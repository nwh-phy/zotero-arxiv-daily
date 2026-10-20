from datetime import datetime, timedelta, timezone
from email import message_from_string
import json
import smtplib
from types import SimpleNamespace

from omegaconf import open_dict
import pytest

from tests.canned_responses import make_sample_paper, make_stub_openai_client, make_stub_zotero_client, make_stub_smtp
from zotero_arxiv_daily.history import DeliveryHistory, paper_identity
from zotero_arxiv_daily.executor import Executor
from zotero_arxiv_daily.utils import EmailDeliveryError, send_email
from zotero_arxiv_daily.construct_email import render_email, html_to_text

END = datetime(2026, 10, 12, tzinfo=timezone.utc)


def test_failed_week_backfills_and_overlap_has_explicit_boundaries(tmp_path):
    state = DeliveryHistory(tmp_path, {'categories': ['physics.optics']})
    assert state.window_start(END, 7, 2, 90) == END - timedelta(days=7)
    previous_end = END - timedelta(days=14)
    state.complete_empty([], previous_end.isoformat())
    restored = DeliveryHistory(tmp_path, {'categories': ['physics.optics']})
    with restored.locked():
        assert restored.window_start(END, 7, 2, 90) == END - timedelta(days=16)
        assert restored.data['covered_until'] == previous_end.isoformat()


def test_first_attempt_is_not_success_but_survives_a_failed_week(tmp_path):
    state = DeliveryHistory(tmp_path, {})
    start = state.window_start(END, 7, 2, 90)
    state.remember_first_attempt(start)
    assert state.data['covered_until'] is None
    assert not state.data['seen'] and state.data['pending'] is None
    restored = DeliveryHistory(tmp_path, {})
    with restored.locked():
        assert restored.window_start(END + timedelta(days=7), 7, 2, 90) == start
        with pytest.raises(RuntimeError, match='max_backfill_days'):
            restored.window_start(END + timedelta(days=90), 7, 2, 90)
        restored.complete_empty([], (END + timedelta(days=7)).isoformat())
        assert restored.data['first_attempt_start'] is None


def test_future_initial_attempt_fails_explicitly(tmp_path):
    state = DeliveryHistory(tmp_path, {})
    state.remember_first_attempt(END + timedelta(days=1))
    with pytest.raises(RuntimeError, match='ahead'):
        state.window_start(END, 7, 2, 90)


@pytest.mark.parametrize('field', ['covered_until', 'first_attempt_start'])
@pytest.mark.parametrize('value', ['', False])
def test_invalid_coverage_timestamp_does_not_reset_to_first_week(tmp_path, field, value):
    state = DeliveryHistory(tmp_path, {})
    state.save()
    data = json.loads(state.path.read_text())
    data[field] = value
    state.path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match='unreadable'):
        with DeliveryHistory(tmp_path, {}).locked():
            pass


def test_previous_history_without_first_attempt_field_remains_compatible(tmp_path):
    state = DeliveryHistory(tmp_path, {})
    state.complete_empty([], END.isoformat())
    data = json.loads(state.path.read_text())
    del data['first_attempt_start']
    state.path.write_text(json.dumps(data))
    with DeliveryHistory(tmp_path, {}).locked() as restored:
        assert restored.window_start(END+timedelta(days=7), 7, 2, 90) == END-timedelta(days=2)


def test_seen_includes_unselected_candidates_and_ignores_versions(tmp_path):
    state = DeliveryHistory(tmp_path, {})
    selected = make_sample_paper(url='https://arxiv.org/abs/2610.00001v1', published=END.isoformat())
    below_cutoff = make_sample_paper(url='http://arxiv.org/abs/2610.00002v1', published=END.isoformat())
    late = make_sample_paper(url='https://arxiv.org/abs/2610.00003v1')
    state.begin([selected], [selected, below_cutoff], END.isoformat())
    state.accepted()
    revision = make_sample_paper(url='http://arxiv.org/abs/2610.00001v2')
    assert state.unseen([revision, below_cutoff, late, late]) == [late]
    assert paper_identity(revision) == paper_identity(selected)


def test_failed_send_does_not_advance_cursor_and_uncertain_send_blocks(tmp_path):
    state = DeliveryHistory(tmp_path, {})
    before = (END - timedelta(days=7)).isoformat()
    state.complete_empty([], before)
    paper = make_sample_paper()
    batch = state.begin([paper], [paper], END.isoformat())
    with pytest.raises(RuntimeError, match='uncertain SMTP outcome'):
        state.assert_ready()
    restored = DeliveryHistory(tmp_path, {})
    with restored.locked():
        assert restored.data['pending']['batch_id'] == batch
        assert restored.data['covered_until'] == before
        with pytest.raises(RuntimeError, match='uncertain'):
            restored.window_start(END, 7, 2, 90)
        restored.rejected()  # Explicitly confirmed not accepted.
        assert restored.unseen([paper]) == [paper]
        assert restored.data['covered_until'] == before


def test_corrupt_history_stops_delivery_instead_of_forgetting_duplicates(tmp_path):
    state = DeliveryHistory(tmp_path, {})
    state.complete_empty([], END.isoformat())
    state.path.write_text('broken json')
    with pytest.raises(RuntimeError, match='unreadable'):
        with DeliveryHistory(tmp_path, {}).locked():
            pass


def test_future_and_overlong_coverage_gaps_fail_explicitly(tmp_path):
    state = DeliveryHistory(tmp_path, {})
    state.complete_empty([], (END - timedelta(days=100)).isoformat())
    with pytest.raises(RuntimeError, match='max_backfill_days'):
        state.window_start(END, 7, 2, 90)
    state.complete_empty([], (END + timedelta(days=1)).isoformat())
    with pytest.raises(RuntimeError, match='ahead'):
        state.window_start(END, 7, 2, 90)


def _legacy_history(tmp_path, name, covered=None, seen=None, pending=None):
    data = {'schema': 1, 'covered_until': covered, 'first_attempt_start': None,
            'seen': seen or {}, 'pending': pending}
    (tmp_path / f'history-{name}.json').write_text(json.dumps(data))


def test_histories_from_earlier_collection_scopes_are_merged_once(tmp_path):
    # Two collection scopes delivered the same week (the 2026-10-19 duplicate brief).
    _legacy_history(tmp_path, 'old', (END - timedelta(days=7)).isoformat(), {'arxiv:2610.00001': END.isoformat()})
    _legacy_history(tmp_path, 'new', END.isoformat(), {'arxiv:2610.00002': END.isoformat()})
    delivered = [make_sample_paper(url=f'https://arxiv.org/abs/2610.0000{i}v1') for i in (1, 2)]
    fresh = make_sample_paper(url='https://arxiv.org/abs/2610.00003v1')
    with DeliveryHistory(tmp_path, {'categories': ['physics.optics']}).locked() as state:
        assert state.window_start(END + timedelta(days=7), 7, 2, 90) == END - timedelta(days=2)
        assert state.unseen([*delivered, fresh]) == [fresh]
        assert not state.path.exists()  # Adoption alone writes nothing, e.g. for a dry run.
        state.complete_empty([fresh], (END + timedelta(days=7)).isoformat())
    _legacy_history(tmp_path, 'new', END.isoformat())  # Ignored once the merged history exists.
    with DeliveryHistory(tmp_path, {'categories': ['physics.optics']}).locked() as restored:
        assert restored.unseen([*delivered, fresh]) == []


def test_uncertain_or_unreadable_earlier_history_blocks_adoption(tmp_path):
    pending = {'status': 'sending', 'batch_id': 'abc', 'end': END.isoformat(), 'seen': {}}
    _legacy_history(tmp_path, 'a', END.isoformat(), pending=pending)
    with DeliveryHistory(tmp_path, {}).locked() as state:
        with pytest.raises(RuntimeError, match='uncertain SMTP outcome'):
            state.assert_ready()
        assert state.data['pending']['batch_id'] == 'abc'  # Reconcilable by its exact batch ID.
    _legacy_history(tmp_path, 'b', END.isoformat(), pending={**pending, 'batch_id': 'def'})
    with pytest.raises(RuntimeError, match='Several earlier histories'):
        with DeliveryHistory(tmp_path, {}).locked():
            pass
    (tmp_path / 'history-b.json').write_text('broken json')
    with pytest.raises(RuntimeError, match='unreadable'):
        with DeliveryHistory(tmp_path, {}).locked():
            pass


def test_collection_changes_keep_the_same_delivery_history(weekly_executor):
    config, _, _ = weekly_executor
    before = Executor(config).history.path
    with open_dict(config):
        config.zotero.include_path = ['2026/other/**']
        config.zotero.ignore_path = ['archive/**']
    assert Executor(config).history.path == before
    with open_dict(config):
        config.source.arxiv.category = ['physics.optics']
    assert Executor(config).history.path != before


def test_local_lock_prevents_parallel_delivery(tmp_path):
    with DeliveryHistory(tmp_path, {}).locked():
        with pytest.raises(RuntimeError, match='Another literature delivery'):
            with DeliveryHistory(tmp_path, {}).locked():
                pass


@pytest.fixture()
def weekly_executor(config, tmp_path, monkeypatch):
    with open_dict(config):
        config.executor.progress_dir = str(tmp_path / 'progress')
        config.executor.history_enabled = True
        config.executor.output_dir = str(tmp_path / 'report')
        config.source.arxiv.lookback_days = 7
        config.executor.max_paper_num = 1
        config.llm.generate_affiliations = False
    monkeypatch.setattr('zotero_arxiv_daily.executor.zotero.Zotero', lambda *a, **kw: make_stub_zotero_client())
    monkeypatch.setattr('zotero_arxiv_daily.executor.OpenAI', lambda **kw: make_stub_openai_client())
    monkeypatch.setattr('zotero_arxiv_daily.reranker.api.OpenAI', lambda **kw: make_stub_openai_client())
    from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever
    actual_end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    def retrieve(self):
        self.window = {'start_utc': self.window_start.isoformat(), 'end_utc_exclusive': actual_end.isoformat(), 'count': 2}
        return [make_sample_paper(title=f'Paper {i}', url=f'https://arxiv.org/abs/2610.0000{i}v1',
                                 published=(actual_end - timedelta(days=1)).isoformat()) for i in range(2)]
    monkeypatch.setattr(ArxivRetriever, 'retrieve_papers', retrieve)
    monkeypatch.delenv('DELIVERY_MARKER', raising=False)
    return config, tmp_path, actual_end


def test_delivery_marks_entire_evaluated_pool_and_second_run_sends_nothing(weekly_executor, monkeypatch):
    config, tmp_path, end = weekly_executor
    sent = []
    monkeypatch.setattr('zotero_arxiv_daily.executor.send_email', lambda *a, **kw: sent.append((a, kw)))
    first = Executor(config)
    first.run()
    assert len(sent) == 1 and sent[0][1]['message_id'].startswith('<')
    assert first.history.data['covered_until'] == end.isoformat()
    assert len(first.history.data['seen']) == 2
    def forbidden(*a, **kw):
        raise AssertionError('No Zotero or embeddings needed for an already evaluated pool')
    monkeypatch.setattr('zotero_arxiv_daily.executor.zotero.Zotero', forbidden)
    Executor(config).run()
    assert len(sent) == 1
    report = json.loads((tmp_path / 'report/report.json').read_text())
    assert report['status'] == 'no_new_candidates'


def test_dry_run_writes_previews_without_smtp_marker_or_history_advancement(weekly_executor, monkeypatch):
    config, tmp_path, end = weekly_executor
    with open_dict(config):
        config.executor.dry_run = True
    marker = tmp_path / 'sent'
    monkeypatch.setenv('DELIVERY_MARKER', str(marker))
    monkeypatch.setattr('zotero_arxiv_daily.executor.send_email', lambda *a, **kw: pytest.fail('SMTP in dry run'))
    executor = Executor(config)
    executor.run()
    assert not marker.exists()
    assert executor.history.data['covered_until'] is None
    assert executor.history.data['first_attempt_start'] is None
    assert not executor.history.path.exists()
    report = json.loads((tmp_path / 'report/report.json').read_text())
    assert report['status'] == 'preview_only'
    assert 'Paper 0' in (tmp_path / 'report/brief.html').read_text()
    assert 'https://arxiv.org/abs/' in (tmp_path / 'report/brief.txt').read_text()


@pytest.mark.parametrize('failure_stage', ['retrieval', 'summary'])
def test_first_failed_run_retries_missed_week_before_any_success(weekly_executor, monkeypatch, failure_stage):
    config, tmp_path, end = weekly_executor
    from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever
    from zotero_arxiv_daily.protocol import Paper
    class Clock(datetime):
        current = end
        @classmethod
        def now(cls, tz=None):
            return cls.current if tz is not None else cls.current.replace(tzinfo=None)
    monkeypatch.setattr('zotero_arxiv_daily.executor.datetime', Clock)
    windows, sent = [], []
    def retrieve(self):
        windows.append(self.window_start)
        if len(windows) == 1 and failure_stage == 'retrieval':
            raise RuntimeError('First retrieval failed')
        self.window = {'start_utc': self.window_start.isoformat(),
                       'end_utc_exclusive': Clock.current.isoformat(), 'count': 1}
        return [make_sample_paper(url='https://arxiv.org/abs/2610.00001v1',
                                 published=(end-timedelta(days=1)).isoformat())]
    attempts = []
    def summarize(self, client, params):
        attempts.append(self.url)
        if len(attempts) == 1 and failure_stage == 'summary':
            raise RuntimeError('First summary failed')
        return 'Validated summary'
    monkeypatch.setattr(ArxivRetriever, 'retrieve_papers', retrieve)
    monkeypatch.setattr(Paper, '_generate_tldr_with_llm', summarize)
    monkeypatch.setattr('zotero_arxiv_daily.executor.send_email', lambda *a, **kw: sent.append(a))
    with pytest.raises(RuntimeError):
        Executor(config).run()
    restored = Executor(config)
    restored.history.load()
    assert restored.history.data['covered_until'] is None
    assert restored.history.data['first_attempt_start'] == (end-timedelta(days=7)).isoformat()
    assert not sent
    Clock.current = end + timedelta(days=7)
    restored.run()
    assert windows == [end-timedelta(days=7)] * 2
    assert len(sent) == 1
    assert restored.history.data['covered_until'] == Clock.current.isoformat()
    assert restored.history.data['first_attempt_start'] is None


@pytest.mark.parametrize('uncertain', [False, True])
def test_smtp_failure_preserves_summaries_and_only_uncertainty_blocks_retry(weekly_executor, monkeypatch, uncertain):
    config, tmp_path, end = weekly_executor
    def failed(*a, **kw):
        raise EmailDeliveryError('submission', 'TimeoutError', uncertain=uncertain)
    monkeypatch.setattr('zotero_arxiv_daily.executor.send_email', failed)
    with pytest.raises(EmailDeliveryError):
        Executor(config).run()
    restored = Executor(config)
    restored.history.load()
    assert restored.history.data['covered_until'] is None
    assert bool(restored.history.data['pending']) == uncertain
    if uncertain:
        with pytest.raises(RuntimeError, match='uncertain'):
            restored.run()
    else:
        sent = []
        monkeypatch.setattr('zotero_arxiv_daily.executor.send_email', lambda *a, **kw: sent.append(a))
        restored.run()
        assert len(sent) == 1
        report = json.loads((tmp_path / 'report/report.json').read_text())
        assert report['summaries_reused'] == 1


def test_smtp_acceptance_marker_survives_history_write_failure(weekly_executor, monkeypatch):
    config, tmp_path, end = weekly_executor
    marker = tmp_path / 'sent'
    monkeypatch.setenv('DELIVERY_MARKER', str(marker))
    monkeypatch.setattr('zotero_arxiv_daily.executor.send_email', lambda *a, **kw: None)
    monkeypatch.setattr(DeliveryHistory, 'accepted', lambda self: (_ for _ in ()).throw(OSError('disk failure')))
    with pytest.raises(RuntimeError, match='SMTP was accepted'):
        Executor(config).run()
    assert marker.read_text() == 'SMTP accepted\n'


@pytest.mark.parametrize('phase,exception,uncertain', [
    ('login', smtplib.SMTPAuthenticationError(535, b'sensitive upstream details'), False),
    ('sendmail', smtplib.SMTPDataError(550, b'sensitive upstream details'), False),
    ('sendmail', smtplib.SMTPServerDisconnected('sensitive upstream details'), True),
])
def test_smtp_error_classification_and_redaction(config, monkeypatch, phase, exception, uncertain):
    Stub = make_stub_smtp([])
    class Failing(Stub):
        pass
    def fail(self, *a, **kw):
        raise exception
    setattr(Failing, phase, fail)
    monkeypatch.setattr(smtplib, 'SMTP', Failing)
    with pytest.raises(EmailDeliveryError) as caught:
        send_email(config, '<p>Brief</p>')
    assert caught.value.uncertain == uncertain
    assert 'sensitive upstream details' not in str(caught.value)


def test_ssl_port_is_direct_and_quit_failure_does_not_undo_acceptance(config, monkeypatch):
    with open_dict(config):
        config.email.smtp_port = 465
    sent = []
    Stub = make_stub_smtp(sent)
    class DisconnectOnQuit(Stub):
        def quit(self):
            raise OSError('lost connection after successful DATA')
    monkeypatch.setattr(smtplib, 'SMTP', lambda *a, **kw: pytest.fail('Plain SMTP on implicit TLS port'))
    monkeypatch.setattr(smtplib, 'SMTP_SSL', DisconnectOnQuit)
    message_id = send_email(config, '<p>中文总结</p><a href="https://arxiv.org/abs/2610.00001">原文</a>')
    assert len(sent) == 1
    message = message_from_string(sent[0][2])
    assert message['Message-ID'] == message_id and message['Date']
    assert [part.get_content_type() for part in message.get_payload()] == ['text/plain', 'text/html']
    plain = message.get_payload()[0].get_payload(decode=True).decode('utf-8')
    assert '中文总结' in plain and 'https://arxiv.org/abs/2610.00001' in plain


def test_email_period_counts_theme_and_absent_metadata_are_clear():
    paper = make_sample_paper(score=7.5, tldr='Summary', recommendation_theme='20_主题/EELS')
    context = {'retrieved_count': 455, 'candidate_count': 400, 'windows': {'arxiv': {
        'start_utc': '2026-10-05T00:00:00+00:00', 'end_utc_exclusive': '2026-10-12T00:00:00+00:00'}}}
    html = render_email([paper], context)
    plain = html_to_text(html)
    assert '2026-10-05 08:00' in plain and '2026-10-12 08:00' in plain
    assert '455' in plain and '400' in plain and 'EELS 1篇' in plain
    assert 'Unknown Affiliation' not in html and '未提供（预印本）' not in html
    assert '.star-wrapper' not in plain


def test_reconcile_cli_inspects_and_requires_exact_batch_without_sending(weekly_executor):
    import subprocess
    import sys
    from pathlib import Path
    from omegaconf import OmegaConf
    config, tmp_path, end = weekly_executor
    (tmp_path / 'config').mkdir()
    OmegaConf.save(config, tmp_path / 'config/default.yaml')
    executor = Executor(config)
    batch = executor.history.begin([make_sample_paper()], [make_sample_paper()], end.isoformat())
    command = [sys.executable, str(Path('scripts/reconcile_delivery.py').resolve())]
    result = subprocess.run(command + ['--outcome', 'inspect'], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and batch in result.stdout
    wrong = subprocess.run(command + ['--outcome', 'accepted', '--batch-id', 'wrong'], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert wrong.returncode != 0
    executor.history.load()
    assert executor.history.data['pending']['batch_id'] == batch
    accepted = subprocess.run(command + ['--outcome', 'accepted', '--batch-id', batch], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert accepted.returncode == 0 and 'no email was sent' in accepted.stdout
    executor.history.load()
    assert executor.history.data['covered_until'] == end.isoformat()
    assert executor.history.data['pending'] is None
