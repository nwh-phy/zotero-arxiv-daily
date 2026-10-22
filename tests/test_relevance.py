"""LLM relevance stage, candidate archive, and saved-paper feedback."""
import json
from types import SimpleNamespace

from omegaconf import OmegaConf, open_dict
import pytest

from tests.canned_responses import make_sample_paper, make_stub_zotero_client
from zotero_arxiv_daily.construct_email import render_email
from zotero_arxiv_daily.executor import Executor, library_identifiers
from zotero_arxiv_daily.relevance import judge, select

PARAMS = {'api_mode': 'chat_completion', 'generation_kwargs': {'model': 'stub'}}
OPTIONS = {'profile': 'q-EELS 与低能集体激发', 'top_k': 4, 'min_score': 3}


def llm_client(replies):
    """Reply per paper title; a missing title raises like a provider error."""
    def create(**kwargs):
        text = str(kwargs.get('messages') or kwargs.get('input'))
        reply = next((value for title, value in replies.items() if f'Title: {title}' in text), None)
        if reply is None:
            raise RuntimeError('provider error')
        message = SimpleNamespace(content=reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], output_text=reply)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                           responses=SimpleNamespace(create=create))


def scored(score, reason='理由'):
    return json.dumps({'score': score, 'reason': reason}, ensure_ascii=False)


@pytest.mark.parametrize('reply, expected', [
    ('```json\n{"score": 4, "reason": "测量石墨声子色散"}\n```', (4, '测量石墨声子色散')),
    ('{"score": "5", "reason": " 同一技术 "}', (5, '同一技术')),
])
def test_judge_accepts_fenced_or_string_scores(reply, expected):
    paper = make_sample_paper(title='P')
    assert judge(llm_client({'P': reply}), PARAMS, 'profile', paper) == expected


@pytest.mark.parametrize('reply', ['not json', '{"score": 7, "reason": "x"}', '{"score": true, "reason": "x"}',
                                   '{"score": 3, "reason": ""}'])
def test_judge_rejects_malformed_judgements(reply):
    with pytest.raises(ValueError):
        judge(llm_client({'P': reply}), PARAMS, 'profile', make_sample_paper(title='P'))


def test_select_orders_by_judgement_and_drops_low_scores():
    papers = [make_sample_paper(title=t) for t in ['Cold atoms', 'Phonon EELS', 'Plasmon', 'Unjudged', 'Beyond shortlist']]
    client = llm_client({'Cold atoms': scored(1, '冷原子'), 'Phonon EELS': scored(5), 'Plasmon': scored(3)})
    selected, summary = select(papers, client, PARAMS, OPTIONS, limit=3)
    # Judged papers first by score; an unjudged paper stays eligible after them; below 3 is dropped.
    assert [p.title for p in selected] == ['Phonon EELS', 'Plasmon', 'Unjudged']
    assert {k: v for k, v in summary.items() if k != 'seconds'} == {
        'mode': 'applied', 'attempted': 4, 'scored': 3, 'reused': 0, 'failed': 1, 'skipped': 0,
        'passed_threshold': 2, 'min_score': 3, 'selected': 3, 'selected_unscored': 1}
    assert papers[0].relevance_reason == '冷原子'
    assert [p.relevance_status for p in papers] == ['scored', 'scored', 'scored', 'failed', None]


def test_unscored_fill_in_is_not_reported_as_passing():
    # Review 2026-10-22 case: one 4, two 1s, one provider failure, two slots.
    papers = [make_sample_paper(title=t) for t in 'ABCD']
    client = llm_client({'A': scored(4, '同一技术'), 'B': scored(1), 'C': scored(1)})
    selected, summary = select(papers, client, PARAMS, OPTIONS, limit=2)
    assert [p.relevance_score for p in selected] == [4, None]
    assert summary['passed_threshold'] == 1 and summary['selected_unscored'] == 1
    html = render_email(selected, {'relevance': summary})
    assert '成功评分 3 篇，其中 1 篇达到 3 分' in html and '另有 1 篇评分不可用，按相似度补选' in html
    assert 'LLM 评分不可用，按相似度补选' in html and '2 篇达到' not in html


def test_select_keeps_embedding_order_when_most_judgements_fail():
    papers = [make_sample_paper(title=t) for t in 'ABCD']
    selected, summary = select(papers, llm_client({'A': scored(1)}), PARAMS, OPTIONS, limit=2)
    assert [p.title for p in selected] == ['A', 'B']
    assert summary['mode'] == 'fallback' and summary['scored'] == 1 and summary['selected_unscored'] == 1
    assert papers[0].relevance_score == 1  # Kept for diagnosis; selection used the embedding order.
    assert '仅成功评分 1/4 篇，本期按相似度排序' in render_email(selected, {'relevance': summary})


def test_select_requires_profile_and_enough_shortlist():
    with pytest.raises(ValueError, match='profile'):
        select([], llm_client({}), PARAMS, {**OPTIONS, 'profile': ' '}, limit=1)
    with pytest.raises(ValueError, match='parameters'):
        select([], llm_client({}), PARAMS, {**OPTIONS, 'top_k': 2}, limit=3)


def test_executor_archives_every_ranked_candidate():
    executor = Executor.__new__(Executor)
    executor.config = OmegaConf.create({'executor': {'max_paper_num': 1}, 'llm': PARAMS,
                                        'reranker': {'relevance': {**OPTIONS, 'enabled': True}}})
    executor.openai_client = llm_client({'A': scored(2), 'B': scored(4, '振动谱学')})
    from zotero_arxiv_daily.progress import PaperProgress
    executor.progress = PaperProgress(None, 'library')
    papers = [make_sample_paper(title=t, url=f'https://arxiv.org/abs/2610.0000{i}v1', score=9 - i)
              for i, t in enumerate('AB')]
    report = {}
    assert [p.title for p in executor._select(papers, report)] == ['B']
    assert [(c['id'], c['relevance_score'], c['selected']) for c in report['candidates']] == [
        ('arxiv:2610.00000', 2, False), ('arxiv:2610.00001', 4, True)]
    html = render_email([papers[1]], report)
    assert 'LLM 判断 4/5：振动谱学' in html and 'LLM 按研究画像成功评分 2 篇，其中 1 篇达到 3 分' in html


def test_library_identifiers_match_preprints_and_titles_without_false_dois():
    ids = library_identifiers({'title': 'Plasmon <i>collapse</i>', 'url': 'http://arxiv.org/abs/2610.21209v1',
                               'DOI': '10.1016/j.ultramic.2026.114402', 'extra': 'arXiv: 2511.08819'})
    assert ids == {'title:plasmoncollapse', 'arxiv:2610.21209', 'arxiv:2511.08819'}


@pytest.fixture()
def relevance_executor(config, tmp_path, monkeypatch):
    from datetime import datetime, timezone, timedelta
    from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever
    with open_dict(config):
        config.executor.progress_dir = str(tmp_path / 'progress')
        config.executor.history_enabled = True
        config.executor.output_dir = str(tmp_path / 'report')
        config.source.arxiv.lookback_days = 7
        config.executor.max_paper_num = 1
        config.llm.generate_affiliations = False
        config.reranker.relevance.update({'enabled': True, 'top_k': 2, 'min_score': 3, 'profile': 'q-EELS'})
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    batches = [['2610.00001', '2610.00002'], ['2610.00003', '2610.00004']]
    def retrieve(self):
        self.window = {'start_utc': self.window_start.isoformat(), 'end_utc_exclusive': end.isoformat(), 'count': 2}
        return [make_sample_paper(title=f'Paper {pid}', url=f'https://arxiv.org/abs/{pid}v1',
                                  published=(end - timedelta(days=1)).isoformat()) for pid in batches.pop(0)]
    monkeypatch.setattr(ArxivRetriever, 'retrieve_papers', retrieve)
    monkeypatch.setattr('zotero_arxiv_daily.reranker.api.OpenAI', lambda **kw: SimpleNamespace(
        embeddings=SimpleNamespace(create=lambda **kw: SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3], index=i) for i in range(len(kw['input']))]))))
    monkeypatch.delenv('DELIVERY_MARKER', raising=False)
    return config, tmp_path


def test_irrelevant_week_sends_nothing_but_advances_coverage(relevance_executor, monkeypatch):
    config, tmp_path = relevance_executor
    monkeypatch.setattr('zotero_arxiv_daily.executor.zotero.Zotero', lambda *a, **kw: make_stub_zotero_client())
    monkeypatch.setattr('zotero_arxiv_daily.executor.OpenAI', lambda **kw: llm_client(
        {'Paper 2610.00001': scored(1), 'Paper 2610.00002': scored(2)}))
    monkeypatch.setattr('zotero_arxiv_daily.executor.send_email', lambda *a, **kw: pytest.fail('No email expected'))
    executor = Executor(config)
    executor.run()
    report = json.loads((tmp_path / 'report/report.json').read_text())
    assert report['status'] == 'no_relevant_candidates'
    assert len(report['candidates']) == 2 and not any(c['selected'] for c in report['candidates'])
    assert executor.history.data['covered_until'] and len(executor.history.data['seen']) == 2


def test_next_brief_reports_how_many_delivered_papers_were_saved(relevance_executor, monkeypatch):
    config, tmp_path = relevance_executor
    replies = {f'Paper 2610.0000{i}': scored(4) for i in range(1, 5)}
    monkeypatch.setattr('zotero_arxiv_daily.executor.OpenAI', lambda **kw: llm_client(replies))
    monkeypatch.setattr('zotero_arxiv_daily.executor.zotero.Zotero', lambda *a, **kw: make_stub_zotero_client())
    sent = []
    monkeypatch.setattr('zotero_arxiv_daily.executor.send_email', lambda config, html, **kw: sent.append(html))
    first = Executor(config)
    first.run()
    delivered = first.history.data['last_delivery']['papers']
    assert len(delivered) == 1
    saved = {'data': {'title': 'Saved later', 'abstractNote': '', 'dateAdded': '2026-10-21T00:00:00Z',
                      'collections': [], 'url': 'https://arxiv.org/abs/' + delivered[0]['id'].split(':')[1]}}
    from tests.canned_responses import _DEFAULT_ITEMS
    monkeypatch.setattr('zotero_arxiv_daily.executor.zotero.Zotero',
                        lambda *a, **kw: make_stub_zotero_client(items=[*_DEFAULT_ITEMS, saved]))
    Executor(config).run()
    report = json.loads((tmp_path / 'report/report.json').read_text())
    assert report['previous_delivery']['delivered'] == 1 and report['previous_delivery']['saved'] == 1
    assert '上期推荐 1 篇，目前已有 1 篇在你的 Zotero 中' in sent[-1]


def counting_client(replies, calls, advance=None):
    """llm_client that records requested titles and optionally advances a fake clock."""
    inner = llm_client(replies)
    def create(**kwargs):
        text = str(kwargs.get('messages') or kwargs.get('input'))
        calls.append(next((t for t in 'ABCDEF' if f'Title: {t}' in text), '?'))
        if advance:
            advance()
        return inner.responses.create(**kwargs)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                           responses=SimpleNamespace(create=create))


def test_judgements_are_cached_and_invalidated_by_profile_model_or_text(tmp_path):
    from zotero_arxiv_daily.progress import PaperProgress
    cache = PaperProgress(tmp_path, 'library')
    run = lambda client, options=OPTIONS, params=PARAMS: select(
        [make_sample_paper(title=t, abstract=f'About {t}') for t in 'ABC'], client, params, options, 2, cache=cache)
    calls = []
    _, summary = run(counting_client({'A': scored(4), 'C': scored(2)}, calls))  # B fails: not cached.
    assert calls == ['A', 'B', 'C'] and summary['reused'] == 0
    calls.clear()
    selected, summary = run(counting_client({'B': scored(5)}, calls))  # Rerun after an interruption.
    assert calls == ['B'] and summary['reused'] == 2 and [p.title for p in selected] == ['B', 'A']
    default_entries = sorted(tmp_path.rglob('relevance-*.json'))  # The three judgements read by run().
    for changed in ({'options': {**OPTIONS, 'profile': '只关心冷原子'}},
                    {'params': {**PARAMS, 'generation_kwargs': {'model': 'other'}}}):
        calls.clear()
        run(counting_client({t: scored(3) for t in 'ABC'}, calls), **changed)
        assert calls == ['A', 'B', 'C']
    default_entries[0].write_text('broken')  # Not an entry of the changed profile or model.
    calls.clear()
    run(counting_client({t: scored(3) for t in 'ABC'}, calls))
    assert len(calls) == 1  # Only the unreadable entry is judged again.


def test_consecutive_failures_stop_provider_calls():
    calls = []
    papers = [make_sample_paper(title=t) for t in 'ABCD']
    selected, summary = select(papers, counting_client({}, calls), PARAMS,
                               {**OPTIONS, 'max_consecutive_failures': 2}, limit=2)
    assert calls == ['A', 'B'] and [p.relevance_status for p in papers] == ['failed', 'failed', 'skipped', 'skipped']
    assert summary['mode'] == 'fallback' and summary['failed'] == 2 and summary['skipped'] == 2
    assert [p.title for p in selected] == ['A', 'B']


def test_stage_never_starts_a_request_that_could_exceed_its_budget(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr('zotero_arxiv_daily.relevance.monotonic', lambda: clock[0])
    def tick():
        clock[0] += 60  # Every request takes its full timeout.
    calls = []
    papers = [make_sample_paper(title=t) for t in 'ABCDEF']
    options = {**OPTIONS, 'top_k': 6, 'time_budget_seconds': 300, 'request_timeout': 60, 'max_retries': 0}
    _, summary = select(papers, counting_client({t: scored(4) for t in 'ABCDEF'}, calls, tick), PARAMS, options, 2)
    assert calls == list('ABCDE') and papers[5].relevance_status == 'skipped'
    assert clock[0] <= 300 and summary['skipped'] == 1 and summary['seconds'] == 300


def test_requests_use_stage_timeout_retries_and_output_budget():
    seen = {}
    inner = counting_client({'A': scored(4)}, [])
    def with_options(**kwargs):
        seen['client'] = kwargs
        def create(**request):
            seen['max_tokens'] = request.get('max_tokens')
            return inner.chat.completions.create(**request)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = SimpleNamespace(with_options=with_options)
    options = {**OPTIONS, 'top_k': 1, 'request_timeout': 30, 'max_retries': 0, 'max_tokens': 512}
    select([make_sample_paper(title='A')], client, PARAMS, options, limit=1)
    assert seen == {'client': {'timeout': 30.0, 'max_retries': 0}, 'max_tokens': 512}


def test_relevance_cache_stores_no_credentials_endpoint_or_profile(tmp_path):
    from zotero_arxiv_daily.progress import PaperProgress
    params = {**PARAMS, 'api': {'key': 'sk-secret-value', 'base_url': 'https://llm.example.invalid/v1'}}
    select([make_sample_paper(title='A')], llm_client({'A': scored(4)}), params, {**OPTIONS, 'top_k': 1}, 1,
           cache=PaperProgress(tmp_path, 'library'))
    [record] = [p.read_text() for p in tmp_path.rglob('relevance-*.json')]
    assert all(secret not in record for secret in ('sk-secret-value', 'llm.example.invalid', OPTIONS['profile']))
