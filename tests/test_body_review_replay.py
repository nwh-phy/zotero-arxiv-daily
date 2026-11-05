"""Offline boundaries for archived body review and strict-summary retries."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts import replay_body_review as cli
from zotero_arxiv_daily.protocol import Paper
from zotero_arxiv_daily.reading_value import PROMPT_DIGEST, READING_VALUE_VERSION


@pytest.fixture(autouse=True)
def offline_tokenizer(monkeypatch):
    # Token-budget correctness is covered in evidence tests. This retry wiring
    # suite must not download a missing tokenizer table on a clean machine.
    import tiktoken
    monkeypatch.setattr(tiktoken, 'get_encoding', lambda name: SimpleNamespace(encode=list, decode=''.join))


@pytest.fixture
def inputs(tmp_path):
    directory = tmp_path / 'config'
    directory.mkdir()
    options = {'enabled': True, 'policy': 'reading_value', 'top_k': 5, 'min_score': 3,
               'profile': 'A reader investigating EELS and transferable analysis methods.',
               'focus_limit': 2, 'explore_limit': 1,
               'shortlist': {'embedding_k': 5, 'core_k': 0, 'core_patterns': []},
               'fulltext_review': {'request_timeout': 90}}
    public = {'llm': {'api': {'key': '${oc.env:UNRELATED_LLM_SECRET}'},
                      'api_mode': 'response', 'generation_kwargs': {'model': 'deepseek-v4.1-flash', 'max_tokens': 4096},
                      'language': '简体中文', 'summary_instructions': 'Use five labelled Chinese fields and supplied sources.',
                      'strict_summary': True, 'summary_min_chars': 150, 'summary_max_chars': 250},
              'reranker': {'relevance': options}, 'executor': {'max_paper_num': 3},
              'zotero': {'api_key': '${oc.env:ZOTERO_SECRET}'},
              'email': {'sender_password': '${oc.env:SMTP_SECRET}'}}
    (directory / 'base.yaml').write_text(json.dumps(public))
    (directory / 'custom.yaml').write_text('{}')
    candidates = []
    for i, (route, score) in enumerate([('current', 5), ('method', 4), ('current', 4), ('exploration', 3), ('exclude', 2)]):
        quote = 'A concrete physics result with a transferable method.'
        candidates.append({'id': f'arxiv:2611.{i+1:05d}', 'source': 'arxiv',
                           'url': f'https://arxiv.org/abs/2611.{i+1:05d}v1',
                           'title': f'Paper {i+1}', 'abstract': quote + ' Background.', 'stage1_rank': i+1,
                           'score': 8-i/10, 'relevance_status': 'scored', 'relevance_score': score,
                           'relevance_reason': '有具体阅读价值',
                           'relevance_details': {'route': route, 'contribution': 'Concrete result',
                                                 'target': 'Target question', 'use': 'Compare a response',
                                                 'evidence': quote if route != 'exclude' else ''},
                           'relevance_review': {'status': 'failed', 'reason': '等待重试', 'evidence_ids': []} if i < 2 else {},
                           'recommendation_tier': 'explore' if i == 3 else None,
                           'shortlist_sources': ['embedding'], 'selected': i == 3})
    windows = {'arxiv': {'start_utc': '2026-10-31T00:00:00+00:00',
                         'end_utc_exclusive': '2026-11-05T00:00:00+00:00', 'count': 5}}
    relevance = {'policy': 'reading_value', 'complete': True, 'incomplete': False,
                 'attempted': 5, 'scored': 5, 'failed': 0, 'skipped': 0, 'selected': 1,
                 'focus_count': 0, 'explore_count': 1, 'shortlist': {'selected': 5}}
    bundle = {'schema': 2, 'settings': {'llm_model': 'deepseek-v4.1-flash', 'max_paper_num': 3,
                                      'relevance': {**copy.deepcopy(options), 'prompt_digest': PROMPT_DIGEST,
                                                    'version': READING_VALUE_VERSION}},
              'window': windows, 'candidates': candidates, 'corpus': [], 'similarity': []}
    report = {'status': 'review_incomplete', 'windows': windows, 'candidate_count': 5,
              'relevance': relevance, 'candidates': copy.deepcopy(candidates)}
    bundle_path, report_path = tmp_path / 'evaluation.json', tmp_path / 'report.json'
    bundle_path.write_text(json.dumps(bundle))
    report_path.write_text(json.dumps(report))
    return bundle_path, report_path, directory


def argv(inputs, output):
    return [str(inputs[0]), '--report', str(inputs[1]), '--config-dir', str(inputs[2]),
            '--live', '--output', str(output)]


def mutate(path, operation):
    data = json.loads(path.read_text())
    operation(data)
    path.write_text(json.dumps(data))


def forbid(*args, **kwargs):
    pytest.fail('No external request or delivery operation is permitted here')


def summary():
    return '\n'.join(label + '：' + '中'*30 + '[全文片段确认:E1]'
                     for label in ['问题', '方法', '结论', '创新', '参考价值'])


def stub_live(monkeypatch, statuses=('confirmed', 'confirmed'), *, invalid_summary=False):
    calls = {'clients': [], 'bodies': [], 'reviews': [], 'summaries': []}
    def client(**kwargs):
        calls['clients'].append(kwargs)
        return SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(cli, 'OpenAI', client)
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-only-env-secret')
    monkeypatch.setenv('OPENAI_API_BASE', 'https://only-env-private-provider.example/v1')
    def body(paper):
        calls['bodies'].append(paper.url)
        assert paper.authors == [] and paper.published is None
        paper.full_text = '# Results\nA concrete body finding supports this method and target response.'
    monkeypatch.setattr(cli, 'extract_body', body)
    class Review:
        def __init__(self, client, params, options, cache):
            self.stats = {'complete': True, 'attempted': 0, 'reused': 0,
                          **{key: 0 for key in ('confirmed', 'downgraded', 'rejected', 'unavailable', 'failed', 'skipped')}}
            assert 'progress' in cache.directory.parts
        def _finish(self, paper, review):
            paper.relevance_review = review
            self.stats[review['status']] += 1
            if review['status'] in ('failed', 'skipped'):
                self.stats['complete'] = False
        def review(self, paper):
            calls['reviews'].append(paper.title)
            self.stats['attempted'] += 1
            assert paper.relevance_review == {}
            self._finish(paper, {'status': statuses[self.stats['attempted']-1], 'reason': '正文片段支持所述用途',
                                 'evidence_id': 'E1', 'evidence_ids': ['E1'],
                                 'evidence': paper.evidence_excerpts[0]['text'][:40]})
    monkeypatch.setattr(cli, 'ReadingReview', Review)
    def generate(paper, client, params):
        calls['summaries'].append(paper.title)
        assert params['strict_summary'] is True
        paper.tldr = 'English invalid draft' if invalid_summary else summary()
        return paper.tldr
    monkeypatch.setattr(Paper, 'generate_tldr', generate)
    return calls


def test_restores_original_focus_isolates_requests_and_leaves_sources_immutable(inputs, tmp_path, monkeypatch, capsys):
    import smtplib
    import zotero_arxiv_daily.history as history
    import zotero_arxiv_daily.utils as utils
    monkeypatch.setattr(history.DeliveryHistory, '__init__', forbid)
    monkeypatch.setattr(utils, 'send_email', forbid)
    monkeypatch.setattr(smtplib, 'SMTP', forbid)
    monkeypatch.setattr(smtplib, 'SMTP_SSL', forbid)
    calls = stub_live(monkeypatch)
    before = [path.read_bytes() for path in inputs[:2]]
    output = tmp_path / 'body-replay'
    assert cli.main(argv(inputs, output)) == 0
    report = json.loads((output / 'report.json').read_text())
    assert report['status'] == 'body_replay_completed'
    assert report['original_selection'] == [{'id': 'arxiv:2611.00001', 'tier': 'focus'},
                                             {'id': 'arxiv:2611.00002', 'tier': 'focus'},
                                             {'id': 'arxiv:2611.00004', 'tier': 'explore'}]
    assert calls['bodies'] == ['https://arxiv.org/abs/2611.00001v1', 'https://arxiv.org/abs/2611.00002v1']
    assert calls['reviews'] == calls['summaries'] == ['Paper 1', 'Paper 2']
    assert len(calls['clients']) == 1 and calls['clients'][0]['api_key'] == 'sk-only-env-secret'
    assert report['summaries_generated'] == 2 and report['focus_count'] == 2 and report['explore_count'] == 1
    assert report['provenance']['current_api_mode'] == 'response'
    assert len(report['provenance']['summary_instructions_sha256']) == 64
    assert all(report[key] is False for key in ('fresh_retrieval', 'fresh_ranking', 'email_sent',
                                               'delivery_history_accessed', 'reserve_backfill'))
    evidence = json.loads((output / 'review-evidence.json').read_text())
    assert len(evidence) == 2 and evidence[0]['excerpts'][0]['id'] == 'E1'
    assert evidence[0]['summary'] == summary() and 'concrete body finding' in evidence[0]['excerpts'][0]['text']
    assert [path.read_bytes() for path in inputs[:2]] == before
    for name in ('brief.txt', 'brief.html'):
        assert '存档正文复核重试预览' in (output / name).read_text()
    assert '抓取 5 篇' not in (output / 'brief.txt').read_text()
    raw = ''.join(path.read_text() for path in output.rglob('*') if path.is_file()) + capsys.readouterr().out
    for private in ('sk-only-env-secret', 'only-env-private-provider', 'ZOTERO_SECRET', 'SMTP_SECRET', 'UNRELATED_LLM_SECRET'):
        assert private not in raw
    assert not list(output.rglob('history-*.json'))


def test_real_review_strict_summary_assertion_and_renderer_run_together_offline(inputs, tmp_path, monkeypatch):
    requests, assertions = [], []
    def create(**kwargs):
        requests.append(kwargs)
        messages = kwargs['input']
        if '复核员' in messages[0]['content']:
            payload = json.loads(messages[1]['content'])
            content = json.dumps({'status': 'confirmed', 'reason': '正文片段支持贡献及具体用途',
                                  'evidence_id': 'E1', 'evidence': payload['excerpts'][0]['text']})
        else:
            content = summary()
        return SimpleNamespace(output_text=content)
    client = SimpleNamespace(responses=SimpleNamespace(create=create), close=lambda: None)
    client.with_options = lambda **kwargs: client
    monkeypatch.setattr(cli, 'OpenAI', lambda **kwargs: client)
    monkeypatch.setenv('OPENAI_API_KEY', 'fake-key')
    monkeypatch.setenv('OPENAI_API_BASE', 'https://fake-provider.example/v1')
    def body(paper):
        paper.full_text = '# Results\nA concrete body finding supports this method and target response.'
    monkeypatch.setattr(cli, 'extract_body', body)
    real_assertion = cli.assert_reading_value_delivery
    def check(papers, report):
        real_assertion(papers, report)
        assertions.append([p.title for p in papers])
    monkeypatch.setattr(cli, 'assert_reading_value_delivery', check)
    output = tmp_path / 'real-stages'
    assert cli.main(argv(inputs, output)) == 0
    assert len(requests) == 4 and all(request['model'] == 'deepseek-v4.1-flash' for request in requests)
    assert assertions == [['Paper 1', 'Paper 2', 'Paper 4']]
    report = json.loads((output / 'report.json').read_text())
    assert report['reading_review']['confirmed'] == 2 and report['summaries_generated'] == 2
    assert '重点推荐 2 篇' in (output / 'brief.txt').read_text()


@pytest.mark.parametrize('statuses', [('rejected', 'rejected'), ('downgraded', 'unavailable')])
def test_no_reserve_backfill_and_no_summaries_for_nonfocus(inputs, tmp_path, monkeypatch, statuses):
    calls = stub_live(monkeypatch, statuses)
    output = tmp_path / 'outcomes'
    assert cli.main(argv(inputs, output)) == 0
    report = json.loads((output / 'report.json').read_text())
    assert report['focus_count'] == 0 and report['summaries_generated'] == 0 and calls['summaries'] == []
    assert calls['reviews'] == ['Paper 1', 'Paper 2']
    assert all(p['id'] != 'arxiv:2611.00003' for p in report['papers'])


def test_failed_review_blocks_summaries_and_retains_evidence(inputs, tmp_path, monkeypatch):
    calls = stub_live(monkeypatch, ('confirmed', 'failed'))
    output = tmp_path / 'incomplete'
    assert cli.main(argv(inputs, output)) == 1
    report = json.loads((output / 'report.json').read_text())
    assert report['status'] == 'review_incomplete' and report['reading_review']['complete'] is False
    assert calls['summaries'] == []
    assert len(json.loads((output / 'review-evidence.json').read_text())) == 2
    assert '筛选未完成' in (output / 'brief.txt').read_text()


def test_invalid_summary_blocks_success_with_safe_diagnostics(inputs, tmp_path, monkeypatch):
    stub_live(monkeypatch, invalid_summary=True)
    output = tmp_path / 'invalid-summary'
    assert cli.main(argv(inputs, output)) == 1
    report = json.loads((output / 'report.json').read_text())
    assert report['status'] == 'summary_failed' and report['failed_count'] == 2
    assert report['failures'][0]['error_type'] == 'ReplayRejected'
    assert '本次重试未完成' in (output / 'brief.txt').read_text()


@pytest.mark.parametrize('change,expected', [
    (lambda d: d['relevance'].update(complete=False), 'archived_screening_incomplete_or_unpaired'),
    (lambda d: d['relevance'].update(skipped=1), 'archived_screening_incomplete_or_unpaired'),
    (lambda d: d['relevance'].update(scored=4), 'archived_shortlist_incomplete'),
    (lambda d: d.update(candidate_count=4), 'archived_candidate_count_mismatch'),
    (lambda d: d['candidates'][0].update(relevance_reason='Different judgement'), 'archived_report_candidates_mismatch'),
])
def test_incomplete_or_unpaired_archive_refused_before_client(inputs, tmp_path, monkeypatch, change, expected):
    mutate(inputs[1], change)
    monkeypatch.setattr(cli, 'OpenAI', forbid)
    output = tmp_path / 'refused'
    assert cli.main(argv(inputs, output)) == 1
    assert json.loads((output / 'report.json').read_text())['failure']['code'] == expected
    assert not (output / 'progress').exists()


@pytest.mark.parametrize('change,expected', [
    (lambda d: d['settings'].update(llm_model='another-model'), 'screening_model_or_limit_mismatch'),
    (lambda d: d['settings']['relevance'].update(profile='Changed questions'), 'screening_configuration_mismatch'),
    (lambda d: d['settings']['relevance'].update(prompt_digest='old-prompt'), 'screening_prompt_mismatch'),
    (lambda d: d['settings']['relevance'].update(version=999), 'screening_prompt_mismatch'),
])
def test_mismatched_model_profile_or_abstract_prompt_refused(inputs, tmp_path, monkeypatch, change, expected):
    mutate(inputs[0], change)
    monkeypatch.setattr(cli, 'OpenAI', forbid)
    output = tmp_path / 'mismatch'
    assert cli.main(argv(inputs, output)) == 1
    assert json.loads((output / 'report.json').read_text())['failure']['code'] == expected


def test_body_review_budget_change_allowed(inputs, tmp_path, monkeypatch):
    mutate(inputs[2] / 'base.yaml', lambda d: d['reranker']['relevance']['fulltext_review'].update(request_timeout=80))
    stub_live(monkeypatch)
    assert cli.main(argv(inputs, tmp_path / 'review-budget')) == 0


@pytest.mark.parametrize('url', ['https://arxiv.org.evil.example/abs/2611.00001v1',
                                  'https://arxiv.org@evil.example/abs/2611.00001v1',
                                  'https://arxiv.org/abs/2611.00001v1?fetch=https://evil.example',
                                  'https://arxiv.org:443/abs/2611.00001v1',
                                  'https://arxiv.org/abs/2611.00001v1/../../private',
                                  'https://arxiv.org/abs/2613.00001v1'])
def test_untrusted_body_targets_refused_before_request(inputs, tmp_path, monkeypatch, url):
    mutate(inputs[0], lambda d: d['candidates'][0].update(url=url))
    monkeypatch.setattr(cli, 'OpenAI', forbid)
    monkeypatch.setattr(cli, 'extract_text_from_tar', forbid)
    output = tmp_path / 'bad-url'
    assert cli.main(argv(inputs, output)) == 1
    assert json.loads((output / 'report.json').read_text())['status'] == 'failed'


def test_download_container_uses_canonical_urls_and_never_fakes_metadata(monkeypatch):
    paper = Paper(source='arxiv', title='Archived title', authors=[], abstract='Abstract',
                  url='http://arxiv.org/abs/cond-mat/0501234v2')
    cli._canonicalize([paper])
    calls = []
    def tar(raw):
        calls.append((raw.entry_id, raw.pdf_url, raw.source_url()))
        return None
    monkeypatch.setattr(cli, 'extract_text_from_tar', tar)
    monkeypatch.setattr(cli, 'extract_text_from_html', lambda raw: '# Results\nPublic body content')
    monkeypatch.setattr(cli, 'extract_text_from_pdf', forbid)
    cli.extract_body(paper)
    assert calls == [('https://arxiv.org/abs/cond-mat/0501234v2', 'https://arxiv.org/pdf/cond-mat/0501234v2',
                      'https://arxiv.org/src/cond-mat/0501234v2')]
    assert paper.full_text and paper.authors == [] and paper.published is None


def test_safe_provider_failure_and_existing_output_refusal(inputs, tmp_path, monkeypatch, capsys):
    stub_live(monkeypatch)
    def fail(**kwargs):
        raise RuntimeError('sk-sensitive-provider-error https://private-provider.example/v1')
    monkeypatch.setattr(cli, 'OpenAI', fail)
    output = tmp_path / 'provider-error'
    assert cli.main(argv(inputs, output)) == 1
    assert json.loads((output / 'report.json').read_text())['failure'] == {'error_type': 'RuntimeError'}
    before = (output / 'report.json').read_bytes()
    assert cli.main(argv(inputs, output)) == 2
    assert (output / 'report.json').read_bytes() == before
    raw = ''.join(path.read_text() for path in output.rglob('*') if path.is_file()) + capsys.readouterr().out
    assert 'sk-sensitive-provider-error' not in raw and 'private-provider.example' not in raw
    with pytest.raises(SystemExit) as exc:
        cli.main([arg for arg in argv(inputs, tmp_path / 'missing-live') if arg != '--live'])
    assert exc.value.code == 2


def test_no_delivery_history_ranking_or_dotenv_entrypoint_dependency():
    source = Path(cli.__file__).read_text()
    assert 'from zotero_arxiv_daily.executor' not in source
    assert 'send_email' not in source and 'DeliveryHistory' not in source and 'load_dotenv' not in source
    assert 'retrieve_papers(' not in source and 'rerank(' not in source


def test_body_workflow_separate_llm_only_same_repository_and_preserves_model():
    workflow = yaml.safe_load(Path('.github/workflows/body-replay.yml').read_text())
    events = workflow.get('on', workflow.get(True))
    assert set(events['workflow_call']['secrets']) == {'OPENAI_API_KEY', 'OPENAI_API_BASE'}
    steps = workflow['jobs']['body-replay']['steps']
    commands = '\n'.join(step.get('run', '') for step in steps)
    assert '--repo "$GITHUB_REPOSITORY"' in commands and '--name weekly-brief-report' in commands
    assert 'body-replay-source/report.json' in commands and 'body-replay-source/evaluation.json' in commands
    assert 'apply_config_override.py --repository-recommendations' in commands
    assert '--live --output verification/body-replay' in commands
    assert 'ZOTERO_' not in json.dumps(workflow) and 'SENDER_PASSWORD' not in json.dumps(workflow)
    assert not any(step.get('uses', '').startswith('actions/cache') for step in steps)
    test = yaml.safe_load(Path('.github/workflows/test.yml').read_text())
    assert 'body_replay_run_id' in test.get('on', test.get(True))['workflow_dispatch']['inputs']
    assert test['jobs']['body-replay']['uses'] == './.github/workflows/body-replay.yml'
    for job in ('probe', 'replay', 'verify'):
        assert "inputs.body_replay_run_id == ''" in test['jobs'][job]['if']
