"""Read-only shortlist planning and isolated LLM preview without application services."""
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import replay_reading_value as cli
from tests.canned_responses import make_sample_paper
from zotero_arxiv_daily.evaluation import build_bundle, restore_candidates
from zotero_arxiv_daily.history import paper_identity


def old_bundle(count=65):
    candidates = [{'id': f'arxiv:2611.{i:05d}', 'title': f'Paper {i}',
                   'abstract': 'Momentum-resolved EELS in layered materials.' if i == count - 1 else 'Physics abstract.',
                   'stage1_rank': i + 1, 'score': 8 - i / 100, 'selection_score': 8 - i / 100,
                   'theme': 'q-EELS', 'relevance_score': 3 if i < 2 else None,
                   'relevance_reason': '旧评分' if i < 2 else None,
                   'relevance_status': 'scored' if i < 2 else None, 'selected': i < 2}
                  for i in range(count)]
    return {'schema': 1, 'candidates': list(reversed(candidates)), 'settings': {'relevance': {'top_k': 30}},
            'window': {'start': '2026-10-26', 'end': '2026-11-02'}, 'corpus': [], 'similarity': []}


@pytest.fixture
def inputs(tmp_path):
    bundle = tmp_path / 'evaluation.json'
    bundle.write_text(json.dumps(old_bundle()), encoding='utf-8')
    config_dir = tmp_path / 'config'
    config_dir.mkdir()
    base = {'llm': {'api_mode': 'response', 'api': {'key': '${oc.env:UNRELATED_SECRET}',
                                                'base_url': '${oc.env:PRIVATE_ENDPOINT}'},
                    'generation_kwargs': {'model': 'fake-model', 'max_tokens': 4096}},
            'reranker': {'relevance': {'top_k': 60, 'profile': 'Physics reader',
                                      'focus_limit': 6, 'explore_limit': 2,
                                      'shortlist': {'embedding_k': 40, 'core_k': 20,
                                                    'core_patterns': [r'\bEELS\b']}}},
            'executor': {'max_paper_num': 8},
            'email': {'sender_password': '${oc.env:SMTP_SECRET}'},
            'zotero': {'api_key': '${oc.env:ZOTERO_SECRET}'}}
    (config_dir / 'base.yaml').write_text(json.dumps(base), encoding='utf-8')
    (config_dir / 'custom.yaml').write_text('{}', encoding='utf-8')
    return bundle, config_dir


def args(inputs, output, mode='--plan-only'):
    bundle, config_dir = inputs
    return [str(bundle), mode, '--config-dir', str(config_dir), '--output', str(output)]


def test_plan_is_offline_immutable_and_reports_recall_without_new_recommendations(inputs, tmp_path, monkeypatch):
    original = inputs[0].read_bytes()
    def forbidden(*a, **kw):
        pytest.fail('Plan mode must not instantiate an API client or call LLM selection')
    monkeypatch.setattr(cli, 'OpenAI', forbidden)
    monkeypatch.setattr(cli, 'select', forbidden)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('OPENAI_API_BASE', raising=False)
    output = tmp_path / 'plan'
    assert cli.main(args(inputs, output)) == 0
    report = json.loads((output / 'comparison.json').read_text())
    assert report['status'] == 'planned' and report['mode'] == 'plan_only'
    assert report['complete'] is None and report['changes']['computed'] is False
    assert report['changes']['removed'] == [] and report['new_selected'] == []
    assert [row['stage1_rank'] for row in report['planned_shortlist']] == list(range(1, 60)) + [65]
    assert report['planned_shortlist'][-1]['shortlist_sources'] == ['core']
    assert [row['stage1_rank'] for row in report['old_selected']] == [1, 2]
    assert report['full_text_review'] == {'performed': False, 'status': 'not_run'}
    assert report['quality_improvement'] == 'not_evaluated'
    screening = report['screening']
    assert screening['model'] == 'fake-model' and screening['reading_value_version'] >= 1
    assert len(screening['prompt_digest']) == len(screening['fingerprint']) == 64
    assert screening['settings']['relevance']['shortlist']['embedding_k'] == 40
    assert screening['settings']['llm']['generation_kwargs']['max_output_tokens'] == 4096
    assert inputs[0].read_bytes() == original
    assert not (output / 'cache').exists()
    labels = list(csv.DictReader((output / 'human_labels.csv').open(encoding='utf-8-sig')))
    assert len(labels) == 60 and all(row['label'] == '' for row in labels)
    assert set(labels[0]) == {'id', 'title', 'abstract', 'label', 'notes'}
    assert '旧评分' not in (output / 'human_labels.csv').read_text()


def _fake_client(monkeypatch):
    created = []
    def factory(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(cli, 'OpenAI', factory)
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-process-environment-only')
    monkeypatch.setenv('OPENAI_API_BASE', 'https://private-process-provider.example/v1')
    return created


def _select_tail(papers, client, params, options, limit, cache):
    assert [p.title for p in papers] == [f'Paper {i}' for i in range(65)]
    assert all(p.relevance_score is None for p in papers)
    assert options['policy'] == 'reading_value' and limit == 8
    assert params['api']['key'] == 'sk-process-environment-only'
    assert cache.directory is not None and 'cache' in cache.directory.parts
    first, last = papers[0], papers[-1]
    for paper in (first, last):
        paper.relevance_score, paper.relevance_reason, paper.relevance_status = 4, '有明确阅读价值', 'scored'
        paper.relevance_details = {'route': 'current', 'contribution': 'New method', 'target': 'layered materials',
                                   'use': 'Compare dispersion', 'evidence': 'EELS'}
        paper.recommendation_tier = 'focus'
    return [first, last], {'policy': 'reading_value', 'attempted': 60, 'scored': 60, 'reused': 0,
                           'failed': 0, 'skipped': 0, 'complete': True, 'incomplete': False,
                           'focus_count': 2, 'explore_count': 0}


def test_live_wiring_has_isolated_cache_and_compares_added_removed_retained(inputs, tmp_path, monkeypatch, capsys):
    created = _fake_client(monkeypatch)
    monkeypatch.setattr(cli, 'select', _select_tail)
    output = tmp_path / 'live'
    assert cli.main(args(inputs, output, '--live')) == 0
    report = json.loads((output / 'comparison.json').read_text())
    assert report['status'] == 'completed' and report['complete'] is True
    assert report['changes'] == {'computed': True, 'added': ['arxiv:2611.00064'],
                                 'removed': ['arxiv:2611.00001'], 'retained': ['arxiv:2611.00000'],
                                 'added_from_stage1_beyond30': ['arxiv:2611.00064']}
    assert report['new_selected'][1]['stage1_rank'] == 65
    assert report['new_selected'][1]['details']['route'] == 'current'
    assert len(created) == 1 and created[0]['api_key'] == 'sk-process-environment-only'
    raw = ''.join(path.read_text(encoding='utf-8-sig') for path in output.iterdir() if path.is_file()) + capsys.readouterr().out
    for private in ('sk-process-environment-only', 'private-process-provider', 'UNRELATED_SECRET',
                    'PRIVATE_ENDPOINT', 'SMTP_SECRET', 'ZOTERO_SECRET'):
        assert private not in raw


def test_existing_output_is_refused_before_any_request(inputs, tmp_path, monkeypatch):
    output = tmp_path / 'occupied'
    output.mkdir()
    sentinel = output / 'comparison.json'
    sentinel.write_text('Do not replace')
    def forbidden(**kwargs):
        pytest.fail('Existing output must be rejected before client creation')
    monkeypatch.setattr(cli, 'OpenAI', forbidden)
    assert cli.main(args(inputs, output, '--live')) == 2
    assert sentinel.read_text() == 'Do not replace' and list(output.iterdir()) == [sentinel]


def test_incomplete_live_run_saves_diagnostics_and_returns_failure(inputs, tmp_path, monkeypatch):
    _fake_client(monkeypatch)
    def incomplete(*a, **kw):
        chosen, summary = _select_tail(*a, **kw)
        summary.update(complete=False, incomplete=True, scored=1, failed=1, skipped=58)
        return chosen, summary
    monkeypatch.setattr(cli, 'select', incomplete)
    output = tmp_path / 'incomplete'
    assert cli.main(args(inputs, output, '--live')) == 1
    report = json.loads((output / 'comparison.json').read_text())
    assert report['status'] == 'incomplete' and report['incomplete'] is True
    assert report['relevance']['failed'] == 1 and (output / 'comparison.csv').exists()


def test_api_exception_saves_only_error_type(inputs, tmp_path, monkeypatch, capsys):
    _fake_client(monkeypatch)
    def failed(*a, **kw):
        raise ConnectionError('https://private-process-provider.example/v1?api_key=sk-process-environment-only')
    monkeypatch.setattr(cli, 'select', failed)
    output = tmp_path / 'failed'
    assert cli.main(args(inputs, output, '--live')) == 1
    report = json.loads((output / 'comparison.json').read_text())
    assert report['failure'] == {'error_type': 'ConnectionError'}
    assert report['status'] == 'failed' and report['complete'] is False
    raw = (output / 'comparison.json').read_text() + capsys.readouterr().out
    assert 'sk-process' not in raw and 'private-process-provider' not in raw


def test_live_uses_real_selector_with_fake_llm_and_writes_only_isolated_caches(inputs, tmp_path, monkeypatch):
    calls, closed = [], []
    def create(**kwargs):
        user = kwargs['input'][1]['content']
        calls.append(user)
        data = {'score': 1, 'reason': '无具体用途', 'route': 'exclude',
                'contribution': '', 'target': '', 'use': '', 'evidence': ''}
        if 'Title: Paper 0\n' in user or 'Title: Paper 64\n' in user:
            tail = 'Title: Paper 64\n' in user
            data.update(score=5 if tail else 4, reason='比较具体测量方法', route='method',
                        contribution='动量分辨测量', target='低能谱学', use='比较色散分析方法',
                        evidence='Momentum-resolved EELS in layered materials.' if tail else 'Physics abstract.')
        return SimpleNamespace(output_text=json.dumps(data, ensure_ascii=False))
    client = SimpleNamespace(responses=SimpleNamespace(create=create), close=lambda: closed.append(True))
    client.with_options = lambda **kwargs: client
    monkeypatch.setattr(cli, 'OpenAI', lambda **kwargs: client)
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-isolated-live-stub')
    monkeypatch.setenv('OPENAI_API_BASE', 'https://isolated-live-stub.example/v1')
    output = tmp_path / 'live-selector'
    original = inputs[0].read_bytes()
    assert cli.main(args(inputs, output, '--live')) == 0
    report = json.loads((output / 'comparison.json').read_text())
    assert len(calls) == 60 and closed == [True]
    assert report['relevance']['scored'] == 60 and report['relevance']['reused'] == 0
    assert report['relevance']['focus_count'] == 2
    assert [row['stage1_rank'] for row in report['new_selected']] == [65, 1]
    assert report['changes']['added_from_stage1_beyond30'] == ['arxiv:2611.00064']
    caches = list((output / 'cache').rglob('reading-value-*.json'))
    assert len(caches) == 60
    raw = ''.join(path.read_text(encoding='utf-8-sig') for path in output.rglob('*') if path.is_file())
    assert 'sk-isolated-live-stub' not in raw and 'isolated-live-stub.example' not in raw
    assert inputs[0].read_bytes() == original


def test_live_missing_environment_never_uses_config_credentials(inputs, tmp_path, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('OPENAI_API_BASE', raising=False)
    def forbidden(**kwargs):
        pytest.fail('Missing environment credentials must not fall back to other configuration')
    monkeypatch.setattr(cli, 'OpenAI', forbidden)
    output = tmp_path / 'missing'
    assert cli.main(args(inputs, output, '--live')) == 1
    report = json.loads((output / 'comparison.json').read_text())
    assert report['failure'] == {'error_type': 'MissingCredentials'}


def test_requires_explicit_mode_and_rejects_screening_environment_interpolations(inputs, tmp_path):
    with pytest.raises(SystemExit) as error:
        cli.main([str(inputs[0]), '--config-dir', str(inputs[1]), '--output', str(tmp_path / 'unused')])
    assert error.value.code == 2 and not (tmp_path / 'unused').exists()
    (inputs[1] / 'custom.yaml').write_text('{"reranker": {"relevance": {"profile": "${oc.env:ZOTERO_SECRET}"}}}')
    output = tmp_path / 'invalid'
    assert cli.main(args(inputs, output)) == 1
    assert json.loads((output / 'comparison.json').read_text())['failure']['error_type'] == 'ValueError'


def test_new_structured_fields_roundtrip_and_old_bundle_stays_readable():
    papers = [make_sample_paper(title=f'Paper {i}', url=f'https://arxiv.org/abs/2611.{i:05d}v1') for i in range(2)]
    paper = papers[1]
    paper.relevance_details = {'route': 'method', 'contribution': 'New fit', 'target': 'EELS',
                               'use': 'Fit spectra', 'evidence': 'An evidence sentence'}
    paper.recommendation_tier = 'explore'
    paper.relevance_review = {'status': 'confirmed', 'reason': '支持', 'evidence_ids': ['E1']}
    paper.shortlist_sources = ['embedding', 'core']
    bundle = build_bundle(papers, list(reversed(papers)), [paper], [], np.empty((2, 0)), {}, {})
    restored = restore_candidates(json.loads(json.dumps(bundle)))
    assert [p.title for p in restored] == ['Paper 1', 'Paper 0']
    assert restored[0].relevance_details == paper.relevance_details
    assert restored[0].recommendation_tier == 'explore'
    assert restored[0].relevance_review == paper.relevance_review
    assert restored[0].shortlist_sources == ['embedding', 'core']
    assert bundle['candidates'][1]['selected_rank'] == 1
    paper.relevance_details['route'] = 'exclude'
    assert bundle['candidates'][1]['relevance_details']['route'] == 'method'
    legacy = restore_candidates(old_bundle(3))
    assert [paper_identity(p) for p in legacy] == [f'arxiv:2611.{i:05d}' for i in range(3)]
    assert legacy[0].relevance_score == 3 and legacy[0].relevance_details == {}
    assert legacy[0].relevance_review == {} and legacy[0].shortlist_sources == []


def test_public_settings_fingerprint_tracks_rules_without_api_or_unknown_secret_fields(inputs):
    options, params, limit = cli.load_screening_config(inputs[1] / 'base.yaml', inputs[1] / 'custom.yaml')
    options.update(api_key='do-not-save-policy-key', base_url='https://do-not-save-policy.example')
    params['api'] = {'key': 'do-not-save-client-key', 'base_url': 'https://do-not-save-client.example'}
    params['generation_kwargs'].update(api_key='do-not-save-generation-key',
                                       extra_headers={'Authorization': 'do-not-save-header'},
                                       reasoning={'effort': 'low', 'api_key': 'do-not-save-reasoning-key'})
    first = cli._screening_snapshot(options, params, limit)
    raw = json.dumps(first)
    assert 'do-not-save' not in raw and 'base_url' not in raw and 'api_key' not in raw
    assert first['settings']['llm']['generation_kwargs']['reasoning'] == {'effort': 'low'}
    params['api'] = {'key': 'different-key', 'base_url': 'https://different-endpoint.example'}
    assert cli._screening_snapshot(options, params, limit)['fingerprint'] == first['fingerprint']
    options['profile'] += ' with a changed research question'
    second = cli._screening_snapshot(options, params, limit)
    assert first['fingerprint'] != second['fingerprint']
    assert first['settings_sha256'] != second['settings_sha256']
