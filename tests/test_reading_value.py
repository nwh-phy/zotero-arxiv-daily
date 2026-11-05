"""Offline contracts for evidence-based reading value and fail-closed selection."""
import json
from types import SimpleNamespace

import pytest

from tests.canned_responses import make_sample_paper
from zotero_arxiv_daily import reading_value
from zotero_arxiv_daily.progress import PaperProgress
from zotero_arxiv_daily.reading_value import (
    choose_reading_value, judge_reading_value, select_reading_value, validate_judgement,
)
from zotero_arxiv_daily.relevance import select

PARAMS = {'api_mode': 'chat_completion', 'generation_kwargs': {'model': 'stub'}}
OPTIONS = {
    'policy': 'reading_value', 'profile': '当前：q-EELS；方法：谱学分析；探索：低能集体激发',
    'top_k': 60, 'focus_limit': 6, 'explore_limit': 2,
}
ABSTRACT = 'We measure momentum-resolved phonon spectra using electron energy-loss spectroscopy.'


def paper(title, **overrides):
    return make_sample_paper(title=title, url=f'https://example.invalid/papers/{title}',
                             abstract=overrides.pop('abstract', ABSTRACT), **overrides)


def judgement(score=4, route='current', **overrides):
    return {'score': score, 'route': route, 'reason': '可比较动量分辨声子谱',
            'contribution': '测量动量分辨声子谱', 'target': '声子色散',
            'use': '比较q-EELS采集与谱线分析方法', 'evidence': 'momentum-resolved phonon spectra', **overrides}


def client(replies, calls=None, tick=None):
    """One fake response per exact paper title; missing replies fail without network."""
    def create(**kwargs):
        messages = kwargs.get('messages') or kwargs.get('input')
        title = messages[-1]['content'].split('\n', 1)[0].removeprefix('Title: ')
        if calls is not None:
            calls.append(title)
        if tick:
            tick()
        reply = replies.get(title)
        if reply is None:
            raise RuntimeError('fake provider failure')
        text = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))], output_text=text)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                           responses=SimpleNamespace(create=create))


def rated(title, score=4, route='current', review=None):
    result = paper(title)
    data = judgement(score, route)
    result.relevance_score, result.relevance_reason = data.pop('score'), data.pop('reason')
    result.relevance_details, result.relevance_status = data, 'scored'
    result.relevance_review = review or {}
    return result


def test_judgement_preserves_concrete_use_and_exact_source_with_whitespace_normalization():
    candidate = paper('P', abstract='We measure momentum-resolved\n\t phonon spectra.')
    data = judgement(evidence='momentum-resolved phonon spectra')
    result = judge_reading_value(client({'P': '```json\n' + json.dumps(data) + '\n```'}), PARAMS, '画像', candidate)
    assert result == data
    candidate.title = 'Direct measurement of phonon dispersion'
    assert validate_judgement(judgement(evidence='measurement of phonon dispersion'), candidate)


@pytest.mark.parametrize('changes', [
    {'evidence': 'momentum-resolved electron spectra'},
    {'evidence': '动量分辨声子谱'},
    {'evidence': 'Momentum-resolved phonon spectra'},
    {'evidence': 'momentum-resolved ... phonon spectra'},
    {'evidence': 'phonon'}, {'reason': 'x' * 201}, {'use': 'x' * 501}, {'evidence': 'x' * 1201},
    {'use': ''}, {'target': ''}, {'contribution': ''}, {'evidence': ''},
    {'route': 'exclude', 'score': 4}, {'route': 'exploration', 'score': 4},
    {'route': 'current', 'score': 2}, {'route': 'method', 'score': 2},
    {'route': 'other'}, {'score': True}, {'score': 3.5}, {'score': 6}, {'reason': ' '},
])
def test_judgement_rejects_unverifiable_or_inconsistent_recommendations(changes):
    with pytest.raises(ValueError):
        validate_judgement(judgement(**changes), paper('P'))


def test_evidence_cannot_join_the_end_of_title_to_start_of_abstract():
    with pytest.raises(ValueError, match='evidence'):
        validate_judgement(judgement(evidence='P We measure'), paper('P'))


@pytest.mark.parametrize('route,score', [('current', 3), ('method', 5), ('exploration', 3), ('exclude', 1), ('exclude', 2)])
def test_each_supported_route_has_a_valid_score(route, score):
    data = judgement(score, route)
    if route == 'exclude':
        data.update(contribution='', target='', use='', evidence='')
    assert validate_judgement(data, paper('P')) == data


def test_select_applies_separate_caps_score_order_and_original_order_for_ties():
    titles = list('ABCDEFGHIJKL')
    papers = [paper(title) for title in titles]
    replies = {title: judgement(4) for title in titles[:7]}
    replies.update({'H': judgement(5, 'method'), 'I': judgement(3, 'exploration'),
                    'J': judgement(3, 'current'), 'K': judgement(3, 'method'), 'L': judgement(2, 'exclude')})
    selected, summary = select(papers, client(replies), PARAMS, OPTIONS, limit=8)
    assert [p.title for p in selected] == list('HABCDEIJ')
    assert [p.recommendation_tier for p in selected] == ['focus'] * 6 + ['explore'] * 2
    assert summary['complete'] and not summary['incomplete']
    assert (summary['attempted'], summary['scored'], summary['passed_threshold']) == (12, 12, 11)
    assert (summary['focus_count'], summary['explore_count'], summary['selected_unscored']) == (6, 2, 0)
    assert summary['shortlist']['selected'] == 12
    assert papers[0].relevance_details['use'] == judgement()['use']


def test_vacant_focus_or_explore_slots_are_not_filled_from_the_other_tier():
    papers = [rated(str(i), 3, 'exploration') for i in range(5)]
    assert len(choose_reading_value(papers, OPTIONS, 8)) == 2
    papers = [rated(str(i), 4) for i in range(9)]
    assert len(choose_reading_value(papers, OPTIONS, 8)) == 6
    assert len(choose_reading_value(papers, OPTIONS, 2)) == 2
    assert not choose_reading_value(papers, OPTIONS, 0)
    assert all(p.recommendation_tier is None for p in papers)


def test_review_reselection_downgrades_and_rejects_then_cleans_old_tiers_deterministically():
    papers = [rated('confirmed', 4, review={'status': 'confirmed'}),
              rated('downgraded', 5, review={'status': 'downgraded'}),
              rated('unavailable', 4, review={'status': 'unavailable'}),
              rated('rejected', 5, review={'status': 'rejected'}),
              rated('failed', 5, review={'status': 'failed'}),
              rated('skipped', 5, review={'status': 'skipped'}),
              rated('provisional', 5)]
    selected = choose_reading_value(papers, OPTIONS, 8)
    assert [p.title for p in selected] == ['provisional', 'confirmed', 'downgraded', 'unavailable']
    assert [p.recommendation_tier for p in selected] == ['focus', 'focus', 'explore', 'explore']
    assert choose_reading_value(papers, OPTIONS, 8) == selected
    papers[-1].relevance_review = {'status': 'rejected'}
    assert [p.title for p in choose_reading_value(papers, OPTIONS, 8)] == ['confirmed', 'downgraded', 'unavailable']
    assert papers[-1].recommendation_tier is None


def test_failed_judgements_produce_incomplete_results_without_similarity_fillins():
    papers = [paper(title) for title in 'ABCD']
    calls = []
    selected, summary = select_reading_value(papers, client({'C': judgement(5)}, calls), PARAMS, OPTIONS, 8)
    assert calls == list('ABCD') and [p.title for p in selected] == ['C']
    assert summary['policy'] == 'reading_value' and summary['mode'] == 'applied'
    assert not summary['complete'] and summary['incomplete']
    assert (summary['failed'], summary['scored'], summary['selected_unscored']) == (3, 1, 0)
    assert papers[0].relevance_score is None and papers[0].recommendation_tier is None


def test_repeated_selection_clears_stale_scores_reviews_and_tiers_before_new_failures():
    candidate = paper('P')
    select_reading_value([candidate], client({'P': judgement(5)}), PARAMS, OPTIONS, 8)
    candidate.relevance_review = {'status': 'confirmed'}
    selected, summary = select_reading_value([candidate], client({}), PARAMS, OPTIONS, 8)
    assert not selected and summary['incomplete']
    assert candidate.relevance_status == 'failed' and candidate.relevance_score is None
    assert candidate.relevance_reason is None and candidate.recommendation_tier is None
    assert candidate.relevance_details == {} and candidate.relevance_review == {}


def test_consecutive_failures_trip_breaker_and_never_select_unscored_papers():
    papers = [paper(str(i)) for i in range(7)]
    calls = []
    selected, summary = select_reading_value(papers, client({}, calls), PARAMS, OPTIONS, 8)
    assert calls == [str(i) for i in range(5)]
    assert [p.relevance_status for p in papers] == ['failed'] * 5 + ['skipped'] * 2
    assert not selected and summary['failed'] == 5 and summary['skipped'] == 2 and summary['incomplete']


def test_budget_blocks_request_whose_full_retry_window_would_overrun_stage(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(reading_value, 'monotonic', lambda: clock[0])
    def tick():
        clock[0] += 60
    papers = [paper(title) for title in 'ABCDEF']
    calls = []
    options = {**OPTIONS, 'time_budget_seconds': 300, 'request_timeout': 60, 'max_retries': 1}
    _, summary = select_reading_value(papers, client({t: judgement() for t in 'ABCDEF'}, calls, tick), PARAMS, options, 8)
    assert calls == list('ABCD')  # At 240 s there is no room for a 120 s worst-case request.
    assert summary['seconds'] == 240 and summary['skipped'] == 2 and summary['incomplete']


def test_stage_uses_default_90_second_one_retry_client_and_4096_output_budget():
    seen = {}
    inner = client({'P': judgement()})
    def with_options(**kwargs):
        seen['options'] = kwargs
        def create(**request):
            seen['max_tokens'] = request['max_tokens']
            return inner.chat.completions.create(**request)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    select_reading_value([paper('P')], SimpleNamespace(with_options=with_options), PARAMS, OPTIONS, 8)
    assert seen == {'options': {'timeout': 90.0, 'max_retries': 1}, 'max_tokens': 4096}


@pytest.mark.parametrize('api_mode,generation,override,expected_key,expected_value', [
    ('chat_completion', {'model': 'stub', 'max_tokens': 512}, None, 'max_tokens', 512),
    ('response', {'model': 'stub', 'max_output_tokens': 768}, None, 'max_output_tokens', 768),
    ('response', {'model': 'stub', 'max_output_tokens': 768}, 256, 'max_output_tokens', 256),
])
def test_none_token_override_inherits_model_budget_and_explicit_override_is_effective(
        api_mode, generation, override, expected_key, expected_value):
    seen = {}
    inner = client({'P': judgement()})
    def create(**request):
        seen.update(request)
        return inner.responses.create(**request)
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                           responses=SimpleNamespace(create=create))
    params = {'api_mode': api_mode, 'generation_kwargs': generation}
    select_reading_value([paper('P')], fake, params, {**OPTIONS, 'max_tokens': override}, 8)
    assert seen[expected_key] == expected_value
    assert params['generation_kwargs'] == generation  # Stage settings do not mutate shared summary parameters.


def test_successful_judgements_resume_per_paper_and_failed_evidence_is_never_cached(tmp_path):
    cache = PaperProgress(tmp_path, 'library')
    run = lambda replies, calls: select_reading_value([paper(t) for t in 'ABC'], client(replies, calls),
                                                      PARAMS, OPTIONS, 8, cache)
    calls = []
    _, summary = run({'A': judgement(), 'B': judgement(evidence='fabricated quote')}, calls)
    assert calls == list('ABC') and summary['failed'] == 2
    assert len(list(tmp_path.rglob('reading-value-*.json'))) == 1
    calls.clear()
    selected, summary = run({'B': judgement(5, 'method'), 'C': judgement(2, 'exclude')}, calls)
    assert calls == list('BC') and summary['reused'] == 1 and summary['complete']
    assert [p.title for p in selected] == ['B', 'A']


@pytest.mark.parametrize('change', ['profile', 'model', 'endpoint', 'abstract', 'prompt', 'version'])
def test_reading_value_cache_invalidates_on_policy_input_and_provider_changes(tmp_path, monkeypatch, change):
    cache = PaperProgress(tmp_path, 'library')
    candidate, options, params = paper('P'), dict(OPTIONS), dict(PARAMS)
    select_reading_value([candidate], client({'P': judgement()}), params, options, 8, cache)
    if change == 'profile':
        options['profile'] = '不同研究画像'
    elif change == 'model':
        params['generation_kwargs'] = {'model': 'other-model'}
    elif change == 'endpoint':
        params['api'] = {'base_url': 'https://other.invalid/v1'}
    elif change == 'abstract':
        candidate.abstract += ' Additional measurement.'
    elif change == 'prompt':
        monkeypatch.setattr(reading_value, 'PROMPT_DIGEST', 'changed-prompt')
    elif change == 'version':
        monkeypatch.setattr(reading_value, 'READING_VALUE_VERSION', 2)
    calls = []
    _, summary = select_reading_value([candidate], client({'P': judgement()}, calls), params, options, 8, cache)
    assert calls == ['P'] and summary['reused'] == 0


def test_cache_ignores_corruption_and_does_not_store_configuration_secrets(tmp_path):
    cache = PaperProgress(tmp_path, 'library')
    params = {**PARAMS, 'api': {'key': 'sk-fake-secret', 'base_url': 'https://private-endpoint.invalid/v1'}}
    select_reading_value([paper('P')], client({'P': judgement()}), params, OPTIONS, 8, cache)
    [path] = list(tmp_path.rglob('reading-value-*.json'))
    contents = path.read_text()
    assert all(value not in contents for value in ('sk-fake-secret', 'private-endpoint.invalid', OPTIONS['profile']))
    record = json.loads(contents)
    record['judgement']['score'] = 5
    path.write_text(json.dumps(record))  # Valid JSON with a mismatched checksum.
    calls = []
    _, summary = select_reading_value([paper('P')], client({'P': judgement()}, calls), params, OPTIONS, 8, cache)
    assert calls == ['P'] and summary['reused'] == 0


def test_structurally_valid_cache_still_requires_evidence_in_the_current_paper(tmp_path):
    cache = PaperProgress(tmp_path, 'library')
    select_reading_value([paper('P')], client({'P': judgement()}), PARAMS, OPTIONS, 8, cache)
    [path] = list(tmp_path.rglob('reading-value-*.json'))
    key = path.stem.removeprefix('reading-value-')
    cache.save_reading_value(key, judgement(evidence='Fabricated yet structurally valid quotation'))
    calls = []
    _, summary = select_reading_value([paper('P')], client({'P': judgement()}, calls), PARAMS, OPTIONS, 8, cache)
    assert calls == ['P'] and summary['reused'] == 0 and summary['complete']


def test_new_policy_cannot_reuse_legacy_score_reason_tuple(tmp_path):
    cache, calls = PaperProgress(tmp_path, 'library'), []
    legacy = {**OPTIONS, 'policy': 'score', 'top_k': 1}
    select([paper('P')], client({'P': {'score': 4, 'reason': '旧评分'}}), PARAMS, legacy, 1, cache)
    selected, summary = select([paper('P')], client({'P': judgement()}, calls), PARAMS, OPTIONS, 1, cache)
    assert calls == ['P'] and summary['reused'] == 0
    assert selected[0].relevance_details['route'] == 'current'
    assert len(list(tmp_path.rglob('relevance-*.json'))) == 1
    assert len(list(tmp_path.rglob('reading-value-*.json'))) == 1


def test_legacy_policy_absent_and_explicit_score_keep_the_previous_selection_contract():
    for options in ({'profile': 'q-EELS', 'top_k': 2}, {'profile': 'q-EELS', 'top_k': 2, 'policy': 'score'}):
        selected, summary = select([paper('A'), paper('B')], client({'A': {'score': 4, 'reason': '旧评分'}}),
                                   PARAMS, options, 2)
        assert [p.title for p in selected] == ['A', 'B'] and summary['selected_unscored'] == 1
        assert 'policy' not in summary


@pytest.mark.parametrize('options', [
    {**OPTIONS, 'profile': ''}, {**OPTIONS, 'profile': ['非字符串']},
    {**OPTIONS, 'focus_limit': -1}, {**OPTIONS, 'explore_limit': True},
    {**OPTIONS, 'time_budget_seconds': float('nan')}, {**OPTIONS, 'max_consecutive_failures': 0},
])
def test_invalid_reading_value_configuration_is_rejected_before_any_request(options):
    with pytest.raises(ValueError):
        select_reading_value([paper('P')], client({}), PARAMS, options, 8)
