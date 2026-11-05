"""Deterministic bounded recall broadens inspection without recommending papers."""
import pytest
from omegaconf import OmegaConf

from tests.canned_responses import make_sample_paper
from zotero_arxiv_daily.shortlist import build_shortlist


def papers(count=100):
    return [make_sample_paper(title=f'Candidate {i}', abstract='Generic physics.',
                              url=f'https://arxiv.org/abs/2611.{i:05d}v1') for i in range(count)]


def options(**shortlist):
    return {'top_k': 60, 'shortlist': {'embedding_k': 40, 'core_k': 20,
                                     'core_patterns': [r'\bEELS\b'], **shortlist}}


def test_bounded_core_recall_preserves_stage1_order_without_recommending():
    candidates = papers()
    for paper in candidates[70:95]:
        paper.abstract = 'New EELS experiment'
    selected, info = build_shortlist(candidates, options())
    assert selected == candidates[:40] + candidates[70:90]
    assert info == {'total_candidates': 100, 'unique_candidates': 100, 'duplicates_removed': 0,
                    'selected': 60, 'embedding': 40, 'core_added': 20, 'fill': 0, 'core_matches': 25}
    assert selected[0].shortlist_sources == ['embedding']
    assert selected[-1].shortlist_sources == ['core']
    assert all(p.relevance_status is None and p.recommendation_tier is None for p in selected)
    assert candidates[90].shortlist_sources == []


def test_insufficient_core_matches_fill_from_original_order_and_keep_multiple_sources():
    candidates = papers(70)
    for i in (0, 62, 69):
        candidates[i].abstract = 'EELS'
    selected, info = build_shortlist(candidates, options())
    assert selected == candidates[:58] + [candidates[62], candidates[69]]
    assert candidates[0].shortlist_sources == ['embedding', 'core']
    assert candidates[40].shortlist_sources == ['fill']
    assert info['core_added'] == 2 and info['fill'] == 18


def test_deduplicates_arxiv_versions_before_counting_and_handles_small_or_empty_sets():
    candidates = papers(5)
    duplicate = make_sample_paper(url=candidates[0].url.replace('v1', 'v2'), title='Duplicate')
    selected, info = build_shortlist([candidates[0], duplicate, *candidates[1:]], options())
    assert selected == candidates and info['duplicates_removed'] == 1
    assert duplicate.shortlist_sources == []
    assert build_shortlist([], options())[0] == []


def test_normalizes_subscripts_dashes_case_and_resets_previous_shortlist_sources():
    candidates = papers(3)
    candidates[2].title = 'q–EELS in TaSe₂'
    selected, _ = build_shortlist(candidates, {'top_k': 2, 'shortlist': {
        'embedding_k': 1, 'core_k': 1, 'core_patterns': [r'q-EELS.*TaSe2']}})
    assert selected == [candidates[0], candidates[2]]
    selected, _ = build_shortlist(candidates, {'top_k': 1})
    assert selected == candidates[:1] and candidates[2].shortlist_sources == []


@pytest.mark.parametrize('settings', [
    {'top_k': True}, {'top_k': 0}, {'top_k': 2.5},
    {'shortlist': []}, {'shortlist': False}, {'shortlist': {'embedding_k': -1}},
    {'shortlist': {'embedding_k': True}}, {'shortlist': {'embedding_k': 50, 'core_k': 20}},
    {'shortlist': {'core_patterns': '['}}, {'shortlist': {'core_patterns': ''}},
    {'shortlist': {'core_patterns': ['']}}, {'shortlist': {'core_patterns': ['(']}},
    {'shortlist': {'core_patterns': [1]}},
])
def test_rejects_invalid_sizes_mappings_and_patterns(settings):
    with pytest.raises(ValueError):
        build_shortlist(papers(2), settings)


def test_accepts_omegaconf_and_has_independent_bounded_defaults():
    selected, _ = build_shortlist(papers(100), OmegaConf.create(options()))
    assert len(selected) == 60
    selected, _ = build_shortlist(papers(100), {'top_k': 5})
    assert len(selected) == 5
    with pytest.raises(ValueError):
        build_shortlist([], [])
