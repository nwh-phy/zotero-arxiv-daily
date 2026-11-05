from types import SimpleNamespace
import json

import pytest

from tests.canned_responses import make_sample_paper
from zotero_arxiv_daily.progress import PaperProgress
from zotero_arxiv_daily.reading_review import ReadingReview, validate_review

BODY = 'We measure momentum dependent plasmon dispersion in a confined metal.'
PARAMS = {'api_mode': 'response', 'generation_kwargs': {'model': 'fake'}, 'language': '简体中文'}


def paper():
    return make_sample_paper(full_text='# Results\n' + BODY,
                             relevance_details={'route': 'current', 'use': '比较动量依赖', 'contribution': '测量色散'})


def client(reply, calls):
    def create(**kwargs):
        calls.append(kwargs)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(output_text=json.dumps(reply))
    return SimpleNamespace(responses=SimpleNamespace(create=create))


def confirmation():
    return {'status': 'confirmed', 'reason': '片段支持色散测量', 'evidence_id': 'E1', 'evidence': BODY}


def test_review_reuses_cache_but_changed_body_or_use_requires_new_request(tmp_path):
    calls, cache = [], PaperProgress(tmp_path, 'library')
    for _ in range(2):
        review = ReadingReview(client(confirmation(), calls), PARAMS, {}, cache)
        result = review.review(paper())
        assert result['status'] == 'confirmed' and result['evidence_ids'] == ['E1']
    assert len(calls) == 1 and review.stats['reused'] == 1
    changed = paper()
    changed.relevance_details['use'] = '比较厚度依赖'
    ReadingReview(client(confirmation(), calls), PARAMS, {}, cache).review(changed)
    changed.full_text += ' Additional body evidence.'
    ReadingReview(client(confirmation(), calls), PARAMS, {}, cache).review(changed)
    assert len(calls) == 3


@pytest.mark.parametrize('change', [
    {'evidence_id': 'E99'}, {'evidence': 'A fabricated measurement not in the supplied body.'},
    {'evidence_id': None, 'evidence': ''}, {'status': 'unknown'}, {'reason': ''},
])
def test_false_or_unavailable_review_evidence_is_rejected(change):
    with pytest.raises(ValueError):
        validate_review({**confirmation(), **change}, [{'id': 'E1', 'text': BODY}])


def test_missing_body_downgrades_without_request_and_failed_review_is_retryable(tmp_path):
    calls, cache = [], PaperProgress(tmp_path, 'library')
    review = ReadingReview(client(RuntimeError('private upstream credential'), calls), PARAMS, {}, cache)
    missing = paper()
    missing.full_text = None
    assert review.review(missing)['status'] == 'unavailable'
    assert not calls and review.stats['complete']
    assert review.review(paper())['status'] == 'failed'
    assert not review.stats['complete'] and len(calls) == 1
    assert not list(tmp_path.rglob('reading-review-*.json'))
    assert 'private' not in json.dumps(review.stats)
    retry = ReadingReview(client(confirmation(), calls), PARAMS, {}, cache)
    assert retry.review(paper())['status'] == 'confirmed'
    assert retry.stats['complete'] and len(calls) == 2


def test_review_budget_skips_without_request_or_success_cache(tmp_path):
    calls = []
    review = ReadingReview(client(confirmation(), calls), PARAMS, {'time_budget_seconds': 1}, PaperProgress(tmp_path, 'lib'))
    assert review.review(paper())['status'] == 'skipped'
    assert not calls and not review.stats['complete']


def test_disabled_cache_does_not_turn_a_successful_review_into_failure():
    calls = []
    review = ReadingReview(client(confirmation(), calls), PARAMS, {}, PaperProgress(None, 'lib'))
    assert review.review(paper())['status'] == 'confirmed'
    assert review.stats['complete'] and len(calls) == 1


def test_nonobject_cache_is_ignored_and_repaired(tmp_path):
    calls, cache = [], PaperProgress(tmp_path, 'lib')
    ReadingReview(client(confirmation(), calls), PARAMS, {}, cache).review(paper())
    path = next(tmp_path.rglob('reading-review-*.json'))
    path.write_text('[]')
    review = ReadingReview(client(confirmation(), calls), PARAMS, {}, cache)
    assert review.review(paper())['status'] == 'confirmed'
    assert len(calls) == 2 and review.stats['reused'] == 0
    assert isinstance(json.loads(path.read_text()), dict)
