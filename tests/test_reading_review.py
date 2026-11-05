from types import SimpleNamespace
from io import StringIO
import json

from loguru import logger
import pytest

from tests.canned_responses import make_sample_paper
from zotero_arxiv_daily.progress import PaperProgress
from zotero_arxiv_daily.reading_review import ReadingReview, ReviewValidationError, validate_review

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


@pytest.mark.parametrize('change,code', [
    ({'status': 'unknown'}, 'invalid_status'),
    ({'reason': ''}, 'invalid_reason'),
    ({'reason': 'x' * 501}, 'invalid_reason'),
    ({'evidence': None}, 'invalid_evidence'),
    ({'evidence': 'x' * 801}, 'invalid_evidence'),
    ({'evidence_id': 'E99'}, 'unknown_excerpt'),
    ({'evidence': ''}, 'evidence_ref_mismatch'),
    ({'evidence_id': None}, 'evidence_ref_mismatch'),
    ({'evidence': 'A fabricated measurement not in the supplied body.'}, 'quote_mismatch'),
    ({'evidence_id': None, 'evidence': ''}, 'confirmed_without_evidence'),
    ({'evidence': 'We measure'}, 'confirmed_without_evidence'),
])
def test_false_or_unavailable_review_evidence_is_rejected(change, code):
    with pytest.raises(ReviewValidationError) as caught:
        validate_review({**confirmation(), **change}, [{'id': 'E1', 'text': BODY}])
    assert isinstance(caught.value, ValueError)
    assert caught.value.code == code
    assert str(caught.value) == code


def test_missing_body_downgrades_without_request_and_failed_review_is_retryable(tmp_path):
    calls, cache = [], PaperProgress(tmp_path, 'library')
    review = ReadingReview(client(RuntimeError('private upstream credential'), calls), PARAMS, {}, cache)
    missing = paper()
    missing.full_text = None
    assert review.review(missing)['status'] == 'unavailable'
    assert not calls and review.stats['complete']
    assert review.review(paper())['status'] == 'failed'
    assert review.stats['failure_codes'] == {'request_failed': 1}
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


@pytest.mark.parametrize('response,code', [
    ('{"status": "confirmed", "reason": "PRIVATE_RAW_RESPONSE", "evidence_id": ???}', 'invalid_json'),
    (json.dumps({**confirmation(), 'evidence_id': 'PRIVATE_RAW_RESPONSE'}), 'unknown_excerpt'),
    (json.dumps({**confirmation(), 'evidence': 'PRIVATE_RAW_RESPONSE'}), 'quote_mismatch'),
    (RuntimeError('PRIVATE_PROVIDER_CREDENTIAL'), 'request_failed'),
    (ValueError('PRIVATE_PROVIDER_CREDENTIAL'), 'request_failed'),
])
def test_failure_diagnostics_are_safe_fail_closed_and_not_cached(tmp_path, response, code):
    calls, logs = [], StringIO()

    def create(**kwargs):
        calls.append(kwargs)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(output_text=response)

    fake = SimpleNamespace(responses=SimpleNamespace(create=create))
    review = ReadingReview(fake, PARAMS, {}, PaperProgress(tmp_path, 'library'))
    sink = logger.add(logs, format='{message}')
    try:
        result = review.review(paper())
    finally:
        logger.remove(sink)
    assert len(calls) == 1  # Diagnostics do not add repair requests or retries.
    assert result['status'] == 'failed' and result['failure_code'] == code
    assert not review.stats['complete'] and review.stats['failed'] == 1
    assert review.stats['failure_codes'] == {code: 1}
    assert not list(tmp_path.rglob('reading-review-*.json'))
    if isinstance(response, Exception):
        assert result['error_type'] == type(response).__name__
    safe_output = logs.getvalue() + json.dumps(result) + json.dumps(review.stats)
    assert 'PRIVATE_RAW_RESPONSE' not in safe_output
    assert 'PRIVATE_PROVIDER_CREDENTIAL' not in safe_output
    assert BODY not in safe_output


@pytest.mark.parametrize('screening,other_excerpt,expected', [
    (True, False, (True, False)),
    (False, True, (False, True)),
    (True, True, (True, True)),
    (False, False, (False, False)),
])
def test_quote_mismatch_only_records_source_match_booleans(tmp_path, screening, other_excerpt, expected):
    quote = 'PRIVATE_QUOTE: A candidate source sentence copied from the wrong location.'
    candidate = paper()
    candidate.full_text = '# Results\n' + BODY
    if other_excerpt:
        candidate.full_text += '\n\n# Methods\n' + quote
    if screening:
        candidate.relevance_details['evidence'] = quote
    calls, logs = [], StringIO()
    review = ReadingReview(client({**confirmation(), 'evidence': quote}, calls), PARAMS, {},
                           PaperProgress(tmp_path, 'library'))
    sink = logger.add(logs, format='{message}')
    try:
        result = review.review(candidate)
    finally:
        logger.remove(sink)
    assert result['failure_code'] == 'quote_mismatch'
    assert (result['matches_screening_evidence'], result['matches_other_excerpt']) == expected
    assert all(isinstance(result[key], bool) for key in ('matches_screening_evidence', 'matches_other_excerpt'))
    assert quote not in json.dumps(result) + json.dumps(review.stats) + logs.getvalue()
    assert not review.stats['complete'] and not list(tmp_path.rglob('reading-review-*.json'))
