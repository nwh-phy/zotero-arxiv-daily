from types import SimpleNamespace
from io import StringIO
import json

from loguru import logger
import pytest

from tests.canned_responses import make_sample_paper
from zotero_arxiv_daily import reading_review as review_module
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


def sequence_client(replies, calls, on_request=None):
    replies = iter(replies)

    def create(**kwargs):
        calls.append(kwargs)
        if on_request is not None:
            on_request()
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(output_text=reply if isinstance(reply, str) else json.dumps(reply))

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
    ({'status': []}, 'invalid_status'),
    ({'status': {}}, 'invalid_status'),
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
    (json.JSONDecodeError('PRIVATE_PROVIDER_CREDENTIAL', 'PRIVATE_RAW_RESPONSE', 0), 'request_failed'),
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
    assert len(calls) == (1 if isinstance(response, Exception) else 2)
    assert review.stats['repair_attempted'] == (0 if isinstance(response, Exception) else 1)
    assert review.stats['repaired'] == 0
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


@pytest.mark.parametrize('first_response,code', [
    ('PRIVATE_RAW_DRAFT without JSON', 'invalid_json'),
    ('{"status": "PRIVATE_RAW_DRAFT", "evidence_id": ???}', 'invalid_json'),
    ({**confirmation(), 'status': 'PRIVATE_RAW_DRAFT'}, 'invalid_status'),
    ({**confirmation(), 'evidence': 'PRIVATE_RAW_DRAFT'}, 'quote_mismatch'),
])
def test_one_bounded_repair_uses_original_evidence_and_caches_only_validated_result(tmp_path, first_response, code):
    calls, logs, cache = [], StringIO(), PaperProgress(tmp_path, 'library')
    review = ReadingReview(sequence_client([first_response, confirmation()], calls), PARAMS, {}, cache)
    sink = logger.add(logs, format='{message}')
    try:
        result = review.review(paper())
    finally:
        logger.remove(sink)
    assert result['status'] == 'confirmed' and review.stats['complete']
    assert len(calls) == 2 and review.stats['attempted'] == 1
    assert review.stats['repair_attempted'] == review.stats['repaired'] == 1
    assert review.stats['failed'] == 0 and review.stats['failure_codes'] == {}
    original, repair = calls[0]['input'], calls[1]['input']
    assert len(original) == 2 and len(repair) == 3
    assert repair[:2] == original  # The exact same system and evidence, without a raw assistant draft.
    assert json.loads(repair[-1]['content'])['failure_code'] == code
    assert all(message['role'] != 'assistant' for message in repair)
    [path] = list(tmp_path.rglob('reading-review-*.json'))
    saved = json.loads(path.read_text())
    assert saved['review'] == confirmation()
    safe_output = json.dumps(result) + json.dumps(review.stats) + json.dumps(calls) + path.read_text() + logs.getvalue()
    assert 'PRIVATE_RAW_DRAFT' not in safe_output
    resumed = ReadingReview(sequence_client([], calls), PARAMS, {}, cache)
    assert resumed.review(paper())['status'] == 'confirmed'
    assert resumed.stats['reused'] == 1 and resumed.stats['repair_attempted'] == 0 and len(calls) == 2


def test_repair_is_not_started_when_remaining_budget_cannot_cover_sdk_attempts(tmp_path, monkeypatch):
    clock, calls = {'now': 0}, []
    monkeypatch.setattr(review_module, 'monotonic', lambda: clock['now'])

    def consume_time():
        clock['now'] = 3

    review = ReadingReview(sequence_client([{**confirmation(), 'evidence': 'wrong'}, confirmation()], calls,
                                          consume_time), PARAMS,
                           {'request_timeout': 2, 'max_retries': 1, 'time_budget_seconds': 6},
                           PaperProgress(tmp_path, 'library'))
    result = review.review(paper())
    assert result['failure_code'] == 'quote_mismatch' and not review.stats['complete']
    assert len(calls) == 1 and review.stats['repair_attempted'] == review.stats['repaired'] == 0
    assert not list(tmp_path.rglob('reading-review-*.json'))


def test_provider_failure_during_repair_never_starts_another_repair(tmp_path):
    calls = []
    fake = sequence_client([{**confirmation(), 'evidence': 'wrong'}, RuntimeError('PRIVATE_CREDENTIAL')], calls)
    review = ReadingReview(fake, PARAMS, {}, PaperProgress(tmp_path, 'library'))
    result = review.review(paper())
    assert len(calls) == 2 and result['failure_code'] == 'request_failed'
    assert result['error_type'] == 'RuntimeError'
    assert review.stats['repair_attempted'] == 1 and review.stats['repaired'] == 0
    assert not review.stats['complete'] and not list(tmp_path.rglob('reading-review-*.json'))
    assert 'PRIVATE_CREDENTIAL' not in json.dumps(result) + json.dumps(review.stats)


def test_failed_repair_preserves_consecutive_failure_breaker(tmp_path):
    calls = []
    fake = sequence_client([{**confirmation(), 'evidence': 'wrong'}] * 2, calls)
    review = ReadingReview(fake, PARAMS, {'max_consecutive_failures': 1}, PaperProgress(tmp_path, 'library'))
    assert review.review(paper())['failure_code'] == 'quote_mismatch'
    assert review.review(paper())['status'] == 'skipped'
    assert len(calls) == 2 and review.stats['repair_attempted'] == 1
    assert not review.stats['complete'] and not list(tmp_path.rglob('reading-review-*.json'))


def test_screening_payload_excludes_abstract_evidence_and_unrelated_fields(tmp_path):
    candidate, calls = paper(), []
    candidate.relevance_details.update(evidence='PRIVATE_ABSTRACT_QUOTE', unrelated='PRIVATE_UNUSED_FIELD')
    candidate.relevance_reason = '比较色散测量方法'
    review = ReadingReview(client(confirmation(), calls), PARAMS, {}, PaperProgress(tmp_path, 'library'))
    assert review.review(candidate)['status'] == 'confirmed'
    payload = json.loads(calls[0]['input'][1]['content'])
    assert payload['screening'] == {'route': 'current', 'use': '比较动量依赖',
                                    'contribution': '测量色散', 'reason': '比较色散测量方法'}
    assert 'PRIVATE_ABSTRACT_QUOTE' not in json.dumps(calls)
    assert 'PRIVATE_UNUSED_FIELD' not in json.dumps(calls)
    assert candidate.relevance_details['evidence'] == 'PRIVATE_ABSTRACT_QUOTE'


@pytest.mark.parametrize('quote', [
    'We derive σ_E, with a continuous spectrum.',
    r'We derive $\sigma_E$ with a continuous spectrum.',
    r'We derive $\sigma_E$, ... spectrum.',
])
def test_repair_does_not_relax_latex_punctuation_or_contiguous_quote_validation(quote):
    body = r'We derive $\sigma_E$, with a continuous spectrum.'
    with pytest.raises(ReviewValidationError) as caught:
        validate_review({**confirmation(), 'evidence': quote}, [{'id': 'E1', 'text': body}])
    assert caught.value.code == 'quote_mismatch'
