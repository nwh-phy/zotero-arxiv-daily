from types import SimpleNamespace
import json

import pytest

from scripts.probe_llm import probe, safe_error


def test_probe_preserves_provider_parameters_and_sends_only_a_minimal_prompt():
    calls = []
    class Client:
        def with_options(self, **kwargs):
            assert kwargs == {'timeout': 30, 'max_retries': 0}
            return self
        @property
        def responses(self):
            def create(**kwargs):
                calls.append(kwargs)
                return SimpleNamespace(output_text='OK')
            return SimpleNamespace(create=create)
    result = probe(Client(), {'api_mode': 'response', 'generation_kwargs': {
        'model': 'test-model', 'max_tokens': 4096, 'reasoning': {'effort': 'low'}}})
    assert result == {'status': 'PASS', 'nonempty_response': True}
    assert calls == [{'input': [{'role': 'user', 'content': 'Reply with OK.'}],
                     'model': 'test-model', 'max_output_tokens': 4096, 'reasoning': {'effort': 'low'}}]


@pytest.mark.parametrize('body,status,category,code', [
    ({'error': {'code': 'insufficient_quota', 'message': 'SECRET'}}, 429, 'quota_or_credit_limit', 'insufficient_quota'),
    ({'code': 'rate_limit_exceeded', 'message': 'SECRET'}, 429, 'rate_limit', 'rate_limit_exceeded'),
    ({'error': {'code': 'SECRET', 'message': 'SECRET'}}, 429, 'rate_limit_or_quota_unspecified', 'not_disclosed'),
    ({'error': {'message': 'Insufficient balance: SECRET'}}, 429, 'quota_or_credit_limit', 'not_disclosed'),
    ({'error': 'SECRET'}, 401, 'authentication_or_access', 'not_disclosed'),
    ('SECRET', 503, 'provider_server_error', 'not_disclosed'),
    (None, None, 'request_error', 'not_disclosed'),
])
def test_probe_diagnostics_never_expose_upstream_content(body, status, category, code):
    exc = RuntimeError('SECRET: endpoint, request and key must stay private')
    exc.body, exc.status_code = body, status
    result = safe_error(exc)
    assert result['category'] == category and result['provider_code'] == code
    assert 'SECRET' not in json.dumps(result)


def test_empty_probe_response_is_not_success(monkeypatch):
    monkeypatch.setattr('scripts.probe_llm._request_llm', lambda *a: '')
    client = SimpleNamespace(with_options=lambda **kw: None)
    assert probe(client, {}) == {'status': 'FAIL', 'category': 'empty_response'}
