"""Make one bounded provider request; report safe diagnostics without papers or credentials."""
import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf
from openai import OpenAI

from zotero_arxiv_daily.protocol import _request_llm


_QUOTA_CODES = {'insufficient_quota', 'quota_exceeded', 'insufficient_balance', 'balance_not_enough'}
_RATE_CODES = {'rate_limit_exceeded', 'rate_limit_error', 'too_many_requests'}


def safe_error(exc):
    """Never return upstream messages, arbitrary codes, response bodies or headers."""
    status = getattr(exc, 'status_code', None)
    status = status if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599 else None
    body = getattr(exc, 'body', None)
    body = body if isinstance(body, dict) else {}
    detail = body.get('error', body)
    detail = detail if isinstance(detail, dict) else {}
    code = detail.get('code')
    code = code if isinstance(code, str) else ''
    message = detail.get('message')
    message = message.lower() if isinstance(message, str) else ''
    if code in _QUOTA_CODES or any(term in message for term in (
            'insufficient quota', 'insufficient_quota', 'exceeded your current quota',
            'quota exhausted', 'insufficient balance', 'insufficient credit', '余额不足', '额度用尽', '欠费')):
        category = 'quota_or_credit_limit'
    elif code in _RATE_CODES or any(term in message for term in ('rate limit', 'too many requests', '限流')):
        category = 'rate_limit'
    elif status == 429:
        category = 'rate_limit_or_quota_unspecified'
    elif status in {401, 403}:
        category = 'authentication_or_access'
    elif status is not None and status >= 500:
        category = 'provider_server_error'
    else:
        category = 'request_error'
    return {'status': 'FAIL', 'error_type': type(exc).__name__, 'http_status': status,
            'category': category, 'provider_code': code if code in _QUOTA_CODES | _RATE_CODES else 'not_disclosed'}


def probe(client, params):
    try:
        answer = _request_llm(client.with_options(timeout=30, max_retries=0), params,
                              [{'role': 'user', 'content': 'Reply with OK.'}])
        if not isinstance(answer, str) or not answer.strip():
            return {'status': 'FAIL', 'category': 'empty_response'}
    except Exception as exc:
        return safe_error(exc)
    return {'status': 'PASS', 'nonempty_response': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = OmegaConf.merge(OmegaConf.load('config/base.yaml'), OmegaConf.load('config/custom.yaml'))
        # Resolve only the LLM subtree; Zotero and SMTP secrets are intentionally unavailable.
        params = OmegaConf.to_container(config.llm, resolve=True)
        client = OpenAI(api_key=params['api']['key'], base_url=params['api']['base_url'])
    except Exception as exc:
        result = {'status': 'FAIL', 'category': 'configuration', 'error_type': type(exc).__name__}
    else:
        result = probe(client, params)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
