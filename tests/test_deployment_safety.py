from types import SimpleNamespace
import pytest
from tests.canned_responses import make_sample_paper
from zotero_arxiv_daily.construct_email import render_email


def test_summary_failure_stops_delivery():
    def fail(**kwargs):
        raise RuntimeError('credential-bearing upstream error')
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    with pytest.raises(RuntimeError, match='email was not sent'):
        make_sample_paper().generate_tldr(client, {'strict_summary': True})


def test_email_escapes_untrusted_paper_and_summary():
    paper = make_sample_paper(title='<script>alert(1)</script>', tldr='<img src=x onerror=alert(1)>',
                              url='https://arxiv.org/abs/2601.12345', pdf_url='javascript:alert(1)')
    html = render_email([paper])
    assert '<script>' not in html and '<img' not in html
    assert '&lt;script&gt;' in html
    assert 'javascript:' not in html
    assert paper.url in html
    assert '推荐 1 篇' in html


def test_noncompliant_summary_cannot_be_sent_after_bounded_repair():
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='English only'))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    params = {'strict_summary': True, 'summary_min_chars': 150, 'summary_max_chars': 250,
              'summary_instructions': 'Use Chinese and supplied evidence only.'}
    with pytest.raises(RuntimeError, match='email was not sent'):
        make_sample_paper().generate_tldr(client, params)
    assert len(calls) == 2
    assert make_sample_paper().title in calls[1]['messages'][1]['content']
