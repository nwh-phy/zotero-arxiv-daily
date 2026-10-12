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


def test_abstract_only_summary_rejects_false_full_text_evidence():
    calls = []
    summary = '问题方法结论创新参考价值[摘要确认]全文片段确认' + '中' * 160
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=summary))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    params = {'strict_summary': True, 'summary_min_chars': 150, 'summary_max_chars': 250,
              'summary_instructions': 'Use supplied evidence only.'}
    paper = make_sample_paper(full_text=None)
    with pytest.raises(RuntimeError, match='email was not sent'):
        paper.generate_tldr(client, params)
    assert len(calls) == 2


def test_atom_arxiv_id_builds_valid_pdf_and_source_urls():
    from zotero_arxiv_daily.retriever.arxiv_retriever import _entry_to_arxiv_result
    entry = SimpleNamespace(id='http://arxiv.org/abs/2610.12406v1', title='Test', summary='Abstract')
    result = _entry_to_arxiv_result(entry)
    assert result.pdf_url == 'https://arxiv.org/pdf/2610.12406v1'
    assert result.source_url() == 'https://arxiv.org/src/2610.12406v1'


@pytest.mark.parametrize('full_text', [None, '# Results\nMeasured result'])
def test_repair_receives_actual_violations_and_available_evidence(full_text):
    params = {'strict_summary': True, 'summary_min_chars': 150, 'summary_max_chars': 250}
    valid = '\n'.join(f'{label}：' + '中' * 30 + '[摘要确认]'
                      for label in ['问题', '方法', '结论', '创新', '参考价值'])
    invalid = '问题：' + '中' * 300 + '[全文片段确认:E99]'
    responses = iter([invalid, valid])
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    paper = make_sample_paper(full_text=full_text)
    assert paper.generate_tldr(client, params) == valid
    feedback = calls[1]['messages'][-1]['content']
    count = sum('\u4e00' <= c <= '\u9fff' for c in invalid)
    assert f'实际{count}个汉字' in feedback
    assert '五项字段缺失或重复' in feedback
    assert '全文标记格式或编号无效' in feedback
    assert len(calls) == 2
    if full_text:
        assert '允许的全文标记仅为：[全文片段确认:E1]' in feedback
    else:
        assert '未提供全文' in feedback


def test_invalid_summary_diagnostics_exclude_model_content():
    paper = make_sample_paper(full_text=None)
    issues = paper.summary_validation_issues('PRIVATE_MODEL_DRAFT',
                                            {'summary_min_chars': 150, 'summary_max_chars': 250})
    assert '实际0个汉字，要求150～250个' in issues
    assert 'PRIVATE_MODEL_DRAFT' not in ';'.join(issues)


def test_empty_model_response_gets_one_bounded_repair():
    valid = '\n'.join(f'{label}：' + '中' * 30 + '[摘要确认]'
                      for label in ['问题', '方法', '结论', '创新', '参考价值'])
    replies = iter([None, valid])
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(replies)))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    params = {'strict_summary': True, 'summary_min_chars': 150, 'summary_max_chars': 250}
    assert make_sample_paper(full_text=None).generate_tldr(client, params) == valid
    assert len(calls) == 2
    assert '摘要为空' in calls[1]['messages'][-1]['content']
    assert calls[1]['messages'][-2]['content']
