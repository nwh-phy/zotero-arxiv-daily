"""Reading-value presentation must preserve the evidence and processing boundary."""
import json
from pathlib import Path
import runpy

import pytest

from tests.canned_responses import make_sample_paper
from zotero_arxiv_daily.construct_email import assert_reading_value_delivery, html_to_text, render_email


def reading_paper(title='Focus paper', tier='focus', **overrides):
    defaults = {
        'title': title,
        'abstract': 'We measure momentum-resolved spectra at an interface.',
        'tldr': '结论：测得界面谱的动量依赖[全文片段确认:E1]。\n参考价值：比较峰位[摘要确认]。',
        'score': 7.4,
        'relevance_score': 4 if tier == 'focus' else 3,
        'relevance_reason': '有具体测量和比较用途',
        'relevance_status': 'scored',
        'relevance_details': {
            'route': 'current' if tier == 'focus' else 'exploration',
            'contribution': '测量界面谱的动量依赖',
            'target': '有限动量下的峰位如何变化',
            'use': '可用于设计动量采样并比较色散',
            'evidence': 'We measure momentum-resolved spectra',
        },
        'recommendation_tier': tier,
        'relevance_review': {'status': 'confirmed', 'reason': '测量及分析结果支持该用途',
                             'evidence_id': 'E1', 'evidence_ids': ['E1'],
                             'evidence': 'Measured interface spectra.'} if tier == 'focus' else {},
        'evidence_excerpts': [{'id': 'E1', 'section': 'Results', 'text': 'Measured interface spectra.'}],
        'recommendation_theme': '20_主题/q-EELS',
        'related_papers': ['Reference title'],
    }
    defaults.update(overrides)
    return make_sample_paper(**defaults)


def reading_context(**options):
    return {'relevance': {'policy': 'reading_value', 'complete': True, **options},
            'reading_review': {'complete': True}}


def test_reading_groups_keep_input_order_and_bound_group_sizes():
    papers = [reading_paper(f'Explore {i}', 'explore') for i in range(3)]
    papers[1:1] = [reading_paper(f'Focus {i}') for i in range(7)]
    html = render_email(papers, reading_context())
    assert '重点推荐 6 篇；拓展阅读 2 篇' in html
    assert html.index('<h2>重点推荐') < html.index('Focus 0') < html.index('Focus 5')
    assert html.index('<h2>拓展阅读') < html.index('Explore 0') < html.index('Explore 1')
    assert html.index('Focus 5') < html.index('Explore 0')
    assert 'Focus 6' not in html and 'Explore 2' not in html


def test_reading_group_limits_honor_smaller_context_values():
    papers = [reading_paper('Focus 0'), reading_paper('Focus 1'), reading_paper('Explore', 'explore')]
    html = render_email(papers, reading_context(focus_limit=1, explore_limit=0))
    assert '重点推荐 1 篇；拓展阅读 0 篇' in html
    assert 'Focus 1' not in html and 'Explore' not in html
    with pytest.raises(ValueError, match='delivery validation failed'):
        assert_reading_value_delivery(papers, reading_context(focus_limit=1, explore_limit=0))


def test_nondefault_limits_are_shared_by_display_and_delivery():
    papers = [reading_paper(f'Focus {i}', url=f'https://arxiv.org/abs/2611.{i:05d}') for i in range(7)]
    papers += [reading_paper(f'Explore {i}', 'explore', url=f'https://arxiv.org/abs/2611.{i + 7:05d}')
               for i in range(3)]
    context = reading_context(focus_limit=7, explore_limit=3)
    assert assert_reading_value_delivery(papers, context) is None
    html = render_email(papers, context)
    assert '重点推荐 7 篇；拓展阅读 3 篇' in html
    assert 'Focus 6' in html and 'Explore 2' in html
    with pytest.raises(ValueError, match='delivery validation failed'):
        assert_reading_value_delivery(papers, reading_context())


@pytest.mark.parametrize('limit', [True, -1, None, '6'])
def test_invalid_limit_cannot_pass_delivery_assertion(limit):
    with pytest.raises(ValueError, match='delivery validation failed'):
        assert_reading_value_delivery([reading_paper()], reading_context(focus_limit=limit))


def test_delivery_assertion_rejects_duplicate_normalized_identities():
    papers = [reading_paper('Version 1', url='https://arxiv.org/abs/2611.12345v1'),
              reading_paper('Version 2', 'explore', url='http://arxiv.org/pdf/2611.12345v2.pdf')]
    with pytest.raises(ValueError, match='delivery validation failed'):
        assert_reading_value_delivery(papers, reading_context())


@pytest.mark.parametrize('summary', ['relevance', 'reading_review'])
def test_delivery_assertion_requires_explicit_complete_summaries(summary):
    context = reading_context()
    context[summary].pop('complete')
    with pytest.raises(ValueError, match='delivery validation failed'):
        assert_reading_value_delivery([reading_paper()], context)


def test_changed_body_cannot_reuse_confirmed_review_with_the_same_excerpt_id():
    paper = reading_paper('Stale body review')
    assert assert_reading_value_delivery([paper], reading_context()) is None
    paper.evidence_excerpts[0]['text'] = 'This replacement body does not contain the reviewed measurement.'
    assert 'Stale body review' not in render_email([paper], reading_context())
    with pytest.raises(ValueError, match='delivery validation failed'):
        assert_reading_value_delivery([paper], reading_context())


def test_rendered_review_ids_must_match_the_validated_body_quote():
    paper = reading_paper('Mismatched body review')
    paper.evidence_excerpts.append({'id': 'E2', 'section': 'Methods', 'text': 'Another excerpt.'})
    paper.relevance_review['evidence_ids'] = ['E2']
    assert 'Mismatched body review' not in render_email([paper], reading_context())
    with pytest.raises(ValueError, match='delivery validation failed'):
        assert_reading_value_delivery([paper], reading_context())


def test_focus_leads_with_use_and_keeps_summary_citation_sources():
    paper = reading_paper()
    html = render_email([paper], reading_context())
    assert html.index('推荐给你的原因') < html.index('对应问题') < html.index('实际贡献') < html.index('中文总结')
    assert '正文片段复核' in html and '[E1] Results' in html
    assert '内容来源：论文摘要；正文片段' in html
    assert '[全文片段确认:E1]' not in html
    assert paper.tldr.endswith('[摘要确认]。')
    assert '匹配收藏夹：20_主题/q-EELS' in html
    assert '收藏相似度：7.4' in html
    assert '相关度：' not in html and 'LLM 判断' not in html and '⭐' not in html


def test_explore_is_a_short_abstract_card_without_summary():
    paper = reading_paper('Explore only', 'explore', tldr=None, full_text=None)
    html = render_email([paper], reading_context())
    assert 'Explore only' in html and '实际贡献' in html and '可以怎样用' in html
    assert '摘要依据' in html and '未作正文片段复核' in html
    assert '中文总结' not in html and '对应问题' not in html
    assert '正文片段复核：</strong>' not in html


def test_explore_never_displays_or_validates_a_leftover_summary():
    paper = reading_paper('Explore only', 'explore', tldr='STALE_SUMMARY[全文片段确认:E99]')
    def reject_summary_validation(*args):
        raise AssertionError('Explore cards must not validate or display summaries')
    paper.summary_is_acceptable = reject_summary_validation
    html = render_email([paper], reading_context())
    assert 'Explore only' in html and 'STALE_SUMMARY' not in html


def test_title_quotation_is_identified_as_title_evidence():
    paper = reading_paper('Momentum spectroscopy method', 'explore')
    paper.relevance_details['evidence'] = paper.title
    html = render_email([paper], reading_context())
    assert '标题依据：</strong>“Momentum spectroscopy method”' in html
    assert '摘要依据' not in html


@pytest.mark.parametrize('status,notice', [
    ('downgraded', '正文片段复核后降为拓展阅读'),
    ('unavailable', '正文不可用，保留为拓展阅读'),
])
def test_review_downgrades_are_visible_as_exploration(status, notice):
    paper = reading_paper('Downgraded', 'explore', relevance_review={'status': status}, tldr=None)
    assert notice in render_email([paper], reading_context())


@pytest.mark.parametrize('tier', ['focus', 'explore'])
@pytest.mark.parametrize('status', ['rejected', 'failed', 'skipped'])
def test_bad_review_status_never_reaches_cards(tier, status):
    paper = reading_paper('Hidden title', tier, relevance_review={'status': status, 'evidence_ids': ['E1']})
    assert 'Hidden title' not in render_email([paper], reading_context())
    with pytest.raises(ValueError, match='delivery validation failed'):
        assert_reading_value_delivery([paper], reading_context())


@pytest.mark.parametrize('change', [
    {'relevance_status': 'failed'},
    {'relevance_status': 'skipped'},
    {'relevance_score': 1},
    {'relevance_score': 3},
    {'relevance_details': {}},
    {'recommendation_tier': None},
    {'relevance_review': {}},
    {'relevance_review': {'status': 'confirmed', 'evidence_ids': ['E99']}},
    {'relevance_review': {'status': 'confirmed', 'evidence_ids': []}},
    {'tldr': None},
    {'tldr': '结论：错标签[全文片段确认:E99]'},
    {'tldr': '结论：裸标签[全文片段确认]'},
])
def test_incomplete_or_mislabelled_focus_is_not_recommended(change):
    paper = reading_paper('Hidden title', **change)
    assert 'Hidden title' not in render_email([paper], reading_context())
    with pytest.raises(ValueError, match='delivery validation failed'):
        assert_reading_value_delivery([paper], reading_context())


@pytest.mark.parametrize('field,value', [
    ('route', 'exclude'), ('route', 'exploration'), ('use', ''), ('target', None),
    ('contribution', ' '), ('evidence', ['We measure']), ('evidence', 'Invented quotation'),
])
def test_unsubstantiated_structured_judgement_is_not_rendered(field, value):
    paper = reading_paper('Hidden title')
    paper.relevance_details[field] = value
    assert 'Hidden title' not in render_email([paper], reading_context())
    with pytest.raises(ValueError, match='delivery validation failed'):
        assert_reading_value_delivery([paper], reading_context())


def test_delivery_error_excludes_hidden_model_text():
    paper = reading_paper('PRIVATE_TITLE', relevance_status='failed', relevance_reason='PRIVATE_MODEL_BODY')
    with pytest.raises(ValueError) as caught:
        assert_reading_value_delivery([paper], reading_context())
    assert 'PRIVATE_' not in str(caught.value)


def test_reading_value_escapes_all_model_content_and_keeps_safe_links():
    unsafe = '<script>draft</script>'
    paper = reading_paper(unsafe, abstract=f'Quote {unsafe} measured.',
                          authors=['<img src=x onerror=alert(1)>'],
                          pdf_url='javascript:alert(1)',
                          url='https://arxiv.org/abs/2601.12345?q="unsafe"&x=1',
                          related_papers=['<b>reference</b>'], recommendation_theme='<i>collection</i>',
                          affiliations=['<em>institute</em>'], journal_ref='<span>journal</span>')
    paper.relevance_details.update(contribution=unsafe, target=unsafe, use=unsafe, evidence=unsafe)
    paper.relevance_review['reason'] = unsafe
    paper.evidence_excerpts[0]['section'] = unsafe
    paper.tldr = '结论：' + unsafe + '[全文片段确认:E1]'
    html = render_email([paper], reading_context())
    for tag in ['<script>', '<img', '<b>', '<i>collection', '<em>', '<span>journal']:
        assert tag not in html
    assert '&lt;script&gt;draft&lt;/script&gt;' in html
    assert '&lt;b&gt;reference&lt;/b&gt;' in html
    assert 'javascript:' not in html
    assert 'https://arxiv.org/abs/2601.12345?q=&quot;unsafe&quot;&amp;x=1' in html


@pytest.mark.parametrize('summary_name,options', [
    ('relevance', {'complete': False}), ('relevance', {'incomplete': True}),
    ('reading_review', {'complete': False}), ('reading_review', {'incomplete': True}),
])
def test_incomplete_screening_preview_never_claims_completion(summary_name, options):
    context = reading_context()
    context.setdefault(summary_name, {}).update(options)
    html = render_email([reading_paper()], context)
    assert '筛选未完成，暂不投递' in html and '部分预览' in html
    assert '本期暂无新增推荐' not in html
    with pytest.raises(ValueError, match='delivery validation failed'):
        assert_reading_value_delivery([reading_paper()], context)


def test_legacy_policy_keeps_existing_score_rendering():
    paper = reading_paper()
    paper.relevance_score, paper.relevance_reason = 4, '旧策略判断'
    html = render_email([paper], {'relevance': {'policy': 'score'}})
    assert '相关度：' in html and 'LLM 判断 4/5：旧策略判断' in html
    assert '<h2>重点推荐' not in html


def test_plain_text_retains_reading_groups_and_evidence():
    text = html_to_text(render_email([reading_paper(), reading_paper('Explore', 'explore')], reading_context()))
    assert '\n重点推荐\n' in text and '\n拓展阅读\n' in text
    assert '推荐给你的原因' in text and '[E1] Results' in text


def test_report_script_prints_review_aggregates_without_paper_bodies(tmp_path, monkeypatch, capsys):
    report = {
        'status': 'reading_review_incomplete',
        'relevance': {'policy': 'reading_value', 'mode': 'applied', 'complete': True,
                      'attempted': 60, 'scored': 60, 'focus_count': 4, 'explore_count': 2},
        'reading_review': {'complete': False, 'attempted': 6, 'confirmed': 4, 'downgraded': 1,
                           'rejected': 0, 'unavailable': 0, 'failed': 1, 'skipped': 0, 'reused': 2},
        'papers': [{'title': 'PRIVATE_TITLE', 'relevance_details': {'use': 'PRIVATE_MODEL_BODY'}}],
    }
    output = tmp_path / 'verification' / 'brief'
    output.mkdir(parents=True)
    (output / 'report.json').write_text(json.dumps(report))
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'report_brief.py'
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(script), run_name='__main__')
    text = capsys.readouterr().out
    assert 'focus 4, explore 2' in text
    assert 'Full-text reading review: complete False; attempted 6, confirmed 4, downgraded 1' in text
    assert 'failed 1, skipped 0, reused 2' in text
    assert 'PRIVATE_' not in text and 'passed threshold' not in text
