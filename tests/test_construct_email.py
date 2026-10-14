"""Tests for zotero_arxiv_daily.construct_email: render_email, get_stars, get_block_html."""

from zotero_arxiv_daily.construct_email import render_email, get_stars, get_block_html, get_empty_html
from tests.canned_responses import make_sample_paper


def test_render_email_with_papers():
    papers = [make_sample_paper(score=7.5, tldr="A great paper.", affiliations=["MIT"])]
    html = render_email(papers)
    assert "Sample Paper Title" in html
    assert "A great paper." in html
    assert "MIT" in html


def test_render_email_empty_list():
    html = render_email([])
    assert "本期暂无新增推荐" in html


def test_render_email_author_truncation():
    authors = [f"Author {i}" for i in range(10)]
    paper = make_sample_paper(authors=authors, score=7.0, tldr="ok")
    html = render_email([paper])
    assert "Author 0" in html
    assert "Author 1" in html
    assert "Author 2" in html
    assert "..." in html
    assert "Author 8" in html
    assert "Author 9" in html
    # Middle authors should be truncated
    assert "Author 5" not in html


def test_render_email_affiliation_truncation():
    affiliations = [f"Uni {i}" for i in range(8)]
    paper = make_sample_paper(affiliations=affiliations, score=7.0, tldr="ok")
    html = render_email([paper])
    assert "Uni 0" in html
    assert "Uni 4" in html
    assert "..." in html
    assert "Uni 7" not in html


def test_render_email_no_affiliations():
    paper = make_sample_paper(affiliations=None, score=7.0, tldr="ok")
    html = render_email([paper])
    assert "Unknown Affiliation" not in html


def test_get_stars_low_score():
    assert get_stars(5.0) == ""
    assert get_stars(6.0) == ""


def test_get_stars_high_score():
    stars = get_stars(8.0)
    assert stars.count("full-star") == 5


def test_get_stars_mid_score():
    stars = get_stars(7.0)
    assert "star" in stars
    assert stars.count("full-star") + stars.count("half-star") > 0


def test_get_block_html_contains_all_fields():
    html = get_block_html("Title", "Auth", "3.5", "Summary", "http://pdf.url", "MIT")
    assert "Title" in html
    assert "Auth" in html
    assert "3.5" in html
    assert "Summary" in html
    assert "http://pdf.url" in html
    assert "MIT" in html


def test_get_empty_html():
    html = get_empty_html()
    assert "本期暂无新增推荐" in html


def test_readable_summary_preserves_validation_and_uncertainty():
    from zotero_arxiv_daily.construct_email import html_to_text
    raw = ('问题：研究目标[摘要确认]。\n方法：实验方法[摘要确认]。\n'
           '结论：测量结果[摘要确认]。\n创新：尚无法确认。\n参考价值：可能有帮助[推测]。')
    paper = make_sample_paper(tldr=raw)
    html = render_email([paper])
    text = html_to_text(html)
    assert '[摘要确认]' not in text
    assert text.count('内容来源：论文摘要') == 1
    assert '（推测）' in text and '尚无法确认' in text
    assert paper.tldr == raw


def test_compact_citations_retain_correspondence_and_escape_body():
    from zotero_arxiv_daily.construct_email import html_to_text
    raw = '结论：<script>unsafe</script>[全文片段确认:E2]。\n参考价值：应用[摘要确认]。'
    paper = make_sample_paper(tldr=raw)
    paper.evidence_excerpts = [{'id':'E1','section':'Unused methods'},
                              {'id':'E2','section':'<Results>'}]
    assert paper.summary_is_acceptable(raw, {})
    html = render_email([paper])
    assert '<script>' not in html and '&lt;script&gt;' in html
    assert '[全文片段确认:E2]' not in html and '[E2]' in html
    assert 'Unused methods' not in html and '&lt;Results&gt;' in html
    assert '论文摘要；正文片段' in html_to_text(html)
    assert paper.tldr == raw and paper.summary_is_acceptable(paper.tldr, {})
