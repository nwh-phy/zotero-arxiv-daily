from .protocol import Paper
import math
import re
from html import escape
from html.parser import HTMLParser
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo


framework = """
<!DOCTYPE HTML>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    .star-wrapper {
      font-size: 1.3em; /* 调整星星大小 */
      line-height: 1; /* 确保垂直对齐 */
      display: inline-flex;
      align-items: center; /* 保持对齐 */
    }
    .half-star {
      display: inline-block;
      width: 0.5em; /* 半颗星的宽度 */
      overflow: hidden;
      white-space: nowrap;
      vertical-align: middle;
    }
    .full-star {
      vertical-align: middle;
    }
  </style>
</head>
<body style="margin:0;padding:16px;background:#fff;color:#333;line-height:1.65;">

<div style="max-width:780px;margin:0 auto;">
    __CONTENT__
</div>

<br><br>
<div>
本简报根据你的 Zotero 收藏推荐。片段编号用于说明模型输入来源，不代表已审阅完整论文。<br>
暂停推送：在仓库 Actions Variables 中将 DELIVERY_ENABLED 设为 false。
</div>

</body>
</html>
"""

def get_empty_html():
  block_template = """
  <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
  <tr>
    <td style="font-size: 20px; font-weight: bold; color: #333;">
        本期暂无新增推荐。
    </td>
  </tr>
  </table>
  """
  return block_template


def _summary_sources(summary, excerpts=()):
    summary = summary or ''
    sources = ['论文摘要'] if '[摘要确认]' in summary else []
    ids = list(dict.fromkeys(re.findall(r'\[全文片段确认:(E\d+)\]', summary)))
    if ids or '[全文片段确认]' in summary:
        sections = {e['id']: e['section'] for e in excerpts}
        details = '；'.join(f'[{ref}] {sections[ref]}' if ref in sections else f'[{ref}]' for ref in ids)
        sources.append('正文片段' + (f'（{details}）' if details else ''))
    return '；'.join(sources) or None


def _summary_html(summary):
    # Keep the validated original untouched; compact only the reader-facing text.
    text = (summary or '').replace('[摘要确认]', '').replace('[全文片段确认]', '')
    text = re.sub(r'\[全文片段确认:(E\d+)\]', r'[\1]', text)
    text = text.replace('[推测]', '（推测）')
    text = escape(text)
    text = re.sub(r'(?m)^(问题|方法|结论|创新|参考价值)(\s*[:：])', r'<strong>\1\2</strong>', text)
    return text.replace('\n', '<br>')

def get_block_html(title:str, authors:str, rate:str, tldr:str, pdf_url:str, affiliations:str=None, url:str=None, publication:str=None, recommendation:str=None, evidence:str=None, source_label:str="arXiv 原文"):
    block_template = """
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
    <tr>
        <td style="font-size: 20px; font-weight: bold; color: #333;">
            <a href="{url}" style="color:#222;text-decoration:none;">{title}</a>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #666; padding: 8px 0;">
            {authors}
            <br>
            {affiliations}
            {publication}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>相关度：</strong> {rate}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>中文总结：</strong> {tldr}
            {recommendation}
            {evidence}
        </td>
    </tr>

    <tr>
        <td style="padding: 8px 0;">
            <a href="{url}">{source_label}</a>
            <a href="{pdf_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #d9534f; padding: 12px 16px; border-radius: 4px;">PDF</a>
        </td>
    </tr>
</table>
"""
    def safe_url(value):
        return escape(value, quote=True) if value and value.startswith(('https://', 'http://')) else '#'
    evidence = evidence or _summary_sources(tldr)
    return block_template.format(title=escape(title), authors=escape(authors), rate=rate, source_label=escape(source_label),
                                 tldr=_summary_html(tldr),
                                 pdf_url=safe_url(pdf_url), url=safe_url(url),
                                 affiliations='<i>' + escape(affiliations) + '</i><br>' if affiliations else '', publication=escape(publication or ''),
                                 recommendation='<br><strong>推荐依据：</strong> ' + escape(recommendation) if recommendation else '',
                                 evidence='<div style="font-size:12px;color:#777;margin-top:10px;">内容来源：' + escape(evidence) + '</div>' if evidence else '')

def get_stars(score:float):
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'
    low = 6
    high = 8
    if score <= low:
        return ''
    elif score >= high:
        return full_star * 5
    else:
        interval = (high-low) / 10
        star_num = math.ceil((score-low) / interval)
        full_star_num = int(star_num/2)
        half_star_num = star_num - full_star_num * 2
        return '<div class="star-wrapper">'+full_star * full_star_num + half_star * half_star_num + '</div>'


def _safe_url(value):
    return escape(value, quote=True) if isinstance(value, str) and value.startswith(('https://', 'http://')) else '#'


def _reading_value_details(paper):
    """Fail closed when an old or partially processed paper reaches the renderer."""
    from .reading_value import validate_judgement
    from .reading_review import validate_review

    if paper.relevance_status != 'scored':
        return None
    if not isinstance(paper.relevance_details, dict):
        return None
    try:
        details = validate_judgement({**paper.relevance_details, 'score': paper.relevance_score,
                                      'reason': paper.relevance_reason}, paper)
    except (TypeError, ValueError):
        return None
    if details['route'] == 'exclude':
        return None
    review = paper.relevance_review
    if not isinstance(review, dict):
        return None
    status = review.get('status')
    if paper.recommendation_tier == 'focus':
        if details['route'] not in {'current', 'method'} or details['score'] < 4 or status != 'confirmed':
            return None
        try:
            checked_review = validate_review(review, paper.evidence_excerpts)
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
        if review.get('evidence_ids') != [checked_review['evidence_id']]:
            return None
        if not paper.summary_is_acceptable(paper.tldr, {}):
            return None
    elif paper.recommendation_tier == 'explore':
        if status not in {None, 'confirmed', 'downgraded', 'unavailable'}:
            return None
    else:
        return None
    return details


def _reading_value_card(paper, details, focus):
    authors = list(paper.authors)
    if len(authors) > 5:
        authors = authors[:3] + ['...'] + authors[-2:]
    published = paper.published or '未提供'
    if paper.published:
        try:
            published = datetime.fromisoformat(paper.published).strftime('%Y-%m-%d')
        except ValueError:
            pass
    label = 'arXiv 发布（UTC）' if paper.source == 'arxiv' else '发布'
    source_label = 'arXiv 原文' if paper.source == 'arxiv' else '论文原文'
    parts = [
        '<table border="0" cellpadding="0" cellspacing="0" width="100%" '
        'style="font-family:Arial,sans-serif;border:1px solid #ddd;border-radius:8px;padding:16px;background:#f9f9f9;">',
        '<tr><td style="font-size:20px;font-weight:bold;">'
        f'<a href="{_safe_url(paper.url)}" style="color:#222;text-decoration:none;">{escape(paper.title)}</a></td></tr>',
        '<tr><td style="font-size:14px;color:#666;padding:8px 0;">'
        + escape(', '.join(authors)) + '<br>'
        + escape(f'{label}：{published}；期刊：{paper.journal_ref or "未提供"}') + '</td></tr>',
    ]
    if paper.affiliations:
        affiliations = list(paper.affiliations[:5]) + (['...'] if len(paper.affiliations) > 5 else [])
        parts.append('<tr><td style="font-size:14px;color:#666;">' + escape(', '.join(affiliations)) + '</td></tr>')
    explore_use_label = ('初筛用途（正文片段未确认）'
                         if paper.relevance_review.get('status') == 'downgraded' else '可以怎样用')
    fields = [('推荐给你的原因', details['use']), ('对应问题', details['target']),
              ('实际贡献', details['contribution'])] if focus else [
                  ('实际贡献', details['contribution']), (explore_use_label, details['use'])]
    for name, value in fields:
        parts.append(f'<tr><td style="padding:4px 0;"><strong>{name}：</strong>{escape(value)}</td></tr>')
    evidence_label = '摘要依据' if ' '.join(details['evidence'].split()) in ' '.join(paper.abstract.split()) else '标题依据'
    parts.append(f'<tr><td style="padding:6px 0;"><strong>{evidence_label}：</strong>“'
                 + escape(details['evidence']) + '”</td></tr>')
    if focus:
        review = paper.relevance_review
        sections = {excerpt['id']: excerpt.get('section', '') for excerpt in paper.evidence_excerpts}
        citations = '；'.join(f'[{ref}] {sections[ref]}' for ref in review['evidence_ids'])
        reason = review.get('reason')
        parts.append('<tr><td style="font-size:13px;color:#666;padding:6px 0;"><strong>正文片段复核：</strong>'
                     + escape(reason if isinstance(reason, str) else '支持上述阅读用途')
                     + '（' + escape(citations) + '）</td></tr>')
        parts.append('<tr><td style="padding:8px 0;"><strong>中文总结：</strong><br>'
                     + _summary_html(paper.tldr) + '</td></tr>')
        if sources := _summary_sources(paper.tldr, paper.evidence_excerpts):
            parts.append('<tr><td style="font-size:12px;color:#777;">内容来源：' + escape(sources) + '</td></tr>')
    else:
        note = '依据标题与摘要筛选，未作正文片段复核。'
        if paper.relevance_review.get('status') in {'downgraded', 'unavailable'}:
            note = '正文片段复核后降为拓展阅读。' if paper.relevance_review['status'] == 'downgraded' else '正文不可用，保留为拓展阅读。'
            if paper.relevance_review['status'] == 'downgraded':
                reason = paper.relevance_review.get('reason')
                if isinstance(reason, str) and reason.strip():
                    note += '复核理由：' + reason.strip()
        elif paper.relevance_review.get('status') == 'confirmed':
            note = '依据标题与摘要保留为拓展阅读；正文片段复核通过。'
        parts.append('<tr><td style="font-size:12px;color:#777;">' + escape(note) + '</td></tr>')
    trace = []
    if paper.recommendation_theme:
        trace.append('匹配收藏夹：' + paper.recommendation_theme)
    if paper.related_papers:
        trace.append('相关收藏：' + '；'.join(paper.related_papers))
    if isinstance(paper.score, (int, float)) and not isinstance(paper.score, bool) and math.isfinite(paper.score):
        trace.append(f'收藏相似度：{paper.score:.1f}')
    if trace:
        parts.append('<tr><td style="font-size:12px;color:#777;padding-top:10px;">'
                     + escape('；'.join(trace)) + '。仅供追溯候选来源。</td></tr>')
    parts.append('<tr><td style="padding-top:10px;">'
                 f'<a href="{_safe_url(paper.url)}">{source_label}</a>'
                 + (f' · <a href="{_safe_url(paper.pdf_url)}">PDF</a>' if paper.pdf_url else '')
                 + '</td></tr></table>')
    return ''.join(parts)


def _group_reading_value(papers, context):
    """Use the same eligibility and configurable caps for preview and delivery."""
    relevance = context.get('relevance') or {}
    limits = {tier: relevance.get(tier + '_limit', default)
              for tier, default in [('focus', 6), ('explore', 2)]}
    if any(isinstance(limit, bool) or not isinstance(limit, int) or limit < 0 for limit in limits.values()):
        raise ValueError('Invalid reading-value display limits')
    groups = {'focus': [], 'explore': []}
    for paper in papers:
        try:
            details = _reading_value_details(paper)
        except (AttributeError, KeyError, TypeError, ValueError):
            details = None
        if details:
            group = groups[paper.recommendation_tier]
            if len(group) < limits[paper.recommendation_tier]:
                group.append((paper, details))
    return groups


def assert_reading_value_delivery(papers, context):
    """Reject incomplete or hidden selections before SMTP/history preparation.

    Previews may omit invalid/over-limit papers. A delivery must display every
    supplied identity exactly once, with both screening stages complete.
    """
    from .history import paper_identity

    message = 'Reading-value delivery validation failed; email was not sent.'
    try:
        relevance, review = context.get('relevance') or {}, context.get('reading_review') or {}
        if relevance.get('policy') != 'reading_value' or any(
                summary.get('complete') is not True or summary.get('incomplete') is True
                for summary in (relevance, review)):
            raise ValueError(message)
        groups = _group_reading_value(papers, context)
        if sum(len(group) for group in groups.values()) != len(papers):
            raise ValueError(message)
        identities = [paper_identity(paper) for paper in papers]
        if len(set(identities)) != len(identities):
            raise ValueError(message)
    except (AttributeError, KeyError, TypeError, ValueError):
        # Never include a title, model draft or untrusted field in the exception.
        raise ValueError(message) from None


def _render_reading_value(papers, context):
    relevance, review = context.get('relevance') or {}, context.get('reading_review') or {}
    groups = _group_reading_value(papers, context)
    incomplete = any(summary.get('complete') is False or summary.get('incomplete') is True
                     for summary in (relevance, review))
    today = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d')
    content = f'<h1>科研文献简报 · {today}（北京时间）</h1>'
    if incomplete:
        content += '<p><strong>筛选未完成，暂不投递。</strong>以下仅为已完成项目的部分预览。</p>'
    content += (f'<p>重点推荐 {len(groups["focus"])} 篇；拓展阅读 {len(groups["explore"])} 篇。'
                '按阅读价值分组，组内保留筛选顺序；名额不补足。</p>')
    if previous := context.get('previous_delivery'):
        content += f'<p>上期推荐 {int(previous["delivered"])} 篇，目前已有 {int(previous["saved"])} 篇在你的 Zotero 中。</p>'
    for source, window in (context.get('windows') or {}).items():
        start = datetime.fromisoformat(window['start_utc']).astimezone(ZoneInfo('Asia/Shanghai'))
        end = datetime.fromisoformat(window['end_utc_exclusive']).astimezone(ZoneInfo('Asia/Shanghai'))
        content += f'<p>{escape(source)} 覆盖：{start:%Y-%m-%d %H:%M} 至 {end:%Y-%m-%d %H:%M}（北京时间，结束时间不含）。</p>'
    if 'retrieved_count' in context:
        content += (f'<p>抓取 {int(context["retrieved_count"])} 篇；'
                    f'新增候选 {int(context.get("candidate_count", context["retrieved_count"]))} 篇。</p>')
    if not any(groups.values()) and not incomplete:
        content += get_empty_html()
    for tier, heading in [('focus', '重点推荐'), ('explore', '拓展阅读')]:
        if groups[tier]:
            content += f'<h2>{heading}</h2>' + '<br><br>'.join(
                _reading_value_card(paper, details, tier == 'focus') for paper, details in groups[tier])
    return framework.replace('__CONTENT__', content)


def render_email(papers:list[Paper], context: dict | None = None) -> str:
    parts = []
    context = context or {}
    if (context.get('relevance') or {}).get('policy') == 'reading_value':
        return _render_reading_value(papers, context)
    if len(papers) == 0:
        parts.append(get_empty_html())
    
    for p in papers:
        #rate = get_stars(p.score)
        rate = round(p.score, 1) if p.score is not None else '未计算'
        author_list = [a for a in p.authors]
        num_authors = len(author_list)
        if num_authors <= 5:
            authors = ', '.join(author_list)
        else:
            authors = ', '.join(author_list[:3] + ['...'] + author_list[-2:])
        if p.affiliations is not None:
            affiliations = p.affiliations[:5]
            affiliations = ', '.join(affiliations)
            if len(p.affiliations) > 5:
                affiliations += ', ...'
        else:
            affiliations = None
        published = p.published or '未提供'
        if p.published:
            try:
                published = datetime.fromisoformat(p.published).strftime('%Y-%m-%d')
            except ValueError:
                pass
        label = 'arXiv 发布（UTC）' if p.source == 'arxiv' else '发布'
        publication = f"{label}：{published}；期刊：{p.journal_ref or '未提供'}"
        recommendation = None
        if p.related_papers:
            theme = f"主题：{p.recommendation_theme}；" if p.recommendation_theme else ''
            recommendation = theme + '相关收藏：' + '；'.join(p.related_papers)
        judged = None
        if p.relevance_score is not None:
            judged = f'LLM 判断 {p.relevance_score}/5：{p.relevance_reason}'
        elif p.relevance_status in ('failed', 'skipped'):
            judged = 'LLM 评分不可用，按相似度补选'
        if judged:
            recommendation = judged + ('；' + recommendation if recommendation else '')
        evidence = _summary_sources(p.tldr, p.evidence_excerpts)
        parts.append(get_block_html(p.title, authors, rate, p.tldr, p.pdf_url, affiliations, p.url, publication,
                                    recommendation, evidence, 'arXiv 原文' if p.source == 'arxiv' else '论文原文'))

    today = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d')
    ordering = '综合相关度、主题覆盖与内容重复程度排列' if any(p.recommendation_theme for p in papers) else '按相关度排序'
    relevance = context.get('relevance') or {}
    if relevance.get('mode') == 'applied':
        ordering = (f"先按 Zotero 收藏相似度初筛 {int(relevance['attempted'])} 篇，LLM 按研究画像成功评分 "
                    f"{int(relevance['scored'])} 篇，其中 {int(relevance['passed_threshold'])} 篇达到 "
                    f"{int(relevance['min_score'])} 分，按判断分数排列")
        if relevance.get('selected_unscored'):
            ordering += f"；另有 {int(relevance['selected_unscored'])} 篇评分不可用，按相似度补选"
    elif relevance.get('mode') == 'fallback':
        ordering += f"（LLM 仅成功评分 {int(relevance['scored'])}/{int(relevance['attempted'])} 篇，本期按相似度排序）"
    content = f'<h1>科研文献简报 · {today}（北京时间）</h1><p>推荐 {len(papers)} 篇，{ordering}。</p>'
    if previous := context.get('previous_delivery'):
        content += f"<p>上期推荐 {int(previous['delivered'])} 篇，目前已有 {int(previous['saved'])} 篇在你的 Zotero 中。</p>"
    windows = context.get('windows', {})
    for source, window in windows.items():
        start = datetime.fromisoformat(window['start_utc']).astimezone(ZoneInfo('Asia/Shanghai'))
        end = datetime.fromisoformat(window['end_utc_exclusive']).astimezone(ZoneInfo('Asia/Shanghai'))
        content += f'<p>{escape(source)} 覆盖：{start:%Y-%m-%d %H:%M} 至 {end:%Y-%m-%d %H:%M}（北京时间，结束时间不含）。</p>'
    if 'retrieved_count' in context:
        content += (f"<p>抓取 {int(context['retrieved_count'])} 篇；"
                    f"新增候选 {int(context.get('candidate_count', context['retrieved_count']))} 篇。</p>")
    themes = Counter(p.recommendation_theme for p in papers if p.recommendation_theme)
    if themes:
        content += '<p>本期主题：' + escape('；'.join(f'{name} {count}篇' for name, count in themes.items())) + '</p>'
    content += '<br>' + '<br><br>'.join(parts)
    return framework.replace('__CONTENT__', content)


class _TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {'head', 'style', 'script'}:
            self.hidden += 1
        elif not self.hidden:
            if tag in {'br', 'p', 'tr', 'h1', 'h2', 'h3', 'div', 'table'}:
                self.parts.append('\n')
            if tag == 'a':
                href = dict(attrs).get('href', '')
                if href.startswith(('https://', 'http://')):
                    self.parts.append(href + ' ')

    def handle_endtag(self, tag):
        if tag in {'head', 'style', 'script'}:
            self.hidden = max(0, self.hidden - 1)
        elif not self.hidden and tag in {'p', 'tr', 'h1', 'h2', 'h3', 'div', 'table'}:
            self.parts.append('\n')

    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.parts.append(data.strip() + ' ')


def html_to_text(html: str) -> str:
    parser = _TextParser()
    parser.feed(html)
    return '\n'.join(line.strip() for line in ''.join(parser.parts).splitlines() if line.strip())
