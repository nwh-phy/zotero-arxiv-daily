"""Retry body review and strict summaries from a completed archived screening.

  uv run --frozen scripts/replay_body_review.py evaluation.json --report report.json \
      --live --output verification/body-review-retry

This is an isolated live body/summary preview, not a fresh retrieval/ranking run.
It never loads .env, Zotero, SMTP, embeddings or delivery history. Only the
originally selected focus papers receive body downloads/review; reserves never
fill vacancies. The source files remain read-only and an existing output is refused.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from time import monotonic

from arxiv import Result
from loguru import logger
from omegaconf import OmegaConf
from openai import OpenAI

from zotero_arxiv_daily.construct_email import assert_reading_value_delivery, html_to_text, render_email
from zotero_arxiv_daily.evidence import EVIDENCE_VERSION
from zotero_arxiv_daily.evaluation import restore_candidates
from zotero_arxiv_daily.history import paper_identity
from zotero_arxiv_daily.progress import PaperProgress
from zotero_arxiv_daily.reading_review import ReadingReview, PROMPT_DIGEST as REVIEW_PROMPT_DIGEST, REVIEW_VERSION
from zotero_arxiv_daily.reading_value import PROMPT_DIGEST, READING_VALUE_VERSION, choose_reading_value, validate_judgement
from zotero_arxiv_daily.retriever.arxiv_retriever import extract_text_from_html, extract_text_from_pdf, extract_text_from_tar
from zotero_arxiv_daily.shortlist import build_shortlist

ROOT = Path(__file__).resolve().parents[1]
NOTICE = ('存档正文复核重试预览：沿用已完成的检索、CPU排序和标题/摘要判断，'
          '仅重新提取原重点候选的正文片段、复核并生成合格摘要；未重新检索或排序。'
          '不发送邮件，不读取或更新投递历史，不检验真实阅读质量。')
_PAPER_ID = r'(?:\d{2}(?:0[1-9]|1[0-2])\.\d{4,5}|[a-z][a-z.-]*(?:\.[A-Z]{2})?/\d{2}(?:0[1-9]|1[0-2])\d{3})(?:v[1-9]\d*)?'
_ARXIV_URL = re.compile(r'https?://arxiv\.org/abs/(' + _PAPER_ID + r')\Z')
_LLM_FIELDS = ('api_mode', 'generation_kwargs', 'language', 'summary_instructions',
               'strict_summary', 'summary_min_chars', 'summary_max_chars')


class ReplayRejected(ValueError):
    """Fixed diagnostic codes, never upstream/provider exception messages."""
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--report', type=Path, required=True, help='Matching archived report.json')
    parser.add_argument('--live', action='store_true', required=True, help='Explicitly allow arXiv and LLM requests')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; progress stays inside it')
    parser.add_argument('--config-dir', type=Path, default=ROOT / 'config')
    return parser


def _plain(value):
    if isinstance(value, dict):
        return all(_plain(item) for item in value.values())
    if isinstance(value, list):
        return all(_plain(item) for item in value)
    return not isinstance(value, str) or ('${' not in value and value != '???')


def load_config(directory):
    """Discard unrelated services before resolving any public settings."""
    selected = []
    for name in ('base.yaml', 'custom.yaml'):
        raw = OmegaConf.to_container(OmegaConf.load(directory / name), resolve=False)
        if not isinstance(raw, dict) or any(not isinstance(raw.get(key, {}), dict)
                                           for key in ('llm', 'reranker', 'executor')):
            raise ReplayRejected('invalid_configuration')
        llm = raw.get('llm', {})
        selected.append({'llm': {key: llm[key] for key in _LLM_FIELDS if key in llm},
                         'relevance': raw.get('reranker', {}).get('relevance', {}),
                         **({'limit': raw['executor']['max_paper_num']}
                            if 'max_paper_num' in raw.get('executor', {}) else {})})
    public = OmegaConf.to_container(OmegaConf.merge(*selected), resolve=False)
    if not _plain(public):
        raise ReplayRejected('nonpublic_configuration')
    options, params, limit = public['relevance'], public['llm'], public.get('limit')
    if (not isinstance(options, dict) or options.get('enabled') is not True
            or options.get('policy') != 'reading_value'
            or isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
            or params.get('api_mode') not in ('response', 'chat_completion')
            or not isinstance(params.get('generation_kwargs'), dict)
            or not isinstance(params['generation_kwargs'].get('model'), str)
            or not params['generation_kwargs']['model'].strip()):
        raise ReplayRejected('invalid_screening_configuration')
    if (params.get('strict_summary') is not True or params.get('summary_min_chars') != 150
            or params.get('summary_max_chars') != 250
            or not isinstance(params.get('summary_instructions'), str)
            or not params['summary_instructions'].strip()):
        raise ReplayRejected('strict_summary_configuration_required')
    return options, params, limit


def _canonicalize(papers):
    """Never trust archived network targets, PDF links, userinfo, ports or query strings."""
    seen = set()
    for paper in papers:
        match = _ARXIV_URL.fullmatch(paper.url) if paper.source == 'arxiv' else None
        if not match or paper_identity(paper) in seen:
            raise ReplayRejected('invalid_or_duplicate_arxiv_candidate')
        seen.add(paper_identity(paper))
        paper.url = 'https://arxiv.org/abs/' + match.group(1)
        paper.pdf_url = 'https://arxiv.org/pdf/' + match.group(1)


def restore_selection(bundle, archived, options, params, limit):
    """Validate provenance and complete judgements, then recover the pre-review set."""
    settings = bundle.get('settings') or {}
    previous = settings.get('relevance') or {}
    if not isinstance(settings, dict) or not isinstance(previous, dict):
        raise ReplayRejected('missing_screening_provenance')
    if (settings.get('llm_model') != params['generation_kwargs']['model']
            or settings.get('max_paper_num') != limit):
        raise ReplayRejected('screening_model_or_limit_mismatch')
    if previous.get('prompt_digest') != PROMPT_DIGEST or previous.get('version') != READING_VALUE_VERSION:
        raise ReplayRejected('screening_prompt_mismatch')
    # Body-review code, prompt and budgets may change without redoing abstract screening.
    original_options = {key: value for key, value in previous.items()
                        if key not in ('prompt_digest', 'version', 'fulltext_review')}
    current_options = {key: value for key, value in options.items() if key != 'fulltext_review'}
    if original_options != current_options:
        raise ReplayRejected('screening_configuration_mismatch')
    relevance = archived.get('relevance') or {}
    if (relevance.get('policy') != 'reading_value' or relevance.get('complete') is not True
            or relevance.get('incomplete') is not False or relevance.get('failed') != 0
            or relevance.get('skipped') != 0 or bundle.get('window') != archived.get('windows')):
        raise ReplayRejected('archived_screening_incomplete_or_unpaired')
    papers = restore_candidates(bundle)
    _canonicalize(papers)
    if len(papers) != archived.get('candidate_count'):
        raise ReplayRejected('archived_candidate_count_mismatch')
    # Require the report to describe the same public judgements, including the failed reviews.
    rows = archived.get('candidates')
    if not isinstance(rows, list) or len(rows) != len(papers):
        raise ReplayRejected('archived_report_candidates_mismatch')
    fields = ('id', 'title', 'relevance_score', 'relevance_reason', 'relevance_status',
              'relevance_details', 'recommendation_tier', 'relevance_review', 'shortlist_sources', 'selected')
    by_id = {row.get('id'): row for row in rows if isinstance(row, dict)}
    if len(by_id) != len(rows) or any(
            {key: row.get(key) for key in fields} !=
            {key: by_id.get(row.get('id'), {}).get(key) for key in fields}
            for row in bundle['candidates']):
        raise ReplayRejected('archived_report_candidates_mismatch')
    shortlist, recall = build_shortlist(papers, options)
    if (relevance.get('attempted') != len(shortlist) or relevance.get('scored') != len(shortlist)
            or (relevance.get('shortlist') or {}).get('selected') != len(shortlist)):
        raise ReplayRejected('archived_shortlist_incomplete')
    shortlist_ids = {id(paper) for paper in shortlist}
    for paper in papers:
        if id(paper) in shortlist_ids:
            if paper.relevance_status != 'scored':
                raise ReplayRejected('archived_shortlist_incomplete')
            validate_judgement({**paper.relevance_details, 'score': paper.relevance_score,
                                'reason': paper.relevance_reason}, paper)
        elif paper.relevance_status is not None:
            raise ReplayRejected('archived_shortlist_inconsistent')
        paper.relevance_review = {}
        paper.recommendation_tier = None
    return papers, choose_reading_value(shortlist, options, limit), recall


def extract_body(paper):
    # Result is a download container only. Never copy its default author/date metadata.
    raw = Result(entry_id=paper.url, title=paper.title, links=[
        Result.Link(href=paper.pdf_url, title='pdf', rel='related'),
        Result.Link(href=paper.url.replace('/abs/', '/src/'), title='source', rel='related')])
    # The generic extractor logs subprocess error strings. Suppress that logger here
    # so a failed download contributes only safe aggregate diagnostics to this replay.
    module = 'zotero_arxiv_daily.retriever.arxiv_retriever'
    logger.disable(module)
    try:
        for extractor in (extract_text_from_tar, extract_text_from_html, extract_text_from_pdf):
            text = extractor(raw)
            if text:
                paper.full_text = text
                return
    finally:
        logger.enable(module)


def _record(paper):
    return {'id': paper_identity(paper), 'title': paper.title, 'url': paper.url,
            'recommendation_tier': paper.recommendation_tier, 'relevance_score': paper.relevance_score,
            'relevance_reason': paper.relevance_reason, 'relevance_details': paper.relevance_details,
            'relevance_review': paper.relevance_review, 'summary': paper.tldr,
            'evidence_sections': [{'id': item['id'], 'section': item['section']} for item in paper.evidence_excerpts]}


def _write(output, report, selected, focus):
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    evidence = [{**_record(paper), 'abstract': paper.abstract, 'excerpts': paper.evidence_excerpts} for paper in focus]
    (output / 'review-evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    html = render_email(selected, report)
    warning = '' if report['status'] == 'body_replay_completed' else '<p><strong>本次重试未完成，暂不投递。</strong></p>'
    html = html.replace(
        '<div style="max-width:780px;margin:0 auto;">',
        '<div style="max-width:780px;margin:0 auto;"><p><strong>' + NOTICE + '</strong></p>' + warning, 1)
    (output / 'brief.html').write_text(html, encoding='utf-8')
    (output / 'brief.txt').write_text(html_to_text(html), encoding='utf-8')
    (output / 'README.txt').write_text(NOTICE + '\n\n'
        'report.json：重试状态、归档输入哈希、原名单与本次名单、失败类型。\n'
        'review-evidence.json：原重点候选、实际提供的E编号正文片段、复核判断和严格摘要，供人工核对。\n'
        'brief.html / brief.txt：带有存档重试说明的诊断预览。失败状态不得作为完整验证。\n'
        'progress/：本次独立缓存；不会读取正式 .paper-progress 或投递历史。\n'
        '存档未保存作者、发表日期，预览不补造；下载容器的默认元数据不进入产物。\n'
        '已校验归档模型、研究画像、摘要筛选提示词摘要/版本及筛选配置；'
        '旧评估包未保存API模式和完整生成参数，不能据此核对这两项历史值。\n', encoding='utf-8')


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        print(f'Refusing unavailable or existing output directory ({type(exc).__name__}).')
        return 2
    started = monotonic()
    selected, focus = [], []
    report = {'schema': 1, 'status': 'processing', 'mode': 'archived_body_review_retry',
              'evidence_level': 'archived_body_summary_preview', 'notice': NOTICE, 'dry_run': True,
              'fresh_retrieval': False, 'fresh_ranking': False, 'abstract_screening_reused': True,
              'email_sent': False, 'delivery_history_accessed': False, 'reserve_backfill': False,
              'metadata_unavailable': ['authors', 'published'], 'source': {}, 'papers': [],
              'relevance': {'policy': 'reading_value', 'complete': False, 'incomplete': True},
              'reading_review': {'complete': False}, 'summaries_generated': 0, 'failed_count': 0,
              'failures': [], 'stage': 'validation', 'timings': {}}
    try:
        sources = {}
        for name, path in [('evaluation', args.bundle), ('report', args.report)]:
            payload = path.read_bytes()
            sources[name] = json.loads(payload)
            report['source'][name + '_sha256'] = hashlib.sha256(payload).hexdigest()
        bundle, archived = sources['evaluation'], sources['report']
        options, params, limit = load_config(args.config_dir)
        papers, proposed, recall = restore_selection(bundle, archived, options, params, limit)
        focus = [paper for paper in proposed if paper.recommendation_tier == 'focus']
        selected = proposed
        report.update(original_selection=[{'id': paper_identity(p), 'tier': p.recommendation_tier} for p in proposed],
                      archived_candidate_count=len(papers), windows=bundle.get('window'),
                      archived_source_status=archived.get('status'), shortlist=recall,
                      provenance={'model': params['generation_kwargs']['model'],
                                  'abstract_prompt_digest': PROMPT_DIGEST, 'abstract_version': READING_VALUE_VERSION,
                                  'current_api_mode': params['api_mode'],
                                  'current_public_llm_policy_sha256': hashlib.sha256(
                                      json.dumps(params, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                                  'summary_instructions_sha256': hashlib.sha256(params['summary_instructions'].encode()).hexdigest(),
                                  'strict_summary': True, 'summary_min_chars': 150, 'summary_max_chars': 250,
                                  'review_prompt_digest': REVIEW_PROMPT_DIGEST, 'review_version': REVIEW_VERSION,
                                  'evidence_version': EVIDENCE_VERSION,
                                  'screening_configuration_matched': True,
                                  'historical_api_mode_and_generation_parameters': 'not_recorded'})
        report['relevance'] = {**archived['relevance'], 'focus_limit': options.get('focus_limit', 6),
                               'explore_limit': options.get('explore_limit', 2)}
        key, base_url = os.environ.get('OPENAI_API_KEY'), os.environ.get('OPENAI_API_BASE')
        if not key or not base_url:
            raise ReplayRejected('missing_llm_credentials')
        params['api'] = {'key': key, 'base_url': base_url}
        cache = PaperProgress(args.output / 'progress', 'isolated-body-review-replay')
        client = OpenAI(api_key=key, base_url=base_url, timeout=90, max_retries=1)
        try:
            report['stage'] = 'reading_review'
            stage_started = monotonic()
            reviewer = ReadingReview(client, params, options.get('fulltext_review') or {}, cache)
            report['reading_review'] = reviewer.stats
            for paper in focus:
                attempted = reviewer.stats['attempted']
                try:
                    extract_body(paper)
                    cache.save(paper, params, status='extracted')
                    reviewer.review(paper)
                except Exception as exc:
                    if reviewer.stats['attempted'] == attempted:
                        reviewer.stats['attempted'] += 1
                    reviewer._finish(paper, {'status': 'failed', 'reason': '正文提取或复核失败，等待重试。'})
                    report['failures'].append({'id': paper_identity(paper), 'stage': 'reading_review',
                                               'error_type': type(exc).__name__})
            report['timings']['reading_review'] = round(monotonic() - stage_started, 2)
            selected = choose_reading_value(proposed, options, limit)
            if not reviewer.stats['complete']:
                report['status'] = 'review_incomplete'
            else:
                report['stage'] = 'summaries'
                stage_started = monotonic()
                for paper in selected:
                    if paper.recommendation_tier != 'focus':
                        continue
                    try:
                        paper.generate_tldr(client, params)
                        if not paper.summary_is_acceptable(paper.tldr, params):
                            raise ReplayRejected('summary_validation_failed')
                        cache.save(paper, params, status='complete')
                        report['summaries_generated'] += 1
                    except Exception as exc:
                        cache.save(paper, params, status='failed', error_type=type(exc).__name__)
                        report['failed_count'] += 1
                        report['failures'].append({'id': paper_identity(paper), 'stage': 'summaries',
                                                   'error_type': type(exc).__name__})
                report['timings']['summaries'] = round(monotonic() - stage_started, 2)
                report['status'] = 'summary_failed' if report['failed_count'] else 'body_replay_completed'
                if report['status'] == 'body_replay_completed':
                    assert_reading_value_delivery(selected, report)
        finally:
            client.close()
    except Exception as exc:
        report['status'] = 'failed'
        report['failure'] = {'error_type': type(exc).__name__}
        if isinstance(exc, ReplayRejected):
            report['failure']['code'] = exc.code
    report.update(selected_count=len(selected),
                  focus_count=sum(p.recommendation_tier == 'focus' for p in selected),
                  explore_count=sum(p.recommendation_tier == 'explore' for p in selected),
                  papers=[_record(paper) for paper in selected],
                  duration_seconds=round(monotonic() - started, 2), failed_stage=report['stage'])
    if report['status'] == 'body_replay_completed':
        report['stage'] = report['failed_stage'] = None
    report['relevance'].update(selected=len(selected), focus_count=report['focus_count'],
                               explore_count=report['explore_count'])
    _write(args.output, report, selected, focus)
    print(f'Archived body/summary preview: {report["status"]}; isolated reports retained.')
    return 0 if report['status'] == 'body_replay_completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
