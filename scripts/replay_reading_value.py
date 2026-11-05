"""Compare abstract screening with an archived evaluation bundle, without delivery.

  uv run --frozen scripts/replay_reading_value.py evaluation.json --plan-only --output verification/recall-plan
  uv run --frozen scripts/replay_reading_value.py evaluation.json --live --output verification/reading-value-preview

An explicit mode is required. Plan mode builds the shortlist with zero API calls.
Live mode makes new LLM judgements using only OPENAI_API_KEY/OPENAI_API_BASE from
the process environment. It never loads .env, Zotero, embeddings, SMTP or delivery
history. Both modes require a new output directory and leave the input untouched.
These are abstract screening previews; full-text review and reading-value quality
improvement require separate verification.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from omegaconf import OmegaConf
from openai import OpenAI

from zotero_arxiv_daily.evaluation import restore_candidates
from zotero_arxiv_daily.history import paper_identity
from zotero_arxiv_daily.progress import PaperProgress
from zotero_arxiv_daily.relevance import select
from zotero_arxiv_daily.shortlist import build_shortlist


ROOT = Path(__file__).resolve().parents[1]
PREVIEW_NOTICE = ('仅为标题和摘要筛选预览；未执行全文复核，未检验真实阅读质量是否提高。'
                  'plan-only 模式不产生新的推荐名单。')


class MissingCredentials(ValueError):
    pass


def _parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bundle', type=Path, help='Read-only evaluation.json from an earlier run')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan-only', action='store_true', help='Build the candidate shortlist with zero API calls')
    mode.add_argument('--live', action='store_true', help='Explicitly allow LLM abstract-screening API requests')
    parser.add_argument('--output', type=Path, required=True, help='New isolated output directory; existing paths are refused')
    parser.add_argument('--base-config', type=Path, default=ROOT / 'config/base.yaml')
    parser.add_argument('--custom-config', type=Path, default=ROOT / 'config/custom.yaml')
    parser.add_argument('--config-dir', type=Path, help='Directory containing base.yaml/custom.yaml; overrides their default paths')
    parser.add_argument('--limit', type=int, help='Override executor.max_paper_num for this preview')
    return parser


def load_screening_config(base_path, custom_path, limit=None):
    """Read only public screening settings; never resolve other services' secrets.

    We intentionally do not use Hydra's application composition or resolve the
    complete config. In particular, llm.api and all Zotero/email settings are
    discarded before merging. Interpolations in screening settings are refused
    rather than resolving arbitrary environment variables or user resolvers.
    """
    selected = []
    for path in (base_path, custom_path):
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
        if not isinstance(raw, dict):
            raise ValueError('Screening configuration must be a mapping')
        llm = raw.get('llm') or {}
        reranker = raw.get('reranker') or {}
        executor = raw.get('executor') or {}
        if not all(isinstance(section, dict) for section in (llm, reranker, executor)):
            raise ValueError('Invalid screening configuration section')
        selected.append({'llm': {key: llm[key] for key in ('api_mode', 'generation_kwargs') if key in llm},
                         'relevance': reranker.get('relevance') or {},
                         **({'limit': executor['max_paper_num']} if 'max_paper_num' in executor else {})})
    merged = OmegaConf.to_container(OmegaConf.merge(*selected), resolve=False)

    def check_plain(value):
        if isinstance(value, dict):
            for item in value.values():
                check_plain(item)
        elif isinstance(value, list):
            for item in value:
                check_plain(item)
        elif isinstance(value, str) and ('${' in value or value == '???'):
            raise ValueError('Screening settings must be explicit public values')

    check_plain(merged)
    options = merged['relevance']
    if not isinstance(options, dict):
        raise ValueError('Relevance options must be a mapping')
    options.update(policy='reading_value', enabled=True)
    limit = merged.get('limit', 8) if limit is None else limit
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError('Preview limit must be a positive integer')
    params = merged['llm']
    if (params.get('api_mode', 'chat_completion') not in ('response', 'chat_completion')
            or not isinstance(params.get('generation_kwargs'), dict)
            or not isinstance(params['generation_kwargs'].get('model'), str)
            or not params['generation_kwargs']['model'].strip()):
        raise ValueError('An explicit LLM model and valid API mode are required')
    return options, params, limit


def _screening_snapshot(options, params, limit):
    """Fingerprint explicit public policy/model settings, excluding API configuration."""
    from zotero_arxiv_daily.reading_value import PROMPT_DIGEST, READING_VALUE_VERSION

    top_k = options.get('top_k', 60)
    shortlist = options.get('shortlist') or {}
    embedding_k = shortlist.get('embedding_k', min(40, top_k))
    relevance = {key: options.get(key, default) for key, default in {
        'policy': 'reading_value', 'enabled': True, 'profile': None, 'top_k': 60,
        'focus_limit': 6, 'explore_limit': 2, 'time_budget_seconds': 900,
        'request_timeout': 90, 'max_retries': 1, 'max_consecutive_failures': 5,
    }.items()}
    relevance['shortlist'] = {
        'embedding_k': embedding_k,
        'core_k': shortlist.get('core_k', min(20, max(0, top_k - embedding_k))),
        'core_patterns': shortlist.get('core_patterns') or [],
    }
    # Explicit allowlist: free-form metadata, headers, endpoint and credential
    # fields must never appear in this saved provenance, including nested fields.
    generation = {key: value for key, value in params.get('generation_kwargs', {}).items() if key in {
        'model', 'max_tokens', 'max_output_tokens', 'temperature', 'top_p', 'seed', 'stop',
        'frequency_penalty', 'presence_penalty', 'logit_bias', 'n', 'reasoning_effort',
        'verbosity', 'logprobs', 'top_logprobs', 'truncation', 'service_tier',
    }}
    reasoning = params.get('generation_kwargs', {}).get('reasoning')
    if isinstance(reasoning, dict):
        generation['reasoning'] = {key: reasoning[key] for key in ('effort', 'summary') if key in reasoning}
    max_tokens = options.get('max_tokens')
    if max_tokens is None:
        max_tokens = generation.get('max_tokens')
        if max_tokens is None and generation.get('max_output_tokens') is None:
            max_tokens = 4096
    else:
        generation.pop('max_output_tokens', None)
    if max_tokens is not None:
        generation['max_tokens'] = max_tokens
    api_mode = params.get('api_mode', 'chat_completion')
    if api_mode == 'response':
        tokens = generation.pop('max_tokens', None)
        if tokens is not None and 'max_output_tokens' not in generation:
            generation['max_output_tokens'] = tokens
    settings = {'relevance': relevance, 'limit': limit,
                'llm': {'api_mode': api_mode, 'generation_kwargs': generation}}

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                         separators=(',', ':')).encode()).hexdigest()

    snapshot = {'settings': settings, 'settings_sha256': digest(settings),
                'reading_value_version': READING_VALUE_VERSION, 'prompt_digest': PROMPT_DIGEST,
                'model': generation['model']}
    snapshot['fingerprint'] = digest(snapshot)
    return snapshot


def _fresh_judgements(papers):
    # Old scores remain in the input/comparison only; this run makes a new policy judgement.
    for paper in papers:
        paper.relevance_score = paper.relevance_reason = paper.relevance_status = None
        paper.relevance_details = {}
        paper.recommendation_tier = None
        paper.relevance_review = {}
        paper.shortlist_sources = []


def _record(paper, rank, selected_rank=None):
    return {'id': paper_identity(paper), 'stage1_rank': rank, 'title': paper.title,
            'abstract': paper.abstract, 'selected_rank': selected_rank,
            'score': paper.relevance_score, 'reason': paper.relevance_reason,
            'status': paper.relevance_status, 'details': paper.relevance_details,
            'tier': paper.recommendation_tier, 'review': paper.relevance_review,
            'shortlist_sources': list(paper.shortlist_sources)}


def _comparison(bundle, papers, shortlist, selected, *, mode, status, summary, recall, source,
                screening=None, failure=None):
    old = {}
    for candidate in sorted(bundle.get('candidates', []), key=lambda item: item['stage1_rank']):
        old.setdefault(candidate['id'], candidate)
    new_ranks = {paper_identity(p): i + 1 for i, p in enumerate(selected)}
    old_selected = [candidate for candidate in old.values() if candidate.get('selected')]
    old_selected.sort(key=lambda item: (item.get('selected_rank') or item['stage1_rank']))
    old_ids = {candidate['id'] for candidate in old_selected}
    new_ids = set(new_ranks)
    judged = mode == 'live' and status in ('completed', 'incomplete')
    rows = []
    seen = set()
    for paper in papers:
        identity = paper_identity(paper)
        if identity in seen:
            continue
        seen.add(identity)
        previous = old[identity]
        row = _record(paper, previous['stage1_rank'], new_ranks.get(identity))
        before, after = identity in old_ids, identity in new_ids
        change = ('retained' if before and after else 'added' if after else
                  'removed' if before else 'unselected') if judged else 'pending'
        row.update(old_selected=before, old_selected_rank=previous.get('selected_rank'),
                   old_score=previous.get('relevance_score'), old_reason=previous.get('relevance_reason'),
                   old_status=previous.get('relevance_status'),
                   new_selected=after if judged else None, change=change,
                   new_from_stage1_beyond30=(change == 'added' and previous['stage1_rank'] > 30))
        rows.append(row)
    ranks = {candidate['id']: candidate['stage1_rank'] for candidate in old.values()}
    added = [paper_identity(p) for p in selected if paper_identity(p) not in old_ids] if judged else []
    return {'schema': 1, 'mode': mode, 'status': status, 'notice': PREVIEW_NOTICE,
            'evidence_level': 'abstract_screening_preview',
            'full_text_review': {'performed': False, 'status': 'not_run'},
            'quality_improvement': 'not_evaluated', 'source': source, 'screening': screening,
            'complete': False if status == 'failed' else summary.get('complete') if mode == 'live' else None,
            'incomplete': True if status == 'failed' else summary.get('incomplete') if mode == 'live' else None,
            'failure': failure, 'relevance': summary, 'shortlist': recall,
            'planned_shortlist': [_record(p, ranks[paper_identity(p)]) for p in shortlist],
            'old_selected': [{'id': c['id'], 'title': c['title'], 'stage1_rank': c['stage1_rank'],
                              'selected_rank': c.get('selected_rank'),
                              'score': c.get('relevance_score'), 'reason': c.get('relevance_reason')}
                             for c in old_selected],
            'new_selected': [_record(p, ranks[paper_identity(p)], i + 1) for i, p in enumerate(selected)],
            'changes': {'computed': judged, 'added': added,
                        'removed': [c['id'] for c in old_selected if c['id'] not in new_ids] if judged else [],
                        'retained': [paper_identity(p) for p in selected if paper_identity(p) in old_ids] if judged else [],
                        'added_from_stage1_beyond30': [identity for identity in added if ranks[identity] > 30]},
            'candidates': rows}


def _write_reports(output, comparison):
    (output / 'comparison.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding='utf-8')
    columns = ['id', 'stage1_rank', 'title', 'old_selected', 'old_selected_rank', 'old_score', 'old_reason',
               'new_selected', 'selected_rank', 'change', 'new_from_stage1_beyond30', 'score', 'reason',
               'status', 'tier', 'route', 'contribution', 'target', 'use', 'evidence', 'shortlist_sources',
               'review_status']
    with (output / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in comparison['candidates']:
            writer.writerow({**row, **row['details'], 'shortlist_sources': ';'.join(row['shortlist_sources']),
                             'review_status': row['review'].get('status')})
    # Blind labels deliberately omit both algorithms' choices, scores and reasons.
    review_ids = {row['id'] for row in comparison['planned_shortlist']} | {
        row['id'] for row in comparison['old_selected']}
    with (output / 'human_labels.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'title', 'abstract', 'label', 'notes'])
        for row in comparison['candidates']:
            if row['id'] in review_ids:
                writer.writerow([row['id'], row['title'], row['abstract'], '', ''])
    (output / 'README.txt').write_text(
        comparison['notice'] + '\n\ncomparison.json / comparison.csv：旧名单、新名单与增删保留变化。\n'
        'human_labels.csv：独立阅读标题和摘要后填“精读 / 浏览 / 不相关”；留空表示尚未判断。\n'
        '旧包未记录最终序号时，旧名单按初排顺序展示；不据此声称原邮件顺序。\n'
        'comparison.json 的 screening 保存公开筛选规则、模型、提示词版本与指纹，不含端点或凭据。\n'
        '缓存仅写入本输出目录的 cache/，不会读取或更新正式运行的缓存与历史。\n', encoding='utf-8')


def main(argv=None):
    args = _parser().parse_args(argv)
    # Reserve the whole destination before reading credentials or making requests.
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print('Refusing to overwrite an existing output path.')
        return 2
    except OSError as exc:
        print(f'Cannot create output directory ({type(exc).__name__}).')
        return 2
    mode = 'plan_only' if args.plan_only else 'live'
    bundle, papers, shortlist, selected = {}, [], [], []
    recall, source, summary = {}, {}, {'policy': 'reading_value'}
    screening = None
    status, failure = 'planned', None
    try:
        payload = args.bundle.read_bytes()
        bundle = json.loads(payload)
        papers = restore_candidates(bundle)
        source = {'sha256': hashlib.sha256(payload).hexdigest(), 'schema': bundle.get('schema'),
                  'candidates': len(papers), 'window': bundle.get('window')}
        base_path = args.config_dir / 'base.yaml' if args.config_dir else args.base_config
        custom_path = args.config_dir / 'custom.yaml' if args.config_dir else args.custom_config
        options, params, limit = load_screening_config(base_path, custom_path, args.limit)
        _fresh_judgements(papers)
        shortlist, recall = build_shortlist(papers, options)
        screening = _screening_snapshot(options, params, limit)
        if args.live:
            # Credentials are accepted only from these two environment variables.
            key, base_url = os.environ.get('OPENAI_API_KEY'), os.environ.get('OPENAI_API_BASE')
            if not key or not base_url:
                raise MissingCredentials('Missing required OPENAI environment variables')
            params['api'] = {'key': key, 'base_url': base_url}
            client = OpenAI(api_key=key, base_url=base_url)
            try:
                cache = PaperProgress(args.output / 'cache', 'reading-value-replay')
                selected, summary = select(papers, client, params, options, limit, cache=cache)
            finally:
                client.close()
            status = 'completed' if summary.get('complete') and not summary.get('incomplete') else 'incomplete'
        comparison = _comparison(bundle, papers, shortlist, selected, mode=mode, status=status,
                                 summary=summary, recall=recall, source=source, screening=screening)
    except Exception as exc:
        # Upstream messages can contain endpoints, tokens or echoed request data.
        failure = {'error_type': type(exc).__name__}
        summary.update(complete=False, incomplete=True)
        try:
            comparison = _comparison(bundle, papers, shortlist, selected, mode=mode, status='failed',
                                     summary=summary, recall=recall, source=source,
                                     screening=screening, failure=failure)
        except Exception:
            comparison = _comparison({}, [], [], [], mode=mode, status='failed',
                                     summary=summary, recall=recall, source=source,
                                     screening=screening, failure=failure)
    _write_reports(args.output, comparison)
    print(f'Abstract screening preview: {comparison["status"]}; reports written to the new output directory.')
    return 0 if comparison['status'] in ('planned', 'completed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
