from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig, ListConfig
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper
from dataclasses import replace
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
import json
import os
import re
from time import monotonic
from .reranker import get_reranker_cls
from .construct_email import render_email, html_to_text
from .utils import send_email, EmailDeliveryError
from openai import OpenAI
from tqdm import tqdm
from .progress import PaperProgress
from .history import DeliveryHistory, paper_identity
from .relevance import select as select_relevant


def normalize_path_patterns(patterns: list[str] | ListConfig | None, config_key: str) -> list[str] | None:
    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(f"config.zotero.{config_key} must contain only glob pattern strings.")

    return list(patterns)


_MARKUP_TAG = re.compile(r'</?[A-Za-z][\w:.-]*(?:\s[^<>]*)?/?>')
_MATHML = re.compile(r'<(?:mml:)?math\b.*?</(?:mml:)?math>', re.DOTALL)


def plain_title(title: str) -> str:
    # Zotero keeps publisher markup (MathML, <sub>, <i>) in titles; emails need readable text.
    # Whitespace between MathML tags is layout, unlike the space in "<i>a</i> <i>b</i>".
    text = _MATHML.sub(lambda match: re.sub(r'>\s+<', '><', match.group(0)), title)
    return unescape(re.sub(r'\s+', ' ', _MARKUP_TAG.sub('', text))).strip()


_ARXIV_REFERENCE = re.compile(r'arxiv[^0-9]{0,20}(\d{4}\.\d{4,5})', re.IGNORECASE)


def title_key(title: str) -> str:
    return 'title:' + re.sub(r'[\W_]+', '', plain_title(title).lower())


def library_identifiers(data: dict) -> set[str]:
    # A delivered paper may be saved as the arXiv preprint or later under its journal title.
    text = ' '.join(str(data.get(field) or '') for field in ('url', 'DOI', 'extra', 'archiveID'))
    return {title_key(data.get('title') or '')} | {'arxiv:' + found for found in _ARXIV_REFERENCE.findall(text)}


def original_abstract(abstract: str) -> str:
    # A Zotero translation plugin appends "\n\n【摘要翻译】..."; embed only the original abstract.
    return abstract.split('【摘要翻译】', 1)[0].strip() or abstract


class Executor:
    def __init__(self, config:DictConfig):
        self.config = config
        self.include_path_patterns = normalize_path_patterns(config.zotero.include_path, "include_path")
        self.ignore_path_patterns = normalize_path_patterns(config.zotero.ignore_path, "ignore_path")
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        self.openai_client = OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url,
                                    timeout=90, max_retries=2)
        self.progress = PaperProgress(config.executor.get('progress_dir'), config.zotero.user_id)
        self.history_enabled = bool(config.executor.get('history_enabled', False))
        if int(config.executor.max_paper_num) <= 0:
            raise ValueError('executor.max_paper_num must be positive')
        if self.history_enabled and (self.progress.directory is None
                or list(config.executor.source) != ['arxiv']
                or not config.source.arxiv.get('lookback_days', 0)):
            raise ValueError('Delivery history requires progress_dir and weekly arXiv-only retrieval')
        # Zotero collections only change ranking, not which papers were already evaluated.
        scope = {'receiver': config.email.receiver,
                 'categories': sorted(config.source.arxiv.category or []),
                 'cross_list': config.source.arxiv.get('include_cross_list', False)}
        # Collection lists used by earlier versions, whose history file names also hashed them.
        legacy_scopes = [{**scope,
                          'include': sorted(normalize_path_patterns(entry.get('include'), 'legacy include') or []),
                          'ignore': sorted(normalize_path_patterns(entry.get('ignore'), 'legacy ignore') or [])}
                         for entry in config.executor.get('legacy_history_scopes') or []]
        self.history = DeliveryHistory(self.progress.directory if self.history_enabled else None, scope, legacy_scopes)
        self.library_ids = set()

    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")
        zot = zotero.Zotero(self.config.zotero.user_id, 'user', self.config.zotero.api_key)
        collections = zot.everything(zot.collections())
        collections = {c['key']:c for c in collections}
        corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
        self.library_ids = set().union(*(library_identifiers(c['data']) for c in corpus))
        corpus = [c for c in corpus if c['data']['abstractNote'] != '']
        def get_collection_path(col_key:str) -> str:
            if p := collections[col_key]['data']['parentCollection']:
                return get_collection_path(p) + '/' + collections[col_key]['data']['name']
            else:
                return collections[col_key]['data']['name']
        for c in corpus:
            paths = [get_collection_path(col) for col in c['data']['collections']]
            c['paths'] = paths
        logger.info(f"Fetched {len(corpus)} zotero papers")
        return [CorpusPaper(
            title=plain_title(c['data']['title']),
            abstract=original_abstract(c['data']['abstractNote']),
            added_date=datetime.strptime(c['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
            paths=c['paths']
        ) for c in corpus]
    
    def filter_corpus(self, corpus:list[CorpusPaper]) -> list[CorpusPaper]:
        if self.include_path_patterns:
            logger.info(f"Selecting zotero papers matching include_path: {self.include_path_patterns}")
            # Keep every collection for traceability, but rank themes only within included ones.
            corpus = [
                replace(c, theme_paths=matched)
                for c in corpus
                if (matched := [
                    path for path in c.paths
                    if any(glob_match(path, pattern) for pattern in self.include_path_patterns)
                ])
            ]
        if self.ignore_path_patterns:
            logger.info(f"Excluding zotero papers matching ignore_path: {self.ignore_path_patterns}")
            corpus = [
                c for c in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.ignore_path_patterns
                )
            ]
        if self.include_path_patterns or self.ignore_path_patterns:
            logger.info(f"Selected {len(corpus)} Zotero papers with abstracts")
        return corpus

    
    def run(self):
        with self.history.locked():
            self.history.assert_ready()
            self._run()

    def _run(self):
        started = monotonic()
        dry_run = bool(self.config.executor.get('dry_run', False))
        report = {'status': 'processing', 'windows': {}, 'retrieved_count': 0,
                  'candidate_count': 0, 'selected_count': 0,
                  'summaries_reused': 0, 'summaries_generated': 0,
                  'failed_count': 0, 'dry_run': dry_run}
        all_papers = []
        if self.history_enabled:
            end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            options = self.config.source.arxiv
            self.retrievers['arxiv'].window_start = self.history.window_start(
                end, int(options.lookback_days), int(options.get('overlap_days', 2)),
                int(options.get('max_backfill_days', 90)),
            )
            if not dry_run:
                self.history.remember_first_attempt(self.retrievers['arxiv'].window_start)
        for source, retriever in self.retrievers.items():
            logger.info(f'Retrieving {source} papers...')
            papers = retriever.retrieve_papers()
            all_papers.extend(papers)
            if hasattr(retriever, 'window'):
                report['windows'][source] = retriever.window
        report['retrieved_count'] = len(all_papers)
        candidates = self.history.unseen(all_papers) if self.history_enabled else all_papers
        report['candidate_count'] = len(candidates)
        logger.info(f'Retrieved {len(all_papers)} papers; {len(candidates)} new candidates')
        coverage_end = report['windows'].get('arxiv', {}).get('end_utc_exclusive')
        if self.history_enabled and not coverage_end:
            raise RuntimeError('Weekly retrieval did not establish a complete coverage window')
        ranked = []
        if candidates:
            corpus = self.filter_corpus(self.fetch_zotero_corpus())
            if not corpus:
                raise RuntimeError('No Zotero papers with abstracts match the configured collections.')
            report['previous_delivery'] = self._previous_delivery()
            ranked = self._select(self.reranker.rerank(candidates, corpus), report)
            report['selected_count'] = len(ranked)
            report['corpus_count'] = len(corpus)
            for paper in tqdm(ranked):
                try:
                    reused = self.progress.restore(paper, self.config.llm)
                    if reused:
                        report['summaries_reused'] += 1
                    else:
                        retriever = self.retrievers.get(paper.source)
                        if (not self.progress.has_extraction(paper) and retriever is not None
                                and hasattr(retriever, 'enrich_selected')):
                            retriever.enrich_selected(paper)
                        self.progress.save(paper, self.config.llm, status='extracted')
                        paper.generate_tldr(self.openai_client, self.config.llm)
                        self.progress.save(paper, self.config.llm, status='complete')
                        report['summaries_generated'] += 1
                    if self.config.llm.get('generate_affiliations', True):
                        paper.generate_affiliations(self.openai_client, self.config.llm)
                except Exception as exc:
                    self.progress.save(paper, self.config.llm, status='failed', error_type=type(exc).__name__)
                    report['failed_count'] += 1
                    logger.warning(f'Paper processing failed ({type(exc).__name__}); progress retained')
            if report['failed_count']:
                report['status'] = 'summary_failed'
                self._write_report(report, started)
                raise RuntimeError(f"{report['failed_count']} selected papers failed; email was not sent. Rerun to resume.")
            if self.config.llm.get('strict_summary', False) and any(
                    not paper.summary_is_acceptable(paper.tldr, self.config.llm) for paper in ranked):
                raise RuntimeError('Selected summaries did not pass validation; email was not sent.')
        if not ranked and not self.config.executor.send_empty:
            # Candidates judged irrelevant were still evaluated, so coverage advances.
            if self.history_enabled and not dry_run:
                self.history.complete_empty(all_papers, coverage_end)
            report['status'] = 'no_relevant_candidates' if candidates else 'no_new_candidates'
            self._write_report(report, started)
            logger.info('No paper to recommend. No email will be sent.')
            return
        html = render_email(ranked, report)
        report['papers'] = [{'title': p.title, 'url': p.url, 'score': p.score,
                             'selection_score': p.selection_score, 'theme': p.recommendation_theme,
                             'relevance_score': p.relevance_score, 'relevance_reason': p.relevance_reason,
                             'relevance_status': p.relevance_status,
                             'evidence_sections': [{'id': e['id'], 'section': e['section']} for e in p.evidence_excerpts]}
                            for p in ranked]
        if dry_run:
            report['status'] = 'preview_only'
            self._write_report(report, started, html)
            logger.info('Dry run complete: preview written; no email or coverage advancement')
            return
        batch_id = self.history.begin(ranked, all_papers, coverage_end) if self.history_enabled else None
        report['batch_id'] = batch_id
        report['status'] = 'submission_prepared'
        self._write_report(report, started, html)
        try:
            if batch_id:
                send_email(self.config, html, message_id=f'<{batch_id}@zotero-arxiv-daily.local>')
            else:
                send_email(self.config, html)
        except EmailDeliveryError as exc:
            if not exc.uncertain:
                self.history.rejected()
            report['status'] = 'smtp_uncertain' if exc.uncertain else 'smtp_rejected'
            self._write_report(report, started, html)
            raise
        except Exception:
            report['status'] = 'smtp_uncertain'
            self._write_report(report, started, html)
            raise RuntimeError('Unexpected delivery failure; acceptance must be checked before retrying') from None
        # Preserve the workflow's SMTP acceptance proof even if saving history fails.
        if marker := os.environ.get('DELIVERY_MARKER'):
            try:
                path = Path(marker)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('SMTP accepted\n')
            except OSError:
                logger.warning('SMTP was accepted but its workflow marker could not be written')
        try:
            self.history.accepted()
        except OSError:
            report['status'] = 'smtp_accepted_history_failed'
            self._write_report(report, started, html)
            raise RuntimeError('SMTP was accepted but history could not be saved; reconcile before retrying') from None
        report['status'] = 'smtp_accepted'
        self._write_report(report, started, html)
        logger.info('SMTP accepted the brief; recipient inbox arrival remains unverified')

    def _select(self, ranked, report):
        limit = int(self.config.executor.max_paper_num)
        options = self.config.reranker.get('relevance') or {}
        if options.get('enabled', False):
            selected, report['relevance'] = select_relevant(ranked, self.openai_client, self.config.llm, options, limit)
        else:
            selected = ranked[:limit]
        chosen = {id(p) for p in selected}
        # Every ranked candidate is archived so ranking changes can be compared offline.
        report['candidates'] = [{'id': paper_identity(p), 'title': p.title, 'score': p.score,
                                 'theme': p.recommendation_theme, 'relevance_score': p.relevance_score,
                                 'relevance_reason': p.relevance_reason, 'relevance_status': p.relevance_status,
                                 'selected': id(p) in chosen}
                                for p in ranked]
        return selected

    def _previous_delivery(self):
        last = self.history.data.get('last_delivery') if self.history_enabled else None
        if not last:
            return None
        saved = [p for p in last['papers'] if p['id'] in self.library_ids or title_key(p['title']) in self.library_ids]
        return {'end_utc_exclusive': last['end'], 'delivered': len(last['papers']), 'saved': len(saved)}

    def _write_report(self, report, started, html=None):
        report['duration_seconds'] = round(monotonic() - started, 2)
        output_dir = self.config.executor.get('output_dir')
        if not output_dir:
            return
        try:
            directory = Path(output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            if html is not None:
                (directory / 'brief.html').write_text(html, encoding='utf-8')
                (directory / 'brief.txt').write_text(html_to_text(html), encoding='utf-8')
        except OSError:
            logger.warning('Brief report could not be written')
