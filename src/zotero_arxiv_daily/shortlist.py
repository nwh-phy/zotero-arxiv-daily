"""Bounded, deterministic recall before reading-value judgement.

Keyword matches add candidates for the LLM to inspect; they never establish
relevance or consume a recommendation slot by themselves.
"""
import re
import unicodedata

from omegaconf import DictConfig, ListConfig

from .history import paper_identity


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a non-negative integer')
    return value


def build_shortlist(papers, options):
    """Return a unique shortlist in original ranking order and recall counts.

    Take ``embedding_k`` unique ranked papers, add up to ``core_k`` matching
    papers from the remaining ranking, then fill unused slots in ranking order.
    The total never exceeds ``top_k``. Matching uses Unicode normalization so
    chemical subscripts and ordinary digits behave consistently.
    """
    if not isinstance(options, (dict, DictConfig)):
        raise ValueError('shortlist options must be a mapping')
    papers = list(papers)
    top_k = _count(options.get('top_k', 60), 'top_k')
    if top_k == 0:
        raise ValueError('top_k must be positive')
    settings = options.get('shortlist')
    if settings is None:
        settings = {}
    if not isinstance(settings, (dict, DictConfig)):
        raise ValueError('shortlist must be a mapping')
    embedding_k = _count(settings.get('embedding_k', min(40, top_k)), 'embedding_k')
    core_k = _count(settings.get('core_k', min(20, max(0, top_k - embedding_k))), 'core_k')
    if embedding_k + core_k > top_k:
        raise ValueError('embedding_k + core_k must not exceed top_k')
    patterns = settings.get('core_patterns', [])
    if patterns is None:
        patterns = []
    if not isinstance(patterns, (list, tuple, ListConfig)) or any(
            not isinstance(pattern, str) or not pattern.strip() for pattern in patterns):
        raise ValueError('core_patterns must contain nonempty regular expressions')
    try:
        patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    except re.error:
        raise ValueError('Invalid core_patterns regular expression') from None

    unique, seen = [], set()
    for paper in papers:
        paper.shortlist_sources = []
        identity = paper_identity(paper)
        if identity not in seen:
            seen.add(identity)
            unique.append(paper)

    matching = set()
    for i, paper in enumerate(unique):
        text = unicodedata.normalize('NFKC', f'{paper.title}\n{paper.abstract}')
        text = re.sub('[‐‑‒–—−]', '-', text)
        if any(pattern.search(text) for pattern in patterns):
            matching.add(i)
    embedding = set(range(min(embedding_k, len(unique))))
    core = set(i for i in sorted(matching) if i not in embedding)
    core = set(sorted(core)[:core_k])
    chosen = embedding | core
    fill = set()
    for i in range(len(unique)):
        if len(chosen) >= top_k:
            break
        if i not in chosen:
            chosen.add(i)
            fill.add(i)

    result = []
    for i in sorted(chosen):
        paper = unique[i]
        if i in embedding:
            paper.shortlist_sources.append('embedding')
        if i in core or (i in embedding and i in matching):
            paper.shortlist_sources.append('core')
        if i in fill:
            paper.shortlist_sources.append('fill')
        result.append(paper)
    return result, {'total_candidates': len(papers), 'unique_candidates': len(unique),
                    'duplicates_removed': len(papers) - len(unique), 'selected': len(result),
                    'embedding': len(embedding), 'core_added': len(core), 'fill': len(fill),
                    'core_matches': len(matching)}
