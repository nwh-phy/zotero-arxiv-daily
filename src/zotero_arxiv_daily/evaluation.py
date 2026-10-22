"""Credential-free evaluation bundles: replay a run's ranking offline and score it against reading labels.

A bundle stores the candidates (title, abstract, stage-1 rank, relevance judgement, selection), the
reference library as ranked (title, date, theme paths), the embedding similarity matrix and the
effective settings, so ranking options can be compared on a fixed sample without the model or API.
"""
import base64
import csv
from datetime import datetime

import numpy as np
from omegaconf import OmegaConf

from .history import paper_identity
from .protocol import CorpusPaper, Paper
from .reranker.base import BaseReranker

BUNDLE_VERSION = 1
# 精读 = worth reading closely, 浏览 = worth browsing, 不相关 = not relevant.
LABELS = {'精读': 2, '浏览': 1, '不相关': 0}


def build_bundle(candidates, ordered, selected, corpus, similarity, settings, window):
    rank = {id(p): i + 1 for i, p in enumerate(ordered)}
    chosen = {id(p) for p in selected}
    return {'schema': BUNDLE_VERSION, 'window': window, 'settings': settings,
            'candidates': [{'id': paper_identity(p), 'title': p.title, 'abstract': p.abstract,
                            'stage1_rank': rank[id(p)], 'score': p.score, 'selection_score': p.selection_score,
                            'theme': p.recommendation_theme, 'relevance_score': p.relevance_score,
                            'relevance_reason': p.relevance_reason, 'relevance_status': p.relevance_status,
                            'selected': id(p) in chosen} for p in candidates],
            'corpus': [{'title': c.title, 'added_date': c.added_date.isoformat(),
                        'theme_paths': c.paths if c.theme_paths is None else c.theme_paths} for c in corpus],
            'similarity': encode_matrix(similarity)}


def encode_matrix(matrix):
    # Exact little-endian bytes: rounding flipped near-tied greedy picks when replaying a real week.
    matrix = np.ascontiguousarray(matrix)
    dtype = matrix.dtype.newbyteorder('<')
    return {'dtype': dtype.str, 'shape': list(matrix.shape),
            'base64': base64.b64encode(matrix.astype(dtype).tobytes()).decode('ascii')}


def decode_matrix(value):
    if isinstance(value, list):  # Rounded format of bundles written before exact storage.
        return np.array(value, dtype=float)
    return np.frombuffer(base64.b64decode(value['base64']), dtype=value['dtype']).reshape(value['shape'])


def write_label_template(bundle, path):
    with open(path, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'stage1_rank', 'selected', 'relevance_score', 'theme', 'title', 'label'])
        for c in sorted(bundle['candidates'], key=lambda c: c['stage1_rank']):
            writer.writerow([c['id'], c['stage1_rank'], 'yes' if c['selected'] else '',
                             c['relevance_score'] or '', c['theme'] or '', c['title'], ''])


def read_labels(path):
    with open(path, newline='', encoding='utf-8-sig') as file:
        labels = {row['id']: row['label'].strip() for row in csv.DictReader(file) if (row.get('label') or '').strip()}
    unknown = sorted({label for label in labels.values() if label not in LABELS})
    if unknown:
        raise ValueError(f'Unknown labels {unknown}; use {"/".join(LABELS)}')
    return labels


def stage1_order(bundle):
    return [c['id'] for c in sorted(bundle['candidates'], key=lambda c: c['stage1_rank'])]


def score(bundle, labels, order=None):
    """Usefulness of the recorded selection and of a stage-1 order, counted over labelled papers only."""
    order = order or stage1_order(bundle)
    value = {key: LABELS[label] for key, label in labels.items()}
    selected = [c['id'] for c in bundle['candidates'] if c['selected']]
    def useful(ids):
        judged = [value[i] for i in ids if i in value]
        return {'labelled': len(judged), 'useful': sum(v >= 1 for v in judged), 'close_reading': sum(v == 2 for v in judged)}
    return {'selected': {'count': len(selected), **useful(selected)},
            'stage1_top10': useful(order[:10]), 'stage1_top30': useful(order[:30]),
            'missed_useful': [{'id': i, 'stage1_rank': order.index(i) + 1, 'label': labels[i]}
                              for i in order if value.get(i, 0) >= 1 and i not in selected]}


class _StoredSimilarity(BaseReranker):
    def __init__(self, options, similarity):
        super().__init__(OmegaConf.create({'reranker': {'ranking': options}}))
        self.similarity = similarity

    def get_similarity_score(self, s1, s2):
        return self.similarity


def replay(bundle, **overrides):
    """Stage-1 order of candidate IDs under changed ranking options, from the stored similarities."""
    options = {**bundle['settings']['ranking'], **overrides}
    similarity = decode_matrix(bundle['similarity'])
    size = len(bundle['corpus'])
    if options.get('mode', 'legacy') != 'theme_balanced':
        similarity = similarity[:, :size]
    elif similarity.shape[1] != size + len(bundle['candidates']):
        raise ValueError('This bundle has no candidate-to-candidate similarities (recorded in legacy mode)')
    # The candidate ID travels in url; the stored corpus is already in the ranker's date order.
    candidates = [Paper(source='arxiv', title=c['title'], authors=[], abstract=c['abstract'], url=c['id'])
                  for c in bundle['candidates']]
    corpus = [CorpusPaper(c['title'], '', datetime.fromisoformat(c['added_date']), c['theme_paths'])
              for c in bundle['corpus']]
    return [p.url for p in _StoredSimilarity(options, similarity).rerank(candidates, corpus)]
