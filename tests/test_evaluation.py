"""Evaluation bundles, offline replay and scoring, stage timings and early-failure reports."""
from datetime import datetime, timedelta
import json
import subprocess
import sys

import numpy as np
from omegaconf import OmegaConf
import pytest

from tests.canned_responses import make_sample_paper
from zotero_arxiv_daily.evaluation import build_bundle, read_labels, replay, score, stage1_order, write_label_template
from zotero_arxiv_daily.executor import Executor
from zotero_arxiv_daily.protocol import CorpusPaper
from zotero_arxiv_daily.reranker.base import BaseReranker

THEME = {'mode': 'theme_balanced', 'theme_path_depth': 2, 'neighbors_per_theme': 3, 'recency_half_life_days': 180,
         'theme_penalty': 0.04, 'diversity_weight': 0.15, 'duplicate_similarity_threshold': 0.85}


class FixedReranker(BaseReranker):
    def __init__(self, similarity):
        super().__init__(OmegaConf.create({'reranker': {'ranking': THEME}}))
        self.similarity = similarity

    def get_similarity_score(self, s1, s2):
        return self.similarity


def recorded_bundle():
    date = datetime(2026, 10, 1)
    corpus = [CorpusPaper('EELS ref', 'a', date, ['20_主题/q-EELS']),
              CorpusPaper('Phonon ref', 'b', date - timedelta(days=30), ['20_主题/界面声子与振动谱学'])]
    papers = [make_sample_paper(title=f'P{i}', url=f'https://arxiv.org/abs/2610.0000{i}v1') for i in range(4)]
    corpus_sim = np.array([[.9, .1], [.85, .1], [.1, .8], [.2, .2]])
    candidate_sim = np.eye(4)
    candidate_sim[0, 1] = candidate_sim[1, 0] = .97  # P1 nearly duplicates P0.
    reranker = FixedReranker(np.concatenate([corpus_sim, candidate_sim], axis=1))
    ordered = reranker.rerank(papers, corpus)
    last = reranker.last_run
    return build_bundle(last['candidates'], ordered, ordered[:2], last['corpus'], last['similarity'],
                        {'ranking': THEME}, {})


def test_replay_reproduces_recorded_order_and_reacts_to_options():
    bundle = recorded_bundle()
    assert replay(bundle) == stage1_order(bundle) == ['arxiv:2610.00000', 'arxiv:2610.00002',
                                                      'arxiv:2610.00001', 'arxiv:2610.00003']
    # Without the diversity penalty the near-duplicate is no longer pushed below the phonon paper.
    assert replay(bundle, diversity_weight=0.0, theme_penalty=0.0)[:2] == ['arxiv:2610.00000', 'arxiv:2610.00001']
    assert replay(bundle, mode='legacy')[0] == 'arxiv:2610.00000'


def test_labels_score_selection_top_ranks_and_missed_papers(tmp_path):
    bundle = recorded_bundle()
    template = tmp_path / 'labels.csv'
    write_label_template(bundle, template)
    rows = template.read_text(encoding='utf-8-sig').splitlines()
    assert rows[0] == 'id,stage1_rank,selected,relevance_score,theme,title,label' and len(rows) == 5
    labels = {'arxiv:2610.00000': '精读', 'arxiv:2610.00002': '不相关', 'arxiv:2610.00001': '浏览'}
    template.write_text('id,label\n' + ''.join(f'{k},{v}\n' for k, v in labels.items()), encoding='utf-8')
    assert read_labels(template) == labels
    result = score(bundle, labels)
    assert result['selected'] == {'count': 2, 'labelled': 2, 'useful': 1, 'close_reading': 1}
    assert result['missed_useful'] == [{'id': 'arxiv:2610.00001', 'stage1_rank': 3, 'label': '浏览'}]
    template.write_text('id,label\narxiv:2610.00000,maybe\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Unknown labels'):
        read_labels(template)


def test_cli_scores_and_replays_a_bundle(tmp_path):
    path = tmp_path / 'evaluation.json'
    path.write_text(json.dumps(recorded_bundle(), ensure_ascii=False), encoding='utf-8')
    (tmp_path / 'labels.csv').write_text('id,label\narxiv:2610.00001,精读\n', encoding='utf-8')
    run = subprocess.run([sys.executable, 'scripts/evaluate_ranking.py', str(path), '--labels', str(tmp_path / 'labels.csv'),
                          '--set', 'diversity_weight=0.0', '--set', 'theme_penalty=0.0'],
                         capture_output=True, text=True, check=True)
    result = json.loads(run.stdout)
    assert result['recorded']['stage1_top10']['useful'] == 1
    assert result['replayed']['top10_changes'] == {'added': [], 'removed': []}  # Only 4 candidates.


def test_run_archives_bundle_timings_and_no_credentials(config, tmp_path, monkeypatch):
    from omegaconf import open_dict
    from tests.canned_responses import make_stub_openai_client, make_stub_zotero_client
    from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever
    with open_dict(config):
        config.executor.output_dir = str(tmp_path / 'report')
        config.executor.dry_run = True
        config.executor.max_paper_num = 1
        config.llm.generate_affiliations = False
        config.reranker.ranking.mode = 'theme_balanced'
    monkeypatch.setattr(ArxivRetriever, 'retrieve_papers', lambda self: [
        make_sample_paper(title=f'Paper {i}', url=f'https://arxiv.org/abs/2610.0000{i}v1') for i in range(3)])
    monkeypatch.setattr('zotero_arxiv_daily.executor.zotero.Zotero', lambda *a, **kw: make_stub_zotero_client())
    monkeypatch.setattr('zotero_arxiv_daily.executor.OpenAI', lambda **kw: make_stub_openai_client())
    monkeypatch.setattr('zotero_arxiv_daily.reranker.api.OpenAI', lambda **kw: make_stub_openai_client())
    Executor(config).run()
    report = json.loads((tmp_path / 'report/report.json').read_text())
    assert report['status'] == 'preview_only' and report['stage'] is None
    assert {'retrieval', 'zotero', 'ranking', 'relevance', 'extraction', 'summaries'} <= set(report['timings'])
    raw = (tmp_path / 'report/evaluation.json').read_text()
    bundle = json.loads(raw)
    assert len(bundle['candidates']) == 3 and bundle['similarity']['shape'] == [3, len(bundle['corpus']) + 3]
    assert replay(bundle) == stage1_order(bundle)
    assert 'fake-zotero-key' not in raw and 'sk-fake' not in raw and 'localhost:30000' not in raw


def test_failure_before_rendering_leaves_a_report_with_stage_and_error_type(config, tmp_path, monkeypatch):
    from omegaconf import open_dict
    from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever
    with open_dict(config):
        config.executor.output_dir = str(tmp_path / 'report')
    def throttled(self):
        raise ConnectionError('429 Too Many Requests for https://export.arxiv.org/api/query?secret=1')
    monkeypatch.setattr(ArxivRetriever, 'retrieve_papers', throttled)
    with pytest.raises(ConnectionError):
        Executor(config).run()
    raw = (tmp_path / 'report/report.json').read_text()
    report = json.loads(raw)
    assert report['status'] == 'failed' and report['failed_stage'] == 'retrieval'
    assert report['error_type'] == 'ConnectionError' and 'retrieval' in report['timings']
    assert 'secret=1' not in raw  # Only the error type is reported, never the upstream message.


def test_replay_is_exact_where_rounding_would_flip_a_near_tie():
    date = datetime(2026, 10, 1)
    corpus = [CorpusPaper('Ref', 'a', date, ['20_主题/q-EELS'])]
    papers = [make_sample_paper(title=f'P{i}', url=f'https://arxiv.org/abs/2610.0000{i}v1') for i in range(2)]
    # Equal after rounding to 4 decimals (the index would then break the tie); exact values favour P1.
    similarity = np.array([[0.70001, 1, 0], [0.70004, 0, 1]], dtype=np.float32)
    reranker = FixedReranker(similarity)
    ordered = reranker.rerank(papers, corpus)
    last = reranker.last_run
    bundle = json.loads(json.dumps(build_bundle(last['candidates'], ordered, ordered[:1], last['corpus'],
                                                last['similarity'], {'ranking': THEME}, {})))
    assert stage1_order(bundle)[0] == 'arxiv:2610.00001' and replay(bundle) == stage1_order(bundle)
    rounded = {**bundle, 'similarity': np.round(similarity, 4).tolist()}  # Older bundle format.
    assert replay(rounded)[0] == 'arxiv:2610.00000'
