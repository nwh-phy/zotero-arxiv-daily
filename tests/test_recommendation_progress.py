"""Regression cases for balanced recommendations, bounded evidence, and resumable work."""
from datetime import datetime, timedelta
import json

import numpy as np
from omegaconf import OmegaConf, open_dict
import pytest

from tests.canned_responses import make_sample_paper, make_sample_corpus, make_stub_openai_client, make_stub_zotero_client
from zotero_arxiv_daily.reranker.base import BaseReranker
from zotero_arxiv_daily.evidence import build_summary_prompt, select_excerpts
from zotero_arxiv_daily.progress import PaperProgress
from zotero_arxiv_daily.protocol import CorpusPaper, Paper
from zotero_arxiv_daily.executor import Executor
from zotero_arxiv_daily.construct_email import render_email


class MatrixReranker(BaseReranker):
    def __init__(self, corpus_sim, candidate_sim=None, **options):
        config = OmegaConf.create({'reranker': {'ranking': {'mode': 'theme_balanced', **options}}})
        super().__init__(config)
        self.sim = np.concatenate([corpus_sim, candidate_sim if candidate_sim is not None else np.eye(len(corpus_sim))], axis=1)

    def get_similarity_score(self, s1, s2):
        return self.sim


def test_minority_theme_not_diluted_by_large_collection():
    date = datetime(2026, 1, 1)
    corpus = [CorpusPaper(str(i), str(i), date, ['20_主题/A']) for i in range(20)]
    corpus.append(CorpusPaper('Minority reference', 'B', date, ['20_主题/B']))
    sim = np.array([[.7] * 20 + [.1], [.1] * 20 + [.9]])
    papers = [make_sample_paper(title='Majority'), make_sample_paper(title='Minority')]
    ranked = MatrixReranker(sim).rerank(papers, corpus)
    assert ranked[0].title == 'Minority'
    assert ranked[0].score == pytest.approx(9)
    assert ranked[0].recommendation_theme == '20_主题/B'
    assert ranked[0].related_papers == ['Minority reference']


def test_decay_depends_on_elapsed_days_not_library_position():
    date = datetime(2026, 1, 1)
    def score(gap):
        corpus = [CorpusPaper('New', 'New', date, ['T']),
                  CorpusPaper('Old', 'Old', date - timedelta(days=gap), ['T'])]
        return MatrixReranker(np.array([[.1, .9]]), recency_half_life_days=30).rerank([make_sample_paper()], corpus)[0].score
    assert score(1) > score(365)


def test_diversity_demotes_near_duplicate_without_forcing_weak_theme():
    date = datetime(2026, 1, 1)
    corpus = [CorpusPaper('A', 'A', date, ['A']), CorpusPaper('B', 'B', date, ['B'])]
    sim = np.array([[.95, .1], [.94, .1], [.1, .9], [.1, .1]])
    similar = np.eye(4)
    similar[0, 1] = similar[1, 0] = .99
    papers = [make_sample_paper(title=name) for name in ['Best A', 'Duplicate A', 'Good B', 'Weak']]
    ranked = MatrixReranker(sim, similar).rerank(papers, corpus)
    assert [p.title for p in ranked][:2] == ['Best A', 'Good B']
    assert ranked[-1].title == 'Weak'
    assert ranked[1].score < ranked[2].score  # Diversity is not misrepresented as raw similarity.
    assert all(a.selection_score >= b.selection_score for a, b in zip(ranked, ranked[1:]))


def test_ranking_rejects_nonfinite_embeddings():
    with pytest.raises(ValueError, match='similarity matrix'):
        MatrixReranker(np.array([[np.nan]])).rerank([make_sample_paper()], make_sample_corpus(1))


def test_key_sections_survive_long_tex_preamble_and_introduction():
    text = (r'\newcommand{\unused}{noise}' * 5000 + '\n' + r'\begin{document}'
            + '\n' + r'\section{Introduction}' + ' background' * 10000
            + '\n' + r'\section{Methods}' + ' MEASURE_EELS'
            + '\n' + r'\section{Results}' + ' OBSERVED_PLASMON'
            + '\n' + r'\section{Conclusion}' + ' QUANTUM_MATERIALS'
            + '\n' + r'\section{References}' + ' UNRELATED_CITATION')
    paper = make_sample_paper(full_text=text)
    prompt = build_summary_prompt(paper, '简体中文')
    assert all(marker in prompt for marker in ['MEASURE_EELS', 'OBSERVED_PLASMON', 'QUANTUM_MATERIALS'])
    assert 'newcommand' not in prompt and 'UNRELATED_CITATION' not in prompt
    import tiktoken
    assert len(tiktoken.get_encoding('o200k_base').encode(prompt)) <= 4000
    assert paper.evidence_excerpts[0]['section'] == 'Conclusion'


@pytest.mark.parametrize('text', ['# Methods\nM\n# Results\nR\n# Conclusion\nC',
                                   '1 Methods\nM\n2 Results\nR\n3 Conclusion\nC'])
def test_markdown_and_plain_text_sections(text):
    excerpts = select_excerpts(text)
    assert [e['section'] for e in excerpts] == ['Conclusion', 'Results', 'Methods']


def test_unstructured_text_includes_end_and_reference_only_has_no_evidence():
    prompt = build_summary_prompt(make_sample_paper(full_text='START ' + 'word ' * 10000 + ' END_RESULT'), 'English')
    assert 'END_RESULT' in prompt
    paper = make_sample_paper(full_text='# References\nOnly bibliography')
    prompt = build_summary_prompt(paper, 'English')
    assert not paper.evidence_excerpts
    assert '只能使用[摘要确认]' in prompt


def test_summary_cannot_cite_nonexistent_excerpt():
    paper = make_sample_paper(full_text='# Results\nA measured result')
    build_summary_prompt(paper, 'Chinese')
    assert paper.summary_is_acceptable('结果[全文片段确认:E1]', {})
    assert not paper.summary_is_acceptable('结果[全文片段确认:E99]', {})
    assert not paper.summary_is_acceptable('结果[全文片段确认]', {})
    paper.full_text = None
    build_summary_prompt(paper, 'Chinese')
    assert not paper.summary_is_acceptable('结果[全文片段确认:E1]', {})


def test_cache_reuse_invalidation_and_credential_exclusion(tmp_path):
    params = {'language': 'Chinese', 'generation_kwargs': {'model': 'm1'},
              'api': {'key': 'SECRET_SHOULD_NOT_BE_SAVED', 'base_url': 'https://example.com/v1'}}
    progress = PaperProgress(tmp_path, 'library1')
    paper = make_sample_paper(tldr='validated summary')
    progress.save(paper, params, status='complete')
    restored = make_sample_paper(full_text=None)
    assert progress.restore(restored, params)
    assert restored.full_text == paper.full_text and restored.tldr == paper.tldr
    raw = next(tmp_path.rglob('*.json')).read_text()
    assert 'SECRET_SHOULD_NOT_BE_SAVED' not in raw and 'https://example.com' not in raw
    assert not progress.restore(make_sample_paper(full_text=None), {**params, 'summary_instructions': 'changed prompt'})
    assert not progress.restore(make_sample_paper(full_text=None), {**params, 'generation_kwargs': {'model': 'm2'}})
    assert not progress.restore(make_sample_paper(full_text=None, url=paper.url + 'v2'), params)
    assert not progress.restore(make_sample_paper(full_text=None, abstract='changed abstract'), params)
    assert not progress.restore(make_sample_paper(full_text='changed body'), params)
    assert not PaperProgress(tmp_path, 'library2').restore(make_sample_paper(full_text=None), params)
    # Rotating credentials does not invalidate a summary made from identical evidence.
    rotated = {**params, 'api': {**params['api'], 'key': 'NEW_KEY'}}
    assert progress.restore(make_sample_paper(full_text=None), rotated)


def test_corrupt_cache_is_ignored_and_failed_record_retains_extraction(tmp_path):
    progress = PaperProgress(tmp_path, 'library')
    paper = make_sample_paper(tldr='old summary')
    progress.save(paper, {}, status='failed', error_type='RuntimeError')
    restored = make_sample_paper(full_text=None)
    assert not progress.restore(restored, {})
    assert restored.full_text == paper.full_text
    assert progress.has_extraction(restored)
    path = next(tmp_path.rglob('*.json'))
    record = json.loads(path.read_text())
    record['payload']['summary'] = 'tampered'
    path.write_text(json.dumps(record))
    assert not progress.restore(make_sample_paper(full_text=None), {})
    path.write_text('not json')
    assert progress.load(paper) == {}


def test_checkpoint_revalidated_against_current_quality_rules(tmp_path):
    progress = PaperProgress(tmp_path, 'library')
    paper = make_sample_paper(tldr='cached')
    progress.save(paper, {}, status='complete')
    assert not progress.restore(make_sample_paper(full_text=None), {'summary_min_chars': 150, 'summary_max_chars': 250})


def test_email_explains_selection_and_escapes_reference_titles():
    paper = make_sample_paper(score=9, tldr='Summary[全文片段确认:E1]', recommendation_theme='20_主题/EELS',
                              related_papers=['<script>bad</script>'])
    paper.evidence_excerpts = [{'id': 'E1', 'section': '<Results>', 'text': 'measured result'}]
    html = render_email([paper])
    assert '推荐依据' in html and '20_主题/EELS' in html and '内容来源' in html
    assert '<script>' not in html and '&lt;Results&gt;' in html
    assert '综合相关度' in html


@pytest.mark.parametrize("failure_stage", ["summary", "smtp"])
def test_failure_saves_other_papers_and_next_run_only_retries_failure(config, tmp_path, monkeypatch, failure_stage):
    with open_dict(config):
        config.executor.progress_dir = str(tmp_path)
        config.executor.max_paper_num = 3
        config.llm.strict_summary = True
        config.llm.generate_affiliations = False
        config.executor.reranker = 'api'
    monkeypatch.setattr('zotero_arxiv_daily.executor.zotero.Zotero', lambda *a, **kw: make_stub_zotero_client())
    monkeypatch.setattr('zotero_arxiv_daily.executor.OpenAI', lambda **kw: make_stub_openai_client())
    monkeypatch.setattr('zotero_arxiv_daily.reranker.api.OpenAI', lambda **kw: make_stub_openai_client())
    from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever
    def retrieve(self):
        return [make_sample_paper(title=f'Paper {i}', url=f'https://arxiv.org/abs/2601.0000{i}v1', full_text=None) for i in range(3)]
    monkeypatch.setattr(ArxivRetriever, 'retrieve_papers', retrieve)
    extracted = []
    def enrich(self, paper):
        extracted.append(paper.title)
        paper.full_text = '# Results\nA measured finding'
    monkeypatch.setattr(ArxivRetriever, 'enrich_selected', enrich)
    generated = []
    def summarize(self, client, params):
        generated.append(self.title)
        if failure_stage == 'summary' and self.title == 'Paper 1' and generated.count(self.title) == 1:
            raise RuntimeError('upstream failure with sensitive detail')
        return 'Validated summary'
    monkeypatch.setattr(Paper, '_generate_tldr_with_llm', summarize)
    sent = []
    send_attempts = []
    def send(*args):
        send_attempts.append(args)
        if failure_stage == 'smtp' and len(send_attempts) == 1:
            raise RuntimeError('SMTP unavailable before acceptance')
        sent.append(args)
    monkeypatch.setattr('zotero_arxiv_daily.executor.send_email', send)
    expected_error = '1 selected papers failed' if failure_stage == 'summary' else 'Unexpected delivery failure'
    with pytest.raises(RuntimeError, match=expected_error):
        Executor(config).run()
    assert not sent
    assert generated == ['Paper 0', 'Paper 1', 'Paper 2']
    assert len(list(tmp_path.rglob('*.json'))) == 3
    assert all('sensitive detail' not in p.read_text() for p in tmp_path.rglob('*.json'))
    Executor(config).run()
    expected_calls = ['Paper 0', 'Paper 1', 'Paper 2'] + (['Paper 1'] if failure_stage == 'summary' else [])
    assert generated == expected_calls
    assert extracted == ['Paper 0', 'Paper 1', 'Paper 2']
    assert len(sent) == 1


def test_api_embeds_shared_candidate_text_only_once(config, monkeypatch):
    from types import SimpleNamespace
    from zotero_arxiv_daily.reranker.api import ApiReranker
    batches = []
    def embed(**kwargs):
        batches.append(kwargs['input'])
        vectors = {'A': [1, 0], 'B': [0, 1], 'C': [1, 1]}
        return SimpleNamespace(data=[SimpleNamespace(embedding=vectors[t]) for t in kwargs['input']])
    monkeypatch.setattr('zotero_arxiv_daily.reranker.api.OpenAI', lambda **kw: SimpleNamespace(embeddings=SimpleNamespace(create=embed)))
    score = ApiReranker(config).get_similarity_score(['A', 'B'], ['C', 'A', 'B'])
    assert batches == [['A', 'B', 'C']]
    assert score.shape == (2, 3)
    np.testing.assert_allclose(score[:, 1:], np.eye(2))


def test_workflow_progress_cache_is_best_effort_and_saved_on_failure():
    from pathlib import Path
    import yaml
    workflow = yaml.safe_load(Path('.github/workflows/main.yml').read_text())
    steps = workflow['jobs']['calculate-and-send']['steps']
    restore = next(step for step in steps if step.get('name') == 'Restore per-paper processing progress')
    save = next(step for step in steps if step.get('name') == 'Save per-paper progress including failed runs')
    marker = next(step for step in steps if step.get('name') == 'Remember SMTP acceptance')
    assert restore['continue-on-error'] and save['continue-on-error']
    assert 'always()' in save['if'] and "steps.process.outcome == 'failure'" in save['if']
    assert '${{ github.run_attempt }}' in save['with']['key']
    assert save['with']['path'] != marker['with']['path']


def test_methods_are_retained_among_many_results_subsections():
    text = '# Methods\nCRITICAL_METHOD\n'
    text += '\n'.join(f'## Results {i}\nResult {i}' for i in range(8))
    text += '\n# Conclusion\nConcluding finding'
    assert any('CRITICAL_METHOD' in e['text'] for e in select_excerpts(text))


def test_plain_text_percentage_does_not_discard_results():
    excerpts = select_excerpts('# Results\nSignal increased by 50% with CONFIRMED_RESULT.')
    assert '50% with CONFIRMED_RESULT' in excerpts[0]['text']


def test_local_embedding_deduplicates_and_preserves_matrix_axes(config, monkeypatch):
    import sys
    from types import SimpleNamespace
    from zotero_arxiv_daily.reranker.local import LocalReranker
    with open_dict(config):
        config.executor.debug = True
    batches = []
    class Encoder:
        def __init__(self, *args, **kwargs):
            pass
        def encode(self, texts, **kwargs):
            batches.append(texts)
            vectors = {'A': [1, 0], 'B': [0, 1], 'C': [1, 1]}
            return np.array([vectors[text] for text in texts])
        def similarity(self, left, right):
            return SimpleNamespace(numpy=lambda: left @ right.T)
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=Encoder))
    score = LocalReranker(config).get_similarity_score(['A', 'B'], ['C', 'A', 'B'])
    assert batches == [['A', 'B', 'C']]
    np.testing.assert_array_equal(score, [[1, 1, 0], [1, 0, 1]])


def test_unwritable_progress_directory_does_not_block_summary(tmp_path):
    blocked = tmp_path / 'file'
    blocked.write_text('not a directory')
    progress = PaperProgress(blocked, 'library')
    paper = make_sample_paper(tldr='Valid summary')
    progress.save(paper, {}, status='complete')
    assert not progress.restore(make_sample_paper(full_text=None), {})


def test_chinese_summary_requires_evidence_for_each_field():
    params = {'summary_min_chars': 150, 'summary_max_chars': 250}
    paper = make_sample_paper(full_text=None)
    build_summary_prompt(paper, 'Chinese')
    summary = '\n'.join(f'{label}：' + '中' * 30 + '[摘要确认]' for label in ['问题', '方法', '结论', '创新', '参考价值'])
    assert paper.summary_is_acceptable(summary, params)
    missing_tag = summary.replace('方法：' + '中' * 30 + '[摘要确认]', '方法：' + '中' * 30)
    assert not paper.summary_is_acceptable(missing_tag, params)
    unsupported_innovation = summary.replace('创新：' + '中' * 30 + '[摘要确认]', '创新：尚无法确认。')
    # Unknown novelty is allowed, provided the whole summary still meets its length rule.
    assert paper.summary_is_acceptable(unsupported_innovation + '中' * 15, params)


def test_api_out_of_order_embeddings_use_authoritative_indices(config, monkeypatch):
    from types import SimpleNamespace
    from zotero_arxiv_daily.reranker.api import ApiReranker
    def embed(**kwargs):
        return SimpleNamespace(data=[SimpleNamespace(index=1, embedding=[0, 1]),
                                     SimpleNamespace(index=0, embedding=[1, 0])])
    monkeypatch.setattr('zotero_arxiv_daily.reranker.api.OpenAI', lambda **kw: SimpleNamespace(embeddings=SimpleNamespace(create=embed)))
    score = ApiReranker(config).get_similarity_score(['A'], ['A', 'B'])
    np.testing.assert_allclose(score, [[1, 0]])


@pytest.mark.parametrize('vectors', [[[0, 0], [0, 1]], [[float('nan'), 1], [0, 1]]])
def test_invalid_embedding_vectors_fail_before_ranking(config, monkeypatch, vectors):
    from types import SimpleNamespace
    from zotero_arxiv_daily.reranker.api import ApiReranker
    client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kw: SimpleNamespace(
        data=[SimpleNamespace(index=i, embedding=vector) for i, vector in enumerate(vectors)])))
    monkeypatch.setattr('zotero_arxiv_daily.reranker.api.OpenAI', lambda **kw: client)
    with pytest.raises(ValueError, match='nonfinite or zero'):
        ApiReranker(config).get_similarity_score(['A'], ['B'])
