"""LLM relevance stage, candidate archive, and saved-paper feedback."""
import json
from types import SimpleNamespace

from omegaconf import OmegaConf, open_dict
import pytest

from tests.canned_responses import make_sample_paper, make_stub_zotero_client
from zotero_arxiv_daily.construct_email import render_email
from zotero_arxiv_daily.executor import Executor, library_identifiers
from zotero_arxiv_daily.relevance import judge, select

PARAMS = {'api_mode': 'chat_completion', 'generation_kwargs': {'model': 'stub'}}
OPTIONS = {'profile': 'q-EELS 与低能集体激发', 'top_k': 4, 'min_score': 3}


def llm_client(replies):
    """Reply per paper title; a missing title raises like a provider error."""
    def create(**kwargs):
        text = str(kwargs.get('messages') or kwargs.get('input'))
        reply = next((value for title, value in replies.items() if f'Title: {title}' in text), None)
        if reply is None:
            raise RuntimeError('provider error')
        message = SimpleNamespace(content=reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], output_text=reply)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                           responses=SimpleNamespace(create=create))


def scored(score, reason='理由'):
    return json.dumps({'score': score, 'reason': reason}, ensure_ascii=False)


@pytest.mark.parametrize('reply, expected', [
    ('```json\n{"score": 4, "reason": "测量石墨声子色散"}\n```', (4, '测量石墨声子色散')),
    ('{"score": "5", "reason": " 同一技术 "}', (5, '同一技术')),
])
def test_judge_accepts_fenced_or_string_scores(reply, expected):
    paper = make_sample_paper(title='P')
    assert judge(llm_client({'P': reply}), PARAMS, 'profile', paper) == expected


@pytest.mark.parametrize('reply', ['not json', '{"score": 7, "reason": "x"}', '{"score": true, "reason": "x"}',
                                   '{"score": 3, "reason": ""}'])
def test_judge_rejects_malformed_judgements(reply):
    with pytest.raises(ValueError):
        judge(llm_client({'P': reply}), PARAMS, 'profile', make_sample_paper(title='P'))


def test_select_orders_by_judgement_and_drops_low_scores():
    papers = [make_sample_paper(title=t) for t in ['Cold atoms', 'Phonon EELS', 'Plasmon', 'Unjudged', 'Beyond shortlist']]
    client = llm_client({'Cold atoms': scored(1, '冷原子'), 'Phonon EELS': scored(5), 'Plasmon': scored(3)})
    selected, summary = select(papers, client, PARAMS, OPTIONS, limit=3)
    # Judged papers first by score; an unjudged paper stays eligible after them; below 3 is dropped.
    assert [p.title for p in selected] == ['Phonon EELS', 'Plasmon', 'Unjudged']
    assert summary == {'mode': 'applied', 'judged': 4, 'failed': 1, 'min_score': 3, 'passed': 3}
    assert papers[0].relevance_reason == '冷原子'


def test_select_keeps_embedding_order_when_most_judgements_fail():
    papers = [make_sample_paper(title=t) for t in 'ABCD']
    selected, summary = select(papers, llm_client({'A': scored(1)}), PARAMS, OPTIONS, limit=2)
    assert [p.title for p in selected] == ['A', 'B']
    assert summary['mode'] == 'fallback' and all(p.relevance_score is None for p in papers)


def test_select_requires_profile_and_enough_shortlist():
    with pytest.raises(ValueError, match='profile'):
        select([], llm_client({}), PARAMS, {**OPTIONS, 'profile': ' '}, limit=1)
    with pytest.raises(ValueError, match='parameters'):
        select([], llm_client({}), PARAMS, {**OPTIONS, 'top_k': 2}, limit=3)


def test_executor_archives_every_ranked_candidate():
    executor = Executor.__new__(Executor)
    executor.config = OmegaConf.create({'executor': {'max_paper_num': 1}, 'llm': PARAMS,
                                        'reranker': {'relevance': {**OPTIONS, 'enabled': True}}})
    executor.openai_client = llm_client({'A': scored(2), 'B': scored(4, '振动谱学')})
    papers = [make_sample_paper(title=t, url=f'https://arxiv.org/abs/2610.0000{i}v1', score=9 - i)
              for i, t in enumerate('AB')]
    report = {}
    assert [p.title for p in executor._select(papers, report)] == ['B']
    assert [(c['id'], c['relevance_score'], c['selected']) for c in report['candidates']] == [
        ('arxiv:2610.00000', 2, False), ('arxiv:2610.00001', 4, True)]
    html = render_email([papers[1]], report)
    assert 'LLM 判断 4/5：振动谱学' in html and '再由 LLM 按研究画像判断相关性' in html


def test_library_identifiers_match_preprints_and_titles_without_false_dois():
    ids = library_identifiers({'title': 'Plasmon <i>collapse</i>', 'url': 'http://arxiv.org/abs/2610.21209v1',
                               'DOI': '10.1016/j.ultramic.2026.114402', 'extra': 'arXiv: 2511.08819'})
    assert ids == {'title:plasmoncollapse', 'arxiv:2610.21209', 'arxiv:2511.08819'}


@pytest.fixture()
def relevance_executor(config, tmp_path, monkeypatch):
    from datetime import datetime, timezone, timedelta
    from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever
    with open_dict(config):
        config.executor.progress_dir = str(tmp_path / 'progress')
        config.executor.history_enabled = True
        config.executor.output_dir = str(tmp_path / 'report')
        config.source.arxiv.lookback_days = 7
        config.executor.max_paper_num = 1
        config.llm.generate_affiliations = False
        config.reranker.relevance.update({'enabled': True, 'top_k': 2, 'min_score': 3, 'profile': 'q-EELS'})
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    batches = [['2610.00001', '2610.00002'], ['2610.00003', '2610.00004']]
    def retrieve(self):
        self.window = {'start_utc': self.window_start.isoformat(), 'end_utc_exclusive': end.isoformat(), 'count': 2}
        return [make_sample_paper(title=f'Paper {pid}', url=f'https://arxiv.org/abs/{pid}v1',
                                  published=(end - timedelta(days=1)).isoformat()) for pid in batches.pop(0)]
    monkeypatch.setattr(ArxivRetriever, 'retrieve_papers', retrieve)
    monkeypatch.setattr('zotero_arxiv_daily.reranker.api.OpenAI', lambda **kw: SimpleNamespace(
        embeddings=SimpleNamespace(create=lambda **kw: SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3], index=i) for i in range(len(kw['input']))]))))
    monkeypatch.delenv('DELIVERY_MARKER', raising=False)
    return config, tmp_path


def test_irrelevant_week_sends_nothing_but_advances_coverage(relevance_executor, monkeypatch):
    config, tmp_path = relevance_executor
    monkeypatch.setattr('zotero_arxiv_daily.executor.zotero.Zotero', lambda *a, **kw: make_stub_zotero_client())
    monkeypatch.setattr('zotero_arxiv_daily.executor.OpenAI', lambda **kw: llm_client(
        {'Paper 2610.00001': scored(1), 'Paper 2610.00002': scored(2)}))
    monkeypatch.setattr('zotero_arxiv_daily.executor.send_email', lambda *a, **kw: pytest.fail('No email expected'))
    executor = Executor(config)
    executor.run()
    report = json.loads((tmp_path / 'report/report.json').read_text())
    assert report['status'] == 'no_relevant_candidates'
    assert len(report['candidates']) == 2 and not any(c['selected'] for c in report['candidates'])
    assert executor.history.data['covered_until'] and len(executor.history.data['seen']) == 2


def test_next_brief_reports_how_many_delivered_papers_were_saved(relevance_executor, monkeypatch):
    config, tmp_path = relevance_executor
    replies = {f'Paper 2610.0000{i}': scored(4) for i in range(1, 5)}
    monkeypatch.setattr('zotero_arxiv_daily.executor.OpenAI', lambda **kw: llm_client(replies))
    monkeypatch.setattr('zotero_arxiv_daily.executor.zotero.Zotero', lambda *a, **kw: make_stub_zotero_client())
    sent = []
    monkeypatch.setattr('zotero_arxiv_daily.executor.send_email', lambda config, html, **kw: sent.append(html))
    first = Executor(config)
    first.run()
    delivered = first.history.data['last_delivery']['papers']
    assert len(delivered) == 1
    saved = {'data': {'title': 'Saved later', 'abstractNote': '', 'dateAdded': '2026-10-21T00:00:00Z',
                      'collections': [], 'url': 'https://arxiv.org/abs/' + delivered[0]['id'].split(':')[1]}}
    from tests.canned_responses import _DEFAULT_ITEMS
    monkeypatch.setattr('zotero_arxiv_daily.executor.zotero.Zotero',
                        lambda *a, **kw: make_stub_zotero_client(items=[*_DEFAULT_ITEMS, saved]))
    Executor(config).run()
    report = json.loads((tmp_path / 'report/report.json').read_text())
    assert report['previous_delivery']['delivered'] == 1 and report['previous_delivery']['saved'] == 1
    assert '上期推荐 1 篇，目前已有 1 篇在你的 Zotero 中' in sent[-1]
