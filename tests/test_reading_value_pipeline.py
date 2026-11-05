from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import json

import pytest
from omegaconf import open_dict

from tests.canned_responses import make_sample_paper, make_stub_openai_client, make_stub_zotero_client
from zotero_arxiv_daily.executor import Executor
from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever

BODY = 'We measure momentum dependent plasmon dispersion in a confined metal.'


def setup(config, tmp_path, monkeypatch, *, body=BODY, review='confirmed', fail_screen=False):
    with open_dict(config):
        config.executor.output_dir = str(tmp_path / 'output')
        config.executor.progress_dir = str(tmp_path / 'cache')
        config.executor.history_enabled = True
        config.executor.legacy_history_scopes = []
        config.executor.max_paper_num = 2
        config.source.arxiv.lookback_days = 7
        config.llm.generate_affiliations = False
        config.reranker.relevance.update({'enabled': True, 'policy': 'reading_value', 'top_k': 2,
                                         'focus_limit': 2, 'explore_limit': 1,
                                         'shortlist': {'embedding_k': 2, 'core_k': 0, 'core_patterns': []}})
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    def retrieve(self):
        self.window = {'start_utc': self.window_start.isoformat(), 'end_utc_exclusive': end.isoformat()}
        return [make_sample_paper(title=f'Paper {i}', abstract=BODY, url=f'https://arxiv.org/abs/2610.0010{i}',
                                  full_text=('# Results\n' + body) if body else None,
                                  published=(end-timedelta(days=1)).isoformat()) for i in range(2)]
    monkeypatch.setattr(ArxivRetriever, 'retrieve_papers', retrieve)
    monkeypatch.setattr('zotero_arxiv_daily.executor.zotero.Zotero', lambda *a, **kw: make_stub_zotero_client())
    monkeypatch.setattr('zotero_arxiv_daily.reranker.api.OpenAI', lambda **kw: make_stub_openai_client())
    calls = {'screen': 0, 'review': 0, 'summary': 0, 'sent': []}
    def create(**kwargs):
        messages = kwargs.get('input', kwargs.get('messages'))
        system = messages[0]['content']
        if '证据复核员' in system:
            calls['review'] += 1
            if review == 'failed':
                raise RuntimeError('upstream failure')
            result = {'status': review, 'reason': '片段核验结果', 'evidence_id': 'E1', 'evidence': BODY}
        elif '文献筛选助手' in system:
            calls['screen'] += 1
            if fail_screen and 'Paper 1' in messages[-1]['content']:
                raise RuntimeError('upstream failure')
            result = {'score': 4, 'reason': '可对照色散', 'route': 'current', 'contribution': '测量动量色散',
                      'target': '二维金属激发', 'use': '比较损失谱的动量依赖', 'evidence': BODY}
        else:
            calls['summary'] += 1
            return SimpleNamespace(output_text='问题：测量集体激发的色散。[全文片段确认:E1]')
        return SimpleNamespace(output_text=json.dumps(result, ensure_ascii=False))
    monkeypatch.setattr('zotero_arxiv_daily.executor.OpenAI', lambda **kw: SimpleNamespace(responses=SimpleNamespace(create=create)))
    monkeypatch.setattr('zotero_arxiv_daily.executor.send_email', lambda *a, **kw: calls['sent'].append(a[1]))
    monkeypatch.delenv('DELIVERY_MARKER', raising=False)
    return Executor(config), calls


@pytest.mark.parametrize('failure, expected', [('screen', 'relevance_incomplete'), ('review', 'review_incomplete')])
def test_incomplete_judgement_or_review_never_sends_or_consumes_coverage(config, tmp_path, monkeypatch, failure, expected):
    executor, calls = setup(config, tmp_path, monkeypatch, fail_screen=failure == 'screen',
                            review='failed' if failure == 'review' else 'confirmed')
    with pytest.raises(RuntimeError, match='incomplete'):
        executor.run()
    report = json.loads((tmp_path / 'output/report.json').read_text())
    assert report['status'] == expected and not calls['sent']
    assert executor.history.data['covered_until'] is None and not executor.history.data['seen']
    assert executor.history.data['pending'] is None
    assert (tmp_path / 'output/evaluation.json').exists()


def test_unavailable_body_can_only_send_short_exploration_without_summary(config, tmp_path, monkeypatch):
    executor, calls = setup(config, tmp_path, monkeypatch, body=None)
    executor.run()
    report = json.loads((tmp_path / 'output/report.json').read_text())
    assert report['status'] == 'smtp_accepted' and report['selected_count'] == 1
    assert report['relevance']['focus_count'] == 0 and report['relevance']['explore_count'] == 1
    assert calls['review'] == calls['summary'] == 0 and len(calls['sent']) == 1
    assert '拓展阅读' in calls['sent'][0] and '正文不可用' in calls['sent'][0]


def test_rejected_papers_are_absent_from_final_report_and_history_completes_empty(config, tmp_path, monkeypatch):
    executor, calls = setup(config, tmp_path, monkeypatch, review='rejected')
    executor.run()
    report = json.loads((tmp_path / 'output/report.json').read_text())
    assert report['status'] == 'no_relevant_candidates' and report['selected_count'] == 0
    assert not any(p['selected'] for p in report['candidates'])
    assert all(p['relevance_review']['status'] == 'rejected' for p in report['candidates'])
    assert executor.history.data['covered_until'] and len(executor.history.data['seen']) == 2
    assert not calls['sent'] and calls['summary'] == 0


def test_confirmed_focus_preview_resumes_all_caches_without_advancing_history(config, tmp_path, monkeypatch):
    executor, calls = setup(config, tmp_path, monkeypatch)
    config.executor.dry_run = True
    executor.run()
    report = json.loads((tmp_path / 'output/report.json').read_text())
    assert report['status'] == 'preview_only' and report['relevance']['focus_count'] == 2
    assert calls['screen'] == calls['review'] == calls['summary'] == 2
    again = Executor(config)
    again.run()
    report = json.loads((tmp_path / 'output/report.json').read_text())
    assert report['relevance']['reused'] == report['reading_review']['reused'] == report['summaries_reused'] == 2
    assert calls['screen'] == calls['review'] == calls['summary'] == 2
    assert not calls['sent'] and again.history.data['covered_until'] is None
    assert report['papers'][0]['relevance_review']['evidence_ids'] == ['E1']


def test_failed_cache_write_never_replaces_body_after_successful_review(config, tmp_path, monkeypatch):
    from zotero_arxiv_daily.progress import PaperProgress
    executor, calls = setup(config, tmp_path, monkeypatch, body=None)
    config.executor.dry_run = True
    extractions = []
    def enrich(self, paper):
        extractions.append(paper.url)
        paper.full_text = '# Results\n' + (BODY if extractions.count(paper.url) == 1 else 'A different, unrelated body.')
    monkeypatch.setattr(ArxivRetriever, 'enrich_selected', enrich)
    monkeypatch.setattr(PaperProgress, '_write', lambda *a, **kw: None)
    executor.run()
    report = json.loads((tmp_path / 'output/report.json').read_text())
    assert report['status'] == 'preview_only' and len(extractions) == 2
    assert calls['review'] == calls['summary'] == 2 and not calls['sent']
    assert all(p['relevance_review']['evidence'] == BODY for p in report['papers'])
