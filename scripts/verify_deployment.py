"""Live verification; sending a labelled test email requires --send-test."""
import json
import argparse
import os
from pathlib import Path
import smtplib
import ssl
import urllib.request
import urllib.parse
import time
import feedparser

import numpy as np
from hydra import compose, initialize_config_dir
from openai import OpenAI
from omegaconf import OmegaConf
from zotero_arxiv_daily.executor import Executor
from zotero_arxiv_daily.construct_email import render_email
from zotero_arxiv_daily.protocol import Paper
from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _entry_to_arxiv_result
from zotero_arxiv_daily.reranker.local import LocalReranker
from zotero_arxiv_daily.utils import send_email

parser = argparse.ArgumentParser()
parser.add_argument('--summary-only', action='store_true')
parser.add_argument('--send-test', action='store_true')
args = parser.parse_args()

with initialize_config_dir(config_dir=str(Path('config').resolve()), version_base=None):
    config = compose(config_name='default')
params = OmegaConf.to_container(config.llm, resolve=True)
out = Path('verification')
out.mkdir(exist_ok=True)
results = {'verification_scope': {'status': 'INFO', 'detail': 'summary-only' if args.summary_only else 'full'}}
failed = False

def check(name, fn):
    global failed
    try:
        detail = fn()
        results[name] = {'status': 'PASS', 'detail': detail}
    except Exception as exc:
        # External exceptions can embed headers, credentials or request bodies.
        results[name] = {'status': 'FAIL', 'error_type': type(exc).__name__}
        failed = True
    print(name + ': ' + results[name]['status'])

executor = Executor(config)
corpus = []
papers = []
ranked = []
candidate_origin = 'current RSS'

def zotero_check():
    global corpus
    all_corpus = executor.fetch_zotero_corpus()
    corpus = executor.filter_corpus(all_corpus)
    assert corpus, 'No usable abstracts'
    return {'paper_types_with_abstract': len(all_corpus), 'selected_with_abstract': len(corpus)}
if not args.summary_only:
    check('zotero', zotero_check)
else:
    results['zotero'] = {'status': 'SKIP', 'reason': 'Targeted summary retest; use prior full-run evidence'}

def embedding_check():
    reranker = LocalReranker(config)
    sim = reranker.get_similarity_score(['Electron energy loss spectroscopy of quantum materials'],
                                      ['Momentum resolved EELS in two dimensional materials'])
    assert sim.shape == (1, 1) and np.isfinite(sim).all()
    return {'model': config.reranker.local.model, 'device': 'cpu', 'shape': list(sim.shape)}
if not args.summary_only:
    check('embedding', embedding_check)
else:
    results['embedding'] = {'status': 'SKIP', 'reason': 'Unchanged; previously verified in full run'}

def arxiv_check():
    global papers
    raw = ArxivRetriever(config)._retrieve_raw_papers()
    papers = [Paper(source='arxiv', title=p.title, authors=[a.name for a in p.authors],
                    abstract=p.summary, url=p.entry_id, pdf_url=p.pdf_url,
                    published=p.published.isoformat(), journal_ref=p.journal_ref or None) for p in raw]
    # Empty is valid on days without new announcements, not an E2E proof.
    return {'new_papers': len(papers), 'categories': list(config.source.arxiv.category)}
check('arxiv', arxiv_check)

# Verification only: empty daily RSS must still suppress production delivery.
# Real older arXiv records exercise ranking/summary without inventing papers.
if not papers:
    def historical_check():
        global papers, candidate_origin
        query = ' OR '.join('cat:' + c for c in config.source.arxiv.category)
        url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode({
            'search_query': query, 'max_results': 10, 'sortBy': 'submittedDate', 'sortOrder': 'descending'})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    feed = feedparser.parse(response.read())
                raw = [_entry_to_arxiv_result(entry) for entry in feed.entries]
                if not raw:
                    raise ValueError('No historical records returned')
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(3)
        papers = [Paper(source='arxiv', title=p.title, authors=[a.name for a in p.authors],
                        abstract=p.summary, url=p.entry_id, pdf_url=p.pdf_url,
                        published=p.published.isoformat(), journal_ref=p.journal_ref or None) for p in raw]
        assert papers, 'No historical records returned'
        candidate_origin = 'historical arXiv API records (verification only)'
        return {'records': len(papers), 'production_fallback': False}
    check('historical_arxiv_sample', historical_check)

def ranking_check():
    global ranked
    ranked = executor.reranker.rerank(papers, corpus)[:config.executor.max_paper_num]
    assert 0 < len(ranked) <= 10
    assert all(ranked[i].score >= ranked[i+1].score for i in range(len(ranked)-1))
    return {'selected': len(ranked), 'candidate_origin': candidate_origin,
            'descending_scores': [round(float(p.score), 4) for p in ranked]}
if papers and corpus:
    check('ranking', ranking_check)
else:
    results['ranking'] = {'status': 'SKIP', 'reason': 'Targeted summary retest' if args.summary_only else 'No live papers or usable corpus'}

def llm_check():
    from zotero_arxiv_daily.protocol import _request_llm
    answer = _request_llm(executor.openai_client, params,
                          [{'role': 'user', 'content': '仅回答：连接正常'}])
    assert answer and '连接正常' in answer
    return {'model': params['generation_kwargs']['model'], 'api_mode': params['api_mode']}
check('llm', llm_check)

def summary_check():
    p = ranked[0] if ranked else papers[0]
    summary = p.generate_tldr(executor.openai_client, params)
    (out/'sample-draft.txt').write_text(p.title+'\n'+p.url+'\n'+summary)
    assert any('\u4e00' <= c <= '\u9fff' for c in summary)
    assert all(label in summary for label in ['问题', '方法', '结论', '创新', '参考价值'])
    assert '摘要确认' in summary
    assert '全文片段确认' not in summary, 'Abstract-only sample claims full-text evidence'
    chinese_characters = sum('\u4e00' <= c <= '\u9fff' for c in summary)
    assert 150 <= chinese_characters <= 250, 'Summary length outside requested range'
    html = render_email([p])
    assert p.url in html and p.pdf_url in html
    assert p.pdf_url.startswith('https://arxiv.org/pdf/') and '://' not in p.pdf_url.removeprefix('https://arxiv.org/pdf/')
    (out/'sample.html').write_text(html)
    (out/'sample.txt').write_text(p.title+'\n'+summary)
    return {'characters': len(summary), 'chinese_characters': chinese_characters, 'candidate_origin': candidate_origin,
            'evidence': 'abstract-only', 'sample_title': p.title}
if papers:
    check('chinese_summary_and_html', summary_check)
else:
    results['chinese_summary_and_html'] = {'status': 'SKIP', 'reason': 'No live papers; no substitute invented'}

smtp_names = ['SENDER', 'RECEIVER', 'SENDER_PASSWORD', 'SMTP_SERVER']
missing = [name for name in smtp_names if not os.environ.get(name)]
if missing:
    results['smtp'] = {'status': 'BLOCKED', 'missing': missing}
    print('smtp: BLOCKED (missing settings)')
else:
    def smtp_check():
        host, port = config.email.smtp_server, int(config.email.smtp_port)
        if port == 465:
            server = smtplib.SMTP_SSL(host, port, timeout=30, context=ssl.create_default_context())
        else:
            server = smtplib.SMTP(host, port, timeout=30)
            server.starttls(context=ssl.create_default_context())
        try:
            server.login(config.email.sender, config.email.sender_password)
        finally:
            server.quit()
        return {'authenticated': True, 'email_sent': False}
    check('smtp', smtp_check)
if args.send_test and not failed and results.get('smtp', {}).get('status') == 'PASS' and results.get('chinese_summary_and_html', {}).get('status') == 'PASS':
    def send_test_check():
        banner = '<p><strong>部署测试邮件</strong>：此邮件用于验证 Gmail 投递与中文简报显示。</p>'
        if candidate_origin != 'current RSS':
            banner += '<p>以下为真实历史 arXiv 样本，仅用于部署测试，不代表今日推荐。</p>'
        send_email(config, banner + (out/'sample.html').read_text(), subject='[部署测试] Zotero arXiv 中文文献简报')
        return {'smtp_accepted': True, 'inbox_confirmed': False, 'candidate_origin': candidate_origin}
    check('end_to_end_email', send_test_check)
else:
    results['end_to_end_email'] = {'status': 'NOT_RUN', 'reason': 'Test sending not requested or prerequisites did not pass'}
(out/'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
summary = '\n'.join(f"- {name}: {r['status']}" for name,r in results.items())
if step_summary := os.environ.get('GITHUB_STEP_SUMMARY'):
    with open(step_summary, 'a') as f:
        f.write('Deployment verification (PASS does not imply email delivery)\n\n'+summary+'\n')
print(summary)
raise SystemExit(1 if failed else 0)
