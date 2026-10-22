"""Score an archived ranking (evaluation.json from a run's artifact) against reading labels; offline, no API calls.

  # 1. Write a CSV to label: put 精读 / 浏览 / 不相关 in the label column (blank = not judged).
  uv run --frozen scripts/evaluate_ranking.py BUNDLE --template labels.csv
  # 2. Score the recorded selection and stage-1 order; optionally replay stage 1 with other options.
  uv run --frozen scripts/evaluate_ranking.py BUNDLE --labels labels.csv --set theme_penalty=0.02 --set mode=legacy

Only stage 1 (embedding ranking) can be replayed; LLM judgements exist only for the recorded shortlist.
"""
import argparse
import json
from pathlib import Path

from zotero_arxiv_daily.evaluation import read_labels, replay, score, stage1_order, write_label_template

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('bundle', type=Path)
parser.add_argument('--template', type=Path, help='Write a label CSV for this bundle and exit')
parser.add_argument('--labels', type=Path, help='Label CSV filled with 精读/浏览/不相关')
parser.add_argument('--set', action='append', default=[], metavar='OPTION=VALUE',
                    help='Ranking option override for a stage-1 replay, e.g. theme_penalty=0.02')
args = parser.parse_args()
bundle = json.loads(args.bundle.read_text(encoding='utf-8'))
if args.template:
    write_label_template(bundle, args.template)
    raise SystemExit(f'Wrote {len(bundle["candidates"])} candidates to {args.template}')
if not args.labels:
    raise SystemExit('Pass --template to create a label file or --labels to score one')
labels = read_labels(args.labels)
result = {'labelled': len(labels), 'recorded': score(bundle, labels)}
if args.set:
    overrides = {}
    for item in args.set:
        key, _, value = item.partition('=')
        overrides[key] = value if key == 'mode' else float(value) if '.' in value else int(value)
    order, recorded = replay(bundle, **overrides), stage1_order(bundle)
    replayed = score(bundle, labels, order)
    result['replayed'] = {'overrides': overrides, 'stage1_top10': replayed['stage1_top10'],
                          'stage1_top30': replayed['stage1_top30'],
                          'top10_changes': {'added': [i for i in order[:10] if i not in recorded[:10]],
                                            'removed': [i for i in recorded[:10] if i not in order[:10]]}}
print(json.dumps(result, ensure_ascii=False, indent=2))
