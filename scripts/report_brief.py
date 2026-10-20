"""Publish aggregate run status, never resolved config or paper/reference bodies."""
import json
from pathlib import Path

path = Path('verification/brief/report.json')
if not path.exists():
    print('Brief report unavailable: delivery may have been skipped or stopped before rendering. Inspect the workflow steps.')
else:
    report = json.loads(path.read_text())
    print('### Literature brief')
    print()
    print('Status: ' + str(report.get('status', 'unknown')))
    print()
    for label, key in [('Retrieved', 'retrieved_count'), ('New candidates', 'candidate_count'),
                       ('Selected', 'selected_count'), ('Summaries reused', 'summaries_reused'),
                       ('Summaries generated', 'summaries_generated'), ('Failed', 'failed_count'),
                       ('Duration (seconds)', 'duration_seconds')]:
        print(f'- {label}: {report.get(key, 0)}')
    if relevance := report.get('relevance'):
        print(f"- LLM relevance: {relevance.get('mode')}; judged {relevance.get('judged', 0)}, "
              f"passed {relevance.get('passed', 0)}, failed {relevance.get('failed', 0)}")
    if previous := report.get('previous_delivery'):
        print(f"- Previous brief saved to Zotero: {previous.get('saved', 0)}/{previous.get('delivered', 0)}")
    print()
    print('SMTP acceptance does not establish recipient inbox arrival. An uncertain submission requires history reconciliation before retrying.')
