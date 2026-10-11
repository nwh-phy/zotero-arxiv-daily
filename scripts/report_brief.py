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
    print()
    print('SMTP acceptance does not establish recipient inbox arrival. An uncertain submission requires history reconciliation before retrying.')
