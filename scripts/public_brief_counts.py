"""Write allowlisted numeric counters; keep private reports on the runner."""
import json
from pathlib import Path

SOURCE = Path('verification/brief/report.json')
OUTPUT = Path('verification/public-brief-counts.json')
COUNT_KEYS = (
    'retrieved_count', 'candidate_count', 'selected_count',
    'summaries_reused', 'summaries_generated', 'failed_count',
)
GROUP_KEYS = {
    'relevance': ('attempted', 'scored', 'focus_count', 'explore_count', 'failed', 'skipped', 'reused'),
    'reading_review': ('attempted', 'confirmed', 'downgraded', 'rejected', 'unavailable', 'failed', 'skipped', 'reused'),
    'previous_delivery': ('saved', 'delivered'),
}


def count_fields(record, keys):
    if not isinstance(record, dict):
        return {}
    return {
        key: record[key] for key in keys
        if type(record.get(key)) is int and record[key] >= 0
    }


def public_counts(report):
    counts = count_fields(report, COUNT_KEYS)
    if isinstance(report, dict):
        for group, keys in GROUP_KEYS.items():
            if nested := count_fields(report.get(group), keys):
                counts[group] = nested
    return counts


def main():
    OUTPUT.unlink(missing_ok=True)
    if not SOURCE.is_file():
        return
    counts = public_counts(json.loads(SOURCE.read_text(encoding='utf-8')))
    OUTPUT.write_text(json.dumps(counts, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
