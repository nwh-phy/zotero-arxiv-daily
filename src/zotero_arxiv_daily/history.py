"""Coverage and recommendation history, separate from summary checkpoints.

Persist only public paper IDs and timestamps. An uncertain SMTP attempt blocks
further sends until explicitly reconciled; SMTP is not an exactly-once protocol.
"""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from urllib.parse import urlparse


HISTORY_VERSION = 1


def paper_identity(paper):
    if paper.source == 'arxiv':
        path = urlparse(paper.url).path
        path = re.sub(r'^/(?:abs|pdf)/', '', path).removesuffix('.pdf')
        return 'arxiv:' + re.sub(r'v\d+$', '', path)
    return paper.source + ':' + paper.url


def utc_date(value):
    date = datetime.fromisoformat(value)
    if date.tzinfo is None:
        raise ValueError('Coverage timestamps must have a timezone')
    return date.astimezone(timezone.utc)


class DeliveryHistory:
    def __init__(self, directory, scope):
        namespace = hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()
        self.namespace = namespace
        self.path = Path(directory) / ('history-' + namespace + '.json') if directory else None
        self.data = {'schema': HISTORY_VERSION, 'covered_until': None, 'seen': {}, 'pending': None}

    @contextmanager
    def locked(self):
        if self.path is None:
            yield self
            return
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_path = self.path.with_suffix('.lock')
        with lock_path.open('a') as lock:
            os.chmod(lock_path, 0o600)
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Another literature delivery is using this history') from None
            try:
                self.load()
                yield self
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def load(self):
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            if (not isinstance(data, dict) or data.get('schema') != HISTORY_VERSION
                    or not isinstance(data.get('seen'), dict)
                    or any(not isinstance(k, str) or not isinstance(v, str) for k, v in data['seen'].items())):
                raise ValueError('Invalid delivery history')
            if data.get('covered_until'):
                utc_date(data['covered_until'])
            for date in data['seen'].values():
                utc_date(date)
            pending = data.get('pending')
            if pending is not None:
                if (not isinstance(pending, dict) or pending.get('status') != 'sending'
                        or not isinstance(pending.get('batch_id'), str)
                        or not isinstance(pending.get('seen'), dict)):
                    raise ValueError('Invalid pending delivery')
                utc_date(pending['end'])
                for key, date in pending['seen'].items():
                    if not isinstance(key, str):
                        raise ValueError('Invalid pending paper ID')
                    utc_date(date)
            self.data = data
        except (OSError, ValueError, TypeError, KeyError):
            raise RuntimeError('Delivery history is unreadable; restore or reconcile it before sending') from None

    def assert_ready(self):
        if self.data.get('pending'):
            batch = self.data['pending']['batch_id']
            raise RuntimeError(f'Delivery {batch} has an uncertain SMTP outcome. Reconcile history before sending again.')

    def window_start(self, end, lookback_days, overlap_days, max_backfill_days):
        self.assert_ready()
        if lookback_days <= 0 or overlap_days < 0 or max_backfill_days < lookback_days:
            raise ValueError('Invalid retrieval coverage parameters')
        covered = self.data.get('covered_until')
        if covered:
            previous_end = utc_date(covered)
            if previous_end > end:
                raise RuntimeError('Delivery coverage is ahead of the current UTC date')
            start = previous_end - timedelta(days=overlap_days)
        else:
            start = end - timedelta(days=lookback_days)
        if start == end:
            # Still check an overlap when rerun on a completed boundary.
            start = end - timedelta(days=1)
        if end - start > timedelta(days=max_backfill_days):
            raise RuntimeError('Missed coverage exceeds max_backfill_days; increase it explicitly to backfill')
        return start

    def unseen(self, papers):
        seen = self.data['seen']
        unique = {}
        for paper in papers:
            identity = paper_identity(paper)
            if identity not in seen:
                unique.setdefault(identity, paper)
        return list(unique.values())

    def _seen_records(self, papers, end):
        records = {}
        for paper in papers:
            try:
                published = utc_date(paper.published).isoformat()
            except (TypeError, ValueError):
                published = end
            records[paper_identity(paper)] = published
        return records

    def begin(self, selected, retrieved, end):
        self.assert_ready()
        if self.path is None:
            return None
        batch_id = hashlib.sha256(json.dumps({'scope': self.namespace, 'end': end, 'ids': [paper_identity(p) for p in selected]}, sort_keys=True).encode()).hexdigest()[:24]
        self.data['pending'] = {'status': 'sending', 'batch_id': batch_id,
                                'end': end, 'seen': self._seen_records(retrieved, end)}
        self.save()
        return batch_id

    def rejected(self):
        self.data['pending'] = None
        self.save()

    def accepted(self):
        pending = self.data.get('pending')
        if not pending:
            return
        self.complete(pending['seen'], pending['end'])

    def complete_empty(self, retrieved, end):
        self.complete(self._seen_records(retrieved, end), end)

    def complete(self, seen, end):
        if self.data.get('covered_until') and utc_date(end) < utc_date(self.data['covered_until']):
            raise ValueError('Coverage cannot move backwards')
        self.data['covered_until'] = end
        self.data['seen'].update(seen)
        cutoff = utc_date(end) - timedelta(days=120)
        self.data['seen'] = {key: date for key, date in self.data['seen'].items() if utc_date(date) >= cutoff}
        self.data['pending'] = None
        self.save()

    def save(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = None
        try:
            with NamedTemporaryFile(mode='w', dir=self.path.parent, delete=False, encoding='utf-8') as file:
                temporary = Path(file.name)
                json.dump(self.data, file, ensure_ascii=False, sort_keys=True)
                file.flush()
                os.fsync(file.fileno())
            temporary.replace(self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
