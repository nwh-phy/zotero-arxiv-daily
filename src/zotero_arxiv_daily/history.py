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
HISTORY_RETENTION_DAYS = 120
HISTORY_PREFIX = 'delivery-history-'
# Earlier histories were also keyed by Zotero include/ignore paths.
LEGACY_PREFIX = 'history-'


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
        self.path = Path(directory) / (HISTORY_PREFIX + namespace + '.json') if directory else None
        self.data = {'schema': HISTORY_VERSION, 'covered_until': None,
                     'first_attempt_start': None, 'seen': {}, 'pending': None}

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
        if self.path is None:
            return
        if self.path.exists():
            self.data = self._read(self.path)
        else:
            self._adopt_legacy()

    def _adopt_legacy(self):
        # Merge histories keyed by earlier collection scopes, so changing include_path
        # neither re-sends evaluated papers nor drops the overlap window. Saved on next write.
        legacy = [self._read(path) for path in sorted(self.path.parent.glob(LEGACY_PREFIX + '*.json'))]
        if not legacy:
            return
        pending = [data['pending'] for data in legacy if data.get('pending')]
        if len(pending) > 1:
            raise RuntimeError('Several earlier histories have uncertain deliveries; reconcile them before sending')
        covered = [utc_date(data['covered_until']) for data in legacy if data.get('covered_until')]
        attempts = [utc_date(data['first_attempt_start']) for data in legacy if data.get('first_attempt_start')]
        seen = {}
        for data in legacy:
            seen.update(data['seen'])
        self.data = {'schema': HISTORY_VERSION,
                     'covered_until': max(covered).isoformat() if covered else None,
                     'first_attempt_start': None if covered or not attempts else min(attempts).isoformat(),
                     'seen': seen, 'pending': pending[0] if pending else None}

    @staticmethod
    def _read(path):
        try:
            data = json.loads(path.read_text())
            if (not isinstance(data, dict) or data.get('schema') != HISTORY_VERSION
                    or 'covered_until' not in data or 'pending' not in data
                    or not isinstance(data.get('seen'), dict)
                    or any(not isinstance(k, str) or not isinstance(v, str) for k, v in data['seen'].items())):
                raise ValueError('Invalid delivery history')
            for field in ('covered_until', 'first_attempt_start'):
                value = data.get(field)
                if value is not None:
                    if not isinstance(value, str):
                        raise ValueError('Invalid coverage timestamp')
                    utc_date(value)
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
            return data
        except (OSError, ValueError, TypeError, KeyError):
            raise RuntimeError('Delivery history is unreadable; restore or reconcile it before sending') from None

    def assert_ready(self):
        if self.data.get('pending'):
            batch = self.data['pending']['batch_id']
            raise RuntimeError(f'Delivery {batch} has an uncertain SMTP outcome. Reconcile history before sending again.')

    def window_start(self, end, lookback_days, overlap_days, max_backfill_days):
        self.assert_ready()
        if (lookback_days <= 0 or not 0 <= overlap_days <= HISTORY_RETENTION_DAYS
                or max_backfill_days < lookback_days):
            raise ValueError('Invalid retrieval coverage parameters')
        covered = self.data.get('covered_until')
        if covered:
            previous_end = utc_date(covered)
            if previous_end > end:
                raise RuntimeError('Delivery coverage is ahead of the current UTC date')
            start = previous_end - timedelta(days=overlap_days)
        elif self.data.get('first_attempt_start'):
            start = utc_date(self.data['first_attempt_start'])
            if start > end:
                raise RuntimeError('Initial attempted coverage is ahead of the current UTC date')
        else:
            start = end - timedelta(days=lookback_days)
        if start == end:
            # Still check an overlap when rerun on a completed boundary.
            start = end - timedelta(days=1)
        if end - start > timedelta(days=max_backfill_days):
            raise RuntimeError('Missed coverage exceeds max_backfill_days; increase it explicitly to backfill')
        return start

    def remember_first_attempt(self, start):
        # Persist intent before retrieval, without claiming successful coverage.
        # A failed first week must still be included in the next week's retry.
        if not self.data.get('covered_until') and not self.data.get('first_attempt_start'):
            self.data['first_attempt_start'] = utc_date(start.isoformat()).isoformat()
            self.save()

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
        self.data['first_attempt_start'] = None
        self.data['seen'].update(seen)
        cutoff = utc_date(end) - timedelta(days=HISTORY_RETENTION_DAYS)
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
