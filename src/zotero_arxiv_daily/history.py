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


def _check_papers(papers):
    if not isinstance(papers, list) or any(
            not isinstance(p, dict) or not isinstance(p.get('id'), str) or not isinstance(p.get('title'), str)
            for p in papers):
        raise ValueError('Invalid delivered paper list')


def scope_namespace(scope):
    return hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()


class DeliveryHistory:
    def __init__(self, directory, scope, legacy_scopes=()):
        self.namespace = scope_namespace(scope)
        self.path = Path(directory) / (HISTORY_PREFIX + self.namespace + '.json') if directory else None
        # Earlier file names also hashed include/ignore paths. Only names recomputed from this
        # subscription's own receiver, categories and cross-list setting may be adopted.
        if any(legacy.get(key) != value for legacy in legacy_scopes for key, value in scope.items()):
            raise ValueError('Legacy history scopes must belong to the current subscription')
        self.legacy_paths = list(dict.fromkeys(
            Path(directory) / (LEGACY_PREFIX + scope_namespace(legacy) + '.json') for legacy in legacy_scopes
        )) if directory else []
        self.legacy_pending = []
        # Non-sensitive scope summary saved for inspection; the receiver stays hashed in the name.
        self.scope_summary = {key: value for key, value in scope.items() if key != 'receiver'}
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
        # Merge this subscription's histories keyed by earlier collection scopes, so changing
        # include_path neither re-sends evaluated papers nor drops the overlap window. Saved on next write.
        legacy = [(path, self._read(path)) for path in self.legacy_paths if path.exists()]
        if not legacy:
            return
        pending = [(path, data['pending']) for path, data in legacy if data.get('pending')]
        if len(pending) > 1:
            # Each batch is reconciled in its own file; sending stays blocked until all are.
            self.legacy_pending = pending
        covered = [utc_date(data['covered_until']) for _, data in legacy if data.get('covered_until')]
        attempts = [utc_date(data['first_attempt_start']) for _, data in legacy if data.get('first_attempt_start')]
        seen = {}
        for _, data in legacy:
            seen.update(data['seen'])
        self.data = {'schema': HISTORY_VERSION,
                     'covered_until': max(covered).isoformat() if covered else None,
                     'first_attempt_start': None if covered or not attempts else min(attempts).isoformat(),
                     'seen': seen, 'pending': pending[0][1] if len(pending) == 1 else None,
                     'migrated_from': [path.name for path, _ in legacy]}
        deliveries = [data['last_delivery'] for _, data in legacy if data.get('last_delivery')]
        if deliveries:
            self.data['last_delivery'] = max(deliveries, key=lambda last: utc_date(last['end']))

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
                _check_papers(pending.get('papers', []))
            if data.get('last_delivery') is not None:
                utc_date(data['last_delivery']['end'])
                _check_papers(data['last_delivery']['papers'])
            return data
        except (OSError, ValueError, TypeError, KeyError):
            raise RuntimeError('Delivery history is unreadable; restore or reconcile it before sending') from None

    def assert_ready(self):
        if self.data.get('pending'):
            batch = self.data['pending']['batch_id']
            raise RuntimeError(f'Delivery {batch} has an uncertain SMTP outcome. Reconcile history before sending again.')
        if self.legacy_pending:
            batches = ', '.join(pending['batch_id'] for _, pending in self.legacy_pending)
            raise RuntimeError(f'Earlier deliveries {batches} have uncertain SMTP outcomes. Reconcile each batch before sending again.')

    def pending_batches(self):
        return ([self.data['pending']] if self.data.get('pending') else []) + [p for _, p in self.legacy_pending]

    def reconcile(self, batch_id, accepted):
        """Resolve one uncertain batch by its exact ID; False when no pending batch matches."""
        if self.data.get('pending') and self.data['pending']['batch_id'] == batch_id:
            target = self
        else:
            match = next((path for path, pending in self.legacy_pending if pending['batch_id'] == batch_id), None)
            if match is None:
                return False
            # Resolve the batch inside its own earlier file; the next run adopts the result.
            target = DeliveryHistory(None, {})
            target.path, target.data = match, self._read(match)
        if accepted:
            target.accepted()
        else:
            target.rejected()
        self.legacy_pending = [(p, b) for p, b in self.legacy_pending if b['batch_id'] != batch_id]
        return True

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
                                'end': end, 'seen': self._seen_records(retrieved, end),
                                'papers': [{'id': paper_identity(p), 'title': p.title} for p in selected]}
        self.save()
        return batch_id

    def rejected(self):
        self.data['pending'] = None
        self.save()

    def accepted(self):
        pending = self.data.get('pending')
        if not pending:
            return
        end, covered, last = pending['end'], self.data.get('covered_until'), self.data.get('last_delivery')
        if pending.get('papers') is not None and (not last or utc_date(last['end']) <= utc_date(end)):
            # Lets the next run report how many delivered papers were saved to Zotero.
            self.data['last_delivery'] = {'end': end, 'papers': pending['papers']}
        # An adopted older batch can end before later coverage: record its papers, keep the later boundary.
        self.complete(pending['seen'], covered if covered and utc_date(end) < utc_date(covered) else end)

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
        if self.scope_summary:
            self.data['scope'] = self.scope_summary
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
