"""Atomic, credential-free per-paper checkpoints. This never records delivery state."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from loguru import logger
from omegaconf import OmegaConf

from .evidence import EVIDENCE_VERSION, build_summary_prompt

CACHE_VERSION = 1
EXTRACTOR_VERSION = 2


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class PaperProgress:
    def __init__(self, directory, library_id):
        # Separate libraries even when a runner or local cache directory is shared.
        self.directory = Path(directory) / _digest(str(library_id)) if directory else None

    def key(self, paper):
        return _digest({'schema': CACHE_VERSION, 'extractor': EXTRACTOR_VERSION,
                        'source': paper.source, 'url': paper.url, 'pdf_url': paper.pdf_url,
                        'title': paper.title, 'abstract': paper.abstract})

    def load(self, paper):
        if self.directory is None:
            return {}
        path = self.directory / (self.key(paper) + '.json')
        try:
            record = json.loads(path.read_text())
            if (not isinstance(record, dict) or record.get('key') != self.key(paper)
                    or record.get('schema') != CACHE_VERSION
                    or record.get('checksum') != _digest(record.get('payload'))):
                raise ValueError('Invalid checkpoint')
            payload = record['payload']
            if (not isinstance(payload, dict)
                    or not isinstance(payload.get('full_text'), (str, type(None)))):
                raise ValueError('Invalid checkpoint payload')
            return payload
        except FileNotFoundError:
            return {}
        except (OSError, ValueError, TypeError):
            logger.warning('Unreadable paper checkpoint; this paper will be processed again')
            return {}

    def summary_key(self, paper, params):
        params = OmegaConf.to_container(params, resolve=True) if OmegaConf.is_config(params) else dict(params)
        # Hash endpoint and generation policy, never save credentials or configuration.
        policy = {key: value for key, value in params.items() if key != 'api'}
        policy['endpoint'] = params.get('api', {}).get('base_url')
        prompt = build_summary_prompt(paper, params.get('language', 'English'))
        return _digest({'evidence_version': EVIDENCE_VERSION, 'policy': policy, 'prompt': prompt})

    def restore(self, paper, params):
        payload = self.load(paper)
        if paper.full_text and 'full_text' in payload and paper.full_text != payload['full_text']:
            return False
        if 'full_text' in payload:
            paper.full_text = payload['full_text']
        summary_key = self.summary_key(paper, params)
        summary = payload.get('summary')
        if (payload.get('summary_key') == summary_key and payload.get('status') == 'complete'
                and paper.summary_is_acceptable(summary, params)):
            paper.tldr = summary
            logger.info('Reusing a validated paper summary')
            return True
        return False

    def has_extraction(self, paper):
        # Retry unsuccessful extraction when there is no reusable summary.
        return bool(self.load(paper).get('full_text'))

    def save(self, paper, params, *, status, error_type=None):
        if self.directory is None:
            return
        payload = {'status': status, 'full_text': paper.full_text,
                   'summary_key': self.summary_key(paper, params)}
        if status == 'complete' and paper.summary_is_acceptable(paper.tldr, params) and paper.tldr != paper.abstract:
            payload['summary'] = paper.tldr
        if error_type:
            payload['error_type'] = error_type
        record = {'schema': CACHE_VERSION, 'key': self.key(paper), 'payload': payload,
                  'checksum': _digest(payload)}
        temporary = None
        try:
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            with NamedTemporaryFile(mode='w', dir=self.directory, delete=False, encoding='utf-8') as file:
                temporary = Path(file.name)
                json.dump(record, file, ensure_ascii=False)
                file.flush()
                os.fsync(file.fileno())
            temporary.replace(self.directory / (record['key'] + '.json'))
        except OSError:
            # Cache availability must not prevent otherwise valid processing.
            logger.warning('Paper checkpoint could not be saved')
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    logger.warning('Temporary paper checkpoint could not be removed')
