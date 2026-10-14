"""Merge public Actions overrides without erasing new repository defaults."""
import os
from pathlib import Path
import re

from omegaconf import OmegaConf

_CREDENTIALS = ('zotero.api_key', 'email.sender_password', 'llm.api.key', 'reranker.api.key')
_ENV_REFERENCE = re.compile(r'\$\{oc\.env:[A-Za-z_][A-Za-z0-9_]*(?:,(?:null)?)?\}')


def apply_override(path: Path, override: str):
    defaults = OmegaConf.load(path)
    custom = OmegaConf.create(override) if override.strip() else OmegaConf.create({})
    if not OmegaConf.is_dict(custom):
        raise ValueError('CUSTOM_CONFIG must be a YAML mapping')
    merged = OmegaConf.merge(defaults, custom)
    raw = OmegaConf.to_container(merged, resolve=False)
    for dotted in _CREDENTIALS:
        value = raw
        for part in dotted.split('.'):
            value = value.get(part) if isinstance(value, dict) else None
        if value is not None and (not isinstance(value, str) or not _ENV_REFERENCE.fullmatch(value)):
            raise ValueError(f'{dotted} must reference an Actions Secret through oc.env')
    OmegaConf.save(merged, path, resolve=False)


if __name__ == '__main__':
    try:
        apply_override(Path('config/custom.yaml'), os.environ.get('CUSTOM_CONFIG', ''))
    except Exception as exc:
        raise SystemExit(f'Configuration override failed ({type(exc).__name__}); use a YAML mapping and environment references for credentials') from None
    print('Public configuration override merged; no resolved values printed')
