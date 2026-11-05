"""Merge public Actions overrides without erasing new repository defaults."""
import argparse
import os
from pathlib import Path
import re

from omegaconf import OmegaConf

_CREDENTIALS = ('zotero.api_key', 'email.sender_password', 'llm.api.key', 'reranker.api.key')
_ENV_REFERENCE = re.compile(r'\$\{oc\.env:[A-Za-z_][A-Za-z0-9_]*(?:,(?:null)?)?\}')
_RECOMMENDATION_DEFAULTS = ('reranker.relevance', 'executor.max_paper_num', 'llm.summary_instructions')


def apply_override(path: Path, override: str, *, repository_recommendations: bool = False):
    defaults = OmegaConf.load(path)
    recommendation_defaults = {}
    if repository_recommendations:
        raw_defaults = OmegaConf.to_container(defaults, resolve=False)
        for dotted in _RECOMMENDATION_DEFAULTS:
            value = raw_defaults
            for part in dotted.split('.'):
                if not isinstance(value, dict) or part not in value:
                    raise ValueError(f'Repository configuration must define {dotted}')
                value = value[part]
            if value is None or value == '???':
                raise ValueError(f'Repository configuration must define {dotted}')
            recommendation_defaults[dotted] = value
    custom = OmegaConf.create(override) if override.strip() else OmegaConf.create({})
    if not OmegaConf.is_dict(custom):
        raise ValueError('CUSTOM_CONFIG must be a YAML mapping')
    merged = OmegaConf.merge(defaults, custom)
    for dotted, value in recommendation_defaults.items():
        # Replace whole subtrees so removed repository settings cannot survive an old override.
        OmegaConf.update(merged, dotted, value, merge=False)
    raw = OmegaConf.to_container(merged, resolve=False)
    for dotted in _CREDENTIALS:
        value = raw
        for part in dotted.split('.'):
            value = value.get(part) if isinstance(value, dict) else None
        if value is not None and (not isinstance(value, str) or not _ENV_REFERENCE.fullmatch(value)):
            raise ValueError(f'{dotted} must reference an Actions Secret through oc.env')
    OmegaConf.save(merged, path, resolve=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository-recommendations', action='store_true',
                        help='Use checked-out recommendation settings while retaining other Actions overrides')
    args = parser.parse_args()
    try:
        apply_override(Path('config/custom.yaml'), os.environ.get('CUSTOM_CONFIG', ''),
                       repository_recommendations=args.repository_recommendations)
    except Exception as exc:
        raise SystemExit(f'Configuration override failed ({type(exc).__name__}); use a YAML mapping and environment references for credentials') from None
    print('Public configuration override merged; no resolved values printed')
