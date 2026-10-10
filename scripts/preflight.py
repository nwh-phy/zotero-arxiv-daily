"""Check syntax and credential presence without logging resolved configuration."""
import argparse
import os
from pathlib import Path
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--syntax-only', action='store_true')
args = parser.parse_args()
for path in [*Path('config').glob('*.yaml'), *Path('.github/workflows').glob('*.yml')]:
    yaml.safe_load(path.read_text())
print('YAML syntax: PASS')
if not args.syntax_only:
    required = ['ZOTERO_ID', 'ZOTERO_KEY', 'OPENAI_API_KEY', 'OPENAI_API_BASE',
                'SENDER', 'RECEIVER', 'SENDER_PASSWORD', 'SMTP_SERVER']
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit('Missing credentials/settings: ' + ', '.join(missing))
    print('Required credential presence: PASS')
