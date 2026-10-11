"""Explicitly reconcile one restored uncertain SMTP attempt; never sends mail."""
import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir

from zotero_arxiv_daily.executor import Executor

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--outcome', required=True, choices=['inspect', 'accepted', 'not-accepted'])
parser.add_argument('--batch-id', help='Exact batch ID from the uncertain-delivery error')
args = parser.parse_args()
with initialize_config_dir(config_dir=str(Path('config').resolve()), version_base=None):
    config = compose(config_name='default')
executor = Executor(config)
if not executor.history_enabled:
    raise SystemExit('Delivery history is disabled')
with executor.history.locked():
    pending = executor.history.data.get('pending')
    if args.outcome == 'inspect':
        if pending:
            print('Pending batch: ' + pending['batch_id'])
            print('Pending coverage end: ' + pending['end'])
        else:
            print('No uncertain delivery pending')
        raise SystemExit(0)
    if not pending or pending['batch_id'] != args.batch_id:
        raise SystemExit('No uncertain delivery matches that batch ID; nothing was changed')
    if args.outcome == 'accepted':
        executor.history.accepted()
    else:
        executor.history.rejected()
print('History reconciled; no email was sent')
