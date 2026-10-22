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
    batches = executor.history.pending_batches()
    if args.outcome == 'inspect':
        # Adopted earlier histories can hold several batches; each is reconciled separately.
        for pending in batches:
            print('Pending batch: ' + pending['batch_id'])
            print('Pending coverage end: ' + pending['end'])
        if not batches:
            print('No uncertain delivery pending')
        raise SystemExit(0)
    if not args.batch_id or not executor.history.reconcile(args.batch_id, args.outcome == 'accepted'):
        raise SystemExit('No uncertain delivery matches that batch ID; nothing was changed')
print('History reconciled; no email was sent')
