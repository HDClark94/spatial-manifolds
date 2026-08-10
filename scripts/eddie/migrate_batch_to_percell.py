"""
Convert existing *batch* xgboost CSVs into *per-target-cell* CSVs.

Old naming (one file per batch, index range):
    {prefix}_M{m}_D{d}_h{h}[_{ENV}]_{cell_start}_{cell_end}.csv
New naming (one file per target cell):
    {prefix}_M{m}_D{d}_h{h}[_{ENV}]_C{target_cluster_id}.csv

The per-cell name is derived generically by stripping the trailing `_{int}_{int}`
and appending `_C{target_cluster_id}`, so it works for every assay (pairwise keeps
its OF1/VR token, medlat/extra keep theirs). Rows are grouped by `target_cluster_id`.

Idempotent: a per-cell file that already exists is left untouched. Writes are atomic
(temp + rename). Batch files are KEPT unless you pass --delete.

Empty (0-byte) batch files are failed/incomplete jobs — they hold nothing to split and
their target cells simply have no output, so the master will re-run them. They're counted
and skipped quietly; pass --clean-empty to delete them.

Usage:
    python migrate_batch_to_percell.py <eddie_base_dir> [--delete] [--clean-empty] [--dry-run]

  <eddie_base_dir>    e.g. the datastore  .../Harry/COHORT12/eddie
                      or scratch          /exports/eddie/scratch/hclark3/COHORT12/eddie
                      or local mirror      /Users/harryclark/Documents/spatial-manifolds/data/eddie
It walks the tree, so any subfolder layout is handled.
"""
import os
import re
import sys
import pandas as pd

BATCH_RE = re.compile(r'^(.*)_(\d+)_(\d+)\.csv$')   # {prefix}_{start}_{end}.csv


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not args:
        print(__doc__); sys.exit(1)
    base = args[0]
    delete = '--delete' in sys.argv
    clean_empty = '--clean-empty' in sys.argv
    dry = '--dry-run' in sys.argv
    if not os.path.isdir(base):
        print(f'Not a directory: {base}'); sys.exit(1)

    total_batches = total_cells = total_empty = 0
    for root, _dirs, files in os.walk(base):
        n_batches = n_cells = n_empty = 0
        for fn in sorted(files):
            if not fn.endswith('.csv'):
                continue
            m = BATCH_RE.match(fn)
            if not m:
                continue  # already per-cell (_C..), aggregated, or not a batch file
            prefix = m.group(1)
            path = os.path.join(root, fn)
            # Empty (0-byte) or unreadable file = a failed/incomplete batch: nothing to
            # split, and its target cells have no output, so they'll be re-run by the master.
            if os.path.getsize(path) == 0:
                n_empty += 1
                if clean_empty and not dry:
                    os.remove(path)
                continue
            try:
                df = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                n_empty += 1
                if clean_empty and not dry:
                    os.remove(path)
                continue
            except Exception as e:
                print(f'  skip (read error) {path}: {e}'); continue
            if 'target_cluster_id' not in df.columns:
                print(f'  skip (no target_cluster_id column) {fn}'); continue
            wrote = 0
            for tid, g in df.groupby('target_cluster_id'):
                out = os.path.join(root, f'{prefix}_C{int(tid)}.csv')
                if os.path.exists(out):
                    continue
                if dry:
                    wrote += 1; continue
                tmp = out + '.tmp'
                g.to_csv(tmp, index=False)
                os.replace(tmp, out)   # atomic
                wrote += 1
            n_batches += 1
            n_cells += wrote
            if delete and not dry:
                os.remove(path)
        if n_batches or n_empty:
            rel = os.path.relpath(root, base)
            empty_note = f'  ({n_empty} empty/failed batches skipped)' if n_empty else ''
            print(f'{rel}: {n_batches} batch files -> {n_cells} new per-cell files{empty_note}'
                  + ('  [dry-run]' if dry else '')
                  + ('  [deleted batch files]' if (delete and not dry) else '')
                  + ('  [removed empties]' if (clean_empty and not dry and n_empty) else ''))
            total_batches += n_batches; total_cells += n_cells; total_empty += n_empty

    print(f'\nTOTAL: split {total_batches} batch files into {total_cells} per-cell files'
          + (f'  |  {total_empty} empty/failed batches skipped' if total_empty else '')
          + ('  (dry-run, nothing written)' if dry else ''))


if __name__ == '__main__':
    main()
