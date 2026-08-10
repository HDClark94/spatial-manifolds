"""
Identify anchoring-switching sessions with the task_anchoring_catalogue criteria.

For every session, compute per-cell anchoring labels (all regions, as the catalogue
does), run PCA on the cell x trial anchoring matrix, and flag the session as a
switcher when  agree_frac > 0.3  and  pc1_enrichment > 1.5  and  pc1_dominance > 2.

Writes a CSV of all sessions + stats + the pass flag.

Usage:
  python identify_switching_sessions.py [--out switching_sessions.csv]
"""
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import anchoring_common as ac
from anchoring_common import compute_vr_tcs, compute_anchoring_labels, session_switching_stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str,
                   default='/Users/harryclark/Documents/spatial-manifolds/data/xgboost_shank_anchoring/switching_sessions.csv')
    p.add_argument('--source_path', type=str, default=ac.SOURCE_PATH)
    p.add_argument('--min_cells', type=int, default=4)
    args = p.parse_args()

    df_all = pd.read_csv(ac.CELLS_CSV)
    for c in ('mouse', 'day', 'cluster_id'):
        df_all[c] = df_all[c].astype(int)
    sessions = sorted(set(map(tuple, df_all[['mouse', 'day']].values.tolist())))
    print(f'{len(sessions)} sessions')

    rows = []
    for mouse, day in sessions:
        try:
            tcs, *_ = compute_vr_tcs(mouse, day, source_path=args.source_path)
        except Exception as e:
            print(f'M{mouse}D{day}: load error ({e})'); continue

        sess = df_all[(df_all['mouse'] == mouse) & (df_all['day'] == day)]
        labels = []
        for _, r in sess.iterrows():
            cid = int(r['cluster_id'])
            lab = compute_anchoring_labels(tcs.get(cid)) if cid in tcs else None
            if lab is not None:
                labels.append(lab)
        if len(labels) < args.min_cells:
            print(f'M{mouse}D{day}: too few cells ({len(labels)})'); continue

        st = session_switching_stats(labels)
        rows.append(dict(mouse=mouse, day=day, n_cells=st['n_cells'],
                         agree_frac=round(st['agree_frac'], 4),
                         var_expl=round(st['var_expl'], 4),
                         pc1_enrichment=round(st['pc1_enrichment'], 4),
                         pc1_dominance=round(st['pc1_dominance'], 4),
                         session_passes=st['session_passes']))
        flag = 'PASS' if st['session_passes'] else 'skip'
        print(f'M{mouse}D{day:02d} {flag}  n={st["n_cells"]}  '
              f'PC1={st["var_expl"]:.1%} ({st["pc1_enrichment"]:.1f}x)  '
              f'dom={st["pc1_dominance"]:.1f}  agree={st["agree_frac"]:.0%}')

    out = pd.DataFrame(rows).sort_values(['session_passes', 'agree_frac'], ascending=False)
    out.to_csv(args.out, index=False)
    n_pass = int(out['session_passes'].sum()) if len(out) else 0
    print(f'\nWrote {args.out}  ({n_pass} switching sessions)')
    if n_pass:
        print(out[out['session_passes']][['mouse', 'day', 'n_cells', 'agree_frac',
                                          'pc1_enrichment', 'pc1_dominance']].to_string(index=False))


if __name__ == '__main__':
    main()
