"""Per-cell agreement with the entorhinal axis, scored OUT-OF-SAMPLE for every cell.

Why this exists. `build_per_cell_pc1.py` builds the axis from all MEC cells and
removes only the scored cell itself (leave-one-out). That is the right correction
for the cell's own contribution, but it leaves a confound that is fatal to any
REGIONAL comparison: an MEC cell is still scored against an axis built from its
own correlated network, while a subicular or visual cell was never in the axis at
all. Part of any MEC > VIS gap is therefore built into the construction rather
than into the brain, and the one contrast free of it -- SUB/PARA vs VIS -- rests
on 15 sessions and is underpowered.

The fix is to put every scored cell in the same position. MEC cells are split at
random into two halves; the axis is built by SVD from half A, and half B, SUB,
VIS and CB cells are all scored against it. Nobody scored is in the axis, so the
comparison is between regions rather than between degrees of membership. The
split is repeated N_SPLIT times and the per-cell results averaged, because a
single split is a coin toss over which cells define the axis.

Chance for the signed correlation is ZERO -- see build_per_cell_pc1.py's
docstring for the measurement showing that the circular-shift null is centred at
zero and that the familiar "+0.15 chance level" is just E|x| = 0.798*sd of that
null. `r` is the effect size; `p` is the two-sided per-cell permutation
significance.

Writes data/population_state/pc1_heldout.csv
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import load_session_labels

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
PS = f'{ROOT}/data/population_state'
CLS_CSV = f'{ROOT}/data/cell_classifications_v2.csv'
N_SHIFT = 200
N_SPLIT = 8
MIN_HALF = 8          # cells per half before an axis is trusted


def axis_from(M, frac):
    Mc = np.nan_to_num(M - np.nanmean(M, axis=1, keepdims=True))
    U, sv, Vt = np.linalg.svd(Mc, full_matrices=False)
    pc = Vt[0]
    if np.corrcoef(pc, np.nan_to_num(frac))[0, 1] < 0:
        pc = -pc
    return pc


def session(mo, dy, rng):
    z = load_session_labels(mo, dy)
    if z is None:
        return None
    L, ids = z['labels'], z['cluster_id']
    in_pop = np.isfinite(z['pc1_load'])
    frac = z['frac_anch']
    pop_idx = np.where(in_pop)[0]
    pop_idx = np.array([i for i in pop_idx if not np.isnan(L[i]).all()])
    if len(pop_idx) < 2 * MIN_HALF:
        return None

    # accumulate over splits; a cell is scored only on splits where it is not in
    # the axis, which for an MEC cell is about half of them
    acc_r = {k: [] for k in range(len(ids))}
    acc_p = {k: [] for k in range(len(ids))}
    for _ in range(N_SPLIT):
        perm = rng.permutation(pop_idx)
        half = len(perm) // 2
        A, B = perm[:half], perm[half:]
        if len(A) < MIN_HALF:
            continue
        pc = axis_from(L[A], frac)
        scored = set(B.tolist()) | {k for k in range(len(ids))
                                    if k not in set(pop_idx.tolist())}
        for k in scored:
            v = np.nan_to_num(L[k])
            if np.std(v) == 0:
                continue
            r = float(np.corrcoef(v, pc)[0, 1])
            null = np.array([np.corrcoef(np.roll(v, s), pc)[0, 1]
                             for s in rng.integers(5, len(v) - 5, N_SHIFT)])
            acc_r[k].append(r)
            acc_p[k].append(float((np.abs(null) >= abs(r)).mean()))

    rows = []
    for k, c in enumerate(ids):
        if not acc_r[k]:
            rows.append(dict(mouse=mo, day=dy, cluster_id=int(c), r=np.nan,
                             p=np.nan, n_split=0, is_mec=bool(in_pop[k]),
                             frac_anch=float(np.nanmean(np.nan_to_num(L[k])))))
            continue
        rows.append(dict(mouse=mo, day=dy, cluster_id=int(c),
                         r=float(np.mean(acc_r[k])),
                         p=float(np.mean(acc_p[k])),
                         n_split=len(acc_r[k]), is_mec=bool(in_pop[k]),
                         frac_anch=float(np.nanmean(np.nan_to_num(L[k])))))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    sess = sorted({(int(f.split('M')[1].split('D')[0]),
                    int(f.split('D')[1].split('.')[0]))
                   for f in os.listdir(f'{PS}/labels') if f.endswith('.npz')})
    out, t0 = [], time.time()
    for k, (mo, dy) in enumerate(sess, 1):
        try:
            R = session(mo, dy, np.random.default_rng(abs(hash((mo, dy))) % 2 ** 32))
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True)
            continue
        if R is None or not len(R):
            continue
        out.append(R)
        if k % 10 == 0:
            print(f'[{k}/{len(sess)}] {time.time() - t0:.0f}s', flush=True)
    A = pd.concat(out, ignore_index=True)
    C = pd.read_csv(CLS_CSV)
    keep = [c for c in ('mouse', 'day', 'cluster_id', 'brain_region',
                        'cell_class_of1', 'coord_SCs_x') if c in C.columns]
    A = A.merge(C[keep].drop_duplicates(['mouse', 'day', 'cluster_id']),
                on=['mouse', 'day', 'cluster_id'], how='left')
    A.to_csv(f'{PS}/pc1_heldout.csv', index=False)
    print(f'\n{len(A)} cells from {A.groupby(["mouse", "day"]).ngroups} sessions')
    print(f'  scored on >=1 split: {int(A.n_split.gt(0).sum())}')
