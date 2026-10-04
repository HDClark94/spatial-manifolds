"""Chance level for the best-offset correlation used in supplement S1.

Usage:  python3 trial_map_null.py [n_sessions]

S1 asks whether a non-anchored trial has a DISPLACED field or NO field, by
correlating each trial's map against the cell's anchored template at every
circular offset and keeping the best. That statistic is a maximum over 100
offsets, so it is high even for unrelated maps, and the raw value cannot
distinguish "coherent field somewhere else" from "noise that happens to align".

Two nulls, both keeping everything about the trial except the thing in question:

  BIN SHUFFLE   position bins of the trial map are permuted, destroying spatial
                structure while preserving the cell's rate distribution exactly.
                This is the floor for "no field at all".
  OTHER CELL    the trial map is matched against a DIFFERENT cell's template
                from the same session. This keeps real spatial structure on both
                sides and asks how well two unrelated real maps align -- the
                stricter and more realistic floor.

Run on a subset of sessions: this is a calibration constant, not a per-cell
quantity, so it does not need every session.

Writes data/population_state/trial_map_null.csv.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
import pynapple as nap
from spatial_manifolds.anchoring import load_session_labels, smooth_nanaware

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = f'{ROOT}/scripts/figures/AnchorDynamics2026'
PS = f'{ROOT}/data/population_state'
OUT = f'{PS}/trial_map_null.csv'
SIGMA, MIN_TEMPLATE, MIN_STATE = 2.0, 8, 5

_g = {}
for _c in json.load(open(f'{FIG}/lick_raster_by_trial_type.ipynb'))['cells']:
    if _c['cell_type'] != 'code':
        continue
    _s = ''.join(_c['source'])
    if _s.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in _s:
        continue
    exec(compile(_s, '<nb>', 'exec'), _g)
globals().update({k: v for k, v in _g.items() if not k.startswith('__')})


def best_r(a, b):
    a = a - np.nanmean(a); b = b - np.nanmean(b)
    d = np.sqrt(np.nansum(a ** 2) * np.nansum(b ** 2))
    if not np.isfinite(d) or d == 0:
        return np.nan
    cc = np.array([np.nansum(np.roll(a, k) * b) for k in range(len(a))]) / d
    return float(np.max(cc))


def session(mo, dy, rng):
    z = load_session_labels(mo, dy)
    if z is None:
        return None
    bp, cp = vr_paths(mo, dy)
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    ta = beh['trials'].as_dataframe()
    trials, orig = clip_trials(ta, clusters)
    keep = np.isin(ta.number.values.astype(int), orig)
    tn, trav = beh['trial_number'], beh['travel']
    dt = trav - (float(np.asarray(tn.values)[0]) - 1) * TL
    moving = beh['S'].threshold(3.0, method='above').time_support
    ids = [int(c) for c in z['cluster_id']
           if int(c) in set(int(x) for x in clusters.index)]
    if len(ids) < 4:
        return None
    tc = nap.compute_1d_tuning_curves(clusters[ids], dt, nb_bins=len(ta) * NBIN,
                                      minmax=[0, len(ta) * TL], ep=moving)
    S = {c: np.array([smooth_nanaware(r, sigma=SIGMA)
                      for r in np.asarray(tc[c]).reshape(len(ta), NBIN)[keep]])
         for c in ids}
    tmpl = {}
    for i, c in enumerate(ids):
        lab = z['labels'][list(z['cluster_id'].astype(int)).index(c)]
        if (lab == 1).sum() >= MIN_TEMPLATE:
            tmpl[c] = np.nanmean(S[c][lab == 1], axis=0)
    rows = []
    for c in ids:
        if c not in tmpl:
            continue
        lab = z['labels'][list(z['cluster_id'].astype(int)).index(c)]
        non = np.where(lab == 0)[0]
        if len(non) < MIN_STATE:
            continue
        others = [o for o in tmpl if o != c]
        shuf, oth = [], []
        for t in non[:25]:                      # 25 trials is plenty per cell
            m = S[c][t]
            shuf.append(best_r(rng.permutation(m), tmpl[c]))
            if others:
                oth.append(best_r(m, tmpl[rng.choice(others)]))
        rows.append(dict(mouse=mo, day=dy, cluster_id=c,
                         null_shuffle=float(np.nanmean(shuf)) if shuf else np.nan,
                         null_othercell=float(np.nanmean(oth)) if oth else np.nan))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    sess = sorted({(int(f.split('M')[1].split('D')[0]),
                    int(f.split('D')[1].split('.')[0]))
                   for f in os.listdir(f'{PS}/labels') if f.endswith('.npz')})
    rng0 = np.random.default_rng(0)
    pick = [sess[i] for i in rng0.choice(len(sess), min(n, len(sess)), replace=False)]
    out = []
    for mo, dy in pick:
        try:
            t = session(mo, dy, np.random.default_rng(abs(hash((mo, dy))) % 2**32))
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        if t is not None and len(t):
            out.append(t); print(f'  M{mo}D{dy}: {len(t)} cells', flush=True)
    A = pd.concat(out, ignore_index=True)
    A.to_csv(OUT, index=False)
    print(f'\n{len(A)} cells over {A.groupby(["mouse","day"]).ngroups} sessions')
    print(f'  null, bin-shuffled trial map : {A.null_shuffle.mean():.3f}')
    print(f'  null, other cell template    : {A.null_othercell.mean():.3f}')
