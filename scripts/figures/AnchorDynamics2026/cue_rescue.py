"""If feedforward drive is what unpins the grid code, a usable landmark should
rescue it. This tests that, and it is the prediction that separates the two-loop
account from every rate- or arousal-based alternative.

The account says the non-anchored state is one of HIGH feedforward gain, and
that high gain hurts here only because the visual scene is aliased: a repeating
corridor supplies a position estimate that is multimodal, so pulling the grid
phase toward it scatters the phase instead of fixing it. The corollary is sharp
and counter-intuitive. On BEACONED trials the reward zone carries a visual
marker, so the sensory estimate is no longer ambiguous, and the same high gain
should now be CORRECTIVE:

    non-anchored trials should hold their spatial correspondence better when
    the trial is cued than when it is not

An account in which the non-anchored state is loss of spatial tuning, or a rate
change, or generic disengagement, predicts no such interaction: those make the
code worse regardless of what is on the screen.

The control is the anchored state, where gain is already low and the cue should
therefore buy much less.

Measure: r0, each trial's smoothed rate map against the cell's own anchored
template at zero offset -- the same quantity the classifier's decision rests on.
Compared within session and within cell, so neither session quality nor cell
identity can carry the effect.

Writes data/population_state/cue_rescue.csv
"""
import json
import os
import sys
import time
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
OUT = f'{PS}/cue_rescue.csv'
SIGMA = 2.0
MIN_TEMPLATE = 8
MIN_PER_CELL = 4          # trials of a given state x cue before a cell counts

_g = {}
for _c in json.load(open(f'{FIG}/lick_raster_by_trial_type.ipynb'))['cells']:
    if _c['cell_type'] != 'code':
        continue
    _s = ''.join(_c['source'])
    if _s.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in _s:
        continue
    exec(compile(_s, '<nb>', 'exec'), _g)
globals().update({k: v for k, v in _g.items() if not k.startswith('__')})

CLS = pd.read_csv(f'{ROOT}/data/cell_classifications_v2.csv')
GRID = set(map(tuple, CLS.loc[CLS.cell_class_of1 == 'GC',
                              ['mouse', 'day', 'cluster_id']].values))
TT = pd.read_csv(f'{PS}/trial_table.csv')


def corr0(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 6:
        return np.nan
    x, y = x[m] - x[m].mean(), y[m] - y[m].mean()
    d = np.sqrt((x ** 2).sum() * (y ** 2).sum())
    return np.nan if d == 0 else float((x * y).sum() / d)


def session(mo, dy):
    z = load_session_labels(mo, dy)
    if z is None:
        return None
    bp, cp = vr_paths(mo, dy)
    if not (os.path.exists(bp) and os.path.exists(cp)):
        return None
    tt = TT[(TT.mouse == mo) & (TT.day == dy)]
    if not len(tt):
        return None
    cue = dict(zip(tt.trial.astype(int), tt.ttype))
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials_all = beh['trials'].as_dataframe()
    trials, orig = clip_trials(trials_all, clusters)
    keep = np.isin(trials_all.number.values.astype(int), orig)
    ids = [int(c) for c in z['cluster_id']]
    have = [c for c in ids
            if c in set(int(x) for x in clusters.index) and (mo, dy, c) in GRID]
    if not have:
        return None
    trial_no = np.asarray(z['trial']).astype(int)
    tn, trav = beh['trial_number'], beh['travel']
    dt = trav - (float(np.asarray(tn.values)[0]) - 1) * TL
    moving = beh['S'].threshold(3.0, method='above').time_support
    n_all = len(trials_all)
    tc = nap.compute_1d_tuning_curves(clusters[have], dt, nb_bins=n_all * NBIN,
                                      minmax=[0, n_all * TL], ep=moving)
    ct = np.array([cue.get(int(t), '?') for t in trial_no])
    rows = []
    for c in have:
        i = ids.index(c)
        lab = np.asarray(z['labels'][i])
        M = np.asarray(tc[c]).reshape(n_all, NBIN)[keep]
        S = np.array([smooth_nanaware(r, sigma=SIGMA) for r in M])
        anch = lab == 1
        if anch.sum() < MIN_TEMPLATE:
            continue
        tmpl = np.nanmean(S[anch], axis=0)
        r0 = np.array([corr0(S[t], tmpl) for t in range(len(S))])
        for state, lm in (('anch', lab == 1), ('non', lab == 0)):
            for cu in ('b', 'nb'):
                m = lm & (ct == cu) & np.isfinite(r0)
                if m.sum() >= MIN_PER_CELL:
                    rows.append(dict(mouse=mo, day=dy, cluster_id=c, state=state,
                                     cue=cu, n=int(m.sum()),
                                     r0=float(np.mean(r0[m]))))
    return pd.DataFrame(rows) if rows else None


if __name__ == '__main__':
    sess = sorted({(int(f.split('M')[1].split('D')[0]),
                    int(f.split('D')[1].split('.')[0]))
                   for f in os.listdir(f'{PS}/labels') if f.endswith('.npz')})
    out, t0 = [], time.time()
    for mo, dy in sess:
        try:
            r = session(mo, dy)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        if r is None:
            continue
        out.append(r)
        print(f'  M{mo}D{dy}: {r.cluster_id.nunique()} grid cells '
              f'({time.time()-t0:.0f}s)', flush=True)
    d = pd.concat(out, ignore_index=True)
    d.to_csv(OUT, index=False)
    print(f'\nwrote {OUT}: {d.cluster_id.nunique()} cells, '
          f'{d.groupby(["mouse","day"]).ngroups} sessions')
