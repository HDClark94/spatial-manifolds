"""A per-lap anchoring score that does not use the classifier.

For each cell and lap:
  template  the cell's mean smoothed map over ALL its other laps (leave-one-out).
            No labels enter: anchored laps agree with one another and
            non-anchored laps do not, so the stable map dominates the mean
            without being told which laps carry it.
  r0        correlation of the lap's smoothed map with that template
  null      the lap's map circularly shifted by every offset of at least
            MIN_SHIFT_CM (computed at once by FFT), correlated with the template
  score     g = r0 - mean(null): 0 is no more alignment than chance
  p         fraction of null offsets reaching r0 (one-sided), per lap

Maps are as the classifier's (running only, 2 cm bins, NaN-aware smoothing at
sigma 1.5 bins), so the score and the labels see the same data; only the
template and the decision differ. Spike count per lap (running only) is kept,
because a sparse lap can fail to reach significance for want of spikes.

The classifier's labels and the population state are attached afterwards, for
grouping only.

Writes data/population_state/anchoring_score/M{mo}D{dy}.npz
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
OUTD = f'{PS}/anchoring_score'
MIN_SHIFT_CM = 30.0
MIN_LAPS = 20

_g = {}
for _c in json.load(open(f'{FIG}/lick_raster_by_trial_type.ipynb'))['cells']:
    if _c['cell_type'] != 'code':
        continue
    _s = ''.join(_c['source'])
    if _s.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in _s:
        continue
    exec(compile(_s, '<nb>', 'exec'), _g)
vr_paths, clip_trials, TL, NBIN = _g['vr_paths'], _g['clip_trials'], _g['TL'], _g['NBIN']
REG = pd.read_csv(f'{PS}/pc1_by_region.csv')


def region(a):
    a = str(a)
    return 'MEC' if a.startswith('ENTm') else 'VIS' if a.startswith('VIS') else 'other'


def score_cell(S):
    """g, r0 and p for every lap of one cell's smoothed maps S [laps, bins]."""
    n, nb = S.shape
    active = S.sum(1) > 0
    tot = S[active].sum(0)
    smin = int(round(MIN_SHIFT_CM / (TL / nb)))
    shifts = np.r_[smin:nb - smin + 1]
    g = np.full(n, np.nan); r0 = np.full(n, np.nan); p = np.full(n, np.nan)
    if active.sum() < MIN_LAPS:
        return g, r0, p
    for t in np.flatnonzero(active):
        tmpl = (tot - S[t]) / (active.sum() - 1)
        a = S[t] - S[t].mean(); b = tmpl - tmpl.mean()
        d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        if d == 0:
            continue
        # circular cross-correlation at every shift: xc[k] = sum a[i] b[i+k]
        xc = np.real(np.fft.ifft(np.conj(np.fft.fft(a)) * np.fft.fft(b))) / d
        r0[t] = xc[0]
        nul = xc[shifts]
        g[t] = xc[0] - nul.mean()
        p[t] = (np.sum(nul >= xc[0]) + 1) / (len(nul) + 1)
    return g, r0, p


def session(mo, dy):
    z = load_session_labels(mo, dy)
    if z is None:
        return None
    bp, cp = vr_paths(mo, dy)
    if not (os.path.exists(bp) and os.path.exists(cp)):
        return None
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials_all = beh['trials'].as_dataframe()
    trials, orig = clip_trials(trials_all, clusters)
    keep = np.isin(trials_all.number.values.astype(int), orig)
    tn, trav = beh['trial_number'], beh['travel']
    dtrav = trav - (float(np.asarray(tn.values)[0]) - 1) * TL
    moving = beh['S'].threshold(3.0, method='above').time_support
    n_all = len(trials_all)
    rr = REG[(REG.mouse == mo) & (REG.day == dy)].set_index('cluster_id')
    ids = [int(c) for c in z['cluster_id']]
    use = [i for i, c in enumerate(ids) if c in clusters.index
           and region(rr.brain_region.get(c, '')) in ('MEC', 'VIS')]
    if not use:
        return None
    cl = [ids[i] for i in use]
    tc = nap.compute_1d_tuning_curves(clusters[cl], dtrav, nb_bins=n_all * NBIN,
                                      minmax=[0, n_all * TL], ep=moving)
    # running spikes per (clipped) lap
    tr = trials.reset_index(drop=True)
    edges = np.c_[tr.start.values, tr.end.values]
    nspk = np.zeros((len(cl), len(tr)), int)
    for j, c in enumerate(cl):
        st = np.asarray(clusters[c].restrict(moving).index)
        nspk[j] = np.searchsorted(st, edges[:, 1]) - np.searchsorted(st, edges[:, 0])
    G = np.full((len(cl), len(tr)), np.nan); R0 = G.copy(); P = G.copy()
    for j, c in enumerate(cl):
        M = np.asarray(tc[c]).reshape(n_all, NBIN)[keep]
        S = np.nan_to_num(np.array([smooth_nanaware(r) for r in M]))
        G[j], R0[j], P[j] = score_cell(S)
    lab = np.asarray(z['labels'], float)[use]
    os.makedirs(OUTD, exist_ok=True)
    np.savez_compressed(f'{OUTD}/M{mo}D{dy}.npz', cluster_id=np.array(cl),
                        region=np.array([region(rr.brain_region.get(c, '')) for c in cl]),
                        g=G.astype(np.float32), r0=R0.astype(np.float32), p=P.astype(np.float32),
                        nspk=nspk, label=lab, pop_anch=np.asarray(z['pc1'], float) > 0,
                        locked=np.nanstd(lab, axis=1) == 0)
    return len(cl)


if __name__ == '__main__':
    w, nw = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    sess = sorted({(int(f.split('M')[1].split('D')[0]), int(f.split('D')[1].split('.')[0]))
                   for f in os.listdir(f'{PS}/labels') if f.endswith('.npz')})
    t0 = time.time()
    for mo, dy in [s for i, s in enumerate(sess) if i % nw == w]:
        if os.path.exists(f'{OUTD}/M{mo}D{dy}.npz'):
            continue
        try:
            n = session(mo, dy)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        if n:
            print(f'  M{mo}D{dy}: {n} cells, {time.time() - t0:.0f}s', flush=True)
