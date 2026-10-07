"""Does the grid network keep its attractor structure on non-anchored trials,
measured WITHOUT reference to track position?

torus_persistence.py matched population vectors against a dictionary built from
anchored trials, and found the fit roughly halved on non-anchored trials. That
result is ambiguous by construction: the dictionary is the ANCHORED repertoire,
so a network that is perfectly intact but occupying different states would score
just as badly as one that had fallen apart. Two of its other measures -- decode
step and decoded-vs-actual velocity -- turned out to be unusable, because grid
cells on a 1D track alias (templates one grid period apart are near-identical),
so the position argmax jumps between aliases: decoded displacement ran at 14 cm
per 100 ms against 3 cm of actual running.

This script asks the question in a way that needs no position and no anchored
dictionary. In a continuous attractor, two cells with similar grid phase are
co-active WHEREVER the bump is, and two cells in antiphase are anti-correlated,
because both facts follow from the connectivity rather than from the location
being represented. So:

    pairwise correlation should depend on pairwise PHASE OFFSET,
    and that dependence should survive wherever the attractor does,
    whether or not the bump is pinned to the track.

Phase offset for each pair is read from the two cells' anchored rate maps (the
circular cross-correlation peak). Correlations are then computed from TIME-binned
spike counts separately within each state. The statistic is the phase-structure
index: mean correlation among near-phase pairs minus mean among antiphase pairs.

    attractor intact, merely unpinned  ->  index preserved in both states
    attractor degraded                 ->  index falls with the state

The null shuffles the phase-offset assignment across pairs, which destroys the
relationship while keeping every pair's own correlation, so the index is judged
against what this set of correlations would give by chance.

Writes data/population_state/attractor_phase_structure.csv
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
from scipy.ndimage import gaussian_filter1d
from spatial_manifolds.anchoring import load_session_labels, smooth_nanaware

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = f'{ROOT}/scripts/figures/AnchorDynamics2026'
PS = f'{ROOT}/data/population_state'
OUT = f'{PS}/attractor_phase_structure.csv'

BIN = 0.25
MIN_GC = 12
MIN_TRIALS = 8
MIN_BINS = 120
SIGMA = 2.0
N_SHUF = 200

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


def phase_offset(a, b, n):
    """Circular cross-correlation peak between two rate maps, in bins."""
    a = np.nan_to_num(a - np.nanmean(a)); b = np.nan_to_num(b - np.nanmean(b))
    d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    if d == 0:
        return np.nan
    cc = np.array([np.sum(np.roll(a, k) * b) for k in range(n)]) / d
    k = int(np.argmax(cc))
    return k if k <= n // 2 else k - n


def session(mo, dy, rng):
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
    ids = [int(c) for c in z['cluster_id']]
    have = [c for c in ids
            if c in set(int(x) for x in clusters.index) and (mo, dy, c) in GRID]
    if len(have) < MIN_GC:
        return None
    lab = np.asarray(z['labels'])[[ids.index(c) for c in have]]
    trial_no = np.asarray(z['trial']).astype(int)
    n_all = len(trials_all)

    tn, trav = beh['trial_number'], beh['travel']
    dt = trav - (float(np.asarray(tn.values)[0]) - 1) * TL
    moving = beh['S'].threshold(3.0, method='above').time_support

    # anchored rate maps -> pairwise phase offsets
    tc = nap.compute_1d_tuning_curves(clusters[have], dt, nb_bins=n_all * NBIN,
                                      minmax=[0, n_all * TL], ep=moving)
    frac = np.nanmean(lab == 1, axis=0)
    anch_tr = set(trial_no[frac > .5]); non_tr = set(trial_no[frac <= .5])
    if len(anch_tr) < MIN_TRIALS or len(non_tr) < MIN_TRIALS:
        return None
    a_mask = np.isin(trial_no, list(anch_tr))
    tmpl = []
    for i, c in enumerate(have):
        M = np.asarray(tc[c]).reshape(n_all, NBIN)[keep]
        tmpl.append(smooth_nanaware(np.nanmean(M[a_mask], axis=0), sigma=SIGMA))
    tmpl = np.array(tmpl)

    # time-binned counts
    cnt = clusters[have].count(BIN, ep=moving)
    t = cnt.index.values
    X = gaussian_filter1d(np.asarray(cnt).astype(float), 1.0, axis=0)
    tn_t = np.asarray(beh['trial_number'].index.values)
    tn_v = np.asarray(beh['trial_number'].values).astype(float)
    tr_b = np.round(np.interp(t, tn_t, tn_v)).astype(int)

    ncell = len(have)
    iu, ju = np.triu_indices(ncell, 1)
    off = np.array([phase_offset(tmpl[i], tmpl[j], NBIN) for i, j in zip(iu, ju)])
    ok = np.isfinite(off)
    if ok.sum() < 30:
        return None
    # near-phase vs antiphase, in bins of the 200 cm track
    aoff = np.abs(off)
    near = ok & (aoff <= NBIN * .10)
    far = ok & (aoff >= NBIN * .35)
    if near.sum() < 8 or far.sum() < 8:
        return None

    rows = []
    for tag, trs in (('anch', anch_tr), ('non', non_tr)):
        m = np.isin(tr_b, list(trs))
        if m.sum() < MIN_BINS:
            continue
        V = X[m]
        V = V - V.mean(0)
        sd = V.std(0); sd[sd == 0] = 1
        V = V / sd
        C = (V.T @ V) / len(V)
        r = C[iu, ju]
        idx = float(np.mean(r[near]) - np.mean(r[far]))
        nul = np.empty(N_SHUF)
        for s in range(N_SHUF):
            p = rng.permutation(r)
            nul[s] = np.mean(p[near]) - np.mean(p[far])
        rows.append(dict(mouse=mo, day=dy, state=tag, n_cells=ncell,
                         n_bins=int(m.sum()), n_near=int(near.sum()),
                         n_far=int(far.sum()), r_near=float(np.mean(r[near])),
                         r_far=float(np.mean(r[far])), index=idx,
                         index_null=float(nul.mean()),
                         z=float((idx - nul.mean()) / (nul.std() + 1e-12))))
    return pd.DataFrame(rows) if len(rows) == 2 else None


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    sess = sorted({(int(f.split('M')[1].split('D')[0]),
                    int(f.split('D')[1].split('.')[0]))
                   for f in os.listdir(f'{PS}/labels') if f.endswith('.npz')})
    out, t0 = [], time.time()
    for mo, dy in sess:
        try:
            r = session(mo, dy, rng)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        if r is None:
            continue
        out.append(r)
        a = r[r.state == 'anch']; n = r[r.state == 'non']
        print(f'  M{mo}D{dy}: {r.n_cells.iloc[0]} cells  index '
              f'anch {a["index"].iloc[0]:+.4f} (z={a.z.iloc[0]:+.1f}) / '
              f'non {n["index"].iloc[0]:+.4f} (z={n.z.iloc[0]:+.1f})  '
              f'({time.time()-t0:.0f}s)', flush=True)
    if out:
        d = pd.concat(out, ignore_index=True)
        d.to_csv(OUT, index=False)
        print(f'\nwrote {OUT}: {d.groupby(["mouse","day"]).ngroups} sessions')
