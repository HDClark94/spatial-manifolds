"""Does the grid population stay on its attractor manifold during NON-ANCHORED
trials, or does the code simply fall apart?

This is the test the whole two-loop account turns on. Trial maps already show
that on non-anchored trials there is no coherent field at any offset, no
alignment between neighbouring trials, and no gradient along the track
(drift_vs_remap.py). Two very different things produce that and trial maps
cannot separate them:

    WANDERING BUMP   the attractor is intact and still holds a localised
                     population state, but its phase moves within a traversal,
                     so averaging over the trial smears the map. Position
                     information is lost; attractor structure is not.

    CODE COLLAPSE    the population no longer holds a bump at all. There is no
                     attractor state to be decoupled from position.

The first keeps a continuous-attractor account alive. The second kills it.

HOW THE TWO ARE SEPARATED. Population vectors are binned in TIME, not position,
so a moving bump is not averaged away. A dictionary of population states is
built from ANCHORED trials indexed by track position -- this is the repertoire
of states the attractor is known to support. Every held-out time bin is then
matched against that dictionary:

    q        how well the best-matching template fits -- is the population in
             SOME attractor state, whether or not it is the right one
    error    distance from the matched position to the true position -- is that
             state the correct one
    step     how far the matched position moves between consecutive bins -- does
             it evolve continuously, as a bump must, or jump at random

A wandering bump gives HIGH q, HIGH error, SMALL steps. A collapse gives LOW q,
high error, and steps at chance. These are opposite predictions on q and step,
which is what makes the test decisive rather than suggestive.

CONTROLS. Templates come from one half of the anchored trials and are never
evaluated on the trials that built them. The null shuffles cell identity within
each time bin, destroying population structure while preserving every cell's own
firing statistics, so q is judged against what this many cells at these rates
would give by chance.

Writes data/population_state/torus_persistence.csv
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
from spatial_manifolds.anchoring import load_session_labels

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = f'{ROOT}/scripts/figures/AnchorDynamics2026'
PS = f'{ROOT}/data/population_state'
OUT = f'{PS}/torus_persistence.csv'

BIN = 0.25          # s per population vector; ~9 cm at the animals' speeds
NPOS = 40           # template positions along the track
MIN_GC = 12         # grid cells before a population vector means anything
MIN_TRIALS = 8      # trials per state
SMOOTH = 1.0        # bins, applied along time to each cell

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


def circ_err(a, b, n):
    return np.abs((a - b + n // 2) % n - n // 2)


def session(mo, dy, rng):
    z = load_session_labels(mo, dy)
    if z is None:
        return None
    bp, cp = vr_paths(mo, dy)
    if not (os.path.exists(bp) and os.path.exists(cp)):
        return None
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    ids = [int(c) for c in z['cluster_id']]
    have = [c for c in ids
            if c in set(int(x) for x in clusters.index) and (mo, dy, c) in GRID]
    if len(have) < MIN_GC:
        return None
    lab = np.asarray(z['labels'])[[ids.index(c) for c in have]]   # grid x trial
    trial_no = np.asarray(z['trial']).astype(int)

    moving = beh['S'].threshold(3.0, method='above').time_support
    cnt = clusters[have].count(BIN, ep=moving)
    t = cnt.index.values
    X = np.asarray(cnt)                                    # time x grid cells
    if len(t) < 200:
        return None
    if SMOOTH > 0:
        from scipy.ndimage import gaussian_filter1d
        X = gaussian_filter1d(X.astype(float), SMOOTH, axis=0)

    # position within trial, and trial number, at each bin centre
    tn_t = np.asarray(beh['trial_number'].index.values)
    tn_v = np.asarray(beh['trial_number'].values).astype(float)
    tr_b = np.round(np.interp(t, tn_t, tn_v)).astype(int)
    tv_t = np.asarray(beh['travel'].index.values)
    tv_v = np.asarray(beh['travel'].values).astype(float)
    trav = np.interp(t, tv_t, tv_v)
    pos = np.mod(trav, TL)
    pb = np.clip((pos / (TL / NPOS)).astype(int), 0, NPOS - 1)

    # population state per trial: is most of the GRID population anchored?
    frac = np.nanmean(lab == 1, axis=0)
    anch_trials = set(trial_no[frac > .5]); non_trials = set(trial_no[frac <= .5])
    if len(anch_trials) < MIN_TRIALS or len(non_trials) < MIN_TRIALS:
        return None

    # z-score each cell over all bins so templates and test vectors compare
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd

    # templates from HALF the anchored trials; evaluated on the other half
    at = np.array(sorted(anch_trials))
    rng.shuffle(at)
    build, test_a = set(at[: len(at) // 2]), set(at[len(at) // 2:])
    m_build = np.isin(tr_b, list(build))
    T = np.full((NPOS, Z.shape[1]), np.nan)
    for p in range(NPOS):
        m = m_build & (pb == p)
        if m.sum() >= 3:
            T[p] = Z[m].mean(0)
    ok = np.isfinite(T).all(1)
    if ok.sum() < NPOS * .7:
        return None
    Tn = T[ok] - T[ok].mean(1, keepdims=True)
    Tn /= np.linalg.norm(Tn, axis=1, keepdims=True)
    pos_ok = np.where(ok)[0]

    rows = []
    for tag, trials in (('anch', test_a), ('non', non_trials)):
        m = np.isin(tr_b, list(trials))
        if m.sum() < 50:
            continue
        V = Z[m] - Z[m].mean(1, keepdims=True)
        nrm = np.linalg.norm(V, axis=1, keepdims=True); nrm[nrm == 0] = 1
        V = V / nrm
        C = V @ Tn.T                                   # bins x positions
        q = C.max(1)
        dec = pos_ok[C.argmax(1)]
        err = circ_err(dec, pb[m], NPOS)
        # continuity: how far the decoded state moves between adjacent bins
        tb = np.where(m)[0]
        adj = np.diff(tb) == 1
        step = circ_err(dec[1:][adj], dec[:-1][adj], NPOS)
        # null: shuffle cell identity within each bin
        Vs = np.array([rng.permutation(v) for v in V])
        Cs = Vs @ Tn.T
        # does the decoded state still advance WITH the animal? Signed decoded
        # displacement against the animal's own displacement over the same bin.
        # Path integration intact -> these track each other even when the phase
        # itself is wrong; a bump driven by something other than velocity -> not.
        d_dec = ((dec[1:][adj] - dec[:-1][adj] + NPOS // 2) % NPOS
                 - NPOS // 2) * (TL / NPOS)
        tr_all = trav[m]
        d_run = np.diff(tr_all)[adj]
        good = np.isfinite(d_dec) & np.isfinite(d_run) & (np.abs(d_run) < TL / 2)
        vcorr = (np.corrcoef(d_dec[good], d_run[good])[0, 1]
                 if good.sum() > 30 else np.nan)
        rows.append(dict(mouse=mo, day=dy, state=tag, n_cells=len(have),
                         bin_s=BIN, n_bins=int(m.sum()), q=float(np.mean(q)),
                         q_null=float(np.mean(Cs.max(1))),
                         err=float(np.mean(err)) * (TL / NPOS),
                         step=float(np.mean(step)) * (TL / NPOS),
                         vcorr=float(vcorr),
                         d_dec=float(np.mean(np.abs(d_dec[good]))),
                         d_run=float(np.mean(np.abs(d_run[good]))),
                         step_chance=float(NPOS / 4) * (TL / NPOS)))
    return pd.DataFrame(rows) if rows else None


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
        print(f'  M{mo}D{dy}: {r.n_cells.iloc[0]} grid cells, '
              f'q anch {r[r.state=="anch"].q.mean():.3f} / '
              f'non {r[r.state=="non"].q.mean() if (r.state=="non").any() else np.nan:.3f}'
              f'  ({time.time()-t0:.0f}s)', flush=True)
    if out:
        d = pd.concat(out, ignore_index=True)
        d.to_csv(OUT, index=False)
        print(f'\nwrote {OUT}: {d.groupby(["mouse","day"]).ngroups} sessions')
