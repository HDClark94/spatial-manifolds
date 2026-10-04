"""Speed's UNIQUE contribution to firing, by anchoring state.

Usage:  python3 hinman_dpr2.py [worker n_workers]

hinman_speed.py asks how well speed alone predicts firing, and finds it predicts
much better when the population is anchored -- Hinman et al.'s signature. But
speed and position are not independent on a linear track, and they are least
independent exactly when the animal runs the track most stereotypically, which
is what an anchored state tends to look like. A cell tuned only to POSITION will
then appear speed-tuned, more so when anchored, with no change in speed coding
at all.

So this factors position out, with the delta-pR2 design used elsewhere in the
project:

    base  =  position
    full  =  position + speed
    dpR2  =  pR2(full) - pR2(base)

dpR2 is what speed adds ON TOP OF position, so a position-driven artefact cannot
produce it.

READ THE DIRECTION CAREFULLY. This control is CONSERVATIVE, not neutral. The
tighter the speed-position coupling in a state, the more of the genuine speed
signal the position model absorbs, and the smaller dpR2 becomes. Since the
coupling is expected to be tighter when anchored, the anchored state is
penalised more heavily. So:

  dpR2 still HIGHER when anchored  ->  strong support; the effect beat a test
                                       that was working against it
  dpR2 equal or lower              ->  ambiguous, NOT a refutation: absorption
                                       and a genuine absence look identical here

`speed_from_position` is recorded per state for exactly this reason -- it says
how much absorption to expect, so the ambiguous case can at least be bounded.

POSITION IS A BASIS, NOT A LINE. Grid and place fields are not monotonic in
position, so position enters as N_POS circular Gaussian bumps tiling the track;
a linear or sin/cos term would leave most position tuning in the residual and
hand it to speed. Speed enters linearly, matching Hinman's model.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/'
                   'AnchorDynamics2026')
import hinman_speed as HS
import pynapple as nap
from sklearn.linear_model import PoissonRegressor
from spatial_manifolds.anchoring import load_session_labels
from spatial_manifolds.mlencoding import poisson_pseudoR2

OUT = '/Users/harryclark/Documents/spatial-manifolds/data/lfp'
N_POS, POS_SD, TL = 14, 16.0, 200.0   # coarser: 3 models x 2 states per cell
ALPHA = 1e-3


def pos_basis(pos):
    """N_POS circular Gaussian bumps tiling the track."""
    c = np.linspace(0, TL, N_POS, endpoint=False)
    d = np.abs(pos[:, None] - c[None, :])
    d = np.minimum(d, TL - d)                 # the track is a ring
    return np.exp(-.5 * (d / POS_SD) ** 2)


def cv_pr2_X(X, y, folds=3):
    """Cross-validated Poisson pR2 for an arbitrary design matrix."""
    n = len(y)
    if n < 4 * HS.MINBIN or y.sum() < 20:
        return np.nan
    idx = np.arange(n)
    out = []
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        if len(te) < 20 or y[tr].sum() < 10:
            continue
        try:
            m = PoissonRegressor(alpha=ALPHA, max_iter=200).fit(X[tr], y[tr])
            out.append(float(poisson_pseudoR2(y[te], m.predict(X[te]),
                                              np.mean(y[te]))))
        except Exception:
            continue
    return float(np.nanmean(out)) if out else np.nan


def binned_with_position(mo, dy, state):
    """As speed_tuning.binned, but keeping track position per bin."""
    bp, cp = HS.vr_paths(mo, dy)
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials, orig = HS.clip_trials(beh['trials'].as_dataframe(), clusters)
    ids = HS.spatial_cell_ids(mo, dy, clusters, classes=('GC', 'NGS', 'NS'))
    if len(ids) < 5:
        return None
    S, P = beh['S'], beh['P']
    ts, vs = np.asarray(S.index), np.asarray(S.values, float)
    ps = np.asarray(P.values, float)[
        np.searchsorted(np.asarray(P.index), ts).clip(0, len(P) - 1)]
    edges = np.arange(ts[0], ts[-1], HS.BIN)
    if len(edges) < 200:
        return None
    ctr = edges[:-1] + HS.BIN / 2
    j = np.searchsorted(edges, ts) - 1
    ok = (j >= 0) & (j < len(ctr)) & np.isfinite(vs) & np.isfinite(ps)
    cnt = np.bincount(j[ok], minlength=len(ctr)).astype(float)
    spd = np.where(cnt > 0, np.bincount(j[ok], weights=vs[ok],
                                        minlength=len(ctr)) / np.maximum(cnt, 1), np.nan)
    pos = np.where(cnt > 0, np.bincount(j[ok], weights=ps[ok],
                                        minlength=len(ctr)) / np.maximum(cnt, 1), np.nan)
    starts = trials.start.values.astype(float)
    ends = trials.end.values.astype(float)
    nums = trials.number.values.astype(int)
    k = np.searchsorted(starts, ctr) - 1
    inside = (k >= 0) & (k < len(starts))
    k = k.clip(0, len(starts) - 1)
    inside &= ctr <= ends[k]
    tno = np.where(inside, nums[k], -1)
    lab = np.array([state.get(int(u), np.nan) for u in tno], dtype=float)
    keep = (np.isfinite(spd) & np.isfinite(pos) & (spd >= HS.RUN)
            & np.isfinite(lab))
    if keep.sum() < 2 * HS.MINBIN:
        return None
    R = np.empty((keep.sum(), len(ids)))
    for i, c in enumerate(ids):
        sp = np.asarray(clusters[c].index, dtype=float)
        R[:, i] = np.histogram(sp, bins=edges)[0][keep]
    return R, spd[keep], pos[keep], lab[keep] > .5, ids


def speed_from_position(pos, speed):
    """How much of speed the position basis can explain -- the absorption risk."""
    X = pos_basis(pos)
    n = len(speed)
    if n < 4 * HS.MINBIN:
        return np.nan
    from sklearn.linear_model import Ridge
    idx = np.arange(n); out = []
    for f in range(HS.NFOLD):
        te = idx[f::HS.NFOLD]; tr = np.setdiff1d(idx, te)
        m = Ridge(alpha=1.0).fit(X[tr], speed[tr])
        p = m.predict(X[te])
        ss = np.sum((speed[te] - p) ** 2) / np.sum((speed[te] - speed[te].mean()) ** 2)
        out.append(1 - ss)
    return float(np.mean(out))


def run_session(mo, dy, rng):
    z = load_session_labels(mo, dy)
    if z is None:
        return None
    state = {int(t): bool(f > .5) for t, f in zip(z['trial'], z['frac_anch'])}
    got = binned_with_position(mo, dy, state)
    if got is None:
        return None
    R, speed, pos, anch, ids = got
    if anch.sum() < HS.MINBIN or (~anch).sum() < HS.MINBIN:
        return None
    ka, kn = HS.matched_bins(speed, anch, rng)
    if len(ka) < HS.MINBIN or len(kn) < HS.MINBIN:
        return None
    sfp = {t: speed_from_position(pos[i], speed[i])
           for t, i in (('a', ka), ('n', kn))}
    rows = []
    for i, c in enumerate(ids):
        y = R[:, i]
        if y.sum() / (len(y) * HS.BIN) < HS.MIN_RATE:
            continue
        rec = dict(mouse=mo, day=dy, cluster_id=int(c),
                   sfp_a=sfp['a'], sfp_n=sfp['n'])
        for tag, idx in (('a', ka), ('n', kn)):
            B = pos_basis(pos[idx])
            sp_ = speed[idx].reshape(-1, 1)
            base = cv_pr2_X(B, y[idx])
            full = cv_pr2_X(np.column_stack([B, sp_]), y[idx])
            rec[f'base_{tag}'] = base
            rec[f'full_{tag}'] = full
            rec[f'dpr2_{tag}'] = full - base
            rec[f'rate_{tag}'] = float(y[idx].mean() / HS.BIN)
        rows.append(rec)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    w, nw = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    d = '/Users/harryclark/Documents/spatial-manifolds/data/population_state/labels'
    sess = sorted({(int(f.split('M')[1].split('D')[0]),
                    int(f.split('D')[1].split('.')[0]))
                   for f in os.listdir(d) if f.endswith('.npz')})
    mine = [s for i, s in enumerate(sess) if i % nw == w]
    path = f'{OUT}/hinman_dpr2_w{w}.csv'
    out, t0 = [], time.time()
    if os.path.exists(path):
        out.append(pd.read_csv(path))
    done = set(map(tuple, out[0][['mouse', 'day']].drop_duplicates().values)) \
        if out else set()
    print(f'worker {w}/{nw}: {len(mine)} sessions, {len(done)} done', flush=True)
    for mo, dy in mine:
        if (mo, dy) in done:
            continue
        try:
            t = run_session(mo, dy, np.random.default_rng(abs(hash((mo, dy))) % 2**32))
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        if t is None or not len(t):
            continue
        out.append(t)
        pd.concat(out, ignore_index=True).to_csv(path, index=False)
        print(f'  M{mo}D{dy}: {len(t)} cells, {time.time()-t0:.0f}s', flush=True)
    print(f'worker {w} done in {time.time()-t0:.0f}s', flush=True)
