"""Speed coding beyond position, by anchoring state, via delta pR2.

Raw speed tuning is preserved when the population is anchored and population
speed decoding is slightly BETTER, which contradicts the velocity-gain account.
But both measures can be satisfied without any speed coding at all: anchored
trials have more reliable position coding and a stereotyped speed profile along
the track, so position can stand in for speed. The question that matters is
what speed explains BEYOND position, which is what the delta pR2 architecture
used for the cross-region models answers.

Per cell and per state, three Poisson xgboost models on 250 ms bins:

    pR2(position)        position alone
    pR2(speed)           speed alone
    pR2(position+speed)  both

Position enters as sin and cos of 2*pi*p/L rather than as centimetres. The
track is a ring -- 0 and 200 cm are the same place -- and a tree split cannot
wrap, so a linear position column represents a field spanning the reward zone
as two unrelated regions at opposite ends of the feature. The two-column
encoding costs a little resolution mid-track (isolating a field needs splits on
both columns) but represents the topology correctly.

    dpR2(speed | position)  = pR2(pos+speed) - pR2(pos)    <- the test
    dpR2(position | speed)  = pR2(pos+speed) - pR2(speed)  <- positive control

The positive control matters: anchoring is DEFINED by position coding becoming
reliable, so dpR2(position | speed) must rise when anchored. If it does not,
the pipeline is not detecting the state difference at all and the speed result
is uninterpretable.

Model and scoring are taken from the existing pipeline rather than reinvented:
src/spatial_manifolds/mlencoding.py, 'count:poisson', 200 rounds, and
poisson_pseudoR2 normalised by the saturated likelihood.

MATCHING. The states are equalised before fitting: bins are matched across
speed deciles and then subsampled to equal count, so neither state gets more
training data or a wider speed range. Matching destroys temporal contiguity, so
folds are shuffled rather than blocked; both states receive identical treatment,
so the comparison stands even though the absolute pR2 values are optimistic
under shuffled folds.

Speed cells are capped at CAP per session, sampled at random from the
positively speed-modulated set, to keep the run within a couple of hours.
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.model_selection import KFold
import pynapple as nap
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.mlencoding import poisson_pseudoR2

_ST = open('/private/tmp/claude-501/-Users-harryclark-Documents-spatial-manifolds/'
           '35f3d8d1-3ead-4d48-8089-491c15b000b7/scratchpad/speed_tuning.py').read()
exec(_ST[:_ST.index(chr(10) + 'if __name__')])          # helpers only

PARAMS = {'objective': 'count:poisson', 'eval_metric': 'logloss', 'seed': 2925,
          'learning_rate': 0.05, 'min_child_weight': 2, 'subsample': 0.6,
          'max_depth': 5, 'gamma': 0.4}
NROUND, FOLDS, CAP = 200, 3, 20


def binned_full(mo, dy, state):
    """Counts, speed, position and state per 250 ms bin."""
    bp, cp = vr_paths(mo, dy)
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials, orig = clip_trials(beh['trials'].as_dataframe(), clusters)
    ids = spatial_cell_ids(mo, dy, clusters, classes=('GC', 'NGS', 'NS'))
    if len(ids) < 5:
        return None
    S, P = beh['S'], beh['P']
    ts, vs = np.asarray(S.index), np.asarray(S.values, dtype=float)
    ps = np.asarray(P.values, dtype=float)
    pt = np.asarray(P.index)
    edges = np.arange(ts[0], ts[-1], BIN)
    if len(edges) < 200:
        return None
    ctr = edges[:-1] + BIN / 2

    j = np.searchsorted(edges, ts) - 1
    ok = (j >= 0) & (j < len(ctr)) & np.isfinite(vs)
    cnt = np.bincount(j[ok], minlength=len(ctr))
    spd = np.bincount(j[ok], weights=vs[ok], minlength=len(ctr))
    spd = np.where(cnt > 0, spd / np.maximum(cnt, 1), np.nan)
    # position is circular: average via unit vectors so 199 and 1 cm do not
    # average to 100
    ang = 2 * np.pi * ps / TL
    jp = np.searchsorted(edges, pt) - 1
    okp = (jp >= 0) & (jp < len(ctr)) & np.isfinite(ang)
    cs = np.bincount(jp[okp], weights=np.cos(ang[okp]), minlength=len(ctr))
    sn = np.bincount(jp[okp], weights=np.sin(ang[okp]), minlength=len(ctr))
    pos = np.mod(np.arctan2(sn, cs), 2 * np.pi) * TL / (2 * np.pi)

    starts, ends = trials.start.values.astype(float), trials.end.values.astype(float)
    nums = trials.number.values.astype(int)
    k = np.searchsorted(starts, ctr) - 1
    inside = (k >= 0) & (k < len(starts))
    k = k.clip(0, len(starts) - 1)
    inside &= ctr <= ends[k]
    lab = np.array([state.get(int(u), np.nan) if ins else np.nan
                    for u, ins in zip(nums[k], inside)], dtype=float)

    keep = (np.isfinite(spd) & (spd >= RUN) & np.isfinite(lab) &
            np.isfinite(pos) & (np.bincount(jp[okp], minlength=len(ctr)) > 0))
    if keep.sum() < 2 * MINBIN:
        return None
    Y = np.empty((keep.sum(), len(ids)))
    for i, c in enumerate(ids):
        sp = np.asarray(clusters[c].index, dtype=float)
        Y[:, i] = np.histogram(sp, bins=edges)[0][keep]      # COUNTS
    return Y, spd[keep], pos[keep], lab[keep] > .5, ids


def cv_pr2(X, y, rng):
    """Cross-validated Poisson pseudo-R2, folds shuffled with a fixed seed."""
    pred = np.empty(len(y))
    for tr, te in KFold(FOLDS, shuffle=True, random_state=0).split(X):
        m = xgb.train(PARAMS, xgb.DMatrix(X[tr], label=y[tr]), NROUND)
        pred[te] = m.predict(xgb.DMatrix(X[te]))
    return float(poisson_pseudoR2(y, pred, np.mean(y)))


def run_session(mo, dy, state, cells, rng):
    got = binned_full(mo, dy, state)
    if got is None:
        return None
    Y, speed, pos, anch, ids = got
    ia, iN = matched_bins(speed, anch, rng)
    k = min(len(ia), len(iN))
    if k < 400:
        return None
    ia, iN = rng.choice(ia, k, replace=False), rng.choice(iN, k, replace=False)

    use = [c for c in cells if c in list(ids)]
    if len(use) > CAP:
        use = list(rng.choice(use, CAP, replace=False))
    rows = []
    for c in use:
        col = list(ids).index(c)
        d = dict(mouse=mo, day=dy, cluster=int(c), n_bins=k)
        for idx, lab in ((ia, 'a'), (iN, 'n')):
            y = Y[idx, col]
            if y.sum() < 50:
                continue
            ang = 2 * np.pi * pos[idx] / TL
            P_ = np.c_[np.sin(ang), np.cos(ang)]      # cyclic position
            S_ = speed[idx][:, None]
            d[f'pr2_pos_{lab}'] = cv_pr2(P_, y, rng)
            d[f'pr2_spd_{lab}'] = cv_pr2(S_, y, rng)
            d[f'pr2_both_{lab}'] = cv_pr2(np.c_[P_, S_], y, rng)
            d[f'rate_{lab}'] = float(y.mean() / BIN)
        for lab in ('a', 'n'):
            if f'pr2_both_{lab}' in d:
                d[f'dspd_{lab}'] = d[f'pr2_both_{lab}'] - d[f'pr2_pos_{lab}']
                d[f'dpos_{lab}'] = d[f'pr2_both_{lab}'] - d[f'pr2_spd_{lab}']
        rows.append(d)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    T = pd.read_csv(f'{OUT}/theta_frequency.csv')
    C = pd.read_csv(f'{OUT}/speed_tuning_cells.csv')
    C = C[(C.p_pool < .05) & (C.r_pool > 0)]                 # speed cells
    sess = sorted(set(zip(T.mouse, T.day)))
    out, t0 = [], time.time()
    for i, (mo, dy) in enumerate(sess, 1):
        st = T[(T.mouse == mo) & (T.day == dy)]
        cells = C[(C.mouse == mo) & (C.day == dy)].cluster.astype(int).tolist()
        if not cells:
            continue
        rng = np.random.default_rng(abs(hash((mo, dy))) % 2**32)
        try:
            R = run_session(int(mo), int(dy), dict(zip(st.trial.astype(int),
                                                       st.anch.astype(bool))), cells, rng)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True)
            continue
        if R is None or not len(R):
            continue
        out.append(R)
        pd.concat(out, ignore_index=True).to_csv(f'{OUT}/speed_deltapr2.csv', index=False)
        print(f'[{i}/{len(sess)}] M{mo}D{dy}: {len(R)} cells, {time.time()-t0:.0f}s',
              flush=True)
    A = pd.concat(out, ignore_index=True)
    print(f'\n{len(A)} cells from {A.groupby(["mouse","day"]).ngroups} sessions')
