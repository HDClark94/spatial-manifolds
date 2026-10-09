"""Phase 1 pilot: cross-region coupling, one session, 10 targets across PC1 deciles.

This is a gate, not a result. It has to show three things before a long run is
worth starting:

  the `shift` floor sits at zero      -- otherwise the folds leak and every
                                         delta downstream is an artifact
  `within` exceeds `cross`            -- otherwise everything predicts everything
  the state matrix is stable          -- across pool draws, or it is noise

DESIGN NOTES that differ from the original notebook, all deliberate:

  n_filters 5, not one per bin. `int(HISTORY_MS / TIME_BS)` gave 100 filters per
  covariate -- 1000 features for a 10-cell pool, and no basis smoothing.

  Shank assignment is round(coord_probe_x / 250). Neuropixels 2.0 puts four
  shanks at 0/250/500/750 um with two columns 32 um apart, so x has 8 values with
  four shanks and 2 with one. The old qcut rule split a single shank's two
  columns into four fake shanks, and its else-branch set every shank to 0, which
  made the within-region pool empty in the 42 of 81 single-shank sessions -- the
  ceiling was silently NaN for half the dataset. The exclusion is now by
  DISTANCE, which is what the shank rule was a proxy for.

  Position enters as sin/cos. The track is a ring.

  The baseline is fitted once per target, not once per pool draw per condition.

  History is built on the CONTIGUOUS session and only then split by state, and
  bins within one history-length of a state transition are dropped -- otherwise a
  bin's history reaches back into the other state.

  ynull for every pR2 comes from the TEST set's own mean. Firing rate differs
  about 7% between states, and using the training mean would score that rate
  difference as a prediction failure.
"""
import os, sys, time, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import xgboost as xgb
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
import pynapple as nap
from spatial_manifolds.mlencoding import MLencoding, poisson_pseudoR2

SP = '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026/cross_region/'
_p0 = open(SP + 'xr_phase0.py').read()
exec(_p0[:_p0.index("\nif __name__")])                    # loaders, geometry, model
_pc = open(SP + 'xr_pc1.py').read()
exec(_pc[:_pc.index("\nif __name__")])                    # pc1_following

OUT = '/Users/harryclark/Documents/spatial-manifolds/data/cross_region_xgboost'
os.makedirs(OUT, exist_ok=True)

N_TARGETS = 10          # one per PC1 decile
N_DRAWS = 2             # random pool draws, averaged
FOLDS_ALL, FOLDS_STATE = 5, 3
NROUND = 200
PARAMS = {'objective': 'count:poisson', 'eval_metric': 'logloss', 'seed': 2925,
          'learning_rate': 0.05, 'min_child_weight': 2, 'subsample': 0.6,
          'max_depth': 5, 'gamma': 0.4}


def contiguous_folds(idx, k):
    """k contiguous blocks of an index array -- never shuffled: neighbouring
    10 ms bins share history, so random folds leak."""
    return np.array_split(idx, k)


def fit_score(Xh, y, tr, te):
    m = xgb.train(PARAMS, xgb.DMatrix(Xh[tr], label=y[tr]), NROUND)
    pred = m.predict(xgb.DMatrix(Xh[te]))
    return float(poisson_pseudoR2(y[te], pred, np.mean(y[te])))


def cv_pr2(Xh, y, idx, k):
    """Contiguous-fold pR2 within one index set."""
    fo = contiguous_folds(idx, k)
    out = []
    for i in range(k):
        te = fo[i]
        tr = np.concatenate([fo[j] for j in range(k) if j != i])
        if len(te) < 100 or y[te].sum() < 20:
            continue
        out.append(fit_score(Xh, y, tr, te))
    return float(np.nanmean(out)) if out else np.nan


def cross_state(Xh, y, ia, iN, k):
    """2x2 train x test matrix. Off-diagonals train on one state and test on the
    whole of the other, which is never in the training set, so no fold is needed
    there -- the diagonal is what needs held-out folds."""
    fa, fn = contiguous_folds(ia, k), contiguous_folds(iN, k)
    res = {}
    for tag, folds, other in (('A', fa, iN), ('N', fn, ia)):
        diag, off = [], []
        for i in range(k):
            tr = np.concatenate([folds[j] for j in range(k) if j != i])
            te = folds[i]
            if len(te) < 100 or y[tr].sum() < 20:
                continue
            m = xgb.train(PARAMS, xgb.DMatrix(Xh[tr], label=y[tr]), NROUND)
            p_d = m.predict(xgb.DMatrix(Xh[te]))
            diag.append(float(poisson_pseudoR2(y[te], p_d, np.mean(y[te]))))
            p_o = m.predict(xgb.DMatrix(Xh[other]))
            off.append(float(poisson_pseudoR2(y[other], p_o, np.mean(y[other]))))
        res[f'{tag}{tag}'] = float(np.nanmean(diag)) if diag else np.nan
        res[f'{tag}{"N" if tag == "A" else "A"}'] = float(np.nanmean(off)) if off else np.nan
    return res


def design(S, y_counts, pool_ids, model):
    """Behaviour + pool, expanded by the history basis, built on the full session."""
    raw = S['base'] if not pool_ids else np.column_stack(
        [S['base']] + [S['counts'][c] for c in pool_ids])
    Xh, yh = model.get_all_with_history(raw, y_counts)
    return np.asarray(Xh), np.asarray(yh)


def run(mouse, day, partner, rng):
    t0 = time.time()
    S = load_session(mouse, day)
    P = pc1_following(mouse, day)
    if P is None:
        return None
    cells = S['cells'].merge(P[['cluster_id', 'r', 'is_mec']], on='cluster_id',
                             how='inner')
    print(f'M{mouse}D{day}: {S["n"]} bins, {len(cells)} cells with PC1 '
          f'({time.time()-t0:.0f}s)', flush=True)

    # per-bin trial and state
    bp, cp = vr_paths(mouse, day)
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials, orig = clip_trials(beh['trials'].as_dataframe(), clusters)
    tr_a, L, frac = population_anchoring(mouse, day)
    # .tolist() gives Python bools: `np.True_ is True` is False, so an
    # `is True` test against numpy booleans silently matches nothing
    st_of = dict(zip(tr_a.number.astype(int),
                     (np.asarray(frac) > .5).tolist()))
    ctr = (np.arange(S['n']) + .5) * TIME_BS / 1000.
    starts, ends = trials.start.values.astype(float), trials.end.values.astype(float)
    nums = trials.number.values.astype(int)
    k = np.searchsorted(starts, ctr) - 1
    inside = (k >= 0) & (k < len(starts))
    k = k.clip(0, len(starts) - 1)
    inside &= ctr <= ends[k]
    tno = np.where(inside, nums[k], -1)
    state = np.array([st_of.get(int(t), np.nan) if t > 0 else np.nan for t in tno],
                     dtype=float)

    # trial-level speed matching between states, then bins of the kept trials
    tsp = {int(n): float(np.nanmean(S['speed'][tno == n])) for n in nums if (tno == n).any()}
    ta = [n for n in nums if st_of.get(int(n)) is True and n in tsp]
    tn_ = [n for n in nums if st_of.get(int(n)) is False and n in tsp]
    qs = np.quantile([tsp[n] for n in ta + tn_], np.linspace(0, 1, 11))
    qs[0] -= 1; qs[-1] += 1
    keep_a, keep_n = [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        A = [n for n in ta if lo <= tsp[n] < hi]
        N = [n for n in tn_ if lo <= tsp[n] < hi]
        m = min(len(A), len(N))
        if m:
            keep_a += list(rng.choice(A, m, replace=False))
            keep_n += list(rng.choice(N, m, replace=False))
    # drop bins whose history reaches into the other state
    edge = np.zeros(S['n'], bool)
    ch = np.where(np.diff(np.nan_to_num(state, nan=-1)) != 0)[0]
    span = int(HISTORY_MS / TIME_BS)
    for c in ch:
        edge[max(0, c - span):c + span] = True
    ia = np.where(np.isin(tno, keep_a) & ~edge)[0]
    iN = np.where(np.isin(tno, keep_n) & ~edge)[0]
    m = min(len(ia), len(iN))
    ia, iN = ia[:m], iN[:m]
    print(f'  state bins: {len(ia)} anchored / {len(iN)} non (speed- and '
          f'count-matched, {edge.sum()} edge bins dropped)', flush=True)

    # 10 targets, one per PC1 decile, in the target region
    tgt_cells = cells[cells.region == 'MEC'].copy()
    tgt_cells['q'] = pd.qcut(tgt_cells.r.rank(method='first'), N_TARGETS, labels=False)
    targets = (tgt_cells.sort_values('r').groupby('q', group_keys=False)
               .apply(lambda d: d.sample(1, random_state=0)))
    print(f'  {len(targets)} targets, PC1 r from {targets.r.min():+.3f} to '
          f'{targets.r.max():+.3f}', flush=True)

    model = make_model()
    cross_src = cells[cells.region == partner].cluster_id.values
    rows = []
    for _, t in targets.iterrows():
        tt = time.time()
        y_raw = S['counts'][t.cluster_id]
        wp = within_pool(cells, t)
        Xb, yb = design(S, y_raw, [], model)
        base_all = cv_pr2(Xb, yb, np.arange(len(yb)), FOLDS_ALL)
        base_st = cross_state(Xb, yb, ia[ia < len(yb)], iN[iN < len(yb)], FOLDS_STATE)
        for cond, src in (('cross', cross_src), ('within', wp), ('shift', cross_src)):
            if len(src) < N_POOL:
                continue
            for d in range(N_DRAWS):
                pick = rng.choice(src, N_POOL, replace=False)
                cols = [S['counts'][c] for c in pick]
                if cond == 'shift':
                    cols = [np.roll(c, int(rng.integers(len(c) // 10, len(c) * 9 // 10)))
                            for c in cols]
                raw = np.column_stack([S['base']] + cols)
                Xh, yh = model.get_all_with_history(raw, y_raw)
                Xh = np.asarray(Xh); yh = np.asarray(yh)
                full_all = cv_pr2(Xh, yh, np.arange(len(yh)), FOLDS_ALL)
                cs = cross_state(Xh, yh, ia[ia < len(yh)], iN[iN < len(yh)],
                                 FOLDS_STATE)
                row = dict(mouse=mouse, day=day, cluster_id=int(t.cluster_id),
                           region=t.region, partner=partner, cond=cond, draw=d,
                           pc1_r=float(t.r), pc1_rank=float(t.q) / (N_TARGETS - 1),
                           n_spikes=int(t.n_spikes),
                           pr2_base=base_all, pr2_full=full_all,
                           d_all=full_all - base_all,
                           d_all_norm=(full_all - base_all) / max(1e-9, 1 - base_all),
                           pool_pc1=float(cells.set_index('cluster_id')
                                          .loc[pick, 'r'].mean()))
                for kk in ('AA', 'AN', 'NA', 'NN'):
                    row[f'pr2_{kk}'] = cs.get(kk, np.nan)
                    row[f'base_{kk}'] = base_st.get(kk, np.nan)
                    row[f'd_{kk}'] = cs.get(kk, np.nan) - base_st.get(kk, np.nan)
                rows.append(row)
        print(f'   target {int(t.cluster_id)} (PC1 r={t.r:+.3f}) done in '
              f'{time.time()-tt:.0f}s', flush=True)
        pd.DataFrame(rows).to_csv(f'{OUT}/pilot_M{mouse}D{day}.csv', index=False)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    R = run(26, 18, 'VIS', np.random.default_rng(0))
    print(f'\n{len(R)} rows')
    g = R.groupby('cond')[['d_all', 'pr2_base', 'd_AA', 'd_NN', 'd_AN', 'd_NA']].mean()
    print(g.round(4).to_string())
