"""Cross-region coupling, second full run: can a region RECONSTRUCT anchoring?

Usage:  python3 xr_full2.py <worker> <n_workers>

Two changes from xr_full.py, both of which need the models refitted.

1. THE TARGET THRESHOLD DROPS FROM 20 CELLS TO 10. `MIN_REGION = 10 + N_POOL`
   demanded enough cells to supply ten targets AND a disjoint ten-cell
   within-region pool. That is the wrong arithmetic: the pool is drawn per
   target from the OTHER cells of the region, so the diagonal needs eleven
   cells, not twenty, and a region taking only cross-region pools needs ten.
   The strict rule silently cost 14 (session, region) target rows -- subiculum
   7 -> 12 sessions, visual 16 -> 21, cerebellum 0 -> 3 -- which is most of the
   weakness in the rows of the coupling matrix. Targets are still one per decile
   of PC1-following, so nothing about their selection changes. Where a region is
   too small to also yield a ten-cell within-pool at MIN_UM separation, no
   `within` row is emitted and the matrix diagonal simply reports fewer sessions.

2. THE PREDICTED TRACE IS TURNED BACK INTO ANCHORING LABELS. dpR2 says how much
   a pool improves the fit; it does not say whether the pool carries the
   trial-to-trial structure this project is about. So every model's out-of-fold
   predictions are kept, reshaped into a trial x position map exactly as the
   real spikes are, and run through the same classifier. The question becomes:
   from cells in region X alone, can you recover WHEN the target cell was
   anchored?

   This is only legitimate because the folds are CONTIGUOUS. Neighbouring 10 ms
   bins share history, so random folds would let the model see the same trial it
   is predicting and reconstruct trial structure it was told. `contiguous_folds`
   splits the index array into k unshuffled blocks, so each bin is predicted
   once by a model that never saw its block. Five folds already give one
   out-of-fold prediction for every bin; ten would only enlarge the training
   set, at twice the compute, so FOLDS_ALL stays at 5.

   THE BASELINE IS A REAL CONTROL, NOT A DEGENERATE ONE. A position-and-speed
   model predicts nearly the same profile on every trial, and would look
   trivially "anchored" everywhere -- but the baseline also carries the target's
   own spike history, which does vary trial to trial. So base_lab_r is a genuine
   floor, and the quantity of interest is the GAIN over it, d_lab_r.

   Two reconstruction metrics, neither needing the gate's 200-shift null:
     lab_r      Pearson r between the predicted and real per-trial mean-
                correlation profiles. Continuous, so it does not inherit the
                classifier's 3-trial resolution floor or its discreteness.
     lab_agree  fraction of trials given the same label, classifier run with
                gate=False on both sides so the two are treated identically.

Everything else -- pools of exactly ten cells, two draws averaged, the shifted
floor, the state-conditioned cells -- is unchanged, so this run replaces the
first one rather than supplementing it.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

SP = ('/Users/harryclark/Documents/spatial-manifolds/scripts/figures/'
      'AnchorDynamics2026/cross_region/')
_f = open(SP + 'xr_full.py').read()
exec(_f[:_f.index("\nif __name__")])        # the whole chain, incl. session_frame

sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import trial_cluster_labels

MIN_REGION = N_TARGETS          # was 10 + N_POOL; see the docstring
NBIN_POS = 100                  # 2 cm bins over the 200 cm track, as elsewhere
MIN_SPEED_BIN = 3.0             # rate maps are built from running samples only
OUT2 = OUT


def cv_pr2_pred(Xh, y, idx, k):
    """Contiguous-fold pR2, and the out-of-fold prediction for every bin.

    Returns (pr2, yhat). yhat is NaN wherever a fold was too small to score,
    which is the same condition under which cv_pr2 drops it from the mean.
    """
    fo = contiguous_folds(idx, k)
    out = []
    yhat = np.full(len(y), np.nan)
    for i in range(k):
        te = fo[i]
        tr = np.concatenate([fo[j] for j in range(k) if j != i])
        if len(te) < 100 or y[te].sum() < 20:
            continue
        m = xgb.train(PARAMS, xgb.DMatrix(Xh[tr], label=y[tr]), NROUND)
        pred = m.predict(xgb.DMatrix(Xh[te]))
        yhat[te] = pred
        out.append(float(poisson_pseudoR2(y[te], pred, np.mean(y[te]))))
    return (float(np.nanmean(out)) if out else np.nan), yhat


def rate_map(v, tno, pbin, moving, trials):
    """(n_trials, NBIN_POS) mean of v per trial and position bin, running only."""
    M = np.full((len(trials), NBIN_POS), np.nan)
    ok = moving & np.isfinite(v)
    for i, n in enumerate(trials):
        m = ok & (tno == n)
        if not m.any():
            continue
        s = np.bincount(pbin[m], weights=v[m], minlength=NBIN_POS)
        c = np.bincount(pbin[m], minlength=NBIN_POS).astype(float)
        with np.errstate(invalid='ignore', divide='ignore'):
            M[i] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return M


def recon(yhat, mc_real, lab_real, tno, pbin, moving, trials):
    """How well does the predicted trace reproduce the cell's anchoring dynamics?

    Both sides go through the identical classifier call, so the comparison
    cannot be contaminated by a difference in method between them.
    """
    if not np.isfinite(yhat).any():
        return np.nan, np.nan
    Mp = rate_map(yhat, tno, pbin, moving, trials)
    if np.isfinite(Mp).sum() < Mp.size * .1:
        return np.nan, np.nan
    try:
        lab_p, mc_p, _, _ = trial_cluster_labels(Mp, gate=False)
    except Exception:
        return np.nan, np.nan
    v = np.isfinite(mc_p) & np.isfinite(mc_real)
    r = (float(np.corrcoef(mc_p[v], mc_real[v])[0, 1])
         if v.sum() >= 6 and np.std(mc_p[v]) > 0 and np.std(mc_real[v]) > 0
         else np.nan)
    w = np.isfinite(lab_p) & np.isfinite(lab_real)
    a = float((lab_p[w] == lab_real[w]).mean()) if w.sum() >= 6 else np.nan
    return r, a


def trial_index(mouse, day, S):
    """Trial number and position bin for every time bin.

    Position is recovered from the design matrix rather than reloaded: base
    columns are sin(ang), cos(ang), speed with ang = 2*pi*pos/TL, so the
    inversion is exact.
    """
    import pynapple as nap
    bp, cp = vr_paths(mouse, day)
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials, _ = clip_trials(beh['trials'].as_dataframe(), clusters)
    ctr = (np.arange(S['n']) + .5) * TIME_BS / 1000.
    starts = trials.start.values.astype(float)
    ends = trials.end.values.astype(float)
    nums = trials.number.values.astype(int)
    k = np.searchsorted(starts, ctr) - 1
    inside = (k >= 0) & (k < len(starts))
    k = k.clip(0, len(starts) - 1)
    inside &= ctr <= ends[k]
    tno = np.where(inside, nums[k], -1)
    ang = np.arctan2(S['base'][:, 0], S['base'][:, 1]) % (2 * np.pi)
    pos = ang * TL / (2 * np.pi)
    pbin = np.clip((pos / TL * NBIN_POS).astype(int), 0, NBIN_POS - 1)
    moving = S['speed'] >= MIN_SPEED_BIN
    return tno, pbin, moving, nums


def run_session2(mouse, day, rng):
    got = session_frame(mouse, day, rng)
    if got is None:
        print(f'  - M{mouse}D{day}: skipped', flush=True); return None
    S, cells, ia, iN = got
    counts = cells.region.value_counts()
    regions = [r for r in counts.index if counts[r] >= MIN_REGION and r != 'OTHER']
    partners = [r for r in counts.index if counts[r] >= PARTNER_MIN and r != 'OTHER']
    if not regions or len(partners) < 2:
        print(f'  - M{mouse}D{day}: regions {dict(counts)} insufficient', flush=True)
        return None
    tno, pbin, moving, trials = trial_index(mouse, day, S)
    model = make_model()
    rows = []
    for treg in regions:
        pool = cells[cells.region == treg].copy()
        pool['q'] = pd.qcut(pool.r.rank(method='first'), N_TARGETS, labels=False)
        targets = (pool.sort_values('r').groupby('q', group_keys=False)
                   .apply(lambda d: d.sample(1, random_state=0)))
        for _, t in targets.iterrows():
            tt = time.time()
            y = S['counts'][t.cluster_id]
            # the cell's OWN anchoring dynamics, from its real spikes, through
            # the same classifier call the predictions will go through
            Mr = rate_map(y.astype(float), tno, pbin, moving, trials)
            try:
                lab_real, mc_real, _, _ = trial_cluster_labels(Mr, gate=False)
            except Exception:
                lab_real = mc_real = np.full(len(trials), np.nan)
            Xb, yb = design(S, y, [], model)
            ja = ia[ia < len(yb)]; jn = iN[iN < len(yb)]
            base_all, base_hat = cv_pr2_pred(Xb, yb, np.arange(len(yb)), FOLDS_ALL)
            b_r, b_a = recon(base_hat, mc_real, lab_real, tno, pbin, moving, trials)
            has_state = len(ja) >= 500 and len(jn) >= 500
            base_st = (cross_state(Xb, yb, ja, jn, FOLDS_STATE) if has_state
                       else {})
            wp = within_pool(cells, t)
            srcs = [('within', wp, treg)] if len(wp) >= N_POOL else []
            for p in partners:
                if p == treg:
                    continue
                s = cells[cells.region == p].cluster_id.values
                srcs += [('cross', s, p), ('shift', s, p)]
            for cond, src, pname in srcs:
                if len(src) < N_POOL:
                    continue
                for d in range(N_DRAWS):
                    pick = rng.choice(src, N_POOL, replace=False)
                    cols = [S['counts'][c] for c in pick]
                    if cond == 'shift':
                        cols = [np.roll(c, int(rng.integers(len(c) // 10,
                                                            len(c) * 9 // 10)))
                                for c in cols]
                    raw = np.column_stack([S['base']] + cols)
                    Xh, yh = model.get_all_with_history(raw, y)
                    Xh = np.asarray(Xh); yh = np.asarray(yh)
                    fa, f_hat = cv_pr2_pred(Xh, yh, np.arange(len(yh)), FOLDS_ALL)
                    l_r, l_a = recon(f_hat, mc_real, lab_real, tno, pbin, moving,
                                     trials)
                    cs = (cross_state(Xh, yh, ia[ia < len(yh)], iN[iN < len(yh)],
                                      FOLDS_STATE) if has_state else {})
                    row = dict(mouse=mouse, day=day, cluster_id=int(t.cluster_id),
                               target_region=treg, partner=pname, cond=cond, draw=d,
                               pc1_r=float(t.r),
                               pc1_rank=float(t.q) / (N_TARGETS - 1),
                               n_spikes=int(t.n_spikes), has_state=has_state,
                               pr2_base=base_all, pr2_full=fa, d_all=fa - base_all,
                               lab_r=l_r, lab_agree=l_a,
                               base_lab_r=b_r, base_lab_agree=b_a,
                               d_lab_r=l_r - b_r, d_lab_agree=l_a - b_a,
                               n_trials=int(np.isfinite(mc_real).sum()),
                               pool_pc1=float(cells.set_index('cluster_id')
                                              .loc[pick, 'r'].mean()))
                    for kk in ('AA', 'AN', 'NA', 'NN'):
                        row[f'pr2_{kk}'] = cs.get(kk, np.nan)
                        row[f'base_{kk}'] = base_st.get(kk, np.nan)
                        row[f'd_{kk}'] = (cs.get(kk, np.nan)
                                          - base_st.get(kk, np.nan)
                                          if has_state else np.nan)
                    rows.append(row)
            print(f'   M{mouse}D{day} {treg} {int(t.cluster_id)} '
                  f'(r={t.r:+.2f}) base_lab_r={b_r:+.2f} {time.time()-tt:.0f}s',
                  flush=True)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    w, nw = int(sys.argv[1]), int(sys.argv[2])
    inv = pd.read_csv(SP + 'xr_inventory.csv')
    sess = [(int(r.mouse), int(r.day)) for _, r in inv.iterrows()
            if sum(r[c] >= PARTNER_MIN for c in ('MEC', 'VIS', 'SUB', 'CB')) >= 2]
    mine = [s for i, s in enumerate(sorted(sess)) if i % nw == w]
    shard = f'{OUT2}/xr_full2_w{w}.csv'
    done, out = set(), []
    if os.path.exists(shard):
        prev = pd.read_csv(shard)
        out.append(prev)
        done = set(map(tuple, prev[['mouse', 'day']].drop_duplicates().values))
    print(f'worker {w}/{nw}: {len(mine)} sessions, {len(done)} already done',
          flush=True)
    t0 = time.time()
    for mo, dy in mine:
        if (mo, dy) in done:
            continue
        try:
            R = run_session2(mo, dy,
                             np.random.default_rng(abs(hash((mo, dy))) % 2**32))
        except Exception as e:
            import traceback
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True)
            traceback.print_exc(limit=4)
            continue
        if R is None or not len(R):
            continue
        out.append(R)
        pd.concat(out, ignore_index=True).to_csv(shard, index=False)
        print(f'[w{w}] M{mo}D{dy} saved, {time.time()-t0:.0f}s elapsed', flush=True)
    print(f'worker {w} finished in {time.time()-t0:.0f}s', flush=True)
