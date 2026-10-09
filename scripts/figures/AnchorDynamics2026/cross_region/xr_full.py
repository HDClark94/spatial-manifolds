"""Cross-region coupling, full run. Usage: python3 xr_full.py <worker> <n_workers>

Built on the pilot (xr_pilot.py), which passed its gates on M26 D18:
shift floor +0.0003, within +0.0469, cross +0.0244 -- floor at zero and a clean
within > cross > shift ordering.

WHAT THE PILOT CHANGED ABOUT THE ANALYSIS. The `shift` floor is NOT zero in the
off-diagonal cells of the state matrix: training on one state and testing on the
other gives a floor of +0.035, because a baseline that generalises badly across
states is strongly negative there, so adding ANY covariate -- even one whose
timing has been destroyed -- appears to help. Raw off-diagonal deltas are
therefore uninterpretable, and every condition carries its own floor within every
cell of the matrix so it can be subtracted. Do not read d_AN or d_NA without
subtracting d_AN/d_NA of the shift condition.

Targets are 10 per region per session, one per decile of PC1 following, so the
rank axis is covered by construction rather than by luck. Every region with
enough cells serves as both target and partner, so MEC->VIS and VIS->MEC are both
measured.

Work is split across worker processes by session index. Each worker writes its
own shard and skips sessions already present in it, so a killed run resumes.
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

SP = '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026/cross_region/'
_pl = open(SP + 'xr_pilot.py').read()
exec(_pl[:_pl.index("\nif __name__")])          # loaders, folds, cross_state, pc1

MIN_REGION = 10 + N_POOL        # need targets AND a pool from the same region
PARTNER_MIN = N_POOL


def session_frame(mouse, day, rng):
    """Everything shared by every target in this session."""
    S = load_session(mouse, day)
    P = pc1_following(mouse, day)
    if P is None:
        return None
    cells = S['cells'].merge(P[['cluster_id', 'r', 'is_mec']], on='cluster_id',
                             how='inner')
    bp, cp = vr_paths(mouse, day)
    import pynapple as nap
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials, orig = clip_trials(beh['trials'].as_dataframe(), clusters)
    tr_a, L, frac = population_anchoring(mouse, day)
    st_of = dict(zip(tr_a.number.astype(int), (np.asarray(frac) > .5).tolist()))
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
    tsp = {int(n): float(np.nanmean(S['speed'][tno == n])) for n in nums if (tno == n).any()}
    ta = [n for n in nums if st_of.get(int(n)) is True and n in tsp]
    tn_ = [n for n in nums if st_of.get(int(n)) is False and n in tsp]
    if len(ta) < 10 or len(tn_) < 10:
        return S, cells, np.array([], int), np.array([], int)
    qs = np.quantile([tsp[n] for n in ta + tn_], np.linspace(0, 1, 11))
    qs[0] -= 1; qs[-1] += 1
    ka, kn = [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        A = [n for n in ta if lo <= tsp[n] < hi]
        N = [n for n in tn_ if lo <= tsp[n] < hi]
        m = min(len(A), len(N))
        if m:
            ka += list(rng.choice(A, m, replace=False))
            kn += list(rng.choice(N, m, replace=False))
    edge = np.zeros(S['n'], bool)
    span = int(HISTORY_MS / TIME_BS)
    for c in np.where(np.diff(np.nan_to_num(state, nan=-1)) != 0)[0]:
        edge[max(0, c - span):c + span] = True
    ia = np.where(np.isin(tno, ka) & ~edge)[0]
    iN = np.where(np.isin(tno, kn) & ~edge)[0]
    m = min(len(ia), len(iN))
    if m < 2000:
        # No usable state contrast -- but the ALL-TRIALS delta pR2 does not need
        # one, and it is the primary comparison. Returning None here discarded
        # the whole session: 36 sessions carry the regions this analysis needs
        # while only 19 also have both states, so insisting on both threw away
        # nearly half the sample for a question that never asked about state.
        return S, cells, np.array([], int), np.array([], int)
    return S, cells, ia[:m], iN[:m]


def run_session(mouse, day, rng):
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
            Xb, yb = design(S, y, [], model)
            ja = ia[ia < len(yb)]; jn = iN[iN < len(yb)]
            base_all = cv_pr2(Xb, yb, np.arange(len(yb)), FOLDS_ALL)
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
                    fa = cv_pr2(Xh, yh, np.arange(len(yh)), FOLDS_ALL)
                    cs = (cross_state(Xh, yh, ia[ia < len(yh)], iN[iN < len(yh)],
                                      FOLDS_STATE) if has_state else {})
                    row = dict(mouse=mouse, day=day, cluster_id=int(t.cluster_id),
                               target_region=treg, partner=pname, cond=cond, draw=d,
                               pc1_r=float(t.r),
                               pc1_rank=float(t.q) / (N_TARGETS - 1),
                               n_spikes=int(t.n_spikes), has_state=has_state,
                               pr2_base=base_all,
                               pr2_full=fa, d_all=fa - base_all,
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
                  f'(r={t.r:+.2f}) {time.time()-tt:.0f}s', flush=True)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    w, nw = int(sys.argv[1]), int(sys.argv[2])
    inv = pd.read_csv(SP + 'xr_inventory.csv')
    sess = [(int(r.mouse), int(r.day)) for _, r in inv.iterrows()
            if sum(r[c] >= PARTNER_MIN for c in ('MEC', 'VIS', 'SUB', 'CB')) >= 2]
    mine = [s for i, s in enumerate(sorted(sess)) if i % nw == w]
    shard = f'{OUT}/xr_full_w{w}.csv'
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
            R = run_session(mo, dy, np.random.default_rng(abs(hash((mo, dy))) % 2**32))
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
