"""Controls and extensions for Figure 4's mediolateral gradient in following.

Four analyses, all on the cells and sessions behind Figure 4D-F: MEC cells
(brain_region ENTm*) in the 27 sessions whose shanks span at least 200 um.

1. LEAVE-ONE-SHANK-OUT. A cell's agreement is its correlation with the session's
   PC1, built from the MEC population with the cell itself removed. That still
   lets a shank dominate the axis: if the medial shanks contribute more cells,
   or more reliable labels, PC1 leans toward them and lateral cells agree less
   partly by construction. Here every cell is scored against a PC1 built only
   from the OTHER shanks, so no cell's own shank helps define its axis. If the
   gradient survives, it is not a construction of the axis.

2. ONE MODEL. The identity confound is reported in the text only as a subset
   test (non-grid cells, rho = -0.100, p = 0.111). Here agreement is modelled on
   within-session mediolateral position together with identity, layer and log
   firing rate, session as a random intercept, so position is tested beyond
   what the other three explain and no cells are discarded. Firing rate is in
   because more spikes make a cell's trial labels more reliable, which raises
   its correlation with any axis. Run on raw agreement and on agreement above
   chance (|r| minus the mean |r| of the cell's own circular-shift null).

3. WHO SWITCHES FIRST. Each cell's lag relative to the population state is the
   centroid of its positive cross-correlation with that state over +-LAG
   trials, the state being the sign of the leave-one-shank-out PC1, so the cell
   and its shank never define the state it is timed against. Negative = the
   cell changes before the population. Within session, lag against position.
   Only cells that follow (leave-shank-out r > R_MIN) have a lag worth reading.

4. BY REGION. The shanks also sample parasubiculum and visual cortex, scored
   against the same MEC axis. Per session, the mean agreement of each region,
   compared within session.

Writes data/population_state/fig4_controls_cells.csv (per cell),
       data/population_state/fig4_controls_summary.json
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from scipy.stats import spearmanr, wilcoxon

from spatial_manifolds.anchoring import load_session_labels

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
PS = f'{ROOT}/data/population_state'
MIN_SPAN_UM = 200.0
MIN_AXIS_CELLS = 5
LAG = 5
R_MIN = .10
MIN_CELLS_SESSION = 20


def axis_from(M, frac):
    """PC1 of a cells x trials label matrix, signed to the anchored fraction."""
    Mc = np.nan_to_num(M - np.nanmean(M, axis=1, keepdims=True))
    _, _, Vt = np.linalg.svd(Mc, full_matrices=False)
    pc = Vt[0]
    if np.corrcoef(pc, np.nan_to_num(frac))[0, 1] < 0:
        pc = -pc
    return pc


def region_class(acr):
    a = str(acr)
    if a.startswith('ENTm'):
        return 'MEC sup' if a[4:5] in ('1', '2', '3') else 'MEC deep'
    if a.startswith(('PAR', 'PRE', 'POST')):
        return 'pre/para'
    if a.startswith('VIS'):
        return 'visual'
    return 'other'


def lag_centroid(v, state):
    """Centroid of the positive cross-correlation of v with state over +-LAG.

    c(l) = corr(v[t], state[t - l]) ... arranged so that NEGATIVE lag means v
    changes BEFORE state (v at t matches state at t + |l|).
    """
    lags = np.arange(-LAG, LAG + 1)
    c = np.full(len(lags), np.nan)
    for i, l in enumerate(lags):
        if l < 0:
            a, b = v[:l], state[-l:]
        elif l > 0:
            a, b = v[l:], state[:-l]
        else:
            a, b = v, state
        if np.std(a) > 0 and np.std(b) > 0:
            c[i] = np.corrcoef(a, b)[0, 1]
    w = np.clip(np.nan_to_num(c), 0, None)
    if w.sum() == 0:
        return np.nan
    return float(np.sum(lags * w) / w.sum())


def session(mo, dy, C, rng):
    z = load_session_labels(mo, dy)
    if z is None:
        return None
    L, ids, frac = z['labels'], [int(c) for c in z['cluster_id']], z['frac_anch']
    in_pop = np.isfinite(z['pc1_load'])
    cs = C[(C.mouse == mo) & (C.day == dy)].set_index('cluster_id')
    ml = np.array([abs(cs.coord_SCs_x.get(c, np.nan)) for c in ids])
    shank = np.round(ml, 0)
    usable = ~np.isnan(L).all(axis=1)
    rows = []
    for s in np.unique(shank[np.isfinite(shank)]):
        others = in_pop & usable & (shank != s)
        if others.sum() < MIN_AXIS_CELLS:
            continue
        pc = axis_from(L[others], frac)
        state = (pc > np.median(pc)).astype(float)
        for k in np.where(shank == s)[0]:
            v = np.nan_to_num(L[k])
            if np.std(v) == 0:
                continue
            r = float(np.corrcoef(v, pc)[0, 1])
            null = np.array([np.corrcoef(np.roll(v, sh), pc)[0, 1]
                             for sh in rng.integers(5, len(v) - 5, 100)])
            rows.append(dict(
                mouse=mo, day=dy, cluster_id=ids[k], ml=ml[k], shank=s,
                in_pop=bool(in_pop[k]), r_lso=r,
                excess_lso=abs(r) - float(np.nanmean(np.abs(null))),
                lag=lag_centroid(v, state) if r > R_MIN else np.nan,
                n_axis=int(others.sum())))
    return pd.DataFrame(rows) if rows else None


def within(df, x, y, min_cells=MIN_CELLS_SESSION):
    ws = []
    for _, g in df.dropna(subset=[x, y]).groupby(['mouse', 'day']):
        if len(g) < min_cells or g[x].nunique() < 2 or g[y].nunique() < 2:
            continue
        r = spearmanr(g[x], g[y])[0]
        if np.isfinite(r):
            ws.append(r)
    ws = np.array(ws)
    return ws, (float(wilcoxon(ws).pvalue) if len(ws) >= 6 else np.nan)


if __name__ == '__main__':
    CLS = pd.read_csv(f'{ROOT}/data/cell_classifications_v2.csv')
    U = pd.read_csv(f'{PS}/unit_table.csv')
    R = pd.read_csv(f'{PS}/pc1_by_region.csv')
    # the 27 multi-shank sessions of Figure 4D/F: MEC cells spanning >= 200 um
    mecC = CLS[CLS.brain_region.astype(str).str.startswith('ENTm')].copy()
    mecC['ml'] = mecC.coord_SCs_x.abs()
    span = mecC.dropna(subset=['ml']).groupby(['mouse', 'day']).ml.agg(lambda v: v.max() - v.min())
    sess = sorted(span[span >= MIN_SPAN_UM].index)
    print(f'{len(sess)} multi-shank sessions')

    out = []
    for mo, dy in sess:
        d = session(int(mo), int(dy), CLS, np.random.default_rng(abs(hash((mo, dy))) % 2 ** 32))
        if d is not None:
            out.append(d)
    A = pd.concat(out, ignore_index=True)
    A = A.merge(CLS[['mouse', 'day', 'cluster_id', 'brain_region']].drop_duplicates(
        ['mouse', 'day', 'cluster_id']), on=['mouse', 'day', 'cluster_id'], how='left')
    A = A.merge(U[['mouse', 'day', 'cluster_id', 'identity', 'firing_rate']],
                on=['mouse', 'day', 'cluster_id'], how='left')
    A = A.merge(R[['mouse', 'day', 'cluster_id', 'r', 'excess']],
                on=['mouse', 'day', 'cluster_id'], how='left')
    A['region'] = A.brain_region.map(region_class)
    A['mec'] = A.brain_region.astype(str).str.startswith('ENTm')
    A.to_csv(f'{PS}/fig4_controls_cells.csv', index=False)
    M = A[A.mec].copy()
    S = {}

    # 1. leave-one-shank-out
    ws_std, p_std = within(M, 'ml', 'r')
    ws_lso, p_lso = within(M, 'ml', 'r_lso')
    S['lso'] = dict(n=len(ws_lso), rho_std=float(np.median(ws_std)), p_std=p_std,
                    rho_lso=float(np.median(ws_lso)), p_lso=p_lso,
                    neg_lso=int((ws_lso < 0).sum()),
                    corr_r_rlso=float(M[['r', 'r_lso']].dropna().corr().iloc[0, 1]))
    print('1 LSO', S['lso'])

    # 2. mixed model
    import statsmodels.formula.api as smf
    M['ml_c'] = (M.ml - M.groupby(['mouse', 'day']).ml.transform('mean')) / 100.0
    M['sup'] = M.region.eq('MEC sup').astype(float)
    M['lrate'] = np.log10(M.firing_rate.clip(lower=.01))
    M['sess'] = M.mouse.astype(str) + '_' + M.day.astype(str)
    S['lmm'] = {}
    for y in ('r', 'excess', 'r_lso'):
        d = M.dropna(subset=[y, 'ml_c', 'identity', 'lrate'])
        d = d[d.identity.isin(['grid', 'non-grid spatial', 'non-spatial', 'putative interneuron'])]
        # Position varies only BETWEEN shanks (3-4 values per session), so cells
        # on one shank are not independent evidence about it. Shank is a
        # random effect nested in session; without it the position p-value
        # counts cells where it should count shanks.
        d = d.assign(shank_id=d.sess + '_' + d.shank.astype(int).astype(str))
        fit = smf.mixedlm(f'{y} ~ ml_c + C(identity, Treatment("non-grid spatial")) + sup + lrate',
                          d, groups=d['sess'], re_formula='1',
                          vc_formula={'shank': '0 + C(shank_id)'}).fit(method='lbfgs')
        coef = {k: dict(b=float(fit.params[k]), se=float(fit.bse[k]), p=float(fit.pvalues[k]))
                for k in fit.params.index if 'Var' not in k}
        S['lmm'][y] = dict(n_cells=len(d), n_sess=d.sess.nunique(), coef=coef)
        b = coef['ml_c']
        print(f'2 LMM {y}: ML per 100 um b={b["b"]:+.4f} se={b["se"]:.4f} p={b["p"]:.3g} '
              f'(n={len(d)} cells, {d.shank_id.nunique()} shanks, {d.sess.nunique()} sessions)')
        for k, v in coef.items():
            if k not in ('Intercept', 'ml_c'):
                print(f'      {k:60s} b={v["b"]:+.4f} p={v["p"]:.3g}')

    # 3. lead/lag
    ws_lag, p_lag = within(M, 'ml', 'lag', min_cells=10)
    S['lag'] = dict(n=len(ws_lag), rho=float(np.median(ws_lag)) if len(ws_lag) else np.nan,
                    p=p_lag, pos=int((ws_lag > 0).sum()),
                    n_cells=int(M.lag.notna().sum()),
                    median_lag=float(M.lag.median()))
    print('3 LAG', S['lag'])

    # 4. by region, within session
    G = A.dropna(subset=['r']).groupby(['mouse', 'day', 'region']).r.mean().unstack()
    S['region'] = {}
    for reg in ('MEC deep', 'pre/para', 'visual'):
        g = G[['MEC sup', reg]].dropna()
        if len(g) >= 6:
            S['region'][reg] = dict(n=len(g), sup=float(g['MEC sup'].median()),
                                    other=float(g[reg].median()),
                                    p=float(wilcoxon(g['MEC sup'], g[reg]).pvalue))
    S['region_means'] = {k: float(v) for k, v in G.median().items()}
    print('4 REGION', S['region'], S['region_means'])
    json.dump(S, open(f'{PS}/fig4_controls_summary.json', 'w'), indent=1, default=float)
    print('wrote', f'{PS}/fig4_controls_summary.json')
