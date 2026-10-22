"""
Tests whether two grid modules within a session behave as facets of one
shared spatial-processing system (H1) or as functionally independent
subsystems (H2), using canonical correlation analysis (CCA) on RESIDUAL
population activity (behavioral covariates regressed out first, so shared
position/speed/head-direction tuning -- which would make ANY two spatially
tuned cells look "coupled" regardless of true module identity -- can't drive
the result).

Anatomical distance between modules is NOT regressed out (unlike behavior):
grid modules are expected to have real anatomical (dorsoventral) structure,
so proximity may be genuine signal, not a nuisance confound. It's reported
descriptively alongside the CCA result instead.

Pipeline per session (module 0 vs module 1, cell_type=='GC' from
cell_classifications.csv):
  1. Load time-binned, smoothed, z-scored firing rates per cell (compute_of_tcs
     for OF1, compute_vr_tcs for VR) and behavioral covariates.
  2. OLS-regress out behavior per cell; keep residuals.
  3. Top-k PCs of each module's residual population activity (k<=3).
  4. CCA between the two modules' PC score matrices -> top canonical
     correlation rho1_obs.
  5. Null: circularly shift module 1's PC-score time series by a random
     offset (preserves each module's own internal covariance structure,
     destroys only cross-module temporal alignment), repeat N times.
  6. Ceiling: split the larger module's cells into two random halves, CCA
     between halves (same-module positive control), if enough cells.

Run across all qualifying OF1 sessions, then their matching VR recordings,
and compare conclusions for agreement.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from spatial_manifolds.detect_grids import compute_of_tcs, compute_vr_tcs

SOURCE_PATH = '/Users/harryclark/Downloads/COHORT12/'
OUT_CSV = '/Users/harryclark/Documents/spatial-manifolds/data/module_independence_cca_results.csv'
N_NULL = 300
K_PC = 3
RNG_SEED = 0
MIN_CELLS_PER_MODULE = 3
MIN_CELLS_FOR_CEILING = 6


def ols_residual(y, X):
    """y: [T], X: [T, d] behavioral covariates (no intercept column needed)."""
    Xd = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return y - Xd @ beta


def build_module_matrix(cell_ids, tcs_time, T):
    cols = []
    for cid in cell_ids:
        if cid in tcs_time:
            v = np.asarray(tcs_time[cid])[:T]
            if len(v) < T:
                v = np.pad(v, (0, T - len(v)))
            cols.append(v)
    if not cols:
        return None
    return np.column_stack(cols)


def top_pcs(mat, k=K_PC):
    k = min(k, mat.shape[1] - 1, mat.shape[1])
    if k < 1:
        return None
    return PCA(n_components=k, random_state=0).fit_transform(mat)


def cca_top_corr(A, B):
    k = min(A.shape[1], B.shape[1])
    cca = CCA(n_components=k, max_iter=1000)
    try:
        A_c, B_c = cca.fit_transform(A, B)
    except Exception:
        return np.nan
    corrs = [np.corrcoef(A_c[:, i], B_c[:, i])[0, 1] for i in range(k)]
    return float(np.nanmax(corrs))


def shift_null(A, B, n_null=N_NULL, rng=None):
    rng = rng or np.random.RandomState(RNG_SEED)
    T = A.shape[0]
    nulls = []
    for _ in range(n_null):
        shift = rng.randint(int(T * 0.1), int(T * 0.9))
        B_shift = np.roll(B, shift, axis=0)
        nulls.append(cca_top_corr(A, B_shift))
    return np.array(nulls)


def ceiling_split(mat, rng=None):
    rng = rng or np.random.RandomState(RNG_SEED)
    n = mat.shape[1]
    if n < MIN_CELLS_FOR_CEILING:
        return np.nan
    idx = rng.permutation(n)
    half = n // 2
    A = top_pcs(mat[:, idx[:half]])
    B = top_pcs(mat[:, idx[half:2*half]])
    if A is None or B is None:
        return np.nan
    return cca_top_corr(A, B)


def run_session(mouse, day, session_type, mod0_ids, mod1_ids, mod0_pos, mod1_pos):
    rng = np.random.RandomState(RNG_SEED)
    try:
        if session_type == 'OF1':
            tcs, tcs_time, beh, clusters, ep = compute_of_tcs(
                mouse, day, apply_zscore=True, apply_guassian_filter=True,
                source_path=SOURCE_PATH, session='OF1')
            T = min(len(v) for v in tcs_time.values())
            px = np.array(beh['P_x'].bin_average(bin_size=10, time_units='ms', ep=ep))[:T]
            py = np.array(beh['P_y'].bin_average(bin_size=10, time_units='ms', ep=ep))[:T]
            sp = np.array(beh['S'].bin_average(bin_size=10, time_units='ms', ep=ep))[:T]
            hd = np.array(beh['H'].bin_average(bin_size=10, time_units='ms', ep=ep))[:T]
            for a in (px, py, sp, hd):
                a[np.isnan(a)] = np.nanmedian(a) if np.any(~np.isnan(a)) else 0
            behav = np.column_stack([px, py, sp, np.cos(hd), np.sin(hd)])
        else:  # VR
            tcs, tcs_time, autocorrs, last_ephys_bin, beh, clusters = compute_vr_tcs(
                mouse, day, apply_zscore=True, apply_guassian_filter=True, source_path=SOURCE_PATH)
            T = min(len(v) for v in tcs_time.values())
            tl = 200
            tns = beh['trial_number']
            last_ephys_time_bin = clusters[clusters.index[0]].count(bin_size=10, time_units='ms').index[-1]
            import pynapple as nap
            ep = nap.IntervalSet(start=0, end=last_ephys_time_bin, time_units='s')
            sp = np.array(beh['S'].bin_average(bin_size=10, time_units='ms', ep=ep))[:T]
            dt = np.array(beh['travel'].bin_average(bin_size=10, time_units='ms', ep=ep)) - ((tns[0] - 1) * tl)
            pos = dt % tl
            for a in (sp, pos):
                a[np.isnan(a)] = np.nanmedian(a) if np.any(~np.isnan(a)) else 0
            behav = np.column_stack([pos, sp])

        m0 = build_module_matrix(mod0_ids, tcs_time, T)
        m1 = build_module_matrix(mod1_ids, tcs_time, T)
        if m0 is None or m1 is None or m0.shape[1] < MIN_CELLS_PER_MODULE or m1.shape[1] < MIN_CELLS_PER_MODULE:
            return None

        m0_resid = np.column_stack([ols_residual(m0[:, i], behav) for i in range(m0.shape[1])])
        m1_resid = np.column_stack([ols_residual(m1[:, i], behav) for i in range(m1.shape[1])])

        pc0 = top_pcs(m0_resid)
        pc1 = top_pcs(m1_resid)
        if pc0 is None or pc1 is None:
            return None

        rho_obs = cca_top_corr(pc0, pc1)
        null = shift_null(pc0, pc1, rng=rng)
        p_val = float((np.sum(null >= rho_obs) + 1) / (len(null) + 1))
        ceiling0 = ceiling_split(m0_resid, rng=rng)
        ceiling1 = ceiling_split(m1_resid, rng=rng)

        return dict(
            mouse=mouse, day=day, session_type=session_type,
            n_mod0=m0.shape[1], n_mod1=m1.shape[1], T=T,
            rho_obs=rho_obs, null_mean=float(np.nanmean(null)), null_p95=float(np.nanpercentile(null, 95)),
            p_value=p_val, ceiling_mod0=ceiling0, ceiling_mod1=ceiling1,
            mod0_x=mod0_pos[0], mod0_y=mod0_pos[1], mod1_x=mod1_pos[0], mod1_y=mod1_pos[1],
        )
    except Exception as e:
        print(f'  M{mouse}D{day} {session_type}: FAILED ({e})', flush=True)
        return None


def main():
    df = pd.read_csv('/Users/harryclark/Documents/spatial-manifolds/data/cell_classifications.csv', keep_default_na=False)
    gc = df[(df['cell_type'] == 'GC') & (df['grid_module'].isin(['0', '1']))].copy()
    gc['grid_module'] = gc['grid_module'].astype(int)
    gc['SC_x'] = pd.to_numeric(gc['SC_x'], errors='coerce')
    gc['SC_y'] = pd.to_numeric(gc['SC_y'], errors='coerce')

    sessions = []
    for (mouse, day), g in gc.groupby(['mouse', 'day']):
        m0 = g[g.grid_module == 0]
        m1 = g[g.grid_module == 1]
        if len(m0) >= MIN_CELLS_PER_MODULE and len(m1) >= MIN_CELLS_PER_MODULE:
            sessions.append((
                mouse, day,
                m0.cluster_id.astype(int).tolist(), m1.cluster_id.astype(int).tolist(),
                (m0.SC_x.mean(), m0.SC_y.mean()), (m1.SC_x.mean(), m1.SC_y.mean()),
            ))
    print(f'{len(sessions)} qualifying sessions', flush=True)

    results = []
    for mouse, day, mod0_ids, mod1_ids, mod0_pos, mod1_pos in sessions:
        mouse, day = int(mouse), int(day)
        for stype in ('OF1', 'VR'):
            r = run_session(mouse, day, stype, mod0_ids, mod1_ids, mod0_pos, mod1_pos)
            if r is not None:
                results.append(r)
                print(f'  M{mouse}D{day} {stype}: rho_obs={r["rho_obs"]:.3f} '
                      f'null_mean={r["null_mean"]:.3f} p={r["p_value"]:.3f} '
                      f'ceiling=({r["ceiling_mod0"]:.3f},{r["ceiling_mod1"]:.3f}) '
                      f'n=({r["n_mod0"]},{r["n_mod1"]})', flush=True)
            else:
                print(f'  M{mouse}D{day} {stype}: skipped', flush=True)

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_CSV, index=False)
    print(f'\nSaved {len(res_df)} rows to {OUT_CSV}', flush=True)
    print('CCA_ANALYSIS_DONE', flush=True)


if __name__ == '__main__':
    main()
