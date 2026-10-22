"""
Three simple tests relating non-grid spatial cells' functional coupling to
grid cells with the anatomical layout established in the "Figure 2" panel
(grid vs non-grid cell anatomical distribution). Reuses the exact
conventions from anatomical_dependence.ipynb (rich baseline
'pos_speed_hd_hing_lfp', Spearman + Mantel-style permutation, 3D SC_x/y/z
distance) and the precomputed pairwise-assay CSVs in data/eddie/xgboost_pairwise/
(target=non-grid cell, covariate=any cell including grid cells -- no new
XGBoost computation needed).

Test 1: does non-grid-to-grid-cell delta-pR2 fall off with anatomical
        distance, same style as the existing grid-to-grid anatomical check?
Test 2: at MATCHED anatomical distance, is coupling still higher for
        "own module" (nearest module) pairs than "other module" pairs?
        (tests whether anatomy alone explains any module-specific signal)
Test 3: per non-grid cell, does a signed module-coupling-specificity score
        correlate with a signed anatomical-proximity-to-module score, pooled
        across sessions?
"""
import glob, os, re, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

DATA_PATH = '/Users/harryclark/Documents/spatial-manifolds/data/eddie/xgboost_pairwise/'
CC_PATH = '/Users/harryclark/Documents/spatial-manifolds/data/cell_classifications.csv'
OUT_DIR = '/Users/harryclark/Documents/spatial-manifolds/data'
HISTORY_LENGTH = 100
SESSION_TYPE = 'OF1'
BASELINE = 'pos_speed_hd_hing_lfp'  # matches anatomical_dependence.ipynb's RICH_BASELINE['OF1']
N_PERM = 1000
RNG_SEED = 42
MIN_CELLS_PER_MODULE = 3
N_DIST_BINS = 4

cc = pd.read_csv(CC_PATH, low_memory=False, keep_default_na=False)
cc['mouse'] = cc['mouse'].astype(int)
cc['day'] = cc['day'].astype(int)
cc['cluster_id'] = cc['cluster_id'].astype(int)
for col in ('SC_x', 'SC_y', 'SC_z'):
    cc[col] = pd.to_numeric(cc[col], errors='coerce')


def get_anatomy(mouse, day):
    sub = cc[(cc.mouse == mouse) & (cc.day == day)].drop_duplicates('cluster_id')
    return sub.set_index('cluster_id')[['SC_x', 'SC_y', 'SC_z', 'cell_type']]


def get_module_grid_cells(mouse, day):
    sub = cc[(cc.mouse == mouse) & (cc.day == day) & (cc.cell_type == 'GC')].drop_duplicates('cluster_id')
    sub = sub[sub.grid_module.isin(['0', '1'])].copy()
    sub['grid_module'] = sub.grid_module.astype(int)
    mod = {int(g): grp.cluster_id.astype(int).tolist() for g, grp in sub.groupby('grid_module')}
    return mod


def load_pairwise(mouse, day):
    pattern = os.path.join(DATA_PATH, f'xgboost_pairwise_M{mouse}_D{day}_h{HISTORY_LENGTH}_{SESSION_TYPE}_*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return pd.concat([pd.read_csv(f, keep_default_na=False) for f in files], ignore_index=True)


def dist3(coords, id_a, id_b):
    a = coords.loc[id_a, ['SC_x', 'SC_y', 'SC_z']].values.astype(float)
    b = coords.loc[id_b, ['SC_x', 'SC_y', 'SC_z']].values.astype(float)
    return float(np.sqrt(((a - b) ** 2).sum()))


def build_session_pairs(mouse, day):
    mod = get_module_grid_cells(mouse, day)
    if 0 not in mod or 1 not in mod or len(mod[0]) < MIN_CELLS_PER_MODULE or len(mod[1]) < MIN_CELLS_PER_MODULE:
        return None
    pw = load_pairwise(mouse, day)
    if pw is None:
        return None
    coords = get_anatomy(mouse, day)
    grid_ids = mod[0] + mod[1]

    bl = (pw[(pw.baseline == BASELINE) & (pw.covariate_cluster_id == -1)]
          [['target_cluster_id', 'pR2_cv']].rename(columns={'pR2_cv': 'pR2_baseline'})
          .drop_duplicates('target_cluster_id'))
    cv = (pw[(pw.baseline == BASELINE) & (pw.target_type == 'NGS') &
             (pw.covariate_cluster_id.isin(grid_ids))]
          [['target_cluster_id', 'covariate_cluster_id', 'pR2_cv']].copy())
    if len(cv) == 0:
        return None
    merged = cv.merge(bl, on='target_cluster_id', how='inner')
    merged['dpr2'] = merged['pR2_cv'] - merged['pR2_baseline']
    merged['module'] = merged['covariate_cluster_id'].apply(lambda c: 0 if c in mod[0] else 1)

    have_coord = set(coords.index[coords[['SC_x', 'SC_y', 'SC_z']].notna().all(axis=1)])
    merged = merged[merged.target_cluster_id.isin(have_coord) & merged.covariate_cluster_id.isin(have_coord)].copy()
    if len(merged) < 10:
        return None
    merged['dist'] = merged.apply(lambda r: dist3(coords, r.target_cluster_id, r.covariate_cluster_id), axis=1)
    return merged, mod, coords


def test1(merged, rng):
    rho_obs, _ = spearmanr(merged.dist, merged.dpr2)
    null = np.empty(N_PERM)
    d = merged.dist.values.copy()
    y = merged.dpr2.values
    for k in range(N_PERM):
        null[k] = spearmanr(rng.permutation(d), y)[0]
    p = (np.sum(np.abs(null) >= np.abs(rho_obs)) + 1) / (N_PERM + 1)
    return rho_obs, p


def test2(merged, coords, mod, rng):
    mean_dist_to = {}
    for tgt, g in merged.groupby('target_cluster_id'):
        d0 = g[g.module == 0].dist.mean()
        d1 = g[g.module == 1].dist.mean()
        if np.isnan(d0) or np.isnan(d1):
            continue
        mean_dist_to[tgt] = 0 if d0 < d1 else 1
    if len(mean_dist_to) < 3:
        return None
    m = merged[merged.target_cluster_id.isin(mean_dist_to.keys())].copy()
    m['nearest'] = m.target_cluster_id.map(mean_dist_to)
    m['own'] = m.module == m.nearest
    try:
        m['dist_bin'] = pd.qcut(m.dist, q=min(N_DIST_BINS, m.dist.nunique()), labels=False, duplicates='drop')
    except Exception:
        return None
    rows = []
    for b, g in m.groupby('dist_bin'):
        own = g[g.own].dpr2
        other = g[~g.own].dpr2
        if len(own) >= 2 and len(other) >= 2:
            rows.append(dict(bin=b, n_own=len(own), n_other=len(other),
                              mean_own=own.mean(), mean_other=other.mean(),
                              mean_dist=g.dist.mean()))
    return pd.DataFrame(rows) if rows else None


def test3(merged, coords, mod, rng):
    c0 = coords.loc[mod[0], ['SC_x', 'SC_y', 'SC_z']].astype(float).mean().values
    c1 = coords.loc[mod[1], ['SC_x', 'SC_y', 'SC_z']].astype(float).mean().values
    rows = []
    for tgt, g in merged.groupby('target_cluster_id'):
        d0 = g[g.module == 0].dpr2
        d1 = g[g.module == 1].dpr2
        if len(d0) < 1 or len(d1) < 1:
            continue
        specificity = d0.mean() - d1.mean()
        tgt_xyz = coords.loc[tgt, ['SC_x', 'SC_y', 'SC_z']].astype(float).values
        anat_pref = np.linalg.norm(tgt_xyz - c1) - np.linalg.norm(tgt_xyz - c0)
        rows.append(dict(target=tgt, specificity=specificity, anat_pref=anat_pref))
    return pd.DataFrame(rows) if rows else None


def main():
    qualifying = [(20, 16), (20, 25), (21, 16), (21, 20), (21, 21), (21, 23), (21, 26),
                  (25, 18), (25, 19), (25, 21), (25, 22), (25, 24), (25, 25), (26, 18),
                  (26, 19), (29, 20), (29, 25)]
    rng = np.random.default_rng(RNG_SEED)

    t1_rows, t2_rows, t3_rows = [], [], []
    for mouse, day in qualifying:
        built = build_session_pairs(mouse, day)
        if built is None:
            print(f'M{mouse}D{day}: skipped (no data)', flush=True)
            continue
        merged, mod, coords = built
        rho, p = test1(merged, rng)
        t1_rows.append(dict(mouse=mouse, day=day, n_pairs=len(merged), rho=rho, p=p))
        print(f'M{mouse}D{day}: test1 rho={rho:.3f} p={p:.3f} n_pairs={len(merged)}', flush=True)

        t2 = test2(merged, coords, mod, rng)
        if t2 is not None:
            t2['mouse'] = mouse; t2['day'] = day
            t2_rows.append(t2)
            diff = (t2.mean_own - t2.mean_other)
            print(f'  test2 own-other by dist bin: {[round(x,4) for x in diff]}', flush=True)

        t3 = test3(merged, coords, mod, rng)
        if t3 is not None and len(t3) >= 3:
            t3['mouse'] = mouse; t3['day'] = day
            t3['specificity_z'] = (t3.specificity - t3.specificity.mean()) / (t3.specificity.std() + 1e-9)
            t3['anat_pref_z'] = (t3.anat_pref - t3.anat_pref.mean()) / (t3.anat_pref.std() + 1e-9)
            t3_rows.append(t3)
            rho3, p3 = spearmanr(t3.specificity, t3.anat_pref)
            print(f'  test3 specificity-vs-anatomy rho={rho3:.3f} p={p3:.3f} n_cells={len(t3)}', flush=True)

    t1_df = pd.DataFrame(t1_rows)
    t1_df.to_csv(f'{OUT_DIR}/nongrid_test1_distance.csv', index=False)
    t2_df = pd.concat(t2_rows, ignore_index=True) if t2_rows else pd.DataFrame()
    t2_df.to_csv(f'{OUT_DIR}/nongrid_test2_matched.csv', index=False)
    t3_df = pd.concat(t3_rows, ignore_index=True) if t3_rows else pd.DataFrame()
    t3_df.to_csv(f'{OUT_DIR}/nongrid_test3_specificity.csv', index=False)

    print('\n=== POOLED SUMMARY ===', flush=True)
    if len(t1_df):
        n_sig = (t1_df['p'] < 0.05).sum()
        print(f'Test1: {n_sig}/{len(t1_df)} sessions significant, '
              f'mean rho={t1_df.rho.mean():.3f}', flush=True)
    if len(t2_df):
        pooled_diff = t2_df.mean_own - t2_df.mean_other
        stat, p2 = wilcoxon(pooled_diff)
        print(f'Test2: pooled own-other diff mean={pooled_diff.mean():.4f}, '
              f'wilcoxon p={p2:.4f} (n_session-bins={len(t2_df)})', flush=True)
    if len(t3_df):
        rho3_pool, p3_pool = spearmanr(t3_df.specificity_z, t3_df.anat_pref_z)
        print(f'Test3: pooled (z-scored within session) rho={rho3_pool:.3f} p={p3_pool:.4f} '
              f'(n_cells={len(t3_df)})', flush=True)
    print('NONGRID_TESTS_DONE', flush=True)


if __name__ == '__main__':
    main()
