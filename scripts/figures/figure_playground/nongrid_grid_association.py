"""
Quantifies whether non-grid spatial (NGS) cells are "grid-associated": is a
given non-grid cell's coupling (delta-pR2, rich behavioral baseline) to grid
cells specifically elevated above its coupling to OTHER non-grid cells (the
generic-spatial-coupling background), rather than merely above zero?

Per-cell score:  D = mean(dpR2 to grid covariates) - mean(dpR2 to non-grid
                      covariates)
Per-cell significance: permute the grid/non-grid covariate labels (fixed
group sizes) to build a null for D, one-sided p-value (grid > non-grid),
BH-FDR corrected across the whole pooled dataset.

Reuses the same pairwise-assay CSVs and rich-baseline convention as
nongrid_grid_module_tests.py / anatomical_dependence.ipynb. Runs across every
OF1 session with cached pairwise CSVs (no module-split requirement, unlike
the module-specific tests -- this question doesn't need 2 modules).
"""
import glob, os, re, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

DATA_PATH = '/Users/harryclark/Documents/spatial-manifolds/data/eddie/xgboost_pairwise/'
CC_PATH = '/Users/harryclark/Documents/spatial-manifolds/data/cell_classifications.csv'
OUT_DIR = '/Users/harryclark/Documents/spatial-manifolds/data'
HISTORY_LENGTH = 100
SESSION_TYPE = 'OF1'
BASELINE = 'pos_speed_hd_hing_lfp'
N_PERM = 2000
RNG_SEED = 42
MIN_COVARIATES = 3   # need >= this many grid AND >= this many non-grid covariates per target

cc = pd.read_csv(CC_PATH, low_memory=False, keep_default_na=False)
cc['mouse'] = cc['mouse'].astype(int)
cc['day'] = cc['day'].astype(int)
cc['cluster_id'] = cc['cluster_id'].astype(int)


def discover_sessions():
    files = glob.glob(os.path.join(DATA_PATH, f'xgboost_pairwise_M*_D*_h{HISTORY_LENGTH}_{SESSION_TYPE}_*.csv'))
    sessions = set()
    for f in files:
        m = re.search(rf'M(\d+)_D(\d+)_h{HISTORY_LENGTH}_{SESSION_TYPE}_', os.path.basename(f))
        if m:
            sessions.add((int(m.group(1)), int(m.group(2))))
    return sorted(sessions)


def load_pairwise(mouse, day):
    pattern = os.path.join(DATA_PATH, f'xgboost_pairwise_M{mouse}_D{day}_h{HISTORY_LENGTH}_{SESSION_TYPE}_*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return pd.concat([pd.read_csv(f, keep_default_na=False) for f in files], ignore_index=True)


def run_session(mouse, day, rng):
    pw = load_pairwise(mouse, day)
    if pw is None:
        return None
    bl = (pw[(pw.baseline == BASELINE) & (pw.covariate_cluster_id == -1)]
          [['target_cluster_id', 'pR2_cv']].rename(columns={'pR2_cv': 'pR2_baseline'})
          .drop_duplicates('target_cluster_id'))
    cv = (pw[(pw.baseline == BASELINE) & (pw.target_type == 'NGS') &
             (pw.covariate_type.isin(['cmGC', 'ncmGC', 'NGS']))]
          [['target_cluster_id', 'covariate_cluster_id', 'covariate_type', 'pR2_cv']].copy())
    if len(cv) == 0:
        return None
    merged = cv.merge(bl, on='target_cluster_id', how='inner')
    merged['dpr2'] = merged['pR2_cv'] - merged['pR2_baseline']
    merged['is_grid'] = merged.covariate_type.isin(['cmGC', 'ncmGC'])

    rows = []
    for tgt, g in merged.groupby('target_cluster_id'):
        grid_vals = g[g.is_grid].dpr2.values
        nongrid_vals = g[~g.is_grid].dpr2.values
        if len(grid_vals) < MIN_COVARIATES or len(nongrid_vals) < MIN_COVARIATES:
            continue
        D_obs = grid_vals.mean() - nongrid_vals.mean()

        pooled = np.concatenate([grid_vals, nongrid_vals])
        n_grid = len(grid_vals)
        null = np.empty(N_PERM)
        for k in range(N_PERM):
            perm = rng.permutation(pooled)
            null[k] = perm[:n_grid].mean() - perm[n_grid:].mean()
        p = (np.sum(null >= D_obs) + 1) / (N_PERM + 1)  # one-sided: grid > non-grid

        rows.append(dict(mouse=mouse, day=day, target_cluster_id=int(tgt),
                          n_grid=n_grid, n_nongrid=len(nongrid_vals),
                          mean_grid_dpr2=grid_vals.mean(), mean_nongrid_dpr2=nongrid_vals.mean(),
                          D=D_obs, p_value=p))
    return pd.DataFrame(rows) if rows else None


def main():
    sessions = discover_sessions()
    print(f'{len(sessions)} OF1 sessions with pairwise CSVs', flush=True)
    rng = np.random.default_rng(RNG_SEED)

    all_rows = []
    for i, (mouse, day) in enumerate(sessions):
        res = run_session(mouse, day, rng)
        if res is not None:
            all_rows.append(res)
            n_sig_uncorrected = (res.p_value < 0.05).sum()
            print(f'[{i+1}/{len(sessions)}] M{mouse}D{day}: {len(res)} NGS cells tested, '
                  f'{n_sig_uncorrected} nominal p<0.05, mean D={res.D.mean():.4f}', flush=True)
        else:
            print(f'[{i+1}/{len(sessions)}] M{mouse}D{day}: skipped', flush=True)

    if not all_rows:
        print('No sessions produced results.', flush=True)
        return
    df = pd.concat(all_rows, ignore_index=True)
    df['q_value'] = multipletests(df.p_value, method='fdr_bh')[1]
    df.to_csv(f'{OUT_DIR}/nongrid_grid_association_results.csv', index=False)

    n_total = len(df)
    n_sig_fdr = (df.q_value < 0.05).sum()
    print(f'\n=== POOLED SUMMARY (n={n_total} non-grid cells across {len(all_rows)} sessions) ===', flush=True)
    print(f'mean D (grid - nongrid dpR2) = {df.D.mean():.4f}, median = {df.D.median():.4f}', flush=True)
    print(f'{n_sig_fdr}/{n_total} ({100*n_sig_fdr/n_total:.1f}%) non-grid cells are '
          f'grid-associated at q<0.05 (BH-FDR corrected)', flush=True)
    print(f'{(df.p_value<0.05).sum()}/{n_total} ({100*(df.p_value<0.05).sum()/n_total:.1f}%) '
          f'at nominal p<0.05 (uncorrected, for comparison)', flush=True)
    print('GRID_ASSOCIATION_DONE', flush=True)


if __name__ == '__main__':
    main()
