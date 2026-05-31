import numpy as np
import pandas as pd
import pynapple as nap
from matplotlib import pyplot as plt
from spatial_manifolds.mlencoding import *
from spatial_manifolds.detect_grids import *
from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore')


"""
XGBoost Cell Number Assay (pR², batched cell runs)
------------------------------------------------------------------
This script fits XGBoost models to predict the firing of reference cells (grid or non-grid spatial)
in VR, using different covariate sets, and is designed to be run in batches (e.g. 20 cells per run) for time-limited cluster jobs.

Key features:
    - Accepts a range of target cell indices (e.g. --cell_start 0 --cell_end 20) to process only a subset of cells per run.
    - Loops over both 'GC' and 'NGS' covariate cell types for each reference cell.
    - Appends a suffix to the output CSV indicating the cell index range (e.g. _0_20.csv).
    - For each reference cell, fits baseline models (pos, speed, lfp, etc) and cell models (baseline + N covariate cells), recording cross-validated pseudo-R² (pR²) for each fit.

Resulting DataFrame columns:
    - mouse, day: session identifiers
    - target_cluster_id: reference cell ID
    - covariate_type: 'cmGC', 'ncmGC', 'NGS', or 'GC' (see logic)
    - baseline: name of baseline covariate set (e.g. 'pos', 'pos_speed', ...)
    - n_covariate_cells: number of covariate cells used (0 for baseline)
    - covariate_cell_type: 'GC' or 'NGS' (type of covariate cells used)
    - pR2_cv: cross-validated pseudo-R²

Each row represents a single XGBoost fit for a given target cell and covariate set.
Run this script multiple times with different cell index ranges to cover all cells in a session.
"""
 

use_parser = True

source_path = '/Users/harryclark/Downloads/COHORT12/'
data_path = '/Users/harryclark/Documents/data/'
fig_path = '/Users/harryclark/Documents/figs/FIGURE1/'
mouse = 25
day = 24
cell_start = 0
cell_end = 20

if use_parser:
    parser = ArgumentParser()
    parser.add_argument('--mouse', type=int, required=True, help='Mouse ID')
    parser.add_argument('--day', type=int, required=True, help='Day of recording')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data directory')
    parser.add_argument('--cell_start', type=int, required=True, help='Start index of target cells (inclusive)')
    parser.add_argument('--cell_end', type=int, required=True, help='End index of target cells (exclusive)')
    args = parser.parse_args()

    mouse = args.mouse
    day = args.day
    data_path = args.data_path
    cell_start = args.cell_start
    cell_end = args.cell_end
    source_path = '/exports/eddie/scratch/hclark3/COHORT12/'


# xgboost parameters 
nfilters = 5 # number of features to represent the covariate history per covariate
history_length = 1000 # in ms

# good examples include 
#mice = [25, 25, 26, 27, 29, 28]
#days = [25, 24, 18, 26, 23, 25]

gcs, ngs, all = classify_cells_both_sessions(mouse, day, percentile_threshold=95, source_path=source_path)
g_m_ids, g_m_cluster_ids, _ = HDBSCAN_grid_modules(gcs, all, mouse, day, min_cluster_size=3, cluster_selection_epsilon=3, 
                                                figpath=fig_path, curate_with_vr=True, curate_with_brain_region=True, source_path=source_path) # create grid modules using HDBSCAN    

# we now have cluster ids classified into modules, non grid spatial cells and non spatial cells 
# as defined by activity in the open field
g_m_cluster_ids = sorted(g_m_cluster_ids, key=len, reverse=True) 
cluster_ids_by_group = []
cluster_ids_by_group.extend(g_m_cluster_ids) # grid cells by module [0,1,2...]
cluster_ids_by_group.append(ngs.cluster_id.values.tolist()) # non grid spatial [-4]
cluster_ids_by_group.append(gcs.cluster_id.values.tolist()) # all grid cells [-2]
cluster_ids_by_group.append(gcs.cluster_id.values.tolist()) # speed cells [-1]

# load the behaviour data
tcs, tcs_time, _, last_ephys_bin, beh, clusters = compute_vr_tcs(mouse,day, apply_zscore=False, apply_guassian_filter=False, source_path=source_path)
last_ephys_time_bin = clusters[clusters.index[0]].count(bin_size=time_bs, time_units = 'ms').index[-1]

# time binned variables for later
ep = nap.IntervalSet(start=0, end=last_ephys_time_bin, time_units = 's')
speed_in_time = np.array(beh['S'].bin_average(bin_size=time_bs, time_units = 'ms', ep=ep))
dt_in_time = np.array(beh['travel'].bin_average(bin_size=time_bs, time_units = 'ms', ep=ep)-((beh['trial_number'][0]-1)*tl))
pos_in_time = dt_in_time%tl
trial_number_in_time = (dt_in_time//tl)+beh['trial_number'][0]

if np.any(np.isnan(pos_in_time)):
    series = pd.Series(dt_in_time)
    filled_series = series.ffill().bfill()
    dt_in_time = np.array(filled_series)
    pos_in_time = dt_in_time%tl
    trial_number_in_time = (dt_in_time//tl)+beh['trial_number'][0]

if np.any(np.isnan(speed_in_time)):
    series = pd.Series(speed_in_time)
    filled_series = series.ffill().bfill()
    speed_in_time = np.array(speed_in_time)

if np.any(np.isnan(trial_number_in_time)):
    series = pd.Series(trial_number_in_time)
    filled_series = series.ffill().bfill()
    trial_number_in_time = np.array(trial_number_in_time)


# create the reference cell population cluster ids
grid_module_population_cluster_ids = np.array(cluster_ids_by_group[0].copy())
grid_non_module_population_cluster_ids = np.setdiff1d(gcs.cluster_id.values, grid_module_population_cluster_ids).astype(int)
non_grid_population_cluster_ids = ngs.cluster_id.values.astype(int).astype(int)


# Prepare covariate cell sets for both GC and NGS
covariate_sets = [
    ('GC', grid_module_population_cluster_ids),
    ('NGS', non_grid_population_cluster_ids)
]

# set up xgboost history model
xgb_history = MLencoding(tunemodel = 'xgboost', cov_history = True, spike_history=False, 
                         window = time_bs, n_filters = nfilters, max_time = history_length)


# --- Results DataFrame: one row per XGBoost fit ---
results_rows = []

# --- Helper function for covariate_type logic ---
def get_covariate_type(target_id, target_type, covariate_cell_type, g_m_cluster_ids, gcs, ngs):
    if target_type == 'GC':
        in_module = any(target_id in module for module in g_m_cluster_ids)
        if covariate_cell_type == 'GC':
            return 'cmGC' if in_module else 'ncmGC'
        elif covariate_cell_type == 'NGS':
            return 'NGS'
    elif target_type == 'NGS':
        return covariate_cell_type
    return 'unknown'

all_target_cells = np.concatenate([
    grid_module_population_cluster_ids,
    grid_non_module_population_cluster_ids,
    non_grid_population_cluster_ids
])

# Apply batching: select only the requested range of target cells
if cell_end is None:
    cell_end = len(all_target_cells)
target_cells_batch = all_target_cells[cell_start:cell_end]

for idx, id in enumerate(target_cells_batch):
    print(f'Processing reference cell {id} ({cell_start+idx+1}/{cell_end})')

    # Get the target variable (reference cell spike train)
    y = np.array(tcs_time[id])
    T = len(y)

    # --- Theta (LFP) trace for this cell's best channel ---
    try:
        theta = get_theta_trace(
            mouse=mouse,
            day=day,
            cluster_id=id,
            time_bs=50,
            resample_bs=time_bs,
            vr_type='VR',
            source_path=source_path,
        )
        theta = np.array(theta)
        if len(theta) < T:
            theta = np.pad(theta, (0, T - len(theta)), mode='constant')
        else:
            theta = theta[:T]
    except Exception as e:
        print(f"    Could not load theta for cluster {id}: {e}. Using zeros.")
        theta = np.zeros(T)

    # Align speed and pos to length T
    pos   = pos_in_time[:T]
    speed = speed_in_time[:T]
    if len(pos) < T:
        pos   = np.pad(pos,   (0, T - len(pos)),   mode='constant')
    if len(speed) < T:
        speed = np.pad(speed, (0, T - len(speed)), mode='constant')

    # --- Baselines: all 7 covariate combos of pos / speed / lfp ---
    BASELINE_COV_NAMES = ['pos', 'speed', 'lfp', 'pos_speed', 'pos_lfp', 'speed_lfp', 'pos_speed_lfp']
    baseline_xs = [
        pos[:, None],                              # pos
        speed[:, None],                            # speed
        theta[:, None],                            # lfp
        np.column_stack((pos, speed)),             # pos_speed
        np.column_stack((pos, theta)),             # pos_lfp
        np.column_stack((speed, theta)),           # speed_lfp
        np.column_stack((pos, speed, theta)),      # pos_speed_lfp
    ]

    for b_idx, x_b in enumerate(baseline_xs):
        Y_hat_b, pR2_cv_b = xgb_history.fit_cv(x_b, y, verbose=0, continuous_folds=True)
        # Determine target_type for this cell
        if id in gcs.cluster_id.values:
            target_type = 'GC'
        elif id in ngs.cluster_id.values:
            target_type = 'NGS'
        else:
            target_type = 'unknown'
        # For baseline fits, covariate_cell_type is not meaningful, but we loop over both for consistency
        for covariate_cell_type, _ in covariate_sets:
            cov_type = get_covariate_type(id, target_type, covariate_cell_type, g_m_cluster_ids, gcs, ngs)
            print(f'  baseline [{BASELINE_COV_NAMES[b_idx]}] pR2 = {np.nanmean(pR2_cv_b):.4f} covariate_type={cov_type} covariate_cell_type={covariate_cell_type}')
            results_rows.append(dict(
                mouse=mouse,
                day=day,
                target_cluster_id=id,
                covariate_type=cov_type,
                baseline=BASELINE_COV_NAMES[b_idx],
                n_covariate_cells=0,
                covariate_cell_type=covariate_cell_type,
                pR2_cv=float(np.nanmean(pR2_cv_b)),
            ))

    # --- Cell models: BASELINE + n covariate cells ---
    SC_x_ref = all[all.cluster_id == id].SC_x.values[0]
    SC_y_ref = all[all.cluster_id == id].SC_y.values[0]

    for covariate_cell_type, cov_cell_population_cluster_ids in covariate_sets:
        cov_cluster_ids = cov_cell_population_cluster_ids.copy()
        if id in cov_cluster_ids:
            cov_cluster_ids = np.setdiff1d(cov_cluster_ids, id)

        cov_clusters_df = all[all.cluster_id.isin(cov_cluster_ids)].copy()
        cov_clusters_df['SC_diff'] = np.sqrt(
            (cov_clusters_df.SC_x - SC_x_ref)**2 + (cov_clusters_df.SC_y - SC_y_ref)**2
        )
        cov_clusters_df = cov_clusters_df.sample(frac=1).reset_index(drop=True)

        cov_tcs_time = {
            cluster_id: tcs_time[cluster_id]
            for cluster_id in cov_clusters_df.cluster_id
            if cluster_id in tcs_time
        }
        if len(cov_tcs_time) == 0:
            continue
        all_x = np.vstack(list(cov_tcs_time.values())).T[:T]  # shape (T, n_available)

        n_neurons_nonzero = np.arange(1, min(11, len(cov_cluster_ids)+1))

        for b_idx, x_b in enumerate(baseline_xs):
            for j, n in enumerate(n_neurons_nonzero):
                np.random.seed(j)
                x = np.column_stack((x_b, all_x[:, :n]))
                Y_hat, pR2_cv = xgb_history.fit_cv(x, y, verbose=0, continuous_folds=True)
                if id in gcs.cluster_id.values:
                    target_type = 'GC'
                elif id in ngs.cluster_id.values:
                    target_type = 'NGS'
                else:
                    target_type = 'unknown'
                cov_type = get_covariate_type(id, target_type, covariate_cell_type, g_m_cluster_ids, gcs, ngs)
                print(f'  n_cells={n} pR2 = {np.nanmean(pR2_cv):.4f} covariate_type={cov_type} covariate_cell_type={covariate_cell_type}')
                results_rows.append(dict(
                    mouse=mouse,
                    day=day,
                    target_cluster_id=id,
                    covariate_type=cov_type,
                    baseline=BASELINE_COV_NAMES[b_idx],
                    n_covariate_cells=int(n),
                    covariate_cell_type=covariate_cell_type,
                    pR2_cv=float(np.nanmean(pR2_cv)),
                ))


# --- Save results DataFrame as CSV ---
results_df = pd.DataFrame(results_rows)
suffix = f'_{cell_start}_{cell_end}'
csv_path = f'{data_path}/xgboost_cell_number_assay_extra_M{mouse}_D{day}{suffix}.csv'
results_df.to_csv(csv_path, index=False)
print(f'Saved results DataFrame to {csv_path}')

# ---
# Each row in results_df represents a single XGBoost fit for a given target cell and covariate set:
#   - Baseline fits (covariate_type='baseline') have n_covariate_cells=0 and specify the baseline set.
#   - Cell fits (covariate_type='cell') have n_covariate_cells > 0 and specify the baseline set used.
#   - Columns:
#       mouse, day: session
#       target_cluster_id: reference cell
#       assay_mode: 'GC' or 'NGS' (covariate cell type)
#       covariate_type: 'baseline' or 'cell'
#       baseline: covariate set name
#       n_covariate_cells: number of covariate cells used
#       covariate_cell_type: type of covariate cells
#       pR2_cv: cross-validated pseudo-R²


