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
XGBoost Cell Number Assay — Open Field 1
------------------------------------------------------------------
Same as xgboost_cell_number_assay_extra.py but uses open field (OF1) data instead of VR.

Key differences from the VR version:
    - Position is 2D (pos_x, pos_y) instead of 1D track position.
    - Uses speed (S), head direction (H), and head-in-goal angle (Hing) from the
      behavior file instead of scalar speed alone.
    - LFP/theta is loaded for the OF1 session if available; zeros used otherwise.
    - Baseline covariate names reflect 2D position and S/H/Hing.

Resulting DataFrame columns:
    - mouse, day: session identifiers
    - target_cluster_id: reference cell ID
    - covariate_type: 'cmGC', 'ncmGC', 'NGS', or 'GC'
    - baseline: name of baseline covariate set (e.g. 'pos', 'speed', 'hd', 'hing', ...)
    - n_covariate_cells: number of covariate cells used (0 for baseline)
    - covariate_cell_type: 'GC' or 'NGS' (type of covariate cells used)
    - pR2_cv: cross-validated pseudo-R²
"""


use_parser = True

source_path = '/Users/harryclark/Downloads/COHORT12/'
data_path = '/Users/harryclark/Documents/data/'
fig_path = '/Users/harryclark/Documents/figs/FIGURE1/'
mouse = 20
day = 14
cell_start = 140
cell_end = 150

# xgboost parameters
nfilters = 5
history_length = 30
max_cells = 10

if use_parser:
    parser = ArgumentParser()
    parser.add_argument('--mouse', type=int, required=True, help='Mouse ID')
    parser.add_argument('--day', type=int, required=True, help='Day of recording')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data directory')
    parser.add_argument('--cell_start', type=int, required=True, help='Start index of target cells (inclusive)')
    parser.add_argument('--cell_end', type=int, required=True, help='End index of target cells (exclusive)')
    parser.add_argument('--history_length', type=int, default=1000, help='History length in ms')
    parser.add_argument('--nfilters', type=int, default=5, help='Number of history filters per covariate')
    parser.add_argument('--max_cells', type=int, default=10, help='Max number of covariate cells to add')
    args = parser.parse_args()

    mouse = args.mouse
    day = args.day
    data_path = args.data_path
    cell_start = args.cell_start
    cell_end = args.cell_end
    nfilters = args.nfilters
    history_length = args.history_length
    max_cells = args.max_cells
    source_path = '/exports/eddie/scratch/hclark3/COHORT12/'

print(f"Running XGBoost Cell Number Assay (OF1) for Mouse {mouse} Day {day} on cells {cell_start} to {cell_end-1} (history={history_length}ms, nfilters={nfilters})")

gcs, ngs, all = classify_cells_both_sessions(mouse, day, percentile_threshold=95, source_path=source_path)
g_m_ids, g_m_cluster_ids, _ = HDBSCAN_grid_modules(gcs, all, mouse, day, min_cluster_size=3, cluster_selection_epsilon=3,
                                                figpath=fig_path, curate_with_vr=True, curate_with_brain_region=True, source_path=source_path)

g_m_cluster_ids = sorted(g_m_cluster_ids, key=len, reverse=True)
cluster_ids_by_group = []
cluster_ids_by_group.extend(g_m_cluster_ids)
cluster_ids_by_group.append(ngs.cluster_id.values.tolist())
cluster_ids_by_group.append(gcs.cluster_id.values.tolist())
cluster_ids_by_group.append(gcs.cluster_id.values.tolist())

# load the open field behaviour data
tcs, tcs_time, beh_OF, clusters_OF, ep = compute_of_tcs(mouse, day, apply_zscore=False, apply_guassian_filter=False, source_path=source_path, session='OF1')
last_ephys_time_bin = clusters_OF[clusters_OF.index[0]].count(bin_size=time_bs, time_units='ms').index[-1]

ep = nap.IntervalSet(start=0, end=last_ephys_time_bin, time_units='s')

# 2D position (x, y)
pos_x_in_time = np.array(beh_OF['P_x'].bin_average(bin_size=time_bs, time_units='ms', ep=ep))
pos_y_in_time = np.array(beh_OF['P_y'].bin_average(bin_size=time_bs, time_units='ms', ep=ep))

if np.any(np.isnan(pos_x_in_time)):
    pos_x_in_time = np.array(pd.Series(pos_x_in_time).ffill().bfill())
if np.any(np.isnan(pos_y_in_time)):
    pos_y_in_time = np.array(pd.Series(pos_y_in_time).ffill().bfill())

# speed, head direction, head-in-goal angle
speed_in_time = np.array(beh_OF['S'].bin_average(bin_size=time_bs, time_units='ms', ep=ep))
hd_in_time = np.array(beh_OF['H'].bin_average(bin_size=time_bs, time_units='ms', ep=ep))
hing_in_time = np.array(beh_OF['Hing'].bin_average(bin_size=time_bs, time_units='ms', ep=ep))

if np.any(np.isnan(speed_in_time)):
    speed_in_time = np.array(pd.Series(speed_in_time).ffill().bfill())
if np.any(np.isnan(hd_in_time)):
    hd_in_time = np.array(pd.Series(hd_in_time).ffill().bfill())
if np.any(np.isnan(hing_in_time)):
    hing_in_time = np.array(pd.Series(hing_in_time).ffill().bfill())


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
xgb_history = MLencoding(tunemodel='xgboost', cov_history=True, spike_history=False,
                         window=time_bs, n_filters=nfilters, max_time=history_length)

results_rows = []

def get_covariate_type(target_id, target_type, covariate_cell_type, g_m_cluster_ids, gcs, ngs):
    if target_type == 'GC':
        in_module = any(target_id in module for module in g_m_cluster_ids)
        if covariate_cell_type == 'GC':
            return 'cmGC' if in_module else 'ncmGC'
        elif covariate_cell_type == 'NGS':
            return 'NGS'
    elif target_type == 'NGS':
        return covariate_cell_type
    elif target_type == 'NS':
        return covariate_cell_type
    return 'unknown'

all_target_cell_ids = np.array(all.cluster_id.values)

if cell_end is None:
    cell_end = len(all_target_cell_ids)
target_cells_batch = all_target_cell_ids[cell_start:cell_end]

for idx, id in enumerate(target_cells_batch):
    print(f'Processing reference cell {id} ({cell_start+idx+1}/{cell_end})')

    y = np.array(tcs_time[id])
    T = len(y)

    # Theta (LFP) trace — try loading for OF1, fall back to zeros
    try:
        theta = get_theta_trace(
            mouse=mouse,
            day=day,
            cluster_id=id,
            time_bs=50,
            resample_bs=time_bs,
            vr_type='OF1',
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

    # Align position, speed, hd, hing to length T
    pos_x = pos_x_in_time[:T]
    pos_y = pos_y_in_time[:T]
    speed = speed_in_time[:T]
    hd    = hd_in_time[:T]
    hing  = hing_in_time[:T]
    if len(pos_x) < T:
        pos_x = np.pad(pos_x, (0, T - len(pos_x)), mode='constant')
    if len(pos_y) < T:
        pos_y = np.pad(pos_y, (0, T - len(pos_y)), mode='constant')
    if len(speed) < T:
        speed = np.pad(speed, (0, T - len(speed)), mode='constant')
    if len(hd) < T:
        hd = np.pad(hd, (0, T - len(hd)), mode='constant')
    if len(hing) < T:
        hing = np.pad(hing, (0, T - len(hing)), mode='constant')

    # Baselines: all covariate combos of pos(x,y) / speed / hd / hing / lfp
    BASELINE_COV_NAMES = [
        'pos', 'speed', 'hd', 'hing', 'lfp',
        'pos_speed', 'pos_hd', 'pos_hing', 'pos_lfp',
        'pos_speed_hd', 'pos_speed_hing', 'pos_speed_lfp',
        'pos_hd_hing', 'pos_speed_hd_hing',
        'pos_speed_hd_hing_lfp',
    ]
    baseline_xs = [
        np.column_stack((pos_x, pos_y)),                              # pos
        speed[:, None],                                                # speed
        hd[:, None],                                                   # hd
        hing[:, None],                                                 # hing
        theta[:, None],                                                # lfp
        np.column_stack((pos_x, pos_y, speed)),                        # pos_speed
        np.column_stack((pos_x, pos_y, hd)),                           # pos_hd
        np.column_stack((pos_x, pos_y, hing)),                         # pos_hing
        np.column_stack((pos_x, pos_y, theta)),                        # pos_lfp
        np.column_stack((pos_x, pos_y, speed, hd)),                    # pos_speed_hd
        np.column_stack((pos_x, pos_y, speed, hing)),                  # pos_speed_hing
        np.column_stack((pos_x, pos_y, speed, theta)),                 # pos_speed_lfp
        np.column_stack((pos_x, pos_y, hd, hing)),                     # pos_hd_hing
        np.column_stack((pos_x, pos_y, speed, hd, hing)),              # pos_speed_hd_hing
        np.column_stack((pos_x, pos_y, speed, hd, hing, theta)),       # pos_speed_hd_hing_lfp
    ]

    for b_idx, x_b in enumerate(baseline_xs):
        Y_hat_b, pR2_cv_b = xgb_history.fit_cv(x_b, y, verbose=0, continuous_folds=True)
        if id in gcs.cluster_id.values:
            target_type = 'GC'
        elif id in ngs.cluster_id.values:
            target_type = 'NGS'
        else:
            target_type = 'NS'

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

    # Cell models: BASELINE + n covariate cells
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
        all_x = np.vstack(list(cov_tcs_time.values())).T[:T]

        n_neurons_nonzero = np.arange(1, min(max_cells+1, len(cov_cluster_ids)+1))

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
                    target_type = 'NS'
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


results_df = pd.DataFrame(results_rows)
suffix = f'_{cell_start}_{cell_end}'
csv_path = f'{data_path}/xgboost_cell_number_assay_extra_of_M{mouse}_D{day}_h{history_length}{suffix}.csv'
results_df.to_csv(csv_path, index=False)
print(f'Saved results DataFrame to {csv_path}')
