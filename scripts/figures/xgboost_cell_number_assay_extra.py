import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import pynapple as nap
from spatial_manifolds.toroidal import *
from spatial_manifolds.behaviour_plots import *
from spatial_manifolds.mlencoding import *
from spatial_manifolds.circular_decoder import circular_decoder, cross_validate_decoder, cross_validate_decoder_time, circular_nanmean
from spatial_manifolds.data.curation import curate_clusters
from scipy.stats import zscore
from spatial_manifolds.util import gaussian_filter_nan
from spatial_manifolds.predictive_grid import compute_travel_projected, wrap_list
from spatial_manifolds.behaviour_plots import *
from spatial_manifolds.detect_grids import *
from argparse import ArgumentParser

import warnings
warnings.filterwarnings('ignore')

import yaml

'''
This script performs an xgboost assay to assess the influence of grid cells or non grid cells 
on the encoding of a reference cell in a VR environment with respect to the distance of the grid cells
or non grid cells to the reference cell.
It uses a subset of grid cells and non-grid spatial cells to predict the reference cells firing 
based on their activity, position and the history of their activity and position. 
The results are saved in a YAML file for further analysis.

This assay is will be optimised for recordings which were recorded in the multishank mode ||||
'''
 
use_parser = True

source_path = '/Users/harryclark/Downloads/COHORT12/'
data_path = '/Users/harryclark/Documents/data/'
fig_path = '/Users/harryclark/Documents/figs/FIGURE1/'
mouse = 25
day = 24
assay_mode = 'GC'         # 'GC' for grid cells, 
                          # 'NGS' for non grid spatial cells

if use_parser:
    parser = ArgumentParser()
    parser.add_argument('--mouse', type=int, required=True, help='Mouse ID')
    parser.add_argument('--day', type=int, required=True, help='Day of recording')
    parser.add_argument('--assay_mode', type=str, required=True, help='Assay mode: GC or NGS')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data directory')
    args = parser.parse_args()

    mouse = args.mouse
    day = args.day
    assay_mode = args.assay_mode
    data_path = args.data_path
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

# set the covariate cell population cluster ids based on the assay mode
if assay_mode == 'GC':
    cov_cell_population_cluster_ids = grid_module_population_cluster_ids
elif assay_mode == 'NGS':
    cov_cell_population_cluster_ids = non_grid_population_cluster_ids

# set up xgboost history model
xgb_history = MLencoding(tunemodel = 'xgboost', cov_history = True, spike_history=False, 
                         window = time_bs, n_filters = nfilters, max_time = history_length)

# All 7 covariate combinations of pos / speed / lfp (used for baselines, n=0)
BASELINE_COV_NAMES = ['pos', 'speed', 'lfp', 'pos_speed', 'pos_lfp', 'speed_lfp', 'pos_speed_lfp']
N_BASELINE_COVS = len(BASELINE_COV_NAMES)

# n_neurons_nonzero: cell conditions only (n > 0)
n_neurons_nonzero = np.arange(1, len(cov_cell_population_cluster_ids), 2)

# baseline_pR2s: shape (n_conditions, N_BASELINE_COVS, n_cells)
# cell_pR2s:     shape (n_conditions, len(n_neurons_nonzero), n_cells)
# last condition row (-1) stores mean pR2 across all trials
N_CONDITIONS = 16

baseline_pR2s_comodular    = np.full((N_CONDITIONS, N_BASELINE_COVS, len(grid_module_population_cluster_ids)), np.nan)
baseline_pR2s_non_comodular = np.full((N_CONDITIONS, N_BASELINE_COVS, len(grid_non_module_population_cluster_ids)), np.nan)
baseline_pR2s_non_grids    = np.full((N_CONDITIONS, N_BASELINE_COVS, len(non_grid_population_cluster_ids)), np.nan)

cell_pR2s_comodular    = np.full((N_CONDITIONS, len(n_neurons_nonzero), len(grid_module_population_cluster_ids)), np.nan)
cell_pR2s_non_comodular = np.full((N_CONDITIONS, len(n_neurons_nonzero), len(grid_non_module_population_cluster_ids)), np.nan)
cell_pR2s_non_grids    = np.full((N_CONDITIONS, len(n_neurons_nonzero), len(non_grid_population_cluster_ids)), np.nan)

def _condition_pR2s(y, Y_hat, beh, trial_number_in_time):
    """Return length-16 array of per-condition pR2, last entry = mean across all trials."""
    out = np.full(N_CONDITIONS, np.nan)
    c = 0
    for context in ['rz1', 'rz2']:
        for ttype in ['b', 'nb']:
            for perf in ['hit', 'try', 'run', 'slow']:
                trial_numbers = np.array(
                    beh['trials'][
                        (beh['trials']['type'] == ttype) &
                        (beh['trials']['context'] == context) &
                        (beh['trials']['performance'] == perf)
                    ]['number']
                )
                if len(trial_numbers) > 0:
                    mask = np.isin(trial_number_in_time, trial_numbers)
                    out[c] = poisson_pseudoR2(y[mask], Y_hat[mask], ynull=np.nanmean(y))
                c += 1
    out[-1] = np.nanmean(poisson_pseudoR2(y, Y_hat, ynull=np.nanmean(y)))
    return out

# loop over the three cell populations
for test_population_cluster_ids, baseline_pR2s, cell_pR2s, label in zip(
    [grid_module_population_cluster_ids, grid_non_module_population_cluster_ids, non_grid_population_cluster_ids],
    [baseline_pR2s_comodular, baseline_pR2s_non_comodular, baseline_pR2s_non_grids],
    [cell_pR2s_comodular, cell_pR2s_non_comodular, cell_pR2s_non_grids],
    ['cmGC', 'ncmGC', 'NGS'],
):
    for i, id in enumerate(test_population_cluster_ids):
        print(f'[{label}] Processing reference cell {id} ({i+1}/{len(test_population_cluster_ids)})')

        # get the target variable
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
            print(f'  baseline [{BASELINE_COV_NAMES[b_idx]}] pR2 = {np.nanmean(pR2_cv_b):.4f}')
            cond_pR2s = _condition_pR2s(y, Y_hat_b, beh, trial_number_in_time)
            cond_pR2s[-1] = np.nanmean(pR2_cv_b)
            baseline_pR2s[:, b_idx, i] = cond_pR2s

        # --- Cell conditions: pos + speed + theta + n cells ---
        # prepare covariate cell matrix (randomised order, same seed per n index)
        SC_x_ref = all[all.cluster_id == id].SC_x.values[0]
        SC_y_ref = all[all.cluster_id == id].SC_y.values[0]

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
        all_x = np.vstack(list(cov_tcs_time.values())).T[:T]  # shape (T, n_available)

        base_x = np.column_stack((pos, speed, theta))  # pos + speed + theta base

        for j, n in enumerate(n_neurons_nonzero):
            np.random.seed(j)
            x = np.column_stack((base_x, all_x[:, :n]))
            Y_hat, pR2_cv = xgb_history.fit_cv(x, y, verbose=0, continuous_folds=True)
            print(f'  n_cells={n} pR2 = {np.nanmean(pR2_cv):.4f}')
            cond_pR2s = _condition_pR2s(y, Y_hat, beh, trial_number_in_time)
            cond_pR2s[-1] = np.nanmean(pR2_cv)
            cell_pR2s[:, j, i] = cond_pR2s

    # --- Save results for this population ---
    baseline_results = {
        str(int(test_population_cluster_ids[i])): {
            cov_name: baseline_pR2s[:, b_idx, i].tolist()
            for b_idx, cov_name in enumerate(BASELINE_COV_NAMES)
        }
        for i in range(len(test_population_cluster_ids))
    }
    cell_results = {
        str(int(test_population_cluster_ids[i])): {
            f'n{int(n)}': cell_pR2s[:, j, i].tolist()
            for j, n in enumerate(n_neurons_nonzero)
        }
        for i in range(len(test_population_cluster_ids))
    }
    yaml_out = {'baseline_covariates': baseline_results, 'cell_covariates': cell_results,
                'baseline_cov_names': BASELINE_COV_NAMES,
                'n_neurons_nonzero': n_neurons_nonzero.tolist()}

    out_path = f'{data_path}/xgboost_{assay_mode}_cell_number_assay_extra_M{mouse}_D{day}_{label}.yaml'
    with open(out_path, 'w') as f:
        yaml.dump(yaml_out, f)
    print(f'Saved {out_path}')


