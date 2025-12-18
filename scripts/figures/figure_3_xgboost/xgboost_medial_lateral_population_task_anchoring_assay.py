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
import itertools 
import warnings
warnings.filterwarnings('ignore')

def compute_task_anchored_labels(cluster_id, tcs, tl, bs, last_ephys_bin, peak_indices=[12, 28, 44, 60, 76], sigma=2.5):
    """
    Compute task-anchored labels for a given grid cell using spectral analysis and k-means anchoring.
    Returns the task-anchored label array (same length as tc, up to last_ephys_bin).
    """
    if cluster_id not in tcs:
        return None
    # Spectrogram and peak detection
    tc = tcs[cluster_id]
    tcs_to_use = {cluster_id: tc}
    tcs_to_use[1000] = tc
    results = spectral_analysis(tcs_to_use, tl, bs=bs)
    spectrograms = results[3]
    S = spectrograms.mean(0)
    max_peaks = np.argmax(S, axis=0)
    labels = np.isin(max_peaks, peak_indices).astype(int)
    # Smooth and trim to ephys bins
    tc = gaussian_filter(np.nan_to_num(tc).astype(np.float64), sigma=sigma)
    tc = tc[:last_ephys_bin]
    # Task-anchored labels
    task_anchored_labels = get_kmeans_spatial_labels(tc, labels, bs=bs, tl=tl)
    return task_anchored_labels

'''
This script performs an xgboost assay to assess the influence of grid cells or non grid cells 
on the encoding of a reference cell in a VR environment with respect to the shank location of the NGS cells.
It uses a subset of grid cells and non-grid spatial cells to predict the reference cells firing 
based on their activity, position and the history of their activity and position. 
The results are saved in a YAML file for further analysis.

This assay is optimised for recordings which were recorded in the multishank mode.
'''

use_parser = True

source_path = '/Users/harryclark/Downloads/COHORT12/'
data_path = '/Users/harryclark/Documents/data/'
fig_path = '/Users/harryclark/Documents/figs/FIGURE1/'
mouse = 26
day = 19

if use_parser: 
    parser = ArgumentParser()
    parser.add_argument('--mouse', type=int, required=True, help='Mouse ID')
    parser.add_argument('--day', type=int, required=True, help='Day of recording')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data directory')
    args = parser.parse_args()

    mouse = args.mouse
    day = args.day
    data_path = args.data_path
    source_path = '/exports/eddie/scratch/hclark3/COHORT12/'
 

# xgboost parameters 
nfilters = 5 # number of features to represent the covariate history per covariate
history_length = 1000 # in ms

locations = pd.read_csv(f'{source_path}/all_cluster_brain_locations_chris.csv')
locations = locations[(locations['mouse'] == mouse) & (locations['day'] == day)]
locations['coord_SCs_x'] = locations['coord_SCs_x'] *-1  # correct for the flipped SCs x coordinates
gcs, ngs, ns, sc, ngs_ns, all = cell_classification_of1(mouse, day, percentile_threshold=95, source_path=source_path, verbose=False) # subset

# Assign shank IDs to NGS cells
gcs = reconstruct_shank_id(gcs, mouse, colname='probe_x')
ngs = reconstruct_shank_id(ngs, mouse, colname='probe_x')

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
    speed_in_time = np.array(filled_series)

# create the reference cell population cluster ids
target_cluster_ids = gcs.cluster_id.values.astype(int)

# set up xgboost history model
xgb_history = MLencoding(tunemodel = 'xgboost', cov_history = True, spike_history=False, 
                         window = time_bs, n_filters = nfilters, max_time = history_length)

print('Starting XGBoost shank-based MEC assay...')
print('Number of target grid cells:', len(target_cluster_ids))

# loop over all grid cells in a session
results_df = pd.DataFrame()
for target_id in target_cluster_ids:
    task_anchored_labels = compute_task_anchored_labels(cluster_id=target_id, 
                                                        tcs=tcs, 
                                                        tl=tl, 
                                                        bs=bs, 
                                                        last_ephys_bin=last_ephys_bin)
    
    # get trial ta labels but in time
    trial_labels_in_time = np.zeros(len(trial_number_in_time), dtype=int)
    trial_labels_trial_numbers = np.arange(trial_number_in_time[0], len(task_anchored_labels)+trial_number_in_time[0], 1)
    for i, tn in enumerate(trial_number_in_time.astype(int)):
        trial_labels_in_time[i] = task_anchored_labels[np.where(trial_labels_trial_numbers == tn)[0][0]].astype(int)
  
    target_location = locations[(locations['cluster_id'] == target_id)]
    target_shank = gcs[gcs['cluster_id'] == target_id]['shank_id'].iloc[0]
    print(f'Processing target cell {target_id} (shank {target_shank})')

    # Only use shank 0, 1, 2, 3, and all shanks as covariate sets
    shank_list = [0, 1, 2, 3]
    shank_combos = [[s] for s in shank_list] + [shank_list]

    for shank_combo in shank_combos:
        shank_combo = np.array(shank_combo)
        if len(shank_combo) == 1:
            shank_combo_label = str(shank_combo[0])
        else:
            shank_combo_label = 'all'

        # Get NGS cells from these shanks
        covariate_cluster_ids = ngs[ngs['shank_id'].isin(shank_combo)]['cluster_id'].values.astype(int)
        cov_clusters_df = all[all.cluster_id.isin(covariate_cluster_ids)]

        # Determine shank relation: 'same', 'medial', 'lateral', or 'mixed'
        if len(shank_combo) == 1:
            if shank_combo[0] == target_shank:
                shank_relation = 'same'
            elif shank_combo[0] < target_shank:
                shank_relation = 'medial'
            elif shank_combo[0] > target_shank:
                shank_relation = 'lateral'
            else:
                shank_relation = 'unknown'
        else:
            # For combos, check if all are medial, all lateral, all same, or mixed
            if np.all(shank_combo == target_shank):
                shank_relation = 'same'
            elif np.all(shank_combo < target_shank):
                shank_relation = 'medial'
            elif np.all(shank_combo > target_shank):
                shank_relation = 'lateral'
            else:
                shank_relation = 'mixed'

        if len(covariate_cluster_ids) > 0:
            cov_tcs_time = {cluster_id: tcs_time[cluster_id] for cluster_id in cov_clusters_df.cluster_id if cluster_id in tcs_time}
            all_x = np.vstack(list(cov_tcs_time.values())).T if len(cov_tcs_time) > 0 else np.empty((len(pos_in_time), 0))

            for use_cells, include_position in zip([False, True, True], [True, True, False]):
                x = all_x.copy()
                if include_position:
                    x = np.column_stack((pos_in_time, x)) if x.shape[1] > 0 else pos_in_time.reshape(-1, 1)
                if (use_cells == False):
                    x = pos_in_time.reshape(-1, 1)
                    n_cells = 0
                else:
                    n_cells = len(covariate_cluster_ids)

                y = np.array(tcs_time[target_id])
                Y_hat, pR2_cv = xgb_history.fit_cv(x, y, verbose=0, continuous_folds=True)

                # For each unique task-anchored label, compute mask, pr2, and proportion
                for label in np.unique(trial_labels_in_time):
                    mask = (trial_labels_in_time == label)
                    if np.sum(mask) == 0:
                        continue
                    pr2 = poisson_pseudoR2(y[mask], Y_hat[mask], ynull=np.nanmean(y[mask]))
                    prop = np.sum(mask) / len(mask)
                    print(f'Cell {target_id} on shank {target_shank}, using shanks {shank_combo_label}, shank relation:{shank_relation}') 
                    print(f'label {label}, pr2 {pr2:.4f}, prop {prop:.4f}, pos={include_position}, ncells={n_cells}')
                    
                    tmp = pd.DataFrame()
                    avg_pR2 = np.nanmean(pR2_cv)
                    tmp['target_cluster_id'] = [target_id]
                    tmp['target_cell_type'] = ['GC']
                    tmp['target_shank'] = [target_shank]
                    tmp['covariate_shanks'] = [shank_combo_label]
                    tmp['covariate_shank_relation'] = [shank_relation]
                    tmp['covariate_cell_type'] = ['NGS']
                    tmp['n_covariate_cells'] = [n_cells]
                    tmp['include_position'] = [include_position]
                    tmp['pR2_cv'] = [avg_pR2]
                    tmp['mouse'] = [mouse]
                    tmp['day'] = [day]
                    tmp['task_anchored_mode'] = [label]
                    tmp['task_anchored_mode_prop'] = [prop]
                    tmp['task_anchored_mode_pr2'] = [pr2]

                    results_df = pd.concat([results_df, tmp], ignore_index=True)
        else:
            print(f'No covariate cells found for shank combo {shank_combo}, skipping.')

results_df.to_pickle(f'{data_path}/xgboost_taskanchoring_shank_assay{mouse}_day{day}.pkl')

print('XGBoost shank-based MEC assay complete.')
print(f'results_df has shape: {results_df.shape}')
print(f'Results saved successfully @ {data_path}/xgboost_taskanchoring_shank_assay{mouse}_day{day}.pkl')