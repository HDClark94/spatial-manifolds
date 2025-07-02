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
from spatial_manifolds.brainrender_helper import *

import warnings
warnings.filterwarnings('ignore')

'''
This script performs an xgboost assay to assess the influence of grid cells or non grid cells 
on the encoding of a reference cell in a VR environment with respect to the distance of the grid cells
or non grid cells to the reference cell.
It uses a subset of grid cells and non-grid spatial cells to predict the reference cells firing 
based on their activity, position and the history of their activity and position. 
The results are saved in a YAML file for further analysis.

This assay is will be optimised for recordings which were recorded in the multishank mode ||||
'''

data_path = '/Users/harryclark/Documents/data/'
fig_path = '/Users/harryclark/Documents/figs/FIGURE1/'
mouse = 25
day = 25
assay_mode = 'GC'         # 'GC' for grid cells, 
                          # 'NGS' for non grid spatial cells
ordering_mode = 'nearest' # 'nearest' for nearest cells first, 
                          # 'farthest' for farthest cells first, 
                          # 'random' for random order

# xgboost parameters 
nfilters = 5 # number of features to represent the covariate history per covariate
history_length = 1000 # in ms

# good examples include 
#mice = [25, 25, 26, 27, 29, 28]
#days = [25, 24, 18, 26, 23, 25]

gcs, ngs, ns, sc, ngs_ns, all = cell_classification_of1(mouse, day, percentile_threshold=99) # subset
rc, rsc, vr_ns = cell_classification_vr(mouse, day)
g_m_ids, g_m_cluster_ids = HDBSCAN_grid_modules(gcs, all, mouse, day, min_cluster_size=3, cluster_selection_epsilon=3, 
                                                figpath=fig_path, curate_with_vr=True, curate_with_brain_region=True) # create grid modules using HDBSCAN    

plot_grid_modules_rate_maps(gcs, g_m_ids, g_m_cluster_ids, mouse, day, figpath=fig_path)


# we now have cluster ids classified into modules, non grid spatial cells and non spatial cells 
# as defined by activity in the open field
g_m_cluster_ids = sorted(g_m_cluster_ids, key=len, reverse=True) 
cluster_ids_by_group = []
cluster_ids_by_group.extend(g_m_cluster_ids) # grid cells by module [0,1,2...]
cluster_ids_by_group.append(ngs.cluster_id.values.tolist()) # non grid spatial [-4]
cluster_ids_by_group.append(ns.cluster_id.values.tolist()) # non spatial cells [-3]
cluster_ids_by_group.append(gcs.cluster_id.values.tolist()) # all grid cells [-2]
cluster_ids_by_group.append(sc.cluster_id.values.tolist()) # speed cells [-1]

# load the behaviour data
tcs, tcs_time, _, last_ephys_bin, beh, clusters = compute_vr_tcs(mouse,day, apply_zscore=False, apply_guassian_filter=False)
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
n_neurons = np.arange(1, len(cov_cell_population_cluster_ids)+1, 5)
pR2s_grids_comodular = np.zeros((len(n_neurons), len(grid_module_population_cluster_ids)))
pR2s_grids_non_comodular = np.zeros((len(n_neurons), len(grid_non_module_population_cluster_ids)))
pR2s_non_grids = np.zeros((len(n_neurons), len(non_grid_population_cluster_ids)))

# loop over the number of neurons to use in the covariate history and the cell population cluster ids
for test_population_cluster_ids, pR2s, in zip([grid_module_population_cluster_ids, grid_non_module_population_cluster_ids, non_grid_population_cluster_ids],
                                              [pR2s_grids_comodular, pR2s_grids_non_comodular, pR2s_non_grids]):
    for j, n in enumerate(n_neurons):
        print(f'I am going to use only {n} grid cells')

        # loop over cell population cluster ids
        for i, id in enumerate(test_population_cluster_ids):
            print(f'{np.sum(pR2s_grids_comodular!=0)}/{np.size(pR2s_grids_comodular)}, {np.sum(pR2s_grids_non_comodular!=0)}/{np.size(pR2s_grids_non_comodular)}, {np.sum(pR2s_non_grids!=0)}/{np.size(pR2s_non_grids)}')

            # get the SC_x coordinates of the test cell
            SC_x_ref = all[all.cluster_id == id].SC_x.values[0]

            # create the covariate history and remove the test cell if necessary  
            cov_cluster_ids = cov_cell_population_cluster_ids.copy()
            if id in cov_cluster_ids:
                cov_cluster_ids = np.setdiff1d(cov_cluster_ids, id)

            # get the SC_x coordinates of the covariate cells and sort them by distance to the test cell
            cov_clusters_df = all[all.cluster_id.isin(cov_cluster_ids)]
            cov_clusters_df['SC_x_diff'] = np.abs(cov_clusters_df.SC_x - SC_x_ref)
            if ordering_mode == 'nearest':
                cov_clusters_df = cov_clusters_df.sort_values(by='SC_x_diff', ascending=False)
            elif ordering_mode == 'farthest':
                cov_clusters_df = cov_clusters_df.sort_values(by='SC_x_diff', ascending=True)
            elif ordering_mode == 'random':
                cov_clusters_df = cov_clusters_df.sample(frac=1).reset_index(drop=True)
            
            cov_tcs_time = {cluster_id: tcs_time[cluster_id] for cluster_id in cov_clusters_df.cluster_id if cluster_id in tcs_time}
            all_x = np.vstack(list(cov_tcs_time.values())).T

            # sub select n cells to use in the covariate history based on distance to the test cell
            x = all_x[:, :n]
            
            # add position to the covariate history as well
            x = np.column_stack((pos_in_time, x))

            # get the target variable
            y = np.array(tcs_time[id])

            # fit the model
            Y_hat, pR2_cv = xgb_history.fit_cv(x, y, verbose = 0, continuous_folds = True)
            print(f'pR2_cv = {np.nanmean(pR2_cv)}')
            pR2s[j, i] = np.nanmean(pR2_cv)


# save the results
import yaml
for test_population_cluster_ids, pR2s, label in zip([grid_module_population_cluster_ids, grid_non_module_population_cluster_ids, non_grid_population_cluster_ids],
                                              [pR2s_grids_comodular, pR2s_grids_non_comodular, pR2s_non_grids], ['cmGC', 'ncmGC', 'NGS']):
    # merge test_population_cluster_ids and pR2s into a dictionary
    results = {test_population_cluster_ids[i]: pR2s[:, i] for i in range(len(test_population_cluster_ids))}

    #Convert to YAML-friendly format
    yaml_friendly_results = {str(int(k)): v.tolist() for k, v in results.items()}

    # Save
    with open(f'{data_path}/xgboost_{assay_mode}_{ordering_mode}_distance_assay_mouse{mouse}_day{day}_{label}.yaml', 'w') as f:
        yaml.dump(yaml_friendly_results, f)


