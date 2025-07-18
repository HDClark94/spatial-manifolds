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
import time
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

time_start = time.time()

use_parser = True

source_path = '/Users/harryclark/Downloads/COHORT12/'
data_path = '/Users/harryclark/Documents/data/'
fig_path = '/Users/harryclark/Documents/figs/FIGURE1/'
mouse = 25
day = 24
cluster_id = 135 # cluster id of the reference cell

if use_parser:
    parser = ArgumentParser()
    parser.add_argument('--mouse', type=int, required=True, help='Mouse ID')
    parser.add_argument('--day', type=int, required=True, help='Day of recording')
    parser.add_argument('--cluster_id', type=str, required=True, help='The cluster ID of the reference cell')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data directory')
    args = parser.parse_args()

    mouse = args.mouse
    day = args.day
    cluster_id = int(args.cluster_id)
    data_path = args.data_path
    source_path = '/exports/eddie/scratch/hclark3/COHORT12/'


# xgboost parameters 
nfilters = 5 # number of features to represent the covariate history per covariate
history_length = 1000 # in ms

# good examples include 
#mice = [25, 25, 26, 27, 29, 28]
#days = [25, 24, 18, 26, 23, 25]

gcs, ngs, ns, sc, ngs_ns, all = cell_classification_of1(mouse, day, percentile_threshold=95, source_path=source_path) # subset
rc, rsc, vr_ns = cell_classification_vr(mouse, day, source_path=source_path)
g_m_ids, g_m_cluster_ids = HDBSCAN_grid_modules(gcs, all, mouse, day, min_cluster_size=3, cluster_selection_epsilon=3, 
                                                figpath=fig_path, curate_with_vr=True, curate_with_brain_region=True, source_path=source_path) # create grid modules using HDBSCAN    

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


# set up xgboost history model
xgb_history = MLencoding(tunemodel = 'xgboost', cov_history = True, spike_history=False, 
                         window = time_bs, n_filters = nfilters, max_time = history_length)

# create a dataframe to store the results
xgboost_results = pd.DataFrame(columns=['mouse', 
                                        'day', 
                                        'cluster_id', 
                                         'trained_on', 
                                         'tested_on', 
                                         'ordering', 
                                         'n_neurons', 
                                         'pR2_cv', 
                                         'trial_type', 
                                         'trial_context', 
                                         'trial_performance'])
 
# loop over the training conditions
for trained_on in ['GC', 'NGS']:
   for ordering in ['nearest', 'farthest', 'random']:
        
        # set the covariate cell population cluster ids based on the assay mode
        if trained_on == 'GC':
            cov_cell_population_cluster_ids = grid_module_population_cluster_ids
        elif trained_on == 'NGS':
            cov_cell_population_cluster_ids = non_grid_population_cluster_ids

        n_neurons = np.array([0,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,30,40,50,60,70,80,90,100,150,200,250,300,350,400,450,500,600,700,800,900,1000])
        n_neurons = n_neurons[n_neurons <= len(cov_cell_population_cluster_ids)] # remove the conditions where the number of neurons is greater than the number of covariate cells
        
        # assign tested_on based on the trained_on and cluster_id
        if trained_on == 'GC':
            if cluster_id in grid_module_population_cluster_ids:
                tested_on = 'cmGC'
            elif cluster_id in gcs.cluster_id.values:
                tested_on = 'ncmGC'
            elif cluster_id in ngs.cluster_id.values:
                tested_on = 'NGS'
            else:
                tested_on = 'other'

        elif trained_on == 'NGS':
            if cluster_id in gcs.cluster_id.values:
                tested_on = 'GC'
            elif cluster_id in ngs.cluster_id.values:
                tested_on = 'NGS'
            else:
                tested_on = 'other'

        # loop over n_neurons
        for j, n in enumerate(n_neurons):
            # get the SC_x coordinates of the test cell
            SC_x_ref = all[all.cluster_id == cluster_id].SC_x.values[0]
            SC_y_ref = all[all.cluster_id == cluster_id].SC_y.values[0]

            # create the covariate history and remove the test cell if necessary  
            cov_cluster_ids = cov_cell_population_cluster_ids.copy()
            if cluster_id in cov_cluster_ids:
                cov_cluster_ids = np.setdiff1d(cov_cluster_ids, cluster_id)

            # get the SC_x coordinates and Sc_y of the covariate cells and sort them by distance to the test cell
            cov_clusters_df = all[all.cluster_id.isin(cov_cluster_ids)]
            cov_clusters_df['SC_diff'] = np.sqrt((cov_clusters_df.SC_x - SC_x_ref)**2 + (cov_clusters_df.SC_y - SC_y_ref)**2)
                        
            if ordering == 'nearest':
                cov_clusters_df = cov_clusters_df.sort_values(by='SC_diff', ascending=True)
            elif ordering == 'farthest':
                cov_clusters_df = cov_clusters_df.sort_values(by='SC_diff', ascending=False)
            elif ordering == 'random':
                cov_clusters_df = cov_clusters_df.sample(frac=1).reset_index(drop=True)
            
            cov_tcs_time = {id: tcs_time[id] for id in cov_clusters_df.cluster_id if id in tcs_time}
            all_x = np.vstack(list(cov_tcs_time.values())).T

            # sub select n cells to use in the covariate history
            np.random.seed(j)
            if n > 0:
                x = all_x[:, :n]
                # add position to the covariate history as well
                x = np.column_stack((pos_in_time, x))
            else:
                x = pos_in_time.reshape(-1, 1)

            # get the target variable
            y = np.array(tcs_time[cluster_id])

            # fit the model
            Y_hat, pR2_cv = xgb_history.fit_cv(x, y, verbose = 0, continuous_folds = True)
            
            for context in ['rz1', 'rz2']:
                for type in ['b', 'nb']:
                    for performance in ['hit', 'try', 'run', 'slow']:
                        # get the trial numbers for the current condition
                        trial_numbers = np.array(beh['trials'][(beh['trials']['type'] == type) & 
                                                               (beh['trials']['context'] == context) & 
                                                               (beh['trials']['performance'] == performance)]['number'])
                        
                        if len(trial_numbers) != 0:
                            # get the indices of the trials in the tcs_time
                            trial_mask = np.isin(trial_number_in_time, trial_numbers)
                            pR2_condition_cv = poisson_pseudoR2(y[trial_mask], Y_hat[trial_mask], ynull=np.nanmean(y))
                        else:
                            pR2_condition_cv = np.nan
                        
                        # save the results for each trial condition
                        df = pd.DataFrame()
                        df['mouse'] = [mouse]
                        df['day'] = [day]
                        df['order_by'] = [ordering]
                        df['trained_on'] = [trained_on]
                        df['tested_on'] = [tested_on]
                        df['n_neurons'] = [n]
                        df['pR2'] = [pR2_condition_cv]
                        df['cluster_id'] = [cluster_id]
                        df['trial_context'] = [context]
                        df['trial_type'] = [type]
                        df['trial_performance'] = [performance]
                        xgboost_results = pd.concat([xgboost_results, df], ignore_index=True)

            # save the results for the session wide condition'
            df = pd.DataFrame()
            df['mouse'] = [mouse]
            df['day'] = [day]
            df['order_by'] = [ordering]
            df['trained_on'] = [trained_on]
            df['tested_on'] = [tested_on]
            df['n_neurons'] = [n]
            df['pR2'] = [np.nanmean(pR2_cv)]
            df['cluster_id'] = [cluster_id]
            df['trial_context'] = ['session']
            df['trial_type'] = ['session']
            df['trial_performance'] = ['session']
            xgboost_results = pd.concat([xgboost_results, df], ignore_index=True)

            print(f'When trained on {trained_on} and ordered by {ordering}, while testing on cluster {cluster_id}, using {n} neurons + position, pR2_cv = {np.nanmean(pR2_cv)}')


# save results in pkl
xgboost_result_path = f'{source_path}xgboost/M{mouse}_D{day}_C{cluster_id}.pkl'
xgboost_results.to_pickle(xgboost_result_path)


time_end = time.time()
print(f'Time taken: {time_end - time_start} seconds')