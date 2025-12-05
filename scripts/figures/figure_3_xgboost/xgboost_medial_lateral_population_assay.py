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
mouse = 21
day = 15
medial_lateral_boundary = 3400  # um in SCs coordinates

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
gcs, ngs, ns, sc, ngs_ns, all = cell_classification_of1(mouse, day, percentile_threshold=95, source_path=source_path) # subset

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

print('Starting XGBoost medial-lateral MEC lateralisation assay...')
print('Number of target grid cells:', len(target_cluster_ids))

# loop over all grid cells in a session
results_df = pd.DataFrame() 
for target_id in target_cluster_ids:
    
    # determione if the grid cell is medial, middle or lateral MEC (250 um bins) starting at 3000
    target_location = locations[(locations['mouse'] == mouse) & (locations['day'] == day) & (locations['cluster_id'] == target_id)]
    target_ML_position = target_location['coord_SCs_x'].iloc[0]
    if (target_ML_position < medial_lateral_boundary):
        target_pos_label = 'medial_MEC'
    elif (target_ML_position >= medial_lateral_boundary):
        target_pos_label = 'lateral_MEC'
    else:
        continue  # skip cells outside medial and lateral MEC

    print(f'Processing target cell {target_id} located at {target_ML_position} um in {target_pos_label}')

    for covariate_pos_label, cov_bounds in zip(['medial_MEC', 'lateral_MEC'], 
                                               [(-np.inf, medial_lateral_boundary), (medial_lateral_boundary, np.inf)]):
        # get NGS covariate cells based on their ML position
        covariate_cluster_ids = ngs.cluster_id.values.astype(int)
        covariate_cluster_ids = covariate_cluster_ids[np.isin(covariate_cluster_ids, locations[
            (locations['brain_region'].str.contains('ENT')) & 
            (locations['coord_SCs_x'] >= cov_bounds[0]) & 
            (locations['coord_SCs_x'] < cov_bounds[1])]['cluster_id'].values.astype(int))]
        
        cov_clusters_df = all[all.cluster_id.isin(covariate_cluster_ids)]

        if len(covariate_cluster_ids) > 0:
            cov_tcs_time = {cluster_id: tcs_time[cluster_id] for cluster_id in cov_clusters_df.cluster_id if cluster_id in tcs_time}
            all_x = np.vstack(list(cov_tcs_time.values())).T
            include_position = False
        
            x = all_x.copy()
            if include_position:
                x = np.column_stack((pos_in_time, x))

            # get the target variable
            y = np.array(tcs_time[target_id])

            # fit the model
            Y_hat, pR2_cv = xgb_history.fit_cv(x, y, verbose = 0, continuous_folds = True)

            print(f'Target ID: {target_id}, Target Pos: {target_pos_label}, Covariate Pos: {covariate_pos_label}, Include Position: {include_position}, pR2_cv = {np.nanmean(pR2_cv)}, using {len(covariate_cluster_ids)} covariate cells')                
            
            tmp = pd.DataFrame()
            avg_pR2 = np.nanmean(pR2_cv)
            tmp['target_cluster_id'] = [target_id]
            tmp['target_cell_type'] = ['GC']
            tmp['target_pos_label'] = [target_pos_label]
            tmp['covariate_pos_label'] = [covariate_pos_label]
            tmp['covariate_cell_type'] = ['NGS']
            tmp['n_covariate_cells'] = [len(covariate_cluster_ids)]
            tmp['include_position'] = [include_position]
            tmp['pR2_cv'] = [avg_pR2]
            tmp['mouse'] = [mouse]
            tmp['day'] = [day]
 
            results_df = pd.concat([results_df, tmp], ignore_index=True)
        else:
            print(f'No covariate cells found for position {covariate_pos_label}, skipping.')

results_df.to_pickle(f'{data_path}/xgboost_MEC_lateralisation_assay{mouse}_day{day}.pkl')

print('XGBoost medial-lateral MEC lateralisation assay complete.')
print(f'results_df has shape: {results_df.shape}')
print(f'Results saved successfully @ {data_path}/xgboost_MEC_lateralisation_assay{mouse}_day{day}.pkl')

