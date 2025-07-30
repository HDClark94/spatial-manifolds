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

time_start = time.time()

use_parser = True

source_path = '/Users/harryclark/Downloads/COHORT12/'
data_path = '/Users/harryclark/Documents/data/'
fig_path = '/Users/harryclark/Documents/figs/FIGURE1/'
mouse = 25
day = 16

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

# set up xgboost history model
xgb_history = MLencoding(tunemodel = 'xgboost', cov_history = True, spike_history=False, 
                         window = time_bs, n_filters = nfilters, max_time = history_length)

# create a dataframe to store the results
xgboost_results = pd.DataFrame(columns=['mouse', 
                             'day', 
                             'cluster_id',
                             'cluster_id_cov',
                             'trained_on', 
                             'tested_on', 
                             'n_neurons',
                             'pR2_cv', 
                             'firing_mode',
                             'n_filters',
                             'history_length',
                             'distance'])

gcs, ngs, ns, sc, ngs_ns, all = cell_classification_of1(mouse, day, percentile_threshold=95, source_path=source_path) # subset
rc, rsc, vr_ns = cell_classification_vr(mouse, day, source_path=source_path)
tcs, tcs_time, _ , last_ephys_bin, beh, clusters_VR = compute_vr_tcs(mouse, day, apply_zscore=False, source_path=source_path) 
_, _, _, _, _, _, _, all_by_anatomy = cell_classification_anatomy(mouse,day, source_path=source_path)
g_m_ids, g_m_cluster_ids = HDBSCAN_grid_modules(gcs, all, mouse, day, min_cluster_size=3, cluster_selection_epsilon=3, 
                                                figpath=fig_path, curate_with_vr=False, curate_with_brain_region=True, source_path=source_path) # create grid modules using HDBSCAN    

# load the behaviour data
last_ephys_time_bin = clusters_VR[clusters_VR.index[0]].count(bin_size=time_bs, time_units = 'ms').index[-1]

# time binned variables for later
ep = nap.IntervalSet(start=0, end=last_ephys_time_bin, time_units = 's')
speed_in_time = np.array(beh['S'].bin_average(bin_size=time_bs, time_units = 'ms', ep=ep))
dt_in_time = np.array(beh['travel'].bin_average(bin_size=time_bs, time_units = 'ms', ep=ep)-((beh['trial_number'][0]-1)*tl))
pos_in_time = dt_in_time%tl
trial_number_in_time = (dt_in_time//tl)+beh['trial_number'][0]
trial_number_in_time = np.array(trial_number_in_time, dtype=int)

if np.any(np.isnan(pos_in_time)):
    series = pd.Series(dt_in_time)
    filled_series = series.ffill().bfill()
    dt_in_time = np.array(filled_series)
    pos_in_time = dt_in_time%tl
    trial_number_in_time = (dt_in_time//tl)+beh['trial_number'][0]
    trial_number_in_time = np.array(trial_number_in_time, dtype=int)

if np.any(np.isnan(speed_in_time)):
    series = pd.Series(speed_in_time)
    filled_series = series.ffill().bfill()
    speed_in_time = np.array(speed_in_time)

if np.any(np.isnan(trial_number_in_time)):
    series = pd.Series(trial_number_in_time)
    filled_series = series.ffill().bfill()
    trial_number_in_time = np.array(trial_number_in_time)
    trial_number_in_time = np.array(trial_number_in_time, dtype=int)

# we now have cluster ids classified into modules, non grid spatial cells and non spatial cells 
# as defined by activity in the open field
g_m_cluster_ids = sorted(g_m_cluster_ids, key=len, reverse=True) 
cluster_ids_by_group = []
cluster_ids_by_group.extend(g_m_cluster_ids) # grid cells by module [0,1,2...]
cluster_ids_by_group.append(ngs.cluster_id.values.tolist()) # non grid spatial [-4]
cluster_ids_by_group.append(ns.cluster_id.values.tolist()) # non spatial cells [-3]
cluster_ids_by_group.append(gcs.cluster_id.values.tolist()) # all grid cells [-2]
cluster_ids_by_group.append(sc.cluster_id.values.tolist()) # speed cells [-1]

# compute the spectrogram and labels from the most adunant grid module
peak_indices = [12, 28, 44, 60, 76]
cluster_ids = g_m_cluster_ids[0]
tcs_to_use = {cluster_id: tcs[cluster_id] for cluster_id in cluster_ids if cluster_id in tcs}
results = spectral_analysis(tcs_to_use, tl, bs=bs)
spectrograms = results[3] 
fvalid = results[5]
S = spectrograms.mean(0)
max_peaks = np.argmax(S, axis=0)
labels = np.isin(max_peaks, peak_indices).astype(int)

# redefine cluster ids by group
cmGC_cluster_ids = g_m_cluster_ids[0] # grid cells in the first module
ncmGC_cluster_ids = np.setdiff1d(gcs.cluster_id.values.tolist(), cmGC_cluster_ids)
ngs_cluster_ids = ngs.cluster_id.values.tolist() # non grid spatial cells

print(f'Processing mouse {mouse}, day {day}')
print(f'{len(cmGC_cluster_ids)} cmGC, {len(ncmGC_cluster_ids)} ncmGC, {len(ngs_cluster_ids)} NGS')

for trained_on, cov_cluster_ids in zip(['GC', 'NGS'], [gcs.cluster_id.values.tolist(), ngs.cluster_id.values.tolist()]):
    for i, cov_cluster_id in enumerate(cov_cluster_ids):
        for j, test_cluster_id in enumerate(gcs.cluster_id.values.tolist() + ngs.cluster_id.values.tolist()):
            if trained_on == 'NGS' and test_cluster_id in gcs.cluster_id.values.tolist():
                tested_on = 'GC'
            elif trained_on == 'NGS' and test_cluster_id in ngs.cluster_id.values.tolist():
                tested_on = 'NGS'
            elif trained_on == 'GC' and test_cluster_id in ngs.cluster_id.values.tolist():
                tested_on = 'NGS'

            elif trained_on == 'GC':
                # Check if both IDs are in the same grid module
                in_same_module = any(
                    (cov_cluster_id in module and test_cluster_id in module)
                    for module in g_m_cluster_ids
                )
                if in_same_module:
                    tested_on = 'cmGC'
                else:
                    tested_on = 'ncmGC'
            
            if cov_cluster_id == test_cluster_id:
                print(f'Skipping covariate cluster {cov_cluster_id} and test cluster {test_cluster_id} as they are the same')
                continue
            
            print(f'Processing trained on {trained_on} and tested on {tested_on}, i={i} covariate cluster {cov_cluster_id}, tested on {test_cluster_id}, j={j}')

            tc = tcs[test_cluster_id]
            tc = gaussian_filter(np.nan_to_num(tc).astype(np.float64), sigma=2.5)
            tc = tc[:last_ephys_bin] # only want bins with ephys data in it
            trial_labels = get_kmeans_spatial_labels(tc, labels, bs=bs, tl=tl) # reuse kmeans function to get the labels based on the task anchoring
            x_pos_cov = clusters_VR.coord_probe_x[cov_cluster_id]
            y_pos_cov = clusters_VR.coord_probe_y[cov_cluster_id]
            x_pos_test = clusters_VR.coord_probe_x[test_cluster_id]
            y_pos_test = clusters_VR.coord_probe_y[test_cluster_id]
            
            cov_tcs_time = {id: tcs_time[id] for id in [cov_cluster_id] if id in tcs_time}
            all_x = np.vstack(list(cov_tcs_time.values())).T
            
            for n in [0,1]:
                if n > 0:
                    x = all_x[:, :n]
                    # add position to the covariate history as well
                    x = np.column_stack((pos_in_time, x))
                else:
                    x = pos_in_time.reshape(-1, 1)

                # get the target variable
                y = np.array(tcs_time[test_cluster_id])

                # fit the model
                Y_hat, pR2_cv = xgb_history.fit_cv(x, y, verbose = 0, continuous_folds = True)
                print(f'Firing mode = whole session, n_neurons {n}, pR2: {np.nanmean(pR2_cv)}')
                
                df = pd.DataFrame({
                    'mouse': mouse,
                    'day': day,
                    'cluster_id': test_cluster_id,
                    'cluster_id_cov': cov_cluster_id,
                    'trained_on': trained_on,
                    'tested_on': tested_on,
                    'n_neurons': n,
                    'pR2_cv': np.nanmean(pR2_cv),
                    'n_filters': nfilters,
                    'history_length': history_length,
                    'firing_mode': 'session',
                    'distance': np.sqrt((x_pos_cov - x_pos_test)**2 + (y_pos_cov - y_pos_test)**2),
                }, index=[0])
                data = pd.concat([data, df], ignore_index=True)
        
                # now we need to calculate the pR2 for each firing mode
                for firing_mode, label in zip(['TA', 'TI'], [1, 0]):
                    trials = beh['trials'][:len(trial_labels)]
                    trials = trials[trial_labels == label]['number'].values
                    mask = np.isin(trial_number_in_time, trials)
                    # use that mask to calcalte the pr2
                    
                    pR2_condition_cv = poisson_pseudoR2(y[mask], Y_hat[mask], ynull=np.nanmean(y[mask]))
                    print(f'Firing mode {firing_mode}, n_neurons {n}, pR2: {pR2_condition_cv}')
                    
                    df = pd.DataFrame({
                        'mouse': mouse,
                        'day': day,
                        'cluster_id': test_cluster_id,
                        'cluster_id_cov': cov_cluster_id,
                        'trained_on': trained_on,
                        'tested_on': tested_on,
                        'n_neurons': n,
                        'pR2_cv': pR2_condition_cv,
                        'n_filters': nfilters,
                        'history_length': history_length,
                        'firing_mode': firing_mode,
                        'distance': np.sqrt((x_pos_cov - x_pos_test)**2 + (y_pos_cov - y_pos_test)**2),
                    }, index=[0])
                    data = pd.concat([data, df], ignore_index=True)

# save results in csv
xgboost_result_path = f'{data_path}M{mouse}_D{day}.csv'
xgboost_results.to_csv(xgboost_result_path)

time_end = time.time()
print(f'Time taken: {time_end - time_start} seconds')