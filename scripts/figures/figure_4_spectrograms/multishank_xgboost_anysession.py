
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


def reconstruct_shank_id(clusters_df, mouse, colname='unit_location_x'):
    shank_ids = []
    for index, cluster in clusters_df.iterrows():

        x_pos = cluster[colname]
        if mouse != 21:
            if x_pos <= 150:
                shank_id = 0
            elif (x_pos > 150 and x_pos <= 400):
                shank_id = 1
            elif (x_pos > 400 and x_pos <= 650):
                shank_id = 2
            elif x_pos > 650:
                shank_id = 3
            shank_ids.append(shank_id)
        # set the reverse shank ids for this mouse as it was 
        # implanted the other way round to all the other mice
        elif mouse == 21:
            if x_pos <= 150:
                shank_id = 3
            elif (x_pos > 150 and x_pos <= 400):
                shank_id = 2
            elif (x_pos > 400 and x_pos <= 650):
                shank_id = 1
            elif x_pos > 650:
                shank_id = 0
            shank_ids.append(shank_id)
    clusters_df['shank_id'] = shank_ids
    return clusters_df


import warnings
warnings.filterwarnings('ignore')


use_parser = True 

source_path = '/Users/harryclark/Downloads/COHORT12/'
data_path = '/Users/harryclark/Documents/data/'
fig_path = '/Users/harryclark/Documents/figs/FIGURE1/'
mouse = 28
day = 25

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


gcs, ngs, ns, sc, ngs_ns, all = cell_classification_of1(mouse, day, percentile_threshold=95, source_path=source_path) # subset
mec, para, pre, sub, vis, cere, other, all_by_anatomy = cell_classification_anatomy(mouse, day, source_path=source_path)

ngs = reconstruct_shank_id(ngs, mouse, colname='probe_x') 
gcs = reconstruct_shank_id(gcs, mouse, colname='probe_x')

g_m_ids, g_m_cluster_ids = HDBSCAN_grid_modules(gcs, all, mouse, day, min_cluster_size=3, cluster_selection_epsilon=3, 
                                                figpath=fig_path, curate_with_vr=False, curate_with_brain_region=True, plot_curate=False) # create grid modules using HDBSCAN    


Mouse = f'M{mouse}'

tcs, tcs_time, _, last_ephys_bin, beh, clusters = compute_vr_tcs(mouse, day, apply_zscore=False, apply_guassian_filter=False, source_path=source_path)

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


xgb_history = MLencoding(tunemodel = 'xgboost',
                         cov_history = True, spike_history=False, # We can choose!
                         window = time_bs, #this dataset has 100ms time bins
                         n_filters = 5,
                         max_time = 1000)


def compute_rolling_pr2(y_true, y_pred, trial_numbers, window_size=10):
    """
    Compute rolling pR2 values across trials using Poisson pseudo-R2.
    
    Parameters:
    -----------
    y_true : array
        True values (ground truth)
    y_pred : array  
        Predicted values
    trial_numbers : array
        Trial numbers for each time point
    window_size : int
        Number of trials to include in each rolling window
        
    Returns:
    --------
    unique_trials : array
        Trial numbers
    rolling_pr2 : array
        Rolling pR2 values for each trial
    """
    # Import the poisson_pseudoR2 function from mlencoding
    from spatial_manifolds.mlencoding import poisson_pseudoR2
    
    unique_trials = np.unique(trial_numbers)
    rolling_pr2 = np.full(len(unique_trials), np.nan)
    
    for i, trial in enumerate(unique_trials):
        # Define window around current trial
        start_trial = max(0, i - window_size//2)
        end_trial = min(len(unique_trials), i + window_size//2 + 1)
        window_trials = unique_trials[start_trial:end_trial]
        
        # Get indices for trials in window
        window_mask = np.isin(trial_numbers, window_trials)
        
        if np.sum(window_mask) > 10:  # Need minimum number of points
            y_true_window = y_true[window_mask]
            y_pred_window = y_pred[window_mask]
            
            # Remove NaN values
            valid_mask = ~(np.isnan(y_true_window) | np.isnan(y_pred_window))
            if np.sum(valid_mask) > 5:
                y_true_clean = y_true_window[valid_mask]
                y_pred_clean = y_pred_window[valid_mask]
                
                # Compute null model (mean activity in the window)
                y_null = np.mean(y_true_clean)
                
                # Compute Poisson pseudo-R2
                if len(y_true_clean) > 1 and np.var(y_true_clean) > 0:
                    pr2 = poisson_pseudoR2(y_true_clean, y_pred_clean, y_null)
                    rolling_pr2[i] = pr2 if not np.isnan(pr2) else 0
    
    return unique_trials, rolling_pr2


def compute_rolling_pr2_for_all_predictions(y_true, trial_numbers, predictions_dict, window_size=10):
    """
    Compute rolling pR2 for all prediction conditions using Poisson pseudo-R2.
    
    Parameters:
    -----------
    y_true : array
        True values (target cell activity)
    trial_numbers : array
        Trial numbers for each time point  
    predictions_dict : dict
        Dictionary with prediction names as keys and predicted values as values
    window_size : int
        Number of trials for rolling window
        
    Returns:
    --------
    results_df : DataFrame
        DataFrame with trial numbers and rolling pR2 for each condition
    """
    unique_trials = np.unique(trial_numbers)
    results = {'trial_number': unique_trials}
    
    for pred_name, y_pred in predictions_dict.items():
        _, rolling_pr2 = compute_rolling_pr2(y_true, y_pred, trial_numbers, window_size)
        results[f'rolling_pr2_{pred_name}'] = rolling_pr2
    
    return pd.DataFrame(results)


# Define the test cell groups
cell_groups = [
    ('Shank 0 NGS', ngs[ngs.shank_id == 0].cluster_id.values.tolist()),
    ('Shank 1 NGS', ngs[ngs.shank_id == 1].cluster_id.values.tolist()),
    ('Shank 2 NGS', ngs[ngs.shank_id == 2].cluster_id.values.tolist()),
    ('Shank 3 NGS', ngs[ngs.shank_id == 3].cluster_id.values.tolist()),
]

all_rolling_results_by_group = {}

for group_name, test_cell_ids in cell_groups:
    all_rolling_results = []

    if len(test_cell_id)>1:
        for ti, test_cell_id in enumerate(test_cell_ids.copy()):
            print(f'processing {group_name}, id {ti}/{len(test_cell_ids)}', flush=True)
            # Remove test cell from the group for covariate construction
            shank_0_ngs = ngs[ngs.shank_id == 0].cluster_id.values.tolist()
            if test_cell_id in shank_0_ngs:
                shank_0_ngs.remove(test_cell_id)
            shank_1_ngs = ngs[ngs.shank_id == 1].cluster_id.values.tolist()
            if test_cell_id in shank_1_ngs:
                shank_1_ngs.remove(test_cell_id)
            shank_2_ngs = ngs[ngs.shank_id == 2].cluster_id.values.tolist()
            if test_cell_id in shank_2_ngs:
                shank_2_ngs.remove(test_cell_id)
            shank_3_ngs = ngs[ngs.shank_id == 3].cluster_id.values.tolist()
            if test_cell_id in shank_3_ngs:
                shank_3_ngs.remove(test_cell_id)
                
            all_cell_ids = all.cluster_id.values.tolist()
            all_cells_except_test = [cid for cid in all_cell_ids if cid != test_cell_id and cid in tcs_time]
            cov_tcs_time_shank_0_ngs = {cluster_id: tcs_time[cluster_id] for cluster_id in shank_0_ngs if cluster_id in tcs_time}
            cov_tcs_time_shank_1_ngs = {cluster_id: tcs_time[cluster_id] for cluster_id in shank_1_ngs if cluster_id in tcs_time}
            cov_tcs_time_shank_2_ngs = {cluster_id: tcs_time[cluster_id] for cluster_id in shank_2_ngs if cluster_id in tcs_time}
            cov_tcs_time_shank_3_ngs = {cluster_id: tcs_time[cluster_id] for cluster_id in shank_3_ngs if cluster_id in tcs_time}
            y = np.array(tcs_time[test_cell_id])
            y_smoothed = gaussian_filter_nan(y, sigma=3)
            X  = np.stack([pos_in_time]).T
            Xs0 = np.column_stack((np.vstack(list(cov_tcs_time_shank_0_ngs.values())).T)).T
            Xs1 = np.column_stack((np.vstack(list(cov_tcs_time_shank_1_ngs.values())).T)).T
            Xs2 = np.column_stack((np.vstack(list(cov_tcs_time_shank_2_ngs.values())).T)).T
            Xs3 = np.column_stack((np.vstack(list(cov_tcs_time_shank_3_ngs.values())).T)).T
            Xall = np.stack([np.array(tcs_time[cid]) for cid in all_cells_except_test]).T
            X_pos_s0 = np.column_stack((pos_in_time, np.vstack(list(cov_tcs_time_shank_0_ngs.values())).T))
            X_pos_s1 = np.column_stack((pos_in_time, np.vstack(list(cov_tcs_time_shank_1_ngs.values())).T))
            X_pos_s2 = np.column_stack((pos_in_time, np.vstack(list(cov_tcs_time_shank_2_ngs.values())).T))
            X_pos_s3 = np.column_stack((pos_in_time, np.vstack(list(cov_tcs_time_shank_3_ngs.values())).T))
            Xpos_all = np.column_stack((pos_in_time, Xall))
            Y_hat_p, _ = xgb_history.fit_cv(X, y, verbose=0, continuous_folds=True)
            Y_hat_s0, _ = xgb_history.fit_cv(Xs0, y, verbose=0, continuous_folds=True)
            Y_hat_s1, _ = xgb_history.fit_cv(Xs1, y, verbose=0, continuous_folds=True)
            Y_hat_s2, _ = xgb_history.fit_cv(Xs2, y, verbose=0, continuous_folds=True)
            Y_hat_s3, _ = xgb_history.fit_cv(Xs3, y, verbose=0, continuous_folds=True)
            Y_hat_all, _ = xgb_history.fit_cv(Xall, y, verbose=0, continuous_folds=True)
            Y_hat_pos_s0, _ = xgb_history.fit_cv(X_pos_s0, y, verbose=0, continuous_folds=True)
            Y_hat_pos_s1, _ = xgb_history.fit_cv(X_pos_s1, y, verbose=0, continuous_folds=True)
            Y_hat_pos_s2, _ = xgb_history.fit_cv(X_pos_s2, y, verbose=0, continuous_folds=True)
            Y_hat_pos_s3, _ = xgb_history.fit_cv(X_pos_s3, y, verbose=0, continuous_folds=True)
            Y_hat_pos_all, _ = xgb_history.fit_cv(Xpos_all, y, verbose=0, continuous_folds=True)
            y_full = np.array(tcs_time[test_cell_id])
            predictions_dict = {
                'position': Y_hat_p,
                'shank_0_ngs': Y_hat_s0,
                'shank_1_ngs': Y_hat_s1,
                'shank_2_ngs': Y_hat_s2,
                'shank_3_ngs': Y_hat_s3,
                'all_cells': Y_hat_all,
                'position_plus_shank_0_ngs': Y_hat_pos_s0,
                'position_plus_shank_1_ngs': Y_hat_pos_s1,
                'position_plus_shank_2_ngs': Y_hat_pos_s2,
                'position_plus_shank_3_ngs': Y_hat_pos_s3,
                'position_plus_all_cells': Y_hat_pos_all,
            }
            rolling_pr2_results = compute_rolling_pr2_for_all_predictions(
                y_true=y_full, 
                trial_numbers=trial_number_in_time,
                predictions_dict=predictions_dict,
                window_size=15
            )
            rolling_pr2_results['test_cell_id'] = test_cell_id
            all_rolling_results.append(rolling_pr2_results)
        
        # Combine all rolling results into a single DataFrame for this group
        all_rolling_df = pd.concat(all_rolling_results, ignore_index=True)
        all_rolling_results_by_group[group_name] = all_rolling_df

# save all_rolling_results_by_group 
import pickle

# Suppose your dictionary is called df_dict
results_path = f'{data_path}M{mouse}_D{day}.pkl'

with open(results_path, 'wb') as f:
    pickle.dump(all_rolling_results_by_group, f)
print(f'saved results in pickle @ {results_path}', flush=True)