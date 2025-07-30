import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix
import pynapple as nap
from spatial_manifolds.toroidal import *
from spatial_manifolds.behaviour_plots import *
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import distance
from spatial_manifolds.circular_decoder import circular_decoder, cross_validate_decoder, cross_validate_decoder_time, circular_nanmean, circular_nansem
from scipy.stats import gaussian_kde
from astropy.convolution import convolve, Gaussian1DKernel
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.cm as cm
from scipy.signal import find_peaks
from scipy.signal import correlate

from spatial_manifolds.tuning_scores.grid_score import autocorr2d
from spatial_manifolds.data.curation import curate_clusters
from scipy.stats import zscore
from spatial_manifolds.util import gaussian_filter_nan
from spatial_manifolds.predictive_grid import compute_travel_projected, wrap_list
from spatial_manifolds.behaviour_plots import *
from spatial_manifolds.behaviour_plots import trial_cat_priority
from spatial_manifolds.anaylsis_parameters import *
import hdbscan
from sklearn.preprocessing import StandardScaler
from spatial_manifolds.anaylsis_parameters import tl, bs, time_bs, rm_figsize, disqualifying_brain_areas_for_grid_cells
 

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'


def cell_classification_vr(mouse, day, percentile_threshold=99, source_path=None):
    if source_path is None:
        source_path = '/Users/harryclark/Downloads/COHORT12/'
    session = 'VR'
    vr_folder = f'{source_path}M{mouse}/D{day:02}/{session}/'
    ramp_path = vr_folder + 'tuning_scores/ramps.parquet'
    speed_path = vr_folder + 'tuning_scores/speed_correlation.parquet'
    spatial_path = vr_folder + 'tuning_scores/spatial_information.parquet'
 
    ramp_table = pd.read_parquet(ramp_path)
    speed_table = pd.read_parquet(speed_path)
    spatial_table = pd.read_parquet(spatial_path)

    ramp_classes = [('+','+'),('+','-'), ('+','/'),('-','+'),('-','-'),('-','/')]

    ramp_cells = pd.DataFrame()
    for class_idx, ramp_class in enumerate(ramp_classes):
        subset_ids1 = ramp_table[(ramp_table['trials'] == 'b+nb')
                                 & (ramp_table['outbound_sign'] == ramp_class[0])
                                ]['cluster_id'].values
        subset_ids2 = ramp_table[(ramp_table['trials'] == 'b+nb')
                                 & (ramp_table['homebound_sign'] == ramp_class[1])
                                 ]['cluster_id'].values
        subset_ids = np.intersect1d(subset_ids1, subset_ids2)
        
        tmp = pd.DataFrame()
        tmp['cluster_id'] = subset_ids
        tmp['ramp_class'] = np.repeat(''.join(ramp_class), len(subset_ids))
        speed_mod = []
        for id in subset_ids:
            speed_mod.append(speed_table[(speed_table['cluster_id'] == id) &
                                         (speed_table['trials'] == 'b+nb') &
                                         (speed_table['context'] == 'rz1')]['sig'].iloc[0])
        tmp['speed_modulated'] = speed_mod
        ramp_cells = pd.concat([ramp_cells, tmp])

    non_spatial_cells = spatial_table.query('sig == False')

    ramp_and_speed_cells = ramp_cells[ramp_cells['speed_modulated'] == True]

    if len(non_spatial_cells)<1:
        non_spatial_cells = pd.DataFrame(columns=spatial_table.columns)
    if len(ramp_cells)<1:
        ramp_cells = pd.DataFrame(columns=speed_table.columns)
    if len(non_spatial_cells)<1:
        ramp_and_speed_cells = pd.DataFrame(columns=speed_table.columns)
    
    return ramp_cells, ramp_and_speed_cells, non_spatial_cells

def cell_classification_anatomy(mouse, day, source_path=None):
    if source_path is None:
        source_path = '/Users/harryclark/Downloads/COHORT12/'
    _,_,_,_,_,clusters_VR = compute_vr_tcs(mouse, day, source_path=source_path)
    session = 'OF1'
    of1_folder = f'{source_path}M{mouse}/D{day:02}/{session}/'
    spatial_path = of1_folder + "tuning_scores/shifted_spatial_information.parquet"
    spatial_information_score_of1 = pd.read_parquet(spatial_path)
    cluster_ids_values = spatial_information_score_of1.query('travel == 0').cluster_id

    MEC_cells = pd.DataFrame(columns=spatial_information_score_of1.columns)
    PARA_cells = pd.DataFrame(columns=spatial_information_score_of1.columns)
    PRE_cells = pd.DataFrame(columns=spatial_information_score_of1.columns)
    VIS_cells = pd.DataFrame(columns=spatial_information_score_of1.columns)
    SUB_cells = pd.DataFrame(columns=spatial_information_score_of1.columns)
    CERE_cells = pd.DataFrame(columns=spatial_information_score_of1.columns)
    other_cells = pd.DataFrame(columns=spatial_information_score_of1.columns)

    print(f'unique brain regions {np.unique(clusters_VR.brain_region)}')
    for index in cluster_ids_values:
        cluster_spatial_information_of1 = spatial_information_score_of1[spatial_information_score_of1.cluster_id==index]
        optimal_lag = cluster_spatial_information_of1.travel.values[np.nanargmax(cluster_spatial_information_of1.spatial_information)]

        brain_region = clusters_VR.brain_region[index]
        SC_x = clusters_VR.coord_SCs_x[index]
        SC_y = clusters_VR.coord_SCs_y[index]
        SC_z = clusters_VR.coord_SCs_z[index]
        probe_x = clusters_VR.coord_probe_x[index]
        probe_y = clusters_VR.coord_probe_y[index]

        cell = cluster_spatial_information_of1[(spatial_information_score_of1.travel == optimal_lag)]
        cell['mouse'] = mouse
        cell['day'] = day
        cell['brain_region'] = brain_region
        cell['optimal_travel_lag'] = optimal_lag
        cell['SC_x'] = SC_x
        cell['SC_y'] = SC_y
        cell['SC_z'] = SC_z
        cell['probe_x'] = probe_x
        cell['probe_y'] = probe_y
        
        if 'ENT' in brain_region:
            MEC_cells = pd.concat([MEC_cells, cell], ignore_index=True)
        elif 'PAR' in brain_region:
            PARA_cells = pd.concat([PARA_cells, cell], ignore_index=True)
        elif 'PRE' in brain_region:
            PRE_cells = pd.concat([PRE_cells, cell], ignore_index=True)
        elif 'VIS' in brain_region:
            VIS_cells = pd.concat([VIS_cells, cell], ignore_index=True)
        elif 'arb' in brain_region:
            CERE_cells = pd.concat([CERE_cells, cell], ignore_index=True)
        elif 'PFL' in brain_region:
            CERE_cells = pd.concat([CERE_cells, cell], ignore_index=True)
        elif 'FL' in brain_region:
            CERE_cells = pd.concat([CERE_cells, cell], ignore_index=True)
        elif 'root' in brain_region:
            CERE_cells = pd.concat([CERE_cells, cell], ignore_index=True)
        elif 'SIM' in brain_region:
            CERE_cells = pd.concat([CERE_cells, cell], ignore_index=True)
        elif 'SUB' in brain_region:
            SUB_cells = pd.concat([SUB_cells, cell], ignore_index=True)
        else:
            other_cells = pd.concat([other_cells, cell], ignore_index=True)
            
    print(f'there are {len(MEC_cells)} MEC cells')
    print(f'there are {len(PARA_cells)} PARA cells')
    print(f'there are {len(PRE_cells)} PRE cells')
    print(f'there are {len(VIS_cells)} VIS cells')
    print(f'there are {len(CERE_cells)} CERE cells')
    print(f'there are {len(other_cells)} other cells')
    print(f'there are {len(SUB_cells)} SUB cells')

    all_cells = pd.concat([MEC_cells, PARA_cells, PRE_cells, SUB_cells, VIS_cells, CERE_cells, other_cells], ignore_index=True)
    print(f'there are {len(all_cells)} all cells')

    return MEC_cells, PARA_cells, PRE_cells, SUB_cells, VIS_cells, CERE_cells, other_cells, all_cells

        

def cell_classification_of1(mouse, day, percentile_threshold=95, source_path=None, 
                            disqualifying_brain_areas_for_grid_cells=disqualifying_brain_areas_for_grid_cells, 
                            disqualifying_brain_areas_for_spatial_cells=disqualifying_brain_areas_for_grid_cells+['VIS'], 
                            use_optimal_travel=True):
    if source_path is None:
        source_path = '/Users/harryclark/Downloads/COHORT12/'
    _,_,_,_,_,clusters_VR = compute_vr_tcs(mouse, day, source_path=source_path)

    print(mouse, day)
    session = 'OF1'
    of1_folder = f'{source_path}M{mouse}/D{day:02}/{session}/'
    shifted_grid_path = of1_folder + "tuning_scores/shifted_grid_score.parquet"
    spatial_path = of1_folder + "tuning_scores/shifted_spatial_information.parquet"
    speed_path = of1_folder + "tuning_scores/shifted_speed_correlation.parquet"

    shifted_grid_scores_of1 = pd.read_parquet(shifted_grid_path)
    spatial_information_score_of1 = pd.read_parquet(spatial_path)
    shifted_speed_score_of1 = pd.read_parquet(speed_path)

    shifted_speed_score_of1 = shifted_speed_score_of1.query('travel == 0')
    cluster_ids_values = shifted_grid_scores_of1.query('travel == 0').cluster_id

    non_grid_cells = pd.DataFrame(columns=shifted_grid_scores_of1.columns)
    grid_cells = pd.DataFrame(columns=shifted_grid_scores_of1.columns)
    non_spatial_cells = pd.DataFrame(columns=shifted_grid_scores_of1.columns)
    speed_cells = pd.DataFrame(columns=shifted_grid_scores_of1.columns)
    cells = pd.DataFrame(columns=shifted_grid_scores_of1.columns)

    # estimate optimal travel using spatial information of all retrohippocampal cells
    travel = np.arange(-50, 50, 2)
    travel_at_max = []
    for id in cluster_ids_values:
        id_scores = np.array(spatial_information_score_of1[spatial_information_score_of1.cluster_id == id].spatial_information)
        id_travels = np.array(spatial_information_score_of1[spatial_information_score_of1.cluster_id == id].travel)

        brain_region = clusters_VR.brain_region[id]
        if 'ENT' in brain_region or 'PAR' in brain_region or 'PRE' in brain_region:
            travel_at_max.append(id_travels[np.nanargmax(id_scores)])

    if len(travel_at_max) > 3:
        kde = gaussian_kde(travel_at_max, bw_method=0.3) # Adjust bw_method for smoothing
        x = np.linspace(-50, 50, 1000)
        kde_values = kde(x)
        optimal_travel = x[np.argmax(kde_values)]
        print(f'optimal travel lag is {optimal_travel} cm based on kde of travel at max spatial information')
        # don't allow optimal_travel to be less than -30
        if optimal_travel < -30:
            optimal_travel = 0  
    else:
        optimal_travel = 0

    if use_optimal_travel:
        print(f'optimal travel lag is {optimal_travel} cm')
    else:
        print(f'using travel lag of 0 cm')
        optimal_travel = 0
    

    # now loop through the clusters and classify them
    for index in cluster_ids_values:
        brain_region = clusters_VR.brain_region[index]
        SC_x = clusters_VR.coord_SCs_x[index]
        SC_y = clusters_VR.coord_SCs_y[index]
        SC_z = clusters_VR.coord_SCs_z[index]
        probe_x = clusters_VR.coord_probe_x[index]
        probe_y = clusters_VR.coord_probe_y[index]

        if brain_region not in disqualifying_brain_areas_for_grid_cells:
            cluster_spatial_information_of1 = spatial_information_score_of1[spatial_information_score_of1.cluster_id==index]
            cluster_shifted_grid_scores_of1 = shifted_grid_scores_of1[shifted_grid_scores_of1.cluster_id==index]
            cluster_speed_correlation_of1 = shifted_speed_score_of1[shifted_speed_score_of1.cluster_id==index]
            cluster_optimal_lag = cluster_spatial_information_of1.travel.values[np.nanargmax(cluster_spatial_information_of1.spatial_information)]
            cluster_optimal_lag_grid_score = cluster_shifted_grid_scores_of1.travel.values[np.nanargmax(cluster_shifted_grid_scores_of1.grid_score)]

            percentile99_grid_score_of1 = np.nanpercentile(cluster_shifted_grid_scores_of1.null_grid_score.iloc[0], percentile_threshold)
            percentile99_spatial_information_of1 = np.nanpercentile(cluster_spatial_information_of1.null_spatial_information.iloc[0], percentile_threshold)

            percentile99_speed_information_of1_pos = np.nanpercentile(cluster_speed_correlation_of1.null_speed_correlation.iloc[0], percentile_threshold)
            percentile99_speed_information_of1_neg = np.nanpercentile(cluster_speed_correlation_of1.null_speed_correlation.iloc[0], 100-percentile_threshold)

            max_grid_score_of1 = cluster_shifted_grid_scores_of1[cluster_shifted_grid_scores_of1['travel'] == np.round(optimal_travel)]['grid_score'].iloc[0]
            spatial_info = cluster_spatial_information_of1[cluster_spatial_information_of1['travel'] == np.round(optimal_travel)]['spatial_information'].iloc[0]
            spatial_info_no_lag = cluster_spatial_information_of1.spatial_information.iloc[0]

            speed_correlation = cluster_speed_correlation_of1.speed_correlation.iloc[0]

            cell = shifted_grid_scores_of1[shifted_grid_scores_of1.grid_score==max_grid_score_of1]
            cell['mouse'] = mouse
            cell['day'] = day
            cell['brain_region'] = brain_region
            cell['session_travel_lag'] = optimal_travel
            cell['optimal_travel_lag'] = cluster_optimal_lag
            cell['optimal_travel_lag_grid_score'] = cluster_optimal_lag_grid_score
            cell['spatial_information_score'] = spatial_info
            cell['spatial_information_score_no_lag'] = spatial_info_no_lag
            cell['SC_x'] = SC_x
            cell['SC_y'] = SC_y
            cell['SC_z'] = SC_z
            cell['probe_x'] = probe_x
            cell['probe_y'] = probe_y

            if (max_grid_score_of1 > percentile99_grid_score_of1) and (spatial_info > percentile99_spatial_information_of1):
                grid_cells = pd.concat([grid_cells, cell], ignore_index=True)
            elif (spatial_info > percentile99_spatial_information_of1) and (brain_region not in disqualifying_brain_areas_for_spatial_cells):
                non_grid_cells = pd.concat([non_grid_cells, cell], ignore_index=True)
            elif (speed_correlation > percentile99_speed_information_of1_pos) or (speed_correlation < percentile99_speed_information_of1_neg):
                speed_cells = pd.concat([speed_cells, cell], ignore_index=True)
            else:
                non_spatial_cells = pd.concat([non_spatial_cells, cell], ignore_index=True)
            cells = pd.concat([cells, cell], ignore_index=True)
        
    all_cells = cells.copy()
    non_grid_and_non_spatial_cells = pd.concat([non_grid_cells, non_spatial_cells], ignore_index=True)

    print(f'there are {len(non_grid_and_non_spatial_cells)} non_grid and non_spatial_cells')
    print(f'there are {len(grid_cells)} grid_cells')
    print(f'there are {len(non_grid_cells)} non grid spatial cells')
    print(f'there are {len(non_spatial_cells)} non spatial cells')
    print(f'there are {len(speed_cells)} speed cells')
    print(f'there are {len(all_cells)} cells')

    if len(non_grid_cells)>0:
        print(f'for the non-grid spatial cells the unique locations are {np.unique(non_grid_cells.brain_region)}')  
    if len(grid_cells)>0:
        print(f'for the grid cells the unique locations are {np.unique(grid_cells.brain_region)}')

    return grid_cells, non_grid_cells, non_spatial_cells, speed_cells, non_grid_and_non_spatial_cells, all_cells



def HDBSCAN_grid_modules(gcs, all, mouse, day, figpath='', min_cluster_size=None, cluster_selection_epsilon=None,
                         curate_with_vr=True, curate_with_brain_region=True, source_path=None, plot_curate=False):
    print(source_path)
    if source_path is None:
        source_path = '/Users/harryclark/Downloads/COHORT12/'

    if min_cluster_size is not None:
        print(f'params min_cluster_size and cluster_selection_epsilon are deprecated, these are now set to default values of 3 and 0.4 respectively')
    
    if len(gcs) <= 1:
        return [], []
    
    gcs['field_spacing'] = pd.to_numeric(gcs['field_spacing'], errors='coerce')
    gcs['orientation'] = pd.to_numeric(gcs['orientation'], errors='coerce')
    gcs.dropna(subset=['field_spacing', 'orientation'], inplace=True)

    # Extract and preprocess features
    X = gcs[["field_spacing", "orientation"]].copy()

    # Standard scale the 'field_spacing'
    scaler = StandardScaler()
    X["field_spacing_scaled"] = scaler.fit_transform(X[["field_spacing"]])

    # Cyclic encoding for 'orientation' (range 0 to 60)
    X["orientation_sin"] = np.sin(2 * np.pi * X["orientation"] / 60)
    X["orientation_cos"] = np.cos(2 * np.pi * X["orientation"] / 60)

    # Scale sine and cosine components to balance feature influence
    scale_factor = 1 / np.sqrt(2)
    X["orientation_sin_scaled"] = X["orientation_sin"] * scale_factor
    X["orientation_cos_scaled"] = X["orientation_cos"] * scale_factor

    # Prepare the feature matrix
    features = X[["field_spacing_scaled", 
                  "orientation_sin_scaled", 
                  "orientation_cos_scaled"]]
    original_features = X[["field_spacing", 
                           "orientation"]]
    
    # Perform HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
                min_cluster_size=3,
                min_samples=1,
                cluster_selection_epsilon=0.4,
                allow_single_cluster=False,
                metric='chebyshev'
            )
    module_labels = clusterer.fit_predict(features)

    # merge labels if centroids are within 10 units (e.g., cm or degrees)
    centroids = original_features.groupby(module_labels).mean().values
    centroids = np.array(centroids)
    # Start with original labels
    merged_labels = module_labels.copy()
    for i, label_i in enumerate(np.unique(module_labels)):
        for j, label_j in enumerate(np.unique(module_labels)):
            if j <= i or label_i == -1 or label_j == -1:
                continue
            if (distance.euclidean(centroids[i], centroids[j]) < 10):
                print(f"Merging labels {label_i} and {label_j} with distance {distance.euclidean(centroids[i], centroids[j])}")
                merged_labels[merged_labels == label_j] = label_i
            else:
                print(f"Not merging labels {label_i} and {label_j} with distance {distance.euclidean(centroids[i], centroids[j])}")
    module_labels = merged_labels.copy()
    
    X["cluster"] = module_labels

    print(f'Found {len(np.unique(module_labels))} modules with HDBSCAN for mouse {mouse} day {day}')
    print(f'for each module , the number of points is: {np.unique(module_labels, return_counts=True)[1]}')
    
    # Plot the results
    plt.figure(figsize=(3, 3))
    sns.scatterplot(data=X, x="field_spacing", y="orientation", hue="cluster", palette="tab10", s=25, legend=False,linewidth=0)

    #label_colors = {label: cm.get_cmap('viridis', len(np.unique(module_labels)))(i) for i, label in enumerate(np.unique(module_labels))}
    #for mi in np.unique(module_labels):
    #    mask = module_labels == mi
    #    print(f'for mi{mi}, there are {np.sum(mask)} points')
    #    plt.scatter(np.array(X['field_spacing'])[mask], 
    #                np.array(X['orientation'])[mask], c=label_colors[mi], s=20, cmap='viridis', label='Clustered Points')
    
    # Highlight unassigned points (label -1)
    plt.scatter(np.array(X['field_spacing'])[module_labels == -1], 
                np.array(X['orientation'])[module_labels == -1], s=21, color='black', label='Unassigned Points')
    plt.scatter(all['field_spacing'], all['orientation'], s=20, color='tab:grey', alpha=0.5,zorder=-1)

    #plt.legend()
    plt.xlabel('Grid spacing (cm)')
    plt.ylabel('Grid orientation ($^\circ$)')
    plt.ylim(0,60)
    plt.title(f'HDBSCAN M{mouse}D{day}')
    plt.tight_layout()
    plt.close()

    if np.unique(module_labels).size == 1 and np.unique(module_labels)[0] == -1:
        module_labels[:] = 0  # Assign all points to a single cluster if no clusters were found
        return [], []
    
    # put cluster ids into modules then rearange from smallest spacing to larger
    grid_module_cluster_ids = []
    grid_module_ids = []
    avg_spacings = []
    for mi, module_label in enumerate(np.unique(module_labels[module_labels != -1])):
        grid_ids = np.array(gcs['cluster_id'])
        cells = gcs[np.isin(gcs['cluster_id'], grid_ids[module_labels == module_label])]
        avg_spacings.append(np.nanmean(cells.field_spacing.values))
        grid_module_cluster_ids.append(cells['cluster_id'].tolist())
        grid_module_ids.append(mi)
        print(f'for module {mi}, there are {len(cells)} cells with average spacing {np.nanmean(cells.field_spacing.values)}')
    grid_module_cluster_ids = [x for _, x in sorted(zip(avg_spacings, grid_module_cluster_ids))]
    grid_module_ids = [x for _, x in sorted(zip(avg_spacings, grid_module_ids))]

    # now curate the modules based on the VR data and anatomy
    _,_,autocorrs,_,_,clusters_VR = compute_vr_tcs(mouse, day, source_path=source_path)
    if curate_with_brain_region:
        for mi, module_ids in zip(grid_module_ids, grid_module_cluster_ids):
            print(f'module {mi} contains cells from {np.unique(clusters_VR[module_ids].brain_region)}')
            new_module_ids = module_ids.copy()
            for id in module_ids:
                br = clusters_VR.brain_region[id]
                if br in disqualifying_brain_areas_for_grid_cells:
                    module_ids.remove(id)
            grid_module_cluster_ids[grid_module_ids.index(mi)] = new_module_ids

    if curate_with_vr:
        tolerance = 30
        prominence = 0.05
        # before performing the median peak check, plot the histogram of peaks
        for mi, module_ids in zip(grid_module_ids, grid_module_cluster_ids):
            matrix = np.array(list(autocorrs.values()))
            matrix_cluster_ids = np.array(list(autocorrs.keys()))
            cluster_id_of_interest = module_ids
            matrix = matrix[np.isin(matrix_cluster_ids, cluster_id_of_interest)]
            matrix_cluster_ids = matrix_cluster_ids[np.isin(matrix_cluster_ids, cluster_id_of_interest)]
            peaks = []
            for array in matrix:
                if len(find_peaks(array,prominence=prominence)[0])>0:
                    peak = find_peaks(array,prominence=prominence)[0][0]
                else:
                    peak = np.nan
                    
                if plot_curate:
                    plt.plot(array, alpha=0.5)
                    plt.axvline(peak, linestyle='--', alpha=0.5)
                    plt.show()

                peaks.append(peak)
            peaks = np.array(peaks)*bs
            median_peak = np.nanmedian(peaks)
            
            if plot_curate:
                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(2,2), squeeze=False)
                if median_peak < 200:
                    max_r = 200
                else:
                    max_r = 400
                ax[0,0].hist(peaks, bins=25, range=(0, max_r))
                ax[0,0].axvline(median_peak-tolerance, color='grey', linestyle='--')
                ax[0,0].axvline(median_peak+tolerance, color='grey', linestyle='--')
                #plt.savefig(f'{figpath}/GC_peaks_{mi}_{mouse}D{day}.pdf')
                plt.show()

            # now check if the peaks are within 30cm of the median peak
            # also check if the rate is really low and should be considered
            for peak, cluster_id in zip(peaks, matrix_cluster_ids):
                if not np.abs(peak-median_peak)<(tolerance): # 30cm tolerance
                    module_ids.remove(cluster_id)
                    print(f'removing cluster {cluster_id} from module {mi} because peak {peak} is not within tolerance of median peak {median_peak}')
                elif nap.TsGroup([clusters_VR[cluster_id]]).rates[0] < 1:
                    module_ids.remove(cluster_id)
                    print(f'removing cluster {cluster_id} from module {mi} because rate is too low (below 1 Hz)')

            grid_module_cluster_ids[grid_module_ids.index(mi)] = module_ids

        # now plot the histogram of peaks again 
        for mi, module_ids in zip(grid_module_ids, grid_module_cluster_ids):
            matrix = np.array(list(autocorrs.values()))
            matrix_cluster_ids = np.array(list(autocorrs.keys()))
            cluster_id_of_interest = module_ids
            matrix = matrix[np.isin(matrix_cluster_ids, cluster_id_of_interest)]
            matrix_cluster_ids = matrix_cluster_ids[np.isin(matrix_cluster_ids, cluster_id_of_interest)]
            peaks = []
            for array in matrix:
                if len(find_peaks(array,prominence=prominence)[0])>0:
                    peak = find_peaks(array,prominence=prominence)[0][0]
                else:
                    peak = np.nan
                peaks.append(peak)
            peaks = np.array(peaks)*bs
            median_peak = np.nanmedian(peaks)

            if plot_curate:
                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(2,2), squeeze=False)
                if median_peak < 200:
                    max_r = 200
                else:
                    max_r = 400
                ax[0,0].hist(peaks, bins=25, range=(0, max_r))
                ax[0,0].axvline(median_peak-tolerance, color='grey', linestyle='--')
                ax[0,0].axvline(median_peak+tolerance, color='grey', linestyle='--')
                #plt.savefig(f'{figpath}/GC_peaks_{mi}_{mouse}D{day}_post_curated.pdf')
                plt.show()
            
    return  grid_module_ids, grid_module_cluster_ids


def plot_grid_modules_rate_maps(gcs, grid_module_ids, grid_module_cluster_ids, mouse, day, figpath, source_path=None):
    print(mouse, day)
    if source_path is None:
        source_path = '/Users/harryclark/Downloads/COHORT12/'
    session = 'OF1'
    of1_folder = f'{source_path}M{mouse}/D{day:02}/{session}/'
    shifted_grid_path = of1_folder + "tuning_scores/shifted_grid_score.parquet"
    spatial_path = of1_folder + "tuning_scores/shifted_spatial_information.parquet"
    spikes_path = of1_folder + f"sub-{mouse}_day-{day:02}_ses-{session}_srt-kilosort4_clusters.npz"
    beh_path = of1_folder + f"sub-{mouse}_day-{day:02}_ses-{session}_beh.nwb"
    shifted_grid_scores_of1 = pd.read_parquet(shifted_grid_path)
    spatial_information_score_of1 = pd.read_parquet(spatial_path)
    beh_OF = nap.load_file(beh_path)
    clusters_OF = nap.load_file(spikes_path)

    shifted_grid_scores_of1 = shifted_grid_scores_of1.query('travel >= 0')
    spatial_information_score_of1 = spatial_information_score_of1.query('travel >= 0')
    cluster_ids_values = shifted_grid_scores_of1.query('travel == 0').cluster_id

    ncols = 10
    rows_per_module = {mi: int(np.ceil(len(module) / ncols)) for mi, module in zip(grid_module_ids, grid_module_cluster_ids)}
    nrows = sum(rows_per_module.values())+len(grid_module_cluster_ids)+1
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 1*nrows), squeeze=False)
    row_counter = 0
    for mi, module_ids in zip(grid_module_ids, grid_module_cluster_ids):
        cells = gcs[gcs['cluster_id'].isin(module_ids)]
        print(f'for module {mi}, there are {len(cells)} cells')
        counter = 0
        for j in range(rows_per_module[mi]):
            for i in range(ncols):
                if counter < len(cells):
                    index = cells['cluster_id'].values[counter]
                    score = cells['grid_score'].values[counter]
                    cluster_shifted_grid_scores = shifted_grid_scores_of1[shifted_grid_scores_of1.cluster_id==index]
                    travel = cluster_shifted_grid_scores.travel.values[np.nanargmax(cluster_shifted_grid_scores.grid_score)]
                    max_score = cluster_shifted_grid_scores.grid_score.values[np.nanargmax(cluster_shifted_grid_scores.grid_score)]
                    field_spacing = cluster_shifted_grid_scores.field_spacing.values[np.nanargmax(cluster_shifted_grid_scores.grid_score)]
                    
                    tcs = {}    
                    position = np.stack([beh_OF['P_x'], beh_OF['P_y']], axis=1)
                    beh_lag = compute_travel_projected(["P_x", "P_y"], position, position, travel)
                    position_lagged = np.stack([beh_lag['P_x'], beh_lag['P_y']], axis=1)
                    for cell in cells['cluster_id'].values:
                        tc = nap.compute_2d_tuning_curves(nap.TsGroup([clusters_OF[cell]]), position_lagged, nb_bins=(40,40))[0]
                        tc = gaussian_filter_nan(tc[0], sigma=(2.5,2.5))
                        tcs[cell] = tc
                    #ax[row_counter, i].text(0,-2, f'id: {index}, mgs: {np.round(max_score, decimals=1)}', size=7)
                    #ax[row_counter, i].text(0,44, f'fs:{int(field_spacing)}', size=7)
                    ax[row_counter, i].imshow(tcs[index], cmap='jet')
                    counter+=1
            row_counter += 1
        row_counter += 1

    for axi in ax.flatten():
        axi.axis('off')
    plt.tight_layout()
    plt.savefig(f'{figpath}/M{mouse}D{day}_GC_rate_maps_modules.pdf', dpi=1000)
    plt.close()    


def compute_vr_tcs(mouse, day, apply_zscore=True, apply_guassian_filter=True, source_path=None, bs_t=None):
    if source_path is None:
        source_path = '/Users/harryclark/Downloads/COHORT12/'
    if bs_t is None:
        bs_t = time_bs

    vr_folder = f'{source_path}M{mouse}/D{day:02}/VR/'
    spikes_path = vr_folder + f"sub-{mouse}_day-{day:02}_ses-VR_srt-kilosort4_clusters.npz"
    beh_path = vr_folder + f"sub-{mouse}_day-{day:02}_ses-VR_beh.nwb"
    beh = nap.load_file(beh_path)
    clusters = nap.load_file(spikes_path)
    #print(f'there are this many clusters before curation {len(clusters)}')
    clusters = curate_clusters(clusters)

    tns = beh['trial_number']
    dt = beh['travel']-((tns[0]-1)*tl)
    n_bins = int(int(((np.ceil(np.nanmax(dt))//tl)+1)*tl)/bs)
    max_bound = int(((np.ceil(np.nanmax(dt))//tl)+1)*tl)
    min_bound = 0
    dt_bins = np.arange(0,max_bound,bs)

    # trick to clip the tc to around the end of the ephys recording
    # take the cell with the highest firing rate, and find the last bin with a spike
    # then work backwards and clip at the end of the last appropriate trials
    tc = nap.compute_1d_tuning_curves(nap.TsGroup([clusters[clusters.index[np.nanargmax(clusters.firing_rate)]]]), 
                                        dt, 
                                        nb_bins=n_bins, 
                                        minmax=[min_bound, max_bound],
                                        ep=beh["moving"])[0]
    
    tc = gaussian_filter(np.nan_to_num(tc).astype(np.float64), sigma=2.5)
    last_ephys_bin = int(np.nonzero(tc)[0][-1] + (tl/bs) - np.nonzero(tc)[0][-1]%(tl/bs))
    last_ephys_time_bin = clusters[clusters.index[0]].count(bin_size=bs_t, time_units = 'ms').index[-1]
    #print(f'last_ephys_bin {last_ephys_bin}')
    #print(f'last_ephys_time_bin {last_ephys_time_bin}')

    # time binned variables for later
    ep = nap.IntervalSet(start=0, end=last_ephys_time_bin, time_units = 's')
    speed_in_time = beh['S'].bin_average(bin_size=bs_t, time_units = 'ms', ep=ep)
    dt_in_time = beh['travel'].bin_average(bin_size=bs_t, time_units = 'ms', ep=ep)-((tns[0]-1)*tl)
    pos_in_time = dt_in_time%tl
    trial_number_in_time = (dt_in_time//tl)+tns[0]

    tcs = {}
    tcs_time = {}
    autocorrs = {}
    for cell in clusters.index:
        tc = nap.compute_1d_tuning_curves(nap.TsGroup([clusters[cell]]), 
                                        dt, 
                                        nb_bins=n_bins, 
                                        minmax=[min_bound, max_bound],
                                        ep=beh["moving"])[0]
        tc = np.array(tc)
        tc = np.nan_to_num(tc).astype(np.float64)
        if apply_guassian_filter:
            tc = gaussian_filter(tc, sigma=2.5)
        if apply_zscore:
            tc = zscore(tc)
        tc = tc[:last_ephys_bin] # only want bins with ephys data in it
        tcs[cell] = tc
        
        tc_time = clusters[cell].count(bin_size=bs_t, time_units = 'ms', ep=ep)
        tc_time = np.array(tc_time)
        tc_time = np.nan_to_num(tc_time).astype(np.float64)
        if apply_guassian_filter:
            tc_time = gaussian_filter(tc_time, sigma=2.5) # 
        if apply_zscore:
             tc_time = zscore(tc_time)
        tcs_time[cell] = tc_time

        lags = np.arange(0, 200, 1) # were looking at 10 timesteps back and 10 forward
        autocorr = []
        for lag in lags:
            if lag < 0:
                tc_offset = np.roll(tc, lag)
                tc_offset[lag:] = 0
            elif lag > 0:
                tc_offset = np.roll(tc, lag)
                tc_offset[:lag] = 0
            else:
                tc_offset = tc
            corr = stats.pearsonr(tc, tc_offset)[0]
            autocorr.append(corr)
        autocorr = np.array(autocorr)
        autocorrs[cell] = autocorr

    # drop beh trials from after last ephys bin
    beh_trials = beh['trials']
    beh_trials = beh_trials[:int(last_ephys_bin/(tl/bs))]

    return tcs, tcs_time, autocorrs, last_ephys_bin, beh, clusters


def get_time_binned_variables(mouse, day, apply_zscore=True, apply_guassian_filter=True, source_path=None, bs_t=None):
    if source_path is None:
        source_path = '/Users/harryclark/Downloads/COHORT12/'
    if bs_t is None:
        bs_t = time_bs

    vr_folder = f'{source_path}M{mouse}/D{day:02}/VR/'
    spikes_path = vr_folder + f"sub-{mouse}_day-{day:02}_ses-VR_srt-kilosort4_clusters.npz"
    beh_path = vr_folder + f"sub-{mouse}_day-{day:02}_ses-VR_beh.nwb"
    beh = nap.load_file(beh_path)
    clusters = nap.load_file(spikes_path)
    #print(f'there are this many clusters before curation {len(clusters)}')
    clusters = curate_clusters(clusters)

    tns = beh['trial_number']
    dt = beh['travel']-((tns[0]-1)*tl)
    n_bins = int(int(((np.ceil(np.nanmax(dt))//tl)+1)*tl)/bs)
    max_bound = int(((np.ceil(np.nanmax(dt))//tl)+1)*tl)
    min_bound = 0
    dt_bins = np.arange(0,max_bound,bs)

    # trick to clip the tc to around the end of the ephys recording
    # take the cell with the highest firing rate, and find the last bin with a spike
    # then work backwards and clip at the end of the last appropriate trials
    tc = nap.compute_1d_tuning_curves(nap.TsGroup([clusters[clusters.index[np.nanargmax(clusters.firing_rate)]]]), 
                                        dt, 
                                        nb_bins=n_bins, 
                                        minmax=[min_bound, max_bound],
                                        ep=beh["moving"])[0]
    
    tc = gaussian_filter(np.nan_to_num(tc).astype(np.float64), sigma=2.5)
    last_ephys_bin = int(np.nonzero(tc)[0][-1] + (tl/bs) - np.nonzero(tc)[0][-1]%(tl/bs))
    last_ephys_time_bin = clusters[clusters.index[0]].count(bin_size=bs_t, time_units = 'ms').index[-1]
    #print(f'last_ephys_bin {last_ephys_bin}')
    #print(f'last_ephys_time_bin {last_ephys_time_bin}')

    # time binned variables for later
    ep = nap.IntervalSet(start=0, end=last_ephys_time_bin, time_units = 's')
    speed_in_time = beh['S'].bin_average(bin_size=bs_t, time_units = 'ms', ep=ep)
    dt_in_time = beh['travel'].bin_average(bin_size=bs_t, time_units = 'ms', ep=ep)-((tns[0]-1)*tl)
    pos_in_time = dt_in_time%tl
    trial_number_in_time = (dt_in_time//tl)+tns[0]

    trial_type_in_time = []
    for tn in trial_number_in_time:
        trial = beh['trials'][beh['trials']['number'] == tn]
        trial_type_in_time.append(trial['type'].values[0])
    trial_type_in_time = np.array(trial_type_in_time)

    return speed_in_time, dt_in_time, pos_in_time, trial_number_in_time, trial_type_in_time


def get_sorted_trials_and_colors(beh, last_ephys_bin, tl, bs):
    sorted_cats = beh['trials'][:int(last_ephys_bin/(tl/bs))].groupby(by=['context','type','performance'])
    sorted_cats = sort_dict_by_priority(sorted_cats, trial_cat_priority)

    sorted_trial_indices = []
    sorted_trial_colors = []
    sorted_block_sizes = []
    for group, cat_indices in zip(sorted_cats.keys(), sorted_cats.values()):
        c = get_color_for_group(group)
        sorted_trial_colors.extend(np.repeat(c, len(cat_indices)).tolist())
        sorted_trial_indices.extend(cat_indices.tolist())
        sorted_block_sizes.append(len(cat_indices))
    sorted_trial_colors = np.array(sorted_trial_colors)
    sorted_trial_indices = np.array(sorted_trial_indices)
    return sorted_trial_indices, sorted_trial_colors

def get_trial_groups_and_colors(beh, last_ephys_bin, tl, bs):
    trial_colors = []
    trial_groups = []

    for trial in beh['trials'][:int(last_ephys_bin/(tl/bs))]:
        group=(trial['context'][0], 
            trial['type'][0],
            trial['performance'][0])
        c = get_color_for_group(group)
        group=''.join(group)
        trial_colors.append(c)
        trial_groups.append(group)
    trial_colors = np.array(trial_colors)
    trial_groups = np.array(trial_groups)
    return trial_groups, trial_colors


def plot_individual_rate_maps(mouse, day, cluster_ids, label='GC', figpath=''):
    tcs, _, _ , last_ephys_bin, beh, _ = compute_vr_tcs(mouse, day)
    trial_groups, trial_colors = get_trial_groups_and_colors(beh, last_ephys_bin, tl, bs)
    sorted_trial_indices, sorted_trial_colors = get_sorted_trials_and_colors(beh, last_ephys_bin, tl, bs)
 
    for id in cluster_ids:
        tc = tcs[id]
        tc = gaussian_filter(np.nan_to_num(tc).astype(np.float64), sigma=2.5)
        tc = zscore(tc)
        tc = tc[:last_ephys_bin] # only want bins with ephys data in it

        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=rm_figsize, width_ratios=[1,0.05], sharey=True)
        plot_firing_rate_map(ax[0], tc, bs=bs, tl=tl,p=95, sort_indices=None)
        ax[1].axis('off')
        ax[1].scatter(np.ones(len(trial_colors)), 
                    np.arange(0,len(trial_colors)), 
                    c = trial_colors,
                    marker='s')
        ax[0].set_xlabel('Pos (cm)')
        fig.savefig(f'{figpath}/M{mouse}D{day}{label}{id}.pdf', dpi=300, bbox_inches='tight')
        plt.show()

        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=rm_figsize, width_ratios=[1,0.05], sharey=True)
        plot_firing_rate_map(ax[0], tc, bs=bs, tl=tl,p=95, sort_indices=sorted_trial_indices)
        ax[1].axis('off')
        ax[1].scatter(np.ones(len(sorted_trial_colors)), 
                    np.arange(0,len(sorted_trial_colors)), 
                    c = sorted_trial_colors,
                    marker='s')
        ax[0].set_xlabel('Pos (cm)')
        fig.savefig(f'{figpath}/M{mouse}D{day}{label}{id}_sorted.pdf', dpi=300, bbox_inches='tight')
        plt.show()


def get_avg_profile(tc, bs=bs, tl=tl, mask=None):
    bpt = tl/bs
    n_trials = int(len(tc)/(bpt))
    trial_rate_map = []
    for i in range(n_trials):
        trial_rate_map.append(tc[int(i*bpt): int((i+1)*bpt)])
    trial_rate_map = np.array(trial_rate_map)

    if mask is None:
        return np.arange(bs/2, tl+(bs/2), bs), np.nanmean(trial_rate_map, axis=0)
    else:
        return np.arange(bs/2, tl+(bs/2), bs), np.nanmean(trial_rate_map[mask], axis=0)



def get_kmeans_spatial_labels(tc, labels, bs, tl):
    """
    Assign spatial labels using k-means labels allowed applied to the spectrogram.
    tc: 1D numpy array of shape (n_bins)
    labels: 1D numpy array of shape (n_windows) with k-means labels
    """
    bpt = tl/bs
    n_trials = int(len(tc)/(bpt))

    n_bins = len(tc)
    n_windows = len(labels)
    nperseg = 1600
    noverlap = 1400
    step = nperseg-noverlap  # nperseg - noverlap

    trial_centres = []
    for i in range(n_windows):
        start = i * step
        end = min(start + nperseg, n_bins)
        trial_centres.append(int(((start + end)/2)/bpt))
    trial_centres = np.array(trial_centres)

    kmean_trial_labels = np.full(n_trials, np.nan)
    for i in range(n_trials):
        trial_labels = labels[trial_centres == i+1]
        if len(trial_labels) != 0:

            modal_label = stats.mode(trial_labels, nan_policy='omit').mode
            kmean_trial_labels[i] = modal_label
 
    # Fill NaN values with the nearest non-NaN value
    for i in range(n_trials):
        if np.isnan(kmean_trial_labels[i]):
            # If the label is NaN, we need to find the nearest non-NaN value
            # walk left and right to find the nearest non-NaN value
            left = i - 1
            right = i + 1
            while left >= 0 and np.isnan(kmean_trial_labels[left]):
                left -= 1
            while right < n_trials and np.isnan(kmean_trial_labels[right]):
                right += 1
            if left >= 0 and right < n_trials:
                # Take the nearest non-NaN value
                if (i - left) <= (right - i):
                    kmean_trial_labels[i] = kmean_trial_labels[left]
                else:
                    kmean_trial_labels[i] = kmean_trial_labels[right]
            elif left >= 0:
                kmean_trial_labels[i] = kmean_trial_labels[left]
            elif right < n_trials:
                kmean_trial_labels[i] = kmean_trial_labels[right]

    return kmean_trial_labels

def plot_individual_rate_maps_with_avg_based_on_task_anchoring(mouse, day, cluster_ids, cluster_ids_for_spectrogram=None, label='GC', figpath='', source_path=None):
    if len(cluster_ids)==0:
        return

    if cluster_ids_for_spectrogram is None:
        cluster_ids_for_spectrogram = cluster_ids
    
    tcs, _, _ , last_ephys_bin, beh, _ = compute_vr_tcs(mouse, day, apply_zscore=False) 

    tcs_to_use = {cluster_id: tcs[cluster_id] for cluster_id in cluster_ids_for_spectrogram if cluster_id in tcs}
    results = spectral_analysis(tcs_to_use, tl, bs=bs)
    spectrograms = results[3] 
    S = spectrograms.mean(0)
    max_peaks = np.argmax(S, axis=0)
    labels = np.isin(max_peaks, [12, 28, 44, 60, 76]).astype(int) # these are the peaks that correspond to the task anchoring
    labels = np.isin(max_peaks, [28, 44, 60, 76]).astype(int) # these are the peaks that correspond to the task anchoring

    # translate the labels back to trial_numbers to use in the avging of the rate map

    for id in cluster_ids:
        tc = tcs[id]
        tc = gaussian_filter(np.nan_to_num(tc).astype(np.float64), sigma=2.5)
        tc = tc[:last_ephys_bin] # only want bins with ephys data in it
        tcz = zscore(tc)
        
        trial_labels = get_kmeans_spatial_labels(tcz, labels, bs=bs, tl=tl) # reuse kmeans function to get the labels based on the task anchoring
    
        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(rm_figsize[0], rm_figsize[1]*1.45), sharex=True, height_ratios=[0.3, 1], width_ratios=[1,0.05], sharey='row')
        plot_firing_rate_map(ax[1,0], tc, bs=bs, tl=tl, p=95, sort_indices=None)
        ax[1,1].axis('off')
        ax[0,1].axis('off')
        ax[1,1].scatter(np.ones(len(trial_labels)), 
                    np.arange(0,len(trial_labels)),
                    c=trial_labels,
                    cmap='cool',
                    marker='s',
                    vmin=0, 
                    vmax=1)
        ax[1,0].set_xlabel('Pos (cm)')
        cmap = plt.get_cmap('cool')

        for group in np.unique(trial_labels):
            if len(trial_labels[trial_labels == group])>5:
                x, y = get_avg_profile(tc, bs, tl, mask=trial_labels==group)
                ax[0,0].plot(x,y, color=cmap(group), linewidth=1)
        fig.savefig(f'{figpath}/M{mouse}D{day}{label}{id}_with_avg_task_anchoring.pdf', dpi=300, bbox_inches='tight')
        plt.close()


def plot_individual_rate_maps_with_avg_k_means_grouped_spectrogram(mouse, day, cluster_ids, label='GC', figpath='', source_path=None):
    if len(cluster_ids)==0:
        return
    tcs, _, _ , last_ephys_bin, beh, _ = compute_vr_tcs(mouse, day, apply_zscore=False) 

    tcs_to_use = {cluster_id: tcs[cluster_id] for cluster_id in cluster_ids if cluster_id in tcs}
    results = spectral_analysis(tcs_to_use, tl, bs=bs)
    spectrograms = results[3] 
    fvalid = results[5]

    S = spectrograms.mean(0)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto')
    labels = kmeans.fit_predict(S.T)
    
    # translate the labels back to trial_numbers to use in the avging of the rate map
 
    for id in cluster_ids:
        tc = tcs[id]
        tc = gaussian_filter(np.nan_to_num(tc).astype(np.float64), sigma=2.5)
        tc = tc[:last_ephys_bin] # only want bins with ephys data in it
        tcz = zscore(tc)
        
        trial_labels = get_kmeans_spatial_labels(tcz, labels, bs=bs, tl=tl)
    
        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(rm_figsize[0], rm_figsize[1]*1.45), sharex=True, height_ratios=[0.3, 1], width_ratios=[1,0.05], sharey='row')
        plot_firing_rate_map(ax[1,0], tc, bs=bs, tl=tl, p=95, sort_indices=None)
        ax[1,1].axis('off')
        ax[0,1].axis('off')
        ax[1,1].scatter(np.ones(len(trial_labels)), 
                    np.arange(0,len(trial_labels)),
                    c=trial_labels,
                    cmap='cool',
                    marker='s')
        ax[1,0].set_xlabel('Pos (cm)')
        cmap = plt.get_cmap('cool')

        for group in np.unique(trial_labels):
            if len(trial_labels[trial_labels == group])>5:
                x, y = get_avg_profile(tc, bs, tl, mask=trial_labels==group)
                ax[0,0].plot(x,y, color=cmap(group), linewidth=1)
        fig.savefig(f'{figpath}/M{mouse}D{day}{label}{id}_with_avg_kmeans_spectrogram.pdf', dpi=300, bbox_inches='tight')
        plt.close()




def plot_individual_rate_maps_with_avg(mouse, day, cluster_ids, label='GC', figpath=''):
    if len(cluster_ids)==0:
        return

    tcs, _, _ , last_ephys_bin, beh, _ = compute_vr_tcs(mouse, day, apply_zscore=False) 
    trial_groups, trial_colors = get_trial_groups_and_colors(beh, last_ephys_bin, tl, bs)
    sorted_trial_indices, sorted_trial_colors = get_sorted_trials_and_colors(beh, last_ephys_bin, tl, bs)
 
    for id in cluster_ids:
        tc = tcs[id]
        tc = gaussian_filter(np.nan_to_num(tc).astype(np.float64), sigma=2.5)
        tc = tc[:last_ephys_bin] # only want bins with ephys data in it
        tcz = zscore(tc)


        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(rm_figsize[0], rm_figsize[1]*1.45), sharex=True, height_ratios=[0.3, 1], width_ratios=[1,0.05], sharey='row')
        plot_firing_rate_map(ax[1,0], tc, bs=bs, tl=tl,p=95, sort_indices=None)
        ax[1,1].axis('off')
        ax[0,1].axis('off')
        ax[1,1].scatter(np.ones(len(trial_colors)), 
                    np.arange(0,len(trial_colors)), 
                    c = trial_colors,
                    marker='s')
        ax[1,0].set_xlabel('Pos (cm)')
        for group in np.unique(trial_groups):
            if len(trial_groups[trial_groups == group])>5:
                x, y = get_avg_profile(tc, bs, tl, mask=trial_groups==group)
                ax[0,0].plot(x,y, color=trial_colors[trial_groups==group][0], linewidth=1)
        fig.savefig(f'{figpath}/M{mouse}D{day}{label}{id}_with_avg.pdf', dpi=300, bbox_inches='tight')
        plt.close()


        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(rm_figsize[0], rm_figsize[1]*1.45), sharex=True, height_ratios=[0.3, 1], width_ratios=[1,0.05], sharey='row')
        plot_firing_rate_map(ax[1,0], tc, bs=bs, tl=tl,p=95, sort_indices=sorted_trial_indices)
        ax[1,1].axis('off')
        ax[0,1].axis('off')
        ax[1,1].scatter(np.ones(len(sorted_trial_colors)), 
                    np.arange(0,len(sorted_trial_colors)), 
                    c = sorted_trial_colors,
                    marker='s')
        ax[1,0].set_xlabel('Pos (cm)')
        for group in np.unique(trial_groups):
            if len(trial_groups[trial_groups == group])>5:
                x, y = get_avg_profile(tc, bs, tl, mask=trial_groups==group)
                ax[0,0].plot(x,y, color=trial_colors[trial_groups==group][0], linewidth=1)
        fig.savefig(f'{figpath}/M{mouse}D{day}{label}{id}_sorted_with_avg.pdf', dpi=300, bbox_inches='tight')
        plt.close()

def plot_stops_mouse_day(mouse, day, figpath, return_fig=True): 
    _, _, _ , last_ephys_bin, beh,_ = compute_vr_tcs(mouse, day)

    plot_stops(beh, tl=tl, sort=False, return_fig=return_fig, last_ephys_bin=last_ephys_bin,
           savepath=f'{figpath}/M{mouse}D{day}_stops')
    plot_stops(beh, tl=200, sort=True, return_fig=return_fig, last_ephys_bin=last_ephys_bin,
           savepath=f'{figpath}/M{mouse}D{day}_stops_sorted')
    

def plot_vr_rate_maps(mouse, day, cluster_ids, label, figpath):
    if len(cluster_ids)==0:
        return
    
    tcs,_,_,_,_,_ = compute_vr_tcs(mouse, day)
    ncols = 10
    nrows = int(np.ceil(len(cluster_ids)/ncols))
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 1.4*nrows), squeeze=False)
    counter = 0
    for j in range(nrows):
        for i in range(ncols):
            if counter<len(cluster_ids):
                index = cluster_ids[counter]
                plot_firing_rate_map(ax[j, i], 
                                    zscore(tcs[index]),
                                    bs=bs,
                                    tl=tl,
                                    p=95)
            else:
                ax[j, i].axis('off')
            counter+=1
            ax[j, i].set_xticks([])
            ax[j, i].set_yticks([])
            ax[j, i].xaxis.set_tick_params(labelbottom=False)
            ax[j, i].yaxis.set_tick_params(labelleft=False)
    plt.tight_layout()
    plt.savefig(f'{figpath}/M{mouse}D{day}_VR_rate_maps_{label}.pdf')
    plt.show()
    #plt.close()    


def plot_spectrogram(mouse, day, cluster_ids, label, figpath=None):
    if len(cluster_ids)==0:
        return
    tcs, _, _ , _, _, _ = compute_vr_tcs(mouse, day)
    tcs_to_use = {cluster_id: tcs[cluster_id] for cluster_id in cluster_ids if cluster_id in tcs}
    results = spectral_analysis(tcs_to_use, tl, bs=bs)
    spectrograms = results[3] 

    plt.figure(figsize=(4,4))
    S = spectrograms.mean(0)
    plt.imshow(S,origin='lower',aspect='auto',vmax=0.25,cmap='magma')
    plt.yticks([0, len(S)/2, len(S)], [0, 1, 2])
    plt.ylabel(f'Frequency (m-1)')
    plt.savefig(f'{figpath}/M{mouse}D{day}_spectrogram_{label}.pdf', dpi=300, bbox_inches='tight')
    plt.close()    


def plot_toroidal_projection(mouse, day, cluster_ids, figpath):
    if len(cluster_ids)==0:
        return
    
    tcs, _, _ , last_ephys_bin, beh,_ = compute_vr_tcs(mouse, day)
    tl=200

    tcs_to_use = {cluster_id: tcs[cluster_id] for cluster_id in cluster_ids if cluster_id in tcs}
    N = len(tcs_to_use)
    zmaps = np.array(list(tcs_to_use.values()))
    results = spectral_analysis(tcs_to_use, tl, bs=bs)
    f_modules =              results[0]
    phi_modules =            results[1]
    grid_cell_idxs_modules = results[2]
    spectrograms =           results[3]
    trial_starts =           results[6]
    grid_cell_idxs = grid_cell_idxs_modules[0]
    phi = phi_modules[0]
    Ng = len(grid_cell_idxs)
    maps = gaussian_filter1d(zmaps[grid_cell_idxs].reshape(Ng, -1), 2, axis=1)
    angles = np.arctan2(np.cos(phi)@maps, np.sin(phi)@maps)

    sorted_trial_indices, sorted_trial_colors = get_sorted_trials_and_colors(beh, last_ephys_bin, tl, bs) 
    trial_groups, trial_colors = get_trial_groups_and_colors(beh, last_ephys_bin, tl, bs)

    angles1 = angles[0].reshape(-1,int(tl/bs))
    angles2 = angles[1].reshape(-1,int(tl/bs))
    angles3 = angles[2].reshape(-1,int(tl/bs))

    for i, angles0 in  enumerate([angles1, angles2, angles3]):
        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(rm_figsize[0], rm_figsize[1]*1.45), sharex=True, height_ratios=[0.3, 1], width_ratios=[1,0.05], sharey='row')

        x = np.arange(1, len(angles0)+1)
        y = np.arange(0, len(angles0[0])*bs, bs)
        X, Y = np.meshgrid(x, y)
        heatmap = ax[1,0].pcolormesh(Y, X, angles0.T, shading='auto', cmap='hsv')
        heatmap.set_rasterized(True)
        ax[1,0].set_xlabel('Pos. (cm)')
        ax[1,1].axis('off')
        ax[0,1].axis('off')
        ax[1,1].scatter(np.ones(len(trial_colors)), 
                    np.arange(0,len(trial_colors)), 
                    c = trial_colors,
                    marker='s',s=1)
        for group in np.unique(trial_groups):
            if len(trial_groups[trial_groups == group])>5:
                x, y = get_avg_profile(angles[i], bs, tl, mask=trial_groups==group)
                ax[0,0].plot(x,y, color=trial_colors[trial_groups==group][0], linewidth=1)
        ax[1,0].set_xlim(0,tl)
        ax[1,0].set_ylim(0,len(angles0))
        ax[1,0].invert_yaxis()
        fig.savefig(f'{figpath}/M{mouse}D{day}A{i}_torus.pdf', dpi=300, bbox_inches='tight')
        plt.close()    


    angles1_sorted = angles[0].reshape(-1,int(tl/bs))[sorted_trial_indices]
    angles2_sorted = angles[1].reshape(-1,int(tl/bs))[sorted_trial_indices]
    angles3_sorted = angles[2].reshape(-1,int(tl/bs))[sorted_trial_indices]

    for i, angles0_sorted in enumerate([angles1_sorted, angles2_sorted, angles3_sorted]):
        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(rm_figsize[0], rm_figsize[1]*1.45), sharex=True, height_ratios=[0.3, 1], width_ratios=[1,0.05], sharey='row')

        x = np.arange(1, len(angles0_sorted)+1)
        y = np.arange(0, len(angles0_sorted[0])*bs, bs)
        X, Y = np.meshgrid(x, y)

        for group in np.unique(trial_groups):
            if len(trial_groups[trial_groups == group])>5:
                x, y = get_avg_profile(angles[i], bs, tl, mask=trial_groups==group)
                ax[0,0].plot(x,y, color=trial_colors[trial_groups==group][0], linewidth=1)

        heatmap = ax[1,0].pcolormesh(Y, X, angles0_sorted.T, shading='auto', cmap='hsv')
        heatmap.set_rasterized(True)
        ax[1,0].set_xlabel('Pos. (cm)')
        ax[1,1].axis('off')
        ax[0,1].axis('off')
        ax[1,1].scatter(np.ones(len(sorted_trial_colors)), 
                    np.arange(0,len(sorted_trial_colors)), 
                    c = sorted_trial_colors,
                    marker='s',s=1)
        ax[1,0].set_xlim(0,tl)
        ax[1,0].set_ylim(0,len(angles0_sorted))
        ax[1,0].invert_yaxis()
        fig.savefig(f'{figpath}/M{mouse}D{day}A{i}_torus_sorted.pdf', dpi=300, bbox_inches='tight')
        plt.close()    


def plot_decoding(mouse, day, cluster_ids, label, figpath):
    if len(cluster_ids)==0:
        return
    
    tcs, tcs_time, autocorrs, last_ephys_bin, beh, clusters = compute_vr_tcs(mouse, day)
    trial_groups, trial_colors = get_trial_groups_and_colors(beh, last_ephys_bin, tl, bs)
    sorted_trial_indices, sorted_trial_colors = get_sorted_trials_and_colors(beh, last_ephys_bin, tl, bs)

    print(len(sorted_trial_indices))

    tns = beh['trial_number']
    dt = beh['travel']-((tns[0]-1)*tl)
    n_bins = int(int(((np.ceil(np.nanmax(dt))//tl)+1)*tl)/bs)
    max_bound = int(((np.ceil(np.nanmax(dt))//tl)+1)*tl)
    min_bound = 0
    dt_bins = np.arange(0,max_bound,bs)

    x_true_dt = dt_bins[:last_ephys_bin]
    true_position = x_true_dt%tl
    trial_numbers = (x_true_dt//tl)+beh['trials']['number'][0]
    tns_to_decode_with = np.array(beh['trials']['number'])
    tns_to_decode_with = tns_to_decode_with[tns_to_decode_with<=np.nanmax(trial_numbers)]
    trial_types = np.array(beh['trials'][:int(last_ephys_bin/(tl/bs))]['type'])

    tns_to_decode = np.array(beh['trials']['number']) # decode all trials to visualise
    tns_to_train = np.array(beh['trials']['number'][(np.isin(beh['trials']['type'], np.array(['b','nb']))) &
                                                    (np.isin(beh['trials']['performance'], np.array(['hit'])))]) 
    tns_to_decode = tns_to_decode[tns_to_decode<=np.nanmax(trial_numbers)] # handles last ephys trials
    tns_to_train = tns_to_train[tns_to_train<=np.nanmax(trial_numbers)] # handles last ephys trials

    tcs = {cluster_id: tcs[cluster_id] for cluster_id in cluster_ids if cluster_id in tcs}

    predictions, errors = cross_validate_decoder(tcs, true_position, trial_numbers, tns_to_decode, tns_to_train, tl, bs, train=0.9, n=10, verbose=False)
    avg_predictions = circular_nanmean(predictions, tl, axis=2)

    sorted_predictions = predictions[sorted_trial_indices]
    sorted_errors = errors[sorted_trial_indices]

    avg_sorted_predictions = circular_nanmean(sorted_predictions, tl, axis=2)
    avg_sorted_errors = circular_nanmean(sorted_errors, tl, axis=2)
    
    # non sorted decoder
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(rm_figsize), width_ratios=[1,0.05], sharey=True)
    x = np.arange(1, len(avg_predictions)+1)
    y = np.arange(0, len(avg_predictions[0])*bs, bs)
    X, Y = np.meshgrid(x, y)
    heatmap = ax[0].pcolormesh(Y, X, avg_predictions.T, shading='auto', cmap='hsv')
    heatmap.set_rasterized(True)
    ax[0].set_xlabel('Pos. (cm)')
    ax[1].axis('off')
    ax[1].scatter(np.ones(len(trial_colors)), 
                    np.arange(0,len(trial_colors)), 
                    c = trial_colors,
                    marker='s',s=1)
    ax[0].set_xlim(0,tl)
    ax[0].set_ylim(0,len(avg_predictions))
    ax[0].invert_yaxis()
    fig.savefig(f'{figpath}/M{mouse}D{day}_Decoder_{label}.pdf', dpi=300, bbox_inches='tight')
    plt.close()    

    # sorted decoder
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(rm_figsize), width_ratios=[1,0.05], sharey=True)
    x = np.arange(1, len(avg_predictions)+1)
    y = np.arange(0, len(avg_predictions[0])*bs, bs)
    X, Y = np.meshgrid(x, y)
    heatmap = ax[0].pcolormesh(Y, X, avg_predictions[sorted_trial_indices].T, shading='auto', cmap='hsv')
    heatmap.set_rasterized(True)
    ax[0].set_xlabel('Pos. (cm)')
    ax[1].axis('off')
    ax[1].scatter(np.ones(len(sorted_trial_colors)), 
                    np.arange(0,len(sorted_trial_colors)), 
                    c = sorted_trial_colors,
                    marker='s',s=1)
    ax[0].set_xlim(0,tl)
    ax[0].set_ylim(0,len(avg_predictions))
    ax[0].invert_yaxis()
    fig.savefig(f'{figpath}/M{mouse}D{day}_Decoder_sorted_{label}.pdf', dpi=300, bbox_inches='tight')
    plt.close()    


    # sorted decoder with avg
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(rm_figsize[0], 1.45*rm_figsize[1]), 
                           height_ratios=[0.3, 1], width_ratios=[1,0.05], sharex=True, sharey='row')
    x = np.arange(1, len(avg_predictions)+1)
    y = np.arange(0, len(avg_predictions[0])*bs, bs)
    X, Y = np.meshgrid(x, y)
    heatmap = ax[1,0].pcolormesh(Y, X, avg_predictions[sorted_trial_indices].T, shading='auto', cmap='hsv')
    heatmap.set_rasterized(True)

    ax[0,0].plot(y,y, color='black', linestyle='dashed')
    ax[0,0].set_ylim(0,tl)
    ax[0,0].plot(y, circular_nanmean(avg_sorted_predictions[:len(trial_types[trial_types=='b'])], tl=tl, axis=0), color='tab:blue', linewidth=1)
    ax[0,0].plot(y, circular_nanmean(avg_sorted_predictions[len(trial_types[trial_types=='b']):], tl=tl, axis=0), color='tab:orange', linewidth=1)
    ax[1,0].set_xlabel('Pos. (cm)')
    ax[1,1].axis('off')
    ax[0,1].axis('off')
    ax[1,1].scatter(np.ones(len(sorted_trial_colors)), 
                    np.arange(0,len(sorted_trial_colors)), 
                    c = sorted_trial_colors,
                    marker='s',s=1)
    ax[1,0].set_xlim(0,tl)
    ax[1,0].set_ylim(0,len(avg_predictions))
    ax[1,0].invert_yaxis()
    fig.savefig(f'{figpath}/M{mouse}D{day}_Decoder_sorted_with_avg_{label}.pdf', dpi=300, bbox_inches='tight')
    plt.close()    

    
    norm = TwoSlopeNorm(vmin=-35, vcenter=0, vmax=35)
    # collated plot for assaying over training sets
    fig, ax = plt.subplots(
        2, 6, layout='constrained', figsize=(8.5*rm_figsize[0], 1.45*rm_figsize[1]), sharey=False, sharex=True, height_ratios=[0.3,1]
    ) 
    for i, train_set in enumerate(zip([['b', 'nb'], ['b'], ['nb']])):
        tns_to_decode = np.array(beh['trials']['number']) # decode all trials to visualise
        tns_to_train = np.array(beh['trials']['number'][np.isin(beh['trials']['type'], np.array(train_set)) &
                                                        np.isin(beh['trials']['performance'], np.array('hit'))]) 
        tns_to_decode = tns_to_decode[tns_to_decode<=np.nanmax(trial_numbers)] # handles last ephys trials
        tns_to_train = tns_to_train[tns_to_train<=np.nanmax(trial_numbers)] # handles last ephys trials

        predictions, errors = cross_validate_decoder(tcs, true_position, trial_numbers, tns_to_decode, tns_to_train,tl,bs,train=0.9, n=50)
        sorted_predictions = predictions[np.argsort(trial_types)]
        sorted_errors = errors[np.argsort(trial_types)]

        avg_sorted_predictions = circular_nanmean(sorted_predictions, tl, axis=2)
        avg_sorted_errors = np.nanmean(sorted_errors, axis=2)
        sem_sorted_errors = stats.sem(sorted_errors, axis=2)
        
        x = np.arange(1, len(avg_sorted_predictions)+1)
        y = np.arange(0, len(avg_sorted_predictions[0])*bs, bs)
        X, Y = np.meshgrid(x, y)

        #ax[0,i*2].set_title(f'train:{train_set}')
        ax[0,i*2].plot(np.arange(bs/2,(tl+bs/2),bs),np.arange(bs/2,(tl+bs/2),bs), color='black', linestyle='dashed')
        ax[0,i*2].plot(np.arange(bs/2,(tl+bs/2),bs), circular_nanmean(avg_sorted_predictions[:len(trial_types[trial_types=='b'])], tl=tl, axis=0), color='tab:blue')
        ax[0,i*2].plot(np.arange(bs/2,(tl+bs/2),bs), circular_nanmean(avg_sorted_predictions[len(trial_types[trial_types=='b']):], tl=tl, axis=0), color='tab:orange')
        heatmap1 = ax[1,i*2].pcolormesh(Y, X, avg_sorted_predictions.T, shading='auto', cmap='hsv')
        heatmap1.set_rasterized(True)
        ax[1,i*2].axhline(y=len(trial_types[trial_types=='b']), color='black')
        b_error = np.nanmean(avg_sorted_errors[:len(trial_types[trial_types=='b'])], axis=0)
        nb_error = np.nanmean(avg_sorted_errors[len(trial_types[trial_types=='b']):], axis=0)
        ax[0,(i*2)+1].plot(np.arange(bs/2,(tl+bs/2),bs), b_error, color='tab:blue')
        ax[0,(i*2)+1].plot(np.arange(bs/2,(tl+bs/2),bs), nb_error, color='tab:orange')

        heatmap2 = ax[1,(i*2)+1].pcolormesh(Y, X, avg_sorted_errors.T, shading='auto', cmap='Purples')
        heatmap2.set_rasterized(True)
        ax[1,(i*2)+1].axhline(y=len(trial_types[trial_types=='b']), color='black')
    for i in range(6):
        ax[1,i].invert_yaxis()
        ax[1,i].set_xlabel(f'Pos (cm)')
        ax[0,i].set_xlim(0,tl)
        ax[1,i].set_xlim(0,tl)
        if i != 0:
            ax[1,i].set_yticklabels([])
    fig.savefig(f'{figpath}/M{mouse}D{day}_different_train_sets_{label}.pdf', dpi=300, bbox_inches='tight')
    plt.close()    



def plot_projected_stops(mouse, day, cluster_ids, label, figpath):
    if len(cluster_ids)==0:
        return
    
    tcs, tcs_time, autocorrs, last_ephys_bin, beh, clusters = compute_vr_tcs(mouse, day)
    trial_groups, trial_colors = get_trial_groups_and_colors(beh, last_ephys_bin, tl, bs)
    sorted_trial_indices, sorted_trial_colors = get_sorted_trials_and_colors(beh, last_ephys_bin, tl, bs)

    tns = beh['trial_number']
    dt = beh['travel']-((tns[0]-1)*tl)
    n_bins = int(int(((np.ceil(np.nanmax(dt))//tl)+1)*tl)/bs)
    max_bound = int(((np.ceil(np.nanmax(dt))//tl)+1)*tl)
    min_bound = 0
    dt_bins = np.arange(0,max_bound,bs)
    x_true_dt = dt_bins[:last_ephys_bin]
    trial_numbers = (x_true_dt//tl)+beh['trials']['number'][0]

    last_ephys_time_bin = clusters[clusters.index[0]].count(bin_size=time_bs, time_units = 'ms').index[-1]

    # time binned variables for later
    ep = nap.IntervalSet(start=0, end=last_ephys_time_bin, time_units = 's')
    speed_in_time = beh['S'].bin_average(bin_size=time_bs, time_units = 'ms', ep=ep)
    dt_in_time = beh['travel'].bin_average(bin_size=time_bs, time_units = 'ms', ep=ep)-((tns[0]-1)*tl)
    time_vals = np.arange(0, len(speed_in_time)*(time_bs/1000),time_bs/1000) # secs
    pos_in_time = dt_in_time%tl
    trial_number_in_time = (dt_in_time//tl)+tns[0]

    # decoding in time
    tcs_time = {cluster_id: tcs_time[cluster_id] for cluster_id in cluster_ids if cluster_id in tcs_time}

    speed_in_time = np.array(speed_in_time)
    pos_in_time = np.array(pos_in_time) 
    trial_number_in_time = np.array(trial_number_in_time)
    dt_in_time = np.array(dt_in_time)
    tns_to_decode_with = np.array(beh['trials']['number'])
    tns_to_decode_with = tns_to_decode_with[tns_to_decode_with<=np.nanmax(trial_numbers)]
    trial_types = np.array(beh['trials']['type'])

    tns_to_decode = np.array(beh['trials']['number']) # decode all trials to visualise
    tns_to_train = np.array(beh['trials']['number'][np.isin(beh['trials']['type'], np.array(['b','nb']))]) 
    tns_to_decode = tns_to_decode[tns_to_decode<=np.nanmax(trial_numbers)] # handles last ephys trials
    tns_to_train = tns_to_train[tns_to_train<=np.nanmax(trial_numbers)] # handles last ephys trials

    predictions_in_time, errors_in_time = cross_validate_decoder_time(tcs_time, 
                                                    true_position=pos_in_time, 
                                                    trial_numbers=trial_number_in_time, 
                                                    tns_to_decode=tns_to_decode, 
                                                    tns_to_train=tns_to_train, 
                                                    tl=tl, bs=bs, train=0.9, n=10, verbose=False)

    avg_predictions_in_time = [np.mean(np.stack(preds_n), axis=0) for preds_n in predictions_in_time]
    avg_predictions_in_time = np.concatenate(avg_predictions_in_time).ravel()

    # create new stop mask
    interpf = interp1d(time_vals, avg_predictions_in_time, kind='nearest', fill_value=np.nan, bounds_error=False)
    avg_predictions_in_time_interp = interpf(dt.index)
    projected_stops_mask = (avg_predictions_in_time_interp>90) & (avg_predictions_in_time_interp<110)

    plot_stops(beh, savepath=f'{figpath}/M{mouse}D{day}_stops'+label,tl=tl,sort=False,return_fig=False,stop_mask=projected_stops_mask, last_ephys_bin=last_ephys_bin)
    plot_stops(beh, savepath=f'{figpath}/M{mouse}D{day}_stops_sorted'+label,tl=tl,sort=True,return_fig=False,stop_mask=projected_stops_mask, last_ephys_bin=last_ephys_bin)
    return




def compare_decodings(mouse, day, cluster_ids_1, cluster_ids_2, label1='', label2='', figpath=''):
    print(f'for {label1}, there are {len(cluster_ids_1)} cells being used for decoding')
    print(f'for {label2}, there are {len(cluster_ids_2)} cells being used for decoding')
    tcs, tcs_time, autocorrs, last_ephys_bin, beh, clusters = compute_vr_tcs(mouse, day)
    trial_groups, trial_colors = get_trial_groups_and_colors(beh, last_ephys_bin, tl, bs)
    sorted_trial_indices, sorted_trial_colors = get_sorted_trials_and_colors(beh, last_ephys_bin, tl, bs)

    tns = beh['trial_number']
    dt = beh['travel']-((tns[0]-1)*tl)
    n_bins = int(int(((np.ceil(np.nanmax(dt))//tl)+1)*tl)/bs)
    max_bound = int(((np.ceil(np.nanmax(dt))//tl)+1)*tl)
    min_bound = 0
    dt_bins = np.arange(0,max_bound,bs)
 
    x_true_dt = dt_bins[:last_ephys_bin]
    true_position = x_true_dt%tl
    trial_numbers = (x_true_dt//tl)+beh['trials']['number'][0]
    tns_to_decode_with = np.array(beh['trials']['number'])
    tns_to_decode_with = tns_to_decode_with[tns_to_decode_with<=np.nanmax(trial_numbers)]
    trial_types = np.array(beh['trials']['type'])

    tns_to_decode = np.array(beh['trials']['number']) # decode all trials to visualise
    tns_to_train = np.array(beh['trials']['number'][np.isin(beh['trials']['type'], np.array(['b','nb']))]) 
    tns_to_decode = tns_to_decode[tns_to_decode<=np.nanmax(trial_numbers)] # handles last ephys trials
    tns_to_train = tns_to_train[tns_to_train<=np.nanmax(trial_numbers)] # handles last ephys trials
    trial_types = trial_types[:len(tns_to_decode)]

    tcs_1 = {cluster_id: tcs[cluster_id] for cluster_id in cluster_ids_1 if cluster_id in tcs}
    predictions_1, errors = cross_validate_decoder(tcs_1, true_position, trial_numbers, tns_to_decode, tns_to_train, tl, bs, train=0.9, n=10, verbose=False)
    avg_predictions_1 = circular_nanmean(predictions_1, tl, axis=2)

    tcs_2 = {cluster_id: tcs[cluster_id] for cluster_id in cluster_ids_2 if cluster_id in tcs}
    predictions_2, errors = cross_validate_decoder(tcs_2, true_position, trial_numbers, tns_to_decode, tns_to_train, tl, bs, train=0.9, n=10, verbose=False)
    avg_predictions_2 = circular_nanmean(predictions_2, tl, axis=2)

    delta = avg_predictions_1-avg_predictions_2

    norm = TwoSlopeNorm(vmin=-35, vcenter=0, vmax=35)
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(0.8, 2), width_ratios=[1,0.05], sharey=True)
    x = np.arange(1, len(avg_predictions_1)+1)
    y = np.arange(0, len(avg_predictions_1[0])*bs, bs)
    X, Y = np.meshgrid(x, y)
    heatmap = ax[0].pcolormesh(Y, X, delta.T, shading='auto', cmap='bwr', norm=norm)
    heatmap.set_rasterized(True)
    ax[0].set_xlabel('Pos. (cm)')
    ax[1].axis('off')
    ax[1].scatter(np.ones(len(trial_colors)), 
                  np.arange(0,len(trial_colors)), 
                  c = trial_colors,
                  marker='s')
    ax[0].set_xlim(0,tl)
    ax[0].set_ylim(0,len(avg_predictions_1))
    ax[0].invert_yaxis()
    fig.savefig(f'{figpath}/compare_decoders_{label1}_{label2}_M{mouse}D{day}.pdf', dpi=300, bbox_inches='tight')


    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(0.8, 2), width_ratios=[1,0.05], sharey=True)
    x = np.arange(1, len(avg_predictions_1)+1)
    y = np.arange(0, len(avg_predictions_1[0])*bs, bs)
    X, Y = np.meshgrid(x, y)
    heatmap = ax[0].pcolormesh(Y, X, delta[sorted_trial_indices].T, shading='auto', cmap='bwr', norm=norm)
    heatmap.set_rasterized(True)
    ax[0].set_xlabel('Pos. (cm)')
    ax[1].axis('off')
    ax[1].scatter(np.ones(len(sorted_trial_colors)), 
                  np.arange(0,len(sorted_trial_colors)), 
                  c = sorted_trial_colors,
                  marker='s')
    ax[0].set_xlim(0,tl)
    ax[0].set_ylim(0,len(avg_predictions_1))
    ax[0].invert_yaxis()
    fig.savefig(f'{figpath}/compare_decoders_{label1}_{label2}_M{mouse}D{day}_sorted.pdf', dpi=300, bbox_inches='tight')

    avg_b_delta = np.nanmean(delta[sorted_trial_indices][:len(trial_types[trial_types=='b'])], axis=0)
    avg_nb_delta = np.nanmean(delta[sorted_trial_indices][len(trial_types[trial_types=='b']):], axis=0)
    sem_b_delta = stats.sem(delta[sorted_trial_indices][:len(trial_types[trial_types=='b'])], axis=0, nan_policy='omit')
    sem_nb_delta = stats.sem(delta[sorted_trial_indices][len(trial_types[trial_types=='b']):], axis=0, nan_policy='omit')

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(7, 2), width_ratios=[1,1], sharey=True)
    y = np.arange(0, len(avg_b_delta)*bs, bs)
    ax[0].set_xlabel('Pos. (cm)')
    ax[1].set_xlabel('Pos. (cm)')
    ax[0].set_ylabel('delta (cm)')
    ax[0].set_xlim(0,tl)
    ax[0].plot(y, avg_b_delta, color='tab:blue')
    ax[0].plot(y, avg_nb_delta, color='tab:orange')
    ax[0].fill_between(y, avg_b_delta+sem_b_delta, avg_b_delta-sem_b_delta, color='tab:blue', alpha=0.3)
    ax[0].fill_between(y, avg_nb_delta+sem_nb_delta, avg_nb_delta-sem_nb_delta, color='tab:orange', alpha=0.3)
    ax[1].plot(y, avg_b_delta-avg_nb_delta, color='black')
    fig.savefig(f'{figpath}/compare_decoders_{label1}_{label2}_M{mouse}D{day}_diff.pdf', dpi=300, bbox_inches='tight')


    # compare decoding accuracies
    unique_trial_groups = np.unique(trial_groups)
    print(unique_trial_groups)
    print(trial_groups.shape)
    print(trial_groups[0])
    
    decoding_accuracy = {}
    for group_label in unique_trial_groups:
        group_mask = trial_groups==group_label
        group_decoding_1 = avg_predictions_1[group_mask]
        group_decoding_2 = avg_predictions_2[group_mask]

        true_locations = np.tile(np.arange(0,tl,bs), len(group_decoding_1)).reshape((len(group_decoding_1), int(tl/bs)))
        errors_1 = true_locations - group_decoding_1
        errors_2 = true_locations - group_decoding_2
        errors_1[errors_1>(tl*0.75)] = tl-errors_1[errors_1>(tl*0.75)]
        errors_1[errors_1<(-tl*0.75)] = tl+errors_1[errors_1<(-tl*0.75)]
        errors_2[errors_2>(tl*0.75)] = tl-errors_2[errors_2>(tl*0.75)]
        errors_2[errors_2<(-tl*0.75)] = tl+errors_2[errors_2<(-tl*0.75)]
        errors_1 = np.abs(errors_1)
        errors_2 = np.abs(errors_2)
        trial_errors_1 = np.nanmean(errors_1,axis=0)
        trial_errors_2 = np.nanmean(errors_2,axis=0)
        decoding_accuracy[label1+group_label] = trial_errors_1
        decoding_accuracy[label2+group_label] = trial_errors_2

    # Create a violin plot
    plt.figure(figsize=(5, 2))
    plt.violinplot([decoding_accuracy[key] for key in decoding_accuracy.keys()], showmeans=True,showmedians=True)
    plt.xticks(np.arange(len(decoding_accuracy.keys())), decoding_accuracy.keys(), rotation=30)
    plt.xlabel('Trial Groups')
    plt.ylabel('Mean Decoding Error (cm)')
    plt.tight_layout()
    plt.show()


    decoding_accuracy = {}
    differences = {}
    for tt_label in ['b', 'nb']:
        tt_mask = trial_types==tt_label
        print(f'np.unque(tt_mask) {np.unique(trial_types)}')
        print(f'shape(tt_mask) {np.shape(tt_mask)}')
        print(f'sum(tt_mask) {np.sum(tt_mask)}')

        decoding_1 = avg_predictions_1[tt_mask]
        decoding_2 = avg_predictions_2[tt_mask]
        diff = decoding_2-decoding_1
        diff[diff>(tl*0.5)] = tl-diff[diff>(tl*0.5)]
        diff[diff<(-tl*0.5)] = tl+diff[diff<(-tl*0.5)]

        true_locations = np.tile(np.arange(0,tl,bs), len(decoding_1)).reshape((len(decoding_1), int(tl/bs)))
        print(f'shape(true_locations) {np.shape(true_locations)}')
        print(f'shape(decoding_1) {np.shape(decoding_1)}')

        errors_1 = true_locations - decoding_1
        errors_2 = true_locations - decoding_2
        errors_1[errors_1>(tl*0.5)] = tl-errors_1[errors_1>(tl*0.5)]
        errors_1[errors_1<(-tl*0.5)] = tl+errors_1[errors_1<(-tl*0.5)]
        errors_2[errors_2>(tl*0.5)] = tl-errors_2[errors_2>(tl*0.5)]
        errors_2[errors_2<(-tl*0.5)] = tl+errors_2[errors_2<(-tl*0.5)]
        errors_1 = np.abs(errors_1)
        errors_2 = np.abs(errors_2)

        trial_errors_1 = np.nanmean(errors_1,axis=0)
        trial_errors_2 = np.nanmean(errors_2,axis=0)
        decoding_accuracy[label1+'_'+tt_label] = trial_errors_1
        decoding_accuracy[label2+'_'+tt_label] = trial_errors_2
        differences[tt_label] = np.nanmean(diff, axis=0)

    # Create a violin plot
    plt.figure(figsize=(2, 2))
    plt.violinplot([decoding_accuracy[key] for key in decoding_accuracy.keys()], showmeans=True,showmedians=True)
    #plt.xticks(np.arange(1,1+len(decoding_accuracy.keys())), decoding_accuracy.keys(), rotation=30)
    plt.xlabel('Trial Groups')
    plt.ylabel('Mean Decoding Error (cm)')
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(2, 2))
    plt.violinplot([differences[key] for key in differences.keys()], showmeans=True,showmedians=True)
    #plt.xticks(np.arange(1, 1+len(differences.keys())), differences.keys(), rotation=30)
    plt.xlabel('Trial Groups')
    plt.ylabel('Diff in Decoding (cm)')
    plt.tight_layout()
    plt.show()


def extract_border(image, color):
    mask = np.all(image == color, axis=-1)

    rows, cols = mask.shape
    border_mask = np.copy(mask)
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if mask[i, j] == 1:
                neighbors = np.array([
                    mask[i-1, j-1], mask[i-1, j], mask[i-1, j+1],
                    mask[i, j-1],                   mask[i, j+1],
                    mask[i+1, j-1], mask[i+1, j], mask[i+1, j+1]
                ])
                if np.all(neighbors):
                    border_mask[i, j] = 0

    border_points = np.column_stack(np.where(border_mask == 1))
    border_points = np.column_stack(np.where(border_mask == 1))

    return border_points



def get_task_anchored_labels_in_time(mouse, day, cluster_ids_for_spectrogram):
    tcs, tcs_time, autocorrs, last_ephys_bin, beh, clusters = compute_vr_tcs(mouse, day, apply_zscore=False) # recompute tcs without z-scoring to use in the spectrogram
    tcs_to_use = {cluster_id: tcs[cluster_id] for cluster_id in cluster_ids_for_spectrogram if cluster_id in tcs}
    results = spectral_analysis(tcs_to_use, tl, bs=bs)
    spectrograms = results[3] 
    S = spectrograms.mean(0)
    max_peaks = np.argmax(S, axis=0)
    labels = np.isin(max_peaks, [12, 28, 44, 60, 76]).astype(int) # these are the peaks that correspond to the task anchoring
    trial_labels = get_kmeans_spatial_labels(tcs[cluster_ids_for_spectrogram[0]], labels, bs=bs, tl=tl) # reuse kmeans function to get the labels based on the task anchoring

    last_ephys_time_bin = clusters[clusters.index[0]].count(bin_size=time_bs, time_units = 'ms').index[-1]

    # time binned variables for later
    ep = nap.IntervalSet(start=0, end=last_ephys_time_bin, time_units = 's')
    speed_in_time = beh['S'].bin_average(bin_size=time_bs, time_units = 'ms', ep=ep)
    dt_in_time = beh['travel'].bin_average(bin_size=time_bs, time_units = 'ms', ep=ep)-((beh['trial_number'][0]-1)*tl)
    time_vals = np.arange(0, len(speed_in_time)*(time_bs/1000),time_bs/1000) # secs
    pos_in_time = dt_in_time%tl
    trial_number_in_time = (dt_in_time//tl)+beh['trial_number'][0]

    trial_number_in_time = pd.Series(np.array(trial_number_in_time)).fillna(method='ffill').fillna(method='bfill').to_numpy()
    trial_labels_trial_numbers = np.arange(trial_number_in_time[0], len(trial_labels)+trial_number_in_time[0], 1)
    trial_labels_in_time = np.zeros(len(trial_number_in_time), dtype=int)
    assert len(trial_labels) == len(trial_labels_trial_numbers), f'len(trial_labels)={len(trial_labels)} != len(trial_labels_trial_numbers)={len(trial_labels_trial_numbers)}'

    for i, tn in enumerate(trial_number_in_time.astype(int)):
        trial_labels_in_time[i] = trial_labels[np.where(trial_labels_trial_numbers == tn)[0][0]].astype(int)
    
    return trial_labels_in_time


from astropy.convolution import convolve, Gaussian1DKernel
def cross_correlation_with_jitter(
    cell1_spikes, cell2_spikes, bin_size=1, max_lag_ms=100, jitter_window_ms=5, n_jitters=1000, duration_s=2000, gauss_sigma_bins=1
):
    """
    Compute cross-correlation between two spike trains and assess significance by jittering cell2_spikes.
    Applies Gaussian smoothing to binned spike trains.
    Returns:
        lags, original_cc, jittered_ccs (n_jitters x len(lags)), pvals (per lag),
        optimal_lag (ms), optimal_corr (value at optimal lag)
    """
    n_bins = int(duration_s * (1000/bin_size))
    # Bin both spike trains
    binned1, _ = np.histogram(cell1_spikes, bins=n_bins, range=(0, duration_s))
    binned2, _ = np.histogram(cell2_spikes, bins=n_bins, range=(0, duration_s))
    # Gaussian smoothing
    gauss_kernel = Gaussian1DKernel(gauss_sigma_bins)
    binned1 = convolve(binned1, gauss_kernel)
    binned2 = convolve(binned2, gauss_kernel)
    # Compute original cross-correlation
    max_lag_bins = int(max_lag_ms // bin_size)
    lags = np.arange(-max_lag_bins, max_lag_bins + 1)
    original_cc = correlate(binned1, binned2, mode='full')
    center = len(original_cc) // 2
    original_cc = original_cc[center - max_lag_bins : center + max_lag_bins + 1]
    # Jitter cell2_spikes and compute surrogate cross-correlations
    jittered_ccs = []
    for _ in range(n_jitters):
        jittered_spikes = cell2_spikes + np.random.uniform(-jitter_window_ms, jitter_window_ms, size=cell2_spikes.shape)
        jittered_spikes = np.clip(jittered_spikes, 0, duration_s)
        binned2_jit, _ = np.histogram(jittered_spikes, bins=n_bins, range=(0, duration_s))
        binned2_jit = convolve(binned2_jit, gauss_kernel)
        cc = correlate(binned1, binned2_jit, mode='full')
        cc = cc[center - max_lag_bins : center + max_lag_bins + 1]
        jittered_ccs.append(cc)
    jittered_ccs = np.array(jittered_ccs)
    pvals = np.mean(jittered_ccs >= original_cc, axis=0)
    optimal_idx = np.argmax(original_cc)
    optimal_lag = lags[optimal_idx] * bin_size
    optimal_corr = original_cc[optimal_idx]


    return lags * bin_size, original_cc, jittered_ccs, pvals, optimal_lag, optimal_corr


def cross_correlation_with_jitter(
    cell1_spikes, cell2_spikes, bin_size=1, max_lag_ms=100, jitter_window_ms=5, 
    n_jitters=1000, duration_s=2000, gauss_sigma_bins=1, plot=True
):
    """
    Compute cross-correlation between two spike trains and assess significance by jittering cell2_spikes.
    Applies Gaussian smoothing to binned spike trains.
    Returns:
        lags, original_cc, jittered_ccs (n_jitters x len(lags)), pvals (per lag),
        optimal_lag (ms), optimal_corr (value at optimal lag), noise_correlation
    """

    def normalized_cross_correlation(x, y, max_lag_bins=max_lag_ms // bin_size):
        x = (x - np.mean(x)) / (np.std(x) + 1e-10)
        y = (y - np.mean(y)) / (np.std(y) + 1e-10)
        cc = correlate(x, y, mode='full')
        center = len(cc) // 2
        cc = cc[center - max_lag_bins : center + max_lag_bins + 1]
        cc = cc / len(x)  # Normalize by number of samples
        return cc

    n_bins = int(duration_s * (1000/bin_size))
    # Bin both spike trains
    binned1, _ = np.histogram(cell1_spikes, bins=n_bins, range=(0, duration_s))
    binned2, _ = np.histogram(cell2_spikes, bins=n_bins, range=(0, duration_s))
    # Gaussian smoothing
    gauss_kernel = Gaussian1DKernel(gauss_sigma_bins)
    binned1 = convolve(binned1, gauss_kernel)
    binned2 = convolve(binned2, gauss_kernel)
    # Compute original cross-correlation
    max_lag_bins = int(max_lag_ms // bin_size)
    lags = np.arange(-max_lag_bins, max_lag_bins + 1)
    original_cc = normalized_cross_correlation(binned1, binned2)
    center = len(original_cc) // 2
    original_cc = original_cc[center - max_lag_bins : center + max_lag_bins + 1]

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 1.7))
        plt.plot(lags * bin_size, original_cc, label='Original CC', color='blue')
        plt.title('Original Cross-Correlation')
        plt.xlabel('Lag (ms)')
        plt.ylabel('Correlation')
        plt.axhline(0, color='black', linestyle='--')
        plt.legend()
        plt.show()

    # Jitter cell2_spikes and compute surrogate cross-correlations
    jittered_ccs = []
    jittered_max_corrs = []
    for i in range(n_jitters):
        if i % 10 == 0:
            print(f'Jitter iteration {i+1}/{n_jitters}')
        jittered_spikes = cell2_spikes + np.random.uniform(-jitter_window_ms, jitter_window_ms, size=cell2_spikes.shape)
        jittered_spikes = np.clip(jittered_spikes, 0, duration_s)
        binned2_jit, _ = np.histogram(jittered_spikes, bins=n_bins, range=(0, duration_s))
        binned2_jit = convolve(binned2_jit, gauss_kernel)
        cc = normalized_cross_correlation(binned1, binned2_jit)
        cc = cc[center - max_lag_bins : center + max_lag_bins + 1]
        jittered_ccs.append(cc)
        jittered_max_corrs.append(np.max(cc))
    jittered_ccs = np.array(jittered_ccs)

    if plot:
        # plot 4 jittered cross-correlations in a subplot nrows=4
        fig, ax = plt.subplots(ncols=1, nrows=4, figsize=(8, 5), squeeze=False,sharex=True)
        ax[0, 0].set_title(f'Jittered Cross-Correlations')
        for j in [0,1,2,3]:
            ax[j, 0].plot(lags * bin_size, jittered_ccs[j], color='red', alpha=1)
            ax[j, 0].set_xlabel('Lag (ms)')
            ax[j, 0].set_ylabel('Correlation')
            ax[j, 0].axhline(0, color='black', linestyle='--')
        plt.tight_layout()
        plt.show()

    jittered_max_corrs = np.array(jittered_max_corrs)
    pvals = np.mean(jittered_ccs >= original_cc, axis=0)
    optimal_idx = np.argmax(original_cc)
    optimal_lag = lags[optimal_idx] * bin_size
    optimal_corr = original_cc[optimal_idx]
    # Noise correlation: original max - median jittered max
    noise_correlation = optimal_corr - np.median(jittered_max_corrs)

    if plot:
        plt.figure(figsize=(3,3))
        plt.hist(jittered_max_corrs, bins=30, alpha=0.5, color='gray', label='Jittered Max Correlations')
        plt.axvline(np.median(jittered_max_corrs), color='red', linestyle='--', label='Median Jittered Max Correlation')
        plt.axvline(optimal_corr, color='blue', linestyle='--', label='Original Max Correlation')
        plt.title('Distribution of Jittered Max Correlations')
        plt.xlabel('Max Correlation')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    return lags * bin_size, original_cc, jittered_ccs, pvals, optimal_lag, optimal_corr, noise_correlation


def load_cluster_locations(clusters, cells):
    for column in ['coord_SCs_x', 
                   'coord_SCs_y', 
                   'coord_SCs_z', 
                   'coord_probe_x', 
                   'coord_probe_y',
                   'brain_region']:
        vals = []
        for id in cells.cluster_id:
            vals.append(clusters[column][id])
        cells[column] = vals
    return cells