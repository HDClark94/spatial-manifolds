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

import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.cm as cm
from scipy.signal import find_peaks

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


def cell_classification_vr(mouse, day, percentile_threshold=99):
    session = 'VR'
    vr_folder = f'/Users/harryclark/Downloads/COHORT12_nwb/M{mouse}/D{day:02}/{session}/'
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

 
def cell_classification_of1(mouse, day, percentile_threshold=99, 
                            disqualifying_brain_regions=None):
    _,_,_,_,_,clusters_VR = compute_vr_tcs(mouse, day)

    if disqualifying_brain_regions is None:
        disqualifying_brain_regions = disqualifying_brain_areas_for_grid_cells
     
    print(mouse, day)
    session = 'OF1'
    of1_folder = f'/Users/harryclark/Downloads/COHORT12_nwb/M{mouse}/D{day:02}/{session}/'
    shifted_grid_path = of1_folder + "tuning_scores/shifted_grid_score.parquet"
    spatial_path = of1_folder + "tuning_scores/shifted_spatial_information.parquet"
    speed_path = of1_folder + "tuning_scores/shifted_speed_correlation.parquet"

    shifted_grid_scores_of1 = pd.read_parquet(shifted_grid_path)
    spatial_information_score_of1 = pd.read_parquet(spatial_path)
    shifted_speed_score_of1 = pd.read_parquet(speed_path)

    shifted_grid_scores_of1 = shifted_grid_scores_of1.query('travel >= 0')
    spatial_information_score_of1 = spatial_information_score_of1.query('travel >= 0')
    shifted_speed_score_of1 = shifted_speed_score_of1.query('travel == 0')
    cluster_ids_values = shifted_grid_scores_of1.query('travel == 0').cluster_id

    non_grid_cells = pd.DataFrame(columns=shifted_grid_scores_of1.columns)
    grid_cells = pd.DataFrame(columns=shifted_grid_scores_of1.columns)
    non_spatial_cells = pd.DataFrame(columns=shifted_grid_scores_of1.columns)
    speed_cells = pd.DataFrame(columns=shifted_grid_scores_of1.columns)
    cells = pd.DataFrame(columns=shifted_grid_scores_of1.columns)

    print(f'there are {len(clusters_VR)} cells to begin with')
    print(f'I wont use these brain regions')
    print(disqualifying_brain_regions)

    for index in cluster_ids_values:
        brain_region = clusters_VR.brain_region[index]
        SC_x = clusters_VR.coord_SCs_x[index]
        SC_y = clusters_VR.coord_SCs_y[index]
        SC_z = clusters_VR.coord_SCs_z[index]
        probe_x = clusters_VR.coord_probe_x[index]
        probe_y = clusters_VR.coord_probe_y[index]

        if brain_region not in disqualifying_brain_regions:
            cluster_spatial_information_of1 = spatial_information_score_of1[spatial_information_score_of1.cluster_id==index]
            cluster_shifted_grid_scores_of1 = shifted_grid_scores_of1[shifted_grid_scores_of1.cluster_id==index]
            cluster_speed_correlation_of1 = shifted_speed_score_of1[shifted_speed_score_of1.cluster_id==index]

            percentile99_grid_score_of1 = np.nanpercentile(cluster_shifted_grid_scores_of1.null_grid_score.iloc[0], percentile_threshold)
            percentile99_spatial_information_of1 = np.nanpercentile(cluster_spatial_information_of1.null_spatial_information.iloc[0], percentile_threshold)

            percentile99_speed_information_of1_pos = np.nanpercentile(cluster_speed_correlation_of1.null_speed_correlation.iloc[0], percentile_threshold)
            percentile99_speed_information_of1_neg = np.nanpercentile(cluster_speed_correlation_of1.null_speed_correlation.iloc[0], 100-percentile_threshold)

            max_grid_score_of1 = cluster_shifted_grid_scores_of1.grid_score.values[np.nanargmax(cluster_shifted_grid_scores_of1.grid_score)]
            spatial_info = cluster_spatial_information_of1.spatial_information.values[np.nanargmax(cluster_shifted_grid_scores_of1.grid_score)]
            spatial_info_no_lag = cluster_spatial_information_of1.spatial_information.iloc[0]

            speed_correlation = cluster_speed_correlation_of1.speed_correlation.iloc[0]

            cell = shifted_grid_scores_of1[shifted_grid_scores_of1.grid_score==max_grid_score_of1]
            cell['brain_region'] = brain_region
            cell['SC_x'] = SC_x
            cell['SC_y'] = SC_y
            cell['SC_z'] = SC_z
            cell['probe_x'] = probe_x
            cell['probe_y'] = probe_y

            if (max_grid_score_of1 > percentile99_grid_score_of1) and (spatial_info > percentile99_spatial_information_of1) and (max_grid_score_of1>0.4):
                grid_cells = pd.concat([grid_cells, cell], ignore_index=True)
            elif (spatial_info_no_lag > percentile99_spatial_information_of1):
                non_grid_cells = pd.concat([non_grid_cells, cell], ignore_index=True)
            elif (speed_correlation > percentile99_speed_information_of1_pos) or (speed_correlation < percentile99_speed_information_of1_neg):
                speed_cells = pd.concat([speed_cells, cell], ignore_index=True)
            else:
                non_spatial_cells = pd.concat([non_spatial_cells, cell], ignore_index=True)
            cells = pd.concat([cells, cell], ignore_index=True)
        else:
            print(f'Cell from {brain_region} being removed')
    all_cells = cells.copy() 
    non_grid_and_non_spatial_cells = pd.concat([non_grid_cells, non_spatial_cells], ignore_index=True)

    print(f'there are {len(non_grid_and_non_spatial_cells)} non_grid and non_spatial_cells')
    print(f'there are {len(grid_cells)} grid_cells')
    print(f'there are {len(non_grid_cells)} non grid spatial cells')
    print(f'there are {len(non_spatial_cells)} non spatial cells')
    print(f'there are {len(speed_cells)} speed cells')
    print(f'there are {len(all_cells)} cells')

    return grid_cells, non_grid_cells, non_spatial_cells, speed_cells, non_grid_and_non_spatial_cells, all_cells



def HDBSCAN_grid_modules(gcs, all, mouse, day, min_cluster_size=5, cluster_selection_epsilon=3, figpath='', 
                         curate_with_vr=True, curate_with_brain_region=True):

    if len(gcs) == 0:
        return [], []

    samples = np.stack([np.array(gcs['field_spacing']),
                        np.cos((np.array(gcs['orientation'])/60) * 2 * np.pi),
                        np.sin((np.array(gcs['orientation'])/60) * 2 * np.pi)]).T

    samples2d = np.stack([np.array(gcs['field_spacing']),
                        np.array(gcs['orientation'])]).T

    # Standardize the data
    scaler = StandardScaler()
    samples_scaled = scaler.fit_transform(samples)
    samples_scaled[:, 1] /= np.sqrt(2)
    samples_scaled[:, 2] /= np.sqrt(2)

    samples_scaled = samples

    # Apply HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, 
                                cluster_selection_epsilon=cluster_selection_epsilon)
    module_labels = clusterer.fit_predict(samples_scaled)

    # Plot the results
    plt.figure(figsize=(3, 3))
    label_colors = {label: cm.get_cmap('viridis', len(np.unique(module_labels)))(i) for i, label in enumerate(np.unique(module_labels))}
    for mi in np.unique(module_labels):
        mask = module_labels == mi
        print(f'for mi{mi}, there are {np.sum(mask)} points')
        plt.scatter(samples2d[:, 0][mask], samples2d[:, 1][mask], c=label_colors[mi], s=20, cmap='viridis', label='Clustered Points')
    # Highlight unassigned points (label -1)
    unassigned = samples2d[module_labels == -1]
    plt.scatter(unassigned[:, 0], unassigned[:, 1], s=21, color='red', label='Unassigned Points')
    plt.scatter(all['field_spacing'], all['orientation'], s=20, color='tab:grey', alpha=0.5,zorder=-1)

    #plt.legend()
    plt.xlabel('Grid Spacing (cm)')
    plt.ylabel('Grid Orientation ($^\circ$)')
    plt.ylim(0,60)
    plt.title(f'HDBSCAN M{mouse}D{day}')
    plt.tight_layout()
    plt.savefig(f'{figpath}/M{mouse}D{day}_HDBSCAN.pdf')
    plt.show()  

    if np.unique(module_labels).size == 1 and np.unique(module_labels)[0] == -1:
        module_labels[:] = 0  # Assign all points to a single cluster if no clusters were found
        label_colors[0] = label_colors[-1]

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


    _,_,autocorrs,_,_,clusters_VR = compute_vr_tcs(mouse, day)
 

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
        # before performing the median peak check, plot the histogram of peaks
        for mi, module_ids in zip(grid_module_ids, grid_module_cluster_ids):
            matrix = np.array(list(autocorrs.values()))
            matrix_cluster_ids = np.array(list(autocorrs.keys()))
            cluster_id_of_interest = module_ids
            matrix = matrix[np.isin(matrix_cluster_ids, cluster_id_of_interest)]
            matrix_cluster_ids = matrix_cluster_ids[np.isin(matrix_cluster_ids, cluster_id_of_interest)]
            peaks = []
            for array in matrix:
                if len(find_peaks(array)[0])>0:
                    peak = find_peaks(array)[0][0]
                else:
                    peak = np.nan
                    
                '''
                plt.plot(array, color=label_colors[mi], alpha=0.5)
                plt.axvline(peak, color=label_colors[mi], linestyle='--', alpha=0.5)
                plt.show()'''

                peaks.append(peak)
            peaks = np.array(peaks)*bs
            median_peak = np.nanmedian(peaks)
            '''
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(2,2), squeeze=False)
            if median_peak < 200:
                max_r = 200
            else:
                max_r = 400
            ax[0,0].hist(peaks, bins=25, range=(0, max_r), color=label_colors[mi])
            ax[0,0].axvline(median_peak-tolerance, color='grey', linestyle='--')
            ax[0,0].axvline(median_peak+tolerance, color='grey', linestyle='--')
            plt.savefig(f'{figpath}/GC_peaks_{mi}_{mouse}D{day}.pdf')
            plt.show()'''

            # now check if the peaks are within 20cm of the median peak
            # also check if the rate is really low and should be considered
            for peak, cluster_id in zip(peaks, matrix_cluster_ids):
                if not np.abs(peak-median_peak)<(tolerance): # 30cm tolerance
                    module_ids.remove(cluster_id)
                elif nap.TsGroup([clusters_VR[cluster_id]]).rates[0] < 1:
                    module_ids.remove(cluster_id)

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
                if len(find_peaks(array)[0])>0:
                    peak = find_peaks(array)[0][0]
                else:
                    peak = np.nan
                peaks.append(peak)
            peaks = np.array(peaks)*bs
            median_peak = np.nanmedian(peaks)

            '''
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(2,2), squeeze=False)
            if median_peak < 200:
                max_r = 200
            else:
                max_r = 400
            ax[0,0].hist(peaks, bins=25, range=(0, max_r), color=label_colors[mi])
            ax[0,0].axvline(median_peak-tolerance, color='grey', linestyle='--')
            ax[0,0].axvline(median_peak+tolerance, color='grey', linestyle='--')
            plt.savefig(f'{figpath}/GC_peaks_{mi}_{mouse}D{day}_post_curated.pdf')
            plt.show()
            '''

    return  grid_module_ids, grid_module_cluster_ids



def plot_grid_modules_rate_maps(gcs, grid_module_ids, grid_module_cluster_ids, mouse, day, figpath):
    print(mouse, day)
    session = 'OF1'
    of1_folder = f'/Users/harryclark/Downloads/COHORT12_nwb/M{mouse}/D{day:02}/{session}/'
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


def compute_vr_tcs(mouse, day, apply_zscore=True, apply_guassian_filter=True):
    vr_folder = f'/Users/harryclark/Downloads/COHORT12_nwb/M{mouse}/D{day:02}/VR/'
    spikes_path = vr_folder + f"sub-{mouse}_day-{day:02}_ses-VR_srt-kilosort4_clusters.npz"
    beh_path = vr_folder + f"sub-{mouse}_day-{day:02}_ses-VR_beh.nwb"
    beh = nap.load_file(beh_path)
    clusters = nap.load_file(spikes_path)
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
    last_ephys_time_bin = clusters[clusters.index[0]].count(bin_size=time_bs, time_units = 'ms').index[-1]
    print(f'last_ephys_bin {last_ephys_bin}')
    print(f'last_ephys_time_bin {last_ephys_time_bin}')

    # time binned variables for later
    ep = nap.IntervalSet(start=0, end=last_ephys_time_bin, time_units = 's')
    speed_in_time = beh['S'].bin_average(bin_size=time_bs, time_units = 'ms', ep=ep)
    dt_in_time = beh['travel'].bin_average(bin_size=time_bs, time_units = 'ms', ep=ep)-((tns[0]-1)*tl)
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
        
        tc_time = clusters[cell].count(bin_size=time_bs, time_units = 'ms', ep=ep)
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

def plot_stops_mouse_day(mouse, day, figpath): 
    _, _, _ , last_ephys_bin, beh,_ = compute_vr_tcs(mouse, day)

    plot_stops(beh, tl=tl, sort=False, return_fig=False, last_ephys_bin=last_ephys_bin,
           savepath=f'{figpath}/M{mouse}D{day}_stops')
    plot_stops(beh, tl=200, sort=True, return_fig=False, last_ephys_bin=last_ephys_bin,
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


def plot_spectrogram(mouse, day, cluster_ids, label, figpath):
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