import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from Elrond.Helpers.array_utility import pandas_collumn_to_2d_numpy_array
from scipy.signal import spectrogram, welch
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter1d
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import pynapple as nap
import os
from probeinterface.plotting import plot_probegroup
import spikeinterface.full as si
from scipy.ndimage import gaussian_filter, rotate
from scipy.signal import find_peaks, peak_prominences, spectrogram
from scipy.stats import zscore
import scipy 
from sklearn.cluster import KMeans
import traceback
import sys

from spatial_manifolds.data.binning import get_bin_config
from spatial_manifolds.data.loading import load_session
from spatial_manifolds.util import gaussian_filter_nan
from spatial_manifolds.toroidal import *
from spatial_manifolds.data.curation import curate_clusters
from spatial_manifolds.behaviour_plots import *

import random
import warnings
warnings.filterwarnings('ignore')

savepath = '/Users/harryclark/Documents/figs/grids'
mouses = [20,20,20,20,20,20,20,20,20,20,20,20,20,21,21,21,21,21,21,21,21,21,21,21,21,22,22,22,22,22,22,22,22,22,25,25,25,25,25,25,25,25,25,25,26,26,26,26,26,26,26,26,26,27,27,27,27,27,27,27,27,27,27,28,28,28,28,28,28,28,28,28,29,29,29,29,29,29,29,29,29]
days =   [14,15,16,17,18,19,20,21,22,23,24,25,26,15,16,17,18,19,20,21,22,23,24,25,26,33,34,35,36,37,38,39,40,41,16,17,18,19,20,21,22,23,24,25,11,12,13,14,15,16,17,18,19,16,17,18,19,20,21,22,23,24,26,16,17,18,19,20,21,22,23,25,16,17,18,19,20,21,22,23,25]


for mouse, day in zip(mouses,days):
    # plot bonafide of1 grid cells 
    session = 'OF1'
    of1_folder = f'/Users/harryclark/Downloads/COHORT12/M{mouse}/D{day:02}/{session}/'
    grid_path = of1_folder + "tuning_scores/grid_score.parquet"
    spikes_path = of1_folder + f"sub-{mouse}_day-{day:02}_ses-{session}_srt-kilosort4_clusters.npz"
    beh_path = of1_folder + f"sub-{mouse}_day-{day:02}_ses-{session}_beh.nwb"
    beh = nap.load_file(beh_path)
    clusters = nap.load_file(spikes_path)
    clusters = curate_clusters(clusters)
    grid_scores = pd.read_parquet(grid_path)
    grid_cells = grid_scores.query('sig == True')
    grid_cell_clusters = clusters[grid_cells.cluster_id]

    tcs = {}
    position = np.stack([beh['P_x'], beh['P_y']], axis=1)
    for cell in grid_cells['cluster_id'].values:
        tc = nap.compute_2d_tuning_curves(nap.TsGroup([clusters[cell]]), position, nb_bins=(40,40))[0]
        tc = gaussian_filter_nan(tc[0], sigma=(2.5,2.5))
        tcs[cell] = tc
    plot_grid_cell_tcs(tcs, cluster_ids=grid_cells['cluster_id'].values, title=f'M{mouse}D{day}_of1_grid_cell_tcs', savepath=savepath)

    # load vr session, compute tcs and autocorrelogram peaks 
    vr_folder = f'/Users/harryclark/Downloads/COHORT12/M{mouse}/D{day:02}/VR/'
    spikes_path = vr_folder + f"sub-{mouse}_day-{day:02}_ses-VR_srt-kilosort4_clusters.npz"
    beh_path = vr_folder + f"sub-{mouse}_day-{day:02}_ses-VR_beh.nwb"
    beh = nap.load_file(beh_path, lazy_loading=False)
    clusters = nap.load_file(spikes_path)
    clusters = curate_clusters(clusters)

    tl=200
    bs=2
    tns = beh['trial_number']
    dt = beh['travel']-((tns[0]-1)*tl)
    n_bins = int(int(((np.ceil(np.nanmax(dt))//tl)+1)*tl)/bs)
    max_bound = int(((np.ceil(np.nanmax(dt))//tl)+1)*tl)
    min_bound = 0
    dt_bins =np.arange(0,max_bound,bs)

    tcs = {}
    autocorrs = {}
    peaks = {}
    cis = clusters.index
    for cell in cis:
        tc = nap.compute_1d_tuning_curves(nap.TsGroup([clusters[cell]]), 
                                        dt, 
                                        nb_bins=n_bins, 
                                        minmax=[min_bound, max_bound],
                                        ep=beh["moving"])[0]
        mask = np.isnan(tc)
        tc = gaussian_filter(np.nan_to_num(tc).astype(np.float64), sigma=2.5)
        tc = zscore(tc)
        tcs[cell] = tc

        lags = np.arange(-200, 200, 1) # were looking at 10 timesteps back and 10 forward
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

        ac_peaks_1D, _ = scipy.signal.find_peaks(autocorr[200:],
                                                width=10,
                                                height=0.05,
                                                prominence=0.05)
        if len(ac_peaks_1D)>0:
            first_peak_loc = lags[ac_peaks_1D[0]+200]
            peaks[cell] = first_peak_loc*bs
        else:
            peaks[cell] = 0

    plot_stops_and_sensitivity(beh, title=f'M{mouse}D{day}_stops', savepath=savepath)
    plot_stops(beh, title=f'M{mouse}D{day}_stops', savepath=savepath)
    plot_all_avg_fr_maps(tcs, cluster_ids=cis, title=f'M{mouse}D{day}_all_avg_frs', savepath=savepath, bs=bs, tl=tl)
    plot_all_fr_maps(tcs, cluster_ids=cis, title=f'M{mouse}D{day}_all_frs', savepath=savepath, bs=bs, tl=tl)
    plot_all_autocorrs(autocorrs, lags, cluster_ids=cis, title=f'M{mouse}D{day}_all_autocorrs', savepath=savepath, bs=bs)
    plt.close()

    # compute k clusters from autocorrelogram
    peak_values = list(peaks.values())
    data = np.array(peak_values).reshape(-1, 1)
    k = 6  # Number of clusters
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    labels = kmeans.labels_
    neuron_clusters = {neuron_id: label for neuron_id, label in zip(peaks.keys(), labels)}
    fig, ax = plt.subplots(7,1, layout='constrained', figsize=(2, 5), height_ratios=[0.2,0.2,0.2,0.2,0.2,0.2,1])
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
              'tab:brown','tab:pink',   'tab:gray',  'tab:olive','tab:cyan']
    for ki in range(k):
        cluster_grid_scores = [score for score, label in zip(np.array(grid_scores['grid_score']), labels) if label == ki]
        cluster_values =      [value for value, label in zip(peak_values, labels) if label == ki]
        ax[ki].hist(cluster_grid_scores, bins=40, range=(-1,1), alpha=0.5, label=f'k={ki}', color=colors[ki])
        ax[ki].axvline(x=0, color='grey')
        ax[6].hist(cluster_values, bins=50, range=(0,250), alpha=0.5, label=f'k={ki}', color=colors[ki])
    ax[5].set_xlabel('grid score')
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[2].set_xticks([])
    ax[3].set_xticks([])
    ax[4].set_xticks([])
    ax[0].set_ylabel(f'k={0}')
    ax[1].set_ylabel(f'k={1}')
    ax[2].set_ylabel(f'k={2}')
    ax[3].set_ylabel(f'k={3}')
    ax[4].set_ylabel(f'k={4}')
    ax[5].set_ylabel(f'k={5}')
    ax[6].set_xlabel('autocorr peak (cm)')
    fig.savefig(savepath+f'/M{mouse}D{day}_K_cluster_autocorreloograms.pdf')
    plt.close()

    '''
    # compute k clusters from autocorrelogram
    data = np.vstack(list(autocorrs.values()))[:, 200:]
    k = 6  # Number of clusters
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    labels = kmeans.labels_
    neuron_clusters = {neuron_id: label for neuron_id, label in zip(peaks.keys(), labels)}
    fig, ax = plt.subplots(7,1, layout='constrained', figsize=(2, 5), height_ratios=[0.2,0.2,0.2,0.2,0.2,0.2,1])
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
              'tab:brown','tab:pink',   'tab:gray',  'tab:olive','tab:cyan']
    k_ordered = []
    for ki in range(k):
        cluster_grid_scores = [score for score, label in zip(np.array(grid_scores['grid_score']), labels) if label == ki]
        ax[ki].hist(cluster_grid_scores, bins=40, range=(-1,1), alpha=0.5, label=f'k={ki}', color=colors[ki])
        ax[ki].axvline(x=0, color='grey')
        k_ordered.extend(data[labels == ki].tolist())
        ax[6].axhline(y=len(k_ordered), zorder=3, color='black')
    k_ordered = np.array(k_ordered)
    k_ordered = np.clip(k_ordered, max=0.4)
    lags_, cells_ = np.meshgrid(lags[200:], np.arange(len(labels)))
    ax[6].pcolormesh(lags_, cells_, k_ordered, shading='auto', cmap='summer')
    #ax[6].imshow(k_ordered, zorder=-2, cmap='summer')
    ax[5].set_xlabel('grid score')
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[2].set_xticks([])
    ax[3].set_xticks([])
    ax[4].set_xticks([])
    ax[0].set_ylabel(f'k={0}')
    ax[1].set_ylabel(f'k={1}')
    ax[2].set_ylabel(f'k={2}')
    ax[3].set_ylabel(f'k={3}')
    ax[4].set_ylabel(f'k={4}')
    ax[5].set_ylabel(f'k={5}')
    ax[6].set_xlabel('autocorr peak (cm)')
    fig.savefig(savepath+f'/M{mouse}D{day}_K_cluster_autocorreloograms2.pdf')
    plt.close()
    '''

    # Extract cluster IDs for each k
    k_cluster_ids = {cluster_id: [] for cluster_id in range(k)}
    for neuron_id, label in neuron_clusters.items():
        k_cluster_ids[label].append(neuron_id)
    for ki in range(k):
        try:
            print(f'For K {ki}, n=({len(k_cluster_ids[ki])}), cluster ids include {np.array(k_cluster_ids[ki])}')

            # run toirodal analysis for given k cluster
            k_tcs = {cluster_id: tcs[cluster_id] for cluster_id in k_cluster_ids[ki] if cluster_id in tcs}
            N = len(k_tcs)
            zmaps = np.array(list(k_tcs.values()))
            results = spectral_analysis(k_tcs, tl, bs=bs)
            f_modules =              results[0]
            phi_modules =            results[1]
            grid_cell_idxs_modules = results[2]
            spectrograms =           results[3]
            trial_starts =           results[6]
            L = tl

            plt.figure(figsize=(3,4))
            nongrid_idxs = np.setdiff1d(np.arange(N), np.concatenate(grid_cell_idxs_modules))
            fmax = 8/L
            count = 0
            Ps = []
            for j in range(len(grid_cell_idxs_modules)):
                grid_cell_idxs = grid_cell_idxs_modules[j]
                for gi in grid_cell_idxs:
                    mp = gaussian_filter1d(zmaps[gi].ravel(), 3)
                    f, Pxx = welch(mp,nperseg=4000,noverlap=3000)
                    # Ps.append(Pxx[f<fmax])
                    Ps.append(Pxx[f<fmax]/(Pxx[f<fmax]).sum())
                count += len(grid_cell_idxs)
                plt.axhline(count, c='grey',linestyle='dashed',linewidth=0.4)
            for ngi in nongrid_idxs:
                mp = gaussian_filter1d(zmaps[ngi].ravel(), 3)
                f, Pxx = welch(mp,nperseg=4000,noverlap=3000)
                Ps.append(Pxx[f<fmax]/(Pxx[f<fmax]).sum())
            Ps = np.stack(Ps)

            plt.pcolormesh(100*f[f<fmax]/2,np.arange(len(Ps)),np.stack(Ps),vmax=0.04)
            plt.xlabel(f'Frequency (m-1)')
            plt.xlim([0,2])
            plt.ylabel('Neuron')
            plt.tight_layout()
            plt.title('PSDs')
            plt.savefig(savepath+f'/M{mouse}D{day}_K{ki}_PSD.pdf')
            plt.close()

            plt.figure(figsize=(20,3))
            for i, grid_cell_idxs in enumerate(grid_cell_idxs_modules):
                plt.subplot(1,5,i+1)
                Ng = len(grid_cell_idxs)
                S = spectrograms[grid_cell_idxs].mean(0)
                plt.imshow(S,origin='lower',aspect='auto',vmax=0.25,cmap='magma')
                plt.yticks([0, len(S)/2, len(S)], [0, 1, 2])
                plt.ylabel(f'Frequency (m-1)')
                plt.xlabel('Trials')
            plt.savefig(savepath+f'/M{mouse}D{day}_K{ki}_joint_spectrogram.pdf')
            plt.close()

            # Plot trajectories on the neural sheet
            grid_cell_idxs = grid_cell_idxs_modules[0]
            phi = phi_modules[0]
            Ng = len(grid_cell_idxs)

            maps = gaussian_filter1d(zmaps[grid_cell_idxs].reshape(Ng, -1), 2, axis=1)
            # maps = zmaps[grid_cell_idxs].reshape(Ng,-1)
            angles = np.arctan2(np.cos(phi)@maps, np.sin(phi)@maps)
            fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(10, 5), squeeze=False)

            ax[0,0].set_title(r'$\theta_1$')
            ax[0,0].imshow(angles[0].reshape(-1,int(L/bs)),cmap='hsv')
            for ts in trial_starts[1:-1]:
                ax[0,0].axhline(ts,color='k',linestyle='dashed',linewidth=1)
            ax[0,0].set_xlabel('Pos. (cm)')
            ax[0,0].set_ylabel('Trial')

            ax[0,1].imshow(angles[1].reshape(-1,int(L/bs)),cmap='hsv')
            ax[0,1].set_title(r'$\theta_2$')
            for ts in trial_starts[1:-1]:
                ax[0,1].axhline(ts,color='k',linestyle='dashed',linewidth=1)
            ax[0,1].set_ylabel('Trial')
            ax[0,1].set_xlabel('Pos. (cm)')

            ax[0,2].imshow(angles[2].reshape(-1,int(L/bs)),cmap='hsv')
            ax[0,2].set_title(r'$\theta_3$')
            for ts in trial_starts[1:-1]:
                ax[0,2].axhline(ts,color='k',linestyle='dashed',linewidth=1)
            ax[0,2].set_ylabel('Trial')
            ax[0,2].set_xlabel('Pos. (cm)')

            plt.tight_layout(w_pad=0.2)
            plt.savefig(savepath+f'/M{mouse}D{day}_K{ki}_toroidal_coords.pdf')
            plt.close()

            sorted_cats = beh['trials'].groupby(by=['context','type','performance'])
            sorted_trial_numbers = []
            sorted_block_sizes = []
            for tns in sorted_cats.values():
                sorted_trial_numbers.extend(tns)
                sorted_block_sizes.append(len(tns))
            sorted_trial_numbers=np.array(sorted_trial_numbers)

            angles1 = angles[0].reshape(-1,int(L/bs))[sorted_trial_numbers]
            angles2 = angles[1].reshape(-1,int(L/bs))[sorted_trial_numbers]
            angles3 = angles[2].reshape(-1,int(L/bs))[sorted_trial_numbers]

            fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(8, 5), sharey=True, squeeze=False, width_ratios=[1,1,1,0.1])
            ax[0,0].set_title(r'$\theta_1$')
            ax[0,0].imshow(angles1,cmap='hsv')
            ax[0,0].set_xlabel('Pos. (cm)')
            ax[0,0].set_ylabel('Trial')
            ax[0,1].imshow(angles2,cmap='hsv')
            ax[0,1].set_title(r'$\theta_2$')
            ax[0,1].set_xlabel('Pos. (cm)')
            ax[0,2].imshow(angles3,cmap='hsv')
            ax[0,2].set_title(r'$\theta_3$')
            ax[0,2].set_xlabel('Pos. (cm)')
            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
                    'tab:brown','tab:pink',   'tab:gray',  'tab:olive','tab:cyan']
            tn_start=0
            for i, block_size in enumerate(sorted_block_sizes):
                ax[0,3].plot([0,0], [tn_start, tn_start+block_size], linewidth=20, color=colors[i])
                tn_start+= block_size
            ax[0,3].set_ylim(0,tn_start)
            plt.savefig(savepath+f'/M{mouse}D{day}_K{ki}_toroidal_coords_sorted.pdf')
            plt.close()

        except Exception as ex:
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
