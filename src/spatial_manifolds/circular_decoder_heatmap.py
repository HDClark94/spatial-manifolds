
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
from scipy.ndimage import gaussian_filter, rotate
from scipy.signal import find_peaks, peak_prominences, spectrogram
from scipy.stats import zscore
import scipy 
from spatial_manifolds.data.binning import get_bin_config
from spatial_manifolds.data.loading import load_session
from spatial_manifolds.util import gaussian_filter_nan
from spatial_manifolds.data.curation import curate_clusters

from spatial_manifolds.circular_decoder import circular_decoder, cross_validate_decoder
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')



mouse = 25
day =  25
session = 'OF1'


of1_folder = f'/Users/harryclark/Downloads/COHORT12/M{mouse}/D{day:02}/{session}/'
grid_path = of1_folder + "tuning_scores/grid_score.parquet"
spikes_path = of1_folder + f"sub-{mouse}_day-{day:02}_ses-{session}_srt-kilosort4_clusters.npz"
beh_path = of1_folder + f"sub-{mouse}_day-{day:02}_ses-{session}_beh.nwb"
active_projects_path = Path("/Volumes/cmvm/sbms/groups/CDBS_SIDB_storage/NolanLab/ActiveProjects/")
anatomy_path = active_projects_path / "Chris/Cohort12/derivatives/labels/anatomy/cluster_annotations.csv"
cluster_locations = pd.read_csv(anatomy_path)
beh = nap.load_file(beh_path)
clusters = nap.load_file(spikes_path)
clusters = curate_clusters(clusters)
grid_scores = pd.read_parquet(grid_path)
grid_cells = grid_scores.query('sig == True')
mouseday_cluster_locations = cluster_locations.query(f'mouse == {mouse} & day == {day}')

tcs = {}
position = np.stack([beh['P_x'], beh['P_y']], axis=1)
for cell in grid_cells['cluster_id'].values:
    tc = nap.compute_2d_tuning_curves(nap.TsGroup([clusters[cell]]), position, nb_bins=(40,40))[0]
    tc = gaussian_filter_nan(tc[0], sigma=(2.5,2.5))
    tcs[cell] = tc
 
ncols = 10
nrows = int(np.ceil(len(tcs)/ncols))
fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 4*(np.ceil(len(tcs)/6)/4)), squeeze=False)
counter = 0
for j in range(nrows):
    for i in range(ncols):
        if counter<len(tcs):
            index = grid_cells['cluster_id'].values[counter]
            ax[j, i].imshow(tcs[index], cmap='jet')
        ax[j, i].set_xticks([])
        ax[j, i].set_yticks([])
        ax[j, i].xaxis.set_tick_params(labelbottom=False)
        ax[j, i].yaxis.set_tick_params(labelleft=False)
        counter+=1
plt.tight_layout()
plt.close()





def plot_firing_rate_map(ax, tc, bs, tl):
    p = np.nanpercentile(tc, 99)
    tc = np.clip(tc, max=p)
    bpt = tl/bs
    n_trials = int(len(tc)/(bpt))
    trial_rate_map = []
    for i in range(n_trials):
        trial_rate_map.append(tc[int(i*bpt): int((i+1)*bpt)])
    trial_rate_map = np.array(trial_rate_map)    
    ax.imshow(trial_rate_map, cmap='binary')
    ax.invert_yaxis()




vr_folder = f'/Users/harryclark/Downloads/COHORT12/M{mouse}/D{day:02}/VR/'
spikes_path = vr_folder + f"sub-{mouse}_day-{day:02}_ses-VR_srt-kilosort4_clusters.npz"
beh_path = vr_folder + f"sub-{mouse}_day-{day:02}_ses-VR_beh.nwb"
beh = nap.load_file(beh_path)
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

cis = grid_cells['cluster_id'].values
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
                                             prominence=0.1)
    if len(ac_peaks_1D)>0:
        first_peak_loc = lags[ac_peaks_1D[0]+200]
        peaks[cell] = first_peak_loc*bs
    else:
        peaks[cell] = np.nan
        
ncols = 10
nrows = int(np.ceil(len(tcs)/ncols))
fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 4*(np.ceil(len(tcs)/6)/4)), squeeze=False)
counter = 0
for j in range(nrows):
    for i in range(ncols):
        if counter<len(tcs):
            index = cis[counter]
            plot_firing_rate_map(ax[j, i], 
                                 zscore(tcs[index]),
                                 bs=bs,
                                tl=tl)
        ax[j, i].set_xticks([])
        ax[j, i].set_yticks([])
        ax[j, i].xaxis.set_tick_params(labelbottom=False)
        ax[j, i].yaxis.set_tick_params(labelleft=False)
        counter+=1
plt.tight_layout()
plt.close()

ncols = 10
nrows = int(np.ceil(len(tcs)/ncols))
fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 2*(np.ceil(len(tcs)/6)/4)), squeeze=False)
counter = 0
for j in range(nrows):
    for i in range(ncols):
        if counter<len(tcs):
            index = cis[counter]       
            ax[j, i].plot(lags, autocorrs[index])
            ax[j, i].axvline(100, color='grey')
            ax[j, i].axvline(-100, color='grey')
            ac_peaks_1D, _ = scipy.signal.find_peaks(autocorrs[index][200:],
                                                     width=10,
                                                     height=0.05,
                                                     prominence=0.1)
            if len(ac_peaks_1D)>0:
                first_peak_loc = lags[ac_peaks_1D[0]+200]
                ax[j, i].axvline(first_peak_loc, color='black')
                ax[j, i].text(1.0,1.1,
                f'l={np.round(first_peak_loc*bs, decimals=1)}',
                fontsize=5,
                ha='right',
                transform=ax[j,i].transAxes,
            )

        ax[j, i].set_xticks([])
        ax[j, i].set_yticks([])
        ax[j, i].xaxis.set_tick_params(labelbottom=False)
        ax[j, i].yaxis.set_tick_params(labelleft=False)
        counter+=1
plt.tight_layout()
plt.close()

plt.hist(np.array(list(map(lambda x: x, peaks.values()))), bins=50, range=(0,250))
plt.close()





lower_bound = 50
upper_bound = 75
subset_tcs = {key: value for key, value in tcs.items() if lower_bound <= peaks.get(key, float('-inf')) <= upper_bound}
print(f'I will use {len(subset_tcs)} in the decoder')

x_true_dt = dt_bins
true_position = x_true_dt%tl
trial_numbers = (x_true_dt//tl)+beh['trials']['trial_number'][0]
tns_to_decode_with = np.array(beh['trials']['trial_number'])


for train_set in [['b', 'nb'], ['b'], ['nb']]:
    tns_to_decode = np.array(beh['trials']['trial_number']) # decode all trials to visualise
    tns_to_train = np.array(beh['trials']['trial_number'][np.isin(beh['trials']['trial_type'], np.array(train_set))]) 

    predictions, errors = cross_validate_decoder(tcs, true_position, trial_numbers, tns_to_decode, tns_to_train,tl,bs,train=0.9, n=50)


