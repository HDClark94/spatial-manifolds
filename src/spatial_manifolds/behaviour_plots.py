import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from probeinterface.plotting import plot_probegroup
import spikeinterface.full as si
from scipy import stats
from scipy.interpolate import interp1d
from scipy.stats import zscore
import scipy
import os
import pynapple as nap
from spatial_manifolds.anaylsis_parameters import tl, bs, rm_figsize


def plot_speeds_multicontext(session, title, figpath):
    fig, ax = plt.subplots(
        1, 1, layout='constrained', figsize=(2, 2)
    )
    for tt, tc, ttc in zip(['nb', 'nb'], ['rz1', 'rz2'], ['tab:blue', 'tab:orange']):
        trials = session['trials'][(session['trials']['trial_type'] == tt) &
                                   (session['trials']['trial_context'] == tc) ]
        print(f'n trials for tc {tc} {len(trials)}')
        tc  = nap.compute_1d_tuning_curves_continuous(
            session['S'],
            session['P'],
            nb_bins=50,
            minmax=[0,230],
            ep=trials)[0]
        ax.plot(tc.index, tc.values, color=ttc)
    ax.set_ylabel('speed (cm/s)')    
    ax.set_xlabel('position (cm)')    

    ax.set_xlim(0,230)
    ax.axvspan(
        90,110,
        alpha=0.2,
        zorder=-10,
        edgecolor='none',
        facecolor='tab:blue',
    )
    ax.axvspan(
        120,140,
        alpha=0.2,
        zorder=-10,
        edgecolor='none',
        facecolor='tab:orange',
    )
    plt.savefig(f'{figpath}/{title}_speeds.pdf', dpi=300)
    plt.show()


def remove_all_from_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=True, bottom=False)
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_xlabel('')
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_ylabel('')
    return ax


def plot_NP2_probe(ax, sorting_analyzer_path=None):
    if sorting_analyzer_path is None:
        project_path = "/Volumes/cmvm/sbms/groups/CDBS_SIDB_storage/NolanLab/ActiveProjects/Chris/Cohort12/derivatives/"
        # get any sorting analyzer from this project folder
        example_day_path = [f.path for f in os.scandir(f"{project_path}{'M25'}/") if f.is_dir()][25]
        sorting_analyzer_path = f"{example_day_path}/full/kilosort4/kilosort4_sa"
    print(sorting_analyzer_path)
    sorting_analyzer = si.load_sorting_analyzer(sorting_analyzer_path, load_extensions=False)
    probe_group = sorting_analyzer.get_probegroup()
    plot_probegroup(probe_group, ax=ax, 
                    probe_shape_kwargs={"alpha":0.5,
                                        "color":"lightgrey"}, 
                    contacts_kargs={"alpha":0.0})
    ax.set_xlim(-200, 1000)
    ax.set_ylim(-300, 3000)
    ax = remove_all_from_ax(ax)


def plot_firing_rate_map(ax, tc, bs, tl, p=99, sort_indices=None):
    p = np.nanpercentile(tc, p)
    tc = np.clip(tc, max=p)
    bpt = tl/bs
    n_trials = int(len(tc)/(bpt))
    trial_rate_map = []
    for i in range(n_trials):
        trial_rate_map.append(tc[int(i*bpt): int((i+1)*bpt)])
    trial_rate_map = np.array(trial_rate_map)    
    if sort_indices is not None:
         trial_rate_map = trial_rate_map[sort_indices]
    
    x = np.arange(1, len(trial_rate_map)+1)
    y = np.arange(0, len(trial_rate_map[0])*bs, bs)
    X, Y = np.meshgrid(x, y)
    heatmap = ax.pcolormesh(Y, X, trial_rate_map.T, shading='auto', cmap='binary')
    heatmap.set_rasterized(True)
    ax.set_xlim(0,tl)
    ax.set_ylim(0,len(trial_rate_map))
    ax.invert_yaxis()


def plot_avg_firing_rate_map(ax, tc, bs, tl, mask=None, c='black'):
    p = np.nanpercentile(tc, 99)
    tc = np.clip(tc, max=p)
    bpt = tl/bs
    n_trials = int(len(tc)/(bpt))
    trial_rate_map = []
    for i in range(n_trials):
        trial_rate_map.append(tc[int(i*bpt): int((i+1)*bpt)])
    trial_rate_map = np.array(trial_rate_map)   
    ax.plot(np.arange(bpt), np.nanmean(trial_rate_map,axis=0), color=c)
    ax.fill_between(np.arange(bpt), np.nanmean(trial_rate_map,axis=0)-stats.sem(trial_rate_map, axis=0,nan_policy="omit"),
                                    np.nanmean(trial_rate_map,axis=0)+stats.sem(trial_rate_map, axis=0,nan_policy="omit"),alpha=0.3, color=c)



def plot_grid_cell_tcs(tcs, cluster_ids, title='', savepath=None):
    ncols = 10
    nrows = int(np.ceil(len(tcs)/ncols))
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 4*(np.ceil(len(tcs)/6)/4)), squeeze=False)
    counter = 0
    for j in range(nrows):
        for i in range(ncols):
            if counter<len(tcs):
                index = cluster_ids[counter]
                ax[j, i].imshow(tcs[index], cmap='jet')
            ax[j, i].set_xticks([])
            ax[j, i].set_yticks([])
            ax[j, i].xaxis.set_tick_params(labelbottom=False)
            ax[j, i].yaxis.set_tick_params(labelleft=False)
            counter+=1
    plt.tight_layout()
    if savepath is not None:
        fig.savefig(f'{savepath}/{title}.pdf')
    plt.close()


def plot_all_autocorrs(autocorrs, lags, cluster_ids, title, savepath, bs):
        
    ncols = 10
    nrows = int(np.ceil(len(autocorrs)/ncols))
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 4*(np.ceil(len(autocorrs)/6)/4)), squeeze=False)
    counter = 0
    for j in range(nrows):
        for i in range(ncols):
            if counter<len(autocorrs):
                index = cluster_ids[counter]       
                ax[j, i].plot(lags, autocorrs[index])
                ax[j, i].axvline(100, color='grey')
                ax[j, i].axvline(-100, color='grey')
                ac_peaks_1D, _ = scipy.signal.find_peaks(autocorrs[index][200:],
                                                        width=10,
                                                        height=0.05,
                                                        prominence=0.05)
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
    if savepath is not None:
        fig.savefig(f'{savepath}/{title}.pdf')
    plt.close()


def plot_all_fr_maps(tcs, cluster_ids, title, savepath, bs, tl):
    ncols = 10
    nrows = int(np.ceil(len(tcs)/ncols))
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 10*(np.ceil(len(tcs)/6)/4)), squeeze=False)
    counter = 0
    for j in range(nrows):
        for i in range(ncols):
            if counter<len(tcs):
                index = cluster_ids[counter]
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
    if savepath is not None:
        fig.savefig(f'{savepath}/{title}.pdf')
    plt.close()


def plot_all_avg_fr_maps(tcs, cluster_ids, title, savepath, bs, tl):
    ncols = 10
    nrows = int(np.ceil(len(tcs)/ncols))
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 4*(np.ceil(len(tcs)/6)/4)), squeeze=False)
    counter = 0
    for j in range(nrows):
        for i in range(ncols):
            if counter<len(tcs):
                index = cluster_ids[counter]
                plot_avg_firing_rate_map(ax[j, i], 
                                        zscore(tcs[index]),
                                        bs=bs,
                                        tl=tl)
            ax[j, i].set_xticks([])
            ax[j, i].set_yticks([])
            ax[j, i].xaxis.set_tick_params(labelbottom=False)
            ax[j, i].yaxis.set_tick_params(labelleft=False)
            counter+=1
    plt.tight_layout()
    if savepath is not None:
        fig.savefig(f'{savepath}/{title}.pdf')
    plt.close()
    

def plot_stops(beh, savepath=None, tl=None, sort=False, return_fig=True, stop_mask=None, last_ephys_bin=None):
    trial_numbers = np.array(beh['trial_number'])
    position = np.array(beh['P'])
    trial_types = np.array(beh['trial_type'])
    speed = np.array(beh['S'])
    if stop_mask is None:
        stop_mask = speed<3

    trial_numbers = trial_numbers[:len(stop_mask)]
    trial_types = trial_types[:len(stop_mask)]
    position = position[:len(stop_mask)]
    speed = speed[:len(stop_mask)]

    speed_ymax=0
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(rm_figsize[0], rm_figsize[1]*1.45), sharex=True, height_ratios=[0.3, 1], width_ratios=[1,0.05], sharey='row')

    if last_ephys_bin is not None:
        sorted_cats = beh['trials'][:int(last_ephys_bin/(tl/bs))].groupby(by=['context','type','performance'])
        n_trials = len( beh['trials'][:int(last_ephys_bin/(tl/bs))])
    else:
        sorted_cats = beh['trials'].groupby(by=['context','type','performance'])
        n_trials = len(beh['trials'])

    sorted_cats = sort_dict_by_priority(sorted_cats, trial_cat_priority)

    sorted_trial_indices = []
    sorted_trial_colors = []
    for group, cat_indices in zip(sorted_cats.keys(), sorted_cats.values()):
        c = get_color_for_group(group)
        sorted_trial_colors.extend(np.repeat(c, len(cat_indices)).tolist())
        sorted_trial_indices.extend(cat_indices.tolist())
        group_speed = speed[np.isin(trial_numbers, np.array(beh['trials'][cat_indices]['number']))]
        group_position = position[np.isin(trial_numbers, np.array(beh['trials'][cat_indices]['number']))]
        speed_profile = np.histogram(group_position, weights=group_speed, range=(0,tl), bins=int(tl/5))[0]/np.histogram(group_position, range=(0,tl), bins=int(tl/5))[0]
        bin_edges = np.histogram(group_position, range=(0,tl), bins=int(tl/5))[1]
        bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        speed_ymax=max(speed_ymax, max(speed_profile))
        if len(cat_indices)>5:
            ax[0,0].plot(bin_centres, speed_profile, color=c, linewidth=1)
    sorted_trial_colors = np.array(sorted_trial_colors)
    sorted_trial_indices = np.array(sorted_trial_indices)

    if sort:
        for i, sti in enumerate(sorted_trial_indices):
            tn_mask = trial_numbers==beh['trials'][sti]['number'].iloc[0]
            stops = position[(stop_mask & tn_mask)]
            ax[1,0].scatter(stops, i*np.ones(len(stops)), alpha=0.025, s=5, c=sorted_trial_colors[i], rasterized=True)
        ax[1,1].scatter(np.ones(len(sorted_trial_colors)), 
                      np.arange(0,len(sorted_trial_colors)), 
                      c = sorted_trial_colors,s=1,
                      marker='s', rasterized=True)
    else: 
        for i, ti in enumerate(beh['trials'].index):
            tn_mask = trial_numbers==beh['trials'][ti]['number'].iloc[0]
            group = (beh['trials'][ti]['context'].iloc[0],
                     beh['trials'][ti]['type'].iloc[0],
                     beh['trials'][ti]['performance'].iloc[0])
            stops = position[(stop_mask & tn_mask)]
            ax[1,0].scatter(stops, i*np.ones(len(stops)), alpha=0.025, s=5, c=get_color_for_group(group), rasterized=True)
            ax[1,1].scatter(1, i, c=get_color_for_group(group),marker='s',s=1,rasterized=True)
    ax[1,1].axis('off')
    ax[0,1].axis('off')
    ax[1,0].set_xlabel('Pos (cm)')
    ax[1,0].set_xlim(0,tl)
    ax[1,0].set_ylim(0, n_trials)
    ax[0,0].set_ylim(0, speed_ymax)
    ax[1,0].invert_yaxis()
    ax[1,0].axvspan(
        90,110,
        alpha=0.2,
        zorder=-10,
        edgecolor='none',
        facecolor='grey',
    )

    if return_fig:
         return 
    else:
        if savepath is not None:
            fig.savefig(savepath+'.pdf', bbox_inches='tight', dpi=300)
            plt.close()    


def curate_behavioural_mask(mask):
    found_true = False
    for i in range(len(mask)):
        if mask[i]:
            if found_true:
                mask[i] = False
            else:
                found_true = True
        else:
            found_true = False
    return mask


trial_cat_priority = [
    ('rz1', 'b', 'hit'),
    ('rz1', 'b', 'try'),
    ('rz1', 'b', 'run'),
    ('rz1', 'nb', 'hit'),
    ('rz1', 'nb', 'try'),
    ('rz1', 'nb', 'run'),
    ('rz1', 'b', 'slow'),
    ('rz1', 'nb', 'slow')
]
def sort_dict_by_priority(data, priority):
    sorted_items = sorted(
        data.items(),
        key=lambda item: priority.index(item[0]) if item[0] in priority else float('inf')
    )
    return dict(sorted_items)
 
def get_color_for_group(group):
    if group == ('rz1', 'b', 'hit'):
        color= '#3071AB'
    elif group == ('rz1', 'b', 'try'):
            color= "#6697CF"
    elif group == ('rz1', 'b', 'run'):
            color= '#A7CAEA'
    elif group == ('rz1', 'b', 'slow'):
            color= 'tab:grey'
    elif group == ('rz1', 'nb', 'hit'):
            color= '#DB752B'
    elif group == ('rz1', 'nb', 'try'):
            color= '#E89E57'
    elif group == ('rz1', 'nb', 'run'):
            color= '#F3C288'
    elif group == ('rz1', 'nb', 'slow'):
            color= 'tab:grey'
    elif group == ('rz2', 'b', 'hit'):
        color= '#5B9241'
    elif group == ('rz2', 'b', 'try'):
            color= '#76B153'
    elif group == ('rz2', 'b', 'run'):
            color= '#A8C982'
    elif group == ('rz2', 'b', 'slow'):
            color= 'tab:grey'
    elif group == ('rz2', 'nb', 'hit'):
            color= '#8D6C59'
    elif group == ('rz2', 'nb', 'try'):
            color= '#B99B81'
    elif group == ('rz2', 'nb', 'run'):
            color= '#D9BDA8'
    elif group == ('rz2', 'nb', 'slow'):
          color= 'tab:grey'
    return color