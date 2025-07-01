
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pynapple as nap
from spatial_manifolds.toroidal import *
from spatial_manifolds.behaviour_plots import *
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import distance
from spatial_manifolds.circular_decoder import circular_decoder, cross_validate_decoder, cross_validate_decoder_time, circular_nanmean

from spatial_manifolds.data.curation import curate_clusters
from scipy.stats import zscore
from spatial_manifolds.util import gaussian_filter_nan
from spatial_manifolds.predictive_grid import compute_travel_projected, wrap_list
from spatial_manifolds.behaviour_plots import *
from spatial_manifolds.behaviour_plots import trial_cat_priority
from spatial_manifolds.detect_grids import *
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from IPython.display import HTML

import warnings
warnings.filterwarnings('ignore')

fig_path = '/Users/harryclark/Documents/figs/FIGURE1/'

sessions_to_use = ['M21D18', 'M21D23', 'M21D24', 'M21D25', 'M21D26',
                   'M22D34', 'M22D35', 'M22D36', 'M25D22', 'M25D23',
                   'M25D24', 'M25D25', 'M28D16', 'M28D17', 'M28D18',
                   'M28D20', 'M28D22', 'M28D23', 'M28D25', 'M29D16',
                   'M29D17', 'M29D18', 'M29D19', 'M29D20', 'M29D21', 
                   'M29D22', 'M29D23', 'M29D25', 'M20D14', 'M20D22',
                   'M20D23', 'M20D25', 'M20D26', 'M21D17', 'M21D21',
                   'M21D22', 'M22D33', 'M22D37', 'M22D38', 'M22D39',
                   'M22D40', 'M22D41', 'M25D17', 'M25D19', 'M25D20',
                   'M26D11', 'M26D12', 'M26D13', 'M26D14', 'M26D15',
                   'M26D16', 'M26D17', 'M26D19', 'M27D16', 'M27D17',
                   'M27D18', 'M27D19', 'M27D20', 'M27D21', 'M27D24',
                   'M27D26', 'M28D19', 'M28D21', 'M20D15', 'M20D16',
                   'M20D17', 'M20D18', 'M20D19', 'M20D20', 'M20D21', 
                   'M20D24', 'M20D27', 'M20D28', 'M21D15', 'M21D16', 
                   'M21D19', 'M21D20', 'M21D27', 'M21D28', 'M25D16',
                   'M26D18', 'M27D23']

for session in sessions_to_use:
    mouse = session.split('M')[-1].split('D')[0]
    day = session.split('M')[-1].split('D')[-1]

    gcs, ngs, ns, sc, ngs_ns, all = cell_classification_of1(mouse, day, percentile_threshold=95) # subset
    rc, rsc = cell_classification_vr(mouse, day)
    tcs, tcs_time, autocorrs, last_ephys_bin, beh, clusters = compute_vr_tcs(mouse, day)

    df = pd.DataFrame()
    for train_set, train_set_label in zip([np.array(['hit']), np.array(['try', 'run'])], ['eng', 'diseng']):
        for test_set, test_set_label in zip([np.array(['hit']), np.array(['try', 'run'])], ['eng', 'diseng']):
            for cluster_ids, cell_type_label in zip([gcs.cluster_id.values, ngs.cluster_id.values], ['GC', 'NGS']):

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
                trial_types = np.array(beh['trials'][:int(last_ephys_bin/(tl/bs))]['type'])

                tns_to_decode = np.array(beh['trials']['number'][(np.isin(beh['trials']['type'], np.array(['b','nb']))) &
                                                                (np.isin(beh['trials']['performance'], test_set))])
                tns_to_train = np.array(beh['trials']['number'][(np.isin(beh['trials']['type'], np.array(['b','nb']))) &
                                                                (np.isin(beh['trials']['performance'], train_set))]) 
                
                tns_to_decode = tns_to_decode[tns_to_decode<=np.nanmax(trial_numbers)] # handles last ephys trials
                tns_to_train = tns_to_train[tns_to_train<=np.nanmax(trial_numbers)] # handles last ephys trials

                tcs_ = {cluster_id: tcs[cluster_id] for cluster_id in cluster_ids if cluster_id in tcs}

                predictions, errors = cross_validate_decoder(tcs_, true_position, trial_numbers, tns_to_decode, 
                                                            tns_to_train, tl, bs, train=0.9, n=10, verbose=False)
                
                avg_predictions = circular_nanmean(predictions, tl, axis=2)
                avg_errors = np.nanmean(errors, tl, axis=2)
                
                tmp = pd.DataFrame()
                tmp['train_set'] = [train_set_label]
                tmp['test_set'] = [test_set_label]
                tmp['cell_type'] = [cell_type_label]
                tmp['error'] = [np.nanmean(avg_errors)]
                tmp['mouse'] = [mouse]
                tmp['day'] = [day]
                tmp['mouse_day'] = [session]

                if len(cluster_ids)>10:
                    df = pd.concat([df, tmp], ignore_index=True)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

for cell_type_label in ['GC', 'NGS']:
    ct_df = df[df['cell_type'] == cell_type_label]

    # Calculate average error for each train_set and test_set combination
    avg_error = ct_df.groupby(['train_set', 'test_set'])['error'].mean().reset_index()
    # Create bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='train_set', y='error', hue='test_set', data=avg_error, errorbar=None)

    # Overlay line plot for individual trajectories
    for mouse_day in df['mouse_day'].unique():
        subset = df[df['mouse_day'] == mouse_day]
        x_labels = subset['train_set'] + '_' + subset['test_set']
        plt.plot(x_labels, subset['error'], marker='o', linestyle='-', label=mouse_day)

    # Customize plot
    plt.xlabel('Train Set and Test Set')
    plt.ylabel('Error')
    plt.title('Comparison of Errors Across Train Sets and Test Sets')
    plt.legend(title='Mouse Day')
    plt.tight_layout()
    plt.show()
