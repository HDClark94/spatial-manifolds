
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
from sklearn.preprocessing import StandardScaler
import umap
from cebra import CEBRA
import cebra.integrations.plotly
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from spatial_manifolds.circular_decoder import circular_decoder, cross_validate_decoder, cross_validate_decoder_time, circular_nanmean
from spatial_manifolds.data.curation import curate_clusters
from scipy.stats import zscore
from spatial_manifolds.util import gaussian_filter_nan
from spatial_manifolds.predictive_grid import compute_travel_projected, wrap_list
from spatial_manifolds.behaviour_plots import *
from spatial_manifolds.behaviour_plots import trial_cat_priority
from spatial_manifolds.detect_grids import *
from spatial_manifolds.brainrender_helper import *

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
mouse = 25
day = 25

# good examples include 
#mice = [25, 25, 26, 27, 29, 28]
#days = [25, 24, 18, 26, 23, 25]

sessions_to_use = ['M25D25', 'M27D16', 'M29D22', 
                   'M21D18', 'M21D23', 'M21D24', 'M21D25', 'M21D26',
                   'M22D34', 'M22D36', 'M25D22', 'M25D23',
                   'M25D24', 'M28D16', 'M28D17', 'M28D18',
                   'M28D20', 'M28D23', 'M28D25', 'M29D16',
                   'M29D17', 'M29D18', 'M29D19', 'M29D20', 'M29D21', 
                   'M29D23', 'M29D25', 'M20D14', 'M20D22', 'M29D22'
                   'M20D23', 'M20D25', 'M20D26', 'M21D17', 'M21D21',
                   'M21D22', 'M22D33', 'M22D37', 'M22D38', 'M22D39',
                   'M22D40', 'M22D41', 'M25D17', 'M25D19', 'M25D20',
                   'M26D11', 'M26D12', 'M26D13', 'M26D15',
                   'M26D16', 'M26D17', 'M26D19', 'M27D17',
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
    rc, rsc, vr_ns = cell_classification_vr(mouse, day)

    g_m_ids, g_m_cluster_ids = HDBSCAN_grid_modules(gcs, all, mouse, day, min_cluster_size=3, cluster_selection_epsilon=3, 
                                                    figpath=fig_path, curate_with_vr=True, curate_with_brain_region=True) # create grid modules using HDBSCAN    

    plot_grid_modules_rate_maps(gcs, g_m_ids, g_m_cluster_ids, mouse, day, figpath=fig_path)

    # we now have cluster ids classified into modules, non grid spatial cells and non spatial cells 
    # as defined by activity in the open field
    g_m_cluster_ids = sorted(g_m_cluster_ids, key=len, reverse=True) 
    cluster_ids_by_group = []
    cluster_ids_by_group.extend(g_m_cluster_ids) # grid cells by module [0,1,2...]
    cluster_ids_by_group.append(ngs.cluster_id.values.tolist()) # non grid spatial [-4]
    cluster_ids_by_group.append(ns.cluster_id.values.tolist()) # non spatial cells [-3]
    cluster_ids_by_group.append(gcs.cluster_id.values.tolist()) # all grid cells [-2]
    cluster_ids_by_group.append(sc.cluster_id.values.tolist()) # speed cells [-1]

    labels = []
    for mi in range(len(g_m_cluster_ids)):
        labels.append(f'GC_{mi}')
    labels.append('NGS')
    labels.append('NS')
    labels.append('all_GC')
    labels.append('S')

    for m, (cluster_ids, label) in enumerate(zip(cluster_ids_by_group, labels)):
        plot_vr_rate_maps(mouse, day, cluster_ids, label=label, figpath=fig_path)

    #plot_vr_rate_maps(mouse, day, rc.cluster_id.values, label=f'ramp_cells', figpath=fig_path)
    #plot_vr_rate_maps(mouse, day, rsc.cluster_id.values, label=f'speed_ramp_cells', figpath=fig_path)


    plot_stops_mouse_day(mouse, day, figpath=fig_path)


    plot_spectrogram(mouse, day, cluster_ids=cluster_ids_by_group[0], figpath=fig_path, label="GC")
    plot_spectrogram(mouse, day, cluster_ids=cluster_ids_by_group[-4], figpath=fig_path, label="NGS")

    plot_toroidal_projection(mouse, day, cluster_ids=cluster_ids_by_group[0], figpath=fig_path)

    plot_projected_stops(mouse, day, cluster_ids=cluster_ids_by_group[-2], label="GC", figpath=fig_path)
    plot_decoding(mouse, day, cluster_ids=cluster_ids_by_group[-2], label="GC", figpath=fig_path)

    plot_projected_stops(mouse, day, cluster_ids=cluster_ids_by_group[-4], label="NGS", figpath=fig_path)
    plot_decoding(mouse, day, cluster_ids=cluster_ids_by_group[-4], label="NGS", figpath=fig_path)


    plot_individual_rate_maps_with_avg(mouse, day, cluster_ids=cluster_ids_by_group[0], label='GC', figpath=fig_path)


    plot_individual_rate_maps_with_avg(mouse, day, cluster_ids=cluster_ids_by_group[-4], label='NGS', figpath=fig_path)


    #plot_decoding(mouse, day, cluster_ids=np.intersect1d(cluster_ids_by_group[-4], rc.cluster_id.values), label="NGS_ramp_cells", figpath=fig_path)


    #plot_decoding(mouse, day, cluster_ids=rc.cluster_id.values, label="ramp_cells", figpath=fig_path)

    #compare_decodings(mouse, day, cluster_ids_1=cluster_ids_by_group[-2], 
    #                  cluster_ids_2=np.intersect1d(cluster_ids_by_group[-4], rc.cluster_id.values), label1='GC', label2='RC_NGS', figpath=fig_path)

    #compare_decodings(mouse, day, cluster_ids_1=cluster_ids_by_group[-2], 
    #                  cluster_ids_2=cluster_ids_by_group[-4], label1='GC', label2='NGS', figpath=fig_path)


