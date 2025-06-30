import pandas as pd
import Elrond.settings as settings
import warnings
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, Isomap
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate
import joblib as jl
import cebra.datasets
from cebra import CEBRA
import cebra.integrations.plotly
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import sklearn.metrics
from matplotlib.collections import LineCollection
import sklearn.metrics
from spatial_manifolds.anaylsis_parameters import tl, time_bs

def encode_1d_to_2d(positions, min_val=0, max_val=tl):
    # Calculate the circumference
    circumference = max_val - min_val
    # Normalize positions to fall between 0 and 1
    normalized_positions = (positions - min_val) / circumference
    # Calculate 2D coordinates
    x_positions = np.cos(2 * np.pi * normalized_positions)
    y_positions = np.sin(2 * np.pi * normalized_positions)
    return np.array([x_positions, y_positions]).T


def decode_2d_to_1d(coordinates, min_val=0, max_val=tl):
    # Calculate the circumference
    circumference = max_val - min_val
    # Transform 2D coordinates back into angles
    angles = np.arctan2(coordinates[:,1], coordinates[:,0])
    # Normalize angles to fall between 0 and 1
    normalized_angles = (angles % (2 * np.pi)) / (2 * np.pi)
    # Calculate original positions
    positions = normalized_angles * circumference + min_val
    return positions


def computer_behaviour_kinematics(position_data, xnew_length, xnew_time_bin_size, track_length):
    resampled_behavioural_data = pd.DataFrame()
    trial_numbers = np.array(position_data['trial_number'], dtype=np.int64)
    x_position_cm = np.array(position_data['x_position_cm'], dtype="float64")
    time_seconds = np.array(position_data['time_seconds'], dtype="float64")
    trial_types = np.array(position_data['trial_type'], dtype="float64")
    x_position_elapsed_cm = (track_length*(trial_numbers-1))+x_position_cm

    x = time_seconds
    y = x_position_elapsed_cm
    f = interpolate.interp1d(x, y)
    xnew = np.arange(xnew_time_bin_size/2, (xnew_length*xnew_time_bin_size)+xnew_time_bin_size, xnew_time_bin_size)
    xnew = xnew[:xnew_length]
    ynew = f(xnew)
    x_position_cm = ynew%track_length
    speed = np.append(0, np.diff(ynew))
    acceleration = np.append(0, np.diff(speed))
    new_trial_numbers = (ynew//track_length).astype(np.int64)+1

    # recalculate trial types 
    tts = []
    for ntn in new_trial_numbers: 
        tt = trial_types[trial_numbers == ntn][0]
        tts.append(tt)
    new_trial_types = np.array(tts)

    resampled_behavioural_data["time_seconds"] = xnew
    resampled_behavioural_data["x_position_cm"] = x_position_cm
    resampled_behavioural_data["speed"] = speed
    resampled_behavioural_data["acceleration"] = acceleration
    resampled_behavioural_data["trial_numbers"] = new_trial_numbers
    resampled_behavioural_data["trial_types"] = new_trial_types
    return resampled_behavioural_data


def decoding_pos(emb_train, emb_test, label_train, n_neighbors=36):
    pos_decoder = KNeighborsRegressor(n_neighbors, metric = 'cosine')
    pos_decoder.fit(emb_train, label_train)
    pos_pred = pos_decoder.predict(emb_test)
    return pos_pred

def split_data(neural_data, label_data, test_ratio):

    split_idx = int(len(neural_data)* (1-test_ratio))
    neural_train = neural_data[:split_idx, :]
    neural_test = neural_data[split_idx:, :]
    label_train = label_data[:split_idx, :]
    label_test = label_data[split_idx:, :]

    return neural_train, neural_test, label_train, label_test



def plot_embeddings(ax, embedding, label, idx_order = (0,1,2), cmap="", viewing_angle=1):
    idx1, idx2, idx3 = idx_order
    if cmap=="track":
        # Define the custom colormap using discrete colors
        colors = ['grey', 'yellow', 'green', 'orange', 'black']
        boundaries = [0, 30, 90, 110, 170, 200]
        custom_cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(boundaries, custom_cmap.N, clip=True)
        r=ax.scatter(embedding[:,idx1],embedding[:, idx2], embedding[:, idx3], c=label, cmap=custom_cmap, norm=norm, s=0.5)
    else:
        r=ax.scatter(embedding[:,idx1],embedding[:, idx2], embedding[:, idx3], c=label, cmap=cmap, s=0.5)
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')

    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Transparent spines
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Transparent panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
 
    # No ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    if viewing_angle == 1:
        ax.view_init(elev=0, azim=0)
    elif viewing_angle == 2:
        ax.view_init(elev=30, azim=45)
    elif viewing_angle == 3:
        ax.view_init(elev=60, azim=30)
    return ax