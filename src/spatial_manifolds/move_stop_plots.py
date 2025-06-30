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

import os
import shutil


def copy_files_with_word(src_folder, dest_folder, word):
    # Create destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # Iterate over all files in the source folder
    for filename in os.listdir(src_folder):
        src_file_path = os.path.join(src_folder, filename)
        
        # Check if it's a file
        if os.path.isfile(src_file_path):
            # Read the file content
            if word in src_file_path:
                # Copy the file to the destination folder
                shutil.copy(src_file_path, dest_folder)
                print(f"Copied: {filename}")

# Example usage
src_folder = '/Users/harryclark/Documents/figs/grids/'
dest_folder = '/Users/harryclark/Documents/figs/stops/'
word = 'stops'

copy_files_with_word(src_folder, dest_folder, word)
