import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
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

from spatial_manifolds.data.binning import get_bin_config
from spatial_manifolds.data.loading import load_session
from spatial_manifolds.util import gaussian_filter_nan

import warnings
warnings.filterwarnings('ignore')


def circ_mean(x, axis=-1, keepdims=False):
    """Calculate circular mean."""
    return np.arctan2(
        np.sin(x).mean(axis, keepdims=keepdims), np.cos(x).mean(axis, keepdims=keepdims)
    )


def spectral_analysis(tcs, tl, bs, windowsize=8):
    """
    tcs: list of 1d np.array
    tl: track length
    bs: size of bin in cm
    windowsize: how many track lengths to compute over
    """
    k=1
    dark_only = False
    """Perform spectral analysis of grid cells."""
    N = len(tcs)
    L = tl/bs
    zmaps = list(tcs.values())
    trial_starts = (0, len(zmaps[0])) # unused in our data as we dont have a dark only epoch
    min_freq, max_freq = 1 / 500, 1 / 20  # cm^-1
    nperseg = windowsize * L
    nperseg = 1600

    Sxs, phase_grams = [], []
    for mp in zmaps:
        f, t, Sxx = spectrogram(mp.ravel(), nperseg=nperseg, noverlap=1400)
        _, _, phase_gram = spectrogram(
            mp.ravel(), nperseg=nperseg, mode="angle", noverlap=1400
        )
        Sxs.append(Sxx[(min_freq < f) & (f < max_freq)])
        phase_grams.append(phase_gram[(min_freq < f) & (f < max_freq)])

    #for pg in phase_grams:
    #    plt.imshow(pg)
    #    plt.show()

    phase_grams = np.stack(phase_grams).T
    Sxs = np.stack(Sxs)
    Sxs = np.nan_to_num(Sxs / (np.linalg.norm(Sxs, axis=1, keepdims=True) + 1e-5))

    fvalid = f[(min_freq < f) & (f < max_freq)]
    cutoff_idx = np.where(min_freq < f)[0][0]
    start_trial, end_trial = trial_starts[0], trial_starts[1]

    dark_idxs = (
        np.concatenate([np.where((t / L > start_trial) & (t / L < end_trial))[0]])
        if dark_only
        else np.arange(len(t))
    )
    RR = np.corrcoef(Sxs[:, :, dark_idxs].reshape(N, -1))
    np.fill_diagonal(RR, 0)

    # Cluster by spectrograms
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=0)
    kmeans.fit(Sxs[:, :, dark_idxs].reshape(N, -1))

    avg_corrs = []
    for mi in range(k):
        grid_cell_idxs = np.where(kmeans.labels_ == mi)[0]
        avg_corrs.append(RR[grid_cell_idxs][:, grid_cell_idxs].mean())
    avg_corrs = np.stack(avg_corrs)
    module_idxs = np.where(avg_corrs > 0.01)[0]
    print("Identified {} candidate modules".format(len(module_idxs)))

    f_modules, phi_modules, grid_cell_idxs_modules = [], [], []
    for mi in module_idxs:
        grid_cell_idxs = np.where(kmeans.labels_ == mi)[0]
        grid_cell_idxs_modules.append(grid_cell_idxs)

        S = Sxs[grid_cell_idxs].mean(0).reshape(-1, len(t)).T
        peak1s = []
        peak2s = []
        peak3s = []
        f1s = []
        f2s = []
        f3s = []
        phi1s, phi2s, phi3s = [], [], []
        colinearity_scores = []
        for phase_gram, s in zip(phase_grams, S):
            peaks, _ = find_peaks(s, distance=1)
            proms, _, _ = peak_prominences(s, peaks)
            top_peaks = peaks[np.argsort(proms)][-2:]
            if len(top_peaks) > 0:
                peak1, peak2 = np.min(top_peaks), np.max(top_peaks)
                if peak1 + peak2 + cutoff_idx >= len(fvalid):
                    peak3 = np.abs(peak1 - peak2) - cutoff_idx
                elif np.abs(peak1 - peak2) - cutoff_idx < 0:
                    peak3 = peak1 + peak2 + cutoff_idx
                else:
                    peak3_options = [
                        peak1 + peak2 + cutoff_idx,
                        np.abs(peak1 - peak2) - cutoff_idx,
                    ]
                    peak3 = peak3_options[np.argmax(s[peak3_options])]
                peak1s.append(peak1)
                peak2s.append(peak2)
                peak3s.append(peak3)
                f1s.append(fvalid[peak1])
                f2s.append(fvalid[peak2])
                f3s.append(fvalid[peak3])
                phi1s.append(phase_gram[peak1])
                phi2s.append(phase_gram[peak2])
                phi3s.append(phase_gram[peak3])
                uniq_peaks = np.unique([peak1, peak2, peak3])
                colinearity_scores.append(
                    100 * (s[uniq_peaks] ** 2).sum() / (s**2).sum()
                )
            else:
                peak1s.append(np.nan)
                peak2s.append(np.nan)
                peak3s.append(np.nan)
                f1s.append(np.nan)
                f2s.append(np.nan)
                f3s.append(np.nan)
                phi1s.append(np.nan)
                phi2s.append(np.nan)
                phi3s.append(np.nan)
                colinearity_scores.append(np.nan)
        f1s = np.stack(f1s)
        f2s = np.stack(f2s)
        f3s = np.stack(f3s)
        phi1s = np.stack(phi1s)
        phi2s = np.stack(phi2s)
        phi3s = np.stack(phi3s)

        phi1_dark = phi1s[dark_idxs][:, grid_cell_idxs]
        phi1_dark = (phi1_dark - circ_mean(phi1_dark, axis=-1, keepdims=True)) % (
            2 * np.pi
        )
        phi2_dark = phi2s[dark_idxs][:, grid_cell_idxs]
        phi2_dark = (phi2_dark - circ_mean(phi2_dark, axis=-1, keepdims=True)) % (
            2 * np.pi
        )
        phi3_dark = phi3s[dark_idxs][:, grid_cell_idxs]
        phi3_dark = (phi3_dark - circ_mean(phi3_dark, axis=-1, keepdims=True)) % (
            2 * np.pi
        )

        xy1 = np.hstack([np.cos(phi1_dark), np.sin(phi1_dark)])
        xy2 = np.hstack([np.cos(phi2_dark), np.sin(phi2_dark)])
        xy3 = np.hstack([np.cos(phi3_dark), np.sin(phi3_dark)])
        xy = [xy1, xy2, xy3]

        kmeans_phases = KMeans(n_clusters=3, n_init="auto", random_state=0)
        kmeans_phases.fit(np.vstack(xy))
        labels = np.stack(np.split(kmeans_phases.labels_, 3)).T

        phi_all = [phi1_dark, phi2_dark, phi3_dark]
        fs_all = [f1s, f2s, f3s]

        fs = {}
        phis = {}
        for j in range(3):
            phis[j] = []
            fs[j] = []
        for i in range(len(dark_idxs)):
            for j in range(3):
                label = np.where(labels[i] == j)[0]
                if len(label) > 0:
                    phis[j].append(phi_all[label[0]][i])
                    fs[j].append(fs_all[label[0]][i])
                else:
                    min_label = (
                        ((np.stack(xy)[:, i] - kmeans_phases.cluster_centers_[j]) ** 2)
                        .sum(-1)
                        .argmin()
                    )
                    fs[j].append(fs_all[min_label][i])
                    phis[j].append(phi_all[min_label][i])

        for j in range(3):
            phis[j] = np.stack(phis[j])
            fs[j] = np.stack(fs[j])
        f_modules.append(fs)

        phases_all = []
        for j in range(3):
            r = np.corrcoef(np.sin(phis[j]))
            np.fill_diagonal(r, 0)
            phases_all.append(circ_mean(phis[j][r.max(0) > 0.5], axis=0))
        phi_modules.append(phases_all)

        scales = []
        for fs in f_modules:
            scale = np.sqrt(2 / (np.stack([f for f in fs.values()])**2).sum(0)).mean()
            scales.append(scale)
        
        scale_sort = np.argsort(scales)
        f_modules = [f_modules[i] for i in scale_sort]
        phi_modules = [phi_modules[i] for i in scale_sort]
        grid_cell_idxs_modules = [grid_cell_idxs_modules[i] for i in scale_sort]

    return [f_modules, 
            phi_modules, 
            grid_cell_idxs_modules, 
            Sxs,
            t,
            fvalid,
            trial_starts]