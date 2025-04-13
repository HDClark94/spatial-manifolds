import numpy as np
import pynapple as nap
from scipy.stats import norm

from spatial_manifolds.data.binning import get_bin_config


def compute_hd_mean_vector_length(
    session, session_type, cluster_spikes, cluster_id, alpha, n_shuffles
):
    results = {'cluster_id': cluster_id}
    shuffles = [
        nap.shift_timestamps(cluster_spikes, min_shift=20.0)
        for _ in range(n_shuffles)
    ]
    bin_config = get_bin_config(session_type)['H']
    tcs = nap.compute_1d_tuning_curves(
        nap.TsGroup([cluster_spikes] + shuffles),
        session['H'],
        nb_bins=bin_config['num_bins'],
        minmax=bin_config['bounds'],
        ep=session['moving'],
    )
    scores = mean_vector_length(tcs.index.values, tcs.values.T)

    results['hd_mean_vector_length'] = scores[0]
    results['null_distribution'] = scores[1:]
    results['sig'] = results['hd_mean_vector_length'] > norm.ppf(
        1 - alpha, loc=np.mean(scores[1:]), scale=np.std(scores[1:])
    )
    return [results]


def mean_vector_length(angles, weights):
    dx = np.cos(angles)
    dy = np.sin(angles)
    totx = np.sum(dx * weights, axis=1) / np.sum(weights, axis=1)
    toty = np.sum(dy * weights, axis=1) / np.sum(weights, axis=1)
    return np.sqrt(totx**2 + toty**2)
