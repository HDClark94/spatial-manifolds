import numpy as np
import pynapple as nap
from scipy.stats import norm

from spatial_manifolds.util import interpolate_nans


def compute_speed_correlation(
    session, session_type, cluster_spikes, cluster_id, alpha, n_shuffles
):
    results = {'cluster_id': cluster_id}
    shuffles = [
        nap.shift_timestamps(cluster_spikes, min_shift=20.0)
        for _ in range(n_shuffles)
    ]
    fr = (
        nap.TsGroup([cluster_spikes] + shuffles)
        .count(0.02)
        .smooth(0.30, windowsize=3, norm=True)
    )
    speed = interpolate_nans(
        session['S'].bin_average(0.02, ep=fr.time_support)
    ).smooth(0.30, windowsize=3, norm=True)
    scores = correlation(
        fr.restrict(session['moving']).values,
        speed.restrict(session['moving']).values,
    )
    null_mean = np.nanmean(scores[1:])
    null_std = np.nanstd(scores[1:])

    results['speed_correlation'] = scores[0]
    results['null_distribution'] = scores[1:]
    results['sig'] = results['speed_correlation'] > norm.ppf(
        1 - (alpha / 2), loc=null_mean, scale=null_std
    ) or results['speed_correlation'] < norm.ppf(
        alpha / 2, loc=null_mean, scale=null_std
    )
    return [results]


def correlation(X, y):
    # Mean-center X and y
    X_mean = np.nanmean(X, axis=0)  # Shape (1, D)
    y_mean = np.nanmean(y)  # Scalar

    X_centered = X - X_mean  # Shape (N, D)
    y_centered = y - y_mean  # Shape (N,)

    # Compute standard deviations
    X_std = np.nanstd(X, axis=0, ddof=1)  # Shape (1, D)
    y_std = np.nanstd(y, ddof=1)  # Scalar

    # Compute correlation
    return (X_centered.T @ y_centered) / ((X.shape[0] - 1) * X_std * y_std)
