import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from numba import njit
from scipy.ndimage import rotate
from scipy.stats import norm
from skimage.feature.peak import peak_local_max

from spatial_manifolds.data.binning import get_bin_config
from spatial_manifolds.util import gaussian_filter_nan


def compute_ellipse_grid_score(
    session, session_type, cluster_spikes, cluster_id, alpha, n_shuffles
):
    return compute_grid_score(
        session,
        session_type,
        cluster_spikes,
        cluster_id,
        alpha,
        n_shuffles,
        do_ellipse_transform=True,
    )


def compute_grid_score(
    session,
    session_type,
    cluster_spikes,
    cluster_id,
    alpha,
    n_shuffles,
    do_ellipse_transform=False,
):
    """
    Computes the grid score for a given cluster.
    Based on the description in:
        https://www.biorxiv.org/content/10.1101/230250v1.full.pdf
    """
    results = {'cluster_id': cluster_id}
    shuffles = [
        nap.shift_timestamps(cluster_spikes, min_shift=20.0)
        for _ in range(n_shuffles)
    ]
    bin_config = get_bin_config(session_type)[('P_x', 'P_y')]
    with np.errstate(invalid='ignore', divide='ignore'):
        tcs = np.stack(
            list(
                nap.compute_2d_tuning_curves(
                    nap.TsGroup([cluster_spikes] + shuffles),
                    np.stack([session['P_x'], session['P_y']], axis=1),
                    nb_bins=bin_config['num_bins'],
                    minmax=bin_config['bounds'],
                    ep=session['moving'],
                )[0].values()
            )
        )
    tcs = gaussian_filter_nan(
        tcs,
        sigma=(0, bin_config['smooth_sigma'], bin_config['smooth_sigma']),
    )
    scores = []
    center = tcs.shape[1:]
    for tc in tcs:
        autocorr = autocorr2d(tc)
        autocorr_old = autocorr.copy()
        peaks = peak_local_max(
            np.nan_to_num(autocorr),
            min_distance=8,
            exclude_border=True,
        )
        distances = np.array([np.linalg.norm(center - peak) for peak in peaks])
        sorted = np.argsort(distances)[1:7]
        peaks = peaks[sorted]
        peaks_old = peaks.copy()
        distances = distances[sorted]
        if len(peaks) < 6:
            scores.append(np.nan)
            continue
        else:
            autocorr, peaks = ellipse_to_circle_transform(
                np.nan_to_num(autocorr), peaks
            )
            distances = np.array(
                [np.linalg.norm(center - peak) for peak in peaks]
            )

        # Define the ring size
        mean_distance = np.mean(distances)
        inner_radius = mean_distance * 0.5
        outer_radius = mean_distance * 1.25

        # Extract a ring around the center
        y, x = np.ogrid[: autocorr.shape[0], : autocorr.shape[1]]
        mask = (x - center[1]) ** 2 + (y - center[0]) ** 2 >= inner_radius**2
        mask &= (x - center[1]) ** 2 + (y - center[0]) ** 2 <= outer_radius**2
        ring = np.where(mask, autocorr, np.nan)

        # Compute the rotational symmetry of the autocorrelation map
        angles = [30, 60, 90, 120, 150]
        angle_scores = {}
        for angle in angles:
            rotated_ring = rotate(
                np.nan_to_num(ring, nan=0.0), angle, reshape=False
            )
            rotated_ring[rotated_ring == 0.0] = np.nan
            rotated_ring = np.where(mask, rotated_ring, np.nan)
            nans = np.isnan(rotated_ring) | np.isnan(ring)
            if np.sum(~nans) < 20:
                angle_scores[angle] = np.nan
                continue
            else:
                angle_scores[angle] = np.corrcoef(
                    ring[~nans], rotated_ring[~nans]
                )[0, 1]

        # Compute the grid score as the difference between the minimum correlation
        # coefficient for rotations of 60 and 120 degrees and the maximum correlation
        # coefficient for rotations of 30, 90, and 150 degrees
        scores.append(
            min(angle_scores[60], angle_scores[120])
            - max(angle_scores[30], angle_scores[90], angle_scores[150])
        )
    results['grid_score'] = scores[0]
    results['null_distribution'] = scores[1:]
    results['type'] = (
        'ellipse_grid_score' if do_ellipse_transform else 'grid_score'
    )
    with np.errstate(invalid='ignore'):
        results['sig'] = results['grid_score'] > norm.ppf(
            1 - alpha, loc=np.nanmean(scores[1:]), scale=np.nanstd(scores[1:])
        )

    return [results]


@njit
def autocorr2d(lambda_matrix, min_n=20):
    height, width = lambda_matrix.shape
    max_tau_x, max_tau_y = 2 * (width - 1), 2 * (height - 1)
    autocorr_map = np.full((max_tau_x + 1, max_tau_y + 1), np.nan)

    for tau_x in range(-width + 1, width):
        for tau_y in range(-height + 1, height):
            sum_lambda = 0.0
            sum_lambda_tau = 0.0
            sum_lambda_product = 0.0
            sum_lambda_sq = 0.0
            sum_lambda_tau_sq = 0.0
            n = 0

            for x in range(width):
                for y in range(height):
                    if 0 <= x + tau_x < width and 0 <= y + tau_y < height:
                        val = lambda_matrix[x, y]
                        val_tau = lambda_matrix[x + tau_x, y + tau_y]
                        if not np.isnan(val) and not np.isnan(val_tau):
                            sum_lambda += val
                            sum_lambda_tau += val_tau
                            sum_lambda_product += val * val_tau
                            sum_lambda_sq += val**2
                            sum_lambda_tau_sq += val_tau**2
                            n += 1

            if n < min_n:
                continue

            numerator = n * sum_lambda_product - sum_lambda * sum_lambda_tau
            denominator = np.sqrt(
                (n * sum_lambda_sq - sum_lambda**2)
                * (n * sum_lambda_tau_sq - sum_lambda_tau**2)
            )

            autocorr_map[tau_x + width - 1, tau_y + height - 1] = (
                numerator / denominator if denominator != 0 else np.nan
            )

    return autocorr_map


def ellipse_to_circle_transform(autocorr, peaks):
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        center, (major_axis, minor_axis), angle = cv2.fitEllipse(peaks)
    angle_rad = np.deg2rad(angle)

    # Get the scaling factors
    if major_axis == 0 or minor_axis == 0:
        return autocorr, peaks
    scale_x = (
        minor_axis / major_axis
    )  # Scale the x-axis to match the y-axis (minor axis)
    scale_y = 1.0

    # Translation to origin
    T1 = np.array(
        [[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]],
        dtype=np.float32,
    )

    # Rotation
    R1 = np.array(
        [
            [np.cos(angle_rad), np.sin(angle_rad), 0],
            [-np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    # Scaling
    S = np.array(
        [[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=np.float32
    )

    # Inverse rotation
    R2 = np.array(
        [
            [np.cos(-angle_rad), np.sin(-angle_rad), 0],
            [-np.sin(-angle_rad), np.cos(-angle_rad), 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    # Translation back
    T2 = np.array(
        [[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]], dtype=np.float32
    )

    # Combine all transformations: T2 * R2 * S * R1 * T1
    M = T2 @ R2 @ S @ R1 @ T1

    # Transform
    peaks_transformed = (
        np.hstack([peaks, np.ones((peaks.shape[0], 1))]) @ M.T
    )[:, :2]

    autocorr_transformed = cv2.warpAffine(
        autocorr, M[:2, :], (autocorr.shape[1], autocorr.shape[0])
    )
    return autocorr_transformed, peaks_transformed
