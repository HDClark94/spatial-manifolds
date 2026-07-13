"""
Border score following Solstad et al. (2008) Science 322, 1865–1868.

Formula: BS = (C_max - d_mean) / (C_max + d_mean)

  C_max  = maximum wall coverage across the four walls, where coverage of wall w
            is the fraction of wall-adjacent bins with firing rate > threshold.
  d_mean = mean distance of firing-field pixels from the nearest wall, normalised
            to [0, 1] by the number of bins.

Threshold is `threshold_fraction` × peak firing rate (default 0.3).
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def border_score_from_rate_map(rate_map, threshold_fraction=0.3):
    """
    Compute the border score for a single 2-D rate map.

    Parameters
    ----------
    rate_map : np.ndarray, shape (N, M)
        Smoothed 2-D firing rate map. NaN for unvisited bins.
    threshold_fraction : float
        Fraction of peak rate used to define the firing field. Default 0.3.

    Returns
    -------
    float : border score in [-1, 1], or np.nan if the map is invalid.
    """
    if rate_map is None or rate_map.size == 0:
        return np.nan
    if np.all(np.isnan(rate_map)):
        return np.nan

    peak_rate = np.nanmax(rate_map)
    if peak_rate <= 0 or not np.isfinite(peak_rate):
        return np.nan

    threshold   = peak_rate * threshold_fraction
    field_mask  = np.nan_to_num(rate_map, nan=0.0) >= threshold

    if not np.any(field_mask):
        return np.nan

    N, M = rate_map.shape

    # Wall coverage: fraction of wall-adjacent bins in the firing field
    coverages = [
        float(np.mean(field_mask[0,  :])),   # south (row 0)
        float(np.mean(field_mask[-1, :])),   # north (row N-1)
        float(np.mean(field_mask[:,  0])),   # west  (col 0)
        float(np.mean(field_mask[:, -1])),   # east  (col M-1)
    ]
    C_max = max(coverages)

    # Mean normalised distance from the nearest wall for field pixels
    row_idx, col_idx = np.where(field_mask)
    d_south = row_idx
    d_north = (N - 1) - row_idx
    d_west  = col_idx
    d_east  = (M - 1) - col_idx
    min_dist_bins = np.minimum(
        np.minimum(d_south, d_north),
        np.minimum(d_west,  d_east),
    )
    # Normalise so the maximum possible distance (arena centre) = 0.5
    d_mean = float(np.mean(min_dist_bins)) / (max(N, M) - 1)

    denom = C_max + d_mean
    if denom == 0:
        return np.nan

    return (C_max - d_mean) / denom


def compute_rate_map(spike_times, pos_x, pos_y, pos_times,
                     n_bins=40, arena_min=0.0, arena_max=100.0,
                     sigma=2.0, min_occupancy_s=0.02):
    """
    Build a smoothed 2-D firing rate map from spike times and position traces.

    Parameters
    ----------
    spike_times : array-like, shape (n_spikes,) — spike times in seconds
    pos_x, pos_y : array-like, shape (n_pos,) — position in cm
    pos_times : array-like, shape (n_pos,) — timestamps for pos_x/pos_y in seconds
    n_bins : int — number of bins per dimension (arena is square)
    arena_min, arena_max : float — spatial bounds in cm
    sigma : float — Gaussian smoothing σ in bins
    min_occupancy_s : float — minimum occupancy (s) for a bin to be included

    Returns
    -------
    rate_map : np.ndarray, shape (n_bins, n_bins) — smoothed rate map (Hz).
               NaN for bins below min_occupancy_s.
    """
    edges = np.linspace(arena_min, arena_max, n_bins + 1)
    dt    = float(np.median(np.diff(pos_times)))   # sample period (s)

    pos_x  = np.asarray(pos_x,    dtype=float)
    pos_y  = np.asarray(pos_y,    dtype=float)
    pos_times = np.asarray(pos_times, dtype=float)
    spike_times = np.asarray(spike_times, dtype=float)

    # Occupancy map (s)
    occ, _, _ = np.histogram2d(pos_x, pos_y, bins=[edges, edges])
    occ = occ * dt

    # Spike map: bin each spike to the position interpolated at spike time
    spk_x = np.interp(spike_times, pos_times, pos_x)
    spk_y = np.interp(spike_times, pos_times, pos_y)
    spk, _, _ = np.histogram2d(spk_x, spk_y, bins=[edges, edges])

    # Smooth both maps then divide
    occ_sm = gaussian_filter(occ,  sigma=sigma)
    spk_sm = gaussian_filter(spk,  sigma=sigma)

    with np.errstate(invalid='ignore', divide='ignore'):
        rm = np.where(occ_sm >= min_occupancy_s, spk_sm / occ_sm, np.nan)

    return rm
