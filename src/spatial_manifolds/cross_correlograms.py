import numpy as np
import pandas as pd
import xarray as xr
from numba import njit
from scipy.signal import convolve
from scipy.stats import poisson

@njit
def _CCH_inner(spikes_i, spikes_j, bin_edges, window):
    counts = np.zeros(len(bin_edges) - 1, dtype=np.int32)
    start_idx = 0

    for t in spikes_i:
        while start_idx < len(spikes_j) and spikes_j[start_idx] < t - window:
            start_idx += 1
        end_idx = start_idx
        while end_idx < len(spikes_j) and spikes_j[end_idx] <= t + window:
            delta = spikes_j[end_idx] - t
            bin_idx = np.searchsorted(bin_edges, delta, side="right") - 1
            if 0 <= bin_idx < len(counts):
                counts[bin_idx] += 1
            end_idx += 1

    return counts

 
def compute_CCH(spike_dict, subset, bin_size=0.001, window=0.05):
    """
    Fast all-to-all cross-correlograms using Numba, returned as xarray.DataArray.
    Includes zero-lag bin.
    """
    N = len(subset)

    # Make bin edges so that 0 is the center of a bin
    half_bins = int(np.floor(window / bin_size))
    bin_edges = np.linspace(
        -half_bins * bin_size, half_bins * bin_size, 2 * half_bins + 1 + 1
    )
    lag_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    B = len(lag_centers)
    cch = np.zeros((N, N, B), dtype=np.int32)

    spike_list = [np.ascontiguousarray(spike_dict[n].times()) for n in subset]

    for i in range(N):
        spikes_i = spike_list[i]
        for j in range(N):
            if i == j:
                continue  # optionally keep, or comment out if you want auto-CCHs too
            spikes_j = spike_list[j]
            cch[i, j] = _CCH_inner(spikes_i, spikes_j, bin_edges, window)

    return xr.DataArray(
        cch,
        dims=("source", "target", "lag"),
        coords={"source": subset, "target": subset, "lag": lag_centers},
        name="cross_correlogram",
    )


def classify_CCH(raw, baseline, lags, bin_size_sec=0.001):
    corrected = raw - baseline

    # peak falls in causal window
    peak_idx = np.argmax(corrected)
    if not 0.0007 <= lags[peak_idx] <= 0.0047:
        return False

    # peak is significant
    pval_peak = (
        1
        - poisson.cdf(raw[peak_idx] - 1, baseline[peak_idx])
        - 0.5 * poisson.pmf(raw[peak_idx], baseline[peak_idx])
    )
    if pval_peak >= 0.001:
        return False

    # peak width < 0.003 s
    def is_in_peak(i):
        if i < 0 or i >= len(corrected):
            return False
        pval = (
            1
            - poisson.cdf(raw[i] - 1, baseline[i])
            - 0.5 * poisson.pmf(raw[i], baseline[i])
        )
        return (
            raw[i] > (raw[peak_idx] / 2) or raw[i] > 2 * np.std(baseline)
        ) and pval < 0.01

    peak_bins = [peak_idx]
    for direction in [-1, 1]:
        i = peak_idx + direction
        while is_in_peak(i):
            peak_bins.append(i)
            i += direction
    if (len(peak_bins) * bin_size_sec) > 0.003:
        return False

    # no overlap with zero-lag bin
    zero_lag_idx = np.where(np.isclose(lags, 0))[0]
    if zero_lag_idx[0] in peak_bins:
        return False

    # Exclusion A: any non-peak bin > 3.5 × std of corrected
    # nonpeak_mask = np.ones_like(corrected, dtype=bool)
    # nonpeak_mask[peak_bins] = False
    # if np.any(corrected[nonpeak_mask] > 3.5 * np.std(corrected)):
    #    return False

    # Exclusion B: any bin in anticausal (<0) with p < 0.005
    anticausal_mask = lags < 0
    raw_antic = raw[anticausal_mask]
    baseline_antic = baseline[anticausal_mask]
    pvals_antic = (
        1
        - poisson.cdf(raw_antic - 1, baseline_antic)
        - 0.5 * poisson.pmf(raw_antic, baseline_antic)
    )
    if np.any(pvals_antic < 0.01):
        return False

    return True