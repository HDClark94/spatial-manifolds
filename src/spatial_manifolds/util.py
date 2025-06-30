from collections import defaultdict

import numpy as np
import pynapple as nap
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter


def tree():
    return defaultdict(tree)


def in_leaves(X, slice):
    return {key: value[slice] for key, value in X.items()}


def get_X(X, model_type):
    if model_type == '':
        return np.ones((len(next(iter(X.values()))), 1))
    else:
        return np.concatenate(
            [X[sub_type] for sub_type in model_type], axis=-1
        )


def interpolate_nans(tsd, pkind='cubic'):
    times = tsd.times()
    arr = tsd.values
    """
    Interpolates data to fill nan values

    Parameters:
        padata : nd array
            source data with np.NaN values

    Returns:
        nd array
            resulting data with interpolated values instead of nans
    """
    aindexes = np.arange(arr.shape[0])
    (agood_indexes,) = np.where(np.isfinite(arr))
    f = interp1d(
        agood_indexes,
        arr[agood_indexes],
        bounds_error=False,
        copy=False,
        fill_value='extrapolate',
        kind=pkind,
    )
    return nap.Tsd(d=f(aindexes), t=times)


def gaussian_filter_nan(X, sigma, mode='reflect'):
    V = X.copy()
    V[np.isnan(X)] = 0
    VV = gaussian_filter(V, sigma=sigma, mode=mode)

    W = 0 * X.copy() + 1
    W[np.isnan(X)] = 0 
    WW = gaussian_filter(W, sigma=sigma, mode=mode)
    Y = VV / WW
    Y[np.isnan(X)] = np.nan
    return Y