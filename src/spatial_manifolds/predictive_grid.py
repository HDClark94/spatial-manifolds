from collections import defaultdict

import numpy as np
import pynapple as nap
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter


def wrap_list(obj):
    return obj if isinstance(obj, list | tuple) else [obj]


def compute_travel_projected(var_label, var_values, P, travel):
    n = len(P)

    # Compute cumulative distances based on dimensionality
    if P.ndim == 1:
        segment_lengths = np.abs(np.diff(P))
    else:
        deltas = np.diff(P, axis=0)
        segment_lengths = np.linalg.norm(deltas, axis=1)

    cum_distances = np.insert(np.cumsum(segment_lengths), 0, 0)

    projected_vals = []
    valid_times = []
    j = 0
    times = P.times()

    for i in range(n):
        target_distance = cum_distances[i] + travel

        # Advance j until we find the segment that contains the projected distance
        while j < n and cum_distances[j] < target_distance:
            j += 1

        if j >= n:
            break  # Stop if out of bounds

        d1 = cum_distances[j - 1]
        d2 = cum_distances[j]
        t = (target_distance - d1) / (d2 - d1)

        interp_val = var_values[j - 1] + t * (
            var_values[j] - var_values[j - 1]
        )

        projected_vals.append(interp_val)
        valid_times.append(times[i])

    projected_vals = np.array(projected_vals)
    valid_times = np.array(valid_times)

    return nap.TsdFrame(
        t=valid_times, d=projected_vals, columns=wrap_list(var_label)
    )