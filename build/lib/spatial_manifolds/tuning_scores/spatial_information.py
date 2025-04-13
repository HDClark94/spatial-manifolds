import numpy as np
import pynapple as nap
from scipy.stats import norm

from spatial_manifolds.data.binning import get_bin_config
from spatial_manifolds.util import gaussian_filter_nan


def compute_spatial_information(
    session, session_type, cluster_spikes, cluster_id, alpha, n_shuffles
):
    results = {'cluster_id': cluster_id}
    shuffles = [
        nap.shift_timestamps(cluster_spikes, min_shift=20.0)
        for _ in range(n_shuffles)
    ]
    bin_config = get_bin_config(session_type)
    with np.errstate(invalid='ignore', divide='ignore'):
        tcs = (
            gaussian_filter_nan(
                nap.compute_1d_tuning_curves(
                    nap.TsGroup([cluster_spikes] + shuffles),
                    session['P'],
                    nb_bins=bin_config['P']['num_bins'],
                    minmax=bin_config['P']['bounds'],
                    ep=session['moving'],
                ),
                sigma=(bin_config['P']['smooth_sigma'], 0),
                mode='wrap',
            )
            if 'trials' in session
            else gaussian_filter_nan(
                np.stack(
                    list(
                        nap.compute_2d_tuning_curves(
                            nap.TsGroup([cluster_spikes] + shuffles),
                            np.stack([session['P_x'], session['P_y']], axis=1),
                            nb_bins=bin_config[('P_x', 'P_y')]['num_bins'],
                            minmax=bin_config[('P_x', 'P_y')]['bounds'],
                            ep=session['moving'],
                        )[0].values()
                    ),
                    axis=0,
                ),
                sigma=(
                    0,
                    bin_config[('P_x', 'P_y')]['smooth_sigma'],
                    bin_config[('P_x', 'P_y')]['smooth_sigma'],
                ),
            )
        )
    scores = (
        nap.compute_1d_mutual_info(
            tcs,
            session['P'],
            ep=session['moving'],
            minmax=bin_config['P']['bounds'],
        )
        if 'trials' in session
        else nap.compute_2d_mutual_info(
            tcs,
            np.stack([session['P_x'], session['P_y']], axis=1),
            ep=session['moving'],
            minmax=bin_config[('P_x', 'P_y')]['bounds'],
        )
    )['SI'].values

    results['spatial_information'] = scores[0]
    results['null_distribution'] = scores[1:]
    results['sig'] = results['spatial_information'] > norm.ppf(
        1 - alpha, loc=np.nanmean(scores[1:]), scale=np.nanstd(scores[1:])
    )
    return [results]
