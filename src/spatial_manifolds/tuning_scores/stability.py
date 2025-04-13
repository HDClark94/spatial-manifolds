import warnings

import numpy as np
import pynapple as nap
from scipy.stats import norm

from spatial_manifolds.data.binning import get_bin_config
from spatial_manifolds.util import gaussian_filter_nan


def compute_stability(
    session, session_type, cluster_spikes, cluster_id, alpha, n_shuffles
):
    shuffles = [
        nap.shift_timestamps(cluster_spikes, min_shift=20.0)
        for _ in range(n_shuffles)
    ]
    bin_config = get_bin_config(session_type)['P']

    results = []
    trial_groups = {}
    for rz, trial_types in enumerate(bin_config['regions'].keys(), 1):
        trial_groups[f'rz{rz}_b+nb'] = trial_types
        for trial_type_label, trial_type in zip(
            ['b', 'nb'], trial_types, strict=False
        ):
            trial_groups[f'rz{rz}_{trial_type_label}'] = [trial_type]
    for group, group_trial_types in trial_groups.items():
        trials = session['trials'][
            session['trials']['trial_type'].isin(group_trial_types)
        ]
        # Compute tuning curves per trial subset
        tcs_splits = []
        for trial_subset in np.array_split(np.arange(len(trials)), 4):
            with np.errstate(invalid='ignore', divide='ignore'):
                tcs_splits.append(
                    gaussian_filter_nan(
                        nap.compute_1d_tuning_curves(
                            nap.TsGroup([cluster_spikes] + shuffles),
                            session['P'],
                            nb_bins=bin_config['num_bins'],
                            minmax=bin_config['bounds'],
                            ep=session['moving'].intersect(
                                trials[trial_subset]
                            ),
                        ),
                        sigma=(bin_config['smooth_sigma'], 0),
                        mode='wrap',
                    )
                )
        # Compute average correlation across trial subsets
        scores = []
        for n in range(1 + n_shuffles):
            with np.errstate(invalid='ignore'):
                corr = np.corrcoef(
                    np.array([tcs_split[:, n] for tcs_split in tcs_splits])
                )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    category=RuntimeWarning,
                    message='Mean of empty slice',
                )
                scores.append(np.nanmean(corr[np.triu_indices(4, k=1)]))
        results.append(
            {
                'cluster_id': cluster_id,
                'group': group,
                'stability': scores[0],
                'null_distribution': scores[1:],
                'sig': scores[0]
                > norm.ppf(
                    1 - alpha,
                    loc=np.mean(scores[1:]),
                    scale=np.std(scores[1:]),
                ),
            }
        )
    return results
