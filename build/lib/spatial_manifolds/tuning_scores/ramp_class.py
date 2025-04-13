import warnings
from collections import defaultdict

import numpy as np
import pynapple as nap
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.multitest import fdrcorrection

from spatial_manifolds.data.binning import get_bin_config
from spatial_manifolds.util import gaussian_filter_nan


def compute_ramp_class(
    session, session_type, cluster_spikes, cluster_id, alpha, n_shuffles
):
    shuffles = [
        nap.shift_timestamps(cluster_spikes, min_shift=20.0)
        for _ in range(n_shuffles)
    ]
    bin_config = get_bin_config(session_type)['P']

    results = []
    trial_groups = {}
    for rz, (trial_types, regions) in enumerate(
        bin_config['regions'].items(), 1
    ):
        trial_groups[f'rz{rz}_b+nb'] = (trial_types, regions)
        for trial_type_label, trial_type in zip(
            ['b', 'nb'], trial_types, strict=False
        ):
            trial_groups[f'rz{rz}_{trial_type_label}'] = (
                [trial_type],
                regions,
            )
    for group, (trial_types, regions) in trial_groups.items():
        trials = session['trials'][
            session['trials']['trial_type'].isin(trial_types)
        ]

        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                category=UserWarning,
                module='pynapple.core.interval_set',
            )

            tcs = nap.compute_1d_tuning_curves(
                nap.TsGroup(shuffles + [cluster_spikes]),
                session['P'],
                nb_bins=bin_config['num_bins'],
                minmax=bin_config['bounds'],
                ep=session['moving'].intersect(trials),
            )
            tcs_index = tcs.index
            tcs = gaussian_filter_nan(
                tcs.values, sigma=(bin_config['smooth_sigma'], 0), mode='wrap'
            )

        # Compute ramp fits
        for region, (start, end) in regions.items():
            null_distribution = defaultdict(list)
            for n in range(1 + n_shuffles):
                mask = (tcs_index > start) & (tcs_index < end)
                model = sm.OLS(
                    tcs[mask, n], sm.add_constant(tcs_index[mask])
                ).fit()
                if n == n_shuffles:
                    results.append(
                        {
                            'cluster_id': cluster_id,
                            'group': group,
                            'region': region,
                            'slope': model.params[1],
                            'intercept': model.params[1],
                            'pval': model.pvalues[1],
                            **null_distribution,
                        }
                    )
                else:
                    null_distribution['null_slope'].append(model.params[1])
                    null_distribution['null_intercept'].append(model.params[1])
                    null_distribution['null_pval'].append(model.params[1])

    for result in results:
        result['sig'] = fdrcorrection(
            [result['pval']] + result['null_pval'], alpha
        )[0][0] and (
            result['slope']
            > norm.ppf(
                1 - alpha,
                loc=np.nanmean(result['null_slope']),
                scale=np.nanstd(result['null_slope']),
            )
        )
        result['sign'] = (
            '/' if not result['sig'] else '+' if result['slope'] > 0 else '-'
        )
    return results
