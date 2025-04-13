from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
import seaborn as sns

from spatial_manifolds.data.binning import get_bin_config
from spatial_manifolds.data.loading import load_session
from spatial_manifolds.util import gaussian_filter_nan

if __name__ == '__main__':
    parser = ArgumentParser('Plotting tuning curves for Nolan lab ephys data.')
    parser.add_argument(
        '--storage',
        type=Path,
        default='./data/',
        help='Path to the storage.',
    )
    parser.add_argument(
        '--sorter',
        type=str,
        default='kilosort4',
        help='Sorting to use.',
    )
    parser.add_argument(
        '--session_type',
        type=str,
        default='VR',
        help='Session type to run the analysis for.',
        choices=['VR', 'MCVR', 'OF1', 'OF2'],
    )
    parser.add_argument(
        '--mouse',
        type=int,
        required=True,
        help='Mouse to run the analysis for.',
    )
    parser.add_argument(
        '--day',
        type=int,
        help='Day to run the analysis for.',
    )
    parser.add_argument(
        '--alpha', type=float, default=0.001, help='Significance level.'
    )
    parser.add_argument(
        '--n_jobs', type=int, default=12, help='Number of parallel jobs.'
    )
    parser.add_argument(
        '--n_shuffles', type=int, default=200, help='Number of shuffles.'
    )
    parser.add_argument('--seed', type=int, default=420, help='Random seed.')
    args = parser.parse_args()

    # Load session
    session, session_path, clusters = load_session(args)
    tuning_scores = {
        path.with_suffix('').name: pd.read_parquet(path)
        for path in (session_path / 'tuning_scores').iterdir()
    }
    ramp_scores = tuning_scores['ramp_class']
    bin_config = get_bin_config(args.session_type)['P']

    # Plot
    groups = [
        ('rz1_b', (0, 1), 'blue'),
        ('rz1_nb', (0, 1), 'blue'),
        ('rz2_b', (2, 3), 'red'),
        ('rz2_nb', (2, 3), 'red'),
    ]
    fig, axs = plt.subplots(2, 2, sharex=True)
    for (group, trial_group, color), ax in zip(groups, axs.flatten()):
        regions = bin_config['regions'][trial_group]
        tcs = nap.compute_1d_tuning_curves(
            clusters[
                ramp_scores[
                    (ramp_scores['group'] == group)
                    & (ramp_scores['sign'] == '+')
                    & (ramp_scores['region'] == 'outbound')
                ]['cluster_id'].values
            ],
            session['P'],
            nb_bins=bin_config['num_bins'],
            minmax=bin_config['bounds'],
            ep=session['moving'],
        )
        tcs_index = tcs.index
        tcs = gaussian_filter_nan(
            tcs.values,
            sigma=(bin_config['smooth_sigma'], 0),
            mode='wrap',
        )
        mean = tcs.mean(axis=1)
        sem = tcs.std(axis=1) / np.sqrt(tcs.shape[1])
        ax.plot(tcs_index, mean, label=group, color=color)
        ax.fill_between(
            tcs_index, mean - sem, mean + sem, alpha=0.2, color=color
        )
        ax.set_ylabel('spikes/s')
        ax.set_title(f'{group} (N={tcs.shape[1]})')
        ax.axvspan(
            regions['outbound'][1],
            regions['homebound'][0],
            alpha=0.2,
            zorder=-10,
            edgecolor='none',
            facecolor='gray',
        )
        sns.despine(top=True, right=True)
    ax.set_xlabel('position (cm)')
    ax.set_xlim(bin_config['bounds'])
    plt.tight_layout()
    plt.show()
