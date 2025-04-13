from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from spatial_manifolds.tuning_scores import TUNING_SCORES

if __name__ == '__main__':
    parser = ArgumentParser('Summarising tuning for Nolan lab ephys data.')
    parser.add_argument(
        '--storage',
        type=Path,
        default=Path('./data/'),
        help='Path to the storage.',
    )
    args = parser.parse_args()

    # Verify that tuning curves have been computed
    sessions = list((args.storage / 'sessions').glob('*/*/'))
    print(sessions)
    # Filter sessions that have all required subdirectories
    sessions = [
        s
        for s in sessions
        if all(
            (s / subdir / 'tuning_scores').exists()
            for subdir in ['OF1', 'OF2', 'VR']
        )
    ]

    # Get classses
    tuning = []
    for session in sessions:
        for subdir in ['OF1', 'OF2', 'VR']:
            for score_type in TUNING_SCORES[subdir[:2]]:
                scores = pd.read_parquet(
                    session
                    / subdir
                    / 'tuning_scores'
                    / f'{score_type}.parquet'
                )
                for cluster_id in scores.index:
                    tuning.append(
                        (
                            session.parent.name,
                            session.name,
                            subdir,
                            cluster_id,
                            score_type,
                            scores.loc[cluster_id, score_type],
                            scores.loc[cluster_id, 'sig'],
                        )
                    )
    tuning = pd.DataFrame(
        tuning,
        columns=[
            'mouse',
            'day',
            'session_type',
            'cluster_id',
            'score_type',
            'score',
            'sig',
        ],
    )
    tuning.to_csv(args.storage / 'tuning.csv', index=False)
