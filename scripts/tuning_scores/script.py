import re
from argparse import ArgumentParser
from pathlib import Path

from spatial_manifolds.data.loading import load_session
from spatial_manifolds.tuning_scores import TUNING_SCORES, compute_tuning_score

if __name__ == '__main__':
    parser = ArgumentParser(
        'Computing tuning scores for Nolan lab ephys data.'
    )
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
        '--alpha', type=float, default=0.05, help='Significance level.'
    )
    parser.add_argument(
        '--n_jobs', type=int, default=1, help='Number of parallel jobs.'
    )
    parser.add_argument(
        '--n_shuffles', type=int, default=200, help='Number of shuffles.'
    )
    parser.add_argument('--seed', type=int, default=420, help='Random seed.')
    args = parser.parse_args()

    # Load session
    session, session_path, clusters = load_session(args)
    (session_path / 'tuning_scores').mkdir(exist_ok=True)

    # Compute tuning scores
    for tuning_score, (tuning_score_fn, parallelise) in TUNING_SCORES[
        re.sub(r'\d+', '', args.session_type)
    ].items():
        print(f'Computing {tuning_score}...')
        result = (
            compute_tuning_score(
                session_path
                / f'M{args.mouse}D{args.day}{args.session_type}.nwb',
                args.session_type,
                clusters,
                tuning_score_fn,
                log_file=session_path / f'{tuning_score}.log',
                alpha=args.alpha,
                n_shuffles=args.n_shuffles,
                n_jobs=args.n_jobs,
            )
            if parallelise
            else tuning_score_fn(session, args.session_type, clusters)
        )
        print(result)
        print(result['sig'].sum())
        result.to_parquet(
            session_path / 'tuning_scores' / f'{tuning_score}.parquet'
        )
