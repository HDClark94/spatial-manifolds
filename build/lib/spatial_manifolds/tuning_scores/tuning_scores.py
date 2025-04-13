import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import pynapple as nap
from tqdm import tqdm


def compute_tuning_score(
    session_path,
    session_type,
    clusters,
    tuning_score_fn,
    log_file,
    alpha,
    n_shuffles,
    n_jobs,
):
    results = []

    if n_jobs == 1:
        for cluster_id in tqdm(clusters, unit='task', file=sys.stdout):
            results.extend(
                tuning_score_fn(
                    nap.load_file(session_path, lazy_loading=False),
                    session_type,
                    clusters[cluster_id],
                    cluster_id,
                    alpha,
                    n_shuffles,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Prepare tasks
            futures = [
                executor.submit(
                    task,
                    session_path,
                    session_type,
                    tuning_score_fn,
                    clusters[cluster_id],
                    cluster_id,
                    alpha,
                    n_shuffles,
                )
                for cluster_id in clusters
            ]

            # Collect results
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                unit='task',
                file=sys.stdout,
            ):
                results.extend(future.result())
    return pd.DataFrame(results)


def task(
    session_path,
    session_type,
    tuning_score_fn,
    cluster_spikes,
    cluster_id,
    alpha,
    n_shuffles,
):
    return tuning_score_fn(
        nap.load_file(session_path, lazy_loading=False),
        session_type,
        cluster_spikes,
        cluster_id,
        alpha,
        n_shuffles,
    )
