from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import pynapple as nap
import spikeinterface.full as si

from spatial_manifolds.data.preprocessing.ephys import load_raw_recording
from spatial_manifolds.data.preprocessing.paths import get_paths

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Fomat Nolan lab spike-sorting for use with this package.'
    )
    parser.add_argument(
        '--storage',
        type=Path,
        default='./data/',
        help='Path to the storage.',
    )
    parser.add_argument(
        '--mouse',
        required=True,
        type=int,
    )
    parser.add_argument(
        '--day',
        required=True,
        type=int,
    )
    parser.add_argument('--sorter', type=str, default='kilosort4')
    args = parser.parse_args()
    paths = get_paths(**args.__dict__)

    # Load location data
    location_data = pd.read_csv(paths['anatomy'])
    location_data = location_data[
        (location_data['mouse'] == args.mouse)
        & (location_data['day'] == args.day)
    ]
    location_data.infer_objects()

    # Load sorting analyser
    sorting_analyser_path = (
        args.storage
        / 'derivatives'
        / f'M{args.mouse}'
        / f'D{args.day}'
        / 'full'
        / args.sorter
        / f'{args.sorter}_sa'
    )
    sorting_analyser = si.load_sorting_analyzer(sorting_analyser_path)

    # Get cluster metrics
    cluster_metrics = sorting_analyser.get_extension(
        'quality_metrics'
    ).get_data()
    cluster_metrics['extremum_channel'] = si.get_template_extremum_channel(
        sorting_analyser
    )
    cluster_metrics.infer_objects()
    if len(location_data) != 0:
        cluster_metrics = cluster_metrics.merge(
            location_data, left_on='extremum_channel', right_on='channel_id'
        ).drop(['channel_id', 'day', 'mouse', 'Unnamed: 0'], axis=1)
    cluster_metrics = cluster_metrics.apply(
        lambda x: x.astype(str) if x.dtype == 'object' else x
    )

    # Session processing
    accumulative_sample_counts = 0
    for session_type in (
        ['OF1', 'VR', 'OF2'] if 'VR' in paths['session_types'] else ['MCVR']
    ):
        session_type_paths = paths['session_types'][session_type]

        # Load raw recording
        recording = load_raw_recording(session_type_paths['raw'])

        # Get clusters
        sorting = sorting_analyser.sorting.frame_slice(
            start_frame=accumulative_sample_counts,
            end_frame=accumulative_sample_counts + recording.get_num_samples(),
        )
        accumulative_sample_counts += recording.get_num_samples()
        clusters = si.spike_vector_to_spike_trains(
            sorting.to_spike_vector(concatenated=False),
            unit_ids=sorting.unit_ids,
        )[0]
        with np.errstate(divide='ignore', invalid='ignore'):
            clusters = nap.TsGroup(
                {
                    int(cluster_id): nap.Ts(clusters[cluster_id] / 30000.0)
                    for cluster_id in clusters
                },
                metadata=cluster_metrics,
            )

        # Save clusters
        clusters.save(session_type_paths['processed']['clusters'])
