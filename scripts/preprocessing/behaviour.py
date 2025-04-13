import shutil
from argparse import ArgumentParser
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pynapple as nap
import pynwb
from pynwb.ecephys import LFP, ElectricalSeries
from pynwb.epoch import TimeIntervals

from spatial_manifolds.data.preprocessing.behaviour import (
    load_blender,
    load_dlc,
)
from spatial_manifolds.data.preprocessing.ephys import (
    compute_lfp,
    load_raw_recording,
    load_sync_channel,
)
from spatial_manifolds.data.preprocessing.paths import get_paths
from spatial_manifolds.data.preprocessing.sync import sync_pulses

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Format Nolan lab behaviour data for use with this package.'
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
    args = parser.parse_args()
    paths = get_paths(**args.__dict__)

    # Load location data
    location_data = pd.read_csv(paths['anatomy'])
    location_data = location_data[
        (location_data['mouse'] == args.mouse)
        & (location_data['day'] == args.day)
    ]
    location_data.infer_objects()

    for session_type in paths['session_types']:
        session_type_paths = paths['session_types'][session_type]
        # Load raw recording
        recording = load_raw_recording(session_type_paths['raw'])
        sync_channel = load_sync_channel(
            session_type_paths['raw'], session_type_paths['tmp']
        )

        # Get behaviour data
        behaviour = {
            behaviour_data_type: sync_pulses(
                *sync_channel,
                globals()[f'load_{behaviour_data_type}'](
                    session_type_paths, session_type
                ),
                session_type_paths,
                tag=behaviour_data_type,
            )
            for behaviour_data_type in session_type_paths[
                'behaviour_data_types'
            ]
        }

        # Get LFP
        lfp = compute_lfp(recording, session_type_paths['tmp'])

        # Make NWB file
        nwbfile = pynwb.NWBFile(
            session_description=f'MEC {session_type} session.',
            identifier=str(uuid4()),
            session_start_time=datetime.now(UTC),
            session_id=f'M{args.mouse}D{args.day}_{session_type}',
            experimenter=[
                'Clark, Harry',
            ],
            lab='Nolan Lab',
            institution='Centre for Discovery Brain Sciences, University of Edinburgh',
        )

        # Add device
        metadata = recording._annotations['probes_info'][0]
        device = nwbfile.create_device(
            name='Neuropixels 2.0',
            description='A 384-channel array with 4 shanks and 96 channels per shank',
            manufacturer=metadata['manufacturer'],
            model_number=metadata['part_number'],
            model_name=metadata['model_name'],
            serial_number=metadata['serial_number'],
        )
        nwbfile.add_electrode_column(
            name='channel', description='label of channel'
        )
        nwbfile.add_electrode_column(
            name='contact_id', description='label of contact id'
        )
        probe = recording.get_probe()
        shanks = {
            int(shank): nwbfile.create_electrode_group(
                name=f'shank {shank}',
                description=f'Shank {shank}',
                device=device,
                location='MEC',
            )
            for shank in np.unique(probe.shank_ids)
        }
        for channel, shank, contact_id in zip(
            recording.get_channel_ids(),
            probe.shank_ids,
            probe.contact_ids,
            strict=False,
        ):
            nwbfile.add_electrode(
                group=shanks[int(shank)],
                channel=channel,
                contact_id=contact_id,
                location=location_data[
                    location_data['device_contact_id'] == contact_id
                ]['brain_region'].values[0]
                if len(location_data) != 0
                else 'unknown',
            )

        # Add LFP
        lfp = LFP(
            electrical_series=ElectricalSeries(
                name='LFP',
                description='Local field potential',
                data=lfp,
                filtering='Bandpass filtered 1-300 Hz',
                electrodes=nwbfile.create_electrode_table_region(
                    region=list(range(len(recording.get_channel_ids()))),
                    description='all channels',
                ),
                starting_time=0.0,
                rate=30.0,
            )
        )
        lfp_module = nwbfile.create_processing_module(
            name='LFP', description='Local field potential'
        )
        lfp_module.add(lfp)

        # Add behaviour
        for behaviour_data in behaviour.values():
            timestamps = behaviour_data.times()
            for column in behaviour_data.columns:
                nwbfile.add_acquisition(
                    pynwb.TimeSeries(
                        name=column,
                        data=behaviour_data[column].values,
                        unit='',
                        timestamps=timestamps,
                    )
                )

        # Add moving epochs
        moving = TimeIntervals(
            name='moving',
            description='Epochs when the animal is moving, using a 5cm/s threshold.',
        )
        for trial in (
            (
                behaviour['dlc']['S']
                if 'OF' in session_type
                else behaviour['blender']['S']
            )
            .threshold(5, method='above')
            .time_support.drop_short_intervals(0.05)
        ):
            moving.add_row(
                start_time=trial.start[0],
                # to separate them from following trial start
                stop_time=trial.end[0] - 1e-5,
            )
        nwbfile.add_time_intervals(moving)

        # Add stationary epochs
        stationary = TimeIntervals(
            name='stationary',
            description='Epochs when the animal is not moving, using a 5cm/s threshold.',
        )
        for trial in (
            (
                behaviour['dlc']['S']
                if 'OF' in session_type
                else behaviour['blender']['S']
            )
            .threshold(5, method='below')
            .time_support.drop_short_intervals(0.05)
        ):
            stationary.add_row(
                start_time=trial.start[0],
                # to separate them from following trial start
                stop_time=trial.end[0] - 1e-5,
            )
        nwbfile.add_time_intervals(stationary)

        if 'VR' in session_type:
            # Add trial numbers
            nwbfile.add_trial_column(
                name='trial_number', description='trial number'
            )
            nwbfile.add_trial_column(
                name='trial_type', description='trial type'
            )
            for trial_number in behaviour['blender']['trial_number'].unique():
                for trial in (
                    (
                        nap.Tsd(
                            d=(
                                behaviour['blender']['trial_number']
                                == trial_number
                            ).values,
                            t=timestamps,
                        )
                    )
                    .threshold(0.5, method='above')
                    .time_support
                ):
                    nwbfile.add_trial(
                        start_time=trial.start[0],
                        stop_time=trial.end[0]
                        - 1e-5,  # to separate them from following trial start
                        trial_number=behaviour['blender']['trial_number']
                        .restrict(trial)
                        .as_series()
                        .mode()
                        .values[0],
                        trial_type=behaviour['blender']['trial_type']
                        .restrict(trial)
                        .as_series()
                        .mode()
                        .values[0],
                    )

        # Write
        with pynwb.NWBHDF5IO(
            session_type_paths['processed']['nwb'], 'w'
        ) as io:
            io.write(nwbfile)
        print(f'NWB file saved to {session_type_paths["processed"]["nwb"]}')

    # Cleanup
    shutil.rmtree(paths['tmp'], ignore_errors=True)
