"""Convert the social (helper/victim) miniscope dataset into a clark2025-style tree.

Input layout (as delivered):
    {SRC}/deeplabcut_data/{cohort}/processed_dlc/{helper}-{victim}_results.csv
    {SRC}/deeplabcut_data/{cohort}/aligned_dlc/{helper}-{victim}_aligned.csv
    {SRC}/deeplabcut_data/{cohort}/raw_dlc/{helper}-{victim}_meta.pickle
    {SRC}/calcium_imaging_data/{cohort}/npy_data/{helper}_{helper}-{victim}.npy
    {SRC}/calcium_imaging_data/{cohort}/cnmfe_model/{helper}_{helper}-{victim}.hdf5

Output layout (mirrors clark2025 M{mouse}/D{day}/{session}/sub-..._typ-..._*):
    {DST}/M{helper}/{cohort}/SOCIAL/sub-M{helper}_ses-{cohort}_typ-SOCIAL_beh.nwb
    {DST}/M{helper}/{cohort}/SOCIAL/sub-M{helper}_ses-{cohort}_typ-SOCIAL_src-cnmfe_events.npz
    {DST}/M{helper}/{cohort}/SOCIAL/sub-M{helper}_ses-{cohort}_typ-SOCIAL_src-cnmfe_traces.npz

Task geometry: the helper shuttles along the long axis of the arena. A single
victim sits behind a door at the x=0 end; the far end (x=30cm) is empty. Every
session delivered so far is single-victim, so victim identity is constant within
a session and trips are labelled victim-relative rather than A/B.
"""

import argparse
import json
import pickle
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np
import pandas as pd
import pynapple as nap
import pynwb
from pynwb.epoch import TimeIntervals
from scipy.sparse import csc_matrix

# ── constants ────────────────────────────────────────────────────────────────
# The DLC homography maps the four arena corners onto a 15 x 10 box. That box is
# NOT in cm despite the axis labels on the delivered stats/*.png: nose-to-tailbase
# measures 3.9-4.3 box units and a mouse is ~8cm nose-to-tailbase, which puts one
# box unit at ~2cm and the arena at ~30 x 20cm. Measure the real box and correct
# here if it differs; every distance and speed below scales linearly with it.
CM_PER_UNIT = 2.0
ARENA_UNITS = (15.0, 10.0)
ARENA_CM = (ARENA_UNITS[0] * CM_PER_UNIT, ARENA_UNITS[1] * CM_PER_UNIT)

# Door midpoints, recovered by least squares from the allocentric/egocentric angle
# pair in processed_dlc (residual < 0.03 units in every session).
VICTIM_DOOR_CM = (0.0, 5.0 * CM_PER_UNIT)
FAR_END_CM = (ARENA_CM[0], 5.0 * CM_PER_UNIT)

# Zones used to cut end-to-end traversals.
VICTIM_ZONE_CM = 3.5 * CM_PER_UNIT
FAR_ZONE_CM = 10.0 * CM_PER_UNIT

MOVING_THRESHOLD_CM_S = 5.0  # clark2025 uses 5cm/s for moving/stationary
MIN_EPOCH_S = 0.05
EVENT_THRESHOLD_K = 2.0  # x per-cell median of non-zero deconvolved S

BODYPARTS = [
    'Nose',
    'FrontPawR',
    'FrontPawL',
    'BodyMid',
    'HindPawR',
    'HindPawL',
    'TailBase',
    'TailTip',
]


def wrap(angle):
    """Wrap radians to [-pi, pi), matching the clark2025 H/Hing convention."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def discover_sessions(src):
    """Yield every session that has both behaviour and calcium."""
    for results in sorted(
        src.glob('deeplabcut_data/*/processed_dlc/*_results.csv')
    ):
        cohort = results.parent.parent.name
        session_id = results.name.replace('_results.csv', '')
        helper, victim = session_id.split('-')
        calcium = sorted(
            (src / 'calcium_imaging_data' / cohort / 'npy_data').glob(
                f'*_{session_id}.npy'
            )
        )
        if not calcium:
            print(f'  ! {cohort}/{session_id}: no calcium, skipping')
            continue
        cnmfe = (
            src
            / 'calcium_imaging_data'
            / cohort
            / 'cnmfe_model'
            / f'{calcium[0].stem}.hdf5'
        )
        yield cohort, session_id, helper, victim, calcium[0], cnmfe


def load_behaviour(src, cohort, session_id):
    """Processed + aligned DLC on a common frame index, converted to cm."""
    dlc = src / 'deeplabcut_data' / cohort
    with open(dlc / 'raw_dlc' / f'{session_id}_meta.pickle', 'rb') as handle:
        meta = pickle.load(handle)['data']
    fps = float(meta['fps'])

    proc = pd.read_csv(dlc / 'processed_dlc' / f'{session_id}_results.csv')
    aligned = pd.read_csv(
        dlc / 'aligned_dlc' / f'{session_id}_aligned.csv',
        header=[0, 1],
        index_col=0,
    )
    # processed_dlc re-indexes Frame from 0 but does start at raw frame 0 and is
    # truncated at the end; aligned_dlc keeps every raw frame. Cut to the shorter.
    n = min(len(proc), len(aligned))
    proc, aligned = proc.iloc[:n], aligned.iloc[:n]

    beh = pd.DataFrame(index=np.arange(n) / fps)
    beh['P_x'] = proc['X Center'].to_numpy() * CM_PER_UNIT
    beh['P_y'] = proc['Y Center'].to_numpy() * CM_PER_UNIT
    # The long axis of the shuttle, i.e. the "track" the rasters are built on.
    beh['P'] = beh['P_x']
    # 'Instantaneous Velocity' is a smoothed speed in box units per SECOND
    # (checked against mean_step_distance in stats.txt), so it only needs scaling.
    beh['S'] = proc['Instantaneous Velocity'].to_numpy() * CM_PER_UNIT
    beh['H'] = wrap(np.deg2rad(proc['Allocentric orientation'].to_numpy()))
    # Carried verbatim from the delivered pipeline, for cross-checking.
    beh['running_dlc'] = proc['Running'].to_numpy().astype(float)
    beh['reliable'] = proc['reliable Center'].to_numpy().astype(float)

    for name, (door_x, door_y) in [
        ('victim', VICTIM_DOOR_CM),
        ('far', FAR_END_CM),
    ]:
        beh[f'{name}_dist'] = np.hypot(beh['P_x'] - door_x, beh['P_y'] - door_y)
        beh[f'{name}_angle'] = wrap(
            np.arctan2(door_y - beh['P_y'], door_x - beh['P_x']) - beh['H']
        )

    for bodypart in BODYPARTS:
        beh[f'{bodypart}_x'] = aligned[(bodypart, 'x')].to_numpy() * CM_PER_UNIT
        beh[f'{bodypart}_y'] = aligned[(bodypart, 'y')].to_numpy() * CM_PER_UNIT
        beh[f'{bodypart}_likelihood'] = aligned[
            (bodypart, 'likelihood')
        ].to_numpy()

    beh = beh.ffill().bfill()
    return nap.TsdFrame(beh.astype(float)), fps, meta


def find_trips(position, times):
    """Cut end-to-end traversals into to_victim / from_victim intervals.

    A trip runs from the last sample in the origin zone to the first sample in
    the destination zone, so it spans the traverse and nothing else.
    """
    zone = np.where(
        position < VICTIM_ZONE_CM, 0, np.where(position > FAR_ZONE_CM, 1, -1)
    )
    visits = [(i, z) for i, z in enumerate(zone) if z >= 0]
    trips = {'to_victim': [], 'from_victim': []}
    for (i0, z0), (i1, _) in zip(visits, visits[1:]):
        if zone[i0] == zone[i1]:
            continue
        # z0 == 1 means leaving the far end and arriving at the victim door.
        trips['to_victim' if z0 == 1 else 'from_victim'].append(
            (times[i0], times[i1])
        )
    return trips


def load_calcium(calcium_npy, cnmfe_hdf5, fps, n_behaviour_frames):
    """z-scored dF/F traces and deconvolved event onsets on the behaviour clock."""
    traces = np.load(calcium_npy)
    with h5py.File(cnmfe_hdf5, 'r') as handle:
        S = handle['estimates/S'][:]
        snr = handle['estimates/SNR_comp'][:]
        r_values = handle['estimates/r_values'][:]
        dims = tuple(int(d) for d in handle['dims'][:])
        A = csc_matrix(
            (
                handle['estimates/A/data'][:],
                handle['estimates/A/indices'][:],
                handle['estimates/A/indptr'][:],
            ),
            shape=tuple(int(s) for s in handle['estimates/A/shape'][:]),
        )

    n_calcium = traces.shape[1]
    n = min(n_calcium, S.shape[1], n_behaviour_frames)
    traces, S = traces[:, :n], S[:, :n]
    times = np.arange(n) / fps

    # Footprint centroids, the analogue of cluster x/y in clark2025.
    footprints = np.asarray(A.todense())
    weight = footprints.sum(axis=0)
    weight[weight == 0] = np.nan
    pixel_y, pixel_x = np.unravel_index(np.arange(footprints.shape[0]), dims)
    centroid_x = (footprints * pixel_x[:, None]).sum(axis=0) / weight
    centroid_y = (footprints * pixel_y[:, None]).sum(axis=0) / weight

    # CNMF-E leaves S non-zero on ~20% of frames, which is far too dense to be
    # events. Threshold per cell against its own non-zero median and keep the
    # ONSET of each supra-threshold run, giving ~0.5-1Hz transient rates.
    events = {}
    for cell, trace in enumerate(S):
        non_zero = trace[trace > 0]
        if non_zero.size < 10:
            events[cell] = nap.Ts(t=np.array([]))
            continue
        above = trace > EVENT_THRESHOLD_K * np.median(non_zero)
        onsets = np.flatnonzero(above & ~np.roll(above, 1))
        if above[0]:
            onsets = np.union1d(onsets, [0])
        events[cell] = nap.Ts(t=times[onsets])

    metadata = pd.DataFrame(
        {
            'snr': snr[: len(events)],
            'r_value': r_values[: len(events)],
            'footprint_x': centroid_x[: len(events)],
            'footprint_y': centroid_y[: len(events)],
        }
    )
    return traces, times, events, metadata, n_calcium


def build_nwb(beh, trips, fps, meta, helper, victim, cohort, sync_note):
    nwbfile = pynwb.NWBFile(
        session_description=(
            f'Social helper task, single victim. Helper M{helper}, '
            f'victim M{victim}, cohort {cohort}.'
        ),
        identifier=str(uuid4()),
        session_start_time=datetime.now(UTC),
        session_id=f'M{helper}_{cohort}_SOCIAL',
        lab='Nolan Lab',
        institution='Centre for Discovery Brain Sciences, University of Edinburgh',
        notes=sync_note,
    )

    times = beh.times()
    for column in beh.columns:
        nwbfile.add_acquisition(
            pynwb.TimeSeries(
                name=column,
                data=np.asarray(beh[column].values).ravel(),
                unit='cm'
                if column.endswith(('_x', '_y', '_dist')) or column in ('P',)
                else '',
                timestamps=times,
            )
        )

    speed = beh['S']
    for name, method in [('moving', 'above'), ('stationary', 'below')]:
        epochs = TimeIntervals(
            name=name,
            description=(
                f'Epochs when the helper is {name}, using a '
                f'{MOVING_THRESHOLD_CM_S}cm/s threshold.'
            ),
        )
        support = (
            speed.threshold(MOVING_THRESHOLD_CM_S, method=method)
            .time_support.drop_short_intervals(MIN_EPOCH_S)
        )
        for epoch in support:
            epochs.add_row(
                start_time=float(epoch.start[0]),
                stop_time=float(epoch.end[0]) - 1e-5,
            )
        nwbfile.add_time_intervals(epochs)

    for name, intervals in trips.items():
        direction = (
            'towards the victim door (x=0)'
            if name == 'to_victim'
            else 'away from the victim door, towards the empty far end'
        )
        epochs = TimeIntervals(
            name=name,
            description=(
                f'End-to-end traversals {direction}, cut from the last sample '
                f'in the origin zone to the first sample in the destination zone.'
            ),
        )
        for start, stop in intervals:
            epochs.add_row(start_time=float(start), stop_time=float(stop))
        nwbfile.add_time_intervals(epochs)

    return nwbfile


def convert(src, dst, overwrite):
    summary = []
    for (
        cohort,
        session_id,
        helper,
        victim,
        calcium_npy,
        cnmfe_hdf5,
    ) in discover_sessions(src):
        out_dir = dst / f'M{helper}' / cohort / 'SOCIAL'
        stem = f'sub-M{helper}_ses-{cohort}_typ-SOCIAL'
        nwb_path = out_dir / f'{stem}_beh.nwb'
        if nwb_path.exists() and not overwrite:
            print(f'  = {cohort}/{session_id}: exists, skipping')
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        beh, fps, meta = load_behaviour(src, cohort, session_id)
        traces, times, events, cell_metadata, n_calcium = load_calcium(
            calcium_npy, cnmfe_hdf5, fps, len(beh)
        )
        trips = find_trips(np.asarray(beh['P'].values).ravel(), beh.times())

        # Behaviour and calcium were acquired on separate clocks and no timestamp
        # or sync file was delivered. Oct_2021 frame counts match the video to
        # within 3 frames so they were evidently frame-locked; the 2024 sessions
        # hit a 32767-frame cap while the camera kept running, so a common start
        # is ASSUMED and the streams are truncated to the shorter one.
        drift = abs(n_calcium - meta['nframes'])
        sync_note = json.dumps(
            {
                'sync': 'assumed_common_start_no_timestamps',
                'behaviour_frames': int(len(beh)),
                'calcium_frames': int(n_calcium),
                'video_frames': int(meta['nframes']),
                'frame_count_mismatch': int(drift),
                'frame_locked': bool(drift <= 3),
                'fps': fps,
                'cm_per_box_unit': CM_PER_UNIT,
                'arena_cm': list(ARENA_CM),
                'victim_door_cm': list(VICTIM_DOOR_CM),
                'far_end_cm': list(FAR_END_CM),
                'event_threshold_k': EVENT_THRESHOLD_K,
                'helper': helper,
                'victim': victim,
                'n_victims': 1,
                'victim_side': 'x0',
                'region': 'unknown',
            }
        )

        nwbfile = build_nwb(
            beh, trips, fps, meta, helper, victim, cohort, sync_note
        )
        with pynwb.NWBHDF5IO(nwb_path, 'w') as io:
            io.write(nwbfile)

        cells = nap.TsGroup(events, metadata=cell_metadata)
        cells.save(str(out_dir / f'{stem}_src-cnmfe_events.npz'))
        nap.TsdFrame(t=times, d=traces.T).save(
            str(out_dir / f'{stem}_src-cnmfe_traces.npz')
        )

        summary.append(
            {
                'cohort': cohort,
                'session': session_id,
                'helper': helper,
                'cells': traces.shape[0],
                'minutes': len(beh) / fps / 60,
                'to_victim': len(trips['to_victim']),
                'from_victim': len(trips['from_victim']),
                'frame_locked': drift <= 3,
                'median_rate_hz': float(np.median(cells.rates.values)),
            }
        )
        print(
            f'  + {cohort}/{session_id} -> {nwb_path.relative_to(dst)} '
            f'({traces.shape[0]} cells, {len(trips["to_victim"])} to-victim trips)'
        )
    return pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', type=Path, default=Path.home() / 'Downloads/social_data')
    parser.add_argument('--dst', type=Path, default=Path.home() / 'Downloads/social2025')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    print(f'Converting {args.src} -> {args.dst}')
    summary = convert(args.src, args.dst, args.overwrite)
    if not summary.empty:
        print()
        print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
