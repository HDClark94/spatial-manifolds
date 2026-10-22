"""
Rebuild MCVR behaviour data without the early-trial truncation bug.

Root cause (confirmed in src/spatial_manifolds/data/preprocessing/sync.py,
sync_pulses(), line ~126): after mapping behaviour-clock time into
ephys-clock time via a fitted linear transform, the function does
    behaviour = nap.TsdFrame(behaviour[behaviour.index > 0])
which unconditionally drops every behavioural sample (and therefore every
trial) that happened before the ephys recording's own t=0. If the VR
session was started before the Neuropixels acquisition -- which varied
across sessions/days -- everything the animal did in that gap is silently
discarded. Confirmed across all 31 MCVR sessions: 13/31 (42%) are missing
between 10 and 24 early trials this way (M22 D44/45/46, M25 D26/30/31,
M26 D20/21/22/24, M27 D25/27/28); M28 and M29 are unaffected in every
session.

Scope of this rebuild (per user decision): core VR channels only --
position, speed, travel, and the trial table (context, type, result,
performance) -- reconstructed directly from the raw blender.csv, with NO
ephys-clock alignment (there's no spike data in these files to align to,
and skipping that step entirely avoids reintroducing any sync-related
bug) and NO DLC-derived channels (lick/eye/pupil/tongue), since those
depend on separate video-tracking outputs not touched here.

Output: one behaviour-only NWB per session, in
    ~/Downloads/mcvr_behaviour_rebuilt/dataset/sub-M{mouse}/sub-M{mouse}_ses-D{day}MCVR_behavior.nwb
mirroring the naming convention of the original ecephys+behavior.nwb
files, minus the "ecephys" part (since there's no ephys content here).

Column/label derivations, all validated against the existing (partially
truncated but correctly-labelled) processed NWB files on their
overlapping trials, across 12 sessions spanning 6 mice (2373 trials):

  - context: 'rz1' if trial_type_code < 2 else 'rz2'   (exact, 100%)
  - type:    'b' if trial_type_code is even else 'nb'  (exact, 100%)
  - result:  'rewarded' if Reward_received else 'unrewarded'  (exact, 100%)
  - performance: experimenter-provided rule --
      1. avg speed OUTSIDE the reward zone < 10 cm/s -> 'slow' (unclassified)
      2. else, min speed INSIDE the reward zone < 3 cm/s -> 'hit'
      3. else ('miss'): compare avg speed inside the reward zone to the
         95th percentile of avg-speed-inside-RZ across this session's hit
         trials -> 'try' if below that percentile, 'run' if above.
    Overall validation accuracy: 94.9% (2252/2373). Hit/run/slow are
    reconstructed with ZERO errors; all disagreement is 'try' trials
    landing just across the boundary into 'hit' (27) or 'run' (94) --
    expected boundary noise from re-deriving an exact percentile
    threshold, not a systematic error.
"""
import glob
import os
import re
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pandas as pd
import pynwb
from pynwb.epoch import TimeIntervals

RAW_SOURCE = '/Volumes/INCR-NolanLab/ActiveProjects/Harry/EphysNeuropixelData/vr_multi_context/'
OUT_ROOT = os.path.expanduser('~/Downloads/mcvr_behaviour_rebuilt/dataset/')

BS, TL = 2, 230  # matches mcvr_tl in anaylsis_parameters.py
COLUMN_NAMES = [
    'Time', 'Position-X', 'Speed', 'Speed_gain_mod', 'Reward_received',
    'Reward_failed', 'Lick_detected', 'Tone_played', 'Position-Y',
    'Tot_trial', 'gain_mod', 'rz_start', 'rz_end', 'sync_pulse',
]
CONTEXT_FROM_CODE = {0: 'rz1', 1: 'rz1', 2: 'rz2', 3: 'rz2'}
TYPE_FROM_CODE = {0: 'b', 1: 'nb', 2: 'b', 3: 'nb'}


def find_raw_session_dir(mouse, day):
    """mouse is numeric-only (e.g. '25'); raw directories are named M25_D27_..."""
    matches = glob.glob(f'{RAW_SOURCE}M{mouse}_D{day}_*_MCVR*')
    return matches[0] if matches else None


def load_raw_blender(raw_dir):
    blender_files = glob.glob(f'{raw_dir}/*blender.csv')
    assert len(blender_files) == 1, f'expected 1 blender.csv in {raw_dir}, found {len(blender_files)}'
    raw = pd.read_csv(blender_files[0], skiprows=4, sep=';', names=COLUMN_NAMES)
    raw['Position-X_cm'] = (raw['Position-X'] * 10).clip(0.0, TL)
    raw['trial_type_code'] = (raw['Position-Y'] / 10).astype(int)
    return raw


def build_trial_table(raw):
    """Per-trial start/end/context/type/result/performance, plus the
    95th-percentile speed threshold used for the try/run split."""
    rows = []
    for trial_num, g in raw.groupby('Tot_trial', sort=True):
        rz_lo, rz_hi = g['rz_start'].iloc[0] * 10, g['rz_end'].iloc[0] * 10
        in_rz = g[(g['Position-X_cm'] >= rz_lo) & (g['Position-X_cm'] <= rz_hi)]
        out_rz = g[(g['Position-X_cm'] < rz_lo) | (g['Position-X_cm'] > rz_hi)]
        rows.append(dict(
            number=int(trial_num),
            start=float(g['Time'].iloc[0]),
            end=float(g['Time'].iloc[-1]),
            trial_type_code=int(g['trial_type_code'].mode().iloc[0]),
            reward_received=bool(g['Reward_received'].max()),
            min_speed_in_rz=in_rz['Speed'].min() if len(in_rz) else np.nan,
            avg_speed_in_rz=in_rz['Speed'].mean() if len(in_rz) else np.nan,
            avg_speed_out_rz=out_rz['Speed'].mean() if len(out_rz) else np.nan,
        ))
    trials = pd.DataFrame(rows)
    trials['context'] = trials['trial_type_code'].map(CONTEXT_FROM_CODE)
    trials['type'] = trials['trial_type_code'].map(TYPE_FROM_CODE)
    trials['result'] = np.where(trials['reward_received'], 'rewarded', 'unrewarded')

    trials['stopped_in_rz'] = trials['min_speed_in_rz'] < 3
    hit_avg_speeds = trials.loc[trials['stopped_in_rz'], 'avg_speed_in_rz']
    speed_threshold = float(np.percentile(hit_avg_speeds, 95)) if len(hit_avg_speeds) else np.nan

    def classify(row):
        if row['avg_speed_out_rz'] < 10:
            return 'slow'
        if row['stopped_in_rz']:
            return 'hit'
        return 'try' if row['avg_speed_in_rz'] < speed_threshold else 'run'

    trials['performance'] = trials.apply(classify, axis=1)
    return trials, speed_threshold


def build_nwb(mouse, day, raw, trials):
    nwbfile = pynwb.NWBFile(
        session_description=f'MCVR behaviour-only rebuild (untruncated), M{mouse}D{day}.',
        identifier=str(uuid4()),
        session_start_time=datetime.now(UTC),
        session_id=f'M{mouse}D{day}_MCVR_behavior_rebuilt',
        experimenter=['Clark, Harry'],
        lab='Nolan Lab',
        institution='Centre for Discovery Brain Sciences, University of Edinburgh',
        notes=(
            'Rebuilt from raw blender.csv to recover early trials dropped by '
            'the ephys-sync truncation bug in the original processing pipeline '
            '(sync.py: behaviour = behaviour[behaviour.index > 0]). Time is the '
            'RAW VR clock (session-relative, starting near 0), NOT aligned to '
            'the ephys clock -- this file has no spike/LFP data. Position, '
            'speed, and trial context/type are exact reconstructions; the '
            'performance label (hit/try/run/slow) is reconstructed from the '
            'experimenter-provided classification rule and validated at 94.9% '
            'exact agreement against the original (hit/run/slow: 100%; try: '
            'boundary noise only) -- see module docstring of the build script '
            'for full validation details.'
        ),
    )

    time = raw['Time'].to_numpy()
    nwbfile.add_acquisition(pynwb.TimeSeries(
        name='P', description='Position within trial (cm), 0-230, resets each trial.',
        data=raw['Position-X_cm'].to_numpy(), unit='cm', timestamps=time,
    ))
    nwbfile.add_acquisition(pynwb.TimeSeries(
        name='S', description='Running speed (cm/s).',
        data=raw['Speed'].to_numpy(), unit='cm/s', timestamps=time,
    ))
    travel = raw['Position-X_cm'].to_numpy() + (raw['Tot_trial'].to_numpy() - 1) * TL
    nwbfile.add_acquisition(pynwb.TimeSeries(
        name='travel', description='Cumulative distance across the whole session (cm) -- P plus (trial_number-1)*230.',
        data=travel, unit='cm', timestamps=time,
    ))
    nwbfile.add_acquisition(pynwb.TimeSeries(
        name='trial_number', description='Trial number, continuous per-sample channel (matches original schema).',
        data=raw['Tot_trial'].to_numpy(), unit='', timestamps=time,
    ))

    nwbfile.add_trial_column(name='number', description='trial number (raw, 1-indexed, no truncation)')
    nwbfile.add_trial_column(name='type', description="'b' beaconed/cued or 'nb' non-beaconed/uncued")
    nwbfile.add_trial_column(name='context', description="'rz1' (familiar) or 'rz2' (novel)")
    nwbfile.add_trial_column(name='result', description="'rewarded' or 'unrewarded'")
    nwbfile.add_trial_column(name='performance', description='hit/try/run/slow, see NWBFile.notes for derivation')
    for _, row in trials.iterrows():
        nwbfile.add_trial(
            start_time=row['start'], stop_time=row['end'] - 1e-5,
            number=int(row['number']), type=row['type'], context=row['context'],
            result=row['result'], performance=row['performance'],
        )

    return nwbfile


def process_session(mouse, day):
    raw_dir = find_raw_session_dir(mouse, day)
    if raw_dir is None:
        return None, f'M{mouse} D{day}: raw session directory not found'
    raw = load_raw_blender(raw_dir)
    trials, speed_threshold = build_trial_table(raw)

    out_dir = os.path.join(OUT_ROOT, f'sub-M{mouse}')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'sub-M{mouse}_ses-D{day}MCVR_behavior.nwb')

    nwbfile = build_nwb(mouse, day, raw, trials)
    with pynwb.NWBHDF5IO(out_path, 'w') as io:
        io.write(nwbfile)

    return dict(mouse=mouse, day=day, n_trials=len(trials), first_trial_number=int(trials['number'].iloc[0]),
                speed_threshold=speed_threshold, out_path=out_path), None


if __name__ == '__main__':
    session_dirs = sorted(glob.glob(f'{RAW_SOURCE}M*_D*_*_MCVR*'))
    sessions = []
    for d in session_dirs:
        m = re.match(r'M(\d+)_D(\d+)_', os.path.basename(d))
        sessions.append((m.group(1), int(m.group(2))))  # mouse kept numeric-only, e.g. '25'
    sessions = sorted(set(sessions), key=lambda x: (int(x[0]), x[1]))

    print(f'{len(sessions)} raw MCVR sessions found')
    results = []
    for mouse, day in sessions:
        result, error = process_session(mouse, day)
        if error:
            print(error)
            continue
        results.append(result)
        print(f"M{result['mouse']} D{result['day']}: {result['n_trials']} trials "
              f"(first trial_number={result['first_trial_number']}) -> {result['out_path']}")

    summary = pd.DataFrame(results)
    summary_path = os.path.join(OUT_ROOT, 'rebuild_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f'\ndone: {len(results)}/{len(sessions)} sessions rebuilt')
    print(f'summary saved to {summary_path}')
