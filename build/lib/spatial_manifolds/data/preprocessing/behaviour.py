import numpy as np
import pandas as pd
import pynapple as nap


def load_blender(paths, session_type):
    blender_data_raw = pd.read_csv(
        paths['behaviour_data_types']['blender'],
        skiprows=4,
        sep=';',
        names=[
            'Time',
            'Position-X',
            'Speed',
            'Speed/gain_mod',
            'Reward_received',
            'Reward_failed',
            'Lick_detected',
            'Tone_played',
            'Position-Y',
            'Tot trial',
            'gain_mod',
            'rz_start',
            'rz_end',
            'sync_pulse',
        ],
    )

    # Preprocess
    blender_data_raw.dropna()
    blender_data_raw['trial_type'] = np.array(
        blender_data_raw['Position-Y'] / 10, dtype=np.int64
    )
    blender_data_raw['Position-X'] = np.array(
        blender_data_raw['Position-X'] * 10
    ).clip(0.0, 200.0 if session_type == 'VR' else 230.0)

    # Rename columns
    blender_data = pd.DataFrame(
        index=blender_data_raw['Time'],
        data=blender_data_raw[
            ['Position-X', 'Speed', 'trial_type', 'Tot trial', 'sync_pulse']
        ].to_numpy(),
        columns=['P', 'S', 'trial_type', 'trial_number', 'sync_pulse'],
    )
    blender_data.infer_objects()
    return nap.TsdFrame(blender_data)


def load_dlc(paths, session_type):
    return globals()[f'load_{session_type[:2]}_dlc'](paths)


def load_VR_dlc(paths):
    # Load DeepLabCut data
    licks = pd.read_csv(
        paths['behaviour_data_types']['dlc']['licks'], usecols=['lick']
    )
    pupil = pd.read_csv(
        paths['behaviour_data_types']['dlc']['pupil'], header=[1, 2]
    )
    pupil.columns = ['_'.join(col).strip() for col in pupil.columns.values]
    pupil.drop(columns=['bodyparts_coords'], inplace=True)
    # Load bonsai data
    bonsai = pd.read_csv(
        paths['bonsai'],
        header=None,
        usecols=[1, 2],
        sep=',',
    )
    bonsai[1] = pd.to_datetime(bonsai[1])
    bonsai.columns = ['time', 'sync_pulse']
    if len(licks) != len(bonsai):
        print(
            f'Licks ({len(licks)}) and Bonsai ({len(bonsai)}) have different lengths'
        )
    if len(pupil) != len(bonsai):
        print(
            f'Pupil ({len(pupil)}) and Bonsai ({len(bonsai)}) have different lengths'
        )
    behaviour = pd.concat([licks, pupil, bonsai], axis=1).dropna()
    behaviour['time'] = pd.to_datetime(behaviour['time'])
    behaviour['time'] = behaviour['time'] - behaviour['time'][0]
    behaviour.set_index('time', inplace=True)
    behaviour.index = behaviour.index.total_seconds()
    return nap.TsdFrame(behaviour)


def load_OF_dlc(paths):
    # Load DeepLabCut data
    dlc = pd.read_csv(paths['behaviour_data_types']['dlc'], header=[1, 2])
    dlc.columns = ['_'.join(col).strip() for col in dlc.columns.values]
    # Load Bonsai data
    bonsai = pd.read_csv(
        paths['bonsai'],
        header=None,
        usecols=[0, 5],
        sep=' ',
    )
    bonsai.columns = ['time', 'sync_pulse']
    if len(dlc) != len(bonsai):
        print(
            f'DeepLabCut ({len(dlc)}) and Bonsai ({len(bonsai)}) have different lengths'
        )
    behaviour = pd.concat([dlc, bonsai], axis=1).dropna()
    behaviour['time'] = pd.to_datetime(behaviour['time'])
    behaviour['time'] = behaviour['time'] - behaviour['time'][0]
    behaviour.set_index('time', inplace=True)
    behaviour.index = behaviour.index.total_seconds()

    # Convert DeepLabCut coordinates to cm
    box_start_x, box_start_y, box_width, box_height = [160, 15, 415, 415]
    behaviour['P_x'] = (
        (behaviour['middle_x'] - box_start_x) / box_width
    ).clip(0.0, 1.0) * 100
    behaviour['P_y'] = (
        (behaviour['middle_y'] - box_start_y) / box_height
    ).clip(0.0, 1.0) * 100

    # Compute head directiong
    behaviour['H'] = np.arctan2(
        behaviour['head_y'] - behaviour['middle_y'],
        behaviour['head_x'] - behaviour['middle_x'],
    )

    # Compute speed
    speed = np.sqrt(
        np.diff(behaviour['P_x']) ** 2 + np.diff(behaviour['P_y']) ** 2
    ) / np.diff(behaviour.index)
    behaviour = behaviour.iloc[1:]
    behaviour['S'] = speed

    return nap.TsdFrame(behaviour)
