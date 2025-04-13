import shutil

from spatial_manifolds.util import tree


def get_paths(storage, mouse, day, sorter=None, dlc_model=None):
    paths = tree()
    paths['anatomy'] = (
        storage
        / 'derivatives'
        / 'labels'
        / 'anatomy'
        / 'mouse_day_channel_ids_brain_location.csv'
    )
    paths['tmp'] = storage / 'tmp' / f'M{mouse}D{day}'

    # Session types
    sessions = list((storage / 'raw').glob(f'*/M{mouse}_D{day}*'))
    assert len(sessions) > 0, (
        f'Could not find any sessions for M{mouse}D{day}.'
    )
    for raw_recording_path in sessions:
        session_type = raw_recording_path.name.split('_')[-1]
        if 'VR' in session_type:
            session_type = session_type.replace('1', '')
        session_type_paths = paths['session_types'][session_type]
        session_type_paths['raw'] = raw_recording_path

        # Blender/Bonsai/DeepLabCut data
        if 'VR' in session_type:
            # DeepLabCut
            derivatives_path = (
                storage / 'derivatives' / f'M{mouse}' / f'D{day}' / 'vr'
            )
            lick_path = derivatives_path / 'licks' / 'lick_mask.csv'
            pupil_path = (
                derivatives_path
                / 'pupil_dilation'
                / f'M{mouse}_D{day}_side_captureDLC_Resnet50_vrJan24shuffle1_snapshot_200.csv'
            )
            if lick_path.exists() and pupil_path.exists():
                session_type_paths['behaviour_data_types']['dlc'] = {
                    'licks': lick_path,
                    'pupil': pupil_path,
                }
            # Blender
            blender_path = list(session_type_paths['raw'].glob('*blender.csv'))
            assert len(blender_path) >= 1, (
                f'Could not find Blender data: {session_type_paths["raw"] / "*blender.csv"}'
            )
            session_type_paths['behaviour_data_types']['blender'] = (
                blender_path[0]
            )
            # Bonsai
            bonsai_path = list(
                session_type_paths['raw'].glob('*side_capture.csv')
            )
            assert len(bonsai_path) >= 1, (
                f'Could not find Bonsai data: {session_type_paths["raw"] / "*.csv"}'
            )
            session_type_paths['bonsai'] = bonsai_path[0]
        else:
            # DeepLabCut
            dlc_path = (
                storage
                / 'derivatives'
                / f'M{mouse}'
                / f'D{day}'
                / session_type.lower()
                / 'dlc'
            )
            dlc_file = list(
                dlc_path.glob(
                    f'*DLC_Resnet50_of_cohort12Oct30shuffle1_snapshot_200.csv'
                )
            )
            assert len(dlc_file) == 1, (
                f'Could not find deeplabcut data: {dlc_path}'
            )
            session_type_paths['behaviour_data_types']['dlc'] = dlc_file[0]
            # Bonsai
            bonsai_path = list(session_type_paths['raw'].glob('*.csv'))
            assert len(bonsai_path) >= 1, (
                f'Could not find Bonsai data: {session_type_paths["raw"] / "*.csv"}'
            )
            if len(bonsai_path) > 1:
                for path in bonsai_path:
                    if '.' not in str(path.with_suffix('')):
                        session_type_paths['bonsai'] = path
                        break
            else:
                session_type_paths['bonsai'] = bonsai_path[0]

        if sorter is not None:
            # Sorting analyzer
            sorting_analyser_path = (
                storage
                / 'derivatives'
                / f'M{mouse}'
                / f'D{day}'
                / 'full'
                / sorter
                / f'{sorter}_sa'
            )
            assert sorting_analyser_path.exists(), (
                f"Sorting analyser doesn't exist: {sorting_analyser_path}"
            )
            session_type_paths['sorting_analyser'] = sorting_analyser_path

        # Temporary data
        session_type_paths['tmp'] = paths['tmp'] / session_type
        if session_type_paths['tmp'].exists():
            shutil.rmtree(session_type_paths['tmp'])

        # Processed data
        processed_path = (
            storage / 'sessions' / f'M{mouse}' / f'D{day}' / session_type
        )
        session_type_paths['processed']['nwb'] = (
            processed_path / f'M{mouse}D{day}{session_type}.nwb'
        )
        session_type_paths['processed']['clusters'] = (
            processed_path / f'{sorter}.npz'
        )

        # Syncing plots
        session_type_paths['sync'] = processed_path / 'sync'
        session_type_paths['sync'].mkdir(parents=True, exist_ok=True)

    return paths
