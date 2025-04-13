import spikeinterface.full as si


def load_raw_recording(raw_recording_path):
    if (raw_recording_path / 'recording.zarr').exists():
        recording = si.read_zarr(raw_recording_path / 'recording.zarr')
    else:
        recording = si.read_openephys(
            raw_recording_path, load_sync_channel=False)
    return recording


def compute_lfp(recording, tmp_path):
    recording_lfp = si.bandpass_filter(recording, freq_min=1, freq_max=300)
    recording_lfp = si.resample(recording_lfp, 60)
    # save&read to avoid memory issues (this is a spikeinterface issue that will be fixed)
    recording_lfp.save_to_folder(folder=tmp_path / 'lfp')
    recording_lfp = si.read_binary_folder(tmp_path / 'lfp')
    return recording_lfp.get_traces()


def load_sync_channel(raw_recording_path, tmp_path):
    if (raw_recording_path / 'recording.zarr').exists():
        sync_data = si.read_zarr(raw_recording_path / 'channel_sync.zarr')
        sync_pulse = sync_data.get_traces(channel_ids=['CH_SYNC'])[:, 0]
    else:
        sync_data = si.read_openephys(raw_recording_path, load_sync_channel=True).select_channels(
            ['CH_SYNC']
        )
        sync_data.save_to_folder(folder=tmp_path / 'channel_sync')
        sync_data = si.read_binary_folder(tmp_path / 'channel_sync')
        sync_pulse = sync_data.get_traces()[:, 0]
    sync_times = sync_data.get_times()
    sync_times -= sync_times[0]
    return sync_pulse, sync_times
