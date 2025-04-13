import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from scipy.spatial import cKDTree


def match_sync_pulses(ts1, ts2):
    """
    Match sync pulse timestamps pairs
    This function results in unshifted but paired timestamps
    which is neccessary because sometimes sync pulses are not paired

    """  # Initial alignment based on cross-correlation
    cross = nap.compute_crosscorrelogram(
        nap.TsGroup([ts2, ts1]),
        binsize=0.01,
        windowsize=150,
    )
    lag = cross.idxmax().values[0]
    ts2_ = nap.Ts(ts2.times() + lag)

    # Format
    ts1_ = np.array(ts1.times())[:, None]
    ts2_ = np.array(ts2_.times())[:, None]

    # Create a KD-tree for fast nearest-neighbor search
    tree = cKDTree(ts2_)

    # Find the closest timestamps in ts2 for each timestamp in ts1
    distances, indices = tree.query(ts1_)

    # Filter out pairs where the distance is too large (optional, to avoid bad matches)
    # You can adjust this threshold as needed
    threshold = np.mean(distances) * 2
    index_ts1 = distances < threshold
    index_ts2 = indices[distances < threshold]

    return ts1[index_ts1], ts2[index_ts2]


def sync_pulses(ephys_sync_pulse, ephys_times, behaviour, paths, tag):
    predefined_lag_path = paths['raw'] / 'lag.npy'
    if predefined_lag_path.exists():
        a, b = 1, -np.load(predefined_lag_path)
    else:
        ephys_sync_pulse = nap.Tsd(
            d=ephys_sync_pulse, t=ephys_times - ephys_times[0]
        )

        # Plot sync pulses
        fig, ax = plt.subplots(figsize=(20, 5))
        plt.plot(behaviour['sync_pulse'], label='behaviour')
        plt.plot(ephys_times[::100], ephys_sync_pulse[::100], label='ephys')

        # Extract timestamps
        ephys_sync_times = ephys_sync_pulse.threshold(
            0.5, method='above'
        ).time_support.get_intervals_center()

        behaviour_sync_times = (
            behaviour['sync_pulse']
            .threshold(
                np.median(behaviour['sync_pulse'].values)
                + 5 * np.std(behaviour['sync_pulse'].values),
                method='above',
            )
            .time_support.get_intervals_center()
        )
        # fix for if the camera moved
        if len(behaviour_sync_times) < 30:
            behaviour_sync_pulse_ = np.diff(behaviour['sync_pulse'], prepend=0)
            behaviour_sync_pulse_[behaviour_sync_pulse_ < 0] = 0
            plt.plot(behaviour_sync_pulse_, label='behaviour (diff fix)')
            behaviour_sync_times = behaviour_sync_pulse_.threshold(
                np.median(behaviour_sync_pulse_.values)
                + 5 * np.std(behaviour_sync_pulse_.values),
                method='above',
            ).time_support.get_intervals_center()
        plt.legend(loc='upper right')
        plt.xlabel('time (s)')
        plt.tight_layout()
        plt.savefig(paths['sync'] / f'{tag}_sync_pulses.png', dpi=300)
        plt.close()

        # Match sync pulses for when they are not perfectly paired
        ephys_sync_times, behaviour_sync_times = match_sync_pulses(
            ephys_sync_times, behaviour_sync_times
        )

        # Compute linear transformation
        A = np.vstack(
            [
                behaviour_sync_times.times(),
                np.ones_like(behaviour_sync_times.times()),
            ]
        ).T
        a, b = np.linalg.lstsq(A, ephys_sync_times.times(), rcond=None)[0]

        # Plot before sync
        aligned = nap.Ts(behaviour_sync_times.times() * a + b)
        x = np.arange(len(ephys_sync_times))
        plt.plot(x, behaviour_sync_times.times() - ephys_sync_times.times())
        plt.scatter(x, behaviour_sync_times.times() - ephys_sync_times.times())
        plt.xlabel('pulse count')
        plt.ylabel('behaviour pulse - ephys pulse (s)')
        plt.tight_layout()
        plt.savefig(paths['sync'] / f'{tag}_before_sync.png')
        plt.close()

        # Plot after sync
        plt.scatter(x, aligned.times() - ephys_sync_times.times())
        plt.plot(x, aligned.times() - ephys_sync_times.times())
        plt.xlabel('pulse count')
        plt.ylabel('behaviour pulse - ephys pulse (s)')
        plt.title(f'slope = {a:.5f}, intercept = {b:.5f}')
        plt.tight_layout()
        plt.savefig(paths['sync'] / f'{tag}_after_sync.png')
        plt.close()

    # Correct behaviour data
    unsynced_time = behaviour.times()
    behaviour = behaviour.as_dataframe()
    behaviour['synced_time'] = unsynced_time * a + b
    behaviour = behaviour.reset_index(drop=True).set_index('synced_time')
    behaviour = behaviour.drop(columns=['sync_pulse'])
    behaviour = nap.TsdFrame(behaviour[behaviour.index > 0])

    return behaviour
