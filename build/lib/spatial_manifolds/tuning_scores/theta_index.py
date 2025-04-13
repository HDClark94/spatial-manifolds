import numpy as np
import pandas as pd
import pynapple as nap


def compute_theta_index(session, session_type, cluster_spikes):
    autocorr = nap.compute_autocorrelogram(
        cluster_spikes, windowsize=0.5, binsize=0.005, ep=session['moving']
    )
    trough = np.mean(
        autocorr[(autocorr.index > 0.05) & (autocorr.index <= 0.07)], axis=0
    )
    peak = np.mean(
        autocorr[(autocorr.index > 0.1) & (autocorr.index <= 0.14)], axis=0
    )
    with np.errstate(invalid='ignore', divide='ignore'):
        return pd.DataFrame(
            [
                {
                    'cluster_id': cluster_id,
                    'theta_index': (trough[cluster_id] - peak[cluster_id])
                    / (trough[cluster_id] + peak[cluster_id]),
                    'sig': (trough[cluster_id] - peak[cluster_id])
                    / (trough[cluster_id] + peak[cluster_id])
                    > 0.07,
                }
                for cluster_id in cluster_spikes
            ]
        )
