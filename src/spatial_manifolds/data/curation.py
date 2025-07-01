import warnings
import pynapple as nap

def curate_clusters(clusters) -> nap.TsGroup:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Some epochs have no duration",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="divide by zero encountered in scalar divide",
        )
    return clusters[
        (clusters["isi_violations_ratio"] < 0.5)
        # & (clusters['amplitude_cutoff'] < 0.1)
        & (clusters["presence_ratio"] > 0.9)
        & (clusters["firing_rate"] > 0.5)
        & (clusters["snr"] > 1)
    ]

