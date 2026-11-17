"""The paper's unit curation, in one place.

A unit is kept unless data/cell_classifications_v2.csv marks it 'EXCL', which
is set by bombcell (noise or non_soma), the ISI rule and the dorsal-island rule
(apply_isi_exclusion.py), exactly as Methods describe. Nothing else is applied.

This replaces spatial_manifolds.data.curation.curate_clusters, whose
SpikeInterface quality-metric thresholds (ISI violation < 0.5, presence ratio >
0.9, rate > 0.5 Hz, SNR > 1) are a different curation from the paper's and
silently removed a third of the curated units from the analyses that used it.

    from paper_units import curate
    clusters = curate(nap.load_file(path), mouse, day)
"""
import pandas as pd

CSV = '/Users/harryclark/Documents/spatial-manifolds/data/cell_classifications_v2.csv'
_KEPT = None


def kept():
    """(mouse, day, cluster_id) of every unit the paper keeps."""
    global _KEPT
    if _KEPT is None:
        c = pd.read_csv(CSV, usecols=['mouse', 'day', 'cluster_id', 'cell_class_of1'])
        c = c[c.cell_class_of1 != 'EXCL']
        _KEPT = set(zip(c.mouse.astype(int), c.day.astype(int), c.cluster_id.astype(int)))
    return _KEPT


def curate(clusters, mouse, day):
    """A session's TsGroup restricted to the paper's curated units."""
    k = kept()
    return clusters[[c for c in clusters.index if (int(mouse), int(day), int(c)) in k]]
