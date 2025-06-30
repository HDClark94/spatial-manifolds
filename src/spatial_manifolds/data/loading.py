import pynapple as nap
from pathlib import Path

from spatial_manifolds.data.curation import curate_clusters

def load_session(args) -> tuple[nap.NWBFile, Path, nap.TsGroup]:
    session_path = (
        args.storage
        / f'M{args.mouse}'
        / f'D{args.day:0>2}'
        / args.session_type
    )
    session_file = (
        session_path
        / f'sub-{args.mouse}_day-{args.day:0>2}_ses-{args.session_type}_beh.nwb'
    )
    assert session_file.exists(), f'Could not find {session_file}.'
    if hasattr(args, 'sorter'):
        clusters_file = (
            session_path
            / f'sub-{args.mouse}_day-{args.day:0>2}_ses-{args.session_type}_srt-{args.sorter}_clusters.npz'
        )
        if clusters_file.exists():
            return (
            nap.load_file(session_file, lazy_loading=False),
            session_path,
            curate_clusters(nap.load_file(clusters_file))
            )
        else:
            return (
            nap.load_file(session_file, lazy_loading=False),
            session_path,
            )
    else:
        return (
            nap.load_file(session_file, lazy_loading=False),
            session_path,
        )

