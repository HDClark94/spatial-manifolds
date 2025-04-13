import pynapple as nap

from spatial_manifolds.data.curation import curate_clusters


def load_session(args, curate=True):
    session_path = (
        args.storage
        / 'sessions'
        / f'M{args.mouse}'
        / f'D{args.day}'
        / args.session_type
    )
    session_file = (
        session_path / f'M{args.mouse}D{args.day}{args.session_type}.nwb'
    )
    clusters_file = session_path / f'{args.sorter}.npz'
    assert session_file.exists(), f'Could not find {session_file}.'
    assert clusters_file.exists(), f'Could not find {clusters_file}.'
    return (
        nap.load_file(session_file, lazy_loading=False),
        session_path,
        curate_clusters(nap.load_file(clusters_file)),
    )
