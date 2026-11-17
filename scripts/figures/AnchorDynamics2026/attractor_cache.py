"""Per-session population cache for the attractor analyses beyond Figure S18.

attractor_coherence.py reads only the grid module of each session. The
reliability analysis (attractor_reliability.py) and the population search
(attractor_population_search.py) need every simultaneously recorded MEC cell,
and both run many times over the same sessions, so the rates are built once
here, exactly as attractor_coherence.session_maps builds them (100 ms counts,
smoothed at 250 ms, z-scored per cell; position maps on the same grid), and the
module is found exactly as there, from the session's grid cells. Sessions are
the ones Figure S18 uses (same criteria).

Writes data/population_state/attractor_cache/M{mouse}D{day}.npz
Usage:  python attractor_cache.py            (ONLY=M25D24 for one session)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attractor_coherence as AC                                  # noqa: E402

OUT = f'{AC.PS}/attractor_cache'


def sessions():
    """The Figure S18 sessions: grid ids, trial states, MEC cell ids."""
    U = pd.read_csv(f'{AC.PS}/unit_table.csv')
    T = pd.read_csv(f'{AC.PS}/anchoring_trials.csv')
    R = pd.read_csv(f'{AC.PS}/pc1_by_region.csv')
    mec = set(zip(R.mouse[R.is_mec], R.day[R.is_mec], R.cluster_id[R.is_mec]))
    for (mo, dy), g in U[U.in_population].groupby(['mouse', 'day']):
        ids = [int(c) for c in g[g.identity == 'grid'].cluster_id]
        if len(ids) < AC.MIN_GRID:
            continue
        t = T[(T.mouse == mo) & (T.day == dy)].sort_values('trial')
        if len(t) < 60:
            continue
        st = t.pc1.values > 0
        if st.mean() < .15 or st.mean() > .85:
            continue
        u = U[(U.mouse == mo) & (U.day == dy)]
        allc = [int(c) for c in u.cluster_id if (mo, dy, int(c)) in mec]
        yield int(mo), int(dy), ids, st, allc, dict(zip(u.cluster_id.astype(int), u.identity))


def build(mo, dy, ids, st, allc, ident):
    S = AC.session_maps(mo, dy, sorted(set(allc) | set(ids)))
    if S is None:
        return None
    gid = [c for c in ids if c in S['pos']]
    phi, mod, phi_all = AC.torus_offsets(S['pos'], gid)
    if mod is None:
        return None
    pos_in = {o: i for i, o in enumerate(S['orig'])}
    idx = np.array([pos_in.get(int(x), -1) for x in S['tnum']])
    ok = (idx >= 0) & (idx < len(st)) & (S['spd'] > AC.MOVE)
    lab = np.where(ok, st[idx.clip(0, len(st) - 1)], False)
    cells = list(S['have'])
    return dict(mouse=mo, day=dy, cells=np.array(cells),
                ident=np.array([ident.get(c, '') for c in cells]),
                is_grid=np.array([c in set(gid) for c in cells]),
                module=np.array(mod), phi=phi, phi_all=phi_all,
                R=np.stack([S['tim'][c] for c in cells]).astype(np.float32),
                P=np.stack([S['pos'][c] for c in cells]).astype(np.float32),
                state=st, idx=idx, ok=ok, anch=ok & lab, non=ok & ~lab,
                spd=S['spd'].astype(np.float32),
                trackpos=S['trackpos'].astype(np.float32), nbin=AC.NBIN)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    only, t0 = os.environ.get('ONLY'), time.time()
    for mo, dy, ids, st, allc, ident in sessions():
        if only and f'M{mo}D{dy}' != only:
            continue
        f = f'{OUT}/M{mo}D{dy}.npz'
        if os.path.exists(f) and not os.environ.get('FORCE'):
            print(f'  = M{mo}D{dy} cached', flush=True)
            continue
        try:
            z = build(mo, dy, ids, st, allc, ident)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True)
            continue
        if z is None:
            print(f'  - M{mo}D{dy}: no module', flush=True)
            continue
        np.savez_compressed(f, **z)
        print(f'  M{mo}D{dy}: {len(z["cells"])} MEC cells, {len(z["module"])} grid in module, '
              f'{time.time() - t0:.0f}s', flush=True)
