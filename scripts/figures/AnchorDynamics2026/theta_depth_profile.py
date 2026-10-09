"""Theta-triggered LFP and CSD against depth, separately for the two states.

S11 asks where in the theta cycle each FREQUENCY peaks, pooled over channels.
This asks the other question: where in the theta cycle each DEPTH peaks. The
laminar phase gradient is the signature of the entorhinal travelling wave
(Hernandez-Perez et al. 2020), and if the anchoring state reorganises the
entorhinal circuit the gradient is a place it could show.

WHAT IS COMPUTED, per session and per state:

  trig   the theta-triggered average LFP for every channel group on one shank,
         triggered on the theta PEAK of a reference group, +/- 200 ms
  csd    its second spatial derivative across depth, the current source
         density, which localises sinks and sources instead of showing the
         volume-conducted field
  phase  the circular mean of (theta phase at this depth - theta phase at the
         reference), taken over all running samples of that state rather than
         from the triggered average, so the gradient has a per-session estimate
         that can be tested

ONE SHANK ONLY. The groups are eight neighbouring contacts averaged, so a group
spans about 60 um of depth; across shanks the same depth is a different
mediolateral position and averaging them would blur the laminar profile that is
the point. The shank carrying the most entorhinal groups is used.

THE REFERENCE IS THE GROUP AT THE PROBE TIP, not the strongest. Picking the
strongest would choose a different lamina in different sessions and the
across-session average would then be taken about a moving origin. The tip is
the superficial end of the entorhinal span in these penetrations -- probe_y
runs ENTm2 at the tip to ENTm6 at the top -- so phase is reported relative to
layer 2 and increases with distance into the deep layers. Sessions are
aligned by LAYER rather than by raw probe depth before averaging, since the
probe sits at a different absolute depth in each.

Trials are speed-matched between states as everywhere else in Figure 7.

Writes data/lfp/theta_depth_profile.npz
"""
import os
import sys
import time
import warnings

import h5py
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from scipy.fft import next_fast_len
from scipy.signal import butter, filtfilt, hilbert

from lfp_batch import FS, LFP_ROOT, clip_trials, nap, speed_matched, vr_paths

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
OUT = f'{ROOT}/data/lfp/theta_depth_profile.npz'

THETA = (6., 10.)
WIN = int(.2 * FS)            # +/- 200 ms, as in the published figures
MIN_GROUPS = 6                # fewer than this is not a laminar profile
MIN_SAMP = int(30 * FS)
MIN_CYCLES = 200


def shank_of(cid):
    """'s1e288' -> 1."""
    try:
        return int(str(cid).split('e')[0].lstrip('s'))
    except Exception:
        return -1


def ent_groups_one_shank(mo, dy, names):
    """Entorhinal channel groups on the best shank, ordered by probe depth."""
    A = pd.read_csv(f'{LFP_ROOT}/M{mo}/D{dy}/anatomy_M{mo}_D{dy}.csv')
    A['shank'] = A.device_contact_id.map(shank_of)
    reg = dict(zip(A.channel_id, A.brain_region.astype(str)))
    dep = dict(zip(A.channel_id, A.coord_probe_y))
    shk = dict(zip(A.channel_id, A.shank))
    rows = []
    for i, n in enumerate(names):
        chs = n.split('_')
        rs = [reg.get(c, '?') for c in chs]
        if sum(r.startswith('ENTm') for r in rs) <= len(chs) / 2:
            continue
        sh = pd.Series([shk.get(c, -1) for c in chs]).mode().iloc[0]
        lay = pd.Series([r for r in rs if r.startswith('ENTm')]).mode().iloc[0]
        rows.append(dict(g=i, shank=int(sh), layer=lay,
                         y=float(np.nanmean([dep.get(c, np.nan) for c in chs]))))
    E = pd.DataFrame(rows)
    if not len(E):
        return None
    best = E.shank.value_counts().idxmax()
    E = E[E.shank == best].sort_values('y').reset_index(drop=True)
    return E if len(E) >= MIN_GROUPS else None


def run_session(mo, dy, rng):
    lf = f'{LFP_ROOT}/M{mo}/D{dy}/VR/sub-M{mo}_ses-D{dy}_typ-VR_beh.nwb'
    if not os.path.exists(lf):
        return None
    fh = h5py.File(lf, 'r')
    if 'processing/ecephys/LFP/lfp/data' not in fh:
        fh.close(); return None
    names = [n.decode() for n in
             fh['general/extracellular_ephys/electrodes/channel_name'][:]]
    E = ent_groups_one_shank(mo, dy, names)
    if E is None:
        fh.close(); return None
    cols = E.g.values

    bpth, cp = vr_paths(mo, dy)
    beh = nap.load_file(bpth); clusters = nap.load_file(cp)
    trials, _ = clip_trials(beh['trials'].as_dataframe(), clusters)
    T = pd.read_csv(f'{ROOT}/data/population_state/anchoring_trials.csv')
    st = T[(T.mouse == mo) & (T.day == dy)]
    if not len(st):
        fh.close(); return None
    state = dict(zip(st.trial.astype(int), st.frac_anch > 0.5))

    d = fh['processing/ecephys/LFP/lfp/data']
    nT = d.shape[0]
    blk = (d.chunks[0] if d.chunks else 26041) * 8
    X = np.empty((nT, len(cols)), np.float32)
    for i in range(0, nT, blk):
        X[i:i + blk] = d[i:i + blk, :][:, cols]   # full-width reads, then select
    fh.close()

    t = np.arange(nT) / FS
    S = beh['S']
    spd = np.asarray(S.values)[np.searchsorted(np.asarray(S.index), t)
                               .clip(0, len(S) - 1)]
    run = spd >= 3.0

    info = []
    for _, tr in trials.iterrows():
        u = int(tr.number)
        if u not in state:
            continue
        m = (t >= tr.start) & (t <= tr.end) & run
        if m.sum() < FS:
            continue
        info.append((bool(state[u]), float(spd[m].mean()), m))
    if len(info) < 10:
        return None
    km = speed_matched(np.array([i[1] for i in info]),
                       np.array([i[0] for i in info]), rng)
    masks = {True: np.zeros(nT, bool), False: np.zeros(nT, bool)}
    for i in km:
        masks[info[i][0]] |= info[i][2]
    if min(masks[True].sum(), masks[False].sum()) < MIN_SAMP:
        return None

    b, a = butter(3, [THETA[0] / (FS / 2), THETA[1] / (FS / 2)], btype='band')
    nf = next_fast_len(nT)
    ph = np.empty((len(cols), nT))
    for j in range(len(cols)):
        ph[j] = np.angle(hilbert(filtfilt(b, a, X[:, j].astype(float)),
                                 N=nf)[:nT])
    ref = 0                      # deepest group on the shank; see docstring

    out = {}
    for s_, tag in ((True, 'anch'), (False, 'non')):
        mm = masks[s_]
        # theta peaks of the reference inside this state
        pk = np.where((np.diff(np.sign(ph[ref])) < 0))[0] + 1
        pk = pk[mm[pk]]
        pk = pk[(pk > WIN) & (pk < nT - WIN - 1)]
        if len(pk) < MIN_CYCLES:
            return None
        idx = pk[:, None] + np.arange(-WIN, WIN + 1)[None, :]
        trig = X[idx, :].mean(0).T               # groups x time
        trig = trig - trig.mean(1, keepdims=True)
        # phase against the reference, over every running sample of the state
        dphi = np.angle(np.mean(np.exp(1j * (ph[:, mm] - ph[ref, mm])), axis=1))
        out[f'trig_{tag}'] = trig.astype(np.float32)
        out[f'phase_{tag}'] = dphi.astype(np.float32)
        out[f'n_{tag}'] = len(pk)
    out['y'] = E.y.values.astype(np.float32)
    out['layer'] = E.layer.values.astype('U8')
    return out


if __name__ == '__main__':
    T = pd.read_csv(f'{ROOT}/data/population_state/anchoring_trials.csv')
    sess = sorted(set(zip(T.mouse.astype(int), T.day.astype(int))))
    store, t0 = {}, time.time()
    for mo, dy in sess:
        rng = np.random.default_rng(abs(hash((mo, dy))) % 2 ** 32)
        try:
            r = run_session(mo, dy, rng)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        if r is None:
            continue
        for k, v in r.items():
            store[f'M{mo}D{dy}__{k}'] = v
        print(f'  M{mo}D{dy}  {len(r["y"])} groups, '
              f'{r["n_anch"]}/{r["n_non"]} cycles  ({time.time()-t0:.0f}s)',
              flush=True)
    np.savez_compressed(OUT, **store)
    n = len({k.split('__')[0] for k in store})
    print(f'\nwrote {OUT}: {n} sessions')
