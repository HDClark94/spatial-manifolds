"""Theta extraction for sessions the main batch never saw.

data/lfp/theta_frequency.csv covers 58 sessions because theta_freq.py takes its
session list -- and its per-trial anchoring labels -- from the eye-tracking
table. M25 D24 is the one session in the dataset with no eye channels, so it was
excluded for a reason that has nothing to do with its LFP or its cells. It is
in fact a good example session: 60 ENTm spatial cells, 223 trials, PC1 carrying
36% of the label variance, and 51.6% of trials anchored, which is the best
balance of any candidate.

This recovers such sessions by deriving the anchoring state from the cell
labels directly rather than from the eye table, and writes to its OWN file.
It does not touch theta_frequency.csv: an earlier single-session run in this
project overwrote the full table, and the extra rows are merged at read time
instead (see lfp_examples.py).
"""
import os, sys, json, time, warnings
import h5py, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from scipy.signal import welch, butter, filtfilt
import pynapple as nap
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/'
                   'AnchorDynamics2026')
from lfp_batch import LFP_ROOT, FS, mec_groups, speed_matched, vr_paths, clip_trials

_g = {}
for _c in json.load(open('/Users/harryclark/Documents/spatial-manifolds/scripts/'
                         'figures/AnchorDynamics2026/lick_raster_by_trial_type.ipynb'))['cells']:
    if _c['cell_type'] != 'code':
        continue
    _s = ''.join(_c['source'])
    if _s.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in _s:
        continue
    exec(compile(_s, '<nb>', 'exec'), _g)
globals().update({k: v for k, v in _g.items() if not k.startswith('__')})
from spatial_manifolds.anchoring import trial_cluster_labels

OUT = '/Users/harryclark/Documents/spatial-manifolds/data/lfp'
BP = butter(3, [6 / (FS / 2), 10 / (FS / 2)], btype='band')
SESSIONS = [(25, 24)]


def population_state(mo, dy):
    """Per-trial anchored/not, from the ENTm cell labels (no eye table needed)."""
    bp, cp = vr_paths(mo, dy)
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials, orig = clip_trials(beh['trials'].as_dataframe(), clusters)
    keep = np.isin(beh['trials'].as_dataframe().number.values.astype(int), orig)
    ids = spatial_cell_ids(mo, dy, clusters)
    if len(ids) < 5:
        return None
    tn, trav = beh['trial_number'], beh['travel']
    dt = trav - (float(np.asarray(tn.values)[0]) - 1) * TL
    moving = beh['S'].threshold(3.0, method='above').time_support
    n_tr = len(beh['trials'].as_dataframe())
    tc = nap.compute_1d_tuning_curves(clusters[ids], dt, nb_bins=n_tr * NBIN,
                                      minmax=[0, n_tr * TL], ep=moving)
    L = []
    for c in ids:
        lab, _, _, _ = trial_cluster_labels(np.asarray(tc[c]).reshape(n_tr, NBIN)[keep])
        if not np.isnan(lab).all():
            L.append(lab)
    frac = np.nanmean(np.array(L), axis=0)
    return dict(zip(trials.number.astype(int), frac > 0.5))


def run_session(mo, dy, state, rng):
    lf = f'{LFP_ROOT}/M{mo}/D{dy}/VR/sub-M{mo}_ses-D{dy}_typ-VR_beh.nwb'
    if not os.path.exists(lf):
        raise FileNotFoundError(lf)
    f = h5py.File(lf, 'r')
    names = [n.decode() for n in f['general/extracellular_ephys/electrodes/channel_name'][:]]
    M = mec_groups(mo, dy, names)
    if len(M) < 4:
        f.close(); raise RuntimeError(f'only {len(M)} MEC groups')
    bp_, cp = vr_paths(mo, dy)
    beh = nap.load_file(bp_); clusters = nap.load_file(cp)
    trials, orig = clip_trials(beh['trials'].as_dataframe(), clusters)

    cols = M.g.values
    rc = np.sort(cols); back = np.array([int(np.where(rc == c)[0][0]) for c in cols])
    L = f['processing/ecephys/LFP/lfp/data'][:, rc][:, back]
    PH = f['processing/ecephys/Processed/theta/data'][:, rc][:, back]
    f.close()
    t = np.arange(L.shape[0]) / FS
    S = beh['S']
    spd = np.asarray(S.values)[np.searchsorted(np.asarray(S.index), t).clip(0, len(S) - 1)]
    run = spd >= 3.0

    d = np.diff(PH, axis=0)
    d = (d + np.pi) % (2 * np.pi) - np.pi
    inst = np.vstack([(d * FS / (2 * np.pi))[:1], d * FS / (2 * np.pi)])
    inst[(inst < 2) | (inst > 20)] = np.nan
    filt = filtfilt(*BP, L, axis=0)

    rows = []
    for _, tr in trials.iterrows():
        u = int(tr.number)
        if u not in state:
            continue
        m = (t >= tr.start) & (t <= tr.end) & run
        if m.sum() < FS:
            continue
        fr, P = welch(L[m], fs=FS, nperseg=512, axis=0)
        sel = (fr >= 4) & (fr < 15)
        rows.append(dict(mouse=mo, day=dy, trial=u, anch=bool(state[u]),
                         speed=float(spd[m].mean()),
                         peak_hz=float(np.nanmean(fr[sel][np.argmax(P[sel], axis=0)])),
                         inst_hz=float(np.nanmedian(inst[m])),
                         amp=float(np.sqrt(np.nanmean(filt[m] ** 2)))))
    R = pd.DataFrame(rows)
    R['z_amp'] = (R.amp - R.amp.mean()) / R.amp.std()
    km = speed_matched(R.speed.values, R.anch.values, rng)
    R['speed_matched'] = False
    R.loc[R.index[km], 'speed_matched'] = True
    return R


if __name__ == '__main__':
    out = []
    for mo, dy in SESSIONS:
        t0 = time.time()
        state = population_state(mo, dy)
        if state is None:
            print(f'  ! M{mo}D{dy}: too few ENTm cells'); continue
        R = run_session(mo, dy, state, np.random.default_rng(abs(hash((mo, dy))) % 2**32))
        out.append(R)
        a, n = R[R.anch], R[~R.anch]
        print(f'M{mo}D{dy}: {len(R)} trials, {100*R.anch.mean():.1f}% anchored, '
              f'amp {a.amp.mean():.1f} vs {n.amp.mean():.1f} '
              f'({100*(a.amp.mean()-n.amp.mean())/n.amp.mean():+.1f}%), '
              f'{time.time()-t0:.0f}s')
    A = pd.concat(out, ignore_index=True)
    p = f'{OUT}/theta_frequency_extra.csv'
    if os.path.exists(p):                       # merge, never clobber
        old = pd.read_csv(p)
        done = set(zip(A.mouse, A.day))
        A = pd.concat([old[~old.set_index(['mouse', 'day']).index.isin(done)], A],
                      ignore_index=True)
    A.to_csv(p, index=False)
    print(f'wrote {p}: {len(A)} trials, {A.groupby(["mouse","day"]).ngroups} sessions')
