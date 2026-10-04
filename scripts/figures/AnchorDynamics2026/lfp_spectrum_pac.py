"""Which frequencies change with the anchoring state, and is theta-nested gamma affected?

Two questions the band analysis in Results §7 cannot answer as it stands.

1. WHICH BANDS. The existing analysis integrates five fixed bands, and reported
   that the change is BROADBAND -- 80-200 Hz falls more than theta does -- which
   is flagged in the paper as a problem, because a weakening confined to theta
   would be the cleaner claim and a general amplitude change is the obvious
   alternative. Five numbers cannot distinguish "theta-specific" from "every
   frequency falls by the same factor". The whole spectrum can: if the log ratio
   between states is flat across frequency, it is a gain change; if it has a
   trough at theta, theta is doing something of its own.

2. THETA-NESTED GAMMA. Never computed. Gamma amplitude in MEC is organised by
   theta phase, and that coupling can change while band powers do not -- it is a
   statement about the TIMING of fast activity within the slow rhythm, not about
   how much of either there is. Tort's modulation index over 18 theta-phase bins,
   for slow (30-55 Hz) and fast (60-100 Hz) gamma.

CONTROLS, all three load-bearing.

  SPEED MATCHING -- theta and gamma both scale with running speed and the states
    differ slightly in speed, so trials are matched within speed deciles.
  EQUAL SAMPLES -- the modulation index is biased upward at small n, and
    anchored trials are not equally numerous, so each state is subsampled to the
    same number of running samples before MI is computed.
  A PHASE-SHUFFLE NULL -- gamma envelopes are rolled by a large random offset
    against theta phase, which destroys the coupling while preserving both
    signals' own statistics. MI minus its null is the quantity reported; raw MI
    is not comparable between states with different signal-to-noise.

Spectra are z-scored within session and channel group before pooling, as
elsewhere, because absolute microvolts depend on impedance and reference.

Usage:  python lfp_spectrum_pac.py [worker n_workers]
Writes  data/lfp/spectrum_by_anchoring.csv   session x frequency x state
        data/lfp/pac_by_anchoring.csv        session x group x state
"""
import os
import sys
import warnings

import h5py
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
import matplotlib

matplotlib.use('Agg')
from scipy.signal import butter, filtfilt, hilbert, welch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lfp_batch import (FS, LFP_ROOT, OUT, clip_trials, mec_groups, nap,
                       speed_matched, vr_paths)

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
MIN_RUN_S = 2.0            # seconds of running before a trial carries a spectrum
NPERSEG = 1024             # ~1 Hz resolution, which is what "which band" needs
FMAX = 250.0
N_PHASE_BINS = 18
N_SURR = 20
MAX_GROUPS = 8             # MEC channel groups per session, spread over depth
GAMMA = [('slow_gamma', 30., 55.), ('fast_gamma', 60., 100.)]
MIN_SAMP_PAC = int(20 * FS)   # 20 s of running per state before MI is trusted


def _bp(lo, hi):
    return butter(3, [lo / (FS / 2), hi / (FS / 2)], btype='band')


def tort_mi(phase, amp, n_bins=N_PHASE_BINS):
    """Tort modulation index: KL divergence of the amplitude-by-phase
    distribution from uniform, normalised to [0, 1].

    The stored theta phase runs 0..2*pi, not -pi..pi, so it is wrapped here
    rather than assumed: binning 0..2*pi data against -pi..pi edges leaves half
    the bins empty, the bin-coverage guard below fires, and every MI comes back
    NaN -- which is exactly what happened first time round.
    """
    phase = (np.asarray(phase) + np.pi) % (2 * np.pi) - np.pi
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx = np.clip(np.digitize(phase, edges) - 1, 0, n_bins - 1)
    m = np.bincount(idx, weights=amp, minlength=n_bins)
    c = np.bincount(idx, minlength=n_bins)
    ok = c > 0
    if ok.sum() < n_bins:
        return np.nan
    m = m[ok] / c[ok]
    if m.sum() <= 0:
        return np.nan
    P = m / m.sum()
    H = -np.sum(P * np.log(P + 1e-12))
    return float((np.log(len(P)) - H) / np.log(len(P)))


def run_session(mo, dy, rng):
    lf = f'{LFP_ROOT}/M{mo}/D{dy}/VR/sub-M{mo}_ses-D{dy}_typ-VR_beh.nwb'
    ana = f'{LFP_ROOT}/M{mo}/D{dy}/anatomy_M{mo}_D{dy}.csv'
    if not (os.path.exists(lf) and os.path.exists(ana)):
        return None, None
    f = h5py.File(lf, 'r')
    if ('processing/ecephys/LFP/lfp/data' not in f
            or 'processing/ecephys/Processed/theta/data' not in f):
        f.close(); return None, None
    names = [n.decode() for n in
             f['general/extracellular_ephys/electrodes/channel_name'][:]]
    M = mec_groups(mo, dy, names)
    if len(M) < 4:
        f.close(); return None, None
    # spread the retained groups over depth rather than taking the first few,
    # since theta-gamma coupling varies steeply through the entorhinal layers
    if len(M) > MAX_GROUPS:
        M = M.iloc[np.linspace(0, len(M) - 1, MAX_GROUPS).round().astype(int)]

    bp, cp = vr_paths(mo, dy)
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials, _ = clip_trials(beh['trials'].as_dataframe(), clusters)
    T = pd.read_csv(f'{ROOT}/data/population_state/anchoring_trials.csv')
    st = T[(T.mouse == mo) & (T.day == dy)]
    if not len(st):
        f.close(); return None, None
    state = dict(zip(st.trial.astype(int), st.frac_anch > 0.5))

    cols = M.g.values
    rc = np.sort(cols); back = np.array([int(np.where(rc == c)[0][0]) for c in cols])
    D = f['processing/ecephys/LFP/lfp/data'][:, rc][:, back].astype(np.float32)
    PH = f['processing/ecephys/Processed/theta/data'][:, rc][:, back].astype(np.float32)
    f.close()

    t = np.arange(D.shape[0]) / FS
    S = beh['S']
    spd = np.asarray(S.values)[np.searchsorted(np.asarray(S.index), t)
                               .clip(0, len(S) - 1)]
    run = spd >= 3.0

    # ---- per-trial spectra ---------------------------------------------------
    rows, tinfo = [], []
    for _, tr in trials.iterrows():
        u = int(tr.number)
        if u not in state:
            continue
        m = (t >= tr.start) & (t <= tr.end) & run
        if m.sum() < MIN_RUN_S * FS:
            continue
        fr, P = welch(D[m], fs=FS, nperseg=NPERSEG, axis=0)
        sel = (fr >= 1) & (fr <= FMAX)
        rows.append((u, bool(state[u]), float(spd[m].mean()), P[sel]))
        tinfo.append((u, bool(state[u]), float(spd[m].mean())))
    if len(rows) < 10:
        return None, None
    fr_sel = fr[sel]
    TI = pd.DataFrame(tinfo, columns=['trial', 'anch', 'speed'])
    km = speed_matched(TI.speed.values, TI.anch.values, rng)
    keep = set(TI.trial.values[km]) if len(km) else set()
    if len(keep) < 10:
        return None, None

    # z-score each frequency within channel group across trials, then average
    # within state: absolute microvolts differ by impedance and reference, and
    # the comparison of interest is within session anyway
    A = np.array([r[3] for r in rows])                    # trials x freq x group
    anch = np.array([r[1] for r in rows])
    inkeep = np.array([r[0] in keep for r in rows])
    L = np.log10(A + 1e-20)
    Z = (L - L.mean(axis=0, keepdims=True)) / (L.std(axis=0, keepdims=True) + 1e-12)
    spec = []
    for tag, msk in (('anch', inkeep & anch), ('non', inkeep & ~anch)):
        if msk.sum() < 5:
            continue
        spec.append(pd.DataFrame(dict(
            mouse=mo, day=dy, state=tag, freq=fr_sel,
            z=Z[msk].mean(axis=(0, 2)),
            log_p=L[msk].mean(axis=(0, 2)), n_trials=int(msk.sum()))))
    SPEC = pd.concat(spec, ignore_index=True) if len(spec) == 2 else None

    # ---- theta-gamma coupling ------------------------------------------------
    tr_state = {int(r[0]): r[1] for r in rows if r[0] in keep}
    mask_state = {True: np.zeros(len(t), bool), False: np.zeros(len(t), bool)}
    for _, tr in trials.iterrows():
        u = int(tr.number)
        if u not in tr_state:
            continue
        m = (t >= tr.start) & (t <= tr.end) & run
        mask_state[tr_state[u]] |= m
    n_each = min(int(mask_state[True].sum()), int(mask_state[False].sum()))
    pac = []
    if n_each >= MIN_SAMP_PAC:
        for nm, lo, hi in GAMMA:
            b, a = _bp(lo, hi)
            env = np.abs(hilbert(filtfilt(b, a, D, axis=0), axis=0))
            for j in range(D.shape[1]):
                for s_ in (True, False):
                    i_ = np.where(mask_state[s_])[0]
                    i_ = rng.choice(i_, n_each, replace=False)
                    i_.sort()
                    ph, am = PH[i_, j], env[i_, j]
                    good = np.isfinite(ph) & np.isfinite(am)
                    if good.sum() < MIN_SAMP_PAC:
                        continue
                    mi = tort_mi(ph[good], am[good])
                    nul = [tort_mi(ph[good],
                                   np.roll(am[good],
                                           int(rng.integers(int(FS * 5),
                                                            good.sum() - int(FS * 5)))))
                           for _ in range(N_SURR)]
                    pac.append(dict(mouse=mo, day=dy, band=nm,
                                    g=int(cols[j]), layer=M.layer.values[j],
                                    dv=float(M.dv.values[j]),
                                    state='anch' if s_ else 'non',
                                    mi=mi, mi_null=float(np.nanmean(nul)),
                                    mi_excess=mi - float(np.nanmean(nul)),
                                    n_samp=int(good.sum())))
    PAC = pd.DataFrame(pac) if pac else None
    return SPEC, PAC


if __name__ == '__main__':
    w, nw = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    T = pd.read_csv(f'{ROOT}/data/population_state/anchoring_trials.csv')
    sess = sorted(set(zip(T.mouse.astype(int), T.day.astype(int))))
    sess = [s for i, s in enumerate(sess) if i % nw == w]
    S_, P_ = [], []
    for k, (mo, dy) in enumerate(sess, 1):
        try:
            a, b = run_session(mo, dy,
                               np.random.default_rng(abs(hash((mo, dy))) % 2 ** 32))
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True)
            continue
        if a is not None:
            S_.append(a)
        if b is not None:
            P_.append(b)
        print(f'[{k}/{len(sess)}] M{mo}D{dy} '
              f'{"spec" if a is not None else "----"} '
              f'{"pac" if b is not None else "---"}', flush=True)
    sfx = '' if nw == 1 else f'_w{w}'
    if S_:
        pd.concat(S_, ignore_index=True).to_csv(
            f'{OUT}/spectrum_by_anchoring{sfx}.csv', index=False)
    if P_:
        pd.concat(P_, ignore_index=True).to_csv(
            f'{OUT}/pac_by_anchoring{sfx}.csv', index=False)
    print(f'\nwrote {OUT}/spectrum_by_anchoring{sfx}.csv and pac_by_anchoring{sfx}.csv')
