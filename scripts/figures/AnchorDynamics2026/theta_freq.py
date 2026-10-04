"""Does theta shift in FREQUENCY, or weaken in amplitude, during anchoring?

Band power in a fixed 6-10 Hz window falls if the rhythm weakens OR if it moves
out of the window. These are different claims about the circuit, and the data
separate them: the file carries theta PHASE alongside the LFP, so instantaneous
frequency is the phase derivative -- no fitting, no windowing.

Three measures per trial, from MEC channel groups, running samples only:
  peak_hz   frequency of the spectral peak in 4-15 Hz (from the LFP)
  inst_hz   median instantaneous frequency from the unwrapped phase derivative
  amp       theta amplitude, as the RMS of the 6-10 Hz filtered LFP -- an
            amplitude that does NOT depend on where in the band the peak sits,
            unlike band power

If theta merely shifts, peak_hz and inst_hz move while amp holds. If it weakens,
amp falls while the frequencies hold. Both can happen.

Speed matching is carried over from the power analysis: anchored and
non-anchored trials differ by 4.7 cm/s, and theta frequency is even more
speed-dependent than theta power.
"""
import os, sys, json, time, warnings
import h5py, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
from scipy.signal import welch, butter, filtfilt
import pynapple as nap
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lfp_batch import (LFP_ROOT, FS, mec_groups, speed_matched,
                       vr_paths, clip_trials)                       # noqa

OUT = '/Users/harryclark/Documents/spatial-manifolds/data/lfp'
BP = butter(3, [6 / (FS / 2), 10 / (FS / 2)], btype='band')


def run_session(mo, dy, rng):
    lf = f'{LFP_ROOT}/M{mo}/D{dy}/VR/sub-M{mo}_ses-D{dy}_typ-VR_beh.nwb'
    if not os.path.exists(lf):
        return None
    f = h5py.File(lf, 'r')
    if 'processing/ecephys/LFP/lfp/data' not in f:
        f.close(); return None
    names = [n.decode() for n in f['general/extracellular_ephys/electrodes/channel_name'][:]]
    M = mec_groups(mo, dy, names)
    if len(M) < 4:
        f.close(); return None
    bp_, cp = vr_paths(mo, dy)
    beh = nap.load_file(bp_); clusters = nap.load_file(cp)
    trials, orig = clip_trials(beh['trials'].as_dataframe(), clusters)
    # Canonical labels, not the eye table. The eye table was only ever the state
    # source because it happened to carry frac_anch; reading the population-state
    # build instead means the LFP analyses use the same labels as every figure,
    # covers the 62 sessions with entorhinal cells rather than the 58 with eye
    # tracking, and does not silently go stale when the classifier changes.
    T = pd.read_csv('/Users/harryclark/Documents/spatial-manifolds/data/'
                    'population_state/anchoring_trials.csv')
    st = T[(T.mouse == mo) & (T.day == dy)]
    if not len(st):
        f.close(); return None
    state = dict(zip(st.trial.astype(int), st.frac_anch > 0.5))

    cols = M.g.values
    rc = np.sort(cols); back = np.array([int(np.where(rc == c)[0][0]) for c in cols])
    L = f['processing/ecephys/LFP/lfp/data'][:, rc][:, back]
    PH = f['processing/ecephys/Processed/theta/data'][:, rc][:, back]
    f.close()
    n = L.shape[0]
    t = np.arange(n) / FS
    S = beh['S']
    spd = np.asarray(S.values)[np.searchsorted(np.asarray(S.index), t).clip(0, len(S) - 1)]
    run = spd >= 3.0

    # Instantaneous frequency from the phase derivative. np.unwrap is NOT used:
    # it propagates NaN forward, so one gap in the phase array turns everything
    # after it into NaN -- which is exactly what happened first time round. The
    # wrapped per-sample difference is local, so a gap costs only that sample.
    d = np.diff(PH, axis=0)
    d = (d + np.pi) % (2 * np.pi) - np.pi            # to [-pi, pi]
    inst = d * FS / (2 * np.pi)
    inst = np.vstack([inst[:1], inst])
    inst[(inst < 2) | (inst > 20)] = np.nan          # physiologically impossible
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
        peak = fr[sel][np.argmax(P[sel], axis=0)]
        rows.append(dict(mouse=mo, day=dy, trial=u, anch=bool(state[u]),
                         speed=float(spd[m].mean()),
                         peak_hz=float(np.nanmean(peak)),
                         inst_hz=float(np.nanmedian(inst[m])),
                         amp=float(np.sqrt(np.nanmean(filt[m] ** 2)))))
    R = pd.DataFrame(rows)
    if not len(R):
        return None
    R['z_amp'] = (R.amp - R.amp.mean()) / R.amp.std()
    # A session with only one state has nothing to match, but its per-trial
    # theta is still worth keeping -- it is the single-state sessions that make
    # the population-level comparison interpretable, and they are useful as
    # controls. Mark them unmatched rather than dropping the session.
    km = speed_matched(R.speed.values, R.anch.values, rng)
    R['speed_matched'] = False
    if len(km):
        R.loc[R.index[km], 'speed_matched'] = True
    return R


if __name__ == '__main__':
    T = pd.read_csv('/Users/harryclark/Documents/spatial-manifolds/data/'
                    'population_state/anchoring_trials.csv')
    sess = sorted(set(zip(T.mouse, T.day)))
    out, t0 = [], time.time()
    for k, (mo, dy) in enumerate(sess, 1):
        rng = np.random.default_rng(abs(hash((mo, dy))) % 2**32)
        try:
            R = run_session(int(mo), int(dy), rng)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        if R is None:
            continue
        out.append(R)
        if k % 10 == 0:
            print(f'[{k}/{len(sess)}] {time.time()-t0:.0f}s', flush=True)
    A = pd.concat(out, ignore_index=True)
    A.to_csv(f'{OUT}/theta_frequency.csv', index=False)
    print(f'\n{len(A)} trials from {A.groupby(["mouse","day"]).ngroups} sessions')
