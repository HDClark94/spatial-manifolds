"""Countable gamma events per theta cycle, against computational demand.

THE HYPOTHESIS being operationalised. Gamma cycles nested in theta act as
discrete computational slots, and what should scale with demand is the NUMBER OF
RELIABLY DISTINGUISHABLE GAMMA EVENTS rather than gamma power.

WHAT IS AND IS NOT AN EMPIRICAL CLAIM HERE. "Slow gamma gives ~5 slots per theta
cycle and fast gamma ~10" is arithmetic, not a finding: 40/8 = 5 and 80/8 = 10 by
definition, and measuring it would only re-measure the two frequencies. The
hypothesis only becomes testable once "slot" means a DETECTED EVENT -- a burst of
gamma that actually occurred -- because the number of events can then be fewer
than the number of available cycles, and can vary while frequency does not. That
is the quantity computed here.

    gamma cycles available per theta cycle   = f_gamma / f_theta   (arithmetic)
    gamma EVENTS detected per theta cycle    = what this measures   (empirical)
    occupancy = events / available cycles                           (0 to 1)

THE DEMAND CONTRAST THE TASK ALREADY PROVIDES. Uncued (non-beaconed) trials can
only be solved by path integration; cued trials offer a visible landmark. That is
a within-session, randomly interleaved manipulation of exactly the demand the
hypothesis is about, on ~15,000 trials. If gamma events index computational
slots engaged for path integration, uncued trials should recruit more of them.

THE CONTROL THAT DECIDES WHETHER THIS IS NEW. Event count and band power are
strongly correlated -- more power crosses a threshold more often -- so an event
effect that disappears once power is partialled out is a power effect wearing a
different name. Both are therefore recorded per trial, and the event analysis is
reported both raw and with power regressed out.

Detection: envelope z-scored within session over running samples; an event is a
supra-threshold excursion lasting at least one gamma cycle; events are assigned
to the theta cycle containing their peak.

WHERE ON THE TRACK. Gamma is also binned by position, because the track is not
homogeneous: the reward zone sits mid-track, and every trial begins with a
teleport that resets position without self-motion. If gamma events index
computational slots for position updating, they have no reason to be uniform
along the track -- and if they peak at the reward zone or at the start, that
locates the computation rather than merely counting it.

Usage:  python gamma_events.py [worker n_workers]
Writes  data/lfp/gamma_events.csv      one row per trial
        data/lfp/gamma_by_position.csv one row per session x position bin x
                                       trial type x state
"""
import os
import sys
import warnings

import h5py
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.signal import filtfilt, hilbert
from lfp_batch import FS, LFP_ROOT, OUT, clip_trials, mec_groups, nap, vr_paths
from lfp_spectrum_pac import _bp

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
BANDS = [('slow_gamma', 30., 48.), ('fast_gamma', 60., 100.)]
THR_SD = 2.0               # envelope z for an event
MAX_GROUPS = 6
MIN_RUN_S = 2.0
TL = 200.0                 # cm
POS_BIN = 5.0              # cm


def detect_events(z, min_len):
    """Start/peak indices of supra-threshold excursions lasting >= min_len."""
    above = z >= THR_SD
    d = np.diff(np.concatenate([[0], above.astype(np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    keep = (ends - starts) >= min_len
    starts, ends = starts[keep], ends[keep]
    if not len(starts):
        return np.empty(0, int)
    return np.array([s + int(np.argmax(z[s:e])) for s, e in zip(starts, ends)])


def run_session(mo, dy):
    lf = f'{LFP_ROOT}/M{mo}/D{dy}/VR/sub-M{mo}_ses-D{dy}_typ-VR_beh.nwb'
    if not os.path.isdir(LFP_ROOT):
        raise RuntimeError(f'LFP root {LFP_ROOT} is not mounted')
    if not os.path.exists(lf):
        return None
    f = h5py.File(lf, 'r')
    if ('processing/ecephys/LFP/lfp/data' not in f
            or 'processing/ecephys/Processed/theta/data' not in f):
        f.close(); return None
    names = [n.decode() for n in
             f['general/extracellular_ephys/electrodes/channel_name'][:]]
    M = mec_groups(mo, dy, names)
    if len(M) < 4:
        f.close(); return None
    if len(M) > MAX_GROUPS:
        M = M.iloc[np.linspace(0, len(M) - 1, MAX_GROUPS).round().astype(int)]
    cols = M.g.values
    rc = np.sort(cols); back = np.array([int(np.where(rc == c)[0][0]) for c in cols])
    D = f['processing/ecephys/LFP/lfp/data'][:, rc][:, back].astype(np.float32)
    PH = f['processing/ecephys/Processed/theta/data'][:, rc][:, back].astype(np.float32)
    f.close()

    bp, cp = vr_paths(mo, dy)
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials, _ = clip_trials(beh['trials'].as_dataframe(), clusters)
    T = pd.read_csv(f'{ROOT}/data/population_state/anchoring_trials.csv')
    stt = T[(T.mouse == mo) & (T.day == dy)]
    if not len(stt):
        return None
    state = dict(zip(stt.trial.astype(int), stt.frac_anch > .5))
    # trial type from the canonical trial table rather than the NWB, so it
    # matches every other figure's cued/uncued split
    N = pd.read_csv(f'{ROOT}/data/population_state/nonmec_population_state.csv')
    nn = N[(N.mouse == mo) & (N.day == dy)]
    ttype = dict(zip(nn.trial.astype(int), nn.ttype.astype(str)))

    t = np.arange(D.shape[0]) / FS
    tn, trav = beh['trial_number'], beh['travel']
    dtv = trav - (float(np.asarray(tn.values)[0]) - 1) * TL
    pos = np.asarray(dtv.values)[np.searchsorted(np.asarray(dtv.index), t)
                                 .clip(0, len(dtv) - 1)] % TL
    S = beh['S']
    spd = np.asarray(S.values)[np.searchsorted(np.asarray(S.index), t)
                               .clip(0, len(S) - 1)]
    run = spd >= 3.0
    if run.sum() < 60 * FS:
        return None

    # theta cycle boundaries, from the stored phase: a new cycle starts wherever
    # the wrapped phase steps backwards
    ph0 = PH[:, 0]
    cyc = np.zeros(len(t), np.int64)
    np.cumsum(np.concatenate([[0], (np.diff(ph0) < -np.pi).astype(np.int64)]),
              out=cyc)

    env, ev = {}, {}
    for nm, lo, hi in BANDS:
        b, a = _bp(lo, hi)
        e = np.abs(hilbert(filtfilt(b, a, D, axis=0), axis=0))
        mu = e[run].mean(0); sd = e[run].std(0) + 1e-12
        env[nm] = (e - mu) / sd
        min_len = int(FS / hi)
        ev[nm] = [detect_events(np.where(run, env[nm][:, j], -9), min_len)
                  for j in range(D.shape[1])]

    rows = []
    for _, tr in trials.iterrows():
        u = int(tr.number)
        if u not in state:
            continue
        m = (t >= tr.start) & (t <= tr.end) & run
        if m.sum() < MIN_RUN_S * FS:
            continue
        i0, i1 = np.where(m)[0][[0, -1]]
        n_cyc = len(np.unique(cyc[m]))
        if n_cyc < 5:
            continue
        theta_hz = n_cyc / (m.sum() / FS)
        rec = dict(mouse=mo, day=dy, trial=u, anch=bool(state[u]),
                   ttype=ttype.get(u, '?'),
                   speed=float(spd[m].mean()), n_cyc=int(n_cyc),
                   theta_hz=float(theta_hz))
        for nm, lo, hi in BANDS:
            cnt = np.mean([np.sum((e >= i0) & (e <= i1)) for e in ev[nm]])
            rec[f'{nm}_ev'] = float(cnt / n_cyc)             # events per theta cycle
            rec[f'{nm}_avail'] = float((lo + hi) / 2 / theta_hz)   # cycles available
            rec[f'{nm}_occ'] = rec[f'{nm}_ev'] / rec[f'{nm}_avail']
            rec[f'{nm}_pow'] = float(env[nm][m].mean())      # z envelope, the control
        rows.append(rec)

    # ---- the same quantities, binned by position along the track -------------
    # Lookups are done through ARRAYS indexed by trial number, not through dict
    # comprehensions over the ~1.8 M samples. The comprehension version ran eight
    # times per session and dominated the runtime; it was also silently wrong,
    # because `state[...] is a_` compares a numpy.bool_ against a Python bool
    # with identity and is therefore always False, so every position bin came
    # back empty.
    edges = np.arange(0, TL + POS_BIN, POS_BIN)
    pb = np.clip(np.digitize(pos, edges) - 1, 0, len(edges) - 2)
    tn_s = np.asarray(tn.values)[np.searchsorted(np.asarray(tn.index), t)
                                 .clip(0, len(tn) - 1)].astype(int)
    hi_tr = int(max(tn_s.max(), max(state) if state else 0)) + 1
    st_arr = np.full(hi_tr, -1, np.int8)
    ty_arr = np.full(hi_tr, 0, np.int8)            # 0 unknown, 1 cued, 2 uncued
    for u_, v_ in state.items():
        if 0 <= u_ < hi_tr:
            st_arr[u_] = int(bool(v_))
    for u_, v_ in ttype.items():
        if 0 <= u_ < hi_tr:
            ty_arr[u_] = 1 if v_ == 'b' else (2 if v_ == 'nb' else 0)
    st_s, ty_s = st_arr[tn_s], ty_arr[tn_s]
    inset = run & (st_s >= 0) & (ty_s > 0)

    prows = []
    for nm, lo, hi in BANDS:
        evmask = np.zeros(len(t), bool)
        for e in ev[nm]:
            evmask[e] = True
        for a_ in (1, 0):
            for ty_code, ty in ((1, 'b'), (2, 'nb')):
                sel = inset & (st_s == a_) & (ty_s == ty_code)
                if sel.sum() < 5 * FS:
                    continue
                for bidx in range(len(edges) - 1):
                    k = sel & (pb == bidx)
                    if k.sum() < 200:
                        continue
                    prows.append(dict(
                        mouse=mo, day=dy, band=nm, anch=bool(a_), ttype=ty,
                        pos=float(edges[bidx] + POS_BIN / 2),
                        ev_rate=float(evmask[k].sum() / (k.sum() / FS)),
                        power=float(env[nm][k].mean()),
                        speed=float(spd[k].mean()), n=int(k.sum())))
    return (pd.DataFrame(rows) if rows else None,
            pd.DataFrame(prows) if prows else None)


if __name__ == '__main__':
    w, nw = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    T = pd.read_csv(f'{ROOT}/data/population_state/anchoring_trials.csv')
    sess = sorted(set(zip(T.mouse.astype(int), T.day.astype(int))))
    sess = [s for i, s in enumerate(sess) if i % nw == w]
    out, pout = [], []
    for k, (mo, dy) in enumerate(sess, 1):
        try:
            r = run_session(mo, dy)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True)
            continue
        if r is None:
            print(f'[{k}/{len(sess)}] M{mo}D{dy} --', flush=True); continue
        a, b = r
        if a is not None:
            out.append(a)
        if b is not None:
            pout.append(b)
        print(f'[{k}/{len(sess)}] M{mo}D{dy} ok', flush=True)
    # report what was ACTUALLY written. An earlier version printed the "wrote"
    # line unconditionally, so when the LFP drive unmounted mid-run and every
    # session returned None the logs still claimed success for all four workers.
    sfx = '' if nw == 1 else f'_w{w}'
    if not out:
        print(f'\nNOTHING WRITTEN: no session yielded data. If the sessions all '
              f'reported "--", check that {LFP_ROOT} is mounted.', flush=True)
        sys.exit(1)
    pd.concat(out, ignore_index=True).to_csv(f'{OUT}/gamma_events{sfx}.csv',
                                             index=False)
    print(f'\nwrote {OUT}/gamma_events{sfx}.csv ({len(out)} sessions)')
    if pout:
        pd.concat(pout, ignore_index=True).to_csv(
            f'{OUT}/gamma_by_position{sfx}.csv', index=False)
        print(f'wrote {OUT}/gamma_by_position{sfx}.csv')
