"""Which structure does each gamma band cohere with, and does that change with
the anchoring state?

This is the test that separates the two versions of the two-loop account, and
they predict OPPOSITE things, so one analysis decides between them:

    v1 (refuted on other grounds)  fast gamma is feedforward SENSORY drive, so
                                   MEC-visual fast-gamma coherence should be
                                   HIGHER in the non-anchored state

    v2 (current)                   slow gamma is the long trans-hippocampal
                                   return loop, so MEC-parasubicular SLOW-gamma
                                   coherence should be HIGHER in the anchored
                                   state, and nothing special should happen
                                   between MEC and visual cortex

It also settles the band assignment itself, which until now has been borrowed
from hippocampal work (Colgin 2009, where fast gamma indexes entorhinal input to
CA1 -- a definition that cannot be carried into MEC, since there the band cannot
index its own input). Coherence with an identified partner structure earns the
assignment in entorhinal cortex or refutes it.

PARASUBICULUM IS THE AVAILABLE RETURN-PATH STRUCTURE. There is no subicular or
CA1 LFP in this collection: of 44 sessions with anatomy, the labels present are
ENTm, VIS*, PAR, a little HPF and a little cerebellum. PAR appears in 20, which
is the one structure on the hippocampal side of the loop with enough sessions to
test. It projects to MEC layer II, so it is on the return path, but it is NOT
the subiculum and the test is correspondingly weaker than the one proposed.

THE SECOND QUESTION, asked at the same time because it needs the same data: is
the band swap entorhinal at all? If visual cortex and cerebellum show the same
state difference, the whole circuit account is unnecessary and this is a global
brain-state signature. Per-region spectra are computed alongside the coherence.

Controls: running only, speed-matched trials between states, and coherence
computed on equal numbers of samples per state.

Writes data/lfp/cross_region_gamma.csv  and  data/lfp/region_spectra.csv
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
from scipy.signal import coherence, welch

from lfp_batch import FS, LFP_ROOT, clip_trials, nap, speed_matched, vr_paths

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
OUT_C = f'{ROOT}/data/lfp/cross_region_gamma.csv'
OUT_S = f'{ROOT}/data/lfp/region_spectra.csv'
SLOW = (30., 48.); FAST = (60., 100.)
MIN_RUN_S = 1.0
MIN_SAMP = int(20 * FS)
NPERSEG = 512


def family(b):
    b = str(b)
    if b.startswith('ENTm'):
        return 'MEC'
    if b.startswith('VIS'):
        return 'VIS'
    if b == 'PAR':
        return 'PAR'
    if b in ('SIM', 'FL', 'PFL', 'AN', 'PRM', 'COPY'):
        return 'CERE'
    if b.startswith('HPF') or b in ('dhc', 'or'):
        return 'HPF'
    return None


def band(f, C, lo, hi):
    m = (f >= lo) & (f <= hi)
    return float(np.nanmean(C[m]))


def run_session(mo, dy, rng):
    lf = f'{LFP_ROOT}/M{mo}/D{dy}/VR/sub-M{mo}_ses-D{dy}_typ-VR_beh.nwb'
    ana = f'{LFP_ROOT}/M{mo}/D{dy}/anatomy_M{mo}_D{dy}.csv'
    if not (os.path.exists(lf) and os.path.exists(ana)):
        return None, None
    A = pd.read_csv(ana)
    A['fam'] = A.brain_region.map(family)
    A = A.dropna(subset=['fam'])
    if 'MEC' not in set(A.fam) or len(set(A.fam)) < 2:
        return None, None
    fh = h5py.File(lf, 'r')
    if 'processing/ecephys/LFP/lfp/data' not in fh:
        fh.close(); return None, None
    names = [n.decode() for n in
             fh['general/extracellular_ephys/electrodes/channel_name'][:]]
    # a channel_name is an underscore-joined group; assign it the family its
    # constituent channels mostly belong to
    reg = dict(zip(A.channel_id.astype(str), A.fam))
    fam_of = {}
    for i, n in enumerate(names):
        fs_ = [reg.get(c) for c in n.split('_') if reg.get(c)]
        if fs_:
            s = pd.Series(fs_)
            if s.value_counts().iloc[0] > len(n.split('_')) / 2:
                fam_of[i] = s.mode().iloc[0]
    fams = sorted(set(fam_of.values()))
    if 'MEC' not in fams or len(fams) < 2:
        fh.close(); return None, None

    T = pd.read_csv(f'{ROOT}/data/population_state/anchoring_trials.csv')
    st = T[(T.mouse == mo) & (T.day == dy)]
    if not len(st):
        fh.close(); return None, None
    state = dict(zip(st.trial.astype(int), st.frac_anch > 0.5))

    bp, cp = vr_paths(mo, dy)
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials, orig = clip_trials(beh['trials'].as_dataframe(), clusters)

    d = fh['processing/ecephys/LFP/lfp/data']
    lfp_t = np.arange(d.shape[0]) / FS
    S = beh['S']
    spd = np.asarray(S.values)[np.searchsorted(np.asarray(S.index), lfp_t)
                               .clip(0, len(S) - 1)]
    run = spd >= 3.0
    cols = sorted(fam_of)
    D = d[:, np.array(cols)]
    fh.close()
    col_fam = np.array([fam_of[c] for c in cols])
    # one representative trace per family: the mean across its groups
    X = {f_: D[:, col_fam == f_].mean(1) for f_ in fams}

    # speed-matched trials
    tinfo = []
    for _, tr in trials.iterrows():
        u = int(tr.number)
        if u not in state:
            continue
        m = (lfp_t >= tr.start) & (lfp_t <= tr.end) & run
        if m.sum() < MIN_RUN_S * FS:
            continue
        tinfo.append((u, bool(state[u]), float(spd[m].mean()), m))
    if len(tinfo) < 10:
        return None, None
    km = speed_matched(np.array([t[2] for t in tinfo]),
                       np.array([t[1] for t in tinfo]), rng)
    sel = [tinfo[i] for i in km]
    masks = {True: np.zeros(len(lfp_t), bool), False: np.zeros(len(lfp_t), bool)}
    for u, s_, sp_, m in sel:
        masks[s_] |= m
    if min(masks[True].sum(), masks[False].sum()) < MIN_SAMP:
        return None, None
    n = min(masks[True].sum(), masks[False].sum())

    crow, srow = [], []
    for s_, tag in ((True, 'anch'), (False, 'non')):
        idx = np.where(masks[s_])[0][:n]
        for f_ in fams:
            fr, P = welch(X[f_][idx], FS, nperseg=NPERSEG)
            w = np.log10(P * fr + 1e-20)
            keep = (fr >= 2) & (fr <= 140)
            z = (w[keep] - w[keep].mean()) / w[keep].std()
            for ff, zz in zip(fr[keep], z):
                srow.append(dict(mouse=mo, day=dy, state=tag, region=f_,
                                 freq=float(ff), z=float(zz)))
        for i, a in enumerate(fams):
            for b in fams[i + 1:]:
                fr, C = coherence(X[a][idx], X[b][idx], FS, nperseg=NPERSEG)
                crow.append(dict(mouse=mo, day=dy, state=tag, pair=f'{a}-{b}',
                                 slow=band(fr, C, *SLOW),
                                 fast=band(fr, C, *FAST),
                                 n_samp=int(n)))
    return pd.DataFrame(crow), pd.DataFrame(srow)


if __name__ == '__main__':
    T = pd.read_csv(f'{ROOT}/data/population_state/anchoring_trials.csv')
    sess = sorted(set(zip(T.mouse.astype(int), T.day.astype(int))))
    C, S, t0 = [], [], time.time()
    for mo, dy in sess:
        rng = np.random.default_rng(abs(hash((mo, dy))) % 2 ** 32)
        try:
            c, s = run_session(mo, dy, rng)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        if c is None:
            continue
        C.append(c); S.append(s)
        print(f'  M{mo}D{dy}: {sorted(set(c.pair))} ({time.time()-t0:.0f}s)',
              flush=True)
    if C:
        pd.concat(C, ignore_index=True).to_csv(OUT_C, index=False)
        pd.concat(S, ignore_index=True).to_csv(OUT_S, index=False)
        print(f'\nwrote {OUT_C} and {OUT_S}')
