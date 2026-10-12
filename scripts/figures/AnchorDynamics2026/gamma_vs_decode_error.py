"""Does fast gamma track the positional mismatch, within non-anchored trials?

THE PREDICTION. can_gate_model.py has the fast-gamma generator driven by the
error between where the landmark says the animal is and where the attractor
says it is, with the gate controlling whether that error is ACTED ON rather
than whether it is transmitted. The sharp consequence is that inside the
non-anchored state, where the error is never corrected, fast-gamma amplitude
should rise and fall with the instantaneous size of that error. A model in
which landmarks are simply not transmitted when non-anchored predicts no such
relationship.

HOW THE MISMATCH IS MEASURED. A Bayesian decoder is trained on ANCHORED trials
only, so its tuning curves define the task frame, and then applied to every
200 ms bin of every non-anchored trial. The circular distance between decoded
and true position is the mismatch: how far the code is from the frame it was
anchored to.

THE CONFOUNDS, and they are serious. Gamma power varies with running speed and
with position on the track; decode error does too. A raw correlation between
them would be carried by both. Each session's gamma envelope and |error| are
therefore residualised on speed and on position -- position entering as the
sine and cosine of track phase plus a 10-bin indicator, so a profile of any
shape is removed -- and the partial correlation taken on the residuals.
Sessions are the unit of analysis, as everywhere else in this paper.

SLOW GAMMA IS THE SPECIFICITY CONTROL. The account says the mismatch drives
the fast band only. If slow gamma tracks the error just as well, the result is
about arousal or engagement rather than about an error signal.

Writes data/lfp/gamma_vs_decode_error.csv, one row per session.
"""
import glob
import json
import os
import sys
import warnings

import h5py
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = os.path.dirname(os.path.abspath(__file__))

from lfp_batch import FS, LFP_ROOT, clip_trials, mec_groups, nap, vr_paths  # noqa

NB = f'{FIG}/position_decoding_by_anchoring.ipynb'
_g = {}
for _c in json.load(open(NB))['cells'][:6]:
    if _c['cell_type'] != 'code':
        continue
    exec(compile(''.join(_c['source']), '<nb>', 'exec'), _g)
globals().update({k: v for k, v in _g.items() if not k.startswith('__')})

OUT_CSV = f'{ROOT}/data/lfp/gamma_vs_decode_error.csv'   # after the
#                      notebook exec, which defines an OUT of its own
GAMMA = [('fast', 60, 100), ('slow', 30, 48)]
MIN_BINS = 150


def _bp(lo, hi):
    return butter(3, [lo / (FS / 2), hi / (FS / 2)], btype='band')


def session(mouse, day):
    """Per 200 ms bin: time, true and decoded position, speed, state."""
    bp, cp = vr_paths(mouse, day)
    beh = nap.load_file(bp)
    clusters = nap.load_file(cp)
    trials_all = beh['trials'].as_dataframe()
    trials, orig = clip_trials(trials_all, clusters)
    if len(trials) < 2 * MIN_TR:
        return None
    cls = CLS[(CLS.mouse == mouse) & (CLS.day == day)]
    ids = [int(c) for c in cls[cls.cell_class_of1.isin(['GC', 'NGS'])].cluster_id
           if int(c) in clusters.index]
    if len(ids) < MIN_CELL:
        return None
    T = pd.read_csv(f'{ROOT}/data/population_state/anchoring_trials.csv')
    stx = T[(T.mouse == mouse) & (T.day == day)]
    if not len(stx):
        return None
    anch = dict(zip(stx.trial.astype(int), stx.pc1 > 0))

    t0, t1 = float(trials.start.min()), float(trials.end.max())
    edges = np.arange(t0, t1, DT)
    ctr = edges[:-1] + DT / 2
    P, S, TNs = beh['P'], beh['S'], beh['trial_number']

    def at(tsd):
        i = np.searchsorted(np.asarray(tsd.index), ctr).clip(0, len(tsd) - 1)
        return np.asarray(tsd.values)[i]

    pos, spd, trn = at(P), at(S), at(TNs).astype(int)
    remap = {int(o): int(n) for o, n in zip(orig, trials.number)}
    tr_new = np.array([remap.get(t, -1) for t in trn])
    ok = (spd >= MIN_SPD) & (tr_new > 0) & np.isfinite(pos)
    pbin = (np.floor(pos / PBIN).astype(int) % NP)
    counts = np.zeros((len(ctr), len(ids)))
    for j, c in enumerate(ids):
        counts[:, j] = np.histogram(np.asarray(clusters[c].index), bins=edges)[0]
    st = np.array([anch.get(int(t), np.nan) for t in tr_new], dtype=float)
    keep = ok & np.isfinite(st)
    return dict(t=ctr[keep], pbin=pbin[keep], spd=spd[keep],
                counts=counts[keep], state=st[keep].astype(bool),
                trial=tr_new[keep])


def gamma_env(mouse, day, t):
    """Band envelopes on the MEC channels, averaged into the decoder's bins."""
    lf = f'{LFP_ROOT}/M{mouse}/D{day}/VR/sub-M{mouse}_ses-D{day}_typ-VR_beh.nwb'
    if not os.path.exists(lf):
        return None
    with h5py.File(lf, 'r') as f_:
        nm = [n.decode() for n in
              f_['general/extracellular_ephys/electrodes/channel_name'][:]]
        Mg = mec_groups(mouse, day, nm)
        if Mg is None or not len(Mg):
            return None
        rc = np.sort(Mg.g.values)
        D = f_['processing/ecephys/LFP/lfp/data'][:, rc].astype(np.float32)
    tt = np.arange(D.shape[0]) / FS
    out = {}
    for nmb, lo, hi in GAMMA:
        b, a = _bp(lo, hi)
        e = np.abs(hilbert(filtfilt(b, a, D, axis=0), axis=0)).mean(1)
        idx = np.searchsorted(tt, t - DT / 2)
        w = int(DT * FS)
        out[nmb] = np.array([e[i:i + w].mean() if i + w <= len(e) else np.nan
                             for i in idx])
    return out


def partial_r(y, x, spd, pbin):
    """Spearman of y against x once speed and position are removed from both."""
    ph = 2 * np.pi * pbin / NP
    Z = np.c_[spd, np.sin(ph), np.cos(ph), np.sin(2 * ph), np.cos(2 * ph),
              np.eye(10)[(pbin / NP * 10).astype(int).clip(0, 9)]]
    Z = np.c_[np.ones(len(Z)), Z]
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < MIN_BINS:
        return np.nan, np.nan
    return spearmanr(res(y[m]), res(x[m]))


def main():
    sess = sorted(glob.glob(f'{LFP_ROOT}/M*/D*/VR'))
    if not sess:
        raise SystemExit(
            f'no sessions under {LFP_ROOT} -- mount the LFP volume. The '
            'behaviour and spikes are read from elsewhere and are available, '
            'but this test needs the field potential.')
    print(f'{len(sess)} LFP sessions')
    rows = []
    for d in sess:
        mo = int(d.split('/M')[1].split('/')[0])
        dy = int(d.split('/D')[1].split('/')[0])
        try:
            S = session(mo, dy)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True)
            continue
        if S is None or S.state.sum() < 50 or (~S.state).sum() < 50:
            continue
        E = gamma_env(mo, dy, S['t'])
        if E is None:
            continue
        # tuning curves from ANCHORED bins define the task frame
        f_tc = tuning(S['counts'][S.state], S['pbin'][S.state])
        dec, _ = decode(S['counts'], f_tc)
        err = circ_err_cm(dec, S['pbin'])
        rec = dict(mouse=mo, day=dy, n_non=int((~S.state).sum()),
                   n_anch=int(S.state.sum()), err_non=float(err[~S.state].mean()),
                   err_anch=float(err[S.state].mean()))
        for nmb, _, _ in GAMMA:
            for tag, m in (('non', ~S.state), ('anch', S.state)):
                r, p = partial_r(E[nmb][m], err[m], S['spd'][m], S['pbin'][m])
                rec[f'r_{nmb}_{tag}'] = r
        rows.append(rec)
        print(f'  M{mo}D{dy}: {rec["n_non"]:4d} non-anchored bins, '
              f'err {rec["err_non"]:5.1f} cm, '
              f'fast r={rec["r_fast_non"]:+.3f} slow r={rec["r_slow_non"]:+.3f}',
              flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT_CSV, index=False)
    print(f'\n{len(R)} sessions')
    for nmb in ('fast', 'slow'):
        for tag in ('non', 'anch'):
            v = R[f'r_{nmb}_{tag}'].dropna()
            if len(v) > 5:
                print(f'  {nmb:<5} gamma, {tag:<4} trials: partial r = '
                      f'{v.mean():+.4f}  p = {wilcoxon(v).pvalue:.3g}  n = {len(v)}')
    v = (R.r_fast_non - R.r_slow_non).dropna()
    if len(v) > 5:
        print(f'  fast minus slow, non-anchored: {v.mean():+.4f}  '
              f'p = {wilcoxon(v).pvalue:.3g}')
    print(f'wrote {OUT_CSV}')


if __name__ == '__main__':
    main()
