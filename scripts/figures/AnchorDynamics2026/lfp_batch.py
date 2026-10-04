"""LFP band power by anchoring state, across every qualifying session.

Raw LFP is in clark2025_lfp; clark2025 itself carries only theta PHASE, from
which power cannot be computed.

DESIGN

Depth is kept, not averaged: theta amplitude varies steeply through the
entorhinal layers, so collapsing the MEC groups blurs the structure worth
seeing. Power is computed per channel group and carries its layer, its
dorsoventral position, and its MEDIOLATERAL position -- the last because the
paper's central axis is mediolateral, and the cell-level analyses already show
medial MEC carrying more of the anchoring signal than lateral. coord_SCs_x is
stored negative, so ML is its magnitude, split at 3400 um to match the cell
analyses.

BOTH absolute and relative power are recorded. Relative power (band / 1-200 Hz
total) is comparable across sessions, since absolute microvolts depend on
electrode impedance and reference -- but it is COMPOSITIONAL: the bands sum to
one, so theta falling mechanically pushes the others up, and four "effects" can
be one. Absolute power breaks that coupling and is reported alongside, z-scored
within session so it can still be pooled.

SPEED MATCHING is the control that matters. Theta power scales steeply with
running speed, and the two anchoring states differ slightly in speed (35.0 vs
34.2 cm/s across the dataset). Without matching, a speed effect would arrive
wearing an anchoring label. Trials are matched by subsampling within speed
deciles so the two states have the same speed distribution, and every statistic
is computed before and after.

The trial is the independent unit throughout.
"""
import os, sys, json, glob, re, time, warnings
import h5py, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
from scipy.signal import welch
import pynapple as nap
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
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

# Prefer the local microSD copy when present: the lab mount has dropped mid-run
# repeatedly, and these analyses read hundreds of MB per session.
import os as _os
_SD = '/Volumes/datasetsSD/clark2025_lfp'
LFP_ROOT = '/Volumes/INCR-NolanLab/ActiveProjects/Wolf/harry/data/clark2025_lfp'
if _os.path.isdir(_SD):
    LFP_ROOT = _SD
OUT = '/Users/harryclark/Documents/spatial-manifolds/data/lfp'
os.makedirs(OUT, exist_ok=True)
FS = 1000.0
BANDS = [('delta', 1, 4), ('theta', 6, 10), ('beta', 10, 20),
         ('gamma', 30, 80), ('high', 80, 200)]
MIN_RUN_S = 1.0
ML_BOUND = 3400.0          # um; matches the cell-level medial/lateral split


def mec_groups(mo, dy, names):
    A = pd.read_csv(f'{LFP_ROOT}/M{mo}/D{dy}/anatomy_M{mo}_D{dy}.csv')
    reg = dict(zip(A.channel_id, A.brain_region.astype(str)))
    dep = dict(zip(A.channel_id, A.coord_SCs_y))
    mlx = dict(zip(A.channel_id, A.coord_SCs_x.abs()))    # stored negative
    rows = []
    for i, n in enumerate(names):
        chs = n.split('_')
        rs = [reg.get(c, '?') for c in chs]
        ent = sum(r.startswith('ENTm') for r in rs)
        if ent <= len(chs) / 2:
            continue
        lay = pd.Series([r for r in rs if r.startswith('ENTm')]).mode().iloc[0]
        ml = float(np.nanmean([mlx.get(c, np.nan) for c in chs]))
        rows.append(dict(g=i, layer=lay,
                         dv=float(np.nanmean([dep.get(c, np.nan) for c in chs])),
                         ml=ml, ml_side='medial' if ml < ML_BOUND else 'lateral'))
    return pd.DataFrame(rows).sort_values('dv')


def speed_matched(tr_speed, tr_state, rng, n_q=10):
    """Trial indices whose speed distribution is equal between the two states."""
    keep = []
    qs = np.quantile(tr_speed, np.linspace(0, 1, n_q + 1))
    qs[0] -= 1; qs[-1] += 1
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (tr_speed > lo) & (tr_speed <= hi)
        a = np.where(m & tr_state)[0]; b = np.where(m & ~tr_state)[0]
        n = min(len(a), len(b))
        if n:
            keep += list(rng.choice(a, n, replace=False))
            keep += list(rng.choice(b, n, replace=False))
    # dtype matters: an empty list gives a float64 array, and using it to index
    # raises "arrays used as indices must be of integer type". Single-state
    # sessions hit this constantly under the gated classifier -- 20 of 62
    # sessions are entirely one state -- where before the gate every session
    # looked mixed because 2-means always split.
    return np.array(sorted(keep), dtype=int)


def run_session(mo, dy, rng):
    lf = f'{LFP_ROOT}/M{mo}/D{dy}/VR/sub-M{mo}_ses-D{dy}_typ-VR_beh.nwb'
    ana = f'{LFP_ROOT}/M{mo}/D{dy}/anatomy_M{mo}_D{dy}.csv'
    if not (os.path.exists(lf) and os.path.exists(ana)):
        return None
    f = h5py.File(lf, 'r')
    if 'processing/ecephys/LFP/lfp/data' not in f:
        f.close(); return None
    names = [n.decode() for n in f['general/extracellular_ephys/electrodes/channel_name'][:]]
    M = mec_groups(mo, dy, names)
    if len(M) < 4:
        f.close(); return None

    bp, cp = vr_paths(mo, dy)
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials, orig = clip_trials(beh['trials'].as_dataframe(), clusters)
    # Canonical labels from the population-state build, not the eye table. The
    # eye table was only ever the state source because it happened to carry
    # frac_anch; this keeps the LFP analyses on the same labels as every figure
    # and stops them going stale unnoticed when the classifier changes.
    T = pd.read_csv('/Users/harryclark/Documents/spatial-manifolds/data/'
                    'population_state/anchoring_trials.csv')
    st = T[(T.mouse == mo) & (T.day == dy)]
    if not len(st):
        f.close(); return None
    state = dict(zip(st.trial.astype(int), st.frac_anch > 0.5))

    d = f['processing/ecephys/LFP/lfp/data']
    lfp_t = np.arange(d.shape[0]) / FS
    S = beh['S']
    spd = np.asarray(S.values)[np.searchsorted(np.asarray(S.index), lfp_t)
                               .clip(0, len(S) - 1)]
    run = spd >= 3.0
    cols = M.g.values
    read_cols = np.sort(cols)                       # h5py needs increasing indices
    back = np.array([int(np.where(read_cols == c)[0][0]) for c in cols])
    D = d[:, read_cols][:, back]
    f.close()

    rows = []
    for _, tr in trials.iterrows():
        u = int(tr.number)
        if u not in state:
            continue
        m = (lfp_t >= tr.start) & (lfp_t <= tr.end) & run
        if m.sum() < MIN_RUN_S * FS:
            continue
        fr, P = welch(D[m], fs=FS, nperseg=512, axis=0)
        tot = P[(fr >= 1) & (fr < 200)].sum(axis=0)
        rec = dict(mouse=mo, day=dy, trial=u, anch=bool(state[u]),
                   speed=float(spd[m].mean()), n_samp=int(m.sum()))
        for nm, lo, hi in BANDS:
            ab = P[(fr >= lo) & (fr < hi)].sum(axis=0)
            for j in range(len(cols)):
                rows.append(dict(**rec, band=nm, g=int(cols[j]),
                                 layer=M.layer.values[j], dv=M.dv.values[j],
                                 ml=M.ml.values[j], ml_side=M.ml_side.values[j],
                                 abs_p=float(ab[j]), rel=float(ab[j] / tot[j])))
    R = pd.DataFrame(rows)
    if not len(R):
        return None
    # z-score absolute power within session and channel group, so it can be
    # pooled despite impedance and reference differing between sessions
    R['z_abs'] = R.groupby(['band', 'g']).abs_p.transform(
        lambda v: (v - v.mean()) / v.std() if v.std() > 0 else np.nan)
    tinfo = R.drop_duplicates('trial')[['trial', 'anch', 'speed']].reset_index(drop=True)
    km = speed_matched(tinfo.speed.values, tinfo.anch.values, rng)
    R['speed_matched'] = R.trial.isin(tinfo.trial.values[km])
    return R


if __name__ == '__main__':
    only = sys.argv[1] if len(sys.argv) > 1 else None
    T = pd.read_csv('/Users/harryclark/Documents/spatial-manifolds/data/'
                    'population_state/anchoring_trials.csv')
    sess = sorted(set(zip(T.mouse, T.day)))
    print(f'{len(sess)} sessions in the ENTm eye table')
    out, t0 = [], time.time()
    for k, (mo, dy) in enumerate(sess, 1):
        if only and f'M{mo}D{dy}' != only:
            continue
        rng = np.random.default_rng(abs(hash((mo, dy))) % 2**32)
        try:
            R = run_session(int(mo), int(dy), rng)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        if R is None:
            continue
        out.append(R)
        n = R.trial.nunique()
        sides = R.drop_duplicates('g').ml_side.value_counts().to_dict()
        print(f'[{k}/{len(sess)}] M{mo}D{dy}: {n} trials, {R.g.nunique()} MEC groups '
              f'{sides}, {R[R.speed_matched].trial.nunique()} speed-matched, '
              f'{time.time()-t0:.0f}s', flush=True)
    A = pd.concat(out, ignore_index=True)
    # a single-session run must not clobber the full table: that has already
    # happened once in this project and cost a 15-minute rerun
    p = (f'{OUT}/lfp_power_by_anchoring.csv' if only is None
         else f'{OUT}/lfp_power_by_anchoring_{only}.csv')
    A.to_csv(p, index=False)
    print(f'\n{A.trial.nunique()} trials from {A.groupby(["mouse","day"]).ngroups} '
          f'sessions -> {p}')
