"""Unfiltered anchoring labels, whole-track and from segments before the reward
zone, with per-trial behaviour, for the test of whether a region's state predicts behaviour or only
maps it (fig1_supp_neighbour_test.py).

Usage:  python3 build_neighbour_labels.py [worker n_workers]
A session file that exists but lacks a segment gets only the missing segments.

Two departures from build_anchoring_labels.py, both needed by that test:

  - NO MEDIAN FILTER. The canonical labels are median-filtered over 5 trials, so
    a trial's label carries its neighbours' firing. A test that predicts one
    trial's outcome from the state on OTHER trials would then see the outcome
    trial's own firing through the filter.
  - SEGMENT VERSIONS. Labels classified from part of the track only, before
    the reward zone (90-110 cm of the 200 cm track). On uncued trials, hits and
    misses differ in speed from ~50 cm on, where the mouse starts slowing for
    the zone, so these labels cannot see the slowing, stopping and licking that
    separate a hit from a miss. Positions below 0 are the end of the previous
    trial, which runs on into this one through the black box (170-200 cm, then
    0-30 cm of the next trial). SEGMENTS gives the variants: pre (0-50 cm),
    pre2 (-30 to 50), bb (the black box, -30 to 30) and bb0 (0-30, this
    trial's own part of the black box).
  - A TASK-REGION VERSION, tz (50-110 cm): from where hits start slowing for the
    reward zone to its end, where hits and misses differ most. Paired with bb it is the
    control: two classifiers built the same way, one blind to the task and one
    centred on it.

Cells are the current curated cells (GC/NGS/NS in cell_classifications_v2.csv,
so after the ISI exclusion), every region. The whole-track gate threshold is
taken from the canonical label file where the cell is there (same profiles, so
the same null); otherwise, and for every pre-zone label, the cell's own null is
computed here. The shared null cache is not written.

Writes data/population_state/neighbour_test/M{mouse}D{day}.npz:
    cluster_id, region          (n_cells,)
    labels_full, labels_<seg>   (n_cells, n_trials) 1 / 0 / NaN, unfiltered
    trial                       (n_trials,) renumbered as in the canonical labels
    ttype, hit                  (n_trials,) 'b' cued / 'nb' uncued; 1 hit
    speed                       mean running speed (>= 3 cm/s) over the trial
    speed_pre, speed_rz         mean speed over 0-50 cm and over the zone
    first_stop                  position of the first stop after the black box
"""
import json, os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026')
import pynapple as nap
from spatial_manifolds.anchoring import (trial_cluster_labels, null_mean_corr,
                                         load_session_labels, smooth_nanaware)
from of_xgb_decomp import fam

NB_PATH = ('/Users/harryclark/Documents/spatial-manifolds/scripts/figures/'
           'AnchorDynamics2026/lick_raster_by_trial_type.ipynb')
_g = {}
for _c in json.load(open(NB_PATH))['cells']:
    if _c['cell_type'] != 'code':
        continue
    _s = ''.join(_c['source'])
    if _s.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in _s:
        continue
    exec(compile(_s, '<nb>', 'exec'), _g)
globals().update({k: v for k, v in _g.items() if not k.startswith('__')})

PS = '/Users/harryclark/Documents/spatial-manifolds/data/population_state'
OUT = f'{PS}/neighbour_test'
os.makedirs(OUT, exist_ok=True)
PRE_CM = 50.0                       # pre-zone segment 0-50 cm: before hits and misses diverge
SEGMENTS = {'pre': (0, 50), 'pre2': (-30, 50), 'bb': (-30, 30), 'bb0': (0, 30),
            'tz': (50, 110)}                                               # cm


def segment(flat, n_tr, lo, hi):
    """(n_tr, bins) maps of lo-hi cm relative to each trial's start, from the
    session's continuous position map; bins before the first trial are NaN."""
    a, b = int(round(lo / BS)), int(round(hi / BS))
    pad = np.full(max(0, -a), np.nan)
    f = np.r_[pad, flat]
    return np.array([f[i * NBIN + a + len(pad): i * NBIN + b + len(pad)] for i in range(n_tr)])


def labels(M, thr=None):
    """Unfiltered labels for one cell's trial x bin map; thr = gate threshold."""
    if thr is None or not np.isfinite(thr):
        sm = np.array([smooth_nanaware(r) for r in np.asarray(M, float)])
        v = sm.sum(axis=1) > 1e-6
        if v.sum() < 6:
            return np.full(len(M), np.nan)
        thr = null_mean_corr(sm[v], use_cache=False)
    return trial_cluster_labels(M, medfilt_size=0, null_thr=thr)[0]


def behaviour(beh, trials):
    """Per-trial speed summaries and first stop, from the speed and position series."""
    S, P = beh['S'], beh['P']
    ts, sv = np.asarray(S.index), np.asarray(S.values, float)
    pv = np.asarray(P.values, float)[np.searchsorted(np.asarray(P.index), ts).clip(0, len(P) - 1)]
    out = []
    for _, t in trials.iterrows():
        m = (ts >= t.start) & (ts < t.end)
        s, p = sv[m], pv[m]
        run = s >= 3.0
        pre = p < PRE_CM
        rz = (p >= RZ[0]) & (p < RZ[1])
        st = np.where((s < STOP_SPEED) & (p >= BLACK_BOX) & (p < TL - BLACK_BOX))[0]
        out.append(dict(speed=s[run].mean() if run.sum() > 10 else np.nan,
                        speed_pre=s[pre].mean() if pre.sum() > 3 else np.nan,
                        speed_rz=s[rz].mean() if rz.sum() > 3 else np.nan,
                        first_stop=p[st[0]] if len(st) else np.nan))
    return pd.DataFrame(out)


def build(mouse, day):
    z = load_session_labels(mouse, day)
    bp, cp = vr_paths(mouse, day)
    if z is None or not (os.path.exists(bp) and os.path.exists(cp)):
        return None
    path = f'{OUT}/M{mouse}D{day}.npz'
    old = dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else None
    todo = [k for k in SEGMENTS if old is None or f'labels_{k}' not in old]
    if not todo:
        return 0
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials_all = beh['trials'].as_dataframe()
    trials, orig = clip_trials(trials_all, clusters)
    keep = np.isin(trials_all.number.values.astype(int), orig)
    assert np.array_equal(trials.number.values.astype(int), z['trial'].astype(int))
    tn, trav = beh['trial_number'], beh['travel']
    dt = trav - (float(np.asarray(tn.values)[0]) - 1) * TL
    moving = beh['S'].threshold(3.0, method='above').time_support
    n_tr = len(trials_all)
    ids = (list(old['cluster_id']) if old is not None else
           spatial_cell_ids(mouse, day, clusters, region=None, classes=('GC', 'NGS', 'NS')))
    if not ids:
        return None
    tc = nap.compute_1d_tuning_curves(clusters[ids], dt, nb_bins=n_tr * NBIN,
                                      minmax=[0, n_tr * TL], ep=moving)
    out = old if old is not None else {}
    if old is None:
        reg = CLS[(CLS.mouse == mouse) & (CLS.day == day)].set_index('cluster_id').brain_region
        thr0 = dict(zip(z['cluster_id'].astype(int), z['null_thr']))
        out['labels_full'] = np.array([
            labels(np.asarray(tc[c]).reshape(n_tr, NBIN)[keep], thr0.get(int(c))) for c in ids])
        out['region'] = np.array([fam(reg.get(int(c))) or '' for c in ids])
        out['cluster_id'] = np.array(ids, int)
        out['trial'] = trials.number.values.astype(int)
        out['ttype'] = trials.type.values.astype(str)
        out['hit'] = (trials.performance.values == 'hit').astype(int)
        out.update({k: v.values for k, v in behaviour(beh, trials).items()})
    for k in todo:
        lo, hi = SEGMENTS[k]
        out[f'labels_{k}'] = np.array([
            labels(segment(np.asarray(tc[c]), n_tr, lo, hi)[keep]) for c in ids])
    np.savez_compressed(path, **out)
    return len(ids)


if __name__ == '__main__':
    w, nw = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    sess = sorted({(int(f[1:3]), int(f.split('D')[1].split('.')[0]))
                   for f in os.listdir(f'{PS}/labels') if f.endswith('.npz')})
    mine = [s for i, s in enumerate(sess) if i % nw == w]
    t0 = time.time()
    for k, (mo, dy) in enumerate(mine, 1):
        try:
            n = build(mo, dy)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        print(f'[w{w} {k}/{len(mine)}] M{mo}D{dy}: {n} cells, {time.time()-t0:.0f}s', flush=True)
