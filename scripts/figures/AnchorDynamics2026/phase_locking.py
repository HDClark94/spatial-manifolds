"""Spike-theta phase locking by anchoring state.

The LFP decouples from running speed when the population is anchored while the
cells' rate coding of speed does not change, which says the state alters how
strongly population activity is ENTRAINED to theta rather than what cells
encode. That word makes a prediction this tests directly: spike-theta phase
locking should weaken when anchored, and its speed dependence should weaken
with it. If locking is unchanged, "entrainment" is the wrong description and the
effect lives in synaptic currents rather than spike timing.

Each cluster is assigned the theta phase of its OWN extremum channel's group
(`extremum_channel` gives 'CH244', group names list their eight contacts), not a
session average: theta phase rotates through the entorhinal layers, so averaging
phase across groups would destroy exactly the quantity being measured.

TWO CONFOUNDS, both handled by one matching step.

  Spike count. The mean resultant length is biased upward at small n, and
  anchored firing rates are ~7% lower, which alone would make anchored cells
  look MORE locked. Counts are equalised between states.

  Speed. Locking depends on running speed, and the states differ in their speed
  distribution. Spikes are matched across speed deciles before counts are
  equalised.

MRL is then averaged over NDRAW random subsamples so the estimate does not ride
on one draw. The speed dependence is measured as MRL in the top speed tercile
minus the bottom, computed within the matched set with counts equalised in each
tercile, so it carries the same protections.
"""
import os, sys, json, time, warnings
import h5py, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import pynapple as nap
from scipy.stats import wilcoxon
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/'
                   'AnchorDynamics2026')
from lfp_batch import LFP_ROOT, FS, vr_paths, clip_trials

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

OUT = '/Users/harryclark/Documents/spatial-manifolds/data/lfp'
RUN_SPEED = 3.0
MINSPK = 300          # per state, after matching
NDRAW = 20


def mrl(ph):
    return float(np.abs(np.mean(np.exp(1j * ph))))


def matched_mrl(pa, pn, rng, ndraw=NDRAW):
    """MRL for each state at equal spike count, averaged over draws."""
    n = min(len(pa), len(pn))
    if n < MINSPK:
        return np.nan, np.nan, 0
    va = np.mean([mrl(rng.choice(pa, n, replace=False)) for _ in range(ndraw)])
    vn = np.mean([mrl(rng.choice(pn, n, replace=False)) for _ in range(ndraw)])
    return va, vn, n


def speed_match(sa, sn, rng, nq=10):
    """Indices into each state's spikes with matched speed distributions."""
    q = np.quantile(np.r_[sa, sn], np.linspace(0, 1, nq + 1))
    q[0] -= 1; q[-1] += 1
    ka, kn = [], []
    for lo, hi in zip(q[:-1], q[1:]):
        ia = np.where((sa >= lo) & (sa < hi))[0]
        iN = np.where((sn >= lo) & (sn < hi))[0]
        k = min(len(ia), len(iN))
        if k:
            ka.append(rng.choice(ia, k, replace=False))
            kn.append(rng.choice(iN, k, replace=False))
    if not ka:
        return np.array([], int), np.array([], int)
    return np.concatenate(ka), np.concatenate(kn)


def run_session(mo, dy, state, rng):
    lf = f'{LFP_ROOT}/M{mo}/D{dy}/VR/sub-M{mo}_ses-D{dy}_typ-VR_beh.nwb'
    if not os.path.exists(lf):
        return None
    f = h5py.File(lf, 'r')
    if 'processing/ecephys/Processed/theta/data' not in f:
        f.close(); return None
    names = [n.decode() for n in f['general/extracellular_ephys/electrodes/channel_name'][:]]
    ch2g = {c: i for i, n in enumerate(names) for c in n.split('_')}

    bp, cp = vr_paths(mo, dy)
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials, orig = clip_trials(beh['trials'].as_dataframe(), clusters)
    ids = spatial_cell_ids(mo, dy, clusters, classes=('GC', 'NGS', 'NS'))
    if len(ids) < 5:
        f.close(); return None
    ec = clusters.metadata['extremum_channel']
    grp = {c: ch2g.get(str(ec[c]), -1) for c in ids}
    use = [c for c in ids if grp[c] >= 0]
    if not use:
        f.close(); return None

    cols = sorted(set(grp[c] for c in use))
    back = {g: i for i, g in enumerate(cols)}
    PH = f['processing/ecephys/Processed/theta/data'][:, cols].astype(np.float32)
    f.close()

    n = PH.shape[0]
    t = np.arange(n) / FS
    S = beh['S']
    spd = np.asarray(S.values)[np.searchsorted(np.asarray(S.index), t).clip(0, len(S) - 1)]
    starts = trials.start.values.astype(float)
    ends = trials.end.values.astype(float)
    nums = trials.number.values.astype(int)
    k = np.searchsorted(starts, t) - 1
    inside = (k >= 0) & (k < len(starts))
    k = k.clip(0, len(starts) - 1)
    inside &= t <= ends[k]
    lab = np.where(inside, [state.get(int(u), np.nan) for u in nums[k]], np.nan)
    ok = (spd >= RUN_SPEED) & np.isfinite(lab)

    rows = []
    for c in use:
        sp = np.asarray(clusters[c].index, dtype=float)
        i = np.round(sp * FS).astype(int)
        i = i[(i >= 0) & (i < n)]
        i = i[ok[i]]
        if len(i) < 2 * MINSPK:
            continue
        ph = PH[i, back[grp[c]]].astype(float)
        sv, st = spd[i], lab[i] > .5
        pa, pn = ph[st], ph[~st]
        sa, sn = sv[st], sv[~st]
        ia, iN = speed_match(sa, sn, rng)
        if len(ia) < MINSPK or len(iN) < MINSPK:
            continue
        pa, pn, sa, sn = pa[ia], pn[iN], sa[ia], sn[iN]
        va, vn, nm = matched_mrl(pa, pn, rng)
        d = dict(mouse=mo, day=dy, cluster=int(c), layer='', n_matched=nm,
                 mrl_a=va, mrl_n=vn,
                 phase_a=float(np.angle(np.mean(np.exp(1j * pa)))),
                 phase_n=float(np.angle(np.mean(np.exp(1j * pn)))),
                 rate_a=float(len(ph[st])), rate_n=float(len(ph[~st])))
        # speed dependence: top minus bottom tercile, counts equalised inside each
        q = np.quantile(np.r_[sa, sn], [1 / 3, 2 / 3])
        for tag, lo, hi in (('lo', -np.inf, q[0]), ('hi', q[1], np.inf)):
            ma, mn = (sa >= lo) & (sa < hi), (sn >= lo) & (sn < hi)
            if ma.sum() >= MINSPK and mn.sum() >= MINSPK:
                xa, xn, _ = matched_mrl(pa[ma], pn[mn], rng, ndraw=10)
                d[f'mrl_{tag}_a'], d[f'mrl_{tag}_n'] = xa, xn
        rows.append(d)
    return pd.DataFrame(rows)


def one_cell(mo, dy, state, cl, rng):
    """The matched spike phases of a single cluster, for plotting."""
    lf = f'{LFP_ROOT}/M{mo}/D{dy}/VR/sub-M{mo}_ses-D{dy}_typ-VR_beh.nwb'
    f = h5py.File(lf, 'r')
    names = [n.decode() for n in f['general/extracellular_ephys/electrodes/channel_name'][:]]
    ch2g = {c: i for i, n in enumerate(names) for c in n.split('_')}
    bp, cp = vr_paths(mo, dy)
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials, orig = clip_trials(beh['trials'].as_dataframe(), clusters)
    g = ch2g[str(clusters.metadata['extremum_channel'][cl])]
    PH = f['processing/ecephys/Processed/theta/data'][:, g].astype(np.float32)
    f.close()
    n = len(PH)
    t = np.arange(n) / FS
    S = beh['S']
    spd = np.asarray(S.values)[np.searchsorted(np.asarray(S.index), t).clip(0, len(S) - 1)]
    starts, ends = trials.start.values.astype(float), trials.end.values.astype(float)
    nums = trials.number.values.astype(int)
    k = np.searchsorted(starts, t) - 1
    inside = (k >= 0) & (k < len(starts))
    k = k.clip(0, len(starts) - 1)
    inside &= t <= ends[k]
    lab = np.where(inside, [state.get(int(u), np.nan) for u in nums[k]], np.nan)
    ok = (spd >= RUN_SPEED) & np.isfinite(lab)
    i = np.round(np.asarray(clusters[cl].index, dtype=float) * FS).astype(int)
    i = i[(i >= 0) & (i < n)]
    i = i[ok[i]]
    ph, sv, stt = PH[i].astype(float), spd[i], lab[i] > .5
    ia, iN = speed_match(sv[stt], sv[~stt], rng)
    pa, pn = ph[stt][ia], ph[~stt][iN]
    m = min(len(pa), len(pn))                  # equal counts, as in the batch
    return rng.choice(pa, m, replace=False), rng.choice(pn, m, replace=False)


if __name__ == '__main__':
    T = pd.read_csv(f'{OUT}/theta_frequency.csv')
    extra = f'{OUT}/theta_frequency_extra.csv'
    if os.path.exists(extra):
        T = pd.concat([T, pd.read_csv(extra)], ignore_index=True)
    sess = sorted(set(zip(T.mouse, T.day)))
    out, t0 = [], time.time()
    for i, (mo, dy) in enumerate(sess, 1):
        st = T[(T.mouse == mo) & (T.day == dy)]
        rng = np.random.default_rng(abs(hash((mo, dy))) % 2**32)
        try:
            R = run_session(int(mo), int(dy),
                            dict(zip(st.trial.astype(int), st.anch.astype(bool))), rng)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        if R is None or not len(R):
            continue
        out.append(R)
        print(f'[{i}/{len(sess)}] M{mo}D{dy}: {len(R)} cells, {time.time()-t0:.0f}s',
              flush=True)
    A = pd.concat(out, ignore_index=True)
    A.to_csv(f'{OUT}/phase_locking.csv', index=False)

    print(f'\n{len(A)} cells, {A.groupby(["mouse","day"]).ngroups} sessions, '
          f'{A.n_matched.median():.0f} matched spikes per state (median)')
    g = A.groupby(['mouse', 'day'])[['mrl_a', 'mrl_n']].median().dropna()
    print(f'  MRL (count- and speed-matched)  anchored {g.mrl_a.median():.4f}  '
          f'non {g.mrl_n.median():.4f}  '
          f'({100*(g.mrl_a.median()-g.mrl_n.median())/g.mrl_n.median():+.1f}%)  '
          f'p = {wilcoxon(g.mrl_a, g.mrl_n).pvalue:.3g}  '
          f'({(g.mrl_a < g.mrl_n).sum()}/{len(g)} lower when anchored)')
    D = A.dropna(subset=['mrl_lo_a', 'mrl_hi_a', 'mrl_lo_n', 'mrl_hi_n']).copy()
    D['dep_a'] = D.mrl_hi_a - D.mrl_lo_a
    D['dep_n'] = D.mrl_hi_n - D.mrl_lo_n
    gd = D.groupby(['mouse', 'day'])[['dep_a', 'dep_n']].median().dropna()
    print(f'  speed dependence (hi - lo MRL)  anchored {gd.dep_a.median():+.4f}  '
          f'non {gd.dep_n.median():+.4f}  p = {wilcoxon(gd.dep_a, gd.dep_n).pvalue:.3g}  '
          f'({len(gd)} sessions, {len(D)} cells)')
    ph = A.dropna(subset=['phase_a', 'phase_n'])
    sh = np.angle(np.mean(np.exp(1j * (ph.phase_a - ph.phase_n))))
    print(f'  preferred phase shift (anchored - non): {np.degrees(sh):+.1f} deg')
