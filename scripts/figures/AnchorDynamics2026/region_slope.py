"""Is the speed-theta decoupling specific to MEC, and does it have a laminar profile?

Three mechanistic accounts of the decoupling have failed (velocity gain,
prediction error, entrainment). What survives is that LFP theta amplitude and
its scaling with speed both fall while the spiking does not, which points at the
magnitude of speed-scaled synaptic drive reaching MEC. That cannot be confirmed
with this dataset -- the prime suspect, the septal/supramammillary input, is not
recorded -- but it can be narrowed, and it can be falsified.

TEST 1, REGIONAL SPECIFICITY. The probes pass through visual cortex, the
pre/para/postsubicular complex and cerebellum on their way to MEC, and every
channel group carries an anatomical assignment. The speed->theta slope is
therefore computed for EVERY group, MEC or not, with the anchoring state still
defined by MEC cells. If the decoupling is confined to entorhinal groups, a
localised change in drive to MEC survives. If visual cortex and cerebellum show
it equally, the account is dead and this is a global brain state.

  Note the asymmetry when reading the result. Theta recorded near MEC can be
  partly MEC theta by volume conduction, so a SHARED effect is ambiguous while a
  DIVERGENT one is informative. A negative result here is much more conclusive
  than a positive one.

TEST 2, LAMINAR PROFILE. Same extraction, no extra cost. Entorhinal afferents
are layer-specific, so a change concentrated in one lamina narrows the candidate
pathway while a flat profile argues against pathway-specific drive. This is a
different quantity from the theta AMPLITUDE effect already known to peak in
ENTm1, and the two need not co-localise.

MEASURE. Per group and state, theta amplitude per trial (RMS of the 6-10 Hz
signal over running samples) regressed on trial mean speed, on the states'
common 5-95% speed support. The normalised slope (slope / mean amplitude) is
primary because absolute LFP amplitude varies by an order of magnitude across
regions and depths, which would otherwise dominate any cross-region comparison.
"""
import os, sys, time, warnings
import h5py, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from scipy.signal import butter, filtfilt
from scipy.stats import wilcoxon
import pynapple as nap
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/'
                   'AnchorDynamics2026')
from lfp_batch import LFP_ROOT, FS, vr_paths, clip_trials

# Read from the local microSD copy when it exists. Three attempts to run this
# straight off the lab volume were each killed partway through by the mount
# dropping (25, 11 and 14 of 59 sessions); this analysis reads all 48 channel
# groups per session, several times the I/O of the other LFP work here.
# Behaviour and clusters still come from the local clark2025 copy via vr_paths.
SD_ROOT = '/Volumes/datasetsSD/clark2025_lfp'
if os.path.isdir(SD_ROOT):
    LFP_ROOT = SD_ROOT
    print(f'reading LFP from {LFP_ROOT}', flush=True)

OUT = '/Users/harryclark/Documents/spatial-manifolds/data/lfp'
BP = butter(3, [6 / (FS / 2), 10 / (FS / 2)], btype='band')
CHUNK = 8           # groups filtered at a time, to bound memory
MINTR = 15

# Families from the region strings that actually occur in the anatomy files.
# 'root', white matter (or, alv, dhc, fp, arb, mcp) and anything unmatched fall
# through to None and are dropped rather than being silently lumped together.
CEREB = ('SIM', 'CUL', 'CENT', 'AN', 'PRM', 'COPY', 'PFL', 'FL', 'DEC', 'FOTU',
         'PYR', 'UVU', 'NOD', 'CB')


def family(r):
    r = str(r)
    if r.startswith('ENTm'):
        return 'MEC'
    if r.startswith('ENTl'):
        return 'LEC'
    if r.startswith(('PRE', 'POST', 'PAR', 'SUB')):
        return 'Pre/para/post/sub'
    if r.startswith('VIS'):
        return 'Visual'
    if r.startswith(CEREB):
        return 'Cerebellum'
    return None


def run_session(mo, dy, state):
    lf = f'{LFP_ROOT}/M{mo}/D{dy}/VR/sub-M{mo}_ses-D{dy}_typ-VR_beh.nwb'
    an = f'{LFP_ROOT}/M{mo}/D{dy}/anatomy_M{mo}_D{dy}.csv'
    if not (os.path.exists(lf) and os.path.exists(an)):
        return None
    A = pd.read_csv(an)
    reg = dict(zip(A.channel_id, A.brain_region.astype(str)))
    dep = dict(zip(A.channel_id, A.coord_SCs_y))

    f = h5py.File(lf, 'r')
    if 'processing/ecephys/LFP/lfp/data' not in f:
        f.close(); return None
    names = [n.decode() for n in f['general/extracellular_ephys/electrodes/channel_name'][:]]
    meta = []
    for i, n in enumerate(names):
        chs = n.split('_')
        fams = [family(reg.get(c, '?')) for c in chs]
        fams = [x for x in fams if x]
        if not fams or len(fams) <= len(chs) / 2:
            continue                       # group not majority-assigned to one family
        fam = pd.Series(fams).mode().iloc[0]
        if sum(x == fam for x in fams) <= len(chs) / 2:
            continue
        lays = [reg.get(c, '?') for c in chs if family(reg.get(c, '?')) == fam]
        meta.append(dict(g=i, family=fam, layer=pd.Series(lays).mode().iloc[0],
                         dv=float(np.nanmean([dep.get(c, np.nan) for c in chs]))))
    if not meta:
        f.close(); return None
    M = pd.DataFrame(meta)

    bp, cp = vr_paths(mo, dy)
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials, _ = clip_trials(beh['trials'].as_dataframe(), clusters)
    n_samp = f['processing/ecephys/LFP/lfp/data'].shape[0]
    t = np.arange(n_samp) / FS
    S = beh['S']
    spd = np.asarray(S.values)[np.searchsorted(np.asarray(S.index), t).clip(0, len(S) - 1)]
    run = spd >= 3.0

    keep = [(int(tr.number), (t >= tr.start) & (t <= tr.end) & run)
            for _, tr in trials.iterrows() if int(tr.number) in state]
    keep = [(u, m) for u, m in keep if m.sum() >= FS]
    if len(keep) < 2 * MINTR:
        f.close(); return None
    tr_spd = np.array([spd[m].mean() for _, m in keep])
    tr_anch = np.array([bool(state[u]) for u, _ in keep])
    if tr_anch.sum() < MINTR or (~tr_anch).sum() < MINTR:
        f.close(); return None
    # common speed support, computed once: speed is per trial, so the same trials
    # are kept for every group
    lo = max(np.quantile(tr_spd[tr_anch], .05), np.quantile(tr_spd[~tr_anch], .05))
    hi = min(np.quantile(tr_spd[tr_anch], .95), np.quantile(tr_spd[~tr_anch], .95))
    cs = (tr_spd >= lo) & (tr_spd <= hi)

    rows = []
    gs = M.g.values
    for c0 in range(0, len(gs), CHUNK):
        cols = gs[c0:c0 + CHUNK]
        L = f['processing/ecephys/LFP/lfp/data'][:, cols].astype(np.float32)
        F = filtfilt(*BP, L.astype(np.float64), axis=0)
        del L
        for gi, g in enumerate(cols):
            amp = np.array([np.sqrt(np.mean(F[m, gi] ** 2)) for _, m in keep])
            d = M[M.g == g].iloc[0].to_dict()
            d.update(mouse=mo, day=dy, n_tr=int(len(keep)))
            ok = True
            for sel, tag in ((tr_anch & cs, 'a'), (~tr_anch & cs, 'n')):
                if sel.sum() < MINTR:
                    ok = False; break
                b = np.polyfit(tr_spd[sel], amp[sel], 1)[0]
                d[f'slope_{tag}'] = b
                d[f'amp_{tag}'] = float(amp[sel].mean())
                d[f'nslope_{tag}'] = b / amp[sel].mean()
            if ok:
                rows.append(d)
        del F
    f.close()
    return pd.DataFrame(rows)


if __name__ == '__main__':
    T = pd.read_csv(f'{OUT}/theta_frequency.csv')
    extra = f'{OUT}/theta_frequency_extra.csv'
    if os.path.exists(extra):
        T = pd.concat([T, pd.read_csv(extra)], ignore_index=True)
    sess = sorted(set(zip(T.mouse, T.day)))
    # Resume: the lab volume has dropped mid-run three times. Sessions already in
    # the CSV are skipped, so repeated drops accumulate progress instead of
    # restarting from nothing.
    out, done = [], set()
    csv = f'{OUT}/region_speed_slope.csv'
    if os.path.exists(csv):
        prev = pd.read_csv(csv)
        out.append(prev)
        done = set(map(tuple, prev[['mouse', 'day']].drop_duplicates().values))
        print(f'resuming: {len(done)} sessions already done', flush=True)
    t0 = time.time()
    for i, (mo, dy) in enumerate(sess, 1):
        if (mo, dy) in done:
            continue
        st = T[(T.mouse == mo) & (T.day == dy)]
        try:
            R = run_session(int(mo), int(dy),
                            dict(zip(st.trial.astype(int), st.anch.astype(bool))))
        except OSError as e:               # volume dropped: wait for it to return
            print(f'  ! M{mo}D{dy}: {type(e).__name__} -- waiting for volume',
                  flush=True)
            for _ in range(60):
                time.sleep(30)
                if os.path.exists(LFP_ROOT):
                    break
            else:
                # ABORT rather than continue. Carrying on silently consumes the
                # rest of the session list and prints a summary over whichever
                # arbitrary subset the mount allowed -- which is exactly how two
                # earlier runs produced plausible-looking but meaningless numbers
                # (MEC -39.5% p=0.29 on 24 sessions, +4.8% p=0.85 on 10).
                raise SystemExit(f'volume gone; stopped after {len(done)} sessions '
                                 f'-- rerun to resume from the saved CSV')
            continue
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        if R is None or not len(R):
            continue
        out.append(R)
        # save after every session: the lab volume has dropped mid-run twice, and
        # an OSError halfway through otherwise costs the whole batch
        pd.concat(out, ignore_index=True).to_csv(csv, index=False)
        print(f'[{i}/{len(sess)}] M{mo}D{dy}: {len(R)} groups, '
              f'{R.family.value_counts().to_dict()}, {time.time()-t0:.0f}s',
              flush=True)
    A = pd.concat(out, ignore_index=True)
    A.to_csv(csv, index=False)
    A['d'] = A.nslope_a - A.nslope_n
    n_done = A.groupby(['mouse', 'day']).ngroups
    # denominator is the sessions actually PRESENT (the SD copy holds only those
    # with non-MEC coverage), not every session in theta_frequency.csv
    avail = sum(os.path.exists(f'{LFP_ROOT}/M{mo}/D{dy}/VR/'
                               f'sub-M{mo}_ses-D{dy}_typ-VR_beh.nwb')
                for mo, dy in sess)
    print(f'\n{len(A)} groups, {n_done} of {avail} available sessions '
          f'({len(sess)} in the trial table)')
    if n_done < .8 * avail:
        print('  INCOMPLETE -- summary below is over a partial, non-random subset '
              'of sessions and should not be interpreted. Rerun to resume.')
    print('\n--- TEST 1: normalised speed->theta slope by region (session medians)')
    for fam, d in A.groupby('family'):
        g = d.groupby(['mouse', 'day'])[['nslope_a', 'nslope_n']].median().dropna()
        if len(g) < 5:
            print(f'  {fam:20s} only {len(g)} sessions, skipped'); continue
        p = wilcoxon(g.nslope_a, g.nslope_n).pvalue
        ch = 100 * (g.nslope_a.median() - g.nslope_n.median()) / abs(g.nslope_n.median())
        r = d.groupby(['mouse', 'day'])[['slope_a', 'slope_n']].median().dropna()
        pr = wilcoxon(r.slope_a, r.slope_n).pvalue
        print(f'  {fam:20s} nslope anchored {g.nslope_a.median():+.5f}  '
              f'non {g.nslope_n.median():+.5f}  ({ch:+6.1f}%)  p = {p:.3g}   '
              f'[raw {r.slope_a.median():+.4f} vs {r.slope_n.median():+.4f}, '
              f'p = {pr:.3g}]  n = {len(g)} sessions, {len(d)} groups')

    print('\n--- TEST 2: MEC laminar profile of the change (nslope anchored - non)')
    E = A[A.family == 'MEC']
    for lay, d in E.groupby('layer'):
        g = d.groupby(['mouse', 'day'])['d'].median().dropna()
        if len(g) < 5:
            continue
        print(f'  {lay:8s} {g.median():+.5f}  p = {wilcoxon(g).pvalue:.3g}  '
              f'({len(g)} sessions, {len(d)} groups)')
    from scipy.stats import kruskal
    # one value per session per layer: channel groups within a session share a
    # behavioural state and a probe, so a per-group test counts the same session
    # dozens of times (it gave p = 1e-17 where the session-level test is null)
    med = E.groupby(['mouse', 'day', 'layer']).d.median().reset_index()
    grp = [g.d.values for _, g in med.groupby('layer') if len(g) >= 5]
    if len(grp) > 2:
        print(f'  Kruskal-Wallis across layers (session medians): '
              f'p = {kruskal(*grp).pvalue:.3g}')
