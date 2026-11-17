"""Does a region's anchoring state predict behaviour, or only map it?

Reads the unfiltered labels of build_neighbour_labels.py; writes
data/population_state/neighbour_test.csv (one row per session x region),
neighbour_persistence.csv (state autocorrelation by lag) and
neighbour_sessions.csv (per session and structure, the black-box and
task-region states' correlation with the whole-track state, with the session's
persistence and, for MEC, grid-cell count). Figure:
fig1_supp_neighbour_test.py.

Per session and region, the state on a trial is its z-scored PC1 score on the
region's own cells x trials labels for that stretch (STATE; MIN_CELLS cells). Three measures, on uncued trials:

  same-trial AUC   the trial's own state against its outcome, from whole-track
                   labels and from each segment of build_neighbour_labels.py:
                   pre (0-50 cm), pre2 (-30 to 50), bb (black box, -30 to 30),
                   bb0 (0-30) and tz (the task region, 50-110 cm)
  neighbour AUC    the state on the nearest PRECEDING cued trial (within NEAR
                   trials) against the uncued trial's outcome, so that nothing
                   from the uncued trial or after it enters. On cued trials the
                   cue sets where the mouse stops, so a label that only records
                   stopping barely varies there; a state that lasts several
                   trials does. (Until 2026-10-31 the nearest cued trial on
                   each side was averaged; the user asked for the past only.)
  residual AUC     the same after regressing out of the neighbour state what the
                   neighbours themselves did: whether they were hits, and their
                   speed-by-position profile (the first N_PC principal
                   components of the 10 cm binned profile, over the session's
                   trials), so slow drift in engagement or in how the mouse
                   runs cannot carry it
  beyond MEC       for the other regions, the residual AUC with the MEC state on
                   the same neighbours also regressed out

Each is scored against its own null: the predictor circularly shifted against
the outcomes (N_SHIFT shifts of at least SHIFT_MIN uncued trials), excess =
AUC - mean null AUC. Signed against signed: both sides are AUCs.
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

PS = '/Users/harryclark/Documents/spatial-manifolds/data/population_state'
NT = f'{PS}/neighbour_test'
REGIONS = ['MEC', 'VIS', 'SUB', 'CB']
MIN_CELLS = {'MEC': 15, 'VIS': 15, 'SUB': 15, 'CB': 10}    # as fig1_supp_nonmec.py
NEAR = 3                        # the cued neighbour must lie within this many trials before
MIN_OUT = 5                     # hits and misses each, among usable uncued trials
N_SHIFT, SHIFT_MIN = 200, 5
COV = ['hit', 'speed', 'speed_pre', 'speed_rz']
N_PC = 5
SPEED_PROF = f'{NT}/speed_profiles.npz'
LAGS = range(1, 9)
VARIANTS = ('full', 'pre', 'pre2', 'bb', 'bb0', 'tz')


# How a trial's state is summarised from its cells' labels: 'pc1' (the default,
# and the measure of every other figure since 2026-10-31), the trial's z-scored
# score on the first principal component of that classifier's own cells x trials
# labels, signed to rise with the fraction anchored; or 'frac', the fraction of
# cells anchored. Run `neighbour_test.py frac` for the second; it writes
# neighbour_test_frac.csv and nothing else (Figure S2I).
STATE = sys.argv[1] if len(sys.argv) > 1 else 'pc1'


def pc1_state(L):
    """z-scored PC1 score per trial of a cells x trials label matrix; trials with
    no labelled cell are NaN. Missing labels take the cell's mean."""
    ok = ~np.isnan(L).all(axis=0)
    X = np.where(np.isnan(L), np.nanmean(L, axis=1, keepdims=True), L)[:, ok]
    X = X - X.mean(axis=1, keepdims=True)
    _, sv, Vt = np.linalg.svd(X, full_matrices=False)
    sc = Vt[0] * sv[0]
    if np.corrcoef(sc, np.nanmean(L[:, ok], axis=0))[0, 1] < 0:
        sc = -sc
    out = np.full(L.shape[1], np.nan)
    out[ok] = (sc - sc.mean()) / sc.std()
    return out


def state_of(L):
    """The per-trial state of a cells x trials label matrix (cells with no label
    already dropped), by STATE."""
    if not len(L):
        return np.full(L.shape[1], np.nan)
    return pc1_state(L) if STATE == 'pc1' else np.nanmean(L, axis=0)


def load(f):
    z = np.load(f'{NT}/{f}', allow_pickle=True)
    mo, dy = int(f[1:3]), int(f.split('D')[1].split('.')[0])
    beh = pd.DataFrame({k: z[k] for k in ['trial', 'ttype', 'hit', 'speed', 'speed_pre',
                                          'speed_rz', 'first_stop']})
    st = {}
    for r in REGIONS:
        m = z['region'] == r
        n = int((~np.isnan(z['labels_full'][m]).all(axis=1)).sum())
        if n < MIN_CELLS[r]:                 # inclusion on whole-track labels, for every variant
            continue
        st[(r, 'n')] = n
        for v in VARIANTS:
            if f'labels_{v}' in z.files:
                L = z[f'labels_{v}'][m]
                L = L[~np.isnan(L).all(axis=1)]
                st[(r, v)] = state_of(L)
    return mo, dy, beh, st


def speed_profiles(sessions):
    """Per-trial speed in 10 cm position bins, from the VR behaviour files; cached."""
    if os.path.exists(SPEED_PROF):
        z = np.load(SPEED_PROF)
        if all(f'M{m}D{d}' in z.files for m, d in sessions):
            return {k: z[k] for k in z.files}
    import json, warnings
    warnings.filterwarnings('ignore')
    g = {}
    nb = '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026/lick_raster_by_trial_type.ipynb'
    for c in json.load(open(nb))['cells']:
        src = ''.join(c['source'])
        if c['cell_type'] == 'code' and not src.startswith('MOUSE, DAY') and 'INV = pd.DataFrame' not in src:
            exec(compile(src, '<nb>', 'exec'), g)
    edges = np.arange(0, 201, 10)
    out = {}
    for m, d in sessions:
        bp, cp = g['vr_paths'](m, d)
        beh, cl = g['nap'].load_file(bp), g['nap'].load_file(cp)
        tr, _ = g['clip_trials'](beh['trials'].as_dataframe(), cl)
        S, P = beh['S'], beh['P']
        ts, sv = np.asarray(S.index), np.asarray(S.values, float)
        pv = np.asarray(P.values, float)[np.searchsorted(np.asarray(P.index), ts).clip(0, len(P) - 1)]
        prof = np.full((len(tr), len(edges) - 1), np.nan)
        for j, (_, t) in enumerate(tr.iterrows()):
            m_ = (ts >= t.start) & (ts < t.end)
            i = np.digitize(pv[m_], edges) - 1
            for k in range(len(edges) - 1):
                if (i == k).any():
                    prof[j, k] = sv[m_][i == k].mean()
        out[f'M{m}D{d}'] = prof
    np.savez_compressed(SPEED_PROF, **out)
    return out


def profile_pcs(prof):
    """First N_PC principal-component scores of a session's speed profiles."""
    X = np.where(np.isfinite(prof), prof, np.nanmean(prof, axis=0))
    X = X - X.mean(0)
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :N_PC] * s[:N_PC]


def neighbours(tt):
    """For each trial, the index of the nearest preceding cued trial within NEAR
    (a list, empty if there is none)."""
    cued = np.where(tt == 'b')[0]
    out = []
    for i in range(len(tt)):
        b = cued[(cued < i) & (cued >= i - NEAR)]
        out.append([b[-1]] if len(b) else [])
    return out


def auc_null(x, y, rng):
    """AUC of x for y, and the mean AUC under circular shifts of x against y."""
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    if y.sum() < MIN_OUT or (1 - y).sum() < MIN_OUT or np.std(x) == 0:
        return np.nan, np.nan
    obs = roc_auc_score(y, x)
    n = len(x)
    sh = rng.integers(SHIFT_MIN, n - SHIFT_MIN, N_SHIFT)
    null = np.mean([roc_auc_score(y, np.roll(x, s)) for s in sh])
    return obs, null


def residual(x, C):
    """x with the columns of C regressed out (least squares, with intercept)."""
    ok = np.isfinite(x) & np.isfinite(C).all(axis=1)
    r = np.full(len(x), np.nan)
    if ok.sum() < 10:
        return r
    A = np.c_[np.ones(ok.sum()), (C[ok] - C[ok].mean(0)) / (C[ok].std(0) + 1e-9)]
    beta = np.linalg.lstsq(A, x[ok], rcond=None)[0]
    r[ok] = x[ok] - A @ beta + x[ok].mean()
    return r


def persistence(s, rng):
    """Autocorrelation of a state series by lag, and its mean with trial order permuted."""
    s = np.nan_to_num(s - np.nanmean(s))
    ac = lambda x, k: np.corrcoef(x[:-k], x[k:])[0, 1]
    perm = [rng.permutation(s) for _ in range(100)]
    return ([ac(s, k) for k in LAGS], [np.mean([ac(p, k) for p in perm]) for k in LAGS])


if __name__ == '__main__':
    files = sorted(f for f in os.listdir(NT) if f.startswith('M') and f.endswith('.npz'))
    SP = speed_profiles([(int(f[1:3]), int(f.split('D')[1].split('.')[0])) for f in files])
    rows, pers = [], []
    for f in files:
        mo, dy, beh, st = load(f)
        tt, hit = beh.ttype.values, beh.hit.values
        nb = neighbours(tt)
        unc = np.where(tt == 'nb')[0]
        use = np.array([i for i in unc if nb[i]])
        if len(use) < 20:
            continue
        y = hit[use]
        nmean = lambda X: np.array([np.nanmean(X[nb[i]], axis=0) for i in use])

        def nstate(r, v):
            """The state on the preceding cued neighbour."""
            return np.array([st[(r, v)][nb[i][0]] for i in use])
        Cn = nmean(beh[COV].values.astype(float))                 # summaries, for panel D
        Cb = np.c_[nmean(hit[:, None].astype(float)), nmean(profile_pcs(SP[f'M{mo}D{dy}']))]
        rng = np.random.default_rng(mo * 100 + dy)
        for r in REGIONS:
            if (r, 'full') not in st:
                continue
            rec = dict(mouse=mo, day=dy, region=r, n_cells=st[(r, 'n')], n_unc=len(use),
                       hit_unc=y.mean(), hit_cued=hit[tt == 'b'].mean())
            for v in [v for v in VARIANTS if (r, v) in st]:
                s = st[(r, v)]
                rec[f'same_{v}'], rec[f'same_{v}_null'] = auc_null(s[use], y, rng)
                xn = nstate(r, v)
                rec[f'nbr_{v}'], rec[f'nbr_{v}_null'] = auc_null(xn, y, rng)
                rec[f'res_{v}'], rec[f'res_{v}_null'] = auc_null(residual(xn, Cb), y, rng)
                if r != 'MEC' and ('MEC', v) in st:
                    rec[f'mec_{v}'], rec[f'mec_{v}_null'] = auc_null(
                        residual(xn, np.c_[Cb, nstate('MEC', v)]), y, rng)
                if v != 'full':
                    ok = np.isfinite(st[(r, 'full')]) & np.isfinite(s)
                    rec[f'r_full_{v}'] = np.corrcoef(st[(r, 'full')][ok], s[ok])[0, 1]
            for k, c in enumerate(COV):
                rec[f'beh_{c}'], rec[f'beh_{c}_null'] = auc_null(Cn[:, k], y, rng)
            rows.append(rec)
            for v in [v for v in VARIANTS if (r, v) in st]:
                ac, an = persistence(st[(r, v)], rng)
                for k, a, b in zip(LAGS, ac, an):
                    pers.append(dict(mouse=mo, day=dy, region=r, labels=v, lag=k, ac=a, ac_null=b))
    if STATE == 'frac':
        R = pd.DataFrame(rows)
        for c in [c for c in R if c.endswith('_null')]:
            R[c.replace('_null', '_x')] = R[c.replace('_null', '')] - R[c]
        R.to_csv(f'{PS}/neighbour_test_frac.csv', index=False)
        sys.exit(f'wrote {PS}/neighbour_test_frac.csv')
    # per session and structure: how far the black-box (and task-region) state
    # follows the whole-track state, and the session properties that might
    # explain it
    C = pd.read_csv(f'{os.path.dirname(PS)}/cell_classifications_v2.csv')
    ses = []
    for f in files:
        z = np.load(f'{NT}/{f}', allow_pickle=True)
        mo, dy = int(f[1:3]), int(f.split('D')[1].split('.')[0])
        u = z['ttype'] == 'nb'
        c = C[(C.mouse == mo) & (C.day == dy) & C.brain_region.astype(str).str.startswith('ENTm')]
        for r in REGIONS:
            m = z['region'] == r
            if (~np.isnan(z['labels_full'][m]).all(axis=1)).sum() < MIN_CELLS[r]:
                continue
            ff, fb, ft = (state_of(L[~np.isnan(L).all(axis=1)])
                          for L in (z[f'labels_{k}'][m] for k in ('full', 'bb', 'tz')))
            ok = np.isfinite(ff) & np.isfinite(fb) & np.isfinite(ft)
            ses.append(dict(mouse=mo, day=dy, region=r, n_cells=int(m.sum()),
                            n_gc=int((c.cell_class_of1 == 'GC').sum()) if r == 'MEC' else np.nan,
                            r_bb=np.corrcoef(ff[ok], fb[ok])[0, 1], r_tz=np.corrcoef(ff[ok], ft[ok])[0, 1],
                            ac_full=np.corrcoef(ff[:-1], ff[1:])[0, 1], hit_unc=z['hit'][u].mean(),
                            speed_pre_sd=np.nanstd(z['speed_pre'])))
    pd.DataFrame(ses).to_csv(f'{PS}/neighbour_sessions.csv', index=False)

    R = pd.DataFrame(rows)
    for c in [c for c in R if c.endswith('_null')]:
        R[c.replace('_null', '_x')] = R[c.replace('_null', '')] - R[c]
    R.to_csv(f'{PS}/neighbour_test.csv', index=False)
    pd.DataFrame(pers).to_csv(f'{PS}/neighbour_persistence.csv', index=False)
    print(R.groupby('region').size().to_dict())
