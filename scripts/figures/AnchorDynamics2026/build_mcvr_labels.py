"""Anchoring labels and a trial table for the multi-context VR sessions.

The MCVR notebooks currently carry an INLINE copy of `trial_cluster_labels`,
copied verbatim from an older notebook. That copy predates two changes to the
shared classifier -- NaN-aware smoothing, and the absolute gate that stops
2-means splitting a cell with no anchoring at all -- so MCVR anchoring is
computed by a different method from every other figure in the collection, and
the numbers are not comparable. This builds MCVR labels with the SAME classifier
the rest of the project uses, and writes them in the same form.

    data/population_state/labels_mcvr/M{mouse}D{day}.npz
    data/population_state/trial_table_mcvr.csv

MCVR differs from the linear-track VR dataset in ways that matter here:

  one combined NWB per session, so there is no separate clusters file and no
      `clip_trials` renumbering -- trial ids are the raw `number` field
  track length 230 cm, not 200, at 2 cm bins (115 bins per trial)
  every trial carries a CONTEXT (rz1 familiar / rz2 novel) as well as a type
      (b cued / nb uncued), so the trial table keeps both
  no cell classification datasheet covers these sessions, so the population is
      defined by anatomy alone -- all curated ENTm cells

CONTEXT IS PASSED TO THE CLASSIFIER. Each trial is scored by its mean
correlation with the other trials OF ITS OWN CONTEXT, never across contexts.
This is not an MCVR-specific classifier: with a single context it reduces
exactly to what the linear-track data has always used (verified -- identical
labels), so one procedure covers both datasets.

Averaging across contexts would be wrong wherever a cell fires in different
places in rz1 and rz2: its within-context correlations are high and its
across-context ones near zero, so pooling halves every trial's score. On
surrogates that costs nothing for a well-driven cell, but at a 0.5 Hz peak rate
a cell anchored in BOTH contexts is reported as 0.67 anchored instead of 0.97,
and a cell anchored in only one context comes back at 0.52-0.55 rather than
exactly 0.50. `remap_r` records each cell's rz1-vs-rz2 rate map correlation so
the effect can be checked against how much a cell actually remaps.

That last point is worth stating plainly: in the linear-track data the population
is all curated ENTm cells, and here it is the same, but the two are not
guaranteed to contain the same CELL TYPES because MCVR has no grid/non-grid
classification. Sessions from M22 carry almost no entorhinal cells at all (its
probe did not reach MEC), so they yield labels but no population state.
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
import pynapple as nap
from spatial_manifolds.data.curation import curate_clusters
from spatial_manifolds.anchoring import (trial_cluster_labels, null_mean_corr,
                                         save_null_cache)

SOURCE = '/Users/harryclark/Downloads/harry_multi_context_vr/dataset/'
PS = '/Users/harryclark/Documents/spatial-manifolds/data/population_state'
LABELS = f'{PS}/labels_mcvr'
TABLE = f'{PS}/trial_table_mcvr.csv'
BS, TL, MIN_SPEED = 2, 230, 3.0
NBIN = int(TL / BS)
MIN_POP = 5


def sessions():
    import glob, re
    out = []
    for p in sorted(glob.glob(f'{SOURCE}sub-M*/sub-M*_ses-D*MCVR_ecephys+behavior.nwb')):
        m = re.search(r'sub-M(\d+)_ses-D(\d+)MCVR', p)
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
    return sorted(set(out))


def gate_action(mc, thr):
    from sklearn.cluster import KMeans
    v = np.isfinite(mc)
    if v.sum() < 6:
        return 'unclassified'
    km = KMeans(2, n_init=10, random_state=0).fit(mc[v].reshape(-1, 1))
    lo, hi = np.sort(km.cluster_centers_.ravel())
    return ('all non-anchored' if hi < thr else
            'all anchored' if lo >= thr else 'split kept')


def build_session(mo, dy):
    p = f'{SOURCE}sub-M{mo}/sub-M{mo}_ses-D{dy}MCVR_ecephys+behavior.nwb'
    if not os.path.exists(p):
        return None
    data = nap.load_file(p)
    cur = curate_clusters(data['units'])
    trials = data['trials'].as_dataframe()
    if len(cur) == 0 or len(trials) == 0:
        return None
    n_tr = len(trials)

    tns, travel = data['trial_number'], data['travel']
    moving = data['S'].threshold(MIN_SPEED, method='above').time_support
    dt = travel - (float(np.asarray(tns.values)[0]) - 1) * TL
    tcs = np.asarray(nap.compute_1d_tuning_curves(cur, dt, nb_bins=n_tr * NBIN,
                                                  minmax=[0, n_tr * TL], ep=moving))
    ids = np.asarray(cur.index)
    regions = np.asarray(cur.metadata['brain_region'].values).astype(str)

    ctx = trials.context.values.astype(str)
    uctx = [c for c in np.unique(ctx) if (ctx == c).sum() >= 6]

    L, thrs, acts, remap = [], [], [], []
    for ci in range(len(ids)):
        M = tcs[:, ci].reshape(n_tr, NBIN)
        lab, mc, sm, _ = trial_cluster_labels(M, context=ctx)
        v = np.isfinite(mc)
        thr = (null_mean_corr(sm[v], context=ctx[v]) if v.sum() >= 6 else np.nan)
        L.append(lab); thrs.append(thr)
        acts.append(gate_action(mc, thr) if np.isfinite(thr) else 'unclassified')

        # how much does this cell remap between the two contexts?
        if len(uctx) == 2:
            m1 = np.nanmean(M[ctx == uctx[0]], axis=0)
            m2 = np.nanmean(M[ctx == uctx[1]], axis=0)
            ok2 = np.isfinite(m1) & np.isfinite(m2)
            remap.append(float(np.corrcoef(m1[ok2], m2[ok2])[0, 1])
                         if ok2.sum() > 10 and np.std(m1[ok2]) > 0
                         and np.std(m2[ok2]) > 0 else np.nan)
        else:
            remap.append(np.nan)
    L = np.array(L)

    in_pop = np.array([r.startswith('ENTm') for r in regions])
    frac = pc1 = None
    full_load = np.full(len(ids), np.nan)
    var = np.nan
    if in_pop.sum() >= MIN_POP:
        P = L[in_pop]
        usable = ~np.isnan(P).all(axis=1)
        P = P[usable]
        if P.shape[0] >= MIN_POP:
            frac = np.nanmean(P, axis=0)
            Pc = np.nan_to_num(P - np.nanmean(P, axis=1, keepdims=True))
            U, sv, Vt = np.linalg.svd(Pc, full_matrices=False)
            pc1, load = Vt[0], U[:, 0]
            if np.corrcoef(pc1, np.nan_to_num(frac))[0, 1] < 0:
                pc1, load = -pc1, -load
            var = float(sv[0] ** 2 / np.sum(sv ** 2))
            full_load[np.where(in_pop)[0][usable]] = load
    if frac is None:
        frac = np.full(n_tr, np.nan); pc1 = np.full(n_tr, np.nan)

    S = data['S']
    st, svl = np.asarray(S.index), np.asarray(S.values, float)
    spd = []
    for s0, e0 in zip(trials.start.values, trials.end.values):
        m = (st >= s0) & (st <= e0) & (svl >= MIN_SPEED)
        spd.append(float(svl[m].mean()) if m.sum() > 10 else np.nan)

    os.makedirs(LABELS, exist_ok=True)
    np.savez_compressed(f'{LABELS}/M{mo}D{dy}.npz', labels=L,
                        cluster_id=ids, region=regions,
                        trial=trials.number.values.astype(int),
                        frac_anch=frac, pc1=pc1,
                        pc1_load=full_load, pc1_var=var,
                        null_thr=np.array(thrs), gate=np.array(acts),
                        remap_r=np.array(remap), context=ctx)
    return pd.DataFrame(dict(
        mouse=mo, day=dy, trial=trials.number.values.astype(int),
        ttype=trials.type.values, context=trials.context.values,
        perf=trials.performance.values, result=trials.result.values,
        hit=(trials.performance.values == 'hit').astype(int), speed=spd,
        frac_anch=frac, pc1=pc1, n_cells=int(np.isfinite(full_load).sum())))


def build_all(verbose=True):
    out, t0 = [], time.time()
    for mo, dy in sessions():
        if os.path.exists(f'{LABELS}/M{mo}D{dy}.npz'):
            z = np.load(f'{LABELS}/M{mo}D{dy}.npz', allow_pickle=True)
            # rebuild the table row from the cached labels without reclassifying
            t = build_session(mo, dy) if not os.path.exists(TABLE) else None
            if t is not None:
                out.append(t)
            continue
        try:
            t = build_session(mo, dy)
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        if t is not None:
            out.append(t)
            save_null_cache()
            if verbose:
                print(f'  M{mo}D{dy}: {len(t)} trials, '
                      f'{int(t.n_cells.iloc[0])} population cells, '
                      f'{time.time()-t0:.0f}s', flush=True)
    if not out:
        return pd.read_csv(TABLE) if os.path.exists(TABLE) else None
    A = pd.concat(out, ignore_index=True)
    A.to_csv(TABLE, index=False)
    if verbose:
        print(f'built trial_table_mcvr.csv: {len(A)} trials, '
              f'{A.groupby(["mouse","day"]).ngroups} sessions')
    return A


def ensure_mcvr_tables(rebuild=False, verbose=True):
    """Labels and trial table for MCVR, built on demand.

    Returns the trial table. Pass rebuild=True after changing the classifier or
    the population definition -- neither invalidates these files automatically.
    """
    if os.path.exists(TABLE) and not rebuild:
        A = pd.read_csv(TABLE)
        if verbose:
            print(f'trial_table_mcvr.csv: {len(A)} trials, '
                  f'{A.groupby(["mouse","day"]).ngroups} sessions (cached)')
        return A
    if rebuild:
        import shutil
        shutil.rmtree(LABELS, ignore_errors=True)
    return build_all(verbose=verbose)


def load_mcvr_labels(mouse, day):
    """Cached MCVR labels for one session, or None."""
    p = f'{LABELS}/M{mouse}D{day}.npz'
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


if __name__ == '__main__':
    A = build_all()
    if A is not None:
        pop = A[A.n_cells >= MIN_POP]
        print(f'  sessions with a population state: '
              f'{pop.groupby(["mouse","day"]).ngroups} of '
              f'{A.groupby(["mouse","day"]).ngroups}')
        print(f'  contexts: {A.context.value_counts().to_dict()}')
        print(f'  frac_anch mean {A.frac_anch.mean():.3f}')
