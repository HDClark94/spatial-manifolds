"""Per-cell agreement with the population anchoring axis, from the cached labels.

Rebuilds `per_cell_pc1.csv` and `pc1_by_region.csv` from
data/population_state/labels/, written by build_anchoring_labels.py. Both files
were previously produced by scratchpad scripts that no longer exist, and both
were built from pre-gate labels, so every figure reading them (Figure 1's
following-fraction pies, Figure 2's regional comparison) was showing numbers
from a superseded classifier.

Reading the cached labels rather than reclassifying means this is seconds per
session rather than minutes, and guarantees the same labels underlie the rasters,
the population state and the per-cell scores.

For a cell inside the population the axis is recomputed by SVD WITHOUT that cell
before correlating: a cell otherwise helps define the component it is scored
against, and with a few dozen cells that self-contribution is not small. Cells
outside the population never enter the decomposition and need no correction --
applying one to them as well would bias the two groups differently.

Each cell is compared against its own circular-shift null, which fixes
SIGNIFICANCE. `p` is the two-sided fraction of shifts reaching |r|.

WHAT THE NULL IS AND IS NOT FOR -- corrected 2026-10-02. An earlier version
subtracted the null as if it were a chance level: `excess = r - r_null` with
`r` signed and `r_null` the mean of |shifted r|. Those are different quantities
and the subtraction manufactured an effect. Measured directly over 1,215 cells,
the SIGNED null mean is -0.003 -- indistinguishable from zero -- while the mean
of |shifted r| is +0.100, which is simply E|x| = 0.798*sd for a null centred at
zero (null sd 0.119). So:

  * Chance for a signed correlation with the axis is ZERO, not +0.15. The
    earlier claim that "a cell anchored on most trials correlates with a
    population that is also anchored on most trials" does not survive
    measurement: the anchored fraction is unrelated to the signed null mean
    (rho = -0.05, p = 0.08) and only weakly related to its SPREAD
    (rho = +0.14). A circular shift preserves the anchored fraction and the run
    structure, and breaks the alignment symmetrically.
  * What the anchored fraction and run structure DO change is the null's
    variance, i.e. how surprising a given |r| is. That is a significance
    question, and `p` answers it per cell.

Subtracting mean|shifted r| from a signed r therefore biases every cell down by
~0.8*sd of its own null, by an amount that differs between regions because their
null widths differ (0.130-0.157). That is what put visual cortex "below its own
null". It is an artifact of mixing a signed statistic with an absolute one.

Columns written: `r` (signed, chance 0) is the effect size; `p` the per-cell
two-sided permutation significance; `r_null_abs`, `r_null_signed` and
`r_null_sd` describe the null itself; `excess` is kept as |r| - r_null_abs,
which at least compares like with like, but `r` and `p` are what to report.
"""
import json, os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import load_session_labels

PS = '/Users/harryclark/Documents/spatial-manifolds/data/population_state'
CLS_CSV = '/Users/harryclark/Documents/spatial-manifolds/data/cell_classifications_v2.csv'
N_SHIFT = 200


def axis_from(M, frac):
    Mc = np.nan_to_num(M - np.nanmean(M, axis=1, keepdims=True))
    U, sv, Vt = np.linalg.svd(Mc, full_matrices=False)
    pc = Vt[0]
    if np.corrcoef(pc, np.nan_to_num(frac))[0, 1] < 0:
        pc = -pc
    return pc, float(sv[0] ** 2 / np.sum(sv ** 2))


def session(mo, dy, rng):
    z = load_session_labels(mo, dy)
    if z is None:
        return None
    L, ids = z['labels'], z['cluster_id']
    in_pop = np.isfinite(z['pc1_load'])
    frac = z['frac_anch']
    P = L[in_pop]
    usable = ~np.isnan(P).all(axis=1)
    P = P[usable]
    if P.shape[0] < 5:
        return None
    pop_ids = list(ids[in_pop][usable])
    pc1_all, var = axis_from(P, frac)

    rows = []
    for k, c in enumerate(ids):
        v = np.nan_to_num(L[k])
        if np.std(v) == 0:
            # A fully gated cell -- every trial anchored or every trial not --
            # has no variance, so its correlation with the axis is undefined.
            # It must still appear in the table: under the gate about a third of
            # cells are like this, and dropping them silently removes them from
            # the denominator of every "fraction of cells following" statistic.
            rows.append(dict(mouse=mo, day=dy, cluster_id=int(c), r=np.nan,
                             r_null=np.nan, r_null_abs=np.nan,
                             r_null_signed=np.nan, r_null_sd=np.nan,
                             p=np.nan, pc1_var=var,
                             frac_anch=float(np.nanmean(v)),
                             is_mec=bool(in_pop[k])))
            continue
        if c in pop_ids and len(pop_ids) > 5:
            pc, _ = axis_from(np.delete(P, pop_ids.index(c), axis=0), frac)
        else:
            pc = pc1_all
        r = float(np.corrcoef(v, pc)[0, 1])
        null = np.array([np.corrcoef(np.roll(v, s), pc)[0, 1]
                         for s in rng.integers(5, len(v) - 5, N_SHIFT)])
        rows.append(dict(mouse=mo, day=dy, cluster_id=int(c), r=r,
                         r_null=float(np.nanmean(np.abs(null))),
                         r_null_abs=float(np.nanmean(np.abs(null))),
                         r_null_signed=float(np.nanmean(null)),
                         r_null_sd=float(np.nanstd(null)),
                         p=float((np.abs(null) >= abs(r)).mean()),
                         pc1_var=var, frac_anch=float(np.nanmean(v)),
                         is_mec=bool(in_pop[k])))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    sess = sorted({(int(f.split('M')[1].split('D')[0]), int(f.split('D')[1].split('.')[0]))
                   for f in os.listdir(f'{PS}/labels') if f.endswith('.npz')})
    out, t0 = [], time.time()
    for k, (mo, dy) in enumerate(sess, 1):
        try:
            R = session(mo, dy, np.random.default_rng(abs(hash((mo, dy))) % 2**32))
        except Exception as e:
            print(f'  ! M{mo}D{dy}: {type(e).__name__}: {e}', flush=True); continue
        if R is None or not len(R):
            continue
        out.append(R)
        if k % 10 == 0:
            print(f'[{k}/{len(sess)}] {time.time()-t0:.0f}s', flush=True)
    A = pd.concat(out, ignore_index=True)

    # join region and class so pc1_by_region.csv keeps the columns its readers expect
    C = pd.read_csv(CLS_CSV)
    keep = [c for c in ('mouse', 'day', 'cluster_id', 'brain_region',
                        'cell_class_of1', 'coord_SCs_x') if c in C.columns]
    A = A.merge(C[keep].drop_duplicates(['mouse', 'day', 'cluster_id']),
                on=['mouse', 'day', 'cluster_id'], how='left')
    # kept for backwards compatibility, now comparing like with like; `r` is
    # the effect size to report and `p` the significance -- see the module
    # docstring for why the old signed-minus-absolute version was wrong
    A['excess'] = A.r.abs() - A.r_null_abs
    # the categories the figures expect: a cell can be significantly ANTI-
    # correlated with the axis, which is a different claim from being unrelated
    # to it, so 'opposes' is kept separate from 'independent'
    A['cls'] = np.where(~np.isfinite(A.r), 'unclassified',
                        np.where(A.p >= .05, 'independent',
                                 np.where(A.r > 0, 'follows', 'opposes')))
    A['cell'] = A.groupby(['mouse', 'day']).cumcount()
    A.to_csv(f'{PS}/pc1_by_region.csv', index=False)
    A[['mouse', 'day', 'cell', 'cluster_id', 'r', 'p', 'cls', 'frac_anch']].to_csv(
        f'{PS}/per_cell_pc1.csv', index=False)
    print(f'\n{len(A)} cells from {A.groupby(["mouse","day"]).ngroups} sessions')
    f = A[A.is_mec].groupby(['mouse', 'day']).apply(lambda d: (d.p < .05).mean())
    print(f'  MEC cells following the axis: median {f.median():.3f} per session '
          f'(range {f.min():.3f}-{f.max():.3f})')
