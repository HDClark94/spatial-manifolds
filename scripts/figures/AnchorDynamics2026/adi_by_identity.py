"""Does the anchoring state read from one cell class predict uncued performance?

The Anchoring Dependence Index is AUC(uncued) - AUC(cued) for the population
state predicting a hit: how much more the state matters when the reward zone
can only be found by path integration. Figure 1 reports it for the population.
This asks it of each cell class separately.

THE CONFOUND IS CELL COUNT, and it is severe. Per session the median is 5 grid
cells against 92 non-grid spatial ones, and a state averaged over more cells is
estimated more reliably, so an unmatched comparison would rank the classes by
how many of them there are. Every class is therefore subsampled to the SAME
FIVE CELLS per session, drawn 60 times, and the index averaged over the draws.

The state is the mean anchoring label across the drawn cells on each trial,
rather than a PC1 of them: with five cells a first component is dominated by
whichever cell happens to vary most.

Writes data/population_state/adi_by_identity.csv, one row per session and
class.
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
PS = f'{ROOT}/data/population_state'
OUT = f'{PS}/adi_by_identity.csv'
ORDER = ['grid', 'putative interneuron', 'non-grid spatial', 'non-spatial']
NSUB, NREP, MIN_TRIALS, MIN_CUE = 5, 60, 40, 10


def main():
    U = pd.read_csv(f'{PS}/unit_table.csv')
    E = pd.read_csv(f'{ROOT}/data/eye_anchoring/eye_anchoring_trials.csv')
    E = E.dropna(subset=['perf', 'ttype'])
    E['hit'] = (E.perf == 'hit').astype(float)
    rng = np.random.default_rng(0)
    rows = []
    for f in sorted(glob.glob(f'{PS}/labels/*.npz')):
        key = os.path.basename(f)[:-4]
        mo = int(key[1:key.index('D')])
        dy = int(key[key.index('D') + 1:])
        b = E[(E.mouse == mo) & (E.day == dy)]
        if len(b) < MIN_TRIALS:
            continue
        z = np.load(f, allow_pickle=True)
        L, tr = z['labels'], z['trial']
        u = U[(U.mouse == mo) & (U.day == dy) & U.in_population]
        idn = dict(zip(u.cluster_id, u.identity))
        ids = np.array([idn.get(int(c)) for c in z['cluster_id']], dtype=object)
        beh = b.set_index('trial').reindex(tr)
        ok = beh.hit.notna().values
        if ok.sum() < MIN_TRIALS:
            continue

        def adi(score):
            out = []
            for t in ('nb', 'b'):
                m = ok & (beh.ttype.values == t) & np.isfinite(score)
                y = beh.hit.values[m]
                if m.sum() < MIN_CUE or len(set(y)) < 2:
                    return np.nan
                out.append(roc_auc_score(y, score[m]))
            return out[0] - out[1]

        with np.errstate(invalid='ignore'):
            rec = {'mouse': mo, 'day': dy, 'n_trials': int(ok.sum()),
                   'all': adi(np.nanmean(L, axis=0))}
            for k in ORDER:
                w = np.where(ids == k)[0]
                rec[f'n_{k}'] = len(w)
                if len(w) < NSUB:
                    continue
                v = [adi(np.nanmean(L[rng.choice(w, NSUB, replace=False)], axis=0))
                     for _ in range(NREP)]
                rec[k] = np.nanmean(v)
        rows.append(rec)
    R = pd.DataFrame(rows)
    R.to_csv(OUT, index=False)
    from scipy.stats import wilcoxon
    print(f'{len(R)} sessions, {NSUB} cells per class, {NREP} draws')
    for k in ['all'] + ORDER:
        v = R[k].dropna()
        print(f'  {k:<22} n={len(v):3d}  ADI {v.mean():+.4f}  '
              f'p = {wilcoxon(v).pvalue:.3g}')
    P = R.dropna(subset=ORDER)
    print(f'  paired on the {len(P)} sessions with all four classes, no pair '
          f'differs (smallest p = '
          f'{min(wilcoxon(P[a], P[b]).pvalue for a in ORDER for b in ORDER if a < b):.2f})')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
