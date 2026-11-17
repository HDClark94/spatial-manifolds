"""Is the chosen cells' agreement with the grid module slow drift?

attractor_search_eval.py removes each cell's position map (per state) before
comparing the bump read from the chosen cells with the one read from half of
the grid module. A rate that drifts slowly through the session in many cells
at once would survive that and still pull both bumps together. Here each
cell's position-removed rate is also high-passed (minus a Gaussian running
mean of sd 60, 20 or 5 s; 0 = not high-passed) before either bump is read,
and the chosen cells' phases are fitted on high-passed training data. The
comparison is as in the eval: chosen cells against grid half B beside grid
half A, and the same number of unchosen cells (10 splits per fold).

Reads the cache and attractor_search_cells{,_null}.csv
Writes data/population_state/attractor_search_drift{,_null}.csv
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
import attractor_search_eval as EV
PS = EV.PS
SIG = [None, 600, 200, 50]          # bins of 100 ms: none, 60 s, 20 s, 5 s
NULLRUN = os.environ.get('NULL') == '1'
CS = pd.read_csv(f'{PS}/attractor_search_cells{"_null" if NULLRUN else ""}.csv')
rng = np.random.default_rng(3)
rows = []
for f in sorted(glob.glob(f'{PS}/attractor_cache/M*D*.npz')):
    z = np.load(f); mo, dy = int(z['mouse']), int(z['day'])
    cells = list(z['cells']); mod_i = np.array([cells.index(c) for c in z['module']]); phi = z['phi']
    R0 = z['R'].astype(float)
    if NULLRUN:
        r_ = np.random.default_rng(7)
        for c in range(len(cells)):
            if c not in set(mod_i):
                R0[c] = np.roll(R0[c], r_.integers(600, R0.shape[1] - 600))
    ok = z['ok'].astype(bool)
    anch_all = z['anch'].astype(bool)
    pbA = np.clip((z['trackpos'] / 200 * 40).astype(int), 0, 39)
    foldA = np.where(z['idx'] >= 0, (z['idx'] // 8) % 5, -1)
    cs = CS[(CS.mouse == mo) & (CS.day == dy)]
    for sg in SIG:
        # slow component: running mean of each cell's position-removed rate, over all bins
        acc = []
        for f_ in range(5):
            sel = np.array([cells.index(c) for c in cs[(cs.fold == f_) & cs.selected].cluster_id], int)
            if len(sel) < 2: continue
            tr, te = ok & (foldA != f_), ok & (foldA == f_)
            P = EV.pos_pred(R0[:, tr], pbA[tr], anch_all[tr], pbA, anch_all)   # all bins
            Res = R0 - P
            if sg is not None:
                Res = Res - gaussian_filter1d(Res, sg, axis=1, mode='nearest')
            Rtr = R0[:, tr] if sg is None else (R0 - gaussian_filter1d(R0, sg, axis=1, mode='nearest'))[:, tr]
            a_, n_ = anch_all[te], ~anch_all[te]
            unsel = np.setdiff1d(np.setdiff1d(np.arange(len(cells)), mod_i), sel)
            hs, sb, ub = [], [], []
            for _ in range(10):
                p_ = rng.permutation(len(mod_i)); h = len(mod_i) // 2
                ia, ib = p_[:h], p_[h:2*h]; A, B = mod_i[ia], mod_i[ib]
                gB = EV.bump(Res[B][:, te], phi[:, ib])
                hs.append(EV.agree(EV.bump(Res[A][:, te], phi[:, ia]), gB, a_, n_))
                ph_A = EV.time_phases(Rtr, EV.bump(Rtr[A], phi[:, ia]))
                sb.append(EV.agree(EV.bump(Res[sel][:, te], ph_A[:, sel]), gB, a_, n_))
                u = rng.choice(unsel, min(len(sel), len(unsel)), replace=False)
                ub.append(EV.agree(EV.bump(Res[u][:, te], ph_A[:, u]), gB, a_, n_))
            acc.append(np.r_[np.nanmean(hs, 0), np.nanmean(sb, 0), np.nanmean(ub, 0)])
        if acc:
            m = np.nanmean(acc, 0)
            rows.append(dict(mouse=mo, day=dy, sig=sg or 0, half_a=m[0], half_n=m[1], sel_a=m[2], sel_n=m[3], un_a=m[4], un_n=m[5]))
    print(mo, dy, flush=True)
D = pd.DataFrame(rows); D.to_csv(f'{PS}/attractor_search_drift{"_null" if NULLRUN else ""}.csv', index=False)
print(D.groupby('sig').median(numeric_only=True).round(3).drop(columns=['mouse','day']))
