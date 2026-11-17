"""Candidate example sessions for Figure 2D-F, drawn side by side to choose from.

D-F must be representative, since the structures differ only in distribution,
but the session nearest the median alone can be one that shows nothing. So
sessions are first required to display the state, then ranked by how typical
their black-box result is:

  eligible   at least MIN_CELLS cells of the structure with labels; at least
             MIN_LABELLED of them labelled on 90% of trials; a whole-track state
             that switches, both states on at least MIN_STATE of trials, in
             blocks (lag-1 autocorrelation of the state, ac_full, at least the
             structure's median); and no dorsal-island units (they are excluded
             upstream, so this is a check)
  ranked by  distance from the structure's median, over all its tested sessions,
             on the black-box state's correlation with the whole-track state
             (r_bb) and the black-box AUC for the trial's own outcome over its
             null (same_bb - same_bb_null), each as a percentile

Writes pick_fig2_examples.pdf (the top N_SHOW of each structure, one page per
structure, each drawn as Figure 2D-F would draw it) and prints the ranking.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_fig2_example_maps as E                                   # noqa: E402

PS = E.B.PS
FIG = os.path.dirname(os.path.abspath(__file__))
MIN_CELLS = {'MEC': 40, 'SUB': 20, 'VIS': 20}
MIN_LABELLED, MIN_STATE, N_SHOW = 0.8, 0.2, 6
CMAP = ListedColormap(['#3f9b8f', '#a8559e'])


def state(L):
    """Whole-track PC1 state, signed to the fraction anchored, and that fraction."""
    L = L[~np.isnan(L).all(1)]
    f = np.nanmean(L, 0)
    sc, _ = pc1(L)
    return sc, f


def pc1(X):
    Xc = np.nan_to_num(X - np.nanmean(X, 1, keepdims=True))
    U, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    sc, ld = Vt[0], U[:, 0]
    if np.corrcoef(sc, np.nanmean(X, 0))[0, 1] < 0:
        sc, ld = -sc, -ld
    return sc, ld


def candidates():
    S = pd.read_csv(f'{PS}/neighbour_sessions.csv')
    T = pd.read_csv(f'{PS}/neighbour_test.csv')
    T['auc_bb'] = T.same_bb - T.same_bb_null
    D = S.merge(T[['mouse', 'day', 'region', 'auc_bb']], on=['mouse', 'day', 'region'])
    out = {}
    for region in ('MEC', 'SUB', 'VIS'):
        R = D[D.region == region].copy()
        for c in ('r_bb', 'auc_bb'):
            R[f'p_{c}'] = R[c].rank(pct=True)
        R['dist'] = np.hypot(R.p_r_bb - .5, R.p_auc_bb - .5)
        rows = []
        for _, r in R.iterrows():
            z = np.load(f'{E.B.OUT}/M{r.mouse}D{r.day}.npz', allow_pickle=True)
            L = z['labels_full'][z['region'] == region]
            n = (~np.isnan(L).all(1)).sum()
            lab = (np.isfinite(L).mean(1) >= .9).mean() if len(L) else 0
            if n < MIN_CELLS[region] or lab < MIN_LABELLED:
                continue
            _, f = state(L)
            on = np.mean(f > .5)
            if min(on, 1 - on) < MIN_STATE or r.ac_full < R.ac_full.median():
                continue
            rows.append(dict(mouse=r.mouse, day=r.day, n=n, labelled=lab, anch=on,
                             ac_full=r.ac_full, r_bb=r.r_bb, auc_bb=r.auc_bb, dist=r.dist))
        C = pd.DataFrame(rows).sort_values('dist')
        print(f'\n{region}: {len(C)} of {len(R)} sessions eligible; medians r_bb '
              f'{R.r_bb.median():.2f}, auc_bb {R.auc_bb.median():+.3f}')
        print(C.head(8).round(3).to_string(index=False))
        out[region] = C.head(N_SHOW)
    return out


def draw(out):
    """One page per structure, each candidate drawn exactly as Figure 2D-F would
    draw it; the session in the figure now is marked."""
    from matplotlib.backends.backend_pdf import PdfPages
    import fig2_example_panel
    name = {'MEC': 'MEC', 'SUB': 'subicular complex', 'VIS': 'visual cortex'}
    with PdfPages(f'{FIG}/pick_fig2_examples.pdf') as pdf:
        for region, C in out.items():
            fig = plt.figure(figsize=(11.8, 10.5))
            gs = fig.add_gridspec(2, 3, wspace=.3, hspace=.35, left=.06, right=.98, top=.9, bottom=.04)
            for k, (_, r) in enumerate(C.iterrows()):
                mo, dy = int(r.mouse), int(r.day)
                z = np.load(f'{E.B.OUT}/M{mo}D{dy}.npz', allow_pickle=True)
                s = E.session_maps(region, mo, dy)
                pick = E.choose_cells(s)
                now = ' (in figure now)' if E.EXAMPLES[region] == (mo, dy) else ''
                fig2_example_panel.draw(
                    fig, gs[k // 3, k % 3], z, region, s['maps'][pick], s['ids'][pick],
                    f'#{k + 1}  {name[region]}, M{mo} D{dy}{now}\n'
                    f'r_bb {r.r_bb:.2f}, AUC_bb {r.auc_bb:+.3f}, PC1 lag-1 ac {r.ac_full:.2f}')
            fig.suptitle(f'{name[region]}: the {len(C)} eligible sessions nearest the median '
                         f'black-box result', fontsize=10, y=.985)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
    print(f'wrote {FIG}/pick_fig2_examples.pdf')


if __name__ == '__main__':
    draw(candidates())
