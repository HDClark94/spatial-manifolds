"""Figure 2D-F: one session of one structure, as drawn in the figure. Shared by
fig1_supp_neighbour_bars.py (the figure) and pick_fig2_examples.py (the sheet
of candidate sessions), so the candidates look exactly as the figure would."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from spatial_manifolds.anchoring import ANCH_COLOR, NONANCH_COLOR, smooth_nanaware
from neighbour_test import pc1_state

CCOL = {'full': '0.62', 'tz': '#e0b43c', 'bb': '0.2'}           # as fig1_supp_neighbour_bars
TXTC = {'full': '0.45', 'tz': '#a07c10', 'bb': '0.15'}
CUED_C, UNCUED_C = '#3f9b4f', '#c0398d'
cmap = ListedColormap([NONANCH_COLOR, ANCH_COLOR])


def draw(fig, spec, z, region, maps, cells, title, letter=None):
    """One session: three example rate maps above the whole-track population
    raster (cells by PC1 loading), with each trial's outcome and the state, PC1 (z),
    from each classifier alongside."""
    m = z['region'] == region
    g2 = spec.subgridspec(2, 1, height_ratios=[.55, 1], hspace=.32)
    # above: three cells that load most on the whole-track state, trial x position
    # from -30 to 200 cm, the two classifiers' stretches marked above each map
    gm = g2[0].subgridspec(1, 3, wspace=.12)
    for j, (M_, c_) in enumerate(zip(maps, cells)):
        am = fig.add_subplot(gm[j])
        S_ = np.array([smooth_nanaware(r_) for r_ in M_])
        am.imshow(S_, aspect='auto', cmap='viridis', extent=[-30, 200, len(S_), 0],
                  vmax=np.nanpercentile(S_, 99), interpolation='nearest')
        for (lo, hi), k in (((-30, 30), 'bb'), ((50, 110), 'tz')):
            am.add_patch(plt.Rectangle((lo, 1.02), hi - lo, .06, transform=am.get_xaxis_transform(),
                                       color=CCOL[k], clip_on=False, lw=0))
        am.axvline(0, color='w', lw=.5, ls=':')
        am.set_xticks([0, 100, 200]); am.tick_params(labelsize=5.5, length=2, pad=1)
        am.set_xlabel('position (cm)', fontsize=5.8, labelpad=1)
        am.text(1, 1.1, f'cell {c_}', transform=am.transAxes, fontsize=5.6, color='0.3',
                ha='right', va='bottom')
        if j == 0:
            am.set_ylabel('trial', fontsize=7)
            if letter:
                am.text(-.42, 1.12, letter, transform=am.transAxes, fontsize=10, weight='bold',
                        va='bottom', ha='right')
            am.text(0, 1.3, title, transform=am.transAxes, fontsize=7, va='bottom')
        else:
            am.set_yticks([])
    # below: the whole-track labels, cells the classifier could not label on any
    # trial left out. As Figure 1: varying cells in order of their PC1 loading,
    # then, past a gap, the locked cells (one label on every trial), anchored first
    g = g2[1].subgridspec(1, 6, width_ratios=[1.9, .1, .1, .33, .33, .33], wspace=.12)
    drop = lambda L: L[~np.isnan(L).all(axis=1)]
    Lc = {k: drop(z[f'labels_{k}'][m]) for k in ('full', 'tz', 'bb')}
    Lf = Lc['full']
    Xc = np.nan_to_num(Lf - np.nanmean(Lf, 1, keepdims=True))
    ld = np.linalg.svd(Xc, full_matrices=False)[0][:, 0]
    ld = ld * np.sign(np.corrcoef(Xc.T @ ld, np.nanmean(Lf, 0))[0, 1])
    varies = np.nanstd(Lf, axis=1) > 0
    ord_v = np.where(varies)[0][np.argsort(ld[varies])[::-1]]
    lk = np.where(~varies)[0]
    ord_l = lk[np.argsort(-np.nanmean(Lf[lk], axis=1))] if len(lk) else lk
    gap = max(3, int(round(.06 * len(Lf))))
    Lb = Lf[ord_v]
    if len(ord_l):
        Lb = np.vstack([Lb, np.full((gap, Lf.shape[1]), np.nan), Lf[ord_l]])
    ax = fig.add_subplot(g[0])
    ax.imshow(Lb.T, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='nearest',
              extent=[0, len(Lb), Lf.shape[1], 0])
    if len(ord_l):
        ax.annotate(f'{len(ord_l)} locked', (len(Lb) - len(ord_l) / 2, .012),
                    xycoords=('data', 'axes fraction'), ha='center', va='bottom',
                    fontsize=4.8, color='0.15')            # inside at the foot, as Figure 1
    ax.set_title('anchoring', fontsize=6.4, loc='left', pad=2)
    ax.set_xticks([0, len(ord_v)]); ax.tick_params(labelsize=5.5, length=2, pad=1)
    ax.set_xlabel(f'cell ({len(Lf)})', fontsize=5.8, labelpad=1)
    ax.set_ylabel('trial', fontsize=7)
    for sp in ax.spines.values():
        sp.set_visible(False)
    n_tr = Lf.shape[1]
    # outcomes, one column per trial type: filled hit, white miss, grey the other type
    for jj, (t_, c, nm) in enumerate((('b', CUED_C, 'cued'), ('nb', UNCUED_C, 'uncued'))):
        ah = fig.add_subplot(g[1 + jj])
        v = np.where(z['ttype'] == t_, z['hit'].astype(float), .5)
        ah.imshow(v[:, None], aspect='auto', cmap=ListedColormap(['white', '0.88', c]), vmin=0, vmax=1,
                  interpolation='nearest')
        ah.set_xticks([]); ah.set_yticks([])
        ah.set_title(nm, fontsize=5.4, pad=2, rotation=90, color=c)
        for sp in ah.spines.values():
            sp.set_linewidth(.5); sp.set_color('0.5')
    # the state, PC1 (z), from each classifier, as S2 and S6E
    for jj, k in enumerate(('full', 'tz', 'bb')):
        ap = fig.add_subplot(g[3 + jj])
        sc = pc1_state(Lc[k])
        tr_ = np.arange(n_tr)
        for w_, c_ in ((sc >= 0, ANCH_COLOR), (sc < 0, NONANCH_COLOR)):     # as Figure 1
            ap.fill_betweenx(tr_, 0, sc, where=w_, color=c_, lw=0, edgecolor='none', interpolate=True)
        ap.axvline(0, color='0.5', lw=.6)
        ap.set_ylim(n_tr - .5, -.5); ap.set_xlim(-3.2, 3.2); ap.set_yticks([])
        ap.set_xticks([-2, 0, 2]); ap.tick_params(labelsize=5, length=1.5, pad=1)
        ap.set_title({'full': 'whole', 'tz': 'task', 'bb': 'box'}[k], fontsize=5.6, pad=2, color=TXTC[k])
        ap.spines[['top', 'right', 'left']].set_visible(False)
        if jj == 1:
            ap.set_xlabel('PC1 (z)', fontsize=5.8, labelpad=1)
