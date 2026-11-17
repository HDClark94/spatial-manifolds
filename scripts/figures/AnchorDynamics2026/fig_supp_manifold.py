"""Figure S17: the MEC population's state space, anchored and non-anchored.

Draws the results of manifold_supp.py (see its docstring for every method).

A-C  One session's windows in PCA, UMAP and CEBRA-Time space, fitted on both
     states together; anchored and non-anchored windows drawn separately,
     coloured by track position.
D    The same three spaces coloured by state.
E    That session's H1 persistence diagrams, one per state, with the
     cell-shift null's 95th percentile for the longest bar.
F    Every session: the longest H1 bar in each state, relative to its own null
     (bar minus the null's 95th percentile), beside the ring control.
G    Dominance of the longest bar over the second: real, and the ring control.
H    Position decoded from each embedding within each state (laps held out),
     against the same with each lap's positions shifted.

Writes fig_supp_manifold.pdf
"""
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from spatial_manifolds.anchoring import ANCH_COLOR, NONANCH_COLOR

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = os.path.dirname(os.path.abspath(__file__))
RES = f'{ROOT}/data/manifold_supp/results'
OUT = f'{FIG}/fig_supp_manifold.pdf'
EXAMPLE = 'M26D19'
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
SCOL = {'anch': ANCH_COLOR, 'non': NONANCH_COLOR}
SNAME = {'anch': 'anchored', 'non': 'non-anchored'}


def load_all():
    R = []
    for f in sorted(glob.glob(f'{RES}/M*D*.npz')):
        z = np.load(f, allow_pickle=True)
        d = dict(zip(z['dec_keys'], z['dec_vals']))
        r = dict(key=os.path.basename(f)[:-4], n_cells=int(z['n_cells']))
        for st in ('anch', 'non'):
            H, Hs, Hc = z[f'H_{st}'], z[f'Hs_{st}'], z[f'Hc_{st}']
            r[f'top_{st}'] = H[0]; r[f'null95_{st}'] = np.percentile(Hs[:, 0], 95)
            r[f'p_{st}'] = (np.sum(Hs[:, 0] >= H[0]) + 1) / (len(Hs) + 1)
            r[f'dom_{st}'] = H[0] / H[1]; r[f'ctrl_top_{st}'] = Hc[0]
            r[f'ctrl_dom_{st}'] = Hc[0] / Hc[1]
            for e in ('pca', 'umap', 'cebra'):
                r[f'dec_{e}_{st}'] = d[f'{e}_{st}']; r[f'dec_{e}_{st}_null'] = d[f'{e}_{st}_null']
        R.append(r)
    import pandas as pd
    return pd.DataFrame(R)


def scatter3(ax, E, c, **kw):
    ax.scatter(E[:, 0], E[:, 1], E[:, 2], c=c, s=1.2, lw=0, rasterized=True, **kw)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for a_ in (ax.xaxis, ax.yaxis, ax.zaxis):
        a_.pane.fill = False; a_.pane.set_edgecolor('0.85')
    ax.view_init(elev=22, azim=-60)


if __name__ == '__main__':
    D = load_all()
    print(f'{len(D)} sessions')
    z = np.load(f'{RES}/{EXAMPLE}.npz', allow_pickle=True)
    a, pos = z['anch'].astype(bool), z['pos']
    fig = plt.figure(figsize=(10.4, 9.2))
    outer = fig.add_gridspec(3, 1, height_ratios=[1, .95, .9], hspace=.26,
                             left=.075, right=.985, top=.95, bottom=.06)
    # ---- A-C
    gA = outer[0].subgridspec(1, 8, width_ratios=[1, 1, .12, 1, 1, .12, 1, 1], wspace=.02)
    rng = np.random.default_rng(0)
    for j, (nm, E, letter) in enumerate((('PCA', z['P3'], 'A'), ('UMAP', z['U'], 'B'),
                                          ('CEBRA-Time', z['C'], 'C'))):
        for k, st in enumerate(('anch', 'non')):
            m = a if st == 'anch' else ~a
            i = rng.permutation(np.flatnonzero(m))[:1500]
            ax = fig.add_subplot(gA[3 * j + k], projection='3d')
            scatter3(ax, E[i], pos[i], cmap='twilight', vmin=0, vmax=200)
            ax.set_title(f'{nm}\n{SNAME[st]}', fontsize=7.5, color=SCOL[st], pad=-2, linespacing=1.1)
            if k == 0:
                ax.text2D(-.05, 1.0, letter, transform=ax.transAxes, fontsize=10, weight='bold')
    sm = matplotlib.cm.ScalarMappable(cmap='twilight', norm=matplotlib.colors.Normalize(0, 200))
    cax = fig.add_axes([.06, outer[0].get_position(fig).y0 - .005, .12, .007])
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=[0, 100, 200])
    cb.ax.tick_params(labelsize=6); cb.outline.set_visible(False)
    cb.set_label('track position (cm)', fontsize=6.5, labelpad=1)

    # ---- D (state) and E (persistence diagrams)
    gB = outer[1].subgridspec(1, 5, width_ratios=[1, 1, 1, .15, 1.6], wspace=.05)
    for j, (nm, E) in enumerate((('PCA', z['P3']), ('UMAP', z['U']), ('CEBRA-Time', z['C']))):
        ax = fig.add_subplot(gB[j], projection='3d')
        i = rng.permutation(len(a))[:3000]
        scatter3(ax, E[i], np.where(a[i], 1, 0),
                 cmap=matplotlib.colors.ListedColormap([NONANCH_COLOR, ANCH_COLOR]), vmin=0, vmax=1,
                 alpha=.6)
        ax.set_title(f'{nm}, by state', fontsize=7.5, pad=-2)
        if j == 0:
            ax.text2D(-.05, 1.0, 'D', transform=ax.transAxes, fontsize=10, weight='bold')
    gE = gB[4].subgridspec(1, 2, wspace=.35)
    ex = D.set_index('key').loc[EXAMPLE]
    for k, st in enumerate(('anch', 'non')):
        ax = fig.add_subplot(gE[k])
        dg = z[f'dgm_{st}']
        mx = max(1e-3, dg[:, 1].max() * 1.05)
        ax.plot([0, mx], [0, mx], color='0.6', lw=.7)
        # null: bars shorter than the null's 95th-percentile longest bar sit
        # inside this band above the diagonal
        ax.fill_between([0, mx], [0, mx], [ex[f'null95_{st}'], mx + ex[f'null95_{st}']],
                        color='0.9', lw=0, edgecolor='none')
        ax.scatter(dg[:, 0], dg[:, 1], s=8, color=SCOL[st], lw=0)
        ax.set_xlim(0, mx); ax.set_ylim(0, mx)
        ax.set_xlabel('birth', fontsize=7); ax.set_ylabel('death', fontsize=7)
        ax.tick_params(labelsize=6); ax.spines[['top', 'right']].set_visible(False)
        ax.set_title(f'H1, {SNAME[st]}', fontsize=7.5, color=SCOL[st])
        ax.set_aspect('equal')
        if k == 0:
            ax.text(-.45, 1.12, 'E', transform=ax.transAxes, fontsize=10, weight='bold')

    # ---- F-H, every session
    gC = outer[2].subgridspec(1, 3, width_ratios=[1, 1, 1.5], wspace=.42)
    ax = fig.add_subplot(gC[0])
    for k, st in enumerate(('anch', 'non')):
        v = D[f'top_{st}'] - D[f'null95_{st}']
        c = D[f'ctrl_top_{st}'] - D[f'null95_{st}']
        x = k + rng.uniform(-.12, .12, len(D))
        ax.scatter(x, v, s=10, color=SCOL[st], lw=0, zorder=3)
        ax.scatter(k + .38 + rng.uniform(-.06, .06, len(D)), c, s=8, facecolor='none',
                   edgecolor='0.5', lw=.6, zorder=2)
        ax.plot([k - .2, k + .2], [np.median(v)] * 2, color='k', lw=1.5, zorder=4)
        print(f'F {st}: top-null95 median {np.median(v):+.3f}, above null (p<.05) in '
              f'{int((D[f"p_{st}"] < .05).sum())}/{len(D)}; control median {np.median(c):+.3f}')
    for i_ in range(len(D)):
        ax.plot([0, 1], [D.top_anch[i_] - D.null95_anch[i_], D.top_non[i_] - D.null95_non[i_]],
                color='0.85', lw=.5, zorder=1)
    p = wilcoxon(D.top_anch - D.null95_anch, D.top_non - D.null95_non).pvalue
    ax.axhline(0, color='0.5', lw=.7, ls=':')
    ax.set_xticks([0, .38, 1, 1.38]); ax.set_xticklabels(['anch', 'ring\nctrl', 'non', 'ring\nctrl'], fontsize=6.5)
    ax.set_ylabel('longest H1 bar\nminus null 95th pct', fontsize=7.5)
    ax.set_title(f'anchored vs non-anchored p = {p:.2g}\n'
                 f'above null: {int((D.p_anch < .05).sum())} and {int((D.p_non < .05).sum())} '
                 f'of {len(D)}', fontsize=7, loc='left')
    ax.tick_params(labelsize=6.5); ax.spines[['top', 'right']].set_visible(False)
    ax.text(-.28, 1.1, 'F', transform=ax.transAxes, fontsize=10, weight='bold')
    print(f'F anch vs non (top - null95): p = {p:.3g}')

    ax = fig.add_subplot(gC[1])
    for k, (col, lab) in enumerate((('dom_anch', 'anch'), ('ctrl_dom_anch', 'ring ctrl\n(anch)'),
                                     ('dom_non', 'non'), ('ctrl_dom_non', 'ring ctrl\n(non)'))):
        v = D[col]
        st = 'anch' if 'anch' in col else 'non'
        ax.scatter(k + rng.uniform(-.12, .12, len(D)), v, s=9, lw=.6,
                   facecolor=SCOL[st] if 'ctrl' not in col else 'none',
                   edgecolor=SCOL[st] if 'ctrl' not in col else '0.5', zorder=3)
        ax.plot([k - .2, k + .2], [np.median(v)] * 2, color='k', lw=1.5, zorder=4)
        print(f'G {col}: median {np.median(v):.2f}, > 1.5 in {int((v > 1.5).sum())}/{len(v)}')
    ax.axhline(1, color='0.5', lw=.7, ls=':')
    ax.set_xticks(range(4)); ax.set_xticklabels(['anch', 'ring\nctrl', 'non', 'ring\nctrl'], fontsize=6.5)
    ax.set_ylabel('longest / second H1 bar', fontsize=7.5)
    ax.set_title('one dominant loop?', fontsize=7, loc='left')
    ax.tick_params(labelsize=6.5); ax.spines[['top', 'right']].set_visible(False)
    ax.text(-.28, 1.1, 'G', transform=ax.transAxes, fontsize=10, weight='bold')

    ax = fig.add_subplot(gC[2])
    xt = []
    for j, e in enumerate(('pca', 'umap', 'cebra')):
        for k, st in enumerate(('anch', 'non')):
            x0 = j * 3 + k
            v, vn = D[f'dec_{e}_{st}'], D[f'dec_{e}_{st}_null']
            ax.scatter(x0 + rng.uniform(-.15, .15, len(D)), v, s=8, color=SCOL[st], lw=0, zorder=3)
            ax.plot([x0 - .25, x0 + .25], [np.median(v)] * 2, color='k', lw=1.5, zorder=4)
            ax.plot([x0 - .25, x0 + .25], [np.median(vn)] * 2, color='0.6', lw=1.2, ls='--', zorder=4)
            print(f'H {e} {st}: median error {np.median(v):.1f} cm (null {np.median(vn):.1f})')
        p_ = wilcoxon(D[f'dec_{e}_anch'], D[f'dec_{e}_non']).pvalue
        ax.text(j * 3 + .5, 1.0, f'p = {p_:.1g}', transform=ax.get_xaxis_transform(),
                ha='center', fontsize=6)
        xt.append(j * 3 + .5)
    ax.set_xticks(xt); ax.set_xticklabels(['PCA', 'UMAP', 'CEBRA-Time'], fontsize=7)
    ax.set_ylabel('position decoding error (cm)', fontsize=7.5)
    ax.set_title('position from the embedding, within each state\n(dashed: laps shifted)',
                 fontsize=7, loc='left', pad=8)
    ax.tick_params(labelsize=6.5); ax.spines[['top', 'right']].set_visible(False)
    ax.text(-.15, 1.1, 'H', transform=ax.transAxes, fontsize=10, weight='bold')
    fig.savefig(OUT + '.partial', format='pdf', dpi=300); os.replace(OUT + '.partial', OUT)
    print('wrote', OUT)
