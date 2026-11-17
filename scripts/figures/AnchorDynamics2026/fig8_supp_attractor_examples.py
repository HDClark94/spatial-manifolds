"""Figure S14 A-C in every session with a large grid module, for choosing examples.

Not a paper figure. One row per session with at least MIN_GRID grid cells in
the module (Figure S14 K, L), largest module first:

    A  the population anchoring state, cell by cell (sorted by PC1 loading),
       and PC1 per trial, as Figure S14A
    B  the module's bump phase along the track, at each position of each trial
    C  the same across the track

B and C are computed as attractor_coherence.py computes them for the example
(EXAMPLE=M..D..): psi_a = angle(sum_c P_c exp(i phi_ca)) over the module's
position maps, on all three lattice axes; the axis that advances most along
the track is drawn as 'along' and the one that advances least as 'across',
as in Figure S14 B, C. Each row's title gives the circular correlation between
the two drawn axes (near 0 means an independent pair) and that between the
two axes that advance (near 1 on a linear track).

Reads data/population_state/attractor_cache/, labels/, anchoring_cells.csv
Writes fig8_supp_attractor_examples.pdf (one page per ROWS sessions)
"""
import glob
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

FIG = os.path.dirname(os.path.abspath(__file__))
PS = '/Users/harryclark/Documents/spatial-manifolds/data/population_state'
OUT = f'{FIG}/fig8_supp_attractor_examples.pdf'
ANCH, NON = '#a8559e', '#3f9b8f'
TA = ListedColormap([NON, ANCH])
NORM = BoundaryNorm([-0.5, 0.5, 1.5], TA.N)
MIN_GRID, ROWS, TL = 15, 4, 200.0
CURRENT = (25, 24)                  # the session Figure S14 uses now


def circ_r(a, b):
    """Circular correlation (Fisher and Lee / Jammalamadaka)."""
    a, b = a[np.isfinite(a) & np.isfinite(b)], b[np.isfinite(a) & np.isfinite(b)]
    sa, sb = np.sin(a - np.angle(np.exp(1j * a).mean())), np.sin(b - np.angle(np.exp(1j * b).mean()))
    return (sa * sb).sum() / np.sqrt((sa ** 2).sum() * (sb ** 2).sum())


def session(f):
    z = np.load(f)
    mo, dy = int(z['mouse']), int(z['day'])
    cells = list(z['cells'])
    mi = [cells.index(c) for c in z['module']]
    nb = int(z['nbin'])
    P = z['P'][mi].astype(float)
    psi = np.stack([np.angle(np.exp(1j * q) @ P) for q in z['phi_all']]).reshape(3, -1, nb)
    adv = [abs(np.median([np.unwrap(r)[-1] - np.unwrap(r)[0] for r in w])) for w in psi]
    order = np.argsort(adv)[::-1]
    return dict(mo=mo, dy=dy, n=len(mi), psi=psi, along=int(order[0]), across=int(order[-1]),
                adv2=int(order[1]), st=np.asarray(z['state']).astype(bool), nb=nb)


def strip(mo, dy):
    Z = np.load(f'{PS}/labels/M{mo}D{dy}.npz', allow_pickle=True)
    C = pd.read_csv(f'{PS}/anchoring_cells.csv')
    C = C[(C.mouse == mo) & (C.day == dy)]
    sel = np.isin(Z['cluster_id'], C.cluster_id[C.in_population].values)
    L, load = Z['labels'][sel], Z['pc1_load'][sel]
    with np.errstate(invalid='ignore'):
        varies = np.array([np.isfinite(r).sum() > 1 and np.nanstd(r) > 0 for r in L])
    return L[np.where(varies)[0][np.argsort(load[varies])[::-1]]], np.asarray(Z['pc1'])


S = [session(f) for f in sorted(glob.glob(f'{PS}/attractor_cache/M*D*.npz'))]
S = sorted([s for s in S if s['n'] >= MIN_GRID], key=lambda s: -s['n'])
print(f'{len(S)} sessions with >= {MIN_GRID} grid cells')

with PdfPages(OUT) as pdf:
    for p0 in range(0, len(S), ROWS):
        fig = plt.figure(figsize=(9.6, 12.4))
        gs = fig.add_gridspec(ROWS, 1, hspace=.42, left=.06, right=.93, top=.95, bottom=.04)
        for r_, s in enumerate(S[p0:p0 + ROWS]):
            mo, dy, st, psi = s['mo'], s['dy'], s['st'], s['psi']
            L, pc1 = strip(mo, dy)
            ntr = L.shape[1]
            g = gs[r_].subgridspec(1, 4, wspace=.16, width_ratios=[.85, .22, 1.3, 1.3])
            ax = fig.add_subplot(g[0])
            ax.imshow(L.T, aspect='auto', cmap=TA, norm=NORM, interpolation='nearest',
                      extent=[0, L.shape[0], ntr, 0])
            ax.set_xticks([]); ax.tick_params(labelsize=6.5)
            ax.set_ylabel('trial', fontsize=7.5)
            ax.set_xlabel(f'{L.shape[0]} MEC cells', fontsize=7.2)
            cur = '   (the current S14 example)' if (mo, dy) == CURRENT else ''
            ax.text(0, 1.12, f'M{mo} D{dy}: {s["n"]} grid cells in the module, '
                    f'{100 * st.mean():.0f}% of {len(st)} trials anchored{cur}',
                    transform=ax.transAxes, fontsize=8.6, weight='bold', va='bottom')
            ax.text(-.2, 1.02, 'A', transform=ax.transAxes, fontsize=9, weight='bold', va='bottom')
            ax2 = fig.add_subplot(g[1], sharey=ax)
            tr = np.arange(len(pc1))
            ax2.fill_betweenx(tr, 0, pc1, where=pc1 >= 0, color=ANCH, lw=0, edgecolor='none',
                              interpolate=True)
            ax2.fill_betweenx(tr, 0, pc1, where=pc1 < 0, color=NON, lw=0, edgecolor='none',
                              interpolate=True)
            ax2.axvline(0, color='0.4', lw=.5)
            ax2.set_xticks([]); ax2.tick_params(labelleft=False, length=0)
            for sp in ax2.spines.values():
                sp.set_visible(False)
            ax2.set_xlabel('PC1', fontsize=7.0)
            ra = circ_r(psi[s['along']].ravel(), psi[s['across']].ravel())
            rv = circ_r(psi[s['along']].ravel(), psi[s['adv2']].ravel())
            for j, (k, lab, pn) in enumerate(((s['along'], r'$\psi_\parallel$  along', 'B'),
                                              (s['across'], r'$\psi_\perp$  across', 'C'))):
                ax = fig.add_subplot(g[2 + j])
                ax.imshow(psi[k][:ntr], cmap='hsv', aspect='auto', interpolation='nearest',
                          extent=[0, TL, min(ntr, psi.shape[1]), 0], vmin=-np.pi, vmax=np.pi)
                ax.set_yticks([]); ax.set_xticks([0, 100, 200]); ax.tick_params(labelsize=6.5)
                ax.set_xlabel('position (cm)', fontsize=7.2)
                ax.set_title(lab, fontsize=8, loc='left', color='0.25')
                ax.text(-.04, 1.02, pn, transform=ax.transAxes, fontsize=9, weight='bold',
                        va='bottom', ha='right')
                if j:
                    ax.text(1.0, 1.12, f'circular r: along vs across {ra:+.2f}; the two advancing '
                            f'axes {rv:+.2f}', transform=ax.transAxes, fontsize=6.6, ha='right',
                            va='bottom', color='0.4')
                    bb = ax.get_position()
                    cax = fig.add_axes([bb.x1 + .008, bb.y0, .007, bb.height])
                    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-np.pi, np.pi),
                                                            cmap='hsv'), cax=cax)
                    cb.set_ticks([-np.pi, 0, np.pi])
                    cb.set_ticklabels([r'$-\pi$', '0', r'$\pi$'])
                    cb.ax.tick_params(labelsize=5.8, length=2, pad=1.5)
                    cb.outline.set_linewidth(.4)
            print(f'  M{mo} D{dy}: {s["n"]} grid, {ntr} trials, {100 * st.mean():.0f}% anchored, '
                  f'along/across r {ra:+.2f}, advancing pair r {rv:+.2f}', flush=True)
        pdf.savefig(fig, dpi=200)
        plt.close(fig)
print('wrote', OUT)
