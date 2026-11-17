"""Figure S4, top section: what the classifier does to one session, step by step.

The intuition behind 'non-anchored', on real data (M26 D18, the session of
Figure 1), before the rest of S4 asks what such a trial is:

  A  two cells' trial x position rate maps, with the labels the classifier gives
  B  one cell's trial x trial correlation matrix (smoothed maps)
  C  how that matrix is condensed: each trial's mean correlation with every
     other trial, one number per trial, on the same trial axis
  D  those numbers split into two groups by 2-means, and the gate: both cluster
     centres are compared with the cell's own chance level (95th percentile of
     the same quantity after circularly shifting each trial's map), so a cell
     that is really in one mode is not split in half
  E  the gate's two other outcomes, on two cells: both centres above chance
     (every trial anchored) and both below (every trial non-anchored)
  F  the label sequence at each step: 2-means alone, after the gate, after the
     5-trial median filter that removes isolated labels

Every number comes from spatial_manifolds.anchoring.trial_cluster_labels, run
with the cell's cached null threshold, and the final labels are checked against
the stored label file.

Writes fig2_supp_classifier_steps.pdf
"""
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
import pynapple as nap
from sklearn.cluster import KMeans
from spatial_manifolds.anchoring import (ANCH_COLOR, NONANCH_COLOR, _mean_corr,
                                         load_session_labels, trial_cluster_labels)

FIG = os.path.dirname(os.path.abspath(__file__))
OUT = f'{FIG}/fig2_supp_classifier_steps.pdf'
MO, DY = 26, 18
MAPS = (109, 153)          # two cells whose split is kept (a grid and a non-grid spatial cell)
MAIN = 109                 # the cell followed through B-D and F
GATED = (139, 116)         # gate: all anchored, all non-anchored
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

_g = {}
for _c in json.load(open(f'{FIG}/lick_raster_by_trial_type.ipynb'))['cells']:
    if _c['cell_type'] != 'code':
        continue
    _s = ''.join(_c['source'])
    if _s.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in _s:
        continue
    exec(compile(_s, '<nb>', 'exec'), _g)
vr_paths, clip_trials, TL, NBIN, CLS = (_g['vr_paths'], _g['clip_trials'], _g['TL'],
                                        _g['NBIN'], _g['CLS'])
LAB_CMAP = ListedColormap([NONANCH_COLOR, ANCH_COLOR]); LAB_CMAP.set_bad('white')


def _lp(ax, s, dx=-.16, dy=1.0):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='right')


def trial_maps(cells):
    bp, cp = vr_paths(MO, DY)
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials_all = beh['trials'].as_dataframe()
    trials, orig = clip_trials(trials_all, clusters)
    keep = np.isin(trials_all.number.values.astype(int), orig)
    tn, trav = beh['trial_number'], beh['travel']
    dtrav = trav - (float(np.asarray(tn.values)[0]) - 1) * TL
    moving = beh['S'].threshold(3.0, method='above').time_support
    n_all = len(trials_all)
    tc = nap.compute_1d_tuning_curves(clusters[list(cells)], dtrav, nb_bins=n_all * NBIN,
                                      minmax=[0, n_all * TL], ep=moving)
    return {c: np.asarray(tc[c]).reshape(n_all, NBIN)[keep] for c in cells}


def steps(M, thr):
    """The classifier's intermediate products for one cell."""
    lab, mc, Sm, _ = trial_cluster_labels(M, null_thr=thr)
    raw, _, _, _ = trial_cluster_labels(M, null_thr=thr, gate=False, medfilt_size=0)
    gated, _, _, _ = trial_cluster_labels(M, null_thr=thr, medfilt_size=0)
    valid = Sm.sum(1) > 1e-6
    C = np.full((len(M), len(M)), np.nan)
    C[np.ix_(valid, valid)] = np.corrcoef(Sm[valid])
    km = KMeans(2, n_init=10, random_state=0).fit(mc[np.isfinite(mc)].reshape(-1, 1))
    return dict(lab=lab, raw=raw, gated=gated, mc=mc, Sm=Sm, C=C, thr=thr,
                centres=np.sort(km.cluster_centers_.ravel()))


def hist_panel(ax, s, title):
    v = s['mc'][np.isfinite(s['mc'])]
    lo, hi = s['centres']
    bins = np.linspace(min(v.min(), s['thr']) - .01, v.max() + .01, 30)
    split = (lo + hi) / 2
    # bars coloured by the FINAL label, so a gated cell shows one colour while
    # the 2-means split it overrode is still drawn (dotted)
    lab = s['lab'][np.isfinite(s['mc'])]
    ax.hist(v[lab == 0], bins, color=NONANCH_COLOR, lw=0, alpha=.8)
    ax.hist(v[lab == 1], bins, color=ANCH_COLOR, lw=0, alpha=.8)
    for c_ in (lo, hi):
        ax.axvline(c_, color='k', lw=1)
    ax.axvline(split, color='k', lw=.7, ls=':')
    ax.axvline(s['thr'], color='0.35', lw=1, ls='--')
    ax.set_title(title, fontsize=6.8, pad=3)
    ax.tick_params(labelsize=6); ax.spines[['top', 'right']].set_visible(False)


if __name__ == '__main__':
    z = load_session_labels(MO, DY)
    ids = [int(c) for c in z['cluster_id']]
    cells = sorted(set(MAPS) | {MAIN} | set(GATED))
    TM = trial_maps(cells)
    S = {c: steps(TM[c], float(z['null_thr'][ids.index(c)])) for c in cells}
    for c in cells:
        stored = np.asarray(z['labels'][ids.index(c)], float)
        same = np.nanmean(S[c]['lab'] == stored)
        print(f'cell {c}: gate {z["gate"][ids.index(c)]}, agreement with stored labels {same:.3f}')
    n = len(z['trial'])
    fig = plt.figure(figsize=(9.8, 5.4))
    G = fig.add_gridspec(2, 1, height_ratios=[1.15, 1], hspace=.45, left=.06, right=.985,
                         top=.94, bottom=.08)
    gT = G[0].subgridspec(1, 7, width_ratios=[1, .07, 1, .07, 1.25, .1, .45], wspace=.12)
    # A: two cells' trial maps, each with its label strip
    for k, c in enumerate(MAPS):
        gc = gT[2 * k].subgridspec(1, 2, width_ratios=[1, .07], wspace=.04)
        ax = fig.add_subplot(gc[0])
        Sm = S[c]['Sm']
        ax.imshow(Sm / np.nanpercentile(Sm, 99), aspect='auto', cmap='viridis', vmin=0, vmax=1,
                  interpolation='nearest', extent=[0, TL, n, 0])
        cl = CLS[(CLS.mouse == MO) & (CLS.day == DY)].set_index('cluster_id').cell_class_of1.get(c, '')
        ax.set_title(f'cell {c} ({"grid" if cl == "GC" else "non-grid spatial"})', fontsize=7, pad=3)
        ax.set_xlabel('position (cm)', fontsize=7); ax.set_xticks([0, 100, 200])
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.set_ylabel('trial', fontsize=7.5); _lp(ax, 'A', dx=-.22)
        else:
            ax.tick_params(labelleft=False)
        axs = fig.add_subplot(gc[1], sharey=ax)
        axs.imshow(S[c]['lab'][:, None], aspect='auto', cmap=LAB_CMAP, vmin=0, vmax=1,
                   interpolation='nearest', extent=[0, 1, n, 0])
        axs.set_xticks([]); axs.tick_params(labelleft=False, left=False)
        axs.set_title('label', fontsize=5.8, pad=3)
    # B: trial x trial matrix
    ax = fig.add_subplot(gT[4])
    im = ax.imshow(S[MAIN]['C'], cmap='magma', vmin=-.2, vmax=.8, interpolation='nearest',
                   extent=[0, n, n, 0])
    ax.set_title(f'cell {MAIN}: trial x trial correlation', fontsize=7, pad=3)
    ax.tick_params(labelsize=6, labelleft=False)
    cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0 - .075, ax.get_position().width * .5, .008])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal'); cb.ax.tick_params(labelsize=5.5)
    cb.outline.set_visible(False); cb.set_label('r', fontsize=6.5, labelpad=1)
    _lp(ax, 'B', dx=-.04)
    # C: condensed to one number per trial
    axc = fig.add_subplot(gT[6])
    mc, lab = S[MAIN]['mc'], S[MAIN]['lab']
    y = np.arange(n) + .5
    axc.scatter(mc, y, c=[ANCH_COLOR if l == 1 else NONANCH_COLOR for l in lab], s=2.5, lw=0)
    axc.axvline(S[MAIN]['thr'], color='0.35', lw=.8, ls='--')
    axc.set_ylim(n, 0); axc.tick_params(labelsize=6, labelleft=False)
    axc.set_xlabel('mean r with\nall other trials', fontsize=6.5)
    axc.spines[['top', 'right']].set_visible(False)
    axc.set_title('row mean', fontsize=6.8, pad=3)
    _lp(axc, 'C', dx=-.08)

    # bottom row: D, E, F
    gB = G[1].subgridspec(1, 5, width_ratios=[1.15, .8, .8, .35, 1.5], wspace=.42)
    ax = fig.add_subplot(gB[0])
    hist_panel(ax, S[MAIN], f'cell {MAIN}: 2-means on the row means\nchance level straddled: split kept')
    ax.set_xlabel('mean r with all other trials', fontsize=6.8); ax.set_ylabel('trials', fontsize=7)
    ax.text(.98, .95, 'solid: cluster centres\ndashed: cell\'s chance level\n(95th pct, shifted maps)',
            transform=ax.transAxes, ha='right', va='top', fontsize=5.6, color='0.3')
    _lp(ax, 'D', dx=-.2)
    for k, (c, t) in enumerate(zip(GATED, ('both centres above chance:\nevery trial anchored',
                                           'both centres below chance:\nevery trial non-anchored'))):
        ax = fig.add_subplot(gB[1 + k])
        hist_panel(ax, S[c], f'cell {c}: {t}')
        ax.set_xlabel('mean r', fontsize=6.8)
        if k == 0:
            _lp(ax, 'E', dx=-.2)
    # F: the label sequence through the steps
    ax = fig.add_subplot(gB[4])
    rows = [('2-means', S[MAIN]['raw']), ('gated', S[MAIN]['gated']),
            ('filtered (final)', S[MAIN]['lab']),
            ('population', (np.asarray(z['pc1'], float) > 0).astype(float))]
    ax.imshow(np.array([r for _, r in rows]), aspect='auto', cmap=LAB_CMAP, vmin=0, vmax=1,
              interpolation='nearest', extent=[0, n, len(rows), 0])
    ax.set_yticks(np.arange(len(rows)) + .5); ax.set_yticklabels([r for r, _ in rows], fontsize=6.2)
    ax.set_xlabel('trial', fontsize=7); ax.tick_params(labelsize=6)
    ax.set_title(f'cell {MAIN}\'s labels at each step,\nand the population state (PC1 > 0)', fontsize=7, pad=3)
    for sp in ax.spines.values():
        sp.set_visible(False)
    _lp(ax, 'F', dx=-.3)
    fig.savefig(OUT, format='pdf', dpi=300)
    print('wrote', OUT)
