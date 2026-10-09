"""One session, drawn as a row: slice, then per structure a raster and two cells.

Figure 6's region examples and its supplement use the same row, so the row lives
here rather than in both -- two copies drift, and a reader comparing the main
figure against the supplement has to be able to trust that "the same layout"
means the same layout.

A row is: the recording slice with this session's cells on it, then for each of
up to three structures a raster built from that structure's OWN cells plus the
two cells with the largest loading on that structure's own PC1, then all the
components overlaid. Nothing in a row is scored against another structure's
axis, so agreement between the groups is agreement between independent
descriptions of the session.

Only what is needed is pulled out of lick_raster_by_trial_type.ipynb, by name,
into this module's namespace. The older scripts did `globals().update(...)` on
the whole notebook, which silently overwrote their own constants -- the notebook
defines MIN_CELLS = 25 and SIGMA, and a figure that had set its own threshold to
10 quietly ran at 25.
"""
import json
import os

import numpy as np
import pandas as pd
import pynapple as nap
from matplotlib.colors import BoundaryNorm, ListedColormap

import anatomy_slice as ANA
from spatial_manifolds.anchoring import (ANCH_COLOR, NONANCH_COLOR,
                                         load_session_labels, smooth_nanaware)

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = f'{ROOT}/scripts/figures/AnchorDynamics2026'

_g = {}
for _c in json.load(open(f'{FIG}/lick_raster_by_trial_type.ipynb'))['cells']:
    if _c['cell_type'] != 'code':
        continue
    _s = ''.join(_c['source'])
    if _s.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in _s:
        continue
    exec(compile(_s, '<nb>', 'exec'), _g)
vr_paths, clip_trials = _g['vr_paths'], _g['clip_trials']
TL, NBIN = _g['TL'], _g['NBIN']

RCOL = {'MEC': '#7b4173', 'SUB': '#c0723a', 'VIS': '#3f9b4f',
        'CB': '#4c72b0', 'LEC': '#8c8c3f'}
RNAME = {'MEC': 'MEC', 'SUB': 'SUB/PARA', 'VIS': 'VIS', 'CB': 'CEREB',
         'LEC': 'LEC'}
ORDER = ['MEC', 'SUB', 'VIS', 'CB', 'LEC']
TA_CMAP = ListedColormap([NONANCH_COLOR, ANCH_COLOR])
TA_NORM = BoundaryNorm([-.5, .5, 1.5], TA_CMAP.N)
MAP_SIGMA = 2.0
N_MAPS = 2
SLICE_STEP = 12
# One slice column, then THREE columns per region (its anchoring raster and two
# example rate maps), then the PC1 overlay. Built from the number of regions the
# row actually has rather than fixed at three: a two-region row left columns
# 7-9 empty and opened a wide band of white between its last rate map and the
# PC1 panel.
def row_widths(n_regions):
    return [1.70] + [1.20, .78, .78] * n_regions + [1.00]


WR = row_widths(3)          # retained for callers that lay out their own rows

_CLS = pd.read_csv(f'{ROOT}/data/cell_classifications_v2.csv')


def fam(r):
    r = str(r)
    if r.startswith('ENTm'):
        return 'MEC'
    if any(r.startswith(x) for x in ('PRE', 'POST', 'PAR', 'SUB')):
        return 'SUB'
    if r.startswith('VIS'):
        return 'VIS'
    if r.startswith('ENTl'):
        return 'LEC'
    if any(k in r for k in ('CENT', 'CUL', 'ARB', 'DEC', 'FOTU', 'PYR', 'SIM',
                            'AN', 'PRM', 'COPY', 'FL', 'NOD', 'UVU', 'CBX',
                            'CBN')):
        return 'CB'
    return 'other'


_CLS['macro'] = _CLS.brain_region.map(fam)
CELLXYZ = _CLS.dropna(subset=['coord_SCs_x', 'coord_SCs_y', 'coord_SCs_z'])


def region_axis(M):
    """PC1 of a structure's own label matrix, with loadings, signed to that
    structure's own anchored fraction."""
    frac = np.nanmean(M, axis=0)
    Mc = np.nan_to_num(M - np.nanmean(M, axis=1, keepdims=True))
    U, sv, Vt = np.linalg.svd(Mc, full_matrices=False)
    pc, load = Vt[0], U[:, 0]
    if np.corrcoef(pc, np.nan_to_num(frac))[0, 1] < 0:
        pc, load = -pc, -load
    return pc, load, float(sv[0] ** 2 / np.sum(sv ** 2))


def collect(mo, dy, min_cells=10, regions=None):
    """{structure: (labels, cluster_ids)} for label-VARYING cells only.

    A cell whose labels never change carries no variance and cannot enter a PCA;
    counting it would also overstate how well a structure is sampled.
    """
    z = load_session_labels(mo, dy)
    if z is None:
        return {}
    L, ids = z['labels'], [int(c) for c in z['cluster_id']]
    g = _CLS[(_CLS.mouse == mo) & (_CLS.day == dy)].set_index(
        'cluster_id').macro.to_dict()
    out = {}
    for k, c in enumerate(ids):
        v = np.nan_to_num(L[k])
        if np.std(v) == 0:
            continue
        r = g.get(c)
        if r in ORDER and (regions is None or r in regions):
            out.setdefault(r, ([], []))
            out[r][0].append(v)
            out[r][1].append(c)
    return {r: (np.array(v), i) for r, (v, i) in out.items()
            if len(v) >= min_cells}


def top_regions(d, n=3):
    """The n best-sampled structures, in anatomical order."""
    best = sorted(d, key=lambda r: -len(d[r][0]))[:n]
    return [r for r in ORDER if r in best]


def trial_maps(mo, dy, want):
    """Trial x position maps for named clusters, NaN-aware smoothed."""
    bp, cp = vr_paths(mo, dy)
    if not (os.path.exists(bp) and os.path.exists(cp)):
        return {}
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials_all = beh['trials'].as_dataframe()
    _, orig = clip_trials(trials_all, clusters)
    keep = np.isin(trials_all.number.values.astype(int), orig)
    tn, trav = beh['trial_number'], beh['travel']
    dt = trav - (float(np.asarray(tn.values)[0]) - 1) * TL
    moving = beh['S'].threshold(3.0, method='above').time_support
    n_all = len(trials_all)
    have = [c for c in want if c in set(int(x) for x in clusters.index)]
    if not have:
        return {}
    tc = nap.compute_1d_tuning_curves(clusters[have], dt, nb_bins=n_all * NBIN,
                                      minmax=[0, n_all * TL], ep=moving)
    return {c: np.array([smooth_nanaware(r, sigma=MAP_SIGMA)
                         for r in np.asarray(tc[c]).reshape(n_all, NBIN)[keep]])
            for c in have}


def draw_row(fig, cell, mo, dy, d, regions, letter='', headline=''):
    """Draw one session into `cell`, a gridspec slot. Returns the axes used."""
    ax_ = {r: region_axis(d[r][0]) for r in regions}
    picks = {r: [d[r][1][i] for i in np.argsort(ax_[r][1])[::-1][:N_MAPS]]
             for r in regions}
    maps = trial_maps(mo, dy, [c for r in regions for c in picks[r]])
    wr = row_widths(len(regions))
    G = cell.subgridspec(1, len(wr), wspace=.26, width_ratios=wr)
    n_tr = d[regions[0]][0].shape[1]

    # ---- the slice, with this session's cells on it --------------------------
    ax = fig.add_subplot(G[0])
    con = ANA.contacts_ccf(mo)
    fit = ANA.fit_best_slice(con)
    u_vals, v_vals, grid = ANA.sample_region_grid(fit, step=SLICE_STEP)
    # neutral greys: the CELL colours are the key, and the default atlas palette
    # puts a green on MEC and a purple on visual cortex -- the same two hues as
    # the cells, pointing at different structures
    ax.imshow(grid, origin='lower', aspect='equal', cmap=ANA.region_cmap_neutral,
              norm=ANA.region_norm_neutral, zorder=1,
              extent=[u_vals[0], u_vals[-1], v_vals[0], v_vals[-1]])
    cu, cv = fit['project'](con)
    ax.scatter(cu, cv, s=.4, color='0.75', alpha=.45, lw=0, zorder=2)
    cells = CELLXYZ[(CELLXYZ.mouse == mo) & (CELLXYZ.day == dy)]
    for r in regions:
        k = cells[cells.macro == r]
        if len(k):
            pu, pv = fit['project'](ANA.to_ccf(
                k, ('coord_SCs_z', 'coord_SCs_y', 'coord_SCs_x')))
            ax.scatter(pu, pv, s=5.0, color=RCOL[r], alpha=.85, lw=0, zorder=3)
    # v increases with depth, so plotting it upward puts MEC above visual cortex
    # -- anatomically upside down. Invert so the slice reads as the brain does.
    ax.set_ylim(v_vals[-1], v_vals[0])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title('slice', fontsize=7.0, color='0.35', pad=4)

    for gi, r in enumerate(regions):
        M, ids_ = d[r]
        pc, load, var = ax_[r]
        order = np.argsort(load)[::-1]
        col0 = 1 + gi * 3

        ax = fig.add_subplot(G[col0])
        ax.imshow(M[order].T, aspect='auto', origin='lower', cmap=TA_CMAP,
                  norm=TA_NORM, interpolation='nearest')
        ax.set_xticks([]); ax.tick_params(labelsize=6)
        if gi == 0:
            ax.set_ylabel('trial', fontsize=7)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel(f'{len(M)} cells', fontsize=6.2)
        ax.set_title(f'{RNAME[r]}   PC1 {100 * var:.0f}%', fontsize=7.4,
                     color=RCOL[r], pad=4)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if gi == 0:
            if letter:
                ax.text(-2.00, 1.16, letter, transform=ax.transAxes,
                        fontsize=10, weight='bold', va='bottom', ha='right')
            ax.text(0, 1.26, headline or f'M{mo} D{dy}',
                    transform=ax.transAxes, fontsize=7.6, va='bottom',
                    ha='left', color='0.2', weight='bold')

        for k, c in enumerate(picks[r]):
            ax = fig.add_subplot(G[col0 + 1 + k])
            if c in maps:
                ax.imshow(maps[c], aspect='auto', origin='lower', cmap='viridis',
                          vmin=0, vmax=np.nanpercentile(maps[c], 99),
                          interpolation='nearest', extent=[0, TL, 0, n_tr])
            lab = M[ids_.index(c)]
            axl = ax.inset_axes([-.26, 0, .20, 1])
            axl.imshow(lab[:, None], aspect='auto', origin='lower',
                       cmap=TA_CMAP, norm=TA_NORM, interpolation='nearest')
            axl.set_xticks([]); axl.set_yticks([])
            for sp in axl.spines.values():
                sp.set_visible(False)
            ax.set_yticks([]); ax.tick_params(labelsize=6)
            ax.set_xticks([0, TL])
            ax.set_xticklabels(['0', f'{int(TL)}'], fontsize=6)
            ax.set_xlabel('cm', fontsize=6.2)
            ax.set_title(f'unit {c}', fontsize=6.6)
            for sp in ax.spines.values():
                sp.set_visible(False)

    ax = fig.add_subplot(G[len(wr) - 1])
    for r in regions:
        p_ = ax_[r][0]
        ax.plot((p_ - p_.mean()) / (p_.std() or 1), np.arange(n_tr),
                color=RCOL[r], lw=.85)
    ax.set_ylim(0, n_tr - 1); ax.set_yticks([])
    ax.tick_params(labelsize=6)
    ax.set_xlabel('PC1 (z)', fontsize=6.6)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    return ax_
