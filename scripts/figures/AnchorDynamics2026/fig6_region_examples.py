"""Figure 6 lead-in: the same anchoring state, seen separately in each structure.

Figure 6's statistics are abstract -- split-half reliabilities, correlations
between principal components corrected for attenuation. They are easier to
believe, and easier to doubt in the right places, after seeing the states
themselves and the single cells underneath them.

Two sessions are shown, chosen to BRACKET the range rather than to represent it:
the strongest and the weakest entorhinal-visual agreement among sessions
recording all three structures with at least MIN_REGION_CELLS label-varying cells
each. Picking the two best would misrepresent Figure 6C, whose disattenuated mean
of +0.863 carries a confidence interval of [0.49, 1.23] precisely because
sessions differ this much.

THE THRESHOLD IS 20 CELLS PER STRUCTURE, NOT 10, AND THAT MATTERS FOR WHICH
SESSIONS APPEAR. At 10 the worst session reaches -0.49, but it has 10 subicular
and 21 visual cells, so its axes are barely estimated and the anti-correlation is
as likely to be noise as signal -- bracketing on extremes is exactly the
selection most vulnerable to it. Requiring 20 keeps the examples legible and the
bracket meaningful. The population statistics in Figure 6 are unaffected: they
use every qualifying session and correct for reliability explicitly, which is
what this figure cannot do for a single session.

NOTHING IS SCORED AGAINST THE ENTORHINAL AXIS. Within a row, each raster is built
only from that region's own cells: ordered by THAT region's PC1 loading, with the
component signed to THAT region's own anchored fraction. Agreement between the
three is therefore agreement between independent descriptions of one session, not
a consequence of a shared frame.

THE RATE MAPS ARE THE POINT OF THE FIGURE. Two cells per region, the two with the
largest PC1 loading in that region, shown as trial x position maps with the
cell's own anchored/non-anchored labels as a strip beside them. They show what
the raster summarises -- firing that is in register across anchored trials and
not across non-anchored ones.

A VISUAL CELL'S MAP IS NOT A PLACE FIELD, and the figure should not be read as
claiming it is. The classifier operates on the trial-to-trial similarity of a
cell's position-binned profile, whatever produces that profile; in visual cortex
on a virtual track that is largely the position-locked visual scene rather than
an allocentric field. The claim is that those profiles are consistent on the same
trials in visual cortex as in MEC, not that visual cells carry a spatial code of
the entorhinal kind.

Writes fig6_region_examples.pdf
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pynapple as nap
from matplotlib.colors import BoundaryNorm, ListedColormap

sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import (ANCH_COLOR, NONANCH_COLOR,
                                         load_session_labels, smooth_nanaware)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anatomy_slice as ANA

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = f'{ROOT}/scripts/figures/AnchorDynamics2026'
PS = f'{ROOT}/data/population_state'
OUT = f'{FIG}/fig6_region_examples.pdf'

# shared session plumbing (vr_paths, clip_trials, TL, NBIN)
_g = {}
for _c in json.load(open(f'{FIG}/lick_raster_by_trial_type.ipynb'))['cells']:
    if _c['cell_type'] != 'code':
        continue
    _s = ''.join(_c['source'])
    if _s.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in _s:
        continue
    exec(compile(_s, '<nb>', 'exec'), _g)
globals().update({k: v for k, v in _g.items() if not k.startswith('__')})

# declared AFTER the notebook exec on purpose: globals().update() above would
# overwrite anything declared before it, and the notebook defines its own
# MIN_CELLS and SIGMA. Prefixed names so the clobbering cannot happen silently
# in either direction; TL and NBIN are taken FROM the notebook and not redefined.
REGIONS = ['MEC', 'SUB', 'VIS']
RNAME = {'MEC': 'MEC', 'SUB': 'SUB/PARA', 'VIS': 'VIS'}
RCOL = {'MEC': '#7b4173', 'VIS': '#3f9b4f', 'SUB': '#c0723a'}
MIN_REGION_CELLS = 20
N_MAPS = 2
MAP_SIGMA = 2.0
TA_CMAP = ListedColormap([NONANCH_COLOR, ANCH_COLOR])
TA_NORM = BoundaryNorm([-.5, .5, 1.5], TA_CMAP.N)


def fam(r):
    r = str(r)
    if r.startswith('ENTm'):
        return 'MEC'
    if any(r.startswith(x) for x in ('PRE', 'POST', 'PAR', 'SUB')):
        return 'SUB'
    if r.startswith('VIS'):
        return 'VIS'
    return 'other'


C = pd.read_csv(f'{ROOT}/data/cell_classifications_v2.csv')
C['macro'] = C.brain_region.map(fam)
CELLXYZ = C.dropna(subset=['coord_SCs_x', 'coord_SCs_y', 'coord_SCs_z'])


def region_axis(M):
    """PC1 of a region's own label matrix, with cell loadings, signed to that
    region's own anchored fraction."""
    frac = np.nanmean(M, axis=0)
    Mc = np.nan_to_num(M - np.nanmean(M, axis=1, keepdims=True))
    U, sv, Vt = np.linalg.svd(Mc, full_matrices=False)
    pc, load = Vt[0], U[:, 0]
    if np.corrcoef(pc, np.nan_to_num(frac))[0, 1] < 0:
        pc, load = -pc, -load
    return pc, load, float(sv[0] ** 2 / np.sum(sv ** 2))


def collect(mo, dy):
    z = load_session_labels(mo, dy)
    if z is None:
        return None
    L, ids = z['labels'], [int(c) for c in z['cluster_id']]
    g = C[(C.mouse == mo) & (C.day == dy)].set_index('cluster_id').macro.to_dict()
    out = {}
    for k, c in enumerate(ids):
        v = np.nan_to_num(L[k])
        if np.std(v) == 0:
            continue
        r = g.get(c)
        if r in REGIONS:
            out.setdefault(r, ([], []))
            out[r][0].append(v)
            out[r][1].append(c)
    out = {r: (np.array(v), ids_) for r, (v, ids_) in out.items()
           if len(v) >= MIN_REGION_CELLS}
    return out if len(out) == len(REGIONS) else None


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
    out = {}
    for c in have:
        M = np.asarray(tc[c]).reshape(n_all, NBIN)[keep]
        out[c] = np.array([smooth_nanaware(r, sigma=MAP_SIGMA) for r in M])
    return out


# ---- choose the two sessions that bracket the range --------------------------
sess = sorted({(int(f.split('M')[1].split('D')[0]),
                int(f.split('D')[1].split('.')[0]))
               for f in os.listdir(f'{PS}/labels') if f.endswith('.npz')})
cands = []
for mo, dy in sess:
    d = collect(mo, dy)
    if d is None:
        continue
    ax_ = {r: region_axis(d[r][0]) for r in REGIONS}
    mv = float(np.corrcoef(ax_['MEC'][0], ax_['VIS'][0])[0, 1])
    cands.append(dict(mo=mo, dy=dy, d=d, ax=ax_, mv=mv))
cands.sort(key=lambda x: -x['mv'])
use = [dict(cands[0], tag='strongest agreement'),
       dict(cands[-1], tag='weakest agreement')]
print(f'{len(cands)} sessions with all three structures >= {MIN_REGION_CELLS} varying '
      f'cells; entorhinal–visual agreement runs '
      f'{cands[-1]["mv"]:+.2f} to {cands[0]["mv"]:+.2f}. Showing the two extremes:')
for u in use:
    print(f'  M{u["mo"]} D{u["dy"]}  ({u["tag"]}, M–V {u["mv"]:+.2f}): '
          + ', '.join(f'{r} {len(u["d"][r][0])}' for r in REGIONS))

# ============================== figure =========================================
# One row per session. Ten columns -- for each structure a raster and two cells,
# then the three components overlaid -- so a reader scans a single session left
# to right and compares sessions top to bottom. This is wider than tall, so the
# page is landscape; the alternative (a three-row block per session) stacked the
# structures vertically and made the across-structure comparison, which is the
# point, the harder of the two to make.
fig = plt.figure(figsize=(10, 4.8))
OUTER = fig.add_gridspec(len(use), 1, hspace=.62)
# leftmost column is the session's recording slice, so the rasters beside it can
# be read against where the contributing cells actually sit
WR = [1.70, 1.20, .78, .78, 1.20, .78, .78, 1.20, .78, .78, 1.00]
SLICE_STEP = 12

for blk, u in enumerate(use):
    mo, dy, d, ax_ = u['mo'], u['dy'], u['d'], u['ax']
    picks = {}
    for r in REGIONS:
        M, ids_ = d[r]
        load = ax_[r][1]
        picks[r] = [ids_[i] for i in np.argsort(load)[::-1][:N_MAPS]]
    maps = trial_maps(mo, dy, [c for r in REGIONS for c in picks[r]])

    G = OUTER[blk].subgridspec(1, len(WR), wspace=.26, width_ratios=WR)
    n_tr = d['MEC'][0].shape[1]

    # ---- the session's slice, with the cells that build the rasters ----------
    ax = fig.add_subplot(G[0])
    con = ANA.contacts_ccf(mo)
    fit = ANA.fit_best_slice(con)
    u_vals, v_vals, grid = ANA.sample_region_grid(fit, step=SLICE_STEP)
    # neutral greys: the CELL colours are the key here, and the default atlas
    # palette would put a green on MEC and a purple on visual cortex, i.e. the
    # same two hues as the cells but pointing at different structures
    ax.imshow(grid, origin='lower', aspect='equal', cmap=ANA.region_cmap_neutral,
              norm=ANA.region_norm_neutral, zorder=1,
              extent=[u_vals[0], u_vals[-1], v_vals[0], v_vals[-1]])
    cu, cv = fit['project'](con)
    ax.scatter(cu, cv, s=.4, color='0.75', alpha=.45, lw=0, zorder=2)
    cells = CELLXYZ[(CELLXYZ.mouse == mo) & (CELLXYZ.day == dy)]
    for r in REGIONS:
        k = cells[cells.macro == r]
        if not len(k):
            continue
        pts = ANA.to_ccf(k, ('coord_SCs_z', 'coord_SCs_y', 'coord_SCs_x'))
        pu, pv = fit['project'](pts)
        ax.scatter(pu, pv, s=5.0, color=RCOL[r], alpha=.85, lw=0, zorder=3)
    # v increases with depth (MEC mean v 479, SUB 426, VIS 117), so plotting it
    # upward puts MEC above visual cortex -- anatomically upside down. Invert so
    # the slice reads the way the brain does: VIS dorsal, MEC ventral to it.
    ax.set_ylim(v_vals[-1], v_vals[0])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title('slice', fontsize=7.0, color='0.35', pad=4)

    for gi, r in enumerate(REGIONS):
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
        ax.set_title(f'{RNAME[r]}   PC1 {100 * var:.0f}%', fontsize=7.6,
                     color=RCOL[r], pad=4)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if gi == 0:
            ax.text(-2.00, 1.16, 'AB'[blk], transform=ax.transAxes, fontsize=10,
                    weight='bold', va='bottom', ha='right')
            ms0 = np.corrcoef(ax_['MEC'][0], ax_['SUB'][0])[0, 1]
            mv0 = np.corrcoef(ax_['MEC'][0], ax_['VIS'][0])[0, 1]
            ax.text(0, 1.26, f'M{mo} D{dy} — {u["tag"]}   '
                             f'(M–S {ms0:+.2f},  M–V {mv0:+.2f})',
                    transform=ax.transAxes, fontsize=7.6, va='bottom',
                    ha='left', color='0.2', weight='bold')

        for k, c in enumerate(picks[r]):
            ax = fig.add_subplot(G[col0 + 1 + k])
            if c in maps:
                # scaled to the 99th percentile, as in Figure 2
                ax.imshow(maps[c], aspect='auto', origin='lower', cmap='viridis',
                          vmin=0, vmax=np.nanpercentile(maps[c], 99),
                          interpolation='nearest', extent=[0, TL, 0, n_tr])
            lab = M[ids_.index(c)]
            axl = ax.inset_axes([-.26, 0, .20, 1])
            axl.imshow(lab[:, None], aspect='auto', origin='lower', cmap=TA_CMAP,
                       norm=TA_NORM, interpolation='nearest')
            axl.set_xticks([]); axl.set_yticks([])
            for sp in axl.spines.values():
                sp.set_visible(False)
            ax.set_yticks([]); ax.tick_params(labelsize=6)
            ax.set_xticks([0, TL]); ax.set_xticklabels(['0', f'{int(TL)}'],
                                                       fontsize=6)
            ax.set_xlabel('cm', fontsize=6.2)
            ax.set_title(f'unit {c}', fontsize=6.6)
            for sp in ax.spines.values():
                sp.set_visible(False)

    ax = fig.add_subplot(G[10])
    for rr in REGIONS:
        p_ = ax_[rr][0]
        z = (p_ - np.mean(p_)) / (np.std(p_) or 1)
        ax.plot(z, np.arange(n_tr), color=RCOL[rr], lw=.85)
    ax.set_ylim(0, n_tr - 1); ax.set_yticks([])
    ax.tick_params(labelsize=6)
    ax.set_xlabel('PC1 (z)', fontsize=6.6)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    if blk == 0:
        ax.set_title('all three', fontsize=6.8, color='0.3', pad=4)

fig.suptitle("Each structure describes the session on its own — nothing scored "
             "against MEC", fontsize=8.2, y=1.06)
fig.savefig(OUT, bbox_inches='tight')
print(f'\nwrote {OUT}')
