"""The visual-cortex cells that stay anchored while MEC switches, and what they track.

Figure 2K shows that MEC's lock-on bias is carried entirely by speed-modulated
cells: strip those out and MEC cells lock slightly more often into the
NON-anchored mode (199 on against 262 off). Doing the same count in every
recorded structure turns up one exception. In visual cortex, cells with no
open-field speed tuning lock into the anchored mode at 98 against 13 -- a 7.5:1
ratio, p = 3e-17, in 15 of the 19 sessions that have visual cells at all.

So a persistently track-anchored, non-speed population does exist outside MEC.
This figure asks what it is tracking, because the obvious candidate is not a
spatial signal at all: the virtual corridor looks the same on every lap, so a
cell driven by the visual scene is anchored to track position by construction,
whatever MEC is doing.

    A   example cells, trial by trial, with the population state shaded behind.
        These hold their field through transitions that reorganise MEC.
    B   every cell's mean profile, sorted by peak position. A scene-driven
        population should concentrate where the corridor has features; a
        spatial one should tile the track.
    C   where the peaks fall, against the uniform expectation, with the reward
        zone marked.
    D   THE TEST THAT DISCRIMINATES. On cued trials a visual beacon marks the
        reward zone and on uncued trials it does not, so the scene differs
        between trial types at one known location. A scene-driven cell should
        respond to that; a position-driven one should not.
    E   the same measures for MEC's non-speed locked-anchored cells, as contrast.

WHAT THIS CANNOT TEST. Screen luminance was not recorded, so "light levels"
cannot be addressed directly -- there is no luminance trace to correlate
against. What is testable is whether firing concentrates at particular track
positions (C) and whether it follows a known change in the scene (D). A uniform
corridor with no features predicts peaks spread evenly; clustering is evidence
of feature-locking, and a cued/uncued difference at the reward zone is evidence
that the visual scene, rather than position, is what the cells follow.

Writes fig2_supp_vis_anchored.pdf
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
from matplotlib.colors import BoundaryNorm, ListedColormap

sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
import pynapple as nap
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.ndimage import median_filter
from scipy.spatial.distance import pdist
from scipy.stats import kstest, wilcoxon
from spatial_manifolds.anchoring import (ANCH_COLOR, NONANCH_COLOR,
                                         load_session_labels, smooth_nanaware)

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = os.path.dirname(os.path.abspath(__file__))
PS = f'{ROOT}/data/population_state'
OUT = f'{FIG}/fig2_supp_vis_anchored.pdf'
CACHE = f'{PS}/vis_anchored_maps.npz'
SIGMA = 2.0
RZ = (90., 110.)          # reward zone
CMAP = 'viridis'          # rate maps are viridis throughout the paper
MIN_BLOCK_FRAC = .05      # a 'major' block is >=5% of the session's trials
N_CLUST = 4               # clusters cut from the profile dendrogram
# Two mice. M21D19 is the clean case -- three transitions, strong PC1 -- and
# M27D23 is the demanding one: 21 transitions over 393 trials, so a visual
# field that survives it is surviving the entorhinal population reorganising
# twenty times. M28D20 was used here before and switches only once, which is
# too weak a test to carry the panel.
EX_SESSIONS = [(21, 19), (27, 23)]

_g = {}
for _c in json.load(open(f'{FIG}/lick_raster_by_trial_type.ipynb'))['cells']:
    if _c['cell_type'] != 'code':
        continue
    _s = ''.join(_c['source'])
    if _s.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in _s:
        continue
    exec(compile(_s, '<nb>', 'exec'), _g)
globals().update({k: v for k, v in _g.items() if not k.startswith('__')})


def _lp(ax, s, dx=-.16, dy=1.0):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='left')


def target_cells():
    u = pd.read_csv(f'{PS}/unit_table.csv')
    reg = pd.read_csv(f'{PS}/pc1_by_region.csv')[['mouse', 'day', 'cluster_id',
                                                  'brain_region']]
    d = u.merge(reg, on=['mouse', 'day', 'cluster_id'], how='left',
                suffixes=('', '_r'))
    d['br'] = d.brain_region_r.fillna(d.brain_region).astype(str)
    base = d[d.sess_switches & (d.label_sd == 0) & (d.cell_frac_anch == 1)
             & (d.speed_p >= .05)]
    return (base[base.br.str.startswith('VIS')].copy(),
            base[base.br.str.startswith('ENTm')].copy())


def build_cache():
    vis, mec = target_cells()
    tt = pd.read_csv(f'{PS}/trial_table.csv')
    want = pd.concat([vis.assign(grp='VIS'), mec.assign(grp='MEC')])
    rows = []
    for (mo, dy), g in want.groupby(['mouse', 'day']):
        z = load_session_labels(int(mo), int(dy))
        if z is None:
            continue
        bp, cp = vr_paths(int(mo), int(dy))
        if not (os.path.exists(bp) and os.path.exists(cp)):
            continue
        beh = nap.load_file(bp); clusters = nap.load_file(cp)
        trials_all = beh['trials'].as_dataframe()
        _, orig = clip_trials(trials_all, clusters)
        keep = np.isin(trials_all.number.values.astype(int), orig)
        ids = [int(c) for c in z['cluster_id']]
        have = [int(c) for c in g.cluster_id
                if int(c) in set(int(x) for x in clusters.index) and int(c) in ids]
        if not have:
            continue
        tn, trav = beh['trial_number'], beh['travel']
        dt = trav - (float(np.asarray(tn.values)[0]) - 1) * TL
        moving = beh['S'].threshold(3.0, method='above').time_support
        n_all = len(trials_all)
        tc = nap.compute_1d_tuning_curves(clusters[have], dt,
                                          nb_bins=n_all * NBIN,
                                          minmax=[0, n_all * TL], ep=moving)
        trial_no = np.asarray(z['trial']).astype(int)
        cue = dict(zip(tt[(tt.mouse == mo) & (tt.day == dy)].trial.astype(int),
                       tt[(tt.mouse == mo) & (tt.day == dy)].ttype))
        ct = np.array([cue.get(int(t), '?') for t in trial_no])
        frac = np.asarray(z['frac_anch'], float)
        for c in have:
            M = np.asarray(tc[c]).reshape(n_all, NBIN)[keep]
            S = np.array([smooth_nanaware(r, sigma=SIGMA) for r in M])
            grp = g[g.cluster_id == c].grp.iloc[0]
            rows.append(dict(mouse=mo, day=dy, cluster_id=c, grp=grp,
                             maps=S.astype(np.float32),
                             cue=ct, pop=(frac > .5)))
        print(f'  M{mo}D{dy}: {len(have)} cells', flush=True)
    np.savez_compressed(CACHE, rows=np.array(rows, dtype=object),
                        allow_pickle=True)
    print(f'cached {len(rows)} cells')
    return rows


def load_rows():
    if os.path.exists(CACHE):
        return list(np.load(CACHE, allow_pickle=True)['rows'])
    return build_cache()



def mec_state(mo, dy):
    """That session's MEC anchoring raster, its PC1, and the major transitions.

    Read straight from the label file rather than cached with the rate maps:
    it costs nothing (the labels are an npz) and keeps the example panels on
    exactly the labels every other figure uses.
    """
    z = load_session_labels(int(mo), int(dy))
    reg = pd.read_csv(f'{PS}/pc1_by_region.csv')
    reg = reg[(reg.mouse == mo) & (reg.day == dy)]
    mec_ids = set(reg[reg.brain_region.astype(str).str.startswith('ENTm')]
                  .cluster_id.astype(int))
    ids = [int(c) for c in z['cluster_id']]
    rows = [i for i, c in enumerate(ids) if c in mec_ids]
    L = np.asarray(z['labels'])[rows].astype(float)
    load = np.asarray(z['pc1_load'], float)[rows]
    # Cells that never vary carry no PC1 loading and would sort arbitrarily
    # among the ones that do, so they are separated to the right of a gap as in
    # Figure 1: the varying cells in PC1 order, then the locked ones.
    varies = np.nanstd(L, axis=1) > 0
    Lv = L[varies][np.argsort(load[varies])[::-1]]
    Ll = L[~varies]
    if len(Ll):
        Ll = Ll[np.argsort(np.nanmean(Ll, axis=1))[::-1]]
    gap = max(2, int(round(.03 * len(L))))
    L = np.vstack([Lv, np.full((gap, L.shape[1]), np.nan), Ll]) if len(Ll) else Lv
    n_var, n_lock = len(Lv), len(Ll)
    pc1 = np.nan_to_num(np.asarray(z['pc1'], float))
    frac = np.asarray(z['frac_anch'], float)
    st = median_filter((frac > .5).astype(float), size=9, mode='nearest') > .5
    # MAJOR transitions only. The median filter still leaves brief excursions,
    # and a session like M27D23 has 21 crossings over 393 trials -- drawn on
    # every panel they occlude the rate maps they are meant to be read against.
    # Runs shorter than MIN_BLOCK are absorbed into the preceding state, so the
    # lines mark the block structure the figure actually refers to.
    # the threshold is a FRACTION of the session, not a fixed count: 20 trials
    # is most of a 107-trial session and a flicker in a 393-trial one, so a
    # fixed value either strips real blocks from the short sessions or leaves
    # the long ones unreadable (M27D23 keeps 21 crossings at a count of 5)
    _minblk = max(5, int(round(MIN_BLOCK_FRAC * len(st))))
    st = st.copy()
    changed = True
    while changed:
        changed = False
        edges = np.r_[0, np.where(np.diff(st.astype(int)) != 0)[0] + 1, len(st)]
        for a, b in zip(edges[:-1], edges[1:]):
            if b - a < _minblk and a > 0:
                st[a:b] = st[a - 1]
                changed = True
                break
    tr = list(np.where(np.diff(st.astype(int)) != 0)[0] + 1)
    return L, pc1, tr, n_var, n_lock, gap


if __name__ == '__main__':
    rows = load_rows()
    V = [r for r in rows if r['grp'] == 'VIS']
    M_ = [r for r in rows if r['grp'] == 'MEC']
    nb = V[0]['maps'].shape[1]
    x = (np.arange(nb) + .5) * (TL / nb)
    print(f'{len(V)} VIS cells, {len(M_)} MEC cells')

    def profile(r):
        p = np.nanmean(r['maps'], axis=0)
        return (p - np.nanmin(p)) / (np.nanmax(p) - np.nanmin(p) + 1e-12)

    def consistency(r):
        Mm = r['maps']; t = np.nanmean(Mm, axis=0) - np.nanmean(Mm)
        if np.nanstd(t) == 0:
            return -1
        v = []
        for row in Mm:
            a = row - np.nanmean(row)
            d = np.sqrt(np.nansum(a ** 2) * np.nansum(t ** 2))
            if d > 0:
                v.append(np.nansum(a * t) / d)
        return float(np.mean(v)) if v else -1

    PV = np.array([profile(r) for r in V])
    PM = np.array([profile(r) for r in M_])

    # ---- the luminance proxy ------------------------------------------------
    # Screen luminance was not recorded, but the pupil constricts in brightness,
    # so z(pupil radius) against track position is an INVERSE luminance profile.
    # Pooled over 23 sessions it is most constricted at 106 cm -- inside the
    # reward zone -- and most dilated near 22 cm, a swing of 2.2 z. The
    # position-locked component is the part that can reflect the scene; the
    # state-related component is tonic across position (Figure 7I).
    _pz = np.load(f'{ROOT}/data/eye_anchoring/eye_position_profiles_gated.npz')
    _pup = np.nanmean(np.concatenate([_pz['anch'], _pz['non']]), axis=0)
    _px = (np.arange(len(_pup)) + .5) * (TL / len(_pup))
    _pup_i = np.interp(x, _px, _pup)

    def _corr(a, b):
        a = a - np.nanmean(a); b = b - np.nanmean(b)
        d = np.sqrt(np.nansum(a ** 2) * np.nansum(b ** 2))
        return np.nan if d == 0 else float(np.nansum(a * b) / d)

    _rv = np.array([_corr(p_, _pup_i) for p_ in PV])
    _rm = np.array([_corr(p_, _pup_i) for p_ in PM])

    fig = plt.figure(figsize=(10.6, 11.4))
    outer = fig.add_gridspec(4, 1, height_ratios=[1.35, 1.15, 1.0, 1.0],
                             hspace=.60, left=.065, right=.975, top=.97,
                             bottom=.045)
    gex = outer[0].subgridspec(1, 2, wspace=.30)
    gclu = outer[1].subgridspec(1, 2, wspace=.34)
    gmid = outer[2].subgridspec(1, 3, width_ratios=[.90, 1.25, 1.05], wspace=.52)
    gbot = outer[3].subgridspec(1, 2, width_ratios=[1.0, .85], wspace=.42)

    # ---- A, B: two sessions, each with its MEC state beside its VIS cells ----
    for si, (mo, dy) in enumerate(EX_SESSIONS):
        cells = sorted([r for r in V if (r['mouse'], r['day']) == (mo, dy)],
                       key=consistency, reverse=True)[:3]
        if len(cells) < 3:
            print(f'  ! M{mo}D{dy} has only {len(cells)} cells'); continue
        L, pc1, tr, n_var, n_lock, gap = mec_state(mo, dy)
        n_tr = cells[0]['maps'].shape[0]
        # only the right-hand block carries the colourbar: both blocks use the
        # same scaling (each map normalised to its own peak) so one bar serves
        # them, and a second would just repeat it in the middle of the row
        _last = si == len(EX_SESSIONS) - 1
        _wr = [1.05, .26, .66, .66, .66] + ([.075] if _last else [])
        # Two rows sharing one trial axis. The mean-rate traces sit in the top
        # row above their own cells, and every raster -- MEC, PC1 and the three
        # visual cells -- sits in the bottom row, so a trial is at the same
        # height in all of them. Previously the traces were nested inside the
        # visual columns only, which pushed those rasters down relative to the
        # MEC raster beside them and made the comparison impossible to read off.
        gg = gex[si].subgridspec(2, len(_wr), height_ratios=[.38, 1.0],
                                 width_ratios=_wr, hspace=.10, wspace=.16)

        # the MEC anchoring raster
        ax = fig.add_subplot(gg[1, 0])
        _tac = ListedColormap([NONANCH_COLOR, ANCH_COLOR])
        _tac.set_bad('white')
        ax.imshow(np.ma.masked_invalid(L.T), aspect='auto',
                  interpolation='nearest', cmap=_tac,
                  norm=BoundaryNorm([-.5, .5, 1.5], 2),
                  extent=[0, L.shape[0], L.shape[1] - .5, -.5])
        if n_lock:
            ax.annotate('varies (PC1 order)', (n_var / 2, 1.004),
                        xycoords=('data', 'axes fraction'), ha='center',
                        va='bottom', fontsize=5.6, color='0.35')
            ax.annotate(f'locked ({n_lock})', (n_var + gap + n_lock / 2, 1.004),
                        xycoords=('data', 'axes fraction'), ha='center',
                        va='bottom', fontsize=5.6, color='0.35')
        for t_ in tr:
            ax.axhline(t_, color='k', lw=.8, ls='--', zorder=4)
        ax.set_ylabel('Trial', fontsize=8)
        ax.set_xlabel('MEC cell', fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        # the session label goes in the empty top-left slot, clear of the
        # column sub-titles beside it
        axt = fig.add_subplot(gg[0, 0]); axt.axis('off')
        axt.text(0, .18, f'M{mo} D{dy}', fontsize=8, va='bottom')
        _lp(axt, 'AB'[si], dx=-.30, dy=.10)

        # PC1, on the same trial axis
        axp = fig.add_subplot(gg[1, 1], sharey=ax)
        y_ = np.arange(len(pc1))
        axp.fill_betweenx(y_, 0, pc1, where=pc1 >= 0, color=ANCH_COLOR, lw=0,
                          interpolate=True)
        axp.fill_betweenx(y_, 0, pc1, where=pc1 < 0, color=NONANCH_COLOR, lw=0,
                          interpolate=True)
        axp.axvline(0, color='0.4', lw=.7)
        for t_ in tr:
            axp.axhline(t_, color='k', lw=.9, ls='--', zorder=4)
        axp.set_xticks([]); axp.tick_params(labelleft=False, left=False)
        axp.set_title('PC1', fontsize=6.8, pad=3)
        for sp in axp.spines.values():
            sp.set_visible(False)

        _im = None
        for k, r in enumerate(cells):
            # mean rate against position IN EACH STATE, above its own cell
            axq = fig.add_subplot(gg[0, 2 + k])
            st_ = median_filter(r['pop'].astype(float), size=9,
                                mode='nearest') > .5
            for m_, c_ in ((st_, ANCH_COLOR), (~st_, NONANCH_COLOR)):
                if m_.sum() >= 5:
                    axq.plot(x, np.nanmean(r['maps'][m_], 0), color=c_, lw=1.0)
            axq.set_xlim(0, TL); axq.set_xticks([])
            axq.tick_params(labelsize=5.5)
            axq.set_title(f'cl {r["cluster_id"]}', fontsize=6, pad=2)
            if k == 0:
                axq.set_ylabel('Hz', fontsize=6)
            axq.spines[['top', 'right']].set_visible(False)

            axc = fig.add_subplot(gg[1, 2 + k], sharey=ax)
            _mx = np.nanmax(r['maps']) or 1.0
            _im = axc.imshow(r['maps'] / _mx, aspect='auto', cmap=CMAP,
                             vmin=0, vmax=1, interpolation='nearest',
                             extent=[0, TL, r['maps'].shape[0] - .5, -.5])
            for t_ in tr:
                if t_ < r['maps'].shape[0]:
                    axc.axhline(t_, color='w', lw=.8, ls='--', alpha=.85,
                                zorder=4)
            axc.set_xticks([0, 100, 200]); axc.tick_params(labelsize=6)
            axc.tick_params(labelleft=False)
            if k == 1:
                axc.set_xlabel('Position (cm)', fontsize=7.5)
            for sp in axc.spines.values():
                sp.set_visible(False)
        if _last:
            axcb = fig.add_subplot(gg[1, 5])
            cb = fig.colorbar(_im, cax=axcb)
            cb.ax.set_title('rate\n(/peak)', fontsize=5.4, pad=3,
                            linespacing=1.1)
            cb.ax.tick_params(labelsize=5.4); cb.outline.set_visible(False)

    # ---- C, D: each population clustered on its profile shape ---------------
    # Sorting by peak position imposes a diagonal whether or not the population
    # has structure. Clustering on profile SHAPE lets the groups declare
    # themselves: the dendrogram shows how separable they are, and each
    # cluster's mean gets its own axes on the right so a flat cluster reads as
    # flat instead of being stretched to fill a band.
    _cpal = ['#c0553a', '#d4a017', '#2f8f7a', '#2b4a7a', '#b5485c']
    for _k, (_P, _nm) in enumerate(((PV, 'VIS'), (PM, 'MEC'))):
        gg = gclu[_k].subgridspec(1, 3, width_ratios=[.30, 1.0, .60],
                                  wspace=.10)
        # correlation distance: cells with the same field shape at different
        # rates should group together, and the profiles are normalised per cell
        Zl = linkage(pdist(_P, metric='correlation'), method='average')
        lab_ = fcluster(Zl, N_CLUST, criterion='maxclust')
        n_ = len(_P)

        # colour each link by the cluster of its descendants, grey above the
        # cut -- done explicitly rather than via color_threshold so the branch
        # colours and the cluster means are guaranteed to be the same mapping
        _lc = {}
        for _i, (_a, _b, _d, _c) in enumerate(Zl):
            _a, _b = int(_a), int(_b)
            _ca = lab_[_a] if _a < n_ else _lc.get(_a)
            _cb = lab_[_b] if _b < n_ else _lc.get(_b)
            _lc[n_ + _i] = _ca if (_ca is not None and _ca == _cb) else None

        axd = fig.add_subplot(gg[0])
        dn = dendrogram(Zl, orientation='left', ax=axd, no_labels=True,
                        link_color_func=lambda i: '0.45')
        order = dn['leaves']
        axd.invert_yaxis()           # leaf order now reads top-down
        ordered = lab_[order]
        # map cluster id -> palette entry by where it first appears, top-down,
        # so the top block is cluster 1
        seen, cmap_ = [], {}
        for c_ in ordered:
            if c_ not in seen:
                seen.append(c_); cmap_[c_] = _cpal[(len(seen) - 1) % len(_cpal)]
        for _coll, _link in zip(axd.collections, [None]):
            pass
        # recolour the drawn links
        axd.clear()
        dn = dendrogram(Zl, orientation='left', ax=axd, no_labels=True,
                        link_color_func=lambda i: cmap_.get(_lc.get(i), '0.45'))
        axd.invert_yaxis()
        axd.set_xticks([]); axd.set_yticks([])
        axd.set_ylabel('Cells', fontsize=8, labelpad=2)
        for sp in axd.spines.values():
            sp.set_visible(False)
        _lp(axd, 'CD'[_k], dx=-.22, dy=1.0)

        axh = fig.add_subplot(gg[1])
        axh.imshow(_P[order], aspect='auto', cmap=CMAP, origin='upper',
                   interpolation='nearest', extent=[0, TL, 10 * n_, 0])
        for b in RZ:
            axh.axvline(b, color='w', lw=1.0, ls='--')
        bounds = np.where(np.diff(ordered) != 0)[0] + 1
        for b_ in bounds:
            axh.axhline(10 * b_, color='w', lw=1.1)
        axh.set_ylim(10 * n_, 0)
        axh.set_yticks([5, 10 * n_ - 5]); axh.set_yticklabels(['1', str(n_)],
                                                              fontsize=6.5)
        axh.set_xlabel('Position (cm)', fontsize=8)
        axh.tick_params(labelsize=7)
        axh.set_title(f'{_nm}: {n_} cells, clustered on profile shape',
                      fontsize=7.5, loc='left')

        # one axes per cluster, stacked, each with its own mean +/- SEM
        edges = np.r_[0, bounds, n_]
        gp = gg[2].subgridspec(len(edges) - 1, 1, hspace=.46)
        for ci, (a_, b_) in enumerate(zip(edges[:-1], edges[1:])):
            memb = _P[np.array(order)[a_:b_]]
            mu = memb.mean(0); se = memb.std(0) / np.sqrt(len(memb))
            c_ = cmap_[ordered[a_]]
            axc2 = fig.add_subplot(gp[ci])
            axc2.fill_between(x, mu - se, mu + se, color=c_, alpha=.30, lw=0)
            axc2.plot(x, mu, color=c_, lw=1.2)
            axc2.axvspan(*RZ, color='#d8e4d0', alpha=.55, lw=0, zorder=0)
            axc2.set_xlim(0, TL)
            axc2.text(1.0, 1.01, f'cluster {ci + 1} (n={b_ - a_})', color=c_,
                      fontsize=5.8, ha='right', va='bottom',
                      transform=axc2.transAxes, clip_on=False)
            axc2.set_yticks([]); axc2.tick_params(labelsize=6.2)
            if ci == len(edges) - 2:
                axc2.set_xticks([0, 100, 200])
                axc2.set_xlabel('Position (cm)', fontsize=7.5)
            else:
                axc2.set_xticks([])
            if ci == 0:
                axc2.set_title('cluster means', fontsize=6.8, loc='left',
                               pad=10)
            for sp in ('top', 'right', 'left'):
                axc2.spines[sp].set_visible(False)

    # ---- E: peak positions, cumulative --------------------------------------
    # Cumulative rather than binned: the KS statistic beneath the panel is a
    # distance between these curves and the diagonal, so the plot and the test
    # are then the same object, and no bin width has to be chosen.
    ax = fig.add_subplot(gmid[0]); _lp(ax, 'E', dx=-.30)
    pk = x[np.argmax(PV, axis=1)]; pkm = x[np.argmax(PM, axis=1)]
    for v_, c_, lab in ((pk, '#8C6BB1', f'VIS ({len(pk)})'),
                        (pkm, '0.35', f'MEC ({len(pkm)})')):
        ax.step(np.r_[0, np.sort(v_), TL],
                np.r_[0, np.arange(1, len(v_) + 1) / len(v_), 1.0],
                where='post', color=c_, lw=1.4, label=lab)
    ax.plot([0, TL], [0, 1], color='0.6', lw=.9, ls=':', label='uniform')
    ax.axvspan(*RZ, color='#d8e4d0', alpha=.55, lw=0, zorder=0)
    ks = kstest(pk / TL, 'uniform'); ksm = kstest(pkm / TL, 'uniform')
    ax.set_xlim(0, TL); ax.set_ylim(0, 1)
    ax.set_xlabel('peak position (cm)', fontsize=8)
    ax.set_ylabel('cumulative fraction', fontsize=8)
    ax.set_title(f'VIS vs uniform p = {ks.pvalue:.1g}\n'
                 f'MEC vs uniform p = {ksm.pvalue:.1g}', fontsize=7, loc='left')
    ax.legend(fontsize=6, frameon=False, loc='upper left')
    ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

    # ---- E: one cell, trials sorted by type, means above ---------------------
    def cue_split(rs):
        a, b = [], []
        for r in rs:
            for tag, acc in (('b', a), ('nb', b)):
                m = r['cue'] == tag
                if m.sum() >= 5:
                    p = np.nanmean(r['maps'][m], axis=0)
                    acc.append((p - np.nanmean(p)) / (np.nanstd(p) + 1e-12))
        return np.array(a), np.array(b)
    CA, CB = cue_split(V)
    n = min(len(CA), len(CB))
    rzm = (x >= RZ[0]) & (x <= RZ[1])
    w = wilcoxon(CA[:n][:, rzm].mean(1), CB[:n][:, rzm].mean(1))

    def _cue_gain(r):
        a, b = r['cue'] == 'b', r['cue'] == 'nb'
        if a.sum() < 5 or b.sum() < 5:
            return -9
        pa = np.nanmean(r['maps'][a], 0); pb = np.nanmean(r['maps'][b], 0)
        sc = np.nanstd(np.nanmean(r['maps'], 0)) + 1e-12
        return float((np.nanmean(pa[rzm]) - np.nanmean(pb[rzm])) / sc)
    ex = max(V, key=_cue_gain)
    gd = gmid[1].subgridspec(2, 1, height_ratios=[.50, 1.0], hspace=.12)
    axm = fig.add_subplot(gd[0])
    for tag, c_, lab in (('b', '#c04744', 'cued (beacon)'),
                         ('nb', '#2b6cb0', 'uncued')):
        axm.plot(x, np.nanmean(ex['maps'][ex['cue'] == tag], 0), color=c_,
                 lw=1.3, label=lab)
    axm.axvspan(*RZ, color='#d8e4d0', alpha=.55, lw=0, zorder=0)
    axm.set_xlim(0, TL); axm.tick_params(labelbottom=False, labelsize=6.5)
    axm.set_ylabel('rate (Hz)', fontsize=7)
    axm.legend(fontsize=5.8, frameon=False, loc='upper right')
    axm.set_title(f'M{ex["mouse"]}D{ex["day"]} cl {ex["cluster_id"]} — trials '
                  f'sorted by type\npopulation: reward zone p = {w.pvalue:.2g} '
                  f'({n} cells)', fontsize=7.4, loc='left')
    axm.spines[['top', 'right']].set_visible(False)
    _lp(axm, 'F', dx=-.17, dy=1.02)
    ax = fig.add_subplot(gd[1], sharex=axm)
    o_cue = np.argsort(ex['cue'] != 'b')
    ax.imshow(ex['maps'][o_cue], aspect='auto', cmap=CMAP,
              interpolation='nearest', extent=[0, TL, len(o_cue) - .5, -.5])
    _nb = int((ex['cue'] == 'b').sum())
    ax.axhline(_nb - .5, color='w', lw=1.2)
    ax.text(TL * .985, _nb * .5, 'cued', color='w', fontsize=6, ha='right',
            va='center', rotation=90)
    ax.text(TL * .985, _nb + (len(o_cue) - _nb) * .5, 'uncued', color='w',
            fontsize=6, ha='right', va='center', rotation=90)
    for b in RZ:
        ax.axvline(b, color='w', lw=.9, ls='--')
    ax.set_xlabel('Position (cm)', fontsize=8)
    ax.set_ylabel('Trial', fontsize=8); ax.tick_params(labelsize=7)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # ---- F: the luminance proxy ---------------------------------------------
    ax = fig.add_subplot(gmid[2]); _lp(ax, 'G', dx=-.26)
    axb = ax.twinx()
    axb.plot(x, _pup_i, color='#b8860b', lw=1.4, zorder=3)
    axb.set_ylabel('z(pupil) — dilated = darker', fontsize=6.8,
                   color='#b8860b', labelpad=2)
    axb.tick_params(labelsize=6.5, colors='#b8860b')
    ax.plot(x, PV.mean(0), color='#8C6BB1', lw=1.5, zorder=4)
    ax.axvspan(*RZ, color='#d8e4d0', alpha=.55, lw=0, zorder=0)
    ax.set_xlabel('Position (cm)', fontsize=8)
    ax.set_ylabel('VIS firing (normalised)', fontsize=8, color='#8C6BB1')
    _wv = wilcoxon(_rv[np.isfinite(_rv)])
    ax.set_title(f'against the luminance proxy\n'
                 f'VIS r = {np.nanmean(_rv):+.2f} (p = {_wv.pvalue:.1g})',
                 fontsize=7.4, loc='left')
    ax.tick_params(labelsize=7); ax.spines[['top']].set_visible(False)

    # ---- G: per-cell correlations, as a boxplot -----------------------------
    ax = fig.add_subplot(gbot[0]); _lp(ax, 'H', dx=-.22)
    _df = pd.DataFrame({'r': np.r_[_rv[np.isfinite(_rv)], _rm[np.isfinite(_rm)]],
                        'region': (['VIS'] * int(np.isfinite(_rv).sum())
                                   + ['MEC'] * int(np.isfinite(_rm).sum()))})
    sns.boxplot(data=_df, x='region', y='r', hue='region', legend=False,
                palette={'VIS': '#8C6BB1', 'MEC': '0.65'}, width=.6,
                fliersize=0, linewidth=.9, ax=ax)
    sns.stripplot(data=_df, x='region', y='r', hue='region', legend=False,
                  palette={'VIS': '#5e4078', 'MEC': '0.35'}, size=2.2,
                  alpha=.55, jitter=.22, ax=ax)
    ax.axhline(0, color='0.6', lw=.8, ls=':')
    for i_, reg_ in enumerate(['VIS', 'MEC']):
        v_ = _df.r[_df.region == reg_]
        ax.text(i_, 1.015, f'p = {wilcoxon(v_).pvalue:.1g}', ha='center',
                va='bottom', fontsize=6.2, color='0.3',
                transform=ax.get_xaxis_transform())
    ax.set_xlabel(''); ax.set_ylabel('corr. with pupil profile', fontsize=8,
                                     labelpad=1)
    ax.set_title('positive = fires where it is darker', fontsize=7.4,
                 loc='left', pad=16)
    ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

    # ---- H: the counts by structure -----------------------------------------
    ax = fig.add_subplot(gbot[1]); _lp(ax, 'I', dx=-.26)
    u = pd.read_csv(f'{PS}/unit_table.csv')
    reg = pd.read_csv(f'{PS}/pc1_by_region.csv')[['mouse', 'day', 'cluster_id',
                                                  'brain_region']]
    d = u.merge(reg, on=['mouse', 'day', 'cluster_id'], how='left',
                suffixes=('', '_r'))
    d['br'] = d.brain_region_r.fillna(d.brain_region).astype(str)
    sw = d[d.sess_switches].dropna(subset=['speed_p'])
    lk = sw[(sw.label_sd == 0) & (sw.speed_p >= .05)]
    labs, ons, offs = [], [], []
    for nm, pref in (('VIS', 'VIS'), ('MEC', 'ENTm'), ('PAR', 'PAR')):
        g = lk[lk.br.str.startswith(pref)]
        labs.append(nm); ons.append(int((g.cell_frac_anch == 1).sum()))
        offs.append(int((g.cell_frac_anch == 0).sum()))
    xx = np.arange(len(labs))
    ax.bar(xx - .19, ons, .38, color=ANCH_COLOR, lw=0, label='locked anchored')
    ax.bar(xx + .19, offs, .38, color=NONANCH_COLOR, lw=0,
           label='locked non-anchored')
    for i, (o_, f_) in enumerate(zip(ons, offs)):
        ax.text(i, max(o_, f_) * 1.05, f'{o_/max(f_,1):.1f}:1', ha='center',
                fontsize=6.4, color='0.3')
    ax.set_xticks(xx); ax.set_xticklabels(labs, fontsize=7.5)
    ax.set_ylabel('cells with no open-field\nspeed tuning', fontsize=8)
    ax.set_title('only VIS locks anchored', fontsize=7.5, loc='left')
    ax.legend(fontsize=5.8, frameon=False, loc='upper right')
    ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

    plt.savefig(OUT, dpi=200, bbox_inches='tight')
    print(f'wrote {OUT}')
