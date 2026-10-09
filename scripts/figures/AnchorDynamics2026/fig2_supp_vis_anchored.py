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

import anatomy_slice as ANA
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


# the probe maps are the same construction as Figure S7's: the session's
# best-fit slice through the atlas in neutral greys, every contact in pale
# grey, and the cells that build the rasters coloured by structure
RCOL = {'MEC': '#7b4173', 'VIS': '#3f9b4f'}
SLICE_STEP = 12


def _macro(r):
    r = str(r)
    if r.startswith('ENTm'):
        return 'MEC'
    if r.startswith('VIS'):
        return 'VIS'
    return 'other'


_CXYZ = pd.read_csv(f'{ROOT}/data/cell_classifications_v2.csv')
_CXYZ['macro'] = _CXYZ.brain_region.map(_macro)
CELLXYZ = _CXYZ.dropna(subset=['coord_SCs_x', 'coord_SCs_y', 'coord_SCs_z'])


def draw_slice(ax, mo, dy):
    """Figure S7's probe map, for one session."""
    con = ANA.contacts_ccf(mo)
    fit = ANA.fit_best_slice(con)
    u_vals, v_vals, grid = ANA.sample_region_grid(fit, step=SLICE_STEP)
    ax.imshow(grid, origin='lower', aspect='equal', zorder=1,
              cmap=ANA.region_cmap_neutral, norm=ANA.region_norm_neutral,
              extent=[u_vals[0], u_vals[-1], v_vals[0], v_vals[-1]])
    cu, cv = fit['project'](con)
    ax.scatter(cu, cv, s=.4, color='0.75', alpha=.45, lw=0, zorder=2)
    cells = CELLXYZ[(CELLXYZ.mouse == mo) & (CELLXYZ.day == dy)]
    for r_, c_ in RCOL.items():
        k = cells[cells.macro == r_]
        if not len(k):
            continue
        pts = ANA.to_ccf(k, ('coord_SCs_z', 'coord_SCs_y', 'coord_SCs_x'))
        pu, pv = fit['project'](pts)
        ax.scatter(pu, pv, s=4.0, color=c_, alpha=.85, lw=0, zorder=3)
    # v increases with depth, so plotting it upward would put MEC above visual
    # cortex; invert so the slice reads the way the brain does
    ax.set_ylim(v_vals[-1], v_vals[0])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


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
    # EVERY locked-anchored cell of each structure, speed-modulated or not.
    # An earlier version kept only cells with no open-field speed tuning, which
    # meant selecting on a variable the figure then goes on to test. The speed
    # tuning is carried as a flag instead, so a panel can split on it where
    # that is the question and pool otherwise. It costs nothing: the census is
    # 7.7:1 for visual cortex against 1.1:1 for entorhinal without the filter,
    # against 7.5:1 and 0.8:1 with it.
    lock = d[d.sess_switches & (d.label_sd == 0) & (d.cell_frac_anch == 1)].copy()
    lock['spd'] = lock.speed_p < .05
    return (lock[lock.br.str.startswith('VIS')].copy(),
            lock[lock.br.str.startswith('ENTm')].copy())


def build_cache():
    vis, ent = target_cells()
    tt = pd.read_csv(f'{PS}/trial_table.csv')
    want = pd.concat([vis.assign(grp='VIS'), ent.assign(grp='ENT')])
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
            _row = g[g.cluster_id == c].iloc[0]
            rows.append(dict(mouse=mo, day=dy, cluster_id=c, grp=_row.grp,
                             spd=bool(_row.spd), maps=S.astype(np.float32),
                             cue=ct, pop=(frac > .5)))
        print(f'  M{mo}D{dy}: {len(have)} cells', flush=True)
    np.savez_compressed(CACHE, rows=np.array(rows, dtype=object),
                        allow_pickle=True)
    print(f'cached {len(rows)} cells')
    return rows


EX_CACHE = f'{PS}/vis_anchored_mec_examples.npz'
SPD_CACHE = f'{PS}/speed_profile.npz'


def speed_profile(sessions, nb):
    """Mean running speed against track position, over the given sessions.

    The track has a stereotyped speed profile -- the animals slow into the
    reward zone and run fastest after it -- so position and speed are
    confounded by construction here. This is the regressor that lets the
    confound be measured instead of argued about.
    """
    if os.path.exists(SPD_CACHE):
        z_ = np.load(SPD_CACHE)
        if len(z_['profile']) == nb:
            return z_['profile']
    acc = []
    for mo, dy in sessions:
        bp, _ = vr_paths(int(mo), int(dy))
        if not os.path.exists(bp):
            continue
        beh = nap.load_file(bp)
        S_ = beh['S']; trav = beh['travel']
        t_ = np.asarray(S_.index); v_ = np.asarray(S_.values)
        pos = np.interp(t_, np.asarray(trav.index),
                        np.asarray(trav.values)) % TL
        ok = v_ >= 3.0
        num, _ = np.histogram(pos[ok], bins=nb, range=(0, TL), weights=v_[ok])
        den, _ = np.histogram(pos[ok], bins=nb, range=(0, TL))
        with np.errstate(invalid='ignore'):
            acc.append(num / np.where(den == 0, np.nan, den))
    prof = np.nanmean(np.array(acc), axis=0)
    np.savez_compressed(SPD_CACHE, profile=prof, n_sessions=len(acc))
    print(f'  cached the speed profile over {len(acc)} sessions')
    return prof


def example_maps(mo, dy, ids):
    """Trial x position maps for named cells, built exactly as the cache is.

    The cached rows hold only the locked-anchored target cells, and the
    entorhinal examples are the opposite kind -- cells that DO switch -- so
    they are built here and kept in their own small npz.
    """
    key = f'M{mo}D{dy}'
    store = dict(np.load(EX_CACHE, allow_pickle=True)) if os.path.exists(
        EX_CACHE) else {}
    want = [int(c) for c in ids]
    if key in store:
        got = store[key].item()
        if all(c in got for c in want):
            return {c: got[c] for c in want}
    bp, cp = vr_paths(int(mo), int(dy))
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials_all = beh['trials'].as_dataframe()
    _, orig = clip_trials(trials_all, clusters)
    keep = np.isin(trials_all.number.values.astype(int), orig)
    have = [c for c in want if c in set(int(x) for x in clusters.index)]
    tn, trav = beh['trial_number'], beh['travel']
    dt = trav - (float(np.asarray(tn.values)[0]) - 1) * TL
    moving = beh['S'].threshold(3.0, method='above').time_support
    n_all = len(trials_all)
    tc = nap.compute_1d_tuning_curves(clusters[have], dt, nb_bins=n_all * NBIN,
                                      minmax=[0, n_all * TL], ep=moving)
    out = {}
    for c in have:
        M = np.asarray(tc[c]).reshape(n_all, NBIN)[keep]
        out[c] = np.array([smooth_nanaware(r, sigma=SIGMA)
                           for r in M]).astype(np.float32)
    store[key] = np.array(out, dtype=object)
    np.savez_compressed(EX_CACHE, **store)
    return out


def load_rows():
    if os.path.exists(CACHE):
        return list(np.load(CACHE, allow_pickle=True)['rows'])
    return build_cache()



def mec_state(mo, dy, prefix='ENTm'):
    """That session's anchoring raster, its PC1, and the major transitions.

    PREFIX picks the structure the raster is built from. The PC1 and the
    transitions are the session's, whichever structure is asked for, so the
    entorhinal and visual rasters are read against one and the same state --
    which is the comparison the panel exists to make.

    Read straight from the label file rather than cached with the rate maps:
    it costs nothing (the labels are an npz) and keeps the example panels on
    exactly the labels every other figure uses.
    """
    z = load_session_labels(int(mo), int(dy))
    reg = pd.read_csv(f'{PS}/pc1_by_region.csv')
    reg = reg[(reg.mouse == mo) & (reg.day == dy)]
    mec_ids = set(reg[reg.brain_region.astype(str).str.startswith(prefix)]
                  .cluster_id.astype(int))
    ids = [int(c) for c in z['cluster_id']]
    rows = [i for i, c in enumerate(ids) if c in mec_ids]
    L = np.asarray(z['labels'])[rows].astype(float)
    load = np.asarray(z['pc1_load'], float)[rows]
    # Cells that never vary carry no PC1 loading and would sort arbitrarily
    # among the ones that do, so they are separated to the right of a gap as in
    # Figure 1: the varying cells in PC1 order, then the locked ones.
    varies = np.nanstd(L, axis=1) > 0
    _idv = np.array([ids[i] for i in rows])[varies]
    _ord = np.argsort(load[varies])[::-1]
    ids_var = _idv[_ord]            # cluster ids, left to right in the raster
    Lv = L[varies][_ord]
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
    return L, pc1, tr, n_var, n_lock, gap, ids_var


if __name__ == '__main__':
    rows = load_rows()
    # Cells with no open-field speed test at all cannot be placed on either
    # side of the speed split, so they are dropped rather than defaulted into
    # the untuned group -- doing that quietly added 29 visual cells and pulled
    # the luminance result from p = 0.014 to p = 0.082.
    _sp = pd.read_csv(f'{PS}/unit_table.csv')[['mouse', 'day', 'cluster_id',
                                               'speed_p']]
    _spd = {(int(a), int(b), int(c)): d for a, b, c, d in
            _sp.itertuples(index=False)}
    for r in rows:
        r['speed_p'] = _spd.get((int(r['mouse']), int(r['day']),
                                 int(r['cluster_id'])), np.nan)
    rows = [r for r in rows if np.isfinite(r['speed_p'])]
    V = [r for r in rows if r['grp'] == 'VIS']
    E_ = [r for r in rows if r['grp'] == 'ENT']
    nb = V[0]['maps'].shape[1]
    x = (np.arange(nb) + .5) * (TL / nb)
    print(f'{len(V)} locked-anchored visual cells '
          f'({sum(r["speed_p"] < .05 for r in V)} speed-modulated), '
          f'{len(E_)} entorhinal ({sum(r["speed_p"] < .05 for r in E_)})')

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
    PE = np.array([profile(r) for r in E_])

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
    _rm = np.array([_corr(p_, _pup_i) for p_ in PE])

    # A and B each take a full row. Side by side there was no width for a
    # probe map and a second population raster, and the two examples are read
    # one after the other rather than against each other anyway.
    fig = plt.figure(figsize=(11.6, 10.4))
    # Five rows. F and G are pinned to the x-extent of the C and D heatmaps,
    # which leaves only about 2.4 in of their row free -- not enough for two
    # more panels -- so the census and the variance-explained panel take a row
    # of their own. Dropping the mean-rate strip from A and B paid for it:
    # the example rasters are no shorter than before.
    outer = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.0, 1.22, .92],
                             hspace=.62, left=.065, right=.975, top=.965,
                             bottom=.050)
    # E sits with C and D: it is the same peak-position information those two
    # heatmaps show, summarised, so it belongs beside them rather than opening
    # a row of its own
    gclu = outer[2].subgridspec(1, 2, wspace=.30)
    # only the row's vertical extent is taken from this: F and G are moved
    # under the C and D heatmaps after those exist, and H and I then fill
    # whatever is left to the right
    grow = outer[3].subgridspec(1, 3, width_ratios=[1.1, 1.0, .95],
                                wspace=.58)

    # ---- A, B: one session per row -----------------------------------------
    # Left to right: where the cells are, then the entorhinal population, its
    # PC1, the visual population, and three visual cells. Both rasters are
    # built the same way and both are cut by the same transition lines, so the
    # claim -- one population reorganises at those lines and the other does
    # not -- is made by the figure rather than asserted in the caption.
    for si, (mo, dy) in enumerate(EX_SESSIONS):
        cells = sorted([r for r in V if (r['mouse'], r['day']) == (mo, dy)],
                       key=consistency, reverse=True)[:3]
        if len(cells) < 3:
            print(f'  ! M{mo}D{dy} has only {len(cells)} cells'); continue
        L, pc1, tr, n_var, n_lock, gap, mec_ids = mec_state(mo, dy)
        LV, _, _, v_var, v_lock, v_gap, _ = mec_state(mo, dy, 'VIS')
        n_tr = cells[0]['maps'].shape[0]
        # The entorhinal examples are the three cells at the LEFT EDGE of the
        # raster beside them -- the largest PC1 loadings among the cells that
        # vary -- so the examples and the population panel are the same object
        # seen at two resolutions. They are the opposite selection to the
        # visual cells, which are picked for never switching.
        _mex = example_maps(mo, dy, mec_ids[:3])
        mcells = [dict(cluster_id=c, maps=_mex[c],
                       pop=np.asarray(load_session_labels(mo, dy)['frac_anch'],
                                      float) > .5)
                  for c in mec_ids[:3] if c in _mex]
        # Only the lower block DRAWS the colourbar -- both use the same scaling,
        # each map normalised to its own peak, so one bar serves them -- but
        # both RESERVE the column. Adding the column to one row only made that
        # row's maps narrower than the other's, so A and B stopped lining up.
        _last = si == len(EX_SESSIONS) - 1
        # the slice holds a square aspect, so a wide column just pads it with
        # whitespace; the rasters take the width instead. Each population sits
        # beside its own examples: raster, three cells, then the next
        # structure.
        # PC1 sits immediately right of the raster, against its locked block,
        # the way Figure 1E has it: the state and the cells it is computed
        # from read as one object instead of being separated by three maps.
        _wr = [.62, 1.20, .26, .70, .70, .70, .95, .70, .70, .70, .07]
        # Two rows sharing one trial axis. The mean-rate traces sit in the top
        # row above their own cells, and every raster -- both populations, PC1
        # and the three visual cells -- sits in the bottom row, so a trial is
        # at the same height in all of them.
        # One row. The mean-rate traces that used to sit above the maps are
        # gone: the maps show the same thing -- a field present in one state
        # and not the other, or present in both -- against the transition
        # lines, and the strip cost a quarter of the row's height to repeat
        # it. Peak rate moves into each map's title.
        gg = outer[si].subgridspec(1, len(_wr), width_ratios=_wr, wspace=.16)

        # the probe map, spanning both rows
        axs = fig.add_subplot(gg[0])
        draw_slice(axs, mo, dy)
        axs.set_title(f'M{mo} D{dy}', fontsize=8, pad=4, loc='left')
        # the slice keeps a square aspect, so its axes box floats inside the
        # gridspec cell and an axes-fraction offset puts the letter somewhere
        # different in each row; anchor it to the cell instead
        _cb = gg[0].get_position(fig)
        fig.text(_cb.x0 - .012, _cb.y1, 'AB'[si], fontsize=10, weight='bold',
                 va='bottom', ha='left')

        _tac = ListedColormap([NONANCH_COLOR, ANCH_COLOR])
        _tac.set_bad('white')
        _ax0 = None
        for gi, (_M, _nv, _nl, _gp, _rn) in enumerate(
                ((L, n_var, n_lock, gap, 'MEC'),
                 (LV, v_var, v_lock, v_gap, 'VIS'))):
            ax = fig.add_subplot(gg[1 if gi == 0 else 6],
                                 **({} if _ax0 is None else {'sharey': _ax0}))
            if _ax0 is None:
                _ax0 = ax
            ax.imshow(np.ma.masked_invalid(_M.T), aspect='auto',
                      interpolation='nearest', cmap=_tac,
                      norm=BoundaryNorm([-.5, .5, 1.5], 2),
                      extent=[0, _M.shape[0], _M.shape[1] - .5, -.5])
            if _nl:
                ax.annotate('varies (PC1 order)', (_nv / 2, 1.004),
                            xycoords=('data', 'axes fraction'), ha='center',
                            va='bottom', fontsize=5.6, color='0.35')
                ax.annotate(f'locked ({_nl})', (_nv + _gp + _nl / 2, 1.004),
                            xycoords=('data', 'axes fraction'), ha='center',
                            va='bottom', fontsize=5.6, color='0.35')
            for t_ in tr:
                ax.axhline(t_, color='k', lw=.8, ls='--', zorder=4)
            ax.set_xlabel(f'{_rn} cell', fontsize=7.5, color=RCOL[_rn])
            ax.tick_params(labelsize=6.5)
            if gi == 0:
                ax.set_ylabel('Trial', fontsize=8)
            else:
                ax.tick_params(labelleft=False)
            for sp in ax.spines.values():
                sp.set_visible(False)
        ax = _ax0

        # PC1, on the same trial axis, between the two populations
        axp = fig.add_subplot(gg[2], sharey=ax)
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
        for _gi, (_grp, _c0, _rn) in enumerate(((mcells, 3, 'MEC'),
                                                (cells, 7, 'VIS'))):
            for k, r in enumerate(_grp):
                # peak of the per-state means, which is what the removed
                # trace strip was read for
                st_ = median_filter(np.asarray(r['pop']).astype(float), size=9,
                                    mode='nearest') > .5
                _pk = 0.
                for m_ in (st_, ~st_):
                    if m_.sum() >= 5:
                        _pk = max(_pk, float(np.nanmax(
                            np.nanmean(r['maps'][m_], 0))))

                axc = fig.add_subplot(gg[_c0 + k], sharey=ax)
                # 99th percentile, not the max: one hot bin otherwise sets
                # the scale and flattens the rest of the map
                _mx = np.nanpercentile(r['maps'], 99) or 1.0
                _im = axc.imshow(np.clip(r['maps'] / _mx, 0, 1),
                                 aspect='auto', cmap=CMAP,
                                 vmin=0, vmax=1, interpolation='nearest',
                                 extent=[0, TL, r['maps'].shape[0] - .5, -.5])
                for t_ in tr:
                    if t_ < r['maps'].shape[0]:
                        axc.axhline(t_, color='w', lw=.8, ls='--', alpha=.85,
                                    zorder=4)
                # the maps are narrow now: only the middle one of each group
                # carries tick labels, which is enough to read the axis off
                axc.set_title(f'cl {r["cluster_id"]}  {_pk:.0f} Hz',
                              fontsize=6, pad=2, color=RCOL[_rn])
                axc.set_xticks([0, 100, 200])
                axc.tick_params(labelsize=6, labelleft=False)
                if k == 1:
                    axc.set_xlabel('Position (cm)', fontsize=7.5)
                else:
                    axc.tick_params(labelbottom=False)
                for sp in axc.spines.values():
                    sp.set_visible(False)
        if _last:
            axcb = fig.add_subplot(gg[10])
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
    # One panel per structure. The entorhinal cells that are speed-modulated in
    # the open field used to have a panel of their own; they are pooled here
    # because the split belongs in the test panel below, where it is measured,
    # rather than in the selection.
    _sets = ((PV, 'VIS', 'visual cortex'),
             (PE, 'ENT', 'entorhinal cortex'))
    for _k, (_P, _nm, _desc) in enumerate(_sets):
        # the inner gaps have to clear the heatmap's tick labels on both
        # sides: the cell numbers sit left of it over the dendrogram, and its
        # '200' sits right of it under the cluster means' '0'
        gg = gclu[_k].subgridspec(1, 3, width_ratios=[.28, 1.0, .58],
                                  wspace=.20)
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
        # each panel says exactly which cells it contains: all three are
        # locked anchored in switching sessions and differ only in region and
        # in whether they are speed-modulated
        axh.set_title(f'{_desc}\n{n_} cells, locked anchored',
                      fontsize=7.6, loc='left')

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

    # The cued/uncued example panel was removed; the population statistic it
    # carried is still computed here and quoted in the caption, since it is one
    # of the two tests of what these cells follow.
    def cue_split(rs):
        a, b = [], []
        for r in rs:
            for tag, acc in (('b', a), ('nb', b)):
                m = r['cue'] == tag
                if m.sum() >= 5:
                    pr = np.nanmean(r['maps'][m], axis=0)
                    acc.append((pr - np.nanmean(pr)) / (np.nanstd(pr) + 1e-12))
        return np.array(a), np.array(b)
    rzm = (x >= RZ[0]) & (x <= RZ[1])
    for _tag, _rows_ in (('all visual', V),
                         ('no speed tuning',
                          [r for r in V if r['speed_p'] >= .05])):
        CA, CB = cue_split(_rows_)
        n = min(len(CA), len(CB))
        w = wilcoxon(CA[:n][:, rzm].mean(1), CB[:n][:, rzm].mean(1))
        print(f'  cued vs uncued in the reward zone, {_tag}: '
              f'p = {w.pvalue:.3g} ({n} cells)')

    # the cumulative peak-position panel is gone, but its test is not: both
    # populations are far from uniform, which is why the caption can say the
    # positional concentration is not peculiar to visual cortex
    for _nm, _P in (('VIS', PV), ('ENT', PE)):
        _pk = x[np.argmax(_P, axis=1)]
        print(f'  {_nm} peak positions vs uniform: '
              f'p = {kstest(_pk / TL, "uniform").pvalue:.3g}')
    for _nm, _r in (('VIS', _rv), ('MEC', _rm)):
        _f = _r[np.isfinite(_r)]
        print(f'  {_nm} corr with pupil: mean {np.mean(_f):+.3f} '
              f'median {np.median(_f):+.3f} p = {wilcoxon(_f).pvalue:.3g}')

    # ---- E, F: the two confounds, tested properly, and where the rest is ----
    # THE CHANCE LEVEL IS THE WHOLE PROBLEM HERE, and two earlier versions of
    # this panel got it wrong. Both the speed profile and the luminance proxy
    # are smooth functions of position, and ANY smooth function of position
    # resembles a smooth firing profile: against zero, or against a rough
    # random regressor, both confounds look several times larger than they
    # are. The null here is a phase-randomised surrogate -- the same power
    # spectrum as the real regressor, so exactly as smooth, with its
    # relationship to position destroyed -- and the test is a permutation over
    # 500 such curves, because one curve is fitted to the whole population at
    # once and the cells are therefore not independent under the null. That
    # dependence is why the null has the spread it does.
    #
    # A variance-explained version of this panel was built first and dropped:
    # cross-validated R2 does not clear this null for ANY regressor in any
    # group (best p = 0.075), so the figure cannot say how MUCH either
    # confound accounts for. The correlation statistic below can still say
    # whether the relationship is there at all, because it asks whether the
    # sign is consistent across cells rather than how much variance one curve
    # soaks up.
    _SPD = speed_profile(sorted({(r['mouse'], r['day'])
                                 for r in V + E_}), len(x))
    N_SUR = 500

    def _z(v):
        v = np.asarray(v, float)
        return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)

    _rng = np.random.default_rng(0)

    def _surrogate(v):
        F_ = np.fft.rfft(np.asarray(v, float) - np.mean(v))
        ph = _rng.uniform(0, 2 * np.pi, len(F_)); ph[0] = 0.
        return _z(np.fft.irfft(np.abs(F_) * np.exp(1j * ph), n=len(v)))

    # Four groups, because the speed split is the question here rather than a
    # selection criterion. Pooling each structure across it was tried and is
    # wrong for visual cortex: the luminance relationship lives entirely in
    # the cells with no open-field speed tuning (+0.34, p = 0.014) and the
    # speed-modulated visual cells have none (+0.05, p = 0.43), so the pooled
    # population reads +0.11, p = 0.21. Entorhinal cortex pools without loss.
    _GL = [('VIS-', [r for r in V if r['speed_p'] >= .05], 'VIS\nno speed',
            '#3f9b4f'),
           ('VIS+', [r for r in V if r['speed_p'] < .05], 'VIS\nspeed-mod',
            '#8fcf96'),
           ('ENT-', [r for r in E_ if r['speed_p'] >= .05], 'ENT\nno speed',
            '#7b4173'),
           ('ENT+', [r for r in E_ if r['speed_p'] < .05], 'ENT\nspeed-mod',
            '#bb8bb0')]
    _PROF = {}
    for _gn, _rows_, _, _ in _GL:
        _PROF[_gn] = np.array([_z(np.nanmean(r['maps'], 0) -
                                  np.nanmean(np.nanmean(r['maps'], 0)))
                               for r in _rows_
                               if np.isfinite(np.nanmean(r['maps'], 0)).all()])

    def _medr(Pm, col):
        return float(np.median(Pm @ _z(col) / Pm.shape[1]))

    _REG = [('running speed', _SPD, '#4a7fb5'),
            ('luminance (pupil)', _pup_i, '#b8860b')]
    _ST = {}
    for _rn_, _reg, _ in _REG:
        for _gn, _, _, _ in _GL:
            obs = _medr(_PROF[_gn], _reg)
            nul = np.array([_medr(_PROF[_gn], _surrogate(_reg))
                            for _ in range(N_SUR)])
            pv = (np.sum(np.abs(nul) >= abs(obs)) + 1) / (N_SUR + 1)
            _ST[(_rn_, _gn)] = (obs, np.percentile(nul, [2.5, 50, 97.5]), pv)
            print(f'  {_rn_:18} {_gn:7} median r {obs:+.3f}, '
                  f'surrogate |r| 95th pct {np.percentile(np.abs(nul), 95):.3f}, '
                  f'p = {pv:.3f}')

    ax = fig.add_subplot(grow[0])
    _lp(ax, 'E', dx=-.24)
    # The null is drawn as what it is -- the spread of 500 random curves --
    # rather than as a band behind a bar, which reads as an error bar on the
    # bar and means the opposite of what it is.
    _gx = np.arange(len(_GL))
    for _j, (_rn_, _reg, _c) in enumerate(_REG):
        off = (_j - .5) * .40
        for i_, (_gn, _, _, _) in enumerate(_GL):
            obs, (lo, mid, hi), pv = _ST[(_rn_, _gn)]
            xx = i_ + off
            ax.add_patch(plt.Rectangle((xx - .13, lo), .26, hi - lo,
                                       facecolor='0.86', edgecolor='none',
                                       zorder=1))
            ax.plot([xx - .13, xx + .13], [mid, mid], color='0.55', lw=.8,
                    zorder=2)
            ax.plot([xx, xx], [mid, obs], color=_c, lw=1.0, zorder=3)
            ax.plot(xx, obs, 'o', ms=5.2, color=_c, zorder=4,
                    label=_rn_ if i_ == 0 else None)
            if pv < .05:
                ax.text(xx, obs + .035, '*', ha='center', va='bottom',
                        fontsize=8, color='0.15')
    ax.axhline(0, color='0.75', lw=.7, ls=':')
    ax.set_xticks(_gx); ax.set_xticklabels([g[2] for g in _GL], fontsize=6.0,
                                           linespacing=1.3)
    ax.set_xlim(-.6, len(_GL) - .4)
    ax.set_ylim(-.26, .46)
    ax.set_ylabel('how well the firing matches\nthe curve (median r)',
                  fontsize=7.5)
    ax.set_title('dot = these cells.  grey = 500 random\ncurves of the same smoothness',
                 fontsize=6.8, loc='left')
    ax.legend(fontsize=5.8, frameon=False, loc='lower left',
              handlelength=.8, handletextpad=.4)
    ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

    # ---- F: where these cells' position structure actually sits -------------
    # Plotted on the raw profiles, not the residuals: since neither regressor
    # clears its null, residualising changes this picture by a few points at
    # most (black-box share 52 -> 50% in VIS) and a residual would imply a
    # subtraction the panel above does not license.
    BB = 30.0                    # black box at each end of the 200 cm track
    ax = fig.add_subplot(grow[1])
    _lp(ax, 'F', dx=-.26)
    _bbs = {}
    for _gn, _lab, _c in (('VIS', 'visual cortex', '#3f9b4f'),
                         ('ENT', 'entorhinal cortex', '#7b4173')):
        v = (np.vstack([_PROF[k] for k in _PROF if k.startswith(_gn)])
             ** 2).mean(0)
        _bbs[_gn] = float(v[(x < BB) | (x > TL - BB)].sum() / v.sum())
        ax.plot(x, v / v.mean(), lw=1.5, color=_c,
                label=f'{_lab}  {_bbs[_gn]:.0%}')
    for _b in ((0, BB), (TL - BB, TL)):
        ax.axvspan(*_b, color='0.84', alpha=.8, lw=0, zorder=0)
    ax.axvspan(*RZ, color='#d8e4d0', alpha=.9, lw=0, zorder=0)
    for _xx, _t in ((15, 'black\nbox'), (185, 'black\nbox'), (100, 'reward')):
        ax.text(_xx, .035, _t, fontsize=5.2, color='0.3', ha='center',
                va='bottom', linespacing=1.1,
                transform=ax.get_xaxis_transform())
    print('  share of profile variance inside the black boxes (30% of track): '
          + ', '.join(f'{g} {v:.0%}' for g, v in _bbs.items()))
    ax.set_xlim(0, TL); ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xlabel('Position (cm)', fontsize=8)
    ax.set_ylabel('variance of the position\nprofile (1 = track average)',
                  fontsize=7.5)
    ax.set_title('the structure sits at the track ends,\nnot at the reward zone',
                 fontsize=6.8, loc='left')
    ax.legend(fontsize=5.6, frameon=False, loc='upper center', ncol=1,
              handlelength=1.2)
    ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

    # ---- G: the counts by structure -----------------------------------------
    ax = fig.add_subplot(grow[2])
    _lp(ax, 'G', dx=-.24)
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
    ax.legend(fontsize=5.8, frameon=False, loc='upper left')
    ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

    plt.savefig(OUT, dpi=200, bbox_inches='tight')
    print(f'wrote {OUT}')
