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

    PV = np.array([profile(r) for r in V])
    PM = np.array([profile(r) for r in M_])

    fig = plt.figure(figsize=(10, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=.60,
                          wspace=.42, left=.07, right=.97, top=.90, bottom=.09)

    # ---- A: example cells -------------------------------------------------
    ga = gs[0, 0].subgridspec(1, 3, wspace=.18)
    # pick by how consistently each trial matches the cell's own mean, not by
    # peak height x trial count, which just favoured the longest sessions
    def consistency(r):
        Mm = r['maps']; t = np.nanmean(Mm, axis=0)
        t = t - np.nanmean(t)
        if np.nanstd(t) == 0:
            return -1
        v = []
        for row in Mm:
            a = row - np.nanmean(row)
            d = np.sqrt(np.nansum(a ** 2) * np.nansum(t ** 2))
            if d > 0:
                v.append(np.nansum(a * t) / d)
        return float(np.mean(v)) if v else -1
    _ranked = sorted(V, key=consistency, reverse=True)
    best, _seen = [], set()
    for r in _ranked:                      # one per session, best first
        if (r['mouse'], r['day']) in _seen:
            continue
        best.append(r); _seen.add((r['mouse'], r['day']))
        if len(best) == 3:
            break
    for k, r in enumerate(best):
        ax = fig.add_subplot(ga[k])
        Mm = r['maps']
        ax.imshow(Mm, aspect='auto', cmap='magma', interpolation='nearest',
                  extent=[0, TL, len(Mm) - .5, -.5])
        # median-filtered state, as everywhere else: the raw per-trial label
        # flickers (25 flips in this session against 9 once filtered) and drew
        # a line at every flicker
        from scipy.ndimage import median_filter
        _st = median_filter(r['pop'].astype(float), size=9, mode='nearest') > .5
        for t_ in np.where(np.diff(_st.astype(int)) != 0)[0]:
            ax.axhline(t_, color='w', lw=.9, ls='--')
        ax.set_xticks([0, 100, 200]); ax.tick_params(labelsize=6)
        if k == 0:
            ax.set_ylabel('Trial', fontsize=8)
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(f'M{r["mouse"]}D{r["day"]} cl {r["cluster_id"]}',
                     fontsize=6, loc='left', pad=2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if k == 1:
            ax.set_xlabel('Position (cm)', fontsize=8)
    fig.text(.055, .935, 'A', fontsize=10, weight='bold')
    fig.text(.088, .935, 'example VIS cells — dashed lines are MEC state transitions',
             fontsize=7.5, color='0.3')

    # ---- B: all cells, sorted by peak -------------------------------------
    ax = fig.add_subplot(gs[0, 1]); _lp(ax, 'B', dx=-.19)
    o = np.argsort(np.argmax(PV, axis=1))
    ax.imshow(PV[o], aspect='auto', cmap='magma', interpolation='nearest',
              extent=[0, TL, len(PV) - .5, -.5])
    ax.axvspan(*RZ, color='w', alpha=.0)
    for b in RZ:
        ax.axvline(b, color='#7fd4a8', lw=1.0, ls='--')
    ax.set_xlabel('Position (cm)', fontsize=8)
    ax.set_ylabel('VIS cell (sorted)', fontsize=8)
    ax.set_title(f'all {len(PV)} cells, normalised', fontsize=7.5, loc='left')
    ax.tick_params(labelsize=7)

    # ---- C: peak positions ------------------------------------------------
    ax = fig.add_subplot(gs[0, 2]); _lp(ax, 'C', dx=-.22)
    pk = x[np.argmax(PV, axis=1)]
    pkm = x[np.argmax(PM, axis=1)]
    bins = np.linspace(0, TL, 21)
    ax.hist(pk, bins=bins, color='#8C6BB1', alpha=.85, lw=0, density=True,
            label=f'VIS ({len(pk)})')
    ax.hist(pkm, bins=bins, histtype='step', color='0.35', lw=1.2,
            density=True, label=f'MEC ({len(pkm)})')
    ax.axhline(1 / TL, color='0.5', lw=.9, ls=':')
    ax.axvspan(*RZ, color='#d8e4d0', alpha=.55, lw=0, zorder=0)
    ks = kstest((pk - 0) / TL, 'uniform')
    ksm = kstest((pkm - 0) / TL, 'uniform')
    ax.set_xlabel('peak position (cm)', fontsize=8)
    ax.set_ylabel('density', fontsize=8)
    ax.set_title(f'VIS vs uniform p = {ks.pvalue:.1g}\n'
                 f'MEC vs uniform p = {ksm.pvalue:.1g}', fontsize=7, loc='left')
    ax.legend(fontsize=6, frameon=False)
    ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

    # ---- D: cued vs uncued ------------------------------------------------
    ax = fig.add_subplot(gs[1, 0]); _lp(ax, 'D', dx=-.19)
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
    for Y, c_, lab in ((CA, '#c04744', 'cued (beacon)'),
                       (CB, '#2b6cb0', 'uncued')):
        mu = Y.mean(0); se = Y.std(0) / np.sqrt(len(Y))
        ax.fill_between(x, mu - se, mu + se, color=c_, alpha=.28, lw=0)
        ax.plot(x, mu, color=c_, lw=1.4, label=lab)
    ax.axvspan(*RZ, color='#d8e4d0', alpha=.55, lw=0, zorder=0)
    rzm = (x >= RZ[0]) & (x <= RZ[1])
    w = wilcoxon(CA[:n][:, rzm].mean(1), CB[:n][:, rzm].mean(1))
    ax.set_xlabel('Position (cm)', fontsize=8)
    ax.set_ylabel('firing (z, within cell)', fontsize=8)
    ax.set_title(f'does the beacon change them?\nin the reward zone: '
                 f'p = {w.pvalue:.2g} ({n} cells)', fontsize=7.5, loc='left')
    ax.legend(fontsize=6, frameon=False)
    ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

    # ---- E: MEC contrast ---------------------------------------------------
    ax = fig.add_subplot(gs[1, 1]); _lp(ax, 'E', dx=-.19)
    om = np.argsort(np.argmax(PM, axis=1))
    ax.imshow(PM[om], aspect='auto', cmap='magma', interpolation='nearest',
              extent=[0, TL, len(PM) - .5, -.5])
    for b in RZ:
        ax.axvline(b, color='#7fd4a8', lw=1.0, ls='--')
    ax.set_xlabel('Position (cm)', fontsize=8)
    ax.set_ylabel('MEC cell (sorted)', fontsize=8)
    ax.set_title(f'MEC non-speed locked-anchored\n({len(PM)} cells), for contrast',
                 fontsize=7.5, loc='left')
    ax.tick_params(labelsize=7)

    # ---- F: what the counts were ------------------------------------------
    ax = fig.add_subplot(gs[1, 2]); _lp(ax, 'F', dx=-.22)
    vis, mec = target_cells()
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

    fig.suptitle('Visual-cortex cells that stay track-anchored while MEC switches',
                 fontsize=9, y=.975)
    plt.savefig(OUT, dpi=200, bbox_inches='tight')
    print(f'wrote {OUT}')
