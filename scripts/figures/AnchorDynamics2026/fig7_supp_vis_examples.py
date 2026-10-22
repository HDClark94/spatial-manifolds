"""Figure S15: Figure 7's dissociation in four sessions, without the rate maps.

An alternative to Figure 7A-B: four sessions, two per row, each drawn the way
Figure 4 draws its examples.

RASTER. One population raster split into two blocks, MEC (ENTm) and visual
cortex, the way Figure 4 splits its raster by shank, each block under a grey
strip giving every cell's region class in Figure 4's grey key (MEC superficial
and deep apart). Within a block, cells that vary come first, in order of their
correlation with the session PC1, then the locked cells past a gap.

PC1. The session's anchoring axis computed SEPARATELY from each structure's own
cells (PC1 of that block's label matrix, signed to the session's anchored
fraction, z-scored), drawn as one line per structure on a shared axis, so
whether visual cortex follows the entorhinal switches is read off two lines.
Dashed lines are the session's transitions, as in Figure 7.

SLICE. The best-fit atlas slice, every cell coloured by agreement above chance
(|r| with the session PC1 minus the mean |r| of its circular-shift null, from
pc1_by_region.csv) on Figure 4's diverging PRGn scale, white at chance. Locked
cells have no agreement -- a constant label correlates with nothing -- and are
drawn small and grey, as Figure 4 draws cells without a value.

E-G. IS THE STATE SHARED? Every session with at least MIN_AXIS varying cells
in each structure. Each structure's state on each trial is its own PC1 (as in
the middle panels); its switches are that PC1's sign changes with epochs
shorter than three trials absorbed (major_transitions, as the dashed lines).
(E) Switch-triggered averages: each structure's fraction of varying cells in
the anchored mode around the OTHER structure's switches and its own, signed so
the new state is up, averaged per session then across sessions, against
circular time shifts of the non-triggering structure. (F) r(MEC PC1, VIS PC1)
against the ceiling set by each axis's own reliability (split-half, Spearman-
Brown corrected): points on the diagonal share as much as their noise allows.
(G) Census: every cell as following an axis (r above the 95th percentile of
its own circular-shift null), opposing it (below the 5th), neither, or locked.
Cached in data/population_state/s15_shared_*.csv.

Sessions: the two of Figure 7 (M21 D19, M27 D23) and the two others with at
least 25 cells in each structure and the clearest contrast between them, from
mice not already shown (M26 D18, M28 D19).
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

import anatomy_slice as ANA
import fig2_supp_vis_anchored as F
from spatial_manifolds.anchoring import (ANCH_COLOR, NONANCH_COLOR,
                                         load_session_labels, major_transitions)

OUT = f'{F.FIG}/fig7_supp_vis_examples.pdf'
SESSIONS = [(21, 19), (27, 23), (26, 18), (28, 19)]
# Figure 4's grey key for the region strip (fig3_anatomy.REGION_GREY): MEC
# superficial, MEC deep, visual cortex
GREY = {'MEC sup': .80, 'MEC deep': .69, 'VIS': .91}
# PC1 line colours: the visual green of Figure 7, and near-black for MEC --
# Figure 7's MEC purple would read as the anchored colour on a PC1 axis
LCOL = {'MEC': '0.15', 'VIS': F.RCOL['VIS']}
TAC = ListedColormap([NONANCH_COLOR, ANCH_COLOR]); TAC.set_bad('white')
REG = pd.read_csv(f'{F.PS}/pc1_by_region.csv')
MIN_AXIS = 10            # varying cells needed to build a structure's PC1
WIN = 15                 # trials either side of a switch (E)
N_TRIG_NULL = 200        # time shifts per session for E's null
SH_SESS = f'{F.PS}/s15_shared_sessions.csv'
SH_CELL = f'{F.PS}/s15_shared_cells.csv'
SH_TRIG = f'{F.PS}/s15_shared_triggered.npz'


def _cls(acr):
    a = str(acr)
    if a.startswith('ENTm'):
        return 'MEC sup' if a[4:5] in ('1', '2', '3') else 'MEC deep'
    return 'VIS' if a.startswith('VIS') else None


def _z(v):
    return (v - np.nanmean(v)) / np.nanstd(v)


def blocks(mo, dy):
    z = load_session_labels(mo, dy)
    ids = np.array([int(c) for c in z['cluster_id']])
    lab = np.asarray(z['labels'], float)
    frac = np.asarray(z['frac_anch'], float)
    rr = REG[(REG.mouse == mo) & (REG.day == dy)].set_index('cluster_id')
    out = {}
    for name in ('MEC', 'VIS'):
        rows = [i for i, c in enumerate(ids) if c in rr.index
                and (_cls(rr.brain_region[c]) or '').startswith(name)
                and not np.isnan(lab[i]).all()]
        L = lab[rows]; cid = ids[rows]
        cls = np.array([_cls(rr.brain_region[c]) for c in cid])
        r = np.array([rr.r.get(c, np.nan) for c in cid], float)
        varies = np.nanstd(L, axis=1) > 0
        ov = np.flatnonzero(varies)[np.argsort(-np.nan_to_num(r[varies], nan=-9))]
        ol = np.flatnonzero(~varies)[np.argsort(-np.nanmean(L[~varies], axis=1))]
        gap = max(2, int(round(.03 * len(L)))) if len(ol) else 0
        M = np.vstack([L[ov], np.full((gap, L.shape[1]), np.nan), L[ol]])
        strip = np.r_[[GREY[c] for c in cls[ov]], [1.] * gap, [GREY[c] for c in cls[ol]]]
        # this structure's own PC1, signed to the session's anchored fraction
        Mc = np.nan_to_num(L[varies] - np.nanmean(L[varies], axis=1, keepdims=True))
        pc = np.linalg.svd(Mc, full_matrices=False)[2][0]
        if np.corrcoef(pc, np.nan_to_num(frac))[0, 1] < 0:
            pc = -pc
        out[name] = dict(M=M, strip=strip, n_var=len(ov), n_lock=len(ol), gap=gap,
                         pc=_z(pc), n=len(L))
    return out, major_transitions(np.nan_to_num(np.asarray(z['pc1'], float))), rr


def _region_axis(L, frac):
    """PC1 of a structure's varying cells, signed to the anchored fraction."""
    v = np.nanstd(L, axis=1) > 0
    Mc = np.nan_to_num(L[v] - np.nanmean(L[v], axis=1, keepdims=True))
    pc = np.linalg.svd(Mc, full_matrices=False)[2][0]
    return -pc if np.corrcoef(pc, np.nan_to_num(frac))[0, 1] < 0 else pc


def _shift_r(x, y, guard=5):
    """r(x, y) and r for every circular shift of x by >= guard trials (FFT)."""
    x = (x - x.mean()) / (x.std() + 1e-12); y = (y - y.mean()) / (y.std() + 1e-12)
    n = len(x)
    c = np.real(np.fft.ifft(np.conj(np.fft.fft(x)) * np.fft.fft(y))) / n
    sh = np.r_[guard:n - guard + 1]
    return float(c[0]), c[sh]


def _switches(pc):
    """Switch trials and direction (+1 into anchored) of a PC1."""
    t = major_transitions(pc)
    d = [np.sign(np.mean(pc[k:k + 3])) or 1. for k in t]
    return np.array(t, int), np.array(d)


def _split_half(L, frac, rng, n=100):
    """Spearman-Brown reliability of a structure's PC1 from random half splits."""
    rs = []
    for _ in range(n):
        p = rng.permutation(len(L)); h = len(L) // 2
        rs.append(abs(np.corrcoef(_region_axis(L[p[:h]], frac),
                                  _region_axis(L[p[h:]], frac))[0, 1]))
    r = float(np.mean(rs))
    return 2 * r / (1 + r)


def shared_state():
    """Per session (F, and the traces behind E) and per cell (G)."""
    if all(os.path.exists(f) for f in (SH_SESS, SH_CELL, SH_TRIG)):
        return pd.read_csv(SH_SESS), pd.read_csv(SH_CELL), dict(np.load(SH_TRIG))
    srows, crows, trig = [], [], {}
    for (mo, dy), g in REG.groupby(['mouse', 'day']):
        z = load_session_labels(int(mo), int(dy))
        if z is None:
            continue
        ids = np.array([int(c) for c in z['cluster_id']])
        lab = np.asarray(z['labels'], float)
        frac = np.asarray(z['frac_anch'], float)
        g = g.set_index('cluster_id')
        struct = np.array([('MEC' if str(g.brain_region.get(c, '')).startswith('ENTm') else
                            'VIS' if str(g.brain_region.get(c, '')).startswith('VIS') else '')
                           for c in ids])
        ok = ~np.isnan(lab).all(axis=1)
        var = ok & (np.nanstd(lab, axis=1) > 0)
        if min((var & (struct == k)).sum() for k in ('MEC', 'VIS')) < MIN_AXIS:
            continue
        rng = np.random.default_rng(int(mo) * 100 + int(dy))
        ax_ = {k: _region_axis(lab[ok & (struct == k)], frac) for k in ('MEC', 'VIS')}
        # fraction of each structure's varying cells in the anchored mode
        fr = {k: np.nanmean(lab[var & (struct == k)], axis=0) for k in ('MEC', 'VIS')}
        n_tr = lab.shape[1]
        # ---- E: switch-triggered averages, per session, both triggers
        for trg in ('MEC', 'VIS'):
            t, d = _switches(ax_[trg])
            t_ok = (t >= WIN) & (t <= n_tr - WIN)
            t, d = t[t_ok], d[t_ok]
            if not len(t):
                continue
            lags = np.arange(-WIN, WIN)
            for k in ('MEC', 'VIS'):
                X = np.array([dd * (fr[k][tt + lags] - fr[k][tt - 5:tt].mean())
                              for tt, dd in zip(t, d)])
                trig.setdefault(f'{trg}>{k}', []).append(X.mean(0))
                # null: the same switches, the traced structure shifted in time
                N = []
                for s_ in rng.integers(WIN, n_tr - WIN, N_TRIG_NULL):
                    f_ = np.roll(fr[k], s_)
                    N.append(np.mean([dd * (f_[tt + lags] - f_[tt - 5:tt].mean())
                                      for tt, dd in zip(t, d)], axis=0))
                trig.setdefault(f'{trg}>{k}_null', []).append(np.array(N))
            trig.setdefault(f'{trg}_n', []).append(len(t))
        # ---- F: r against the reliability ceiling
        r_mv, null = _shift_r(ax_['MEC'], ax_['VIS'], guard=10)
        relM = _split_half(lab[ok & (struct == 'MEC')], frac, rng)
        relV = _split_half(lab[ok & (struct == 'VIS')], frac, rng)
        srows.append(dict(mouse=int(mo), day=int(dy), n_trials=n_tr,
                          n_mec=int((var & (struct == 'MEC')).sum()),
                          n_vis=int((var & (struct == 'VIS')).sum()),
                          r_mv=r_mv, null95=float(np.percentile(np.abs(null), 95)),
                          p_mv=float((np.sum(np.abs(null) >= abs(r_mv)) + 1) / (len(null) + 1)),
                          rel_mec=relM, rel_vis=relV,
                          ceiling=float(np.sqrt(max(relM, 0) * max(relV, 0)))))
        # ---- G: every cell against each structure's axis (own: leave-one-out)
        for i in np.flatnonzero(ok & np.isin(struct, ['MEC', 'VIS'])):
            own = struct[i]
            row = dict(mouse=int(mo), day=int(dy), cluster_id=int(ids[i]), structure=own,
                       locked=bool(not var[i]))
            if var[i]:
                for k in ('MEC', 'VIS'):
                    if k == own:
                        m_ = ok & (struct == k); m_[i] = False
                        pc = _region_axis(lab[m_], frac)
                    else:
                        pc = ax_[k]
                    r, nl = _shift_r(np.nan_to_num(lab[i]), pc)
                    row[f'r_{k}'] = r
                    row[f'cat_{k}'] = ('follows' if r > np.percentile(nl, 95) else
                                       'opposes' if r < np.percentile(nl, 5) else 'neither')
            crows.append(row)
        print(f'  M{mo}D{dy}: r {r_mv:+.2f}, ceiling {srows[-1]["ceiling"]:.2f}', flush=True)
    S, C = pd.DataFrame(srows), pd.DataFrame(crows)
    T = {}
    for k, v in trig.items():
        T[k] = np.array(v) if not k.endswith('_null') else np.stack(v)   # sessions x ...
    for X, f in ((S, SH_SESS), (C, SH_CELL)):
        X.to_csv(f + '.partial', index=False); os.replace(f + '.partial', f)
    np.savez(SH_TRIG + '.partial.npz', **T); os.replace(SH_TRIG + '.partial.npz', SH_TRIG)
    return S, C, T


def draw_slice(ax, mo, dy, rr, vlim):
    cells = F.CELLXYZ[(F.CELLXYZ.mouse == mo) & (F.CELLXYZ.day == dy)
                      & F.CELLXYZ.macro.isin(['MEC', 'VIS'])]
    cpts = ANA.to_ccf(cells, ('coord_SCs_z', 'coord_SCs_y', 'coord_SCs_x'))
    # the plane is fitted to the mouse's contacts and framed on them and the
    # cells together, at the atlas's own aspect; every contact is drawn small
    # and pale, as in Figure S7, so all four shanks show even where no cell sat
    con = ANA.contacts_ccf(mo)
    fit = ANA.fit_best_slice(con, extent_pts=np.vstack([con, cpts]))
    u, v, grid = ANA.sample_region_grid(fit, step=F.SLICE_STEP)
    ax.imshow(grid, origin='lower', aspect='equal', zorder=1,
              cmap=ANA.region_cmap_neutral, norm=ANA.region_norm_neutral,
              extent=[u[0], u[-1], v[0], v[-1]])
    cu, cv = fit['project'](con)
    ax.scatter(cu, cv, s=.6, color='0.55', alpha=.5, lw=0, zorder=2)
    pu, pv = fit['project'](cpts)
    pu = pu + np.random.default_rng(mo * 1000 + dy).normal(0, F.ML_JITTER, len(pu))
    ex = np.array([rr.excess.get(c, np.nan) for c in cells.cluster_id], float)
    ok = np.isfinite(ex)
    ax.scatter(pu[~ok], pv[~ok], s=2.0, color='0.70', alpha=.6, lw=0, zorder=2)
    sc = ax.scatter(pu[ok], pv[ok], c=ex[ok], s=6.0, cmap='PRGn',
                    norm=TwoSlopeNorm(vcenter=0., vmin=-vlim, vmax=vlim),
                    alpha=.95, lw=0, edgecolors='none', zorder=3)
    ax.set_xlim(u[0], u[-1]); ax.set_ylim(v[-1], v[0])
    ax.set_aspect('equal', adjustable='box'); ax.set_anchor('C')
    ax.set_xlabel('medial–lateral (µm)', fontsize=6.5, labelpad=1)
    ax.set_ylabel('dorsal–ventral (µm)', fontsize=6.5, labelpad=1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(2))
    # the slice's v is centred on the contacts' centroid; label depth from
    # the most dorsal contact instead, so the axis reads 0 at the top
    _v0 = float(cv.min())
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(
        _v0 + np.arange(0, v[-1] - _v0, 1000)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda y, _: f'{y - _v0:.0f}'))
    ax.tick_params(labelsize=5.5, length=2, pad=1)
    ax.spines[['top', 'right']].set_visible(False)
    return sc


def example(fig, gs, mo, dy, letter, vlim):
    B, tr, rr = blocks(mo, dy)
    n_tr = B['MEC']['M'].shape[1]
    gg = gs.subgridspec(1, 3, width_ratios=[1.45, .28, 1.05], wspace=.16)
    HR = [.05, 1]
    gR = gg[0].subgridspec(2, 2, height_ratios=HR, hspace=.025, wspace=.06,
                           width_ratios=[B['MEC']['M'].shape[0], B['VIS']['M'].shape[0]])
    ax0 = None
    for j, name in enumerate(('MEC', 'VIS')):
        b = B[name]
        axs = fig.add_subplot(gR[0, j])
        axs.imshow(np.repeat(b['strip'][None, :, None], 3, axis=2), aspect='auto',
                   interpolation='nearest', vmin=0, vmax=1)
        axs.set_xticks([]); axs.set_yticks([])
        for sp in axs.spines.values():
            sp.set_visible(False)
        axs.set_title(f'{name} {b["n"]}, locked {b["n_lock"]} '
                      f'({100 * b["n_lock"] / b["n"]:.0f}%)', fontsize=6.3, pad=2,
                      color='0.2')
        ax = fig.add_subplot(gR[1, j], **({} if ax0 is None else {'sharey': ax0}))
        ax0 = ax0 or ax
        ax.imshow(np.ma.masked_invalid(b['M'].T), aspect='auto', interpolation='nearest',
                  cmap=TAC, norm=BoundaryNorm([-.5, .5, 1.5], 2),
                  extent=[0, b['M'].shape[0], n_tr, 1])
        for t_ in tr:
            ax.axhline(t_ + 1, color='0.1', lw=.5, ls='--', alpha=.7, zorder=4)
        ax.set_xticks([])
        ax.set_xlabel(f'{name} cells', fontsize=7, labelpad=2)
        if j == 0:
            ax.set_ylabel('Trial', fontsize=8, labelpad=1)
            ax.tick_params(labelsize=6.5)
            axs.set_ylabel('region', fontsize=5.8, rotation=0, ha='right', va='center',
                           labelpad=2)
            axs.annotate(f'M{mo} D{dy}', (0, 1), xycoords='axes fraction',
                         xytext=(0, 12), textcoords='offset points', fontsize=7.6,
                         ha='left', va='bottom')
            _p = gR[0, 0].get_position(fig)
            fig.text(_p.x0 - .04, _p.y1 + .02, letter, fontsize=10, weight='bold',
                     va='bottom', ha='right')
        else:
            ax.tick_params(labelleft=False, left=False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    gP = gg[1].subgridspec(2, 1, height_ratios=HR, hspace=.025)
    axp = fig.add_subplot(gP[1], sharey=ax0)
    y = np.arange(1, n_tr + 1)
    for name in ('VIS', 'MEC'):
        axp.plot(B[name]['pc'], y, color=LCOL[name], lw=.8, zorder=3)
    axp.axvline(0, color='0.6', lw=.6)
    for t_ in tr:
        axp.axhline(t_ + 1, color='0.1', lw=.5, ls='--', alpha=.7, zorder=4)
    axp.set_xticks([]); axp.tick_params(labelleft=False, left=False)
    axp.set_xlabel('PC1\n(z)', fontsize=7, labelpad=2)
    for sp in axp.spines.values():
        sp.set_visible(False)
    rmv = np.corrcoef(B['MEC']['pc'], B['VIS']['pc'])[0, 1]
    axp.set_title(f'r = {rmv:+.2f}', fontsize=6.3, pad=2, color='0.2')

    axsl = fig.add_subplot(gg[2])
    sc = draw_slice(axsl, mo, dy, rr, vlim)
    print(f'M{mo} D{dy}: MEC {B["MEC"]["n_lock"]}/{B["MEC"]["n"]} locked, '
          f'VIS {B["VIS"]["n_lock"]}/{B["VIS"]["n"]} locked, {len(tr)} transitions, '
          f'{n_tr} trials, r(MEC PC1, VIS PC1) = {rmv:+.3f}')
    return sc


if __name__ == '__main__':
    ex = REG[REG.set_index(['mouse', 'day']).index.isin(SESSIONS)].excess.dropna()
    vlim = float(max(abs(np.percentile(ex, 2)), abs(np.percentile(ex, 98))))
    fig = plt.figure(figsize=(10.4, 10.0))
    outer = fig.add_gridspec(3, 1, height_ratios=[1, 1, .62], hspace=.42,
                             left=.06, right=.985, top=.955, bottom=.105)
    for i, (mo, dy) in enumerate(SESSIONS):
        cell = outer[i // 2].subgridspec(1, 2, wspace=.12)[i % 2]
        sc = example(fig, cell, mo, dy, 'ABCD'[i], vlim)
    # ---- E-G: is the state shared? ---------------------------------------
    SS, CC, TT = shared_state()
    gB = outer[2].subgridspec(1, 4, width_ratios=[1, 1, 1.05, 1.25], wspace=.42)
    lags = np.arange(-WIN, WIN)
    axE0 = None
    for j, trg in enumerate(('MEC', 'VIS')):
        ax = fig.add_subplot(gB[j], **({} if axE0 is None else {'sharey': axE0}))
        axE0 = axE0 or ax
        oth = 'VIS' if trg == 'MEC' else 'MEC'
        Nm = TT[f'{trg}>{oth}_null'].mean(0)            # shifts x lags, session mean
        lo, hi = np.percentile(Nm, [2.5, 97.5], axis=0)
        ax.fill_between(lags, lo, hi, facecolor='none', hatch='.....', edgecolor='0.75', lw=0,
                        zorder=1)
        for k in (trg, oth):
            X = TT[f'{trg}>{k}']
            m, se = X.mean(0), X.std(0) / np.sqrt(len(X))
            ax.fill_between(lags, m - se, m + se, color=LCOL[k], alpha=.25, lw=0,
                            edgecolor='none', zorder=2)
            ax.plot(lags, m, color=LCOL[k], lw=1.3, zorder=3)
        ax.axvline(-.5, color='0.3', lw=.6, ls='--'); ax.axhline(0, color='0.5', lw=.6, ls=':')
        # the other structure's step (trials 0-4 against the pre-switch
        # baseline the traces are already referred to), as a share of the
        # triggering structure's own, and against the shifted null
        post = slice(WIN, WIN + 5)
        st_o = TT[f'{trg}>{oth}'][:, post].mean(1); st_t = TT[f'{trg}>{trg}'][:, post].mean(1)
        nul = Nm[:, post].mean(1)
        p = (np.sum(nul >= st_o.mean()) + 1) / (len(nul) + 1)
        _ps = (f'p < {1 / (len(nul) + 1):.1g}' if p <= 1 / (len(nul) + 1) + 1e-12
               else f'p = {p:.2g}')
        ax.set_title(f'at {trg} switches\n{oth} steps {100 * st_o.mean() / st_t.mean():.0f}% '
                     f'as far, {_ps}', fontsize=7, pad=3, linespacing=1.15)
        print(f'E {trg} switches: {int(TT[f"{trg}_n"].sum())} switches in {len(st_o)} sessions; '
              f'{trg} step {st_t.mean():.3f}, {oth} step {st_o.mean():.3f} '
              f'({100 * st_o.mean() / st_t.mean():.0f}%), null 97.5% {np.percentile(nul, 97.5):.3f}, p {p:.3g}')
        ax.set_xlabel(f'trials from {trg} switch', fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.spines[['top', 'right']].set_visible(False)
        if j == 0:
            ax.set_ylabel('change in fraction of cells\nin the new state', fontsize=7.5)
            ax.fill_between([], [], [], facecolor='none', hatch='.....', edgecolor='0.75',
                            lw=0, label='time-shift null')
            ax.legend(fontsize=6, frameon=False, loc='upper left', handlelength=1.2)
            ax.text(-.36, 1.12, 'E', transform=ax.transAxes, fontsize=10, weight='bold')
        else:
            ax.tick_params(labelleft=False)

    # F: shared as far as each axis's own reliability allows
    ax = fig.add_subplot(gB[2])
    sig = SS.p_mv < .05
    ax.plot([0, 1], [0, 1], color='0.6', lw=.7, ls='--', zorder=1)
    ax.axhline(0, color='0.5', lw=.6, ls=':')
    ax.scatter(SS.ceiling[sig], SS.r_mv[sig], s=16, color='0.15', lw=0, zorder=3)
    ax.scatter(SS.ceiling[~sig], SS.r_mv[~sig], s=16, facecolor='w', edgecolor='0.15',
               lw=.8, zorder=3)
    for L_, (mo, dy) in zip('ABCD', SESSIONS):
        r_ = SS[(SS.mouse == mo) & (SS.day == dy)]
        if len(r_):
            ax.annotate(L_, (r_.ceiling.iloc[0], r_.r_mv.iloc[0]), xytext=(3, -7),
                        textcoords='offset points', fontsize=6, weight='bold', color='0.3')
    hc = SS.ceiling >= .5
    ratio = (SS.r_mv / SS.ceiling)[hc]
    ax.set_xlim(0, 1); ax.set_ylim(-.6, 1)
    ax.set_xlabel('reliability ceiling', fontsize=7.5)
    ax.set_ylabel('r (MEC PC1, VIS PC1)', fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title(f'{int(sig.sum())} of {len(SS)} beyond shift null\nceiling >= 0.5: r / ceiling '
                 f'median {ratio.median():.2f} (n = {hc.sum()})', fontsize=7, pad=3, linespacing=1.15)
    ax.text(-.3, 1.12, 'F', transform=ax.transAxes, fontsize=10, weight='bold')
    print(f'F: sig {int(sig.sum())}/{len(SS)}; ceiling>=.5: n {hc.sum()}, r/ceiling median '
          f'{ratio.median():.3f}, below .5: {int((ratio < .5).sum())} '
          f'{[f"M{a}D{b}" for a, b in zip(SS.mouse[hc][ratio < .5], SS.day[hc][ratio < .5])]}; '
          f'ceiling<.5: n {(~hc).sum()}, sig {int(sig[~hc].sum())}')

    # G: census of cells against each axis
    ax = fig.add_subplot(gB[3])
    bars = [('VIS', 'MEC'), ('VIS', 'VIS'), ('MEC', 'MEC'), ('MEC', 'VIS')]
    cats = [('follows', '0.20'), ('opposes', '0.55'), ('neither', '0.86'), ('locked', 'w')]
    for y_, (st, k) in enumerate(bars):
        c = CC[CC.structure == st]
        v = c[f'cat_{k}'].where(~c.locked, 'locked').value_counts(normalize=True)
        x0 = 0.
        for cat, col in cats:
            w = float(v.get(cat, 0))
            ax.barh(y_, w, left=x0, color=col, edgecolor='0.4', lw=.4, height=.7,
                    hatch='////' if cat == 'locked' else None)
            if cat == 'follows':
                ax.text(x0 + w / 2, y_, f'{100 * w:.0f}%', ha='center', va='center',
                        fontsize=6, color='w')
            x0 += w
        print(f'G {st} cells vs {k} axis (n={len(c)}): ' +
              ', '.join(f'{cat} {100 * float(v.get(cat, 0)):.1f}%' for cat, _ in cats))
    ax.set_yticks(range(4))
    ax.set_yticklabels([f'{st} cells,\n{"own" if st == k else k} axis' for st, k in bars],
                       fontsize=6.3)
    ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel('fraction of cells', fontsize=7.5)
    ax.tick_params(axis='x', labelsize=6.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(handles=[matplotlib.patches.Patch(facecolor=col, edgecolor='0.4', lw=.4,
                                                hatch='////' if cat == 'locked' else None,
                                                label=cat) for cat, col in cats],
              fontsize=6, frameon=False, ncol=4, loc='lower center',
              bbox_to_anchor=(.4, 1.0), handlelength=1.2, columnspacing=.8)
    ax.text(-.42, 1.12, 'G', transform=ax.transAxes, fontsize=10, weight='bold')

    # the shared key along the bottom: the agreement scale (common to the four
    # slices), the PC1 lines and the region strip greys
    cax = fig.add_axes([.07, .045, .15, .008])
    cb = fig.colorbar(sc, cax=cax, orientation='horizontal')
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6)
    cb.set_label('agreement above chance (A-D)', fontsize=7, labelpad=2)
    h = ([Line2D([], [], color=LCOL[k], lw=1.2, label=k) for k in ('MEC', 'VIS')]
         + [matplotlib.patches.Patch(facecolor=str(g), edgecolor='0.6', lw=.4, label=k)
            for k, g in (('MEC superficial', GREY['MEC sup']), ('MEC deep', GREY['MEC deep']),
                         ('visual cortex', GREY['VIS']))]
         + [Line2D([], [], marker='o', ls='', color='0.70', ms=3, label='locked (no agreement)')])
    fig.legend(handles=h, loc='lower left', bbox_to_anchor=(.27, .028), ncol=6,
               fontsize=6.6, frameon=False, handlelength=1.4, columnspacing=1.2)
    tmp = OUT + '.partial'
    fig.savefig(tmp, format='pdf', dpi=300); os.replace(tmp, OUT)
    print('wrote', OUT)
