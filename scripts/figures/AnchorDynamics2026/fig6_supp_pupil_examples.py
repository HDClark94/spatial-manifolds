'''Figure S17: the pupil effect of Figure 6 in five more sessions, one per mouse.

Each row is one session, drawn as Figure 6 C-G (the functions of
fig5_pupil_arousal): the eye at seven moments of one anchored and one
non-anchored traversal (the most representative trial of each state's longest
block, with the DeepLabCut perimeter landmarks and the radius beneath each
frame); mean pupil radius against position for the two blocks, with those
trials overlaid; the MEC anchoring raster and its PC1; the radius on every
trial on the same trial axis; and that map z-scored over trials at each
position, which removes the position profile every trial shares.

Sessions: ranked by the difference in mean pupil radius between the two
longest blocks (anchored minus non-anchored, over that block's trials,
relative to its standard error), which is what the position panel shows: the
best session of each mouse with a clear difference (M26, M28, M27, M21; in M20,
M25 and M22 the blocks differ little or not at all), then the next best overall
(M28 D25). M21 D22 is a different day of Figure 6's mouse.

Writes fig6_supp_pupil_examples.pdf
'''
import os
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig5_pupil_arousal as F6                                       # noqa: E402

SESSIONS = [(26, 18), (28, 19), (27, 19), (21, 22), (28, 25)]
OUT = f'{F6.FIG}/fig6_supp_pupil_examples.pdf'
NB_MEAN, NB_MAP = 25, 40
ZLIM = 2


def row(fig, gs, mo, dy, letter):
    S = F6.load_session(mo, dy)
    P = F6.population(mo, dy)
    blocks = F6.state_blocks(mo, dy, S, P)
    sel = {}
    for k in ('anch', 'non'):
        a, b, _, n = blocks[k]
        u, prof = F6.pick_trial(S, a, b)
        sel[k] = dict(block=(a, b, n), trial=u, frames=F6.trial_frames(S, u))
    want = [f for k in sel for f in sel[k]['frames'] if f is not None]
    store, ox, oy = F6.get_frames(mo, dy, want)
    ntr = len(P['trial'])
    tick_i = [i for i, t in enumerate(P['trial']) if t % 40 == 0]

    g = gs.subgridspec(1, 6, width_ratios=[3.3, 1.45, 1.25, .32, 1.35, 1.35], wspace=.30)
    # eye frames, one row per state
    gl = g[0].subgridspec(2, len(F6.SHOW_POS), hspace=.30, wspace=.05)
    MCX = np.median(S['cx'][S['good']]) + ox
    MCY = np.median(S['cy'][S['good']]) + oy
    for r, (k, col, lab) in enumerate((('anch', F6.ANCH_COLOR, 'anchored'),
                                       ('non', F6.NONANCH_COLOR, 'non-anch.'))):
        for c, fi in enumerate(sel[k]['frames']):
            ax = fig.add_subplot(gl[r, c], facecolor='k')
            ax.imshow(cv2.cvtColor(store[fi], cv2.COLOR_BGR2RGB))
            for jj, cc in enumerate(F6.LM_COLORS):
                ax.scatter(S['X'][jj, fi] + ox, S['Y'][jj, fi] + oy, s=2.5, color=cc,
                           edgecolor='k', linewidth=.15, zorder=4)
            ax.set_xlim(MCX - F6.ZOOM, MCX + F6.ZOOM)
            ax.set_ylim(MCY + F6.ZOOM * .78, MCY - F6.ZOOM * .78)
            ax.text(.5, -.05, f'{S["rad"][fi]:.1f}', transform=ax.transAxes, fontsize=5.5,
                    color='0.25', ha='center', va='top')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(col); sp.set_linewidth(.9)
            if r == 0:
                ax.set_title(f'{F6.SHOW_POS[c]} cm', fontsize=6, color='0.25', pad=2)
            if c == 0:
                ax.set_ylabel(f'{lab}\ntrial {sel[k]["trial"]}', fontsize=6.3, color=col,
                              labelpad=3, linespacing=1.05)
    # mean radius against position, per block, with the displayed trials
    ax = fig.add_subplot(g[1])
    x = np.arange(NB_MEAN) * (200.0 / NB_MEAN) + 4
    pb = np.clip((S['pos'] / (200.0 / NB_MEAN)).astype(int), 0, NB_MEAN - 1)
    pb1 = np.clip(S['pos'].astype(int), 0, 199)
    for k, col, lab in (('anch', F6.ANCH_COLOR, 'anchored'),
                        ('non', F6.NONANCH_COLOR, 'non-anchored')):
        a, b, _ = sel[k]['block']
        m = S['good'] & (S['spd'] >= 3.0) & (S['tr'] >= a) & (S['tr'] <= b)
        us = np.unique(S['tr'][m])
        mu, se = np.full(NB_MEAN, np.nan), np.full(NB_MEAN, np.nan)
        for i in range(NB_MEAN):
            pt = np.array([S['rad'][m & (pb == i) & (S['tr'] == u)].mean() for u in us])
            pt = pt[np.isfinite(pt)]
            if len(pt) > 1:
                mu[i], se[i] = pt.mean(), pt.std(ddof=1) / np.sqrt(len(pt))
        ax.fill_between(x, mu - se, mu + se, color=col, alpha=.35, linewidth=0,
                        edgecolor='none', zorder=2)
        ax.plot(x, mu, color=col, lw=1.4, zorder=3, label=f'{lab} ({len(us)})')
        mt = S['good'] & (S['spd'] >= 3.0) & (S['tr'] == sel[k]['trial'])
        tv = np.array([S['rad'][mt & (pb1 == i)].mean() if (mt & (pb1 == i)).any() else np.nan
                       for i in range(200)])
        ok = np.isfinite(tv)
        ax.plot(np.arange(200)[ok], tv[ok], '-', color=col, lw=.6, alpha=.75, zorder=4)
    ax.axvspan(90, 110, color='#d8e4d0', alpha=.55, lw=0, zorder=0)
    for x0 in (0, 170):
        ax.axvspan(x0, x0 + 30, color='0.90', lw=0, zorder=0)
    ax.set_xlim(0, 200); ax.set_xticks([0, 100, 200])
    ax.set_xlabel('Position (cm)', fontsize=6.5, labelpad=1)
    ax.set_ylabel('Pupil radius (px)', fontsize=6.5, labelpad=1)
    ax.legend(fontsize=5, frameon=False, loc='upper right', handlelength=1.2,
              borderaxespad=0, labelspacing=.2)
    ax.tick_params(labelsize=5.8, length=2); ax.spines[['top', 'right']].set_visible(False)
    # the raster and PC1
    ax = fig.add_subplot(g[2])
    cmap = ListedColormap([F6.NONANCH_COLOR, F6.ANCH_COLOR])
    nv, nl = P['n_varies'], P['n_locked']
    if nl:
        gap = max(2, int(round(.025 * P['n'])))
        Lr = np.vstack([P['L'][:nv], np.full((gap, P['L'].shape[1]), np.nan), P['L'][nv:]])
    else:
        Lr = P['L']
    ax.imshow(Lr.T, aspect='auto', cmap=cmap, norm=BoundaryNorm([-.5, .5, 1.5], 2),
              interpolation='nearest', extent=[0, Lr.shape[0], ntr - .5, -.5])
    for tb in P['trans']:
        ax.axhline(tb, color='0.15', lw=.7, ls='--', zorder=4)
    ax.set_xlabel(f'MEC cell (n={P["n"]})', fontsize=6.5, labelpad=1)
    ax.set_ylabel('Trial', fontsize=6.5, labelpad=1)
    ax.set_yticks(tick_i); ax.set_yticklabels([int(P['trial'][i]) for i in tick_i])
    ax.set_xticks([]); ax.tick_params(labelsize=5.8, length=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    marks = [ax]
    ax = fig.add_subplot(g[3])
    yy, pc = np.arange(ntr), np.nan_to_num(P['pc1'])
    ax.fill_betweenx(yy, 0, pc, where=pc >= 0, color=F6.ANCH_COLOR, linewidth=0,
                     edgecolor='none', interpolate=True)
    ax.fill_betweenx(yy, 0, pc, where=pc < 0, color=F6.NONANCH_COLOR, linewidth=0,
                     edgecolor='none', interpolate=True)
    ax.axvline(0, color='0.4', lw=.6)
    for tb in P['trans']:
        ax.axhline(tb, color='0.15', lw=.7, ls='--', zorder=4)
    ax.set_ylim(ntr - .5, -.5); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('PC1', fontsize=6.5, labelpad=1)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # pupil radius on every trial
    gh = g[4].subgridspec(1, 2, width_ratios=[1, .07], wspace=.08)
    ax = fig.add_subplot(gh[0])
    H = np.full((ntr, NB_MAP), np.nan)
    pbm = np.clip((S['pos'] / (200.0 / NB_MAP)).astype(int), 0, NB_MAP - 1)
    ok = S['good'] & (S['spd'] >= 3.0)
    for i, u in enumerate(P['trial']):
        mu_ = ok & (S['tr'] == u)
        if mu_.any():
            cnt = np.bincount(pbm[mu_], minlength=NB_MAP).astype(float)
            tot = np.bincount(pbm[mu_], weights=S['rad'][mu_], minlength=NB_MAP)
            H[i] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    cm2 = plt.get_cmap('plasma').copy(); cm2.set_bad('0.9')
    lo, hi = np.nanpercentile(H, [2, 98])
    im = ax.imshow(np.ma.masked_invalid(H), aspect='auto', cmap=cm2, vmin=lo, vmax=hi,
                   interpolation='nearest', extent=[0, 200, ntr - .5, -.5])
    for tb in P['trans']:
        ax.axhline(tb, color='0.15', lw=.7, ls='--', zorder=4)
    ax.set_xticks([0, 100, 200]); ax.set_xlabel('Position (cm)', fontsize=6.5, labelpad=1)
    ax.set_yticks(tick_i); ax.set_yticklabels([]); ax.tick_params(labelsize=5.8, length=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    marks.append(ax)
    cax = fig.add_subplot(gh[1])
    cb = fig.colorbar(im, cax=cax); cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.3, length=1.5, pad=1)
    cax.set_title('px', fontsize=5.5, pad=2)
    # the same map z-scored over trials at each position, so the position
    # profile shared by every trial drops out and only the trial-to-trial
    # change is left
    gz = g[5].subgridspec(1, 2, width_ratios=[1, .07], wspace=.08)
    ax = fig.add_subplot(gz[0])
    with np.errstate(invalid='ignore', divide='ignore'):
        Z = (H - np.nanmean(H, 0)) / np.nanstd(H, 0)
    cm3 = plt.get_cmap('PuOr_r').copy(); cm3.set_bad('0.9')
    im = ax.imshow(np.ma.masked_invalid(Z), aspect='auto', cmap=cm3, vmin=-ZLIM, vmax=ZLIM,
                   interpolation='nearest', extent=[0, 200, ntr - .5, -.5])
    for tb in P['trans']:
        ax.axhline(tb, color='0.15', lw=.7, ls='--', zorder=4)
    ax.set_xticks([0, 100, 200]); ax.set_xlabel('Position (cm)', fontsize=6.5, labelpad=1)
    ax.set_yticks(tick_i); ax.set_yticklabels([]); ax.tick_params(labelsize=5.8, length=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    marks.append(ax)
    cax = fig.add_subplot(gz[1])
    cb = fig.colorbar(im, cax=cax, ticks=[-ZLIM, 0, ZLIM]); cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.3, length=1.5, pad=1)
    cax.set_title('z', fontsize=5.5, pad=2)
    # which traversals the eye frames come from
    for a_ in marks:
        for k, col in (('anch', F6.ANCH_COLOR), ('non', F6.NONANCH_COLOR)):
            iu = int(np.where(P['trial'] == sel[k]['trial'])[0][0])
            a_.plot([-.03], [iu], transform=a_.get_yaxis_transform(), clip_on=False,
                    marker='<', ms=3.2, color=col, zorder=6)
    E = F6.eye_trials()
    e = E[(E.mouse == mo) & (E.day == dy)]
    d = e[e.anch].z_rad.mean() - e[~e.anch].z_rad.mean()
    bb = gs.get_position(fig)
    fig.text(bb.x0 - .035, bb.y1 + .012, letter, fontsize=10, weight='bold', va='bottom')
    fig.text(bb.x0, bb.y1 + .012, f'M{mo} D{dy}   anchored minus non-anchored: '
             f'{d:+.2f} z(pupil radius), {len(e)} trials', fontsize=7.5, va='bottom')
    print(f'M{mo}D{dy}: trials {sel["anch"]["trial"]} / {sel["non"]["trial"]}, '
          f'blocks {sel["anch"]["block"]} / {sel["non"]["block"]}, delta {d:+.3f}')


def main():
    fig = plt.figure(figsize=(11.6, 2.35 * len(SESSIONS)))
    outer = fig.add_gridspec(len(SESSIONS), 1, hspace=.55, left=.06, right=.97,
                             top=.965, bottom=.04)
    for i, (mo, dy) in enumerate(SESSIONS):
        row(fig, outer[i], mo, dy, 'ABCDE'[i])
    fig.savefig(OUT, dpi=220, bbox_inches='tight')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
