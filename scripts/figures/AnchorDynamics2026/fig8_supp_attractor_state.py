"""Supplement to Figure 9: the attractor bump survives while non-anchored.

Figure 9 and the simulation in Figure S13 both rest on one assumption that
neither of them tests: that the non-anchored state is an attractor which is
still running and merely uncorrected, rather than one that has come apart. The
two possibilities make opposite predictions about how concentrated the
module's activity is on the torus, and nothing else in the paper measures it.

The analysis is attractor_coherence.py; this script only draws it. It is a
supplement rather than a main figure because it settles an assumption rather
than adding a claim -- but it is the panel that decides whether the gated
attractor is a model of these recordings at all.

Writes fig8_supp_attractor_state.pdf
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.mplot3d import proj3d
from scipy.stats import wilcoxon

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

FIG = os.path.dirname(os.path.abspath(__file__))
ROOT = '/Users/harryclark/Documents/spatial-manifolds'
PS = f'{ROOT}/data/population_state'
OUT = f'{FIG}/fig8_supp_attractor_state.pdf'

ANCH, NON = '#a8559e', '#3f9b8f'
TA = ListedColormap([NON, ANCH])
NORM = BoundaryNorm([-0.5, 0.5, 1.5], TA.N)

D = pd.read_csv(f'{PS}/attractor_coherence.csv')
D1 = pd.read_csv(f'{PS}/attractor_coherence_s1.csv')
D0 = pd.read_csv(f'{PS}/attractor_coherence_s0.01.csv')
E = np.load(f'{PS}/attractor_example.npz')
BT = pd.read_csv(f'{PS}/attractor_bump_by_trial.csv')
M = np.load(f'{FIG}/can_gate_model.npz')

fig = plt.figure(figsize=(9.6, 9.4))
gs = fig.add_gridspec(3, 20, hspace=.72, wspace=2.6, left=.055, right=.985,
                      top=.935, bottom=.065, height_ratios=[1.0, 3.2, 1.0])


def sheet(phi):
    """The two phases carried onto the rhombic unit cell (Gardner et al.)."""
    return phi[0] - phi[1] / 2.0, np.sqrt(3) / 2.0 * phi[1]


def lp(ax, s, x=-.24, y=1.07):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='right')


def bare(ax):
    ax.tick_params(labelsize=6.5)
    ax.spines[['top', 'right']].set_visible(False)


def torus3d(fig, parent, cycle):
    """A torus with one of its two cycles traced in the phase colour wheel.

    Cycle 1 runs the long way round the hole and cycle 2 the short way through
    the tube: the two independent ways round the surface, which are the two
    phases B and C plot. The long cycle is drawn as a band on the surface so
    that it is hidden correctly where the body passes in front of it; the
    short cycle is drawn over the body: a ring through the tube is hidden
    almost entirely by the surface under this renderer's depth sorting, so the
    circle is taken on the torus surface itself, projected into two dimensions
    through the same camera, and drawn on top. The half whose outward normal
    faces the camera lies on the visible surface and is drawn solid; the half
    that passes behind the tube is drawn faint, so the cycle still reads as a
    complete loop around the tube.
    """
    bb = parent.get_position()
    w, h = bb.width * .62, bb.height * .34
    rect = [bb.x1 - w, bb.y1 + bb.height * .10, w, h]
    ax = fig.add_axes(rect, projection='3d')
    nu, nv = 160, 64
    u, v = np.linspace(0, 2 * np.pi, nu), np.linspace(0, 2 * np.pi, nv)
    U, V = np.meshgrid(u, v)
    R, r = 1.0, .36
    ax.plot_surface((R + r * np.cos(V)) * np.cos(U),
                    (R + r * np.cos(V)) * np.sin(U), r * np.sin(V),
                    color='0.82', shade=True, linewidth=0, edgecolor='none',
                    antialiased=False, rstride=1, cstride=1)
    if cycle == 1:
        rr = r * 1.02
        UU, VV = np.meshgrid(u, np.linspace(.30, .95, 10))
        C = plt.cm.hsv((UU % (2 * np.pi)) / (2 * np.pi))
        ax.plot_surface((R + rr * np.cos(VV)) * np.cos(UU),
                        (R + rr * np.cos(VV)) * np.sin(UU), rr * np.sin(VV),
                        facecolors=C, shade=False, linewidth=0,
                        edgecolor='none', antialiased=False, rstride=1,
                        cstride=1)
    ax.view_init(elev=44, azim=-62); ax.set_axis_off()
    ax.set_box_aspect((1, 1, .45))
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_zlim(-.85, .85)
    if cycle == 2:
        fig.canvas.draw()
        u0, t = -2.1, np.linspace(0, 2 * np.pi, 240)
        X = (R + r * np.cos(t)) * np.cos(u0)
        Y = (R + r * np.cos(t)) * np.sin(u0)
        Z = r * np.sin(t)
        cam = np.array([np.cos(np.radians(44)) * np.cos(np.radians(-62)),
                        np.cos(np.radians(44)) * np.sin(np.radians(-62)),
                        np.sin(np.radians(44))])
        nrm = np.stack([np.cos(t) * np.cos(u0), np.cos(t) * np.sin(u0),
                        np.sin(t)], 1)
        vis = nrm @ cam > .02              # the half lying on the near surface
        pts = np.array([proj3d.proj_transform(x, y, z, ax.get_proj())[:2]
                        for x, y, z in zip(X, Y, Z)])
        ov = fig.add_axes(rect); ov.set_xlim(0, 1); ov.set_ylim(0, 1)
        ov.axis('off'); ov.patch.set_alpha(0)
        f = ov.transAxes.inverted().transform(ax.transData.transform(pts))
        for k in range(len(t) - 1):
            on = vis[k] and vis[k + 1]
            ov.plot(f[k:k + 2, 0], f[k:k + 2, 1], solid_capstyle='round',
                    lw=2.3 if on else 1.3, alpha=1.0 if on else .30,
                    color=plt.cm.hsv(k / (len(t) - 1)))


def star(p):
    return (f'p = {p:.1e}'.replace('e-0', 'e-') if p < 1e-3
            else f'p = {p:.3f}')


NB = int(E['nbin'])
st = E['state'].astype(bool)
mo_, dy_ = int(E['mouse']), int(E['day'])
t0 = int(E['strip_t0'])
str_tr = E['strip_trial']
TA0, TA1 = int(str_tr[0]), int(str_tr[-1])

# ---- A: the population state that defines the two conditions ---------------
Z = np.load(f'{PS}/labels/M{mo_}D{dy_}.npz', allow_pickle=True)
C = pd.read_csv(f'{PS}/anchoring_cells.csv')
C = C[(C.mouse == mo_) & (C.day == dy_)]
sel = np.isin(Z['cluster_id'], C.cluster_id[C.in_population].values)
L, load = Z['labels'][sel], Z['pc1_load'][sel]
with np.errstate(invalid='ignore'):
    varies = np.array([np.isfinite(r).sum() > 1 and np.nanstd(r) > 0
                       for r in L])
L = L[np.where(varies)[0][np.argsort(load[varies])[::-1]]]
ntr = L.shape[1]


def mark(ax, w=.9):
    """Where panel D's trials come from."""
    for y in (TA0, TA1):
        ax.axhline(y, color='k', lw=w)


gA = gs[0, 0:5].subgridspec(1, 2, wspace=.12, width_ratios=[1, .30])
axA = fig.add_subplot(gA[0, 0])
lp(axA, 'A', x=-.30)
axA.imshow(L.T, aspect='auto', cmap=TA, norm=NORM, interpolation='nearest',
           extent=[0, L.shape[0], ntr, 0])
mark(axA)
axA.set_xticks([]); axA.tick_params(labelsize=6.5)
axA.set_ylabel('trial', fontsize=7.5)
axA.set_xlabel(f'{L.shape[0]} MEC cells', fontsize=7.5)
axA.set_title(f'the population state\nM{mo_} D{dy_}', fontsize=7.6,
              loc='left', color='0.25')
axP = fig.add_subplot(gA[0, 1], sharey=axA)
pc1, tr = Z['pc1'], np.arange(ntr)
axP.fill_betweenx(tr, 0, pc1, where=pc1 >= 0, color=ANCH, lw=0,
                  edgecolor='none', interpolate=True)
axP.fill_betweenx(tr, 0, pc1, where=pc1 < 0, color=NON, lw=0,
                  edgecolor='none', interpolate=True)
axP.axvline(0, color='0.4', lw=.5)
mark(axP, w=.7)
axP.set_xticks([]); axP.tick_params(labelleft=False, length=0)
for sp in axP.spines.values():
    sp.set_visible(False)
axP.set_xlabel('PC1', fontsize=7.0)
axP.annotate('E', xy=(1.08, (TA0 + TA1) / 2), xycoords=('axes fraction',
             'data'), fontsize=7.5, weight='bold', va='center', ha='left',
             annotation_clip=False, color='0.1')

# ---- B, C: the module's two torus phases, in trial order -------------------
# B and C are psi, the population's phase along one lattice axis each. Which
# two matters: on a LINEAR track the bump has one degree of freedom, so the
# track direction projects onto two of the three lattice axes almost equally
# and onto the third hardly at all. The two that advance are therefore nearly
# the same phase -- here they correlate at +0.96 and differ by 12 degrees on
# average -- and drawing both would be drawing one twice. B shows the axis
# that runs along the track and C the one that runs across it, which is the
# independent pair (circular r = +0.05).
AX = E['psi_all'].reshape(len(E['psi_all']), -1, NB)
_adv = np.array([abs(np.median([np.unwrap(r)[-1] - np.unwrap(r)[0]
                                for r in w])) for w in AX])
K_ALONG, K_ACROSS = int(np.argmax(_adv)), int(np.argmin(_adv))
PSI = np.stack([AX[K_ALONG], AX[K_ACROSS]])      # 2 x trials x bins
for k, (lab, pn, cols) in enumerate(
        ((r'$\psi_\parallel$', 'B', slice(6, 9)),
         (r'$\psi_\perp$', 'C', slice(9, 12)))):
    ax = fig.add_subplot(gs[0, cols])
    lp(ax, pn, x=-.09)
    ax.imshow(PSI[k], cmap='hsv', aspect='auto', interpolation='nearest',
              extent=[0, 200, ntr, 0], vmin=-np.pi, vmax=np.pi)
    mark(ax)
    ax.set_yticks([])
    ax.set_xlim(0, 200); ax.set_xticks([0, 100, 200])
    ax.set_xlabel('position (cm)', fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    if not k:
        ax.text(0, -.36, f'{int(E["n_grid"])} grid cells, one module',
                transform=ax.transAxes, fontsize=6.6, color='0.35')
    ax.set_title(f'{lab}  ' + ('along' if not k else 'across'),
                 fontsize=8.0, loc='left', color='0.25', pad=13)
    torus3d(fig, ax, k + 1)
    if k:                                   # one colour wheel for both maps
        bb = ax.get_position()
        cax = fig.add_axes([bb.x1 + .010, bb.y0, .007, bb.height])
        cb = fig.colorbar(plt.cm.ScalarMappable(
            norm=plt.Normalize(-np.pi, np.pi), cmap='hsv'), cax=cax)
        cb.set_ticks([-np.pi, 0, np.pi])
        cb.set_ticklabels(['$-\\pi$', '0', '$\\pi$'])
        cb.ax.tick_params(labelsize=6.0, length=2, pad=1.5)
        cb.outline.set_linewidth(.4)
        cb.set_label('population phase', fontsize=6.4, labelpad=1)

# ---- D: the bump's path on the sheet, unwrapped ----------------------------
AX = E['psi_all'].reshape(len(E['psi_all']), -1, NB)
rev = [np.median([np.unwrap(r)[-1] - np.unwrap(r)[0] for r in AX[k]])
       for k in range(len(AX))]
k1, k2 = np.argsort(-np.abs(rev))[:2]       # the two axes that advance
a0 = t0 - 1
while a0 > 0 and st[a0 - 1]:
    a0 -= 1
n1 = t0
while n1 < len(st) - 1 and not st[n1 + 1]:
    n1 += 1


def _traj(t):
    u1, u2 = np.unwrap(AX[k1, t]), np.unwrap(AX[k2, t])
    return (u1 - u2 / 2) / (2 * np.pi), (np.sqrt(3) / 2 * u2) / (2 * np.pi)


axT = fig.add_subplot(gs[0, 15:20])
lp(axT, 'D', x=-.14)
GR = np.arange(-4, 5)
for nn in GR:                                     # the unit cell of the sheet
    axT.plot([-6, 6], [np.sqrt(3) / 2 * nn] * 2, color='0.74', lw=.5,
             ls=(0, (3, 2)), zorder=0)
    axT.plot([nn + 6 / np.sqrt(3), nn - 6 / np.sqrt(3)], [-6, 6],
             color='0.74', lw=.5, ls=(0, (3, 2)), zorder=0)
for idx, c in ((np.linspace(a0, t0 - 1, 12).astype(int), ANCH),
               (np.linspace(t0, n1, 12).astype(int), NON)):
    seg = [_traj(t) for t in idx]
    for x, y in seg:
        axT.plot(x, y, color=c, lw=.7, alpha=.85, zorder=2)
    xm = np.mean([v[0][0] for v in seg]); ym = np.mean([v[1][0] for v in seg])
    axT.plot(xm, ym, 'o', ms=3.4, color='k', zorder=5)
    if c == ANCH:
        axT.annotate('start', (xm, ym), textcoords='offset points',
                     xytext=(0, -9), fontsize=5.8, color='0.25', ha='center')
xa = np.concatenate([_traj(t)[0] for t in range(a0, n1 + 1)])
ya = np.concatenate([_traj(t)[1] for t in range(a0, n1 + 1)])
axT.set_xlim(np.percentile(xa, .5) - .2, np.percentile(xa, 99.5) + .2)
axT.set_ylim(np.percentile(ya, .5) - .2, np.percentile(ya, 99.5) + .2)
axT.set_xlabel(r'revolutions in $\theta_1$', fontsize=7.5)
axT.set_ylabel(r'revolutions in $\theta_2$', fontsize=7.5, labelpad=1)
bare(axT)
axT.set_title('one trial each, the path unwrapped\nacross copies of the sheet',
              fontsize=7.6, loc='left', color='0.25', pad=13)
axT.text(.98, .24, 'anchored', fontsize=6.2, color=ANCH, ha='right',
         transform=axT.transAxes, va='top')
axT.text(.98, .14, 'non-anchored', fontsize=6.2, color=NON, ha='right',
         transform=axT.transAxes, va='top')

# ---- E: the bump's two coordinates through the session, and at the switch --
# Drawn as Gardner et al. draw it: the two sheet coordinates and the animal's
# position as continuous traces, a window showing that the bump's own speed
# follows the animal's, and the unit cell itself either side of the switch --
# tight while the state holds, scattered once it is lost.
GS1 = gs[1, :].subgridspec(2, 1, hspace=.32, height_ratios=[2.8, .75])
GT = GS1[0].subgridspec(4, 1, hspace=.22)
GB = GS1[1].subgridspec(1, 2, wspace=.22, width_ratios=[1.0, 1.25])
_psit = E['psi_t']; _spd = E['spd_t']; _pos = E['pos_t']
_idx = E['idx_t']; _okt = E['ok_t'].astype(bool)
_t = np.arange(len(_spd)) * 0.1
_thc = np.stack(sheet(_psit)) / (2 * np.pi) % 1.0
NW, WIN = 9, 6                       # nine windows of six trials
starts = np.linspace(t0 - (NW // 2) * WIN, t0 + (NW - NW // 2 - 1) * WIN,
                     NW).astype(int)
_pc1t = pc1[np.clip(_idx, 0, len(pc1) - 1)]
# the two windows the panels below are taken from
_W_SPD = (0.0, 100.0)
_wt = _t[(_idx >= starts[0]) & (_idx < starts[-1] + WIN)]
_W_CELL = (float(_wt.min()), float(_wt.max())) if len(_wt) else (0., 1.)
for r_, (v, col, lab) in enumerate((
        (_pc1t, '0.25', 'PC1'),
        (_thc[0], '#4c72b0', r'$\theta_1$'),
        (_thc[1], '#dd8452', r'$\theta_2$'),
        (_pos, 'k', 'pos\n(cm)'))):
    a = fig.add_subplot(GT[r_])
    if r_ == 0:
        a.fill_between(_t, 0, v, where=v >= 0, color=ANCH, lw=0,
                       interpolate=True, rasterized=True)
        a.fill_between(_t, 0, v, where=v < 0, color=NON, lw=0,
                       interpolate=True, rasterized=True)
        a.axhline(0, color='0.4', lw=.4)
        a.set_ylim(-abs(v).max() * 1.1, abs(v).max() * 1.1)
        a.set_yticks([])
    else:
        a.plot(_t, v, color=col, lw=.2, rasterized=True)
        a.set_ylim(0, 1 if r_ < 3 else 200)
        a.set_yticks([0, 1] if r_ < 3 else [0, 200])
    a.set_xlim(0, _t[-1])
    a.tick_params(labelsize=5.8, labelbottom=(r_ == 3))
    a.set_ylabel(lab, fontsize=6.6, labelpad=2)
    a.spines[['top', 'right']].set_visible(False)
    for (x0, x1), c_ in ((_W_SPD, '0.25'), (_W_CELL, '0.25')):
        a.add_patch(plt.Rectangle((x0, a.get_ylim()[0]), x1 - x0,
                                  a.get_ylim()[1] - a.get_ylim()[0],
                                  fill=False, ec=c_, lw=.7, zorder=6,
                                  clip_on=False if r_ == 3 else True))
    if r_ == 0:
        lp(a, 'E', x=-.055, y=1.10)
        a.set_title('the bump through the session; the boxes are what the two '
                    'panels below show', fontsize=7.6, loc='left', color='0.25')
    if r_ == 3:
        a.set_xlabel('time (s)', fontsize=7.0)

axV = fig.add_subplot(GB[0])
_pp = E['psi'].reshape(2, -1, NB)
_dpdx = np.array([np.median(np.diff(np.unwrap(_pp[k][st], axis=1), axis=1))
                  / (200.0 / NB) for k in range(2)])
_bs = np.array([np.diff(np.unwrap(_psit[k])) / 0.1 / _dpdx[k]
                for k in range(2)]).mean(0)
_z = lambda v: (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
_w = (_t[1:] < 100) & _okt[1:]
axV.plot(_t[1:][_w], _z(_spd[1:][_w]), color='0.45', lw=.9, label='the animal')
axV.plot(_t[1:][_w], _z(_bs[_w]), color='#c04744', lw=.9, label='the bump')
axV.set_xlabel('time (s)', fontsize=7.0)
axV.set_ylabel('speed (z)', fontsize=7.0)
axV.legend(fontsize=5.8, frameon=False, loc='upper right', ncol=2,
           handlelength=1.0, columnspacing=.9, borderaxespad=.1)
axV.tick_params(labelsize=6.2)
axV.spines[['top', 'right']].set_visible(False)
_tr = np.clip(_idx[1:], 0, len(st) - 1)
_an = st[_tr] & _okt[1:]
_sl = lambda q: np.polyfit(_spd[1:][q], _bs[q], 1)[0]
axV.text(0, -.50, 'over the session the bump advances '
         f'{_sl(_an):.2f} cm per cm run while the state holds, '
         f'{_sl(_okt[1:] & ~_an):.2f} once it is lost',
         transform=axV.transAxes, fontsize=5.8, color='0.4', va='top')

_P2 = E['psi'].reshape(2, -1, NB)
axR = fig.add_subplot(GB[1]); axR.axis('off')
axR.set_title('the unit cell across the switch', fontsize=7.0, loc='left',
              color='0.25', pad=2)
hh = np.sqrt(3) / 2
for w, s0 in enumerate(starts):
    lo_, hi_ = max(0, s0), min(len(st), s0 + WIN)
    if hi_ <= lo_:
        continue
    use = np.zeros(len(st), bool); use[lo_:hi_] = True
    col = ANCH if st[lo_:hi_].mean() >= .5 else NON
    a_ = axR.inset_axes([w / NW, .18, 1 / NW * .92, .74])
    th = np.stack(sheet(_P2[:, use, 5:15].reshape(2, -1)))
    x_ = th[0] / (2 * np.pi) % 1.0
    y_ = th[1] / (2 * np.pi) % 1.0
    X, Y = x_ - y_ / 2.0, hh * y_
    a_.plot([0, 1, .5, -.5, 0], [0, 0, hh, hh, 0], color='0.65', lw=.5)
    a_.scatter(X, Y, s=1.3, color=col, lw=0, alpha=.65, rasterized=True)
    a_.set_xlim(-.62, 1.12); a_.set_ylim(-.06, hh + .06)
    a_.set_aspect('equal'); a_.axis('off')
    a_.text(.5, -.04, f'{lo_}', fontsize=4.8, color=col, ha='center',
            va='top', transform=a_.transAxes)
axR.text(.5, -.30, 'six trials each, labelled by first trial', fontsize=5.5,
         color='0.4', ha='center', va='top', transform=axR.transAxes)


# ---- F: how concentrated the population is, moment by moment ---------------
ax = fig.add_subplot(gs[2, 0:6])
lp(ax, 'F', x=-.30)
pv, sh = E['pv'], E['sh']
a, n_ = E['anch'].astype(bool), E['non'].astype(bool)
bins = np.linspace(min(pv.min(), sh.min()), max(pv.max(), sh.max()), 46)
for v, c, lab in ((sh[a | n_], '0.55', 'phase shuffled'),
                  (pv[n_], NON, 'non-anchored'), (pv[a], ANCH, 'anchored')):
    ax.hist(v, bins=bins, density=True, histtype='step', color=c, lw=1.2,
            label=lab)
bare(ax)
ax.set_xlabel('bump population vector, per 100 ms', fontsize=7.5)
ax.set_ylabel('density', fontsize=7.5)
ax.legend(fontsize=6.0, frameon=False, loc='upper right', handlelength=1.1,
          borderpad=.1, labelspacing=.25)
ax.set_title('both states above the shuffle', fontsize=7.6, loc='left',
             color='0.25')

# ---- F: every session, against its own shuffle -----------------------------
ax = fig.add_subplot(gs[2, 7:13])
lp(ax, 'G')
for _, r in D.iterrows():
    ax.plot([0, 1], [r.bump_anch, r.bump_non], color='0.78', lw=.6, zorder=1)
rng = np.random.default_rng(0)
for x, col, c in ((0, 'bump_anch', ANCH), (1, 'bump_non', NON)):
    v = D[col].values
    ax.scatter(np.full(len(v), x) + rng.normal(0, .03, len(v)), v, s=9,
               color=c, lw=0, zorder=2)
    ax.plot([x - .18, x + .18], [v.mean()] * 2, color=c, lw=1.8, zorder=3)
ax.axhline(0, color='k', lw=.8, ls=(0, (3, 2)))
ax.set_xlim(-.45, 1.45); ax.set_xticks([0, 1])
ax.set_xticklabels(['anchored', 'non-\nanchored'], fontsize=6.8)
ax.set_ylabel('bump above shuffle', fontsize=7.5)
bare(ax)
pa, pn_ = wilcoxon(D.bump_anch).pvalue, wilcoxon(D.bump_non).pvalue
pd_ = wilcoxon(D.bump_anch - D.bump_non).pvalue
ax.set_ylim(-.004, D.bump_anch.max() * 1.45)
psp = wilcoxon(D.sm_anch - D.sm_non).pvalue
ax.text(.5, .99, f'both above zero: {star(max(pa, pn_))}\n'
        f'anchored higher: {star(pd_)}\n'
        f'speeds equated: {star(psp)}', transform=ax.transAxes,
        fontsize=6.3, ha='center', va='top', color='0.25')
ax.set_title(f'{len(D)} sessions, {int(D.n_grid.sum())} grid cells',
             fontsize=7.6, loc='left', color='0.25')

# ---- G: the bump across the switch, trial by trial -------------------------
WLAG = 8


def _epochs(a):
    """State epochs, with runs shorter than three trials absorbed into the
    preceding one -- the definition of a transition used everywhere else in
    the paper, since a trial or two across the axis is a crossing rather than
    a state the population entered."""
    a, out, i = a.astype(int).copy(), [], 0
    while i < len(a):
        j = i
        while j < len(a) and a[j] == a[i]:
            j += 1
        if out and j - i < 3:
            a[i:j] = a[i - 1]
            out[-1] = (out[-1][0], j, out[-1][2])
        else:
            out.append((i, j, bool(a[i])))
        i = j
    return out


segs, slopes, dist = [], [], []
for (mo_s, dy_s), g in BT.groupby(['mouse', 'day']):
    g = g.sort_values('trial')
    tr, bb, aa = g.trial.values, g.bump.values, g.anch.values
    full = np.full(tr.max() + 1, np.nan); full[tr] = bb
    fa = np.zeros(tr.max() + 1, bool); fa[tr] = aa
    ep = _epochs(fa)
    for i, j, state in ep:
        if j - i >= 8:
            seg = full[i:j]; kk = np.arange(j - i); o = np.isfinite(seg)
            if o.sum() >= 6:
                slopes.append((state, np.polyfit(kk[o], seg[o], 1)[0]))
    for i, j, state in ep[1:]:
        if state or not fa[i - 1]:              # anchored -> non-anchored only
            continue
        if i - WLAG < 0 or i + WLAG > len(full):
            continue
        q = full[i - WLAG:i + WLAG]
        if np.isfinite(q).sum() >= 1.4 * WLAG:
            segs.append(q)
    cl = np.concatenate([np.full(j - i, st_) for i, j, st_ in ep])
    sw = np.flatnonzero(np.diff(cl.astype(int)) != 0) + .5
    d = (np.min(np.abs(np.arange(len(fa))[:, None] - sw[None, :]), 1)
         if len(sw) else np.full(len(fa), np.inf))
    for t, v in zip(tr, bb):
        dist.append((mo_s, dy_s, d[t], bool(cl[t]), v))
SEG = np.array(segs)
D2 = pd.DataFrame(dist, columns=['mouse', 'day', 'dist', 'anch', 'bump'])

ax = fig.add_subplot(gs[2, 14:20])
lp(ax, 'H')
lags = np.arange(-WLAG, WLAG)
m_ = np.nanmean(SEG, 0)
se_ = np.nanstd(SEG, 0) / np.sqrt(np.sum(np.isfinite(SEG), 0))
for sl_, c in ((slice(0, WLAG), ANCH), (slice(WLAG, 2 * WLAG), NON)):
    ax.plot(lags[sl_], m_[sl_], color=c, lw=1.3, marker='o', ms=2.6, mew=0)
    ax.fill_between(lags[sl_], (m_ - se_)[sl_], (m_ + se_)[sl_], color=c,
                    alpha=.22, linewidth=0, edgecolor='none')
ax.axvline(-.5, color='k', lw=.8, ls=(0, (3, 2)))
ax.set_xlabel('trial from the switch', fontsize=7.5)
ax.set_ylabel('bump above shuffle', fontsize=7.5)
step = np.nanmean(SEG[:, WLAG:WLAG + 3], 1) - np.nanmean(
    SEG[:, WLAG - 3:WLAG], 1)
p_step = wilcoxon(step[np.isfinite(step)]).pvalue
sl_non = np.array([v for s_, v in slopes if not s_])
p_slope = wilcoxon(sl_non).pvalue
bare(ax)
ax.set_ylim(.018, .072)
ax.text(.5, .99, f'no step: {star(p_step)}\nno decay after: {star(p_slope)}',
        transform=ax.transAxes, fontsize=6.2, ha='center', va='top',
        color='0.25')
ax.set_title(f'{len(SEG)} switches: it does not\ndecay into the new state',
             fontsize=7.6, loc='left', color='0.25')

# ---- H2: how far from the switch the difference is expressed ---------------
BINS = ((0, 2, '0-2'), (3, 5, '3-5'), (6, 10, '6-10'), (11, 1e9, '>10'))
mm, ee, pp = [], [], []
for lo, hi, _ in BINS:
    q = D2[(D2.dist >= lo) & (D2.dist <= hi)]
    S_ = q.groupby(['mouse', 'day']).apply(
        lambda g: pd.Series({'a': g[g.anch].bump.mean(),
                             'n': g[~g.anch].bump.mean()}),
        include_groups=False).dropna()
    d_ = (S_.a - S_.n).values
    mm.append(d_.mean()); ee.append(d_.std() / np.sqrt(len(d_)))
    pp.append(wilcoxon(d_).pvalue)
# the smoothing series and the model comparison are reported in the Methods
RET = np.stack([100 * T.bump_non.values / T.bump_anch.values
                for T in (D0, D1, D)])
keep = np.median(RET, axis=1)
mo, mn = M['bump_open'], M['bump_shut']
mk = 100 * (mn[0] - mn[1]) / (mo[0] - mo[1])
p_sm = wilcoxon(RET[0] - RET[2]).pvalue

fig.savefig(OUT, dpi=300, bbox_inches='tight')
print(f'wrote {OUT}')
print(f'  model keeps {mk:.0f}%; recordings {keep[0]:.0f}/{keep[1]:.0f}/'
      f'{keep[2]:.0f}% at none/100/250 ms, none vs 250 p = {p_sm:.3f}')
print(f'  anchored {D.bump_anch.mean():+.4f} p={pa:.2g}; non-anchored '
      f'{D.bump_non.mean():+.4f} p={pn_:.2g}; difference p={pd_:.2g}')
print(f'  {len(SEG)} switches: step {np.nanmean(step):+.4f} p={p_step:.3g}; '
      f'non-anchored within-epoch slope {sl_non.mean():+.5f} p={p_slope:.3g}')
print('  difference by distance from switch: ' +
      ', '.join(f'{b[2]} {v:+.4f} (p={q:.3g})'
                for b, v, q in zip(BINS, mm, pp)))
