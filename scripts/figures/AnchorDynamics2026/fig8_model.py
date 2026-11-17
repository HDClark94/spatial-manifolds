"""Figure 10: one circuit, one gate, and what each setting of the gate produces.

THIS FIGURE IS SPECULATIVE AND IS LABELLED AS SUCH ON THE PAGE. It belongs to
the Ideas and speculation section, not to Results. The circuit is a schematic
and the six small panels under it are SIMULATED, drawn from the account rather
than measured from the data; every measured number on the page names the figure
it comes from.

WHAT THE FIGURE SAYS. There is one circuit, not two. Two sources feed the
entorhinal position estimate -- a self-motion drive that is always present, and
a landmark drive carrying the task frame that passes through a gate. The gate
is what the anchoring state IS: open, the estimate is corrected to the task
frame on every traversal; shut, it is not. Engagement sets the gate, and in
this task engagement goes with a CONSTRICTED pupil, not a dilated one (Figure
6: pupil falls 0.41 z on entry to the anchored state and rises 0.49 z on exit,
and hit rate falls monotonically from 0.83 to 0.39 across pupil deciles, so the
constricted state is the focused one). The position estimate then drives
behaviour -- where the animal slows and stops.

WHAT CHANGED, AND WHY.
  1. An early version put the switch INSIDE the entorhinal circuit: two loops,
     recurrent and feedforward, generating slow and fast gamma, with
     cholinergic tone selecting between them. The measurements rule that out.
     The spike-field architecture is the same in both states -- phase locking
     to either gamma band (Holm p >= 0.17), the laminar theta phase gradient
     (every layer p >= 0.06), the theta phase at which each band peaks
     (p = 0.33, 0.16) -- and the cells that would have to reconfigure do not.
     So the switch sits on the input, not in the circuit.
  2. A later version drew the two states as two separate circuits, side by
     side. That said the opposite of what is meant: the circuit is ONE circuit
     and only the gate moves. It is now drawn once.
  3. That version also had the pupil DILATED in the engaged state, which is
     backwards for this task -- see above and Figure 6.

THE TEN SMALL PANELS ARE SIMULATED. Three periodic 1D grid cells of different
spacing, the position decoded from a continuous attractor fed by them, and the
stops the animal makes on uncued traversals, over ten traversals of the 200 cm
track. The decode is drawn as the
decoded position itself -- jet, wrapped onto the track -- rather than as an
error, so the panel shows what the code says about where the animal is.
Anchored: the landmark drive re-references the attractor at the start of each
traversal, so fields recur at the same places and the decoded position runs
0 to 200 cm identically on every lap. Non-anchored: nothing re-references it,
so the phase is re-drawn each traversal and the decode starts from an
arbitrary offset and integrates away within the lap. The stops raster is the
account's prediction about BEHAVIOUR, and it is drawn for uncued traversals
because that is where it has to show: the reward zone at 90-110 cm carries no
visual cue on those trials, so stopping there requires the estimate to be on
the track. Measured, anchoring buys most of its behavioural advantage exactly
there (the uncued-minus-cued anchoring benefit is +0.094, p = 0.012; Figure
1), which is why behaviour is drawn downstream of the gate rather than beside
it.

THE FIELD-POTENTIAL CHANGES ARE DRAWN ON THE SCHEMATIC, beside the estimate,
because the frequency-specific part of the result is what says the change is a
rotation of the input balance rather than a gain change: slow gamma rises,
fast gamma, theta and theta-fast coupling fall, and the difference crosses
zero at a median of 67 Hz in 25 of 26 sessions (Figure 8).
The non-anchored panels are drawn with the phase RE-DRAWN rather than
progressively displaced, because that is what the data show: non-anchored
trials have no coherent field at any offset once the unrelated-cell floor is
applied, and neighbouring non-anchored traversals do not align with one
another, so the correction is missing rather than moved.

Writes fig8_model.pdf
"""
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Bbox

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'
plt.rcParams['pdf.fonttype'] = 42

FIG = os.path.dirname(os.path.abspath(__file__))
OUT = f'{FIG}/fig8_model.pdf'

SELF_C = '#2b6cb0'     # self-motion drive
LAND_C = '#c04744'     # landmark drive
GRID_C = '#c04744'
INT_C = '#2b6cb0'
ACH = '#b8860b'        # engagement / the gate
ANCH, NONANCH = '#a8559e', '#3f9b8f'


def arrow(ax, p0, p1, color, rad=0., lw=1.3, ls='-', shrink=9, z=3, head=True,
          ms=8):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle='-|>' if head else '-', mutation_scale=ms, lw=lw,
        color=color, linestyle=ls, shrinkA=shrink, shrinkB=shrink, zorder=z,
        connectionstyle=f'arc3,rad={rad}'))


def box(ax, rect, s, fc, tc, fs=6.4, ec='0.75', lw=.9, z=1, pad=.010):
    x, y, w, h = rect
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad={pad}',
                                fc=fc, ec=ec, lw=lw, zorder=z))
    if s:
        ax.text(x + w / 2, y + h / 2, s, fontsize=fs, ha='center', va='center',
                color=tc, linespacing=1.3, zorder=z + 1)


# A on the left, B (the ring, one dimension) and C (the torus, two) stacked to
# its right (2026-11-01). The circuit is the same circuit, re-flowed top to
# bottom -- drives, estimate, action -- to fill a column. The simulated rhythm,
# code, decoded position and stops that once sat under it were removed.
fig = plt.figure(figsize=(11.0, 4.9))
ax = fig.add_axes([0, .02, .34, .96]); ax.set_axis_off()
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# ── A: the circuit, drawn once ────────────────────────────────────────────
LAND = (.03, .80, .40, .11)
GATE = (.58, .80, .26, .11)
SELF = (.03, .60, .40, .11)
EST = (.05, .20, .86, .30)
BEH = (.20, .02, .52, .11)
ax.text(0, .99, 'A', fontsize=10, weight='bold', va='top', ha='left', color='0.1')

box(ax, LAND, 'landmarks\n(the task frame)', '#fbf0ef', LAND_C)
box(ax, SELF, 'self-motion\n(speed, head direction)', '#eef3fa', SELF_C)
box(ax, GATE, 'engagement\ngate', '#fdf6e3', ACH, fs=6.6, ec=ACH, lw=1.3)

# the two drives
arrow(ax, (.44, .855), (.575, .855), LAND_C, lw=1.6, shrink=1, head=False)
ax.text(.51, .87, 'gated', fontsize=5.8, color=LAND_C, ha='center')
arrow(ax, (.71, .79), (.71, .51), LAND_C, lw=1.6, shrink=1)
ax.text(.73, .69, 'OPEN \u2192 anchored\nSHUT \u2192\nnon-anchored', fontsize=5.8,
        ha='left', va='center', color=ACH, linespacing=1.3)
arrow(ax, (.23, .59), (.23, .51), SELF_C, lw=1.6, shrink=1)
ax.text(.25, .555, 'always on', fontsize=5.8, color=SELF_C, ha='left', va='center')
ax.text(.25, .527, 'theta\u2013speed slope 0.343 \u2192 0.198', fontsize=5.4,
        color='0.5', ha='left', va='center')

# the estimate, with the cells that carry it drawn inside it
box(ax, EST, '', '#f7f5f1', '0.3', ec='0.68', lw=1.0)
ax.text(.48, .465, 'entorhinal position estimate', fontsize=7.0,
        ha='center', va='center', color='0.25', zorder=4)
ax.text(.48, .438, 'a continuous attractor', fontsize=5.8,
        ha='center', va='center', color='0.55', zorder=4)
box(ax, (.09, .235, .35, .16), 'grid\ncells', '#fbf0ef', GRID_C, fs=6.4, z=4)
box(ax, (.52, .235, .35, .16), 'FS\ninterneurons', '#eef3fa', INT_C, fs=6.4, z=4)
arrow(ax, (.445, .34), (.515, .34), '0.5', lw=1.0, shrink=1, z=6, ms=6)
arrow(ax, (.515, .29), (.445, .29), '0.5', lw=1.0, shrink=1, z=6, ms=6)

# the readout
arrow(ax, (.46, .195), (.46, .135), '0.45', lw=1.4, shrink=1)
box(ax, BEH, 'action (where the animal\nslows and stops)', '#f2f4f1', '0.3')


# ── B: what the gate does to the map, in the world/map rings of Figure 2C ──
def rings(ax, x0, world, mp, col):
    """One world ring over one map ring, with the arrows that tie each point on
    the map to its place in the world, as fig1_supp_neighbour_bars.rings. world
    and mp are (shear, shift, phase)."""
    t = np.linspace(0, 2 * np.pi, 200)
    tk = np.deg2rad([-160, -105, -50, 15, 75, 135])
    xy = lambda tt, cy, sh, dx, ph: (x0 + dx + 1.15 * np.cos(tt + ph),
                                     cy + .3 * np.sin(tt + ph) + sh * np.cos(tt + ph))
    for cy, (sh, dx, ph), c in ((1.35, world, '0.3'), (0, mp, col)):
        ax.plot(*xy(t, cy, sh, dx, 0), color=c, lw=1.4, zorder=2)
    pw, pm = xy(tk, 1.35, *world), xy(tk, 0, *mp)
    for k in range(len(tk)):
        ax.annotate('', xy=(pm[0][k], pm[1][k]), xytext=(pw[0][k], pw[1][k]), zorder=1,
                    arrowprops=dict(arrowstyle='<|-|>', color='0.55', lw=.6, mutation_scale=4.5,
                                    shrinkA=2.2, shrinkB=2.2))
    ax.scatter(*pw, s=9, facecolor='w', edgecolor='0.3', lw=.7, zorder=3)
    ax.scatter(*pm, s=9, facecolor='w', edgecolor=col, lw=.7, zorder=3)


axr = fig.add_axes([.37, .5, .62, .5]); axr.set_axis_off()
# open: the map sits on the world point for point. Shut: the landmark drive no
# longer corrects it, the bump under-integrates, and the error builds lap on
# lap (Figure S17) -- drawn as the map ring slipping further from where the
# anchored map sits (dashed) on each lap, each lap's ring a step lower.
rings(axr, 1.3, (0, 0, 0), (0, 0, 0), ANCH)
axr.text(1.3, -.62, 'gate open: anchored', ha='center', va='top', fontsize=6.2, color=ANCH,
         weight='bold')
axr.text(1.3, -.95, 'map tied to the world', ha='center', va='top', fontsize=5.8, color='0.35')

X1, STEP, NLAP = 9.0, .33, 6
_t = np.linspace(0, 2 * np.pi, 200)
_tk = np.deg2rad([-160, -105, -50, 15, 75, 135])
ring = lambda tt, x0, cy, sh, dx, ph: (x0 + dx + 1.15 * np.cos(tt + ph),
                                       cy + .3 * np.sin(tt + ph) + sh * np.cos(tt + ph))
axr.plot(*ring(_t, X1, 1.35, 0, 0, 0), color='0.3', lw=1.4, zorder=2)          # the world
pw = ring(_tk, X1, 1.35, 0, 0, 0)
axr.plot(*ring(_t, X1, 0, 0, 0, 0), color=ANCH, lw=1.0, ls=(0, (3, 2)), zorder=1.5)  # where anchored sits
for k in range(NLAP):
    sh, dx, ph = -.09 * (k + 1), .17 * (k + 1), .4 * (k + 1)    # the slip grows each lap
    cy = -STEP * k
    al = 1 - .1 * k
    axr.plot(*ring(_t, X1, cy, sh, dx, 0), color=NONANCH, lw=1.2, alpha=al, zorder=3 + .01 * k)
    pm = ring(_tk, X1, cy, sh, dx, ph)
    axr.scatter(*pm, s=7, facecolor='w', edgecolor=NONANCH, lw=.6, alpha=al, zorder=4)
    if k == 0:                                         # lap n tied back to the world
        for j in range(len(_tk)):
            axr.annotate('', xy=(pm[0][j], pm[1][j]), xytext=(pw[0][j], pw[1][j]), zorder=1,
                         arrowprops=dict(arrowstyle='<|-|>', color='0.55', lw=.6,
                                         mutation_scale=4.5, shrinkA=2.2, shrinkB=2.2))
    axr.text(X1 + 1.3 + dx + .25, cy - .02, 'lap n' if k == 0 else f'n + {k}', fontsize=5.6,
             color=NONANCH, va='center', ha='left', alpha=max(al, .6))
axr.scatter(*pw, s=9, facecolor='w', edgecolor='0.3', lw=.7, zorder=3)
axr.text(X1, -STEP * (NLAP - 1) - .62, 'gate shut: non-anchored', ha='center', va='top',
         fontsize=6.2, color=NONANCH, weight='bold')
axr.text(X1, -STEP * (NLAP - 1) - .95, 'the map slips further each lap; dashed, where the anchored map sits',
         ha='center', va='top', fontsize=5.8, color='0.35')
axr.axhline(.68, xmin=0, xmax=1, color='0.85', lw=.5, ls=(0, (2, 2)), zorder=0)
axr.text(-.3, 1.35, 'world', ha='right', va='center', fontsize=6, color='0.3')
axr.text(-.3, 0, 'map', ha='right', va='center', fontsize=6, color='0.3')
axr.set_xlim(-1.6, 13.4); axr.set_ylim(-STEP * (NLAP - 1) - 1.25, 1.95)
axr.text(-1.6, 1.95, 'B', fontsize=10, weight='bold', va='top', ha='left', color='0.1')
axr.text(-1.1, 1.95, 'one dimension: a ring', fontsize=6.6, va='top', ha='left', color='0.3')

# ── C: the same in two dimensions -- the map a torus, the grid-cell attractor ─
# The world is a flat sheet, the map the torus on which grid-cell population
# activity lives (two periodic phases), drawn in 3D. One trajectory through the
# world: anchored, its image on the torus is the anchored image; gate shut, the
# image starts from a re-drawn phase and under-integrates (Figures S16, S17),
# so it ends somewhere else. Dashed on that torus, where the anchored image runs.
from mpl_toolkits.mplot3d import proj3d

axt = fig.add_axes([.37, .0, .62, .47]); axt.set_axis_off()
axt.set_zorder(5); axt.patch.set_visible(False)            # 2D overlay above the tori
axt.set_xlim(-1.6, 13.4); axt.set_ylim(-1.6, 2.3)
TR, Tr, ELEV, AZIM = 1.0, .42, 38, -90
_s = np.linspace(0, 1, 300)                               # one smooth, irregular path
U = .5 + .30 * np.sin(2 * np.pi * .9 * _s + .3) + .12 * np.sin(2 * np.pi * 2.3 * _s + 1.)
V = .5 + .28 * np.sin(2 * np.pi * 1.3 * _s + 2.) + .12 * np.sin(2 * np.pi * 3.1 * _s)
TIES = [0, 100, 200, 299]
# world position -> the two grid phases; the path stays on the near face
TH_A, PH_A = -np.pi * 1.0 + 1.0 * np.pi * U, -.2 + 2.6 * V
TH_N = TH_A[0] + .6 + .65 * (TH_A - TH_A[0])               # re-drawn start, 0.65 gain
PH_N = PH_A[0] - .6 + .65 * (PH_A - PH_A[0])


def torus_pts(th, ph):
    rr = TR + Tr * np.cos(ph)
    return rr * np.cos(th), rr * np.sin(th), Tr * np.sin(ph)


_vw = np.array([np.cos(np.deg2rad(ELEV)) * np.cos(np.deg2rad(AZIM)),
                np.cos(np.deg2rad(ELEV)) * np.sin(np.deg2rad(AZIM)), np.sin(np.deg2rad(ELEV))])


def on_torus(a3, th, ph, col, lw, ls='-', z=3):
    """A path on the torus, solid where the surface faces the viewer and faint
    where it runs round the back."""
    x, y, z_ = torus_pts(th, ph)
    vis = (np.cos(ph) * np.cos(th) * _vw[0] + np.cos(ph) * np.sin(th) * _vw[1]
           + np.sin(ph) * _vw[2]) > 0
    for m, al in ((vis, 1.), (~vis, .25)):
        xx, yy, zz = (np.where(m, q, np.nan) for q in (x, y, z_))
        a3.plot(xx, yy, zz, color=col, lw=lw, ls=ls, alpha=al, zorder=z)


def torus_axes(x0, col):
    p0 = fig.transFigure.inverted().transform(axt.transData.transform([(x0 - 2.2, -1.3),
                                                                       (x0 + 2.2, .95)]))
    a3 = fig.add_axes([*p0[0], *(p0[1] - p0[0])], projection='3d', computed_zorder=False)
    a3.set_axis_off(); a3.patch.set_visible(False)
    a3.view_init(ELEV, AZIM)
    a3.set_box_aspect((1, 1, Tr / (TR + Tr)), zoom=1.2)
    L = TR + Tr
    a3.set_xlim(-L, L); a3.set_ylim(-L, L); a3.set_zlim(-Tr, Tr)
    th, ph = np.meshgrid(np.linspace(0, 2 * np.pi, 60), np.linspace(0, 2 * np.pi, 30))
    a3.plot_surface(*torus_pts(th, ph), color=col, alpha=.16, lw=0, shade=True,
                    antialiased=True, zorder=1)
    a3.plot_wireframe(*torus_pts(th, ph), rstride=3, cstride=5, color=col, lw=.25,
                      alpha=.35, zorder=1.5)
    return a3


def world_sheet(ax, x0, y0):
    A_, B_, D_ = np.array([x0 - 1.15, y0]), np.array([x0 + 1.15, y0]), np.array([x0 - .8, y0 + .8])
    P = lambda u, v: A_ + np.multiply.outer(u, B_ - A_) + np.multiply.outer(v, D_ - A_)
    ax.add_patch(plt.Polygon([P(0, 0), P(1, 0), P(1, 1), P(0, 1)], closed=True, fc='0.96',
                             ec='0.3', lw=1.1, zorder=1))
    w = P(U, V)
    ax.plot(w[:, 0], w[:, 1], color='0.3', lw=1.0, zorder=2)
    ax.scatter(w[TIES, 0], w[TIES, 1], s=9, facecolor='w', edgecolor='0.3', lw=.7, zorder=3)
    return w[TIES]


def to_axt(a3, th, ph):
    x2, y2, _ = proj3d.proj_transform(*torus_pts(th, ph), a3.get_proj())
    return axt.transData.inverted().transform(a3.transData.transform(np.c_[x2, y2]))


TOR = []
for x0, col, th, ph in ((1.3, ANCH, TH_A, PH_A), (X1, NONANCH, TH_N, PH_N)):
    a3 = torus_axes(x0, col)
    if col == NONANCH:                                     # where the anchored image runs
        on_torus(a3, TH_A, PH_A, ANCH, .8, ls=(0, (3, 2)), z=2)
    on_torus(a3, th, ph, col, 1.3)
    TOR.append((a3, x0, col, th, ph))
fig.canvas.draw()                                          # fixes the 3D projections
for a3, x0, col, th, ph in TOR:
    pw = world_sheet(axt, x0, 1.05)
    pm = to_axt(a3, th[TIES], ph[TIES])
    axt.scatter(pm[:, 0], pm[:, 1], s=9, facecolor='w', edgecolor=col, lw=.7, zorder=4)
    for j in range(len(TIES)):
        axt.annotate('', xy=pm[j], xytext=pw[j], zorder=2,
                     arrowprops=dict(arrowstyle='<|-|>', color='0.55', lw=.6, mutation_scale=4.5,
                                     shrinkA=2.2, shrinkB=2.2))
axt.text(1.3, -1.2, 'gate open: anchored', ha='center', va='top', fontsize=6.2, color=ANCH,
         weight='bold')
axt.text(1.3, -1.5, 'the path maps onto the torus', ha='center', va='top', fontsize=5.8,
         color='0.35')
axt.text(X1, -1.2, 'gate shut: non-anchored', ha='center', va='top', fontsize=6.2,
         color=NONANCH, weight='bold')
axt.text(X1, -1.5, 'both phases start elsewhere and lag; dashed, the anchored path',
         ha='center', va='top', fontsize=5.8, color='0.35')
axt.text(-.3, 1.45, 'world', ha='right', va='center', fontsize=6, color='0.3')
axt.text(-.3, -.15, 'map', ha='right', va='center', fontsize=6, color='0.3')
axt.text(-1.6, 2.3, 'C', fontsize=10, weight='bold', va='top', ha='left', color='0.1')
axt.text(-1.1, 2.3, 'two dimensions: a torus', fontsize=6.6, va='top', ha='left', color='0.3')

# the gate, both ways, once for B and C together
axg = fig.add_axes([.37, 0, .62, 1]); axg.set_axis_off(); axg.set_zorder(6)
axg.patch.set_visible(False); axg.set_xlim(-1.6, 13.4); axg.set_ylim(0, 1)
axg.annotate('', xy=(6.9, .56), xytext=(3.4, .56),
             arrowprops=dict(arrowstyle='-|>', color=ACH, lw=1.2, mutation_scale=9))
axg.text(5.15, .575, 'gate shuts', ha='center', va='bottom', fontsize=6.2, color=ACH)
axg.annotate('', xy=(3.4, .44), xytext=(6.9, .44),
             arrowprops=dict(arrowstyle='-|>', color=ACH, lw=1.2, mutation_scale=9))
axg.text(5.15, .425, 'gate opens:\nmap re-anchored', ha='center', va='top', fontsize=6.2,
         color=ACH, linespacing=1.1)

plt.savefig(OUT, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'wrote {OUT}')
