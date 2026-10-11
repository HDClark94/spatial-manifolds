"""Figure 8: the anchoring state drawn as two routes through the same circuit.

THIS FIGURE IS SPECULATIVE AND IS LABELLED AS SUCH ON THE PAGE. It belongs to
the Ideas and speculation section, not to Results. Both panels are schematics;
the numbers annotated on them are measured, and each carries the figure it
comes from.

WHAT THE FIGURE SAYS. One circuit, two routes. A self-motion drive is always
present. A landmark drive -- the task frame -- reaches the entorhinal position
estimate only while an engagement gate is open, and the pupil indexes that
gate. Anchored is the open-gate route: the estimate is corrected to the task
frame on every traversal. Non-anchored is the shut-gate route: no correction
arrives, and the correction is MISSING rather than displaced, because
non-anchored trials show no coherent field at any offset and neighbouring
non-anchored traversals do not align with one another. The circuit is the same
in both; what differs is which of its inputs is being heeded.

WHAT CHANGED, AND WHY.
  1. An early version put the switch INSIDE the entorhinal circuit: two loops,
     recurrent and feedforward, generating slow and fast gamma, with
     cholinergic tone selecting between them. The measurements rule that out.
     The spike-field architecture is the same in both states -- phase locking
     to either gamma band (Holm p >= 0.17), the laminar theta phase gradient
     (every layer p >= 0.06), the theta phase at which each band peaks
     (p = 0.33, 0.16) -- and the cells that would have to reconfigure do not.
     So the switch sits on the input, not in the circuit.
  2. This version drops the four evidence panels that used to accompany the
     schematic (the theta-speed hinge, the field-potential changes, the
     per-class readout, and the list of tested nulls) and the prediction band
     beneath them. All of that is measured elsewhere -- Figures 2, 7, S10 and
     S11 -- and the predictions belong in the text. What is left is the claim
     itself: the two routes, side by side, built from the same five boxes so
     that the only difference a reader has to find is which arrow carries.

Writes fig8_model.pdf
"""
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'
plt.rcParams['pdf.fonttype'] = 42

FIG = os.path.dirname(os.path.abspath(__file__))
OUT = f'{FIG}/fig8_model.pdf'

SELF_C = '#2b6cb0'     # self-motion drive, as the interneuron blue of Figure 7
LAND_C = '#c04744'     # landmark drive
GRID_C = '#c04744'
ACH = '#b8860b'        # the gate
OFF = '#b9b4ae'        # a route that is not carrying
ANCH, NONANCH = '#a8559e', '#3f9b8f'

def arrow(ax, p0, p1, color, rad=0., lw=1.3, ls='-', shrink=9, z=3, head=True):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle='-|>' if head else '-', mutation_scale=8, lw=lw,
        color=color, linestyle=ls, shrinkA=shrink, shrinkB=shrink, zorder=z,
        connectionstyle=f'arc3,rad={rad}'))


def box(ax, x, y, w, h, s, fc, tc, fs=6.2, ec='0.75', lw=.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.018',
                                fc=fc, ec=ec, lw=lw, zorder=1))
    ax.text(x + w / 2, y + h / 2, s, fontsize=fs, ha='center', va='center',
            color=tc, linespacing=1.3)


# Both routes are built from the same five elements in the same places, so the
# only thing a reader has to compare is whether the gated arrow carries.
# Everything that differs between the states is collected in these two dicts.
A_ROUTE = dict(
    label='A', state='ANCHORED', hue=ANCH,
    sub='engagement is high, the gate is open, and the estimate is corrected '
        'to the task frame on every lap',
    gate_fc='#fdf6e3', gate_ec=ACH, gate_tc=ACH,
    eng='engagement HIGH — pupil dilated',
    node='OPEN', node_fc='#fdf6e3', node_ec=ACH, node_tc=ACH,
    land_c=LAND_C, land_fc='#fbf0ef', land_tc=LAND_C, land_ls='-',
    land_lw=1.7, blocked=False, land_note='the gate is open:\nthis drive arrives',
    self_note='theta\u2013speed slope 0.218',
    est_note='corrected to the\ntask frame on\nevery traversal',
    out_note='follow the state most\nstrongly (+0.185, +0.131;\nFigure 2)',
)
B_ROUTE = dict(
    label='B', state='NON-ANCHORED', hue=NONANCH,
    sub='engagement is low, the gate is shut, and no correction reaches the '
        'estimate at all',
    gate_fc='#f2f1ef', gate_ec='0.72', gate_tc='0.45',
    eng='engagement LOW\u2003\u2014\u2003pupil constricted',
    node='SHUT', node_fc='#f2f1ef', node_ec='0.62', node_tc='0.45',
    land_c=OFF, land_fc='#f4f3f1', land_tc='0.55', land_ls=(0, (2.2, 1.8)),
    land_lw=1.2, blocked=True, land_note='stopped at the gate: this\ndrive never arrives',
    self_note='theta\u2013speed slope 0.374',
    est_note='no correction, and\nnone survives at any\noffset: missing,\nnot moved',
    out_note='the same cells, the same\ncoupling to either band\n(Holm p \u2265 0.17; S10, S11)',
)

# geometry, shared by both routes (axes fraction). The gated route is drawn
# ABOVE the ungated one so that the engagement box can drop straight onto the
# gate without crossing the self-motion arrow.
ENG = (.335, .858, .665, .118)
LAND = (.000, .585, .300, .150)
NODE = (.352, .638, .108, .082)
SELF = (.000, .215, .300, .150)
EST = (.512, .395, .250, .172)
OUT_ = (.808, .395, .192, .172)


def route(ax, r):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.text(-.050, 1.105, r['label'], transform=ax.transAxes, fontsize=10,
            weight='bold', va='bottom', ha='left')
    ax.text(.0, 1.105, r['state'], transform=ax.transAxes, fontsize=8.6,
            weight='bold', color=r['hue'], va='bottom', ha='left')
    ax.text(.0, 1.032, r['sub'], transform=ax.transAxes, fontsize=6.5,
            color='0.35', va='bottom', ha='left')

    box(ax, *ENG, r['eng'], r['gate_fc'], r['gate_tc'], fs=6.4,
        ec=r['gate_ec'], lw=1.0)
    box(ax, *LAND, 'landmarks\n(the task frame)', r['land_fc'], r['land_tc'])
    box(ax, *SELF, 'self-motion\n(speed, head direction)', '#eef3fa', SELF_C)
    box(ax, *EST, 'entorhinal\nposition estimate', '#f4f1ec', '0.3')
    box(ax, *OUT_, 'grid cells +\nFS interneurons', '#f7f2f6', GRID_C)

    # the landmark drive runs into the gate, and on only if the gate is open
    arrow(ax, (.302, .660), (.348, .678), r['land_c'], lw=r['land_lw'],
          ls=r['land_ls'], shrink=1, head=False)
    box(ax, *NODE, r['node'], r['node_fc'], r['node_tc'], fs=6.0,
        ec=r['node_ec'], lw=1.1)
    arrow(ax, (.472, .658), (.512, .560), r['land_c'], lw=r['land_lw'],
          ls=r['land_ls'], shrink=1)
    if r['blocked']:
        ax.plot(.4945, .609, marker='x', ms=7.5, mew=1.9, color='#8a1c1c',
                zorder=5)
    ax.text(.150, .565, r['land_note'], fontsize=5.7, color=r['land_tc'],
            ha='center', va='top', linespacing=1.35)

    # engagement sets the gate, straight down onto it
    arrow(ax, (.406, .856), (.406, .726), ACH if not r['blocked'] else '0.62',
          lw=1.2, ls=(0, (2.4, 1.6)), shrink=2, z=2)

    # the self-motion drive is always present, gate or no gate
    arrow(ax, (.302, .300), (.512, .432), SELF_C, rad=.14, lw=1.7, shrink=1)
    ax.text(.150, .196, 'always on\n' + r['self_note'], fontsize=5.7,
            color=SELF_C, ha='center', va='top', linespacing=1.35)

    # the readout
    arrow(ax, (.764, .481), (.804, .481), '0.45', lw=1.3, shrink=1)

    ax.text(.612, .372, r['est_note'], fontsize=5.9, color='0.45',
            ha='center', va='top', linespacing=1.4)
    ax.text(.903, .372, r['out_note'], fontsize=5.9, color='0.45',
            ha='center', va='top', linespacing=1.4)


fig = plt.figure(figsize=(9.5, 2.65))
gs = fig.add_gridspec(1, 2, wspace=.20, left=.045, right=.985, top=.813,
                      bottom=.160)
route(fig.add_subplot(gs[0]), A_ROUTE)
route(fig.add_subplot(gs[1]), B_ROUTE)

# a divider, so the two routes read as alternatives rather than as a sequence
fig.add_artist(plt.Line2D([.513, .513], [.195, .900], color='0.86', lw=.8))

fig.text(.045, .105, 'The state changes what arrives at this circuit, not what '
                     'the circuit does with it.', fontsize=8.2, weight='bold',
         color='#2a2a2a', va='center')
fig.text(.045, .042, 'The same five elements are drawn in both routes because '
                     'the measurements say the circuit is the same in both '
                     'states: only the gated arrow differs.',
         fontsize=6.4, color='0.45', va='center')

fig.text(.045, .960, 'HYPOTHESIS', fontsize=8, weight='bold', color='#8a1c1c')
fig.text(.153, .960, '\u2014 both routes are schematics; the values annotated on '
                     'them are measured, and name the figure they come from',
         fontsize=8, color='0.45')

plt.savefig(OUT, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'wrote {OUT}')
