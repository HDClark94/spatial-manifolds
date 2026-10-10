"""Figure 8: the two-loop account, drawn as a hypothesis rather than a result.

THIS FIGURE IS SPECULATIVE AND IS LABELLED AS SUCH ON THE PAGE. It belongs to
the Ideas and speculation section, not to Results. Panels A and E are drawn
schematics; C and D carry measured values, and so does one feature of B -- the
theta-phase offset between the two gamma bands, which was measured rather than
assumed after an earlier version of the panel drew it wrong (see below).

WHAT CHANGED, AND WHY. An earlier version put the switch inside the entorhinal
circuit: two loops, recurrent and feedforward, generating slow and fast gamma,
with cholinergic tone selecting between them. The measurements in panel E rule
that out. The spike-field architecture is the same in both states -- phase
locking to either gamma band, the laminar theta phase gradient, the theta phase
at which each band peaks -- and the cells that would have to reconfigure do not.
What does change is the balance of drive: the spectrum rotates about 67 Hz and
theta's entrainment to running speed halves, while the speed CELLS code speed
just as well. So the switch is placed on the input, not in the circuit.

    A   THE PROPOSAL. Two sources feed the entorhinal position estimate: a
        self-motion drive that is always present, and a landmark drive that is
        gated by engagement. Anchored means the estimate is corrected to the
        task frame on each traversal; non-anchored means it is not. The
        correction is missing rather than displaced -- no coherent field
        survives at any offset, and neighbouring non-anchored traversals do not
        align with one another -- so this is not a drifting or re-referenced
        map.
    B   THE MEASURED HINGE. Theta's entrainment to running speed roughly halves
        when anchored (slope 0.374 to 0.218, p = 0.045), while the open-field
        speed cells code speed just as well in both states (+11.9%, p = 0.12).
        The velocity signal is intact and the rhythm is heeding it less, which
        is what a change in the mixture of drive looks like and is not what a
        change in the cells would look like.
    C   THE FIELD POTENTIAL (measured, Figure 7).
    D   THE READOUT (measured, Figure 2). Every class follows the state; the
        two populations of the readout motif do so by about twice as much.
    E   WHAT THE STATE DOES NOT CHANGE: six tested nulls, which constrain this
        account more than any of its positive panels do. An earlier version of
        this panel proposed a cross-region coherence test to earn a slow and
        fast band assignment borrowed from hippocampus; that test cannot be
        run, because the field potential exists only for the entorhinal
        recordings.

WHAT THIS FIGURE DELIBERATELY DOES NOT DEPICT. Earlier versions of this account
located anchoring in the GRID PHASE -- the map pinned in the anchored state and
displaced, drifting, or reset to an arbitrary offset in the non-anchored one.
Every phase-level version is excluded by the data (see drift_vs_remap.py):
non-anchored trials have no coherent field at any offset once the
unrelated-cell floor is applied, no alignment between neighbouring trials, and
no gradient along the track. So nothing here shows a displaced grid bump, and
nothing here claims what the phase does on a non-anchored trial. The claim is
about which loop is running and what sets it, not about where the map sits.

Writes fig8_model.pdf
"""
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'
plt.rcParams['pdf.fonttype'] = 42

FIG = os.path.dirname(os.path.abspath(__file__))
OUT = f'{FIG}/fig8_model.pdf'

SLOW = '#2b6cb0'      # slow gamma, as in Figure 7
FAST = '#c04744'      # fast gamma
THETA = '#7b4173'
GRID_C = '#c04744'
INT_C = '#2b6cb0'
ACH = '#b8860b'
ANCH, NONANCH = '#a8559e', '#3f9b8f'


def _lp(ax, s, dx=-.14, dy=1.0):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='left')


def arrow(ax, p0, p1, color, rad=0., lw=1.3, inhib=False, ls='-', shrink=9):
    """Excitatory (arrowhead) or inhibitory (bar) connection."""
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle='-[' if inhib else '-|>',
        mutation_scale=5.5 if inhib else 8, lw=lw, color=color, linestyle=ls,
        shrinkA=shrink, shrinkB=shrink, zorder=3,
        connectionstyle=f'arc3,rad={rad}'))


fig = plt.figure(figsize=(9.5, 7.0))
outer = fig.add_gridspec(3, 1, height_ratios=[1.02, 1.0, 1.02], hspace=.58,
                         left=.055, right=.985, top=.93, bottom=.045)
top = outer[0].subgridspec(1, 2, width_ratios=[1.34, 1.0], wspace=.20)
bot = outer[1].subgridspec(1, 3, width_ratios=[.92, .92, 1.04], wspace=.52)

# ── A: the proposal ───────────────────────────────────────────────────────
# Two sources of positional drive, mixed by a global engagement signal. This
# replaces a two-loop account in which cholinergic tone selected between a
# recurrent and a feedforward gamma generator. That version asked the local
# circuit to reconfigure, and the measurements in E say it does not: the
# spike-field architecture is the same in both states. What the data do show
# changing is the BALANCE OF DRIVE -- a rotation of the spectrum and a halving
# of theta's entrainment to running speed -- so the switch is placed on the
# input rather than inside the circuit.
ax = fig.add_subplot(top[0]); _lp(ax, 'A', dx=-.08)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.set_title('what is proposed to switch: which drive the position estimate '
             'follows', fontsize=7.5, loc='left', color='0.25', pad=5)
for x_, y_, w_, h_, nm, c_, tc in (
        (.02, .62, .26, .15, 'self-motion\n(speed, head direction)', '#eef3fa', INT_C),
        (.02, .20, .26, .15, 'landmarks\n(the task frame)', '#fbf0ef', FAST),
        (.44, .41, .24, .17, 'entorhinal\nposition estimate', '#f4f1ec', '0.3')):
    ax.add_patch(FancyBboxPatch((x_, y_), w_, h_, boxstyle='round,pad=.022',
                                fc=c_, ec='0.75', lw=.9, zorder=1))
    ax.text(x_ + w_ / 2, y_ + h_ / 2, nm, fontsize=6.5, ha='center',
            va='center', color=tc, linespacing=1.25)
arrow(ax, (.29, .69), (.43, .55), INT_C, rad=-.18, lw=1.6, shrink=1)
arrow(ax, (.29, .28), (.43, .45), FAST, rad=.18, lw=1.6, shrink=1)
ax.text(.335, .755, 'always on', fontsize=5.8, color=INT_C, ha='center')
ax.text(.345, .175, 'gated', fontsize=5.8, color=FAST, ha='center')
# the gate
ax.add_patch(FancyBboxPatch((.30, .845), .40, .115,
                            boxstyle='round,pad=.022', fc='#fdf6e3',
                            ec=ACH, lw=1.0, zorder=1))
ax.text(.50, .902, 'engagement — pupil-indexed, septal/cholinergic',
        fontsize=6.3, ha='center', va='center', color=ACH)
arrow(ax, (.42, .845), (.37, .34), ACH, rad=.55, lw=1.2, ls=(0, (2.4, 1.6)),
      shrink=2)
# the readout
ax.add_patch(FancyBboxPatch((.76, .41), .22, .17, boxstyle='round,pad=.022',
                            fc='#f7f2f6', ec='0.75', lw=.9, zorder=1))
ax.text(.87, .495, 'grid cells +\nFS interneurons', fontsize=6.5, ha='center',
        va='center', color=GRID_C, linespacing=1.25)
arrow(ax, (.69, .495), (.75, .495), '0.45', lw=1.3, shrink=1)
ax.text(.87, .33, 'read it out most\nstrongly (D)', fontsize=5.8, ha='center',
        va='top', color='0.5', linespacing=1.25)
ax.text(.0, -.055, 'anchored: the estimate is corrected to the frame each lap.  '
                   'non-anchored: it is not,\nand no coherent field survives at '
                   'any offset — so this is a missing correction,\nnot a '
                   'displaced or drifting map',
        transform=ax.transAxes, fontsize=6.3, color='0.45', linespacing=1.4)

# ── B: the measured hinge ─────────────────────────────────────────────────
# Theta's entrainment to running speed halves in the anchored state while the
# speed CELLS are unchanged. That pair is the single strongest reason to put
# the switch on the input: the velocity signal is still being computed and
# still reaches the cells, while the rhythm that carries it into the circuit
# is listening to it less.
ax = fig.add_subplot(top[1]); _lp(ax, 'B', dx=-.20)
ax.set_title('the measured hinge (Figure 7)', fontsize=7.5, loc='left',
             color='0.25', pad=5)
sp = np.linspace(5, 45, 50)
for sl, c_, nm in ((0.374, NONANCH, 'non-anchored'), (0.218, ANCH, 'anchored')):
    ax.plot(sp, sl * (sp - 5) / 10., color=c_, lw=1.8, label=nm)
ax.set_xlabel('running speed (cm/s)', fontsize=7)
ax.set_ylabel('theta amplitude (a.u.)', fontsize=7)
ax.legend(fontsize=6, frameon=False, loc='upper left')
ax.tick_params(labelsize=6.5); ax.spines[['top', 'right']].set_visible(False)
# inside the axes: below the shallower line is empty, and anything placed
# under the panel runs into the row beneath
ax.text(.97, .04, 'slope 0.374 \u2192 0.218, p = 0.045\n'
                  'frequency slope and both correlations fall with it\n\n'
                  'yet the open-field speed cells code speed just as\n'
                  'well when anchored (+11.9%, p = 0.12): the signal\n'
                  'is intact and the rhythm is heeding it less',
        transform=ax.transAxes, fontsize=6.0, color='0.45', ha='right',
        va='bottom', linespacing=1.45)

# ── C: the field potential (measured) ─────────────────────────────────────
ax = fig.add_subplot(bot[0]); _lp(ax, 'C', dx=-.46)
ax.set_title('what the field potential\ndoes (Figure 7)', fontsize=7.5,
             loc='left', color='0.25', pad=5)
val = [+0.105, -0.108, -0.094, -0.159]
lab = ['slow gamma\n30–48 Hz', 'fast gamma\n60–100 Hz', 'theta\n6–10 Hz',
       'theta–fast\ncoupling']
y = np.arange(len(val))[::-1]
ax.barh(y, val, color=[SLOW, FAST, THETA, FAST], alpha=.85, height=.62, lw=0)
ax.axvline(0, color='0.35', lw=.9, zorder=3)
for yy, v in zip(y, val):
    ax.text(v + (.012 if v > 0 else -.012), yy, f'{v:+.3f}', fontsize=6.3,
            va='center', ha='left' if v > 0 else 'right', color='0.25')
ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=6.5)
ax.set_xlim(-.27, .20); ax.set_xticks([-.2, -.1, 0, .1])
ax.set_xlabel('anchored − non-anchored', fontsize=7)
ax.tick_params(labelsize=6.5); ax.spines[['top', 'right']].set_visible(False)

# ── D: the readout (measured) ─────────────────────────────────────────────
ax = fig.add_subplot(bot[1]); _lp(ax, 'D', dx=-.46)
ax.set_title('which cells express it\n(Figure 2)', fontsize=7.5, loc='left',
             color='0.25', pad=5)
# The superseded values were +0.119, +0.080, -0.000 and -0.023, with the
# lower two at chance. Those came from a signed correlation scored against an
# absolute null; corrected, every class exceeds its own chance level and the
# claim is graded rather than exclusive (see Figure 2D).
val = [+0.185, +0.131, +0.081, +0.056]
lab = ['grid', 'putative\ninterneuron', 'non-grid\nspatial', 'non-spatial']
sig = ['1e-11', '3e-16', '6e-22', '4e-16']
y = np.arange(len(val))[::-1]
ax.barh(y, val, color=[GRID_C, INT_C, '0.72', '0.72'], alpha=.85, height=.62,
        lw=0)
ax.axvline(0, color='0.35', lw=.9, zorder=3)
for yy, v, s in zip(y, val, sig):
    ax.text(v + (.006 if v >= 0 else -.006), yy, s, fontsize=6.1,
            va='center', ha='left' if v >= 0 else 'right', color='0.25')
ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=6.5)
ax.set_xlim(0, .245); ax.set_xticks([0, .05, .10, .15, .20])
ax.set_xlabel('agreement above\nthe cell\'s own null', fontsize=7)
ax.tick_params(labelsize=6.5); ax.spines[['top', 'right']].set_visible(False)
ax.text(.0, -.40, 'every class follows it; the loop\'s two populations\n'
        'by about twice as much', transform=ax.transAxes, fontsize=6.3,
        color='0.45', linespacing=1.35)

# ── E: what the state does NOT change (all measured) ──────────────────────
# This panel used to propose a cross-region coherence test that would have
# earned the slow/fast band assignment. It cannot be run: the field potential
# exists only for the entorhinal recordings. What has been run since is a set
# of tests that came back null, and they constrain the model more sharply than
# the proposal did, so the panel now carries them.
ax = fig.add_subplot(bot[2]); _lp(ax, 'E', dx=-.10)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.set_title('what the state does NOT change', fontsize=7.5, loc='left',
             color='0.25', pad=5)
NULLS = [
    ('spike coupling to gamma, either band', 'Holm p \u2265 0.17', 'S11'),
    ('which class is most gamma-coupled', 'grid, not FS', 'S11'),
    ('the laminar theta phase gradient', 'all layers p \u2265 0.06', 'S10'),
    ('where in theta each gamma band peaks', 'p = 0.33, 0.16', 'S10'),
    ('grid phase on non-anchored trials', 'no coherent field', 'Fig 2'),
    ('sharing between connected pairs', 'no more than matched', 'Fig 4'),
]
for i_, (what, how, src) in enumerate(NULLS):
    yy = .96 - i_ * .148
    ax.plot(.03, yy, marker='x', ms=4.5, mew=1.5, color='#8a1c1c')
    ax.text(.10, yy, what, fontsize=6.3, va='center', color='0.25')
    ax.text(.10, yy - .062, f'{how}   ({src})', fontsize=5.4, va='center',
            color='0.55')
ax.text(.0, -.17, 'the field rotates while the spike\u2013field architecture\n'
                  'holds still: a state imposed on this circuit, not a\n'
                  'reorganisation of it',
        transform=ax.transAxes, fontsize=6.3, color='0.45', linespacing=1.35)

# ── F: what it comes to, and what would break it ──────────────────────────
ax = fig.add_subplot(outer[2]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis('off')
ax.add_patch(FancyBboxPatch((.004, -.12), .992, 1.10,
                            boxstyle='round,pad=.012', fc='#faf8f4',
                            ec='0.82', lw=.9, zorder=0))
ax.text(.018, .90, 'The state changes what arrives at this circuit, not what '
                   'the circuit does with it.', fontsize=8.2, weight='bold',
        color='#2a2a2a', va='center')
# lines are broken by hand and the blocks stacked: left to the renderer the
# longest one sets the tight bounding box and the figure is published a third
# wider than its panels need
BLOCKS = [
    (.74, 'Already consistent:',
     'a decoder trained ON non-anchored trials does no better than one trained on anchored\n'
     'trials and tested out of frame (31.9 against 30.4 cm, p = 0.04, 56 sessions) — there is no\n'
     'second map to learn.'),
    (.46, 'Prediction 1:',
     'move the gate and the state follows; move the gamma circuit and it should not. A\n'
     'cholinergic or attentional manipulation should shift the anchored fraction, while a local\n'
     'manipulation that changes gamma power should move C and leave D alone.'),
    (.18, 'Prediction 2:',
     'landmarks should matter only while the gate is open. A first pass is consistent — the beacon\'s\n'
     'effect on anchoring is +0.005 with the pupil constricted against −0.008 when dilated (interaction\n'
     'p = 0.007, 57 sessions) — but neither simple effect is significant alone, so this needs a manipulation.'),
]
for yy, head, body in BLOCKS:
    ax.text(.018, yy, head, fontsize=6.6, weight='bold', color='0.35',
            va='top')
    ax.text(.125, yy, body, fontsize=6.5, color='0.35', va='top',
            linespacing=1.45)

fig.text(.055, .972, 'HYPOTHESIS', fontsize=8, weight='bold', color='#8a1c1c')
fig.text(.163, .972, '— only A is a schematic; B, C, D and E are measured',
         fontsize=8, color='0.45')

plt.savefig(OUT, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'wrote {OUT}')
