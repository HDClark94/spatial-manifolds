"""Figure 8: one circuit, one gate, and what each setting of the gate produces.

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
5: pupil falls 0.41 z on entry to the anchored state and rises 0.49 z on exit,
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
     backwards for this task -- see above and Figure 5.

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
zero at a median of 67 Hz in 25 of 26 sessions (Figure 7).
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
SLOW, FAST, THETA = '#2b6cb0', '#c04744', '#7b4173'   # bands, as in Figure 7
CMAP = 'viridis'       # rate maps are viridis throughout the paper

TRACK = 200.           # cm
NTRIAL = 10
rng = np.random.default_rng(8)
RZ = np.array([90., 110.])    # reward zone, as in Figure 1


# ── the simulated code under each setting of the gate ─────────────────────
def grid_map(lam, phase0, anchored, k=3.4):
    """A periodic 1D grid cell over ten traversals, re-referenced or not."""
    x = np.linspace(0, TRACK, 120)
    m = np.empty((NTRIAL, x.size))
    for t in range(NTRIAL):
        # anchored: the landmark drive restores the same phase every lap.
        # non-anchored: nothing does, so the phase is drawn afresh.
        ph = phase0 if anchored else rng.uniform(0, lam)
        r = np.exp(k * (np.cos(2 * np.pi * (x - ph) / lam) - 1.))
        m[t] = r * (1. + .22 * rng.standard_normal(x.size))
    return np.clip(m, 0, None)


def decoded_position(anchored):
    """Position decoded from an attractor fed by those cells, over ten laps.

    Returned as the decoded position itself, in cm and wrapped onto the 200 cm
    track, so the panel is a map of what the animal's code says about where it
    is rather than of how wrong it is.
    """
    x = np.linspace(0, TRACK, 160)
    p = np.empty((NTRIAL, x.size))
    for t in range(NTRIAL):
        if anchored:
            # re-referenced at the start of the lap and held there
            w = 2.0 * rng.standard_normal(x.size).cumsum() / 9.
            err = np.clip(w - w.mean(), -7, 7)
        else:
            # starts wherever the attractor happens to be and integrates away
            err = (rng.uniform(-80, 80)
                   + rng.normal(0, 48) * (x / TRACK)
                   + 3.2 * rng.standard_normal(x.size).cumsum() / 7.)
        p[t] = np.mod(x + err, TRACK)
    return x, p


def stops(anchored):
    """Where the animal stops on uncued traversals, trial by trial.

    The prediction the account makes about behaviour, not a measurement. With
    the gate open the decode is on the track, so stopping concentrates at the
    reward zone the animal cannot see; with it shut the decode is off the
    track by an arbitrary amount, so stopping is scattered and only
    occasionally lands in the zone by chance.
    """
    out = []
    for t in range(NTRIAL):
        if anchored:
            v = list(rng.normal(RZ[0] + 6., 6.0, rng.integers(2, 4)))
            if rng.random() < .35:                      # the occasional stray
                v.append(rng.uniform(35, 165))
        else:
            v = list(rng.uniform(32, 168, rng.integers(3, 6)))
            if rng.random() < .30:                      # chance hits
                v.append(rng.normal(RZ.mean(), 9.))
        out.append(np.clip(v, 30, 170))
    return out


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


fig = plt.figure(figsize=(9.5, 6.0))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# ── the circuit, drawn once ───────────────────────────────────────────────
ENG = (.205, .848, .370, .105)
GATE = (.270, .742, .085, .070)
LAND = (.025, .742, .175, .070)
SELF = (.025, .620, .175, .070)
EST = (.460, .598, .272, .228)
BEH = (.795, .668, .180, .085)

box(ax, ENG, '', '#fdf6e3', ACH, ec=ACH, lw=1.0)
ax.text(.390, .928, 'engagement — attentional focus  (septal / cholinergic)',
        fontsize=6.8, ha='center', va='center', color=ACH)
ax.text(.390, .903, 'indexed by a CONSTRICTED pupil, not a dilated one',
        fontsize=6.2, ha='center', va='center', color=ACH)
ax.text(.390, .875, '\u22120.41 z on entry to the anchored state and +0.49 z on '
                    'exit, and the hit rate\nfalls from 0.83 to 0.39 as the '
                    'pupil widens (Figure 5)', fontsize=5.5, ha='center',
        va='center', color='0.5', linespacing=1.35)

box(ax, LAND, 'landmarks\n(the task frame)', '#fbf0ef', LAND_C)
box(ax, SELF, 'self-motion\n(speed, head direction)', '#eef3fa', SELF_C)
box(ax, GATE, 'GATE', '#fdf6e3', ACH, fs=7.0, ec=ACH, lw=1.3)

# engagement sets the gate
arrow(ax, (.312, .836), (.312, .816), ACH, lw=1.2, ls=(0, (2.4, 1.6)),
      shrink=1, z=2)
# the two drives
arrow(ax, (.202, .777), (.267, .777), LAND_C, lw=1.6, shrink=1, head=False)
arrow(ax, (.358, .772), (.457, .755), LAND_C, lw=1.6, shrink=1)
arrow(ax, (.202, .655), (.457, .658), SELF_C, lw=1.6, rad=.06, shrink=1)
ax.text(.234, .792, 'gated', fontsize=5.8, color=LAND_C, ha='center')
ax.text(.312, .712, 'OPEN \u2192 anchored\nSHUT \u2192 non-anchored', fontsize=5.8,
        ha='center', va='center', color=ACH, linespacing=1.35)
ax.text(.240, .672, 'always on', fontsize=5.8, color=SELF_C, ha='center')
ax.text(.330, .603, 'theta\u2013speed slope 0.374 \u2192 0.218 when the gate opens, '
                    'while the\nspeed CELLS do not change (+11.9%, p = 0.12; '
                    'Figure 7)',
        fontsize=5.4, color='0.5', ha='center', va='top', linespacing=1.35)

# the estimate, with the cells that carry it drawn inside it
box(ax, EST, '', '#f7f5f1', '0.3', ec='0.68', lw=1.0)
ax.text(.596, .804, 'entorhinal position estimate', fontsize=7.0,
        ha='center', va='center', color='0.25', zorder=4)
ax.text(.596, .782, 'a continuous attractor', fontsize=5.8,
        ha='center', va='center', color='0.55', zorder=4)
box(ax, (.480, .638, .112, .105), 'grid\ncells', '#fbf0ef', GRID_C, fs=6.4,
    z=4)
box(ax, (.604, .638, .112, .105), 'FS\ninterneurons', '#eef3fa', INT_C,
    fs=6.4, z=4)
arrow(ax, (.594, .712), (.602, .712), '0.5', lw=1.0, shrink=1, z=6, ms=6)
arrow(ax, (.602, .668), (.594, .668), '0.5', lw=1.0, shrink=1, z=6, ms=6)
ax.text(.596, .614, 'the two populations that follow the state most strongly '
                    '(+0.185, +0.131; Figure 2)', fontsize=5.3, ha='center',
        va='center', color='0.5', zorder=5)

# the readout
box(ax, BEH, 'action\n(where the animal\nslows and stops)', '#f2f4f1', '0.3')
arrow(ax, (.734, .710), (.792, .710), '0.45', lw=1.4, shrink=1)

# down into the simulated code
arrow(ax, (.575, .595), (.575, .512), '0.6', lw=1.1, shrink=1)
ax.text(.588, .553, 'what the code looks like under each\nsetting of the gate '
                    '(simulated)', fontsize=5.9, ha='left', va='center',
        color='0.5', linespacing=1.35)

# ── what the field potential does when the gate opens (measured) ─────────
# The frequency-specific part of the story belongs in the schematic, because
# it is what tells you the change is a rotation of the input balance rather
# than a gain change: the difference between states reverses sign across the
# spectrum instead of scaling it.
ax.text(.752, .656, 'and what the field potential does when the gate opens\n'
                    '(measured, Figure 7; anchored \u2212 non-anchored)',
        fontsize=6.0, ha='left', va='top', color='0.3', linespacing=1.4)

_fb = fig.add_axes([.840, .505, .096, .094])
_val = [+0.105, -0.108, -0.094, -0.159]
_lab = ['slow gamma 30\u201348 Hz', 'fast gamma 60\u2013100 Hz', 'theta 6\u201310 Hz',
        'theta\u2013fast coupling']
_y = np.arange(4)[::-1]
_fb.barh(_y, _val, color=[SLOW, FAST, THETA, FAST], alpha=.85, height=.62,
         lw=0)
_fb.axvline(0, color='0.35', lw=.8, zorder=3)
for _yy, _v in zip(_y, _val):
    _fb.text(_v + (.010 if _v > 0 else -.010), _yy, f'{_v:+.3f}', fontsize=5.0,
             va='center', ha='left' if _v > 0 else 'right', color='0.3')
_fb.set_yticks(_y); _fb.set_yticklabels(_lab, fontsize=5.3)
_fb.set_xlim(-.30, .22); _fb.set_xticks([-.2, 0, .2])
_fb.tick_params(labelsize=5.2, length=2, pad=1.2)
_fb.spines[['top', 'right']].set_visible(False)

ax.text(.752, .485, 'The difference REVERSES SIGN across the spectrum rather\n'
                    'than scaling it: it crosses zero at a median of 67 Hz, in\n'
                    '25 of 26 sessions. The spectrum rotates.',
        fontsize=5.5, ha='left', va='top', color='0.5', linespacing=1.45)

# ── the left-hand reading of the two settings ─────────────────────────────
ax.text(.025, .556, 'GATE OPEN  —  ANCHORED', fontsize=7.2, weight='bold',
        color=ANCH, va='top')
ax.text(.025, .530, 'The landmark drive reaches the estimate, which is '
                    're-referenced to\nthe task frame on every traversal. Grid '
                    'fields recur at the same\nplaces lap after lap, the '
                    'attractor\u2019s decoded position stays on the\ntrack, and '
                    'the animal stops where the reward is.',
        fontsize=6.5, color='0.35', va='top', linespacing=1.5)

ax.text(.025, .404, 'GATE SHUT  —  NON-ANCHORED', fontsize=7.2, weight='bold',
        color=NONANCH, va='top')
ax.text(.025, .378, 'Only self-motion arrives. Nothing re-references the '
                    'estimate to the\ntrack, so each traversal starts from '
                    'wherever the attractor\nhappens to be: the grid phase is '
                    'drawn afresh lap by lap and the\ndecoded position '
                    'integrates away within the lap.',
        fontsize=6.5, color='0.35', va='top', linespacing=1.5)
ax.text(.025, .252, 'The correction is MISSING rather than displaced. '
                    'Non-anchored\ntrials show no coherent field at any offset '
                    'once the unrelated-cell\nfloor is applied, and '
                    'neighbouring non-anchored traversals do not\nalign with '
                    'one another \u2014 so this is neither a drifting map nor a\n'
                    're-referenced one, which is why the phase is drawn afresh '
                    'below\nrather than progressively shifted.',
        fontsize=6.2, color='0.5', va='top', linespacing=1.5)

ax.text(.025, .128, 'It is the anchored trials on which the animal localises '
                    'the reward zone\ncorrectly, and most strongly when it '
                    'must do so by path integration\n(Figure 1) \u2014 which is '
                    'what puts behaviour downstream of the gate\nrather than '
                    'beside it.',
        fontsize=6.2, color='0.5', va='top', linespacing=1.5)

# ── the simulated code: 2 states x (3 rate maps + 1 decoded position) ────
COLS = [(.418, .088), (.518, .088), (.618, .088), (.718, .104),
        (.872, .104)]
ROWS = [(.285, .135, True, 'ANCHORED', ANCH),
        (.090, .135, False, 'NON-ANCHORED', NONANCH)]
HEAD = ['grid cell 1', 'grid cell 2', 'grid cell 3', 'decoded position (cm)',
        'stops, uncued trials']
CELLS = [(44., 10.), (62., 30.), (86., 55.)]   # spacing cm, anchored phase

for ri, (ry, rh, anch, rlab, rcol) in enumerate(ROWS):
    ax.text(.386, ry + rh / 2, rlab, fontsize=6.6, weight='bold', color=rcol,
            ha='center', va='center', rotation=90)
    for ci, (cx, cw) in enumerate(COLS):
        a = fig.add_axes([cx, ry, cw, rh])
        a.tick_params(labelsize=5.6, length=2, pad=1.5)
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
        if ci < 3:
            lam, ph = CELLS[ci]
            m = grid_map(lam, ph, anch)
            a.imshow(m / m.max(), aspect='auto', cmap=CMAP, origin='upper',
                     extent=[0, TRACK, NTRIAL + .5, .5],
                     interpolation='nearest', vmin=0, vmax=1)
        elif ci == 3:
            x, p = decoded_position(anch)
            im = a.imshow(p, aspect='auto', cmap='jet', origin='upper',
                          extent=[0, TRACK, NTRIAL + .5, .5],
                          interpolation='nearest', vmin=0, vmax=TRACK)
            if ri == 0:   # one colourbar; both rows share the scale
                cb = fig.add_axes([cx + cw + .006, ry, .009, rh])
                plt.colorbar(im, cax=cb, ticks=[0, 100, 200])
                cb.tick_params(labelsize=5.3, length=2, pad=1.5)
        else:
            a.axvspan(*RZ, color='#e8e2d6', linewidth=0, zorder=0)
            for t, v in enumerate(stops(anch)):
                a.plot(v, np.full(len(v), t + 1), ls='none', marker='|',
                       ms=3.6, mew=.9, color=rcol)
            a.set_ylim(NTRIAL + .62, -.30)   # headroom for the zone label
            if ri == 0:
                a.text(np.mean(RZ), .02, 'reward zone', fontsize=4.9,
                       ha='center', va='center', color='0.45')
        a.set_yticks([1, 5, 10])
        a.set_xlim(0, TRACK); a.set_xticks([0, 100, 200])
        if ri == 0:
            a.set_title(HEAD[ci], fontsize=6.2, color='0.3', pad=3)
            a.set_xticklabels([])
        else:
            a.set_xlabel('position (cm)', fontsize=6.2, labelpad=1)
        if ci == 0:
            a.set_ylabel('trial', fontsize=6.2, labelpad=1)
        else:
            a.set_yticklabels([])

fig.text(.025, .032, 'The state changes what arrives at this circuit, not what '
                     'the circuit does with it \u2014 and what arrives decides '
                     'whether the code is on the track.', fontsize=8.2,
         weight='bold', color='#2a2a2a', va='center')

fig.text(.025, .995, 'HYPOTHESIS', fontsize=8, weight='bold', color='#8a1c1c',
         va='top')
fig.text(.133, .995, '\u2014 the circuit is a schematic and the ten panels below '
                     'it are simulated; every measured value names the figure '
                     'it comes from', fontsize=8, color='0.45', va='top')

plt.savefig(OUT, dpi=300)
plt.close(fig)
print(f'wrote {OUT}')
