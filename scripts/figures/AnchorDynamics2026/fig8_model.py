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
# Bands. Figure 7 draws slow gamma in the interneuron blue and fast gamma in
# the grid-cell red; here those two colours are already spoken for by the two
# populations inside the estimate, so the bands take green and orange.
SLOW, FAST, THETA = '#2f7d32', '#e8701a', '#7b4173'
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
GATE = (.270, .742, .085, .070)
LAND = (.025, .742, .175, .070)
SELF = (.025, .620, .175, .070)
EST = (.460, .598, .272, .228)
BEH = (.795, .668, .180, .085)

# panel letters: A is the circuit, B-E the four things the two rows show
for _lx, _ly, _ll in ((.004, .826, 'A'), (.070, .492, 'B'),
                      (.448, .492, 'C'), (.676, .492, 'D'),
                      (.822, .492, 'E')):
    ax.text(_lx, _ly, _ll, fontsize=10, weight='bold', va='bottom',
            ha='left', color='0.1')

box(ax, LAND, 'landmarks\n(the task frame)', '#fbf0ef', LAND_C)
box(ax, SELF, 'self-motion\n(speed, head direction)', '#eef3fa', SELF_C)
box(ax, GATE, 'engagement\ngate', '#fdf6e3', ACH, fs=6.6, ec=ACH, lw=1.3)

# the two drives
arrow(ax, (.202, .777), (.267, .777), LAND_C, lw=1.6, shrink=1, head=False)
arrow(ax, (.358, .772), (.457, .755), LAND_C, lw=1.6, shrink=1)
arrow(ax, (.202, .655), (.457, .658), SELF_C, lw=1.6, rad=.06, shrink=1)
ax.text(.234, .792, 'gated', fontsize=5.8, color=LAND_C, ha='center')
ax.text(.312, .712, 'OPEN \u2192 anchored\nSHUT \u2192 non-anchored', fontsize=5.8,
        ha='center', va='center', color=ACH, linespacing=1.35)
ax.text(.240, .672, 'always on', fontsize=5.8, color=SELF_C, ha='center')
ax.text(.300, .636, 'theta\u2013speed slope 0.374 \u2192 0.218', fontsize=5.4,
        color='0.5', ha='center', va='top')

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

# the readout
box(ax, BEH, 'action\n(where the animal\nslows and stops)', '#f2f4f1', '0.3')
arrow(ax, (.734, .710), (.792, .710), '0.45', lw=1.4, shrink=1)

# down into the simulated code
arrow(ax, (.575, .595), (.575, .512), '0.6', lw=1.1, shrink=1)
ax.text(.590, .553, 'simulated', fontsize=5.9, ha='left', va='center',
        color='0.5')

# ── two states, one row each: the rhythm, the code, and the behaviour ────
# The left column is the frequency-specific result drawn as a mechanism, in
# the same row as the code it is a claim about. Theta phases are MEASURED
# (Figure S10, 28 sessions): slow gamma 30-48 Hz peaks near 95 deg of theta
# and fast gamma 60-100 Hz near 42 deg, 0 deg being the theta peak. They are
# drawn at the SAME phase in both rows because the state does not move them
# (paired, 6.6 deg and 2.8 deg, p = 0.33 and 0.16). What the state does move
# is amplitude -- slow up, fast down -- and how tightly the fast burst is
# confined to its phase, which is the 15.9% loss of theta-fast coupling.
PH_SLOW, PH_FAST = 95., 42.
_p = np.linspace(-180., 540., 1400)
OFF_T, OFF_S, OFF_F = 4.6, 1.5, -2.2

COLS = [(.082, .330), (.460, .062), (.536, .062), (.612, .062),
        (.692, .076), (.838, .095)]
ROWS = [(.322, .138, True, 'ANCHORED', ANCH),
        (.112, .138, False, 'NON-ANCHORED', NONANCH)]
HEAD = ['theta-nested gamma', 'grid cell 1', 'grid cell 2', 'grid cell 3',
        'decoded position (cm)', 'stops, uncued']
CELLS = [(44., 10.), (62., 30.), (86., 55.)]   # spacing cm, anchored phase
# theta, slow, fast amplitude and the fast burst's concentration, per state
LFP = {True: (.95, 1.55, .55, 2.6), False: (1.05, .55, 1.55, 8.0)}

for ri, (ry, rh, anch, rlab, rcol) in enumerate(ROWS):
    ax.text(.014, ry + rh / 2, rlab, fontsize=7.4, weight='bold', color=rcol,
            ha='center', va='center', rotation=90)
    for ci, (cx, cw) in enumerate(COLS):
        a_ = fig.add_axes([cx, ry, cw, rh])
        a_.tick_params(labelsize=5.8, length=2, pad=1.5)
        for sp in ('top', 'right'):
            a_.spines[sp].set_visible(False)
        if ci == 0:
            _at, _as, _af, _kf = LFP[anch]
            a_.spines['left'].set_visible(False)
            for _ph, _col in ((PH_SLOW, SLOW), (PH_FAST, FAST)):
                for _k in (0, 360):
                    a_.axvline(_ph + _k, color=_col, lw=.7, ls=(0, (2.0, 1.8)),
                               alpha=.6, zorder=0)
            a_.plot(_p, OFF_T + _at * np.cos(np.radians(_p)), color=THETA,
                    lw=1.4)
            for _off, _amp, _ph, _k, _n, _col in (
                    (OFF_S, _as, PH_SLOW, 3.4, 5., SLOW),
                    (OFF_F, _af, PH_FAST, _kf, 10., FAST)):
                _e = _amp * np.exp(_k * (np.cos(np.radians(_p - _ph)) - 1.))
                a_.fill_between(_p, _off - _e, _off + _e, color=_col,
                                alpha=.22, linewidth=0, edgecolor='none')
                a_.plot(_p, _off + _e * np.sin(np.radians(_p) * _n),
                        color=_col, lw=.8)
            a_.set_xlim(-180, 540); a_.set_ylim(-4.2, 6.4)
            a_.set_xticks([-180, 0, 180, 360, 540])
            a_.set_yticks([OFF_T, OFF_S, OFF_F])
            a_.set_yticklabels(['theta', 'slow \u03b3', 'fast \u03b3'],
                               fontsize=6.0)
            for _t, _col in zip(a_.get_yticklabels(), (THETA, SLOW, FAST)):
                _t.set_color(_col)
            a_.tick_params(axis='y', length=0)
            if ri == 0:   # the measured phases, marked once
                for _ph, _col in ((PH_FAST, FAST), (PH_SLOW, SLOW)):
                    a_.text(_ph + 360, 6.3, f'{_ph:.0f}\u00b0', fontsize=5.4,
                            color=_col, ha='center', va='top')
            else:
                a_.set_xlabel('theta phase (deg, 0 = peak)', fontsize=6.4,
                              labelpad=1)
            if ri == 0:
                a_.set_xticklabels([])
            a_.set_title(HEAD[ci], fontsize=6.6, color='0.3', pad=3) \
                if ri == 0 else None
            continue
        if ci < 4:
            lam, ph = CELLS[ci - 1]
            m = grid_map(lam, ph, anch)
            a_.imshow(m / m.max(), aspect='auto', cmap=CMAP, origin='upper',
                      extent=[0, TRACK, NTRIAL + .5, .5],
                      interpolation='nearest', vmin=0, vmax=1)
        elif ci == 4:
            x, p_ = decoded_position(anch)
            im = a_.imshow(p_, aspect='auto', cmap='jet', origin='upper',
                           extent=[0, TRACK, NTRIAL + .5, .5],
                           interpolation='nearest', vmin=0, vmax=TRACK)
            if ri == 0:
                cb = fig.add_axes([cx + cw + .006, ry, .009, rh])
                plt.colorbar(im, cax=cb, ticks=[0, 100, 200])
                cb.tick_params(labelsize=5.4, length=2, pad=1.5)
        else:
            a_.axvspan(*RZ, color='#e8e2d6', linewidth=0, zorder=0)
            for t, v in enumerate(stops(anch)):
                a_.plot(v, np.full(len(v), t + 1), ls='none', marker='|',
                        ms=3.8, mew=.9, color=rcol)
            a_.set_ylim(NTRIAL + .62, -.30)
            if ri == 0:
                a_.text(np.mean(RZ), .02, 'reward zone', fontsize=5.0,
                        ha='center', va='center', color='0.45')
        a_.set_yticks([1, 5, 10])
        a_.set_xlim(0, TRACK); a_.set_xticks([0, 100, 200])
        if ri == 0:
            a_.set_title(HEAD[ci], fontsize=6.6, color='0.3', pad=3)
            a_.set_xticklabels([])
        else:
            a_.set_xlabel('position (cm)', fontsize=6.4, labelpad=1)
        if ci == 1:
            a_.set_ylabel('trial', fontsize=6.4, labelpad=1)
        else:
            a_.set_yticklabels([])

plt.savefig(OUT, dpi=300, bbox_inches=Bbox.from_extents(0, .28, 9.5, 5.08))
plt.close(fig)
print(f'wrote {OUT}')
