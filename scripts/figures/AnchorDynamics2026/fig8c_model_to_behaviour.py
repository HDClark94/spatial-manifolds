"""From two loops to behaviour: one parameter, the whole chain.

The architecture is the v2 two-loop circuit, and this figure asks whether a
SINGLE parameter -- the strength of the return loop -- can carry the whole
phenomenon, from the field potential down to whether the animal finds the
reward zone:

    return loop gain  ->  gamma band balance      (what the LFP shows)
                      ->  pinning of grid phase   (what the maps show)
                      ->  error at the reward zone (what the behaviour shows)

Nothing here is a claim about the real circuit. It is a demonstration that the
architecture is sufficient to produce the observed pattern, which is a weaker
thing than evidence and is labelled as such on the page.

THE ONE KNOB. Loop B gain g sets two things at once, which is the point:

    the slow-gamma share of the simulated LFP, directly, since loop B generates
    the slow band; and

    the restoring force k on grid phase, because the return loop is what
    delivers a track-bound position estimate. Phase obeys an
    Ornstein-Uhlenbeck process, ds = -k s dx + sqrt(2D) dW, which is the form
    the real trial maps required: free diffusion predicts zero residual
    correspondence on non-anchored trials and the data retain +0.041, so the
    non-anchored state is WEAKLY pinned rather than unpinned (k ~ 0.005-0.01
    per cm, a decorrelation length of about one track length).

BEHAVIOUR falls out without further assumption. The reward zone on uncued
trials can only be found by path integration, so the animal's estimate of it is
displaced by the grid phase error s at that point on the track. It stops in the
zone when |s| is smaller than the zone's half-width. Hit rate is therefore the
stationary probability P(|s| < w), and the OU stationary variance is D/k -- so
tightening the return loop improves behaviour through exactly the same parameter
that moves the gamma bands.

Real values marked for comparison are measured, not fitted: r0 +0.381 anchored
and +0.041 non-anchored, and the hit rate running 0.826 to 0.391 across pupil
deciles.

Writes fig8c_model_to_behaviour.pdf
"""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from two_loop_dummy import FS, loop, loop_frequency, spec

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

FIG = os.path.dirname(os.path.abspath(__file__))
OUT = f'{FIG}/fig8c_model_to_behaviour.pdf'

SLOW_C = '#2b6cb0'; FAST_C = '#c04744'; THETA_C = '#7b4173'
ANCH_C = '#1a6b3c'; NON_C = '#8a1c1c'; ACH = '#b8860b'

TL, NB = 200.0, 100          # track length (cm), position bins
D_PHASE = 34.0               # cm^2 per cm, fitted earlier to the map statistics
# Only the RATIO D/k is constrained by the trial maps: phase-error sd is
# sqrt(D/k), so D and k are not separately identifiable from map statistics.
# D is held at the earlier fit and k carries the state, giving sd 7 cm anchored
# (fields visible but jittered) against 65 cm non-anchored (beyond the grid
# spacing, so correspondence is gone).
K_ANCH, K_NON = 0.680, 0.008 # restoring force, per cm
R0_ANCH, R0_NON = 0.381, 0.041
HIT_HI, HIT_LO = 0.826, 0.391
ZONE_W = 15.0                # half-width of the reward zone, cm


def _lp(ax, s, dx=-.16, dy=1.0):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='left')


def phase_trials(k, n_trials, rng, D=D_PHASE, wrap=True):
    """Grid phase error along the track, one row per trial (OU process)."""
    dx = TL / NB
    S = np.empty((n_trials, NB))
    step = np.sqrt(2 * D * dx)
    for t in range(n_trials):
        s = 0.0 if k > .03 else rng.uniform(-TL / 2, TL / 2)
        for i in range(NB):
            sg = ((s + TL / 2) % TL) - TL / 2
            s = s - k * sg * dx + rng.normal(0, step)
            S[t, i] = (((s + TL / 2) % TL) - TL / 2) if wrap else s
    return S


def maps_from_phase(S, rng, spacing=55.0, rate=1.0):
    """A model grid cell's trial x position map, given its phase error."""
    x = np.arange(NB) * (TL / NB)
    lam = rate * (0.5 + 0.5 * np.cos(2 * np.pi * (x[None, :] - S) / spacing))
    return rng.poisson(lam * 12.0).astype(float)


def r0_of(M, tmpl):
    v = []
    t = tmpl - tmpl.mean()
    for m in M:
        a = m - m.mean()
        d = np.sqrt((a ** 2).sum() * (t ** 2).sum())
        if d > 0:
            v.append(float((a * t).sum() / d))
    return float(np.mean(v)) if v else np.nan


if __name__ == '__main__':
    rng = np.random.default_rng(3)
    x = np.arange(NB) * (TL / NB)
    tmpl = 0.5 + 0.5 * np.cos(2 * np.pi * x / 55.0)

    fig = plt.figure(figsize=(10, 5.9))
    outer = fig.add_gridspec(2, 3, width_ratios=[1.22, 1.0, 1.0],
                             height_ratios=[1.0, .95], wspace=.42, hspace=.62,
                             left=.055, right=.985, top=.88, bottom=.085)

    # ---- A: the architecture ------------------------------------------------
    ax = fig.add_subplot(outer[0, 0]); _lp(ax, 'A', dx=-.09)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.set_title('one knob: return-loop gain', fontsize=7.5, loc='left',
                 color='0.25', pad=6)
    ax.add_patch(FancyBboxPatch((.14, .50), .50, .30, boxstyle='round,pad=.03',
                                fc='#f4f1ec', ec='0.8', lw=.9))
    ax.text(.39, .83, 'MEC layer II', fontsize=6.5, color='0.45', ha='center')
    G, I = (.27, .645), (.53, .645)
    ax.add_patch(Circle(G, .062, fc=FAST_C, ec='none', alpha=.85))
    ax.add_patch(Circle(I, .062, fc=SLOW_C, ec='none', alpha=.85))
    ax.text(*G, 'grid', fontsize=6.3, color='w', ha='center', va='center',
            weight='bold')
    ax.text(*I, 'FS', fontsize=6.3, color='w', ha='center', va='center',
            weight='bold')
    ax.add_patch(FancyArrowPatch(G, I, arrowstyle='-|>', mutation_scale=7,
                                 connectionstyle='arc3,rad=-0.45', lw=1.2,
                                 color=FAST_C, shrinkA=7, shrinkB=7))
    ax.add_patch(FancyArrowPatch(I, G, arrowstyle='-[', mutation_scale=5,
                                 connectionstyle='arc3,rad=-0.45', lw=1.2,
                                 color=SLOW_C, shrinkA=7, shrinkB=7))
    ax.text(.39, .435, 'loop A — local, fast gamma', fontsize=6.2,
            color=FAST_C, ha='center')
    # the long return loop
    ax.add_patch(FancyBboxPatch((.70, .50), .27, .30, boxstyle='round,pad=.03',
                                fc='#eef3fa', ec='0.8', lw=.9))
    ax.text(.835, .645, 'hippocampal\nreturn', fontsize=6.3, ha='center',
            va='center', color='0.3', linespacing=1.2)
    ax.add_patch(FancyArrowPatch((.645, .70), (.715, .70), arrowstyle='-|>',
                                 mutation_scale=7, lw=1.2, color='0.5'))
    ax.add_patch(FancyArrowPatch((.715, .58), (.645, .58), arrowstyle='-|>',
                                 mutation_scale=7, lw=1.4, color=SLOW_C))
    ax.text(.835, .435, 'loop B — long, slow gamma', fontsize=6.2, color=SLOW_C,
            ha='center')
    ax.add_patch(FancyArrowPatch((.50, .24), (.50, .375), arrowstyle='-|>',
                                 mutation_scale=7, lw=1.4, color=ACH,
                                 linestyle=(0, (2.2, 1.5))))
    ax.text(.50, .185, 'engagement sets loop B gain $g$', fontsize=6.6,
            color=ACH, ha='center')
    ax.text(.50, .055, '$g$  →  slow/fast balance\n'
                       '$g$  →  pinning $k$  →  map  →  behaviour',
            fontsize=6.5, color='0.3', ha='center', linespacing=1.5)

    # ---- B: the LFP the architecture produces -------------------------------
    ax = fig.add_subplot(outer[0, 1]); _lp(ax, 'B', dx=-.22)
    ax.set_title('what the loops give the LFP', fontsize=7.5, loc='left',
                 color='0.25', pad=6)
    n = int(300 * FS); t = np.arange(n) / FS
    gate_A = np.clip(np.cos(2 * np.pi * 8 * t), 0, None) ** 1.2
    gate_B = np.clip(-np.cos(2 * np.pi * 8 * t), 0, None) ** 1.2
    for gA, gB, col, lab in ((1.22, .80, NON_C, 'low $g$ — non-anchored'),
                             (0.78, 1.30, ANCH_C, 'high $g$ — anchored')):
        dA = rng.normal(0, 1, n) * (0.4 + gate_A) * gA
        dB = rng.normal(0, 1, n) * (0.4 + gate_B) * gB
        lfp = (loop(n, 2.0, 1.0, 4.7, dA, rng) + loop(n, 9.0, 1.0, 3.5, dB, rng)
               + .55 * np.sin(2 * np.pi * 8 * t))
        f, z = spec(lfp)
        ax.plot(f, z, color=col, lw=1.3, label=lab)
    ax.axvspan(30, 48, color=SLOW_C, alpha=.10, lw=0)
    ax.axvspan(60, 100, color=FAST_C, alpha=.10, lw=0)
    ax.text(39, ax.get_ylim()[1] * .92, 'slow', fontsize=6, color=SLOW_C, ha='center')
    ax.text(80, ax.get_ylim()[1] * .92, 'fast', fontsize=6, color=FAST_C, ha='center')
    ax.set_xlim(2, 130); ax.set_xlabel('Frequency (Hz)', fontsize=8)
    ax.set_ylabel('whitened power (z)', fontsize=8)
    ax.legend(fontsize=5.8, frameon=False, loc='lower left')
    ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

    # ---- C: grid phase across trials ---------------------------------------
    ax = fig.add_subplot(outer[0, 2]); _lp(ax, 'C', dx=-.22)
    ax.set_title('grid phase along the track', fontsize=7.5, loc='left',
                 color='0.25', pad=6)
    for k, col, lab in ((K_ANCH, ANCH_C, 'anchored'), (K_NON, NON_C, 'non-anchored')):
        S = phase_trials(k, 6, rng, wrap=False)
        for r_ in S:
            ax.plot(x, r_, color=col, lw=.9, alpha=.75)
    ax.axhline(0, color='0.4', lw=.8, ls=':')
    ax.set_xlabel('Position on track (cm)', fontsize=8)
    ax.set_ylabel('phase error (cm)', fontsize=8)
    ax.set_xlim(0, TL); ax.set_ylim(-100, 100)
    ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)
    ax.text(.03, .04, 'high $g$ holds it; low $g$ lets it wander',
            transform=ax.transAxes, fontsize=6.2, color='0.45')

    # ---- D, E: the maps that result ----------------------------------------
    for j, (k, col, lab) in enumerate(((K_ANCH, ANCH_C, 'anchored'),
                                       (K_NON, NON_C, 'non-anchored'))):
        ax = fig.add_subplot(outer[1, j]); _lp(ax, 'DE'[j], dx=-.22)
        S = phase_trials(k, 45, rng)
        M = maps_from_phase(S, rng)
        ax.imshow(M, aspect='auto', cmap='plasma', interpolation='nearest',
                  extent=[0, TL, len(M) - .5, -.5])
        ax.set_title(f'{lab}   (model $r_0$ = {r0_of(M, tmpl):+.2f}, '
                     f'real {R0_ANCH if j == 0 else R0_NON:+.2f})',
                     fontsize=7, loc='left', color=col, pad=4)
        ax.set_xlabel('Position on track (cm)', fontsize=8)
        if j == 0:
            ax.set_ylabel('Trial', fontsize=8)
        ax.tick_params(labelsize=7)
        for sp in ax.spines.values():
            sp.set_visible(False)

    # ---- F: and the behaviour ----------------------------------------------
    ax = fig.add_subplot(outer[1, 2]); _lp(ax, 'F', dx=-.22)
    ax.set_title('reward-zone localisation', fontsize=7.5, loc='left',
                 color='0.25', pad=6)
    ks = np.logspace(np.log10(0.004), np.log10(1.0), 60)
    # OU stationary sd = sqrt(D/k); the animal stops in the zone when |s| < w
    from scipy.stats import norm
    hit = np.array([2 * norm.cdf(ZONE_W / np.sqrt(D_PHASE / k)) - 1 for k in ks])
    ax.plot(ks, hit, color='0.2', lw=1.6, zorder=3)
    for k, col, lab in ((K_ANCH, ANCH_C, 'anchored'), (K_NON, NON_C, 'non-anchored')):
        h = 2 * norm.cdf(ZONE_W / np.sqrt(D_PHASE / k)) - 1
        ax.plot([k], [h], 'o', ms=6, color=col, zorder=4)
        ax.annotate(lab, (k, h), textcoords='offset points',
                    xytext=(7, -10 if k < .02 else 2), fontsize=6.4, color=col)
    ax.axhspan(HIT_LO, HIT_HI, color='0.85', alpha=.55, lw=0, zorder=0)
    ax.text(.95, (HIT_LO + HIT_HI) / 2, 'observed range\nacross pupil deciles',
            fontsize=6.0, color='0.4', ha='right', va='center', linespacing=1.3)
    ax.text(.5, .035, 'the model swings wider than the animals do:\n'
            'phase error is not their only source of error',
            transform=ax.transAxes, fontsize=5.9, color='0.45', ha='center',
            linespacing=1.35)
    ax.set_xlabel('pinning $k$ (per cm), set by $g$', fontsize=8)
    ax.set_ylabel('P(stop in reward zone)', fontsize=8)
    ax.set_ylim(0, 1); ax.set_xscale('log'); ax.set_xlim(.004, 1.0)
    ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

    fig.text(.055, .955, 'ARCHITECTURE DEMONSTRATION', fontsize=8,
             weight='bold', color='#8a1c1c')
    fig.text(.305, .955, '— simulation only; no panel here is data. Real values '
             'are marked for scale', fontsize=8, color='0.45')
    plt.savefig(OUT, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f'wrote {OUT}')
    for k, nm in ((K_ANCH, 'anchored'), (K_NON, 'non-anchored')):
        sd = np.sqrt(D_PHASE / k)
        from scipy.stats import norm as _n
        print(f'  {nm:13s} k={k:.3f}  phase sd {sd:5.1f} cm  '
              f'P(hit) {2*_n.cdf(ZONE_W/sd)-1:.3f}')
