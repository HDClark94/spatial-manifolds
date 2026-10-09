"""Figure 7 supplement — where in the theta cycle each frequency peaks.

The band analysis in Figure 7 fixes two gamma bands in advance and asks how
strongly each is nested in theta. It cannot ask WHERE in the cycle they peak,
because the modulation index discards phase. This does, as a continuous
function of frequency, for both anchoring states.

    A, B  the phase-frequency map for each state, over 28 sessions
    C     their difference, which is where the state effect lives
    D     the same information reduced to preferred phase against frequency

WHAT IT SHOWS. The relationship is a GRADIENT, not a step between two bands:
preferred phase runs smoothly from about 130 deg of theta at 20-30 Hz to about
41 deg at 60-100 Hz. The whole gradient sits roughly 20 deg later in the cycle
in the non-anchored state, and the slow-minus-fast separation is itself
state-dependent -- +34 deg when anchored against +52 deg when not. None of that
is visible to a two-band test.

EACH FREQUENCY ROW IS NORMALISED TO SUM TO ONE, so colour is where a frequency
peaks and not how much power it carries. Without that the 1/f background would
dominate every row and the panels would redraw the spectrum.

MAINS DOES NOT DRIVE THESE PANELS, and the normalisation is why: a strong line
adds no colour once its row is normalised. Measured, the 48-52 Hz band is if
anything LESS modulated than its neighbours (0.0081 against 0.0102), which is
what a line uncorrelated with theta does -- it dilutes the coupling rather than
creating it. The band is marked on the panels so the reader can see which row
it is.

COLOUR LIMITS ARE THE 1ST AND 99TH PERCENTILES of the two maps, symmetric about
uniform. The maps deviate from uniform by only about 3%, so a scale chosen by
eye is likely to be far too wide: an earlier version ran +/-14% and the
structure filled a fifth of the colour range.

Computed by theta_phase_comodulogram.py. Writes fig7_supp_comodulogram.pdf
"""
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.bf'] = 'Arial:bold'
plt.rcParams['pdf.fonttype'] = 42

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = os.path.dirname(os.path.abspath(__file__))
OUT = f'{FIG}/fig7_supp_comodulogram.pdf'
ANCH_COLOR, NONANCH_COLOR = '#a8559e', '#3f9b8f'


def lp(ax, s, x=-.20, y=1.0):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='right')


z = np.load(f'{ROOT}/data/lfp/theta_phase_comodulogram.npz')
A, N, f = z['anch'], z['non'], z['freqs']
NP = int(z['n_phase'])
cent = np.linspace(-np.pi, np.pi, NP + 1)[:-1] + np.pi / NP
deg = np.degrees(cent)
NS = A.shape[0]


def pref(M):
    """Preferred phase per frequency, on [0, 360).

    The standard [-180, 180) branch cut falls straight through these values,
    which lie between about +40 and +180 deg, and put a full-axis jump at the
    lowest frequency.
    """
    return np.degrees(np.angle((M * np.exp(1j * cent)[None, :]).sum(1))) % 360.


mA, mN = A.mean(0) * NP, N.mean(0) * NP
both = np.concatenate([mA, mN])
vmin, vmax = np.percentile(both, 1), np.percentile(both, 99)
hw = max(1 - vmin, vmax - 1)
vmin, vmax = 1 - hw, 1 + hw
print(f'{NS} sessions, {len(f)} frequencies, {NP} phase bins')
print(f'colour limits {vmin:.3f} - {vmax:.3f}')

fig = plt.figure(figsize=(9.6, 3.1))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.18], wspace=.60,
                      left=.06, right=.975, top=.80, bottom=.20)

for k, (M, tag, col) in enumerate(((mA, 'anchored', ANCH_COLOR),
                                   (mN, 'non-anchored', NONANCH_COLOR))):
    ax = fig.add_subplot(gs[k])
    im = ax.pcolormesh(deg, f, M, cmap='magma', vmin=vmin, vmax=vmax,
                       shading='nearest', rasterized=True)
    ax.axhspan(48, 52, facecolor='none', edgecolor='w', lw=.5, ls=(0, (2, 2)))
    ax.text(178, 50, 'mains', fontsize=5.2, color='w', ha='right', va='center')
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xlabel('theta phase (deg)', fontsize=7.5)
    if k == 0:
        ax.set_ylabel('frequency (Hz)', fontsize=7.5)
    else:
        ax.tick_params(labelleft=False)
    ax.set_title(tag, fontsize=8, color=col)
    ax.tick_params(labelsize=6.5)
    lp(ax, 'AB'[k])
cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.03)
cb.ax.set_ylabel('amplitude (1 = uniform)', fontsize=5.4, labelpad=2)
cb.ax.tick_params(labelsize=5.4); cb.outline.set_visible(False)

# ---- C: the difference, which is where the state effect is ------------------
ax = fig.add_subplot(gs[2])
D = mA - mN
lim = np.percentile(np.abs(D), 99)
imd = ax.pcolormesh(deg, f, D, cmap='RdBu_r', vmin=-lim, vmax=lim,
                    shading='nearest', rasterized=True)
ax.set_xticks([-180, -90, 0, 90, 180])
ax.set_xlabel('theta phase (deg)', fontsize=7.5)
ax.tick_params(labelsize=6.5)
ax.set_title('anchored − non-anchored', fontsize=8, color='0.25')
# the state effect here is a phase SHIFT, which shows as a dipole rather than a
# blob: anchored higher on the earlier flank, lower on the later one, strongest
# where the amplitude itself is (50-75 Hz). It is a small difference on a noisy
# map and the caption says so.
ax.text(.5, -.30, 'a shift, not a gain: red earlier, blue later',
        transform=ax.transAxes, fontsize=5.8, color='0.45', ha='center')
lp(ax, 'C')
cbd = fig.colorbar(imd, ax=ax, fraction=.045, pad=.03)
cbd.ax.set_ylabel('difference', fontsize=5.4, labelpad=2)
cbd.ax.tick_params(labelsize=5.4); cbd.outline.set_visible(False)

# ---- D: preferred phase against frequency -----------------------------------
ax = fig.add_subplot(gs[3])
for M, c, lab in ((A, ANCH_COLOR, 'anchored'), (N, NONANCH_COLOR, 'non-anchored')):
    per = np.array([pref(M[i]) for i in range(M.shape[0])])
    mu = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(per)), 0))) % 360.
    se = np.degrees(np.std(np.radians(per), 0) / np.sqrt(len(per)))
    ax.fill_between(f, mu - se, mu + se, color=c, alpha=.28, lw=0)
    ax.plot(f, mu, color=c, lw=1.5, label=lab)
ax.axvspan(30, 48, color='0.88', alpha=.6, lw=0, zorder=0)
ax.axvspan(60, 100, color='0.88', alpha=.6, lw=0, zorder=0)
ax.text(39, 196, 'slow', fontsize=6, color='0.45', ha='center')
ax.text(80, 196, 'fast', fontsize=6, color='0.45', ha='center')
ax.set_xlabel('frequency (Hz)', fontsize=7.5)
ax.set_ylabel('preferred theta phase (deg)', fontsize=7.5)
ax.set_title('a gradient, not two bands', fontsize=8, color='0.25')
ax.set_ylim(0, 210)
ax.legend(fontsize=6, frameon=False, loc='lower left')
ax.tick_params(labelsize=6.5)
ax.spines[['top', 'right']].set_visible(False)
lp(ax, 'D', x=-.26)

fig.suptitle(f'Where in the theta cycle each frequency peaks, and how it moves '
             f'with the anchoring state ({NS} sessions)',
             fontsize=8.6, x=.06, ha='left', y=.99)
plt.savefig(OUT, dpi=220, bbox_inches='tight')
plt.close(fig)
print(f'wrote {OUT}')
