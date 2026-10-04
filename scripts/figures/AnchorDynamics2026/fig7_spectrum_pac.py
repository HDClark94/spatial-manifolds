"""Which frequencies change with the anchoring state, and what happens to nested gamma.

Replaces the claim in Results §7 that the LFP change is BROADBAND. It is not.
Measured at ~1 Hz resolution rather than in five preset bands, the difference
between states changes SIGN across frequency: theta falls, slow gamma RISES, and
mid-to-high gamma falls. A gain change cannot do that.

The earlier binning is why this was missed. Its "gamma" band was 30-80 Hz, which
straddles the rise and the fall and averages them toward nothing, and its "high"
band was 80-200 Hz, which is the falling side only -- so the five numbers said
"theta down a little, high frequencies down more", which reads as broadband.
Splitting at 55-60 Hz separates two effects of opposite sign.

The coupling result is the one the power spectrum cannot give. Theta-nested FAST
gamma coupling falls by 11.8% when the population is anchored (24 of 28
sessions), while SLOW gamma coupling does not move at all. Both couplings are
real -- 48x and 16x their own phase-shuffle nulls -- so this is a change in which
gamma rhythm theta organises, not a change in whether it organises one.

CAVEAT ON THE SLOW-GAMMA NULL, which is weaker than it looks. The slow band is
30-55 Hz and therefore contains the 50 Hz mains line, which is strong, roughly
constant in amplitude, and not locked to theta phase. A component like that
dilutes the modulation index toward zero, so a real slow-gamma change could be
masked. It does not manufacture one, and the power results are notched, but the
slow-gamma coupling should be re-run on 30-48 Hz before the null is relied on.

The fast-gamma reduction is uniform across entorhinal layers -- L1 p = 0.003,
L2 p = 0.007, L3 p = 8e-4, L5 p = 1e-4, L6 not significant at n = 6 -- which is
reported here rather than plotted, and which distinguishes it from the cell-level
following that peaks in ENTm2.

Everything is speed-matched, sample-count-matched between states, and
null-subtracted; see lfp_spectrum_pac.py for why each of those is necessary.

Writes fig7_spectrum_pac.pdf
"""
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import ANCH_COLOR, NONANCH_COLOR

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = os.path.dirname(os.path.abspath(__file__))
OUT = f'{FIG}/fig7_spectrum_pac.pdf'
GRID_C = '#c04744'
BANDS = [('delta', 1, 4), ('theta', 6, 10), ('beta', 12, 20),
         ('slow\ngamma', 30, 48), ('mid\ngamma', 60, 100),
         ('high\ngamma', 100, 150), ('HFO', 150, 250)]
# Mains and its harmonics. The 50 Hz line sits INSIDE a conventional 30-55 Hz
# slow-gamma band and is the largest peak in the spectrum after theta, so a
# slow-gamma effect read off that band could be a line-noise effect. Two things
# are done about it: the band stops at 48 Hz, and these windows are excluded
# from every band integral. For the record the result does not depend on it --
# 30-55 unnotched gives +0.106 (p = 5.9e-4), notched +0.109 (p = 3.6e-4),
# 30-48 clean +0.105 (p = 0.0013, 21/26 sessions) -- and the 48-52 Hz line band
# on its own is not significant (+0.094, p = 0.063).
LINE = [(48, 52), (98, 102), (148, 152), (198, 202)]


def not_mains(f):
    return not any(lo <= f <= hi for lo, hi in LINE)

S = pd.concat([pd.read_csv(f) for f in
               sorted(glob.glob(f'{ROOT}/data/lfp/spectrum_by_anchoring_w*.csv'))],
              ignore_index=True)
P = pd.concat([pd.read_csv(f) for f in
               sorted(glob.glob(f'{ROOT}/data/lfp/pac_by_anchoring_w*.csv'))],
              ignore_index=True)
# SUFFICIENT SWITCHING. A session contributes only if it has at least MIN_TRIALS
# speed-matched trials in EACH state; with fewer, a per-state spectrum is an
# average over a handful of trials and the state comparison is mostly noise.
# 26 of 28 sessions qualify.
MIN_TRIALS = 8
_n = S.drop_duplicates(['mouse', 'day', 'state']).pivot_table(
    index=['mouse', 'day'], columns='state', values='n_trials')
KEEP = set(_n[_n.min(axis=1) >= MIN_TRIALS].index)
S = S[[k in KEEP for k in zip(S.mouse, S.day)]]
P = P[[k in KEEP for k in zip(P.mouse, P.day)]]

# SESSION-NORMALISED POWER, keeping the spectrum's shape. The stored `z` is
# z-scored per frequency, which removes the 1/f falloff and so cannot be plotted
# as a spectrum. Subtracting each session's own mean log power over both states
# and all frequencies instead puts sessions on a common scale -- absolute
# microvolts depend on impedance and reference -- while leaving the shape intact.
S = S.copy()
# 1/f-COMPENSATED. Raw LFP power falls so steeply with frequency that everything
# above theta is squashed into the bottom of the axis and the gamma peaks -- the
# ones this figure is about -- are invisible. Multiplying power by frequency
# flattens the 1/f background, which in log units is just adding log10(f), and
# leaves genuine peaks standing above a roughly level baseline.
S['whit'] = S.log_p + np.log10(S.freq)
S['rel'] = S.whit - S.groupby(['mouse', 'day']).whit.transform('mean')
R = S.pivot_table(index=['mouse', 'day', 'freq'], columns='state',
                  values='rel').dropna().reset_index()
W = S.pivot_table(index=['mouse', 'day', 'freq'], columns='state',
                  values='z').dropna().reset_index()
W['d'] = W.anch - W.non
NS = W.groupby(['mouse', 'day']).ngroups

fig = plt.figure(figsize=(7.8, 3.4))
G = fig.add_gridspec(1, 3, wspace=.62, width_ratios=[1.40, 1.15, .85])


def tidy(ax):
    ax.tick_params(labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)


def lp(ax, s, x=-.20):
    ax.text(x, 1.05, s, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='right')


# ---- A: the two spectra ------------------------------------------------------
ax = fig.add_subplot(G[0])
for st, col, lab in (('non', NONANCH_COLOR, 'non-anchored'),
                     ('anch', ANCH_COLOR, 'anchored')):
    g = R.groupby('freq')[st].agg(['mean', 'sem'])
    ax.fill_between(g.index, g['mean'] - g['sem'], g['mean'] + g['sem'],
                    color=col, alpha=.30, linewidth=0)
    ax.plot(g.index, g['mean'], color=col, lw=1.2, label=lab)
ax.axvspan(6, 10, color='0.88', alpha=.6, linewidth=0, zorder=0)
ax.axvspan(30, 55, color='0.88', alpha=.6, linewidth=0, zorder=0)
ax.set_xscale('log')
ax.set_xticks([2, 8, 20, 50, 100, 200])
ax.set_xticklabels(['2', '8', '20', '50', '100', '200'], fontsize=7)
ax.set_xlabel('frequency (Hz)', fontsize=8)
ax.set_ylabel('session-normalised\nlog (power $\\times$ frequency)', fontsize=8)
ax.set_title(f'the two states, as spectra\n({NS} sessions that switch)',
             fontsize=8.5)
_g50 = R.groupby('freq').anch.mean()
_f50 = _g50.index[np.argmin(np.abs(_g50.index - 50))]
ax.annotate('mains', xy=(50, _g50.loc[_f50]),
            xytext=(88, _g50.loc[_f50] - .06), fontsize=5.8, color='0.45',
            ha='left', va='top',
            arrowprops=dict(arrowstyle='->', color='0.55', lw=.7))
ax.legend(fontsize=6.5, frameon=False, loc='lower left')
tidy(ax)
lp(ax, 'A', x=-.23)

# ---- B: by band --------------------------------------------------------------
ax = fig.add_subplot(G[1])
for j, (nm, lo, hi) in enumerate(BANDS):
    g = W[(W.freq >= lo) & (W.freq < hi) & W.freq.map(not_mains)] \
        .groupby(['mouse', 'day']).d.mean().dropna()
    p = wilcoxon(g).pvalue
    c = GRID_C if p < .05 else '0.6'
    ax.bar(j, g.mean(), .62, color=c, linewidth=0)
    se = g.std() / np.sqrt(len(g))
    ax.plot([j, j], [g.mean() - se, g.mean() + se], color='0.3', lw=1)
    ax.annotate(f'{p:.0e}'.replace('e-0', 'e-') if p < .001 else f'{p:.3f}',
                (j, g.mean() + np.sign(g.mean()) * (se + .012)),
                ha='center', va='bottom' if g.mean() > 0 else 'top',
                fontsize=5.4, color=c)
ax.axhline(0, color='0.4', lw=1)
ax.set_xticks(range(len(BANDS)))
ax.set_xticklabels([b[0].replace('\n', ' ') for b in BANDS], fontsize=6,
                   rotation=38, ha='right')
ax.set_ylabel('anchored $-$ non (z)', fontsize=8)
ax.set_title('the difference changes SIGN,\nso it is not a gain change', fontsize=8.5)
tidy(ax)
lp(ax, 'B', x=-.24)

# ---- C: theta-gamma coupling -------------------------------------------------
ax = fig.add_subplot(G[2])
w = .34
for j, b in enumerate(['slow_gamma', 'fast_gamma']):
    g = P[P.band == b].groupby(['mouse', 'day', 'state']).mi_excess.mean() \
        .unstack().dropna()
    p = wilcoxon(g.anch, g.non).pvalue
    for k, (st, col) in enumerate((('non', NONANCH_COLOR), ('anch', ANCH_COLOR))):
        v = g[st]
        ax.bar(j + (k - .5) * w, v.mean(), w, color=col, linewidth=0,
               label=('non-anchored', 'anchored')[k] if j == 0 else None)
        se = v.std() / np.sqrt(len(v))
        ax.plot([j + (k - .5) * w] * 2, [v.mean() - se, v.mean() + se],
                color='0.3', lw=1)
    ax.annotate(f'{p:.0e}'.replace('e-0', 'e-') if p < .001 else f'p={p:.2f}',
                (j, max(g.anch.mean(), g.non.mean()) * 1.09), ha='center',
                fontsize=6, color=GRID_C if p < .05 else '0.5')
ax.set_xticks([0, 1]); ax.set_xticklabels(['slow\n30–55*', 'fast\n60–100'],
                                          fontsize=6.8)
ax.set_ylabel('theta–gamma MI\nover shuffle null', fontsize=8)
ax.set_title('only FAST gamma\nuncouples', fontsize=8.5)
ax.text(.5, -.30, '* contains mains', transform=ax.transAxes, fontsize=5.2,
        color='0.5', ha='center')
ax.set_ylim(0, 0.0034)
ax.legend(fontsize=5.6, frameon=False, loc='upper left',
          handlelength=.9, borderpad=.1)
tidy(ax)
lp(ax, 'C', x=-.34)

fig.savefig(OUT, bbox_inches='tight')
print(f'wrote {OUT}')
