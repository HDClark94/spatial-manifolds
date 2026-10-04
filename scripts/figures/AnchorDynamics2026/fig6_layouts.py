"""Figure 5, several panel arrangements and a choice of example session.

Usage:  python3 fig6_layouts.py [mouse day]

The content is fixed -- population anchoring raster, theta amplitude, theta
frequency and speed on a shared trial axis; the three by-state bar pairs; the
two speed-theta gain plots; and the scatters behind them. What varies is how
they are arranged, because the same evidence reads very differently depending on
whether the example or the population is given the dominant position.

    stack       portrait, example on top, population beneath (the incumbent)
    twocol      portrait, example down the left, population down the right
    wide        landscape, example across the top, population in one row
    example     landscape, example given two thirds including both scatters

EXAMPLE SESSION. The default is M28 D19, not the incumbent M28 D23. Of the 19
sessions with enough trials in both states, only five reproduce all four
population effects (amplitude down, frequency down, and both speed-theta
correlations down), and M28 D19 is the only one of those whose state sequence is
also blocky enough to read as a raster (median run 21 trials). M28 D23 gets
three of four, and the one it misses is r(speed, frequency) -- the strongest
population effect at p = 2.7e-4 -- which in that session runs the wrong way.

M28 D25 is the alternative worth considering: it has by far the clearest
decoupling (r falls from +0.44 to -0.10 for amplitude and +0.58 to -0.08 for
frequency) and its speed is nearly identical between states, but its amplitude
goes UP by 2.4%, against the population. Pass it on the command line to see it.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import statsmodels.formula.api as smf
from scipy.ndimage import median_filter
from scipy.stats import pearsonr, wilcoxon

sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import (ANCH_COLOR, NONANCH_COLOR,
                                         load_session_labels)

plt.rcParams['font.family'] = 'Arial'
FIG = '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026'
LFP = '/Users/harryclark/Documents/spatial-manifolds/data/lfp'
MINTR, SMOOTH = 15, 5
TRACES = (('amp', 'Theta\namp.', '#b5651d'),
          ('inst_hz', 'Theta\nHz', '#3b6ea5'),
          ('speed', 'Speed\ncm/s', '#4f8f4f'))

T = pd.read_csv(f'{LFP}/theta_frequency.csv')
T['anch'] = T.anch.astype(bool)


def summarise():
    rows = []
    for (mo, dy), d in T.groupby(['mouse', 'day']):
        a, n = d[d.anch], d[~d.anch]
        if len(a) < MINTR or len(n) < MINTR:
            continue
        r = dict(mouse=mo, day=dy,
                 amp_a=a.amp.mean() / d.amp.mean(), amp_n=n.amp.mean() / d.amp.mean(),
                 hz_a=a.inst_hz.mean(), hz_n=n.inst_hz.mean(),
                 spd_a=a.speed.mean(), spd_n=n.speed.mean())
        for tag, col in (('r_amp', 'amp'), ('r_hz', 'inst_hz')):
            r[f'{tag}_a'] = pearsonr(a.speed, a[col])[0]
            r[f'{tag}_n'] = pearsonr(n.speed, n[col])[0]
        # GAIN AS SLOPE. r is attenuated by however much speed varies, and the
        # anchored state samples 1.5x less speed range, so the r comparison
        # needed speed matching to mean anything. A regression coefficient does
        # not have that problem: it is invariant to the predictor's range, uses
        # every trial, and is what the word "gain" denotes. Amplitude uses the
        # session z-scored column so slopes are comparable across sessions.
        for tag, col in (('sl_amp', 'z_amp'), ('sl_hz', 'inst_hz')):
            r[f'{tag}_a'] = np.polyfit(a.speed, a[col], 1)[0]
            r[f'{tag}_n'] = np.polyfit(n.speed, n[col], 1)[0]
        rows.append(r)
    return pd.DataFrame(rows)


def lmm_p(va, vn, mouse):
    """Paired within day, random intercept for mouse.

    The sessions come from six mice, unevenly (one animal contributes seven),
    so a signed-rank test over sessions treats correlated observations as
    independent and overstates significance -- for theta frequency it gives
    p = 0.049 where this gives 0.38. Day is the pairing unit with exactly two
    observations, so the model collapses to the per-day difference and mouse is
    the random intercept over those.
    """
    D = pd.DataFrame(dict(mouse=np.asarray(mouse), d=np.asarray(va) - np.asarray(vn)))
    try:
        m = smf.mixedlm('d ~ 1', D, groups=D['mouse']).fit(reml=True,
                                                           method='lbfgs')
        p = float(m.pvalues['Intercept'])
    except Exception:
        p = np.nan
    # A REML fit can "succeed" and still return a NaN p when the mouse variance
    # collapses to the boundary -- for running speed it lands at 7e-33, i.e. the
    # model has degenerated to a one-sample test on the differences. Falling
    # back to exactly that is the honest reading, and it must not be skipped:
    # NaN compares False against every threshold, so an unguarded NaN renders as
    # the MOST significant result rather than as a failure.
    return p if np.isfinite(p) else wilcoxon(va, vn).pvalue


S = summarise()


# ── panel primitives ─────────────────────────────────────────────────────────
from functools import lru_cache

_labels = lru_cache(maxsize=None)(load_session_labels)
MAJOR_FILT = 9          # trials; must be odd


def transitions(mo, dy):
    """Trial positions of the MAJOR state changes.

    The raw state flips on single trials, so marking every flip would draw a
    hedge rather than a set of boundaries. The sequence is median-filtered
    first, exactly as fig1_v2 does, which keeps the block structure and drops
    the flicker. Filtering before finding the edges -- rather than finding
    edges and then requiring both neighbouring runs to be long -- is what stops
    a clear boundary going unmarked when a long block is interrupted briefly.
    """
    z = _labels(mo, dy)
    st = median_filter((np.asarray(z['frac_anch']) > .5).astype(float),
                       size=MAJOR_FILT, mode='nearest') > .5
    # trials are drawn at x = 1..n, so a change between st[i] and st[i+1]
    # belongs at x = i + 1.5
    return [i + 1.5 for i in np.where(np.diff(st.astype(int)) != 0)[0]]


def mark_transitions(ax, mo, dy, lw=.9):
    for x in transitions(mo, dy):
        ax.axvline(x, color='0.12', lw=lw, ls='--', zorder=6)


def draw_raster(ax, mo, dy, ylab=True, title=True):
    z = _labels(mo, dy)
    in_pop = np.isfinite(z['pc1_load'])
    L, load_ = z['labels'][in_pop], z['pc1_load'][in_pop]
    order = np.argsort(load_)[::-1]
    n_tr = L.shape[1]
    ax.imshow(L[order], aspect='auto',
              cmap=ListedColormap([NONANCH_COLOR, ANCH_COLOR]),
              norm=BoundaryNorm([-.5, .5, 1.5], 2), interpolation='nearest',
              extent=[1, n_tr, L.shape[0], 1])
    if ylab:
        ax.set_ylabel('Cell (PC1 order)', fontsize=8.5)
    if title:
        ax.set_title(f'M{mo} D{dy}', fontsize=9.5, loc='left')
    mark_transitions(ax, mo, dy)
    ax.set_xticklabels([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return n_tr


def draw_trace(ax, mo, dy, col, lab, colr, n_tr, last=False, ylab=True):
    te = T[(T.mouse == mo) & (T.day == dy)].sort_values('trial')
    v = np.full(n_tr, np.nan)
    ok = (te.trial.values >= 1) & (te.trial.values <= n_tr)
    v[te.trial.values[ok] - 1] = te[col].values[ok]
    tr = np.arange(1, n_tr + 1)
    sm = pd.Series(v).rolling(SMOOTH, center=True, min_periods=2).mean().values
    ax.plot(tr, v, color='0.78', lw=.5)
    ax.plot(tr, sm, color=colr, lw=1.6)
    mark_transitions(ax, mo, dy)
    lo, hi = np.nanpercentile(v, [1, 99]); pad = (hi - lo) * .35
    ax.set_ylim(lo - pad, hi + pad); ax.set_xlim(1, n_tr)
    if ylab:
        ax.set_ylabel(lab, fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))   # short panels fit ~3 ticks
    ax.spines[['top', 'right']].set_visible(False)
    if last:
        ax.set_xlabel('Trial', fontsize=8.5)
    else:
        # the stack shares one trial axis; only the bottom panel draws it
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(bottom=False, labelbottom=False)


def draw_pc1(ax, mo, dy, n_tr, ylab=True, last=False):
    """PC1 of the label matrix: the population classification as one number.

    Drawn as a signed fill rather than a line, matching fig1_v2, so the blocks
    of anchored and non-anchored trials in the raster above read straight down
    into the LFP traces below.
    """
    z = _labels(mo, dy)
    pc = np.nan_to_num(np.asarray(z['pc1'], dtype=float))[:n_tr]
    x = np.arange(1, len(pc) + 1)
    ax.fill_between(x, 0, pc, where=pc >= 0, color=ANCH_COLOR, linewidth=0,
                    edgecolor='none', interpolate=True)
    ax.fill_between(x, 0, pc, where=pc < 0, color=NONANCH_COLOR, linewidth=0,
                    edgecolor='none', interpolate=True)
    ax.axhline(0, color='0.45', lw=.7)
    mark_transitions(ax, mo, dy)
    ax.set_xlim(1, n_tr)
    v = np.nanmax(np.abs(pc)) * 1.15 or 1.0
    ax.set_ylim(-v, v)
    if ylab:
        ax.set_ylabel('PC1', fontsize=7.5)
    ax.annotate(f'{100 * float(z["pc1_var"]):.0f}% var', (.995, .06),
                xycoords='axes fraction', ha='right', fontsize=5.8, color='0.45')
    ax.tick_params(labelsize=6.5)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.spines[['top', 'right']].set_visible(False)
    if last:
        ax.set_xlabel('Trial', fontsize=8.5)
    else:
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(bottom=False, labelbottom=False)


def barpair(ax, va, vn, ylab, title, short=False, p=None):
    # The test is paired within session, so the points are drawn paired: a
    # jittered cloud at each bar shows the spread but hides which value belongs
    # to which session, and the spread BETWEEN sessions is far larger than the
    # within-session effect the bars are claiming.
    for x, v, c in ((0, vn, NONANCH_COLOR), (1, va, ANCH_COLOR)):
        ax.bar(x, np.mean(v), width=.72, color=c, lw=.8, edgecolor='k', zorder=1)
    for i in range(len(va)):
        ax.plot([0, 1], [vn[i], va[i]], color='0.35', lw=.45, alpha=.75, zorder=2)
    ax.scatter(np.zeros(len(vn)), vn, s=7, color='0.15', lw=0, alpha=.8, zorder=3)
    ax.scatter(np.ones(len(va)), va, s=7, color='0.15', lw=0, alpha=.8, zorder=3)
    for x, v in ((0, vn), (1, va)):
        ax.errorbar(x, np.mean(v), yerr=np.std(v, ddof=1) / np.sqrt(len(v)),
                    color='k', lw=1.3, capsize=3, capthick=1.2, zorder=5)
    if p is None:
        p = wilcoxon(va, vn).pvalue
    lo, hi = min(va.min(), vn.min()), max(va.max(), vn.max())
    span = hi - lo
    y = hi + .10 * span
    ax.plot([0, 0, 1, 1], [y, y + .04 * span, y + .04 * span, y], color='0.3', lw=.8)
    star = ('?' if not np.isfinite(p) else
            'n.s.' if p > .05 else '*' if p > .01 else '**' if p > .001 else '***')
    ax.text(.5, y + .07 * span, star, ha='center', fontsize=7.5, color='0.2')
    ax.set_ylim(max(0, lo - .25 * span), y + .22 * span)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['non', 'anch'] if short else ['non-anch', 'anchored'],
                       fontsize=7.5)
    ax.set_ylabel(ylab, fontsize=8.5)
    ax.set_title(title, fontsize=8.5, loc='left')
    ax.tick_params(labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    return p


def pairdots(ax, va, vn, ylab, title, short=False, p=None):
    for i in range(len(va)):
        ax.plot([0, 1], [vn[i], va[i]], color='0.78', lw=.6, zorder=1)
    ax.scatter(np.zeros(len(vn)), vn, s=13, color=NONANCH_COLOR, lw=0, zorder=2)
    ax.scatter(np.ones(len(va)), va, s=13, color=ANCH_COLOR, lw=0, zorder=2)
    for x, v, c in ((0, vn, NONANCH_COLOR), (1, va, ANCH_COLOR)):
        ax.plot([x - .17, x + .17], [np.median(v)] * 2, color=c, lw=2.2, zorder=3)
    ax.axhline(0, color='0.6', lw=.7, ls=':')
    if p is None:
        p = wilcoxon(va, vn).pvalue
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['non', 'anch'] if short else ['non-anch', 'anchored'],
                       fontsize=7.5)
    ax.set_xlim(-.4, 1.4)
    ax.set_ylabel(ylab, fontsize=8.5)
    ax.set_title((f'{title.split(":")[-1].strip()}\np = {p:.2g}' if short
                  else f'{title} — p = {p:.2g}'), fontsize=8.5, loc='left')
    ax.tick_params(labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    return p


def draw_scatter(ax, mo, dy, col, ylab, xlab=True):
    d = T[(T.mouse == mo) & (T.day == dy)]
    for m, c in ((d.anch, ANCH_COLOR), (~d.anch, NONANCH_COLOR)):
        s_ = d[m]
        ax.scatter(s_.speed, s_[col], s=5, color=c, alpha=.5, lw=0)
        xs = np.linspace(s_.speed.min(), s_.speed.max(), 40)
        ax.plot(xs, np.polyval(np.polyfit(s_.speed, s_[col], 1), xs), color=c, lw=1.6)
    ra = pearsonr(d[d.anch].speed, d[d.anch][col])[0]
    rn = pearsonr(d[~d.anch].speed, d[~d.anch][col])[0]
    ax.annotate(f'r  {rn:+.2f} → {ra:+.2f}', (.03, .96), xycoords='axes fraction',
                va='top', fontsize=6.8, color='0.25')
    if xlab:
        ax.set_xlabel('Speed (cm/s)', fontsize=8.5)
    ax.set_ylabel(ylab, fontsize=8.5)
    ax.tick_params(labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)


def legend(fig, y=.002):
    h = [plt.Line2D([], [], marker='s', ls='', color=ANCH_COLOR, label='anchored'),
         plt.Line2D([], [], marker='s', ls='', color=NONANCH_COLOR,
                    label='non-anchored')]
    fig.legend(handles=h, loc='lower center', ncol=2, fontsize=8, frameon=False,
               bbox_to_anchor=(.5, y))


def bars_and_gains(get, short=False):
    """The five population panels. All p values are the mixed model's."""
    mo = S.mouse.values
    barpair(get(0), S.amp_a.values, S.amp_n.values,
            'Theta amplitude\n(session-normalised)', 'amplitude', short,
            p=lmm_p(S.amp_a.values, S.amp_n.values, mo))
    barpair(get(1), S.hz_a.values, S.hz_n.values, 'Theta frequency (Hz)',
            'frequency', short, p=lmm_p(S.hz_a.values, S.hz_n.values, mo))
    barpair(get(2), S.spd_a.values, S.spd_n.values, 'Speed (cm/s)', 'speed',
            short, p=lmm_p(S.spd_a.values, S.spd_n.values, mo))
    pairdots(get(3), S.sl_amp_a.values, S.sl_amp_n.values,
             'd(amplitude)/d(speed)\n(z per cm/s)', 'gain: amplitude', short,
             p=lmm_p(S.sl_amp_a.values, S.sl_amp_n.values, mo))
    pairdots(get(4), S.sl_hz_a.values, S.sl_hz_n.values,
             'd(frequency)/d(speed)\n(Hz per cm/s)', 'gain: frequency', short,
             p=lmm_p(S.sl_hz_a.values, S.sl_hz_n.values, mo))


# ── layouts ──────────────────────────────────────────────────────────────────
def layout_stack(mo, dy):
    """Portrait: example on top, population beneath. The incumbent."""
    fig = plt.figure(figsize=(7.5, 7.2))
    gs = fig.add_gridspec(9, 3,
                          height_ratios=[1.6, .28, .34, .34, .34, .34, 1.15, .24, 1.15],
                          hspace=.32, wspace=.46)
    n_tr = draw_raster(fig.add_subplot(gs[0, :]), mo, dy)
    draw_pc1(fig.add_subplot(gs[1, :]), mo, dy, n_tr)
    for k, (c, lab, colr) in enumerate(TRACES):
        draw_trace(fig.add_subplot(gs[k + 2, :]), mo, dy, c, lab, colr, n_tr,
                   last=(k == 2))
    ax_map = {0: gs[6, 0], 1: gs[6, 1], 2: gs[6, 2], 3: gs[8, 0], 4: gs[8, 1]}
    bars_and_gains(lambda i: fig.add_subplot(ax_map[i]))
    draw_scatter(fig.add_subplot(gs[8, 2]), mo, dy, 'amp', 'Theta amplitude')
    legend(fig)
    return fig


def layout_twocol(mo, dy):
    """Portrait: example down the left, population down the right."""
    fig = plt.figure(figsize=(10.0, 5.2))
    outer = fig.add_gridspec(1, 2, width_ratios=[1, 1.06], wspace=.3)
    gl = outer[0].subgridspec(5, 1, height_ratios=[1.8, .30, .36, .36, .36],
                              hspace=.20)
    n_tr = draw_raster(fig.add_subplot(gl[0]), mo, dy)
    draw_pc1(fig.add_subplot(gl[1]), mo, dy, n_tr)
    for k, (c, lab, colr) in enumerate(TRACES):
        draw_trace(fig.add_subplot(gl[k + 2]), mo, dy, c, lab, colr, n_tr,
                   last=(k == 2))
    gr = outer[1].subgridspec(2, 3, hspace=.78, wspace=.95)
    ax_map = {0: gr[0, 0], 1: gr[0, 1], 2: gr[0, 2], 3: gr[1, 0], 4: gr[1, 1]}
    bars_and_gains(lambda i: fig.add_subplot(ax_map[i]), short=True)
    draw_scatter(fig.add_subplot(gr[1, 2]), mo, dy, 'inst_hz',
                 'Theta frequency (Hz)')
    legend(fig, y=-.01)
    return fig


def layout_wide(mo, dy):
    """Landscape: example across the top, the five population panels in one row."""
    fig = plt.figure(figsize=(11.0, 5.2))
    gs = fig.add_gridspec(7, 5, height_ratios=[1.55, .28, .34, .34, .34, .40, 1.5],
                          hspace=.34, wspace=.46)
    n_tr = draw_raster(fig.add_subplot(gs[0, :]), mo, dy)
    draw_pc1(fig.add_subplot(gs[1, :]), mo, dy, n_tr)
    for k, (c, lab, colr) in enumerate(TRACES):
        draw_trace(fig.add_subplot(gs[k + 2, :]), mo, dy, c, lab, colr, n_tr,
                   last=(k == 2))
    bars_and_gains(lambda i: fig.add_subplot(gs[6, i]))
    legend(fig, y=-.015)
    return fig


def layout_example(mo, dy):
    """Landscape: the example given two thirds, including both scatters."""
    fig = plt.figure(figsize=(11.0, 5.7))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=.26)
    gl = outer[0].subgridspec(6, 2, height_ratios=[1.7, .28, .34, .34, .34, 1.35],
                              hspace=.42, wspace=.34)
    n_tr = draw_raster(fig.add_subplot(gl[0, :]), mo, dy)
    draw_pc1(fig.add_subplot(gl[1, :]), mo, dy, n_tr)
    for k, (c, lab, colr) in enumerate(TRACES):
        draw_trace(fig.add_subplot(gl[k + 2, :]), mo, dy, c, lab, colr, n_tr,
                   last=(k == 2))
    draw_scatter(fig.add_subplot(gl[5, 0]), mo, dy, 'amp', 'Theta amplitude')
    draw_scatter(fig.add_subplot(gl[5, 1]), mo, dy, 'inst_hz', 'Theta freq. (Hz)')
    gr = outer[1].subgridspec(3, 2, hspace=.75, wspace=.72)
    ax_map = {0: gr[0, 0], 1: gr[0, 1], 2: gr[1, 0], 3: gr[1, 1], 4: gr[2, 0]}
    bars_and_gains(lambda i: fig.add_subplot(ax_map[i]))
    legend(fig, y=-.012)
    return fig


LAYOUTS = dict(stack=layout_stack, twocol=layout_twocol, wide=layout_wide,
               example=layout_example)

# the five sessions that reproduce all four population effects and are legible
# as a raster, ordered by typicality (fig6_session_ranking.py)
ALL_FOUR = [(21, 21), (21, 19), (27, 19), (28, 19), (28, 16)]

if __name__ == '__main__':
    if sys.argv[1:2] == ['allfour']:
        name = sys.argv[2] if len(sys.argv) > 2 else 'twocol'
        print(f'{len(S)} sessions in the population panels; '
              f'{name} layout for the {len(ALL_FOUR)} all-four sessions')
        for mo, dy in ALL_FOUR:
            fig = LAYOUTS[name](mo, dy)
            out = f'{FIG}/fig6_layout_{name}_M{mo}D{dy}.pdf'
            fig.savefig(out, dpi=180, bbox_inches='tight'); plt.close(fig)
            print('  saved', os.path.basename(out))
        sys.exit()
    MO, DY = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (28, 19)
    only = sys.argv[3] if len(sys.argv) > 3 else None
    print(f'{len(S)} sessions in the population panels; example M{MO}D{DY}')
    for name, fn in ({only: LAYOUTS[only]} if only else LAYOUTS).items():
        fig = fn(MO, DY)
        out = f'{FIG}/fig6_layout_{name}_M{MO}D{DY}.pdf'
        fig.savefig(out, dpi=180, bbox_inches='tight')
        plt.close(fig)
        print('  saved', os.path.basename(out))
