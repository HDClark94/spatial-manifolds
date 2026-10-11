"""Figure 7: what the entorhinal LFP does between anchoring states.

Portrait, four rows: the example full width, then the six population panels two
to a row. Reading order is A-D down, then E F / G H / I J.

A-D  ONE SESSION, trial by trial. The population axis, the spectrogram, three
     band traces, and theta-nested gamma for BOTH bands on twinned axes, all on
     one trial axis with the major transitions marked. This is what the effect
     looks like before it is averaged over anything. Slow gamma needs its own
     axis because its MI is ~5x smaller than fast gamma's; on a shared axis it
     flattens against the bottom and the reader cannot see that it does not fall.

E-G  THE POPULATION. The two states as spectra, the difference by band, and
     theta-gamma coupling. The headline is that the difference CHANGES SIGN
     across frequency -- theta down, slow gamma up, mid-to-high gamma down --
     which a gain change cannot do, and which replaces the earlier claim that
     the LFP change is broadband. That claim came from a five-band split whose
     "gamma" ran 30-80 Hz, straddling the rise and the fall and averaging them
     away.

H-J  WHETHER IT IS A TRADE-OFF. It is not shown to be one, and these panels say
     so. Slow gamma rises in 21 of 26 sessions and fast gamma falls in 19, but:

     (i) the two changes are UNCORRELATED in magnitude across sessions
     (Spearman rho = -0.08, p = 0.70), where a redistribution of a fixed amount
     of power predicts a negative correlation; and

     (ii) the 16 of 26 sessions in the slow-up/fast-down quadrant are no more
     than the two marginals already imply (expected 15.3, Fisher p = 0.59). An
     earlier version tested that count against a binomial at p = 0.25 and read
     it as evidence of a coupled shift; that null assumes each band is equally
     likely to rise or fall, which the band effects in F have already rejected,
     so it re-tested the marginals and reported them as a conjunction.

     The honest statement is that both bands move in opposite directions, not
     that one is drawn from the other. The null is not decisive -- session-level
     noise could mask a within-session trade-off, and panel C shows the two
     running anti-phase trial by trial in the example -- and the test that would
     settle it is a within-session, trial-by-trial correlation with speed and
     state partialled out.

     J's error band is the SEM across sessions at each frequency, so it shows
     whether the sessions agree on where the sign change sits rather than how
     smooth one averaged curve looks.

Everything is speed-matched (population) or shows the speed trace (example);
coupling is sample-count-matched and null-subtracted. See lfp_spectrum_pac.py.

Writes fig7_theta_gamma.pdf
"""
import glob
import os
import sys
import warnings

import h5py
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d
from scipy.signal import filtfilt, hilbert, welch
from scipy.stats import fisher_exact, pearsonr, spearmanr, wilcoxon

sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lfp_batch import FS, LFP_ROOT, clip_trials, mec_groups, nap, vr_paths
from lfp_spectrum_pac import GAMMA, _bp, tort_mi
from spatial_manifolds.anchoring import (ANCH_COLOR, NONANCH_COLOR,
                                         major_transitions)

plt.rcParams['font.family'] = 'Arial'
# mathtext must resolve to Arial, not the DejaVu/STIX default, or rho, times
# and the panel letters render in a different face from the rest of the figure
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'
plt.rcParams['pdf.fonttype'] = 42

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = os.path.dirname(os.path.abspath(__file__))
OUT = f'{FIG}/fig7_theta_gamma.pdf'

MOUSE, DAY = 26, 18
NPERSEG, FLO, FHI = 1024, 1.9, 150.
MIN_RUN_S, PAC_WIN, SMOOTH = 2.0, 9, 5
N_LOGBINS, TRIAL_SIGMA, MAJOR_FILT = 56, 1.8, 9
MIN_TRIALS = 8
# Emphasis colour for 'significant', not a cell class: this figure has no
# cell types in it, and the grid-cell red it used to borrow is reserved.
SIG_C = '#2f3e46'
LINE = [(48, 52), (98, 102), (148, 152)]
BANDS = [('theta 6–10', 6, 10, '#7b4173'),
         ('slow gamma 30–48', 30, 48, '#2f7d32'),
         ('fast gamma 60–100', 60, 100, '#e8701a')]
# BAND PALETTE. Slow and fast gamma used to be drawn in the non-grid-spatial
# blue and the grid-cell red of the class palette (Figure 2). Those two
# colours are reserved for cell types across the paper, so the bands now take
# green and orange. Theta keeps its purple, which is not a class colour.
# mid gamma (60-100) and HFO (150-250) were dropped from this panel: the
# 60-100 band already has a panel of its own in the band traces above, and HFO
# at this reference carries no state effect worth a column.
PBANDS = [('delta', 2, 5), ('theta', 6, 10), ('beta', 12, 20),
          ('slow\ngamma', 30, 48), ('high\ngamma', 100, 150)]


def not_mains(f):
    return not any(lo <= f <= hi for lo, hi in LINE)


# ============================ the example session ==============================
lf = f'{LFP_ROOT}/M{MOUSE}/D{DAY}/VR/sub-M{MOUSE}_ses-D{DAY}_typ-VR_beh.nwb'
f_ = h5py.File(lf, 'r')
names = [n.decode() for n in f_['general/extracellular_ephys/electrodes/channel_name'][:]]
Mg = mec_groups(MOUSE, DAY, names)
cols = Mg.g.values
rc = np.sort(cols); back = np.array([int(np.where(rc == c)[0][0]) for c in cols])
Dx = f_['processing/ecephys/LFP/lfp/data'][:, rc][:, back].astype(np.float32)
PHx = f_['processing/ecephys/Processed/theta/data'][:, rc][:, back].astype(np.float32)
f_.close()

bp, cp = vr_paths(MOUSE, DAY)
beh = nap.load_file(bp); clusters = nap.load_file(cp)
trials, _ = clip_trials(beh['trials'].as_dataframe(), clusters)
T = pd.read_csv(f'{ROOT}/data/population_state/anchoring_trials.csv')
stx = T[(T.mouse == MOUSE) & (T.day == DAY)]
frac = dict(zip(stx.trial.astype(int), stx.frac_anch))
pc1d = dict(zip(stx.trial.astype(int), stx.pc1))

tt = np.arange(Dx.shape[0]) / FS
Sv = beh['S']
spd = np.asarray(Sv.values)[np.searchsorted(np.asarray(Sv.index), tt)
                            .clip(0, len(Sv) - 1)]
run = spd >= 3.0
env = {}
for nm, lo, hi in GAMMA:
    b, a = _bp(lo, hi)
    env[nm] = np.abs(hilbert(filtfilt(b, a, Dx, axis=0), axis=0))

spec, nums, masks, tspd = [], [], [], []
for _, tr in trials.iterrows():
    u = int(tr.number)
    if u not in frac:
        continue
    m = (tt >= tr.start) & (tt <= tr.end) & run
    if m.sum() < MIN_RUN_S * FS:
        continue
    fr, Pw = welch(Dx[m], fs=FS, nperseg=NPERSEG, axis=0)
    sel = (fr >= FLO) & (fr <= FHI)
    spec.append(Pw[sel].mean(axis=1))
    nums.append(u); masks.append(m); tspd.append(float(spd[m].mean()))
fr_sel = fr[sel]
Ax = np.array(spec); nums = np.array(nums); tspd = np.array(tspd)
pc = np.array([pc1d[u] for u in nums]); pc = (pc - pc.mean()) / (pc.std() or 1)
anch = np.array([frac[u] for u in nums]) > .5
# major transitions are sign changes of PC1, which is the trace in A; the
# shading still follows the thresholded anchored fraction, since the panels
# shade STATE and the lines mark where it turns over
major = anch
TRANS = np.array(major_transitions(pc))

Lx = np.log10(Ax + 1e-20)
Zx = (Lx - Lx.mean(0)) / (Lx.std(0) + 1e-12)
keepf = np.array([not_mains(x) for x in fr_sel])
EDGES = np.geomspace(FLO, FHI, N_LOGBINS + 1)
rows_, centres = [], []
for lo, hi in zip(EDGES[:-1], EDGES[1:]):
    k = (fr_sel >= lo) & (fr_sel < hi) & keepf
    if k.sum():
        rows_.append(Lx[:, k].mean(1)); centres.append(np.sqrt(lo * hi))
LB = np.array(rows_).T
ZB = gaussian_filter1d((LB - LB.mean(0)) / (LB.std(0) + 1e-12), TRIAL_SIGMA,
                       axis=0, mode='nearest')
centres = np.array(centres)


def band_trace(lo, hi):
    k = (fr_sel >= lo) & (fr_sel < hi) & keepf
    return uniform_filter1d(Zx[:, k].mean(1), SMOOTH, mode='nearest')


# Both gamma bands, not just fast. The population panels report that fast gamma
# uncouples from theta while slow gamma does not, so the example has to show both
# traces or it illustrates only half the claim -- and a reader seeing one falling
# trace cannot tell whether the other band did the same thing.
pac_t = []
pac_v = {nm: [] for nm, _, _ in GAMMA}
for i in range(len(nums) - PAC_WIN + 1):
    m = np.zeros(len(tt), bool)
    for k in range(i, i + PAC_WIN):
        m |= masks[k]
    if m.sum() < 15 * FS:
        continue
    pac_t.append(i + PAC_WIN // 2)
    for nm, _, _ in GAMMA:
        pac_v[nm].append(np.nanmean([tort_mi(PHx[m, j], env[nm][m, j])
                                     for j in range(Dx.shape[1])]))
pac_t = np.array(pac_t)
pac_v = {k: np.array(v) for k, v in pac_v.items()}

# ============================ the population ===================================
S = pd.concat([pd.read_csv(f) for f in
               sorted(glob.glob(f'{ROOT}/data/lfp/spectrum_by_anchoring_w*.csv'))],
              ignore_index=True)
P = pd.concat([pd.read_csv(f) for f in
               sorted(glob.glob(f'{ROOT}/data/lfp/pac_by_anchoring_w*.csv'))],
              ignore_index=True)
_n = S.drop_duplicates(['mouse', 'day', 'state']).pivot_table(
    index=['mouse', 'day'], columns='state', values='n_trials')
KEEP = set(_n[_n.min(axis=1) >= MIN_TRIALS].index)
S = S[[k in KEEP for k in zip(S.mouse, S.day)]].copy()
P = P[[k in KEEP for k in zip(P.mouse, P.day)]]
S['whit'] = S.log_p + np.log10(S.freq)
S['rel'] = S.whit - S.groupby(['mouse', 'day']).whit.transform('mean')
R = S.pivot_table(index=['mouse', 'day', 'freq'], columns='state',
                  values='rel').dropna().reset_index()
W = S.pivot_table(index=['mouse', 'day', 'freq'], columns='state',
                  values='z').dropna().reset_index()
W['d'] = W.anch - W.non
NS = W.groupby(['mouse', 'day']).ngroups
Wok = W[W.freq.map(not_mains)]


def pband(lo, hi):
    return Wok[(Wok.freq >= lo) & (Wok.freq < hi)].groupby(['mouse', 'day']).d.mean()


TR = pd.concat([pband(30, 48).rename('slow'), pband(60, 100).rename('fast')],
               axis=1).dropna()
TR['shift'] = TR.slow - TR.fast
TR['sum'] = TR.slow + TR.fast
# The redistribution claim is a WITHIN-session, paired one: do the two bands
# move equal and opposite? That is the sum, not the across-session
# correlation. A uniform see-saw -- every session shifting by much the same
# amount -- produces no across-session correlation at all, so the correlation
# that used to title this panel cannot test the claim it was named after.
TR['broad'] = Wok.groupby(['mouse', 'day']).d.mean()        # 2-200 Hz
# the same contrast in log power rather than in per-frequency z, as a check
# that the conservation is not an artifact of z-scoring each frequency
_LP = S.pivot_table(index=['mouse', 'day', 'freq'], columns='state',
                    values='log_p').dropna().reset_index()
_LP['d'] = _LP.anch - _LP.non
_LP = _LP[_LP.freq.map(not_mains)]
_lb = lambda lo, hi: _LP[(_LP.freq >= lo) & (_LP.freq < hi)] \
    .groupby(['mouse', 'day']).d.mean()
LPSUM = (_lb(30, 48) + _lb(60, 100)).dropna()
# ... and weighted by bandwidth, which is the honest caveat: the fast band is
# 40 Hz wide against slow's 18, so a per-Hz balance is not a power budget
LPSUM_BW = (_lb(30, 48) * 18 + _lb(60, 100) * 40).dropna()
P_SUM = wilcoxon(TR['sum']).pvalue
P_SHIFT = wilcoxon(TR['shift']).pvalue
P_BROAD = wilcoxon(TR['broad']).pvalue
R_SF, _ = pearsonr(TR.slow, TR.fast)
_z, _se = np.arctanh(R_SF), 1 / np.sqrt(len(TR) - 3)
CI_SF = np.tanh([_z - 1.96 * _se, _z + 1.96 * _se])
BOTH = int(((TR.slow > 0) & (TR.fast < 0)).sum())


def crossover(g):
    """Frequency at which a session's anchored - non profile turns negative.

    Computed, not written in: an earlier version of panel J carried a hardcoded
    '43 Hz' that does not reproduce from the data file. Smoothed over 5 bins
    because single 0.98 Hz bins cross zero repeatedly on noise, and the LAST
    upward-to-downward crossing is taken so that a wobble near 25 Hz does not
    get reported as the gamma crossover.
    """
    g = g.sort_values('freq')
    g = g[(g.freq >= 25) & (g.freq <= 90)]
    d = np.convolve(g.d.values, np.ones(5) / 5, mode='same')
    f = g.freq.values
    idx = np.where((d[:-1] > 0) & (d[1:] <= 0))[0]
    if not len(idx):
        return np.nan
    i = idx[-1]
    return f[i] + (f[i + 1] - f[i]) * d[i] / (d[i] - d[i + 1])


CX = Wok.groupby(['mouse', 'day'])[['freq', 'd']].apply(crossover).dropna()

print(f'example M{MOUSE}D{DAY}: {len(nums)} trials, {len(TRANS)} major transitions')
print(f'population: {NS} switching sessions')
# The quadrant count must be tested against the OBSERVED marginals, not against
# p = 0.25. A binomial at 0.25 assumes each band is equally likely to rise or
# fall, which the band effects above have already rejected -- it re-tests the
# marginals and reports them as a conjunction. Given 21/26 up and 19/26 down,
# independence already predicts ~15 of 26 in the quadrant.
_su, _fd = TR.slow > 0, TR.fast < 0
TAB = [[int((_su & _fd).sum()), int((_su & ~_fd).sum())],
       [int((~_su & _fd).sum()), int((~_su & ~_fd).sum())]]
P_QUAD = fisher_exact(TAB)[1]
EXP_QUAD = _su.sum() * _fd.sum() / len(TR)
print(f'  slow up {int(_su.sum())}/{len(TR)}, fast down {int(_fd.sum())}/{len(TR)}, '
      f'both {BOTH}/{len(TR)} (expected under observed marginals {EXP_QUAD:.1f}, '
      f'Fisher p={P_QUAD:.2f})')
print(f'  shift (slow-fast) {TR["shift"].mean():+.3f} p={P_SHIFT:.2g}  |  '
      f'sum (slow+fast) {TR["sum"].mean():+.3f} p={P_SUM:.2g}  |  '
      f'broadband {TR["broad"].mean():+.3f} p={P_BROAD:.2g}')
print(f'  in log power: sum {LPSUM.mean():+.4f} p={wilcoxon(LPSUM).pvalue:.2g}; '
      f'bandwidth-weighted {LPSUM_BW.mean():+.3f} '
      f'p={wilcoxon(LPSUM_BW).pvalue:.2g}')
print(f'  slow vs fast across sessions: r={R_SF:+.3f} '
      f'95% CI [{CI_SF[0]:+.2f}, {CI_SF[1]:+.2f}] (not what the claim rests on)')
print(f'  crossover median {np.median(CX):.1f} Hz in {len(CX)}/{NS} sessions')

# ============================== figure =========================================
# Portrait, sized to A4 with margins (~7 x 10 in). The previous 3-row layout --
# the example, then E-G across, then H-J across -- trimmed to 7.1 x 8.7 in under
# bbox_inches='tight', an aspect of only 1.2:1, so it sat squat on a portrait
# page with the six population panels squeezed three-to-a-row. Four rows with the
# population panels two-to-a-row fills the page and roughly doubles their width.
fig = plt.figure(figsize=(8.4, 5.9))
# left/right/top/bottom are pinned near the page edges because the save uses
# bbox_inches='tight', which discards the figure margins -- so the OUTER gridspec
# proportions, not figsize, decide the final aspect. With matplotlib's default
# margins (gridspec spanning ~78% of width and ~77% of height) a nominal 7x10
# figure trimmed to 6.6x8.4 in, an aspect of 1.28:1, which is why the first
# attempt at "portrait" barely changed anything.
# E and F moved up beside the example stack: A-D are tall and narrow (they share
# the trial axis and have to be stacked), which left the right half of that block
# empty while the spectra sat in a row of their own underneath. Four rows rather
# than five, and the page is wider than it is tall by less.
OUTER = fig.add_gridspec(2, 1, height_ratios=[2.45, 1.0], hspace=.46,
                         left=.10, right=.97, top=.965, bottom=.045)


def tidy(ax):
    ax.tick_params(labelsize=6.5)
    ax.spines[['top', 'right']].set_visible(False)


def lp(ax, s, x=-.085, y=1.0):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='right')


def shade(ax, lines=True):
    """Both states behind the traces, not just one.

    Shading only the non-anchored blocks leaves the anchored ones as bare white,
    which reads as 'no data' rather than as the other state. Colouring both says
    the session is always in one of two states, and lets panels A, C and D be
    read against the state without the reader tracking which colour means
    'something' and which means 'nothing'.
    """
    for st, col in ((major, ANCH_COLOR), (~major, NONANCH_COLOR)):
        d = np.diff(np.concatenate([[0], st.astype(int), [0]]))
        for s_, e_ in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            ax.axvspan(s_ - .5, e_ - .5, color=col, alpha=.15, linewidth=0,
                       zorder=0)
    if lines:
        for tr_ in TRANS:
            ax.axvline(tr_ - .5, color='0.35', ls='--', lw=.7, zorder=5)


# ---- A-D: the example, with E and F beside it ---------------------------------
# A-D take more of the width; E and F are a spectrum and a band summary and
# read perfectly well narrower, while the example spectrogram and the band
# traces do not.
TOP = OUTER[0].subgridspec(1, 2, width_ratios=[1.0, .52], wspace=.40)
GE = TOP[0].subgridspec(4, 2, height_ratios=[.42, 1.55, .95, .78],
                        width_ratios=[1, .022], hspace=.26, wspace=.02)
# generous hspace: E's lower axis carries tick labels and an x-axis label,
# and F carries a two-line title, so the two run into each other at the
# spacing the other rows use
# E's two axes share one frequency axis and belong together, so they are
# nested and set nearly touching; the wide gap goes between E and F, where an
# x label and a panel title have to fit. One flat 3-row grid with a single
# hspace could not do both, and spent a third of the column on white space
# between the spectrum and the deviation beneath it.
GEF = TOP[1].subgridspec(2, 1, height_ratios=[1.46, 1.02], hspace=.62)
GE2 = GEF[0].subgridspec(2, 1, height_ratios=[.46, 1.0], hspace=.12)
x = np.arange(len(nums))

ax = fig.add_subplot(GE[0, 0])
# drawn the way Figure 1 draws it: PC1 filled against zero, anchored colour
# above the line and non-anchored below, so the state reads off the fill
# rather than off a shaded background and a separate trace
ax.fill_between(x, 0, pc, where=pc >= 0, color=ANCH_COLOR, lw=0,
                interpolate=True)
ax.fill_between(x, 0, pc, where=pc < 0, color=NONANCH_COLOR, lw=0,
                interpolate=True)
ax.axhline(0, color='0.4', lw=.7)
ax.set_xlim(-.5, len(x) - .5); ax.set_xticks([])
ax.set_ylabel('population\naxis (PC1, z)', fontsize=7)
ax.set_title(f'M{MOUSE} D{DAY} — one session, trial by trial', fontsize=8.8,
             loc='left')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=ANCH_COLOR, label='anchored'),
                   Patch(facecolor=NONANCH_COLOR, label='non-anchored')],
          fontsize=5.6, frameon=False, ncol=2, loc='lower right',
          bbox_to_anchor=(1.0, 1.02), handlelength=1.0, columnspacing=.9,
          borderpad=.1)
tidy(ax); lp(ax, 'A', x=-.20)
# no bottom spine: the trial axis belongs to the panels beneath, and a spine
# with no ticks under a filled trace reads as a second zero line
ax.spines['bottom'].set_visible(False)
# the speed twin axis is gone: it was a second y scale on the panel that
# carries the state, and speed is not what this figure is about

ax = fig.add_subplot(GE[1, 0])
im = ax.pcolormesh(x, centres, ZB.T, cmap='RdBu_r', vmin=-1.4, vmax=1.4,
                   shading='gouraud', rasterized=True)
for tr_ in TRANS:
    ax.axvline(tr_ - .5, color='0.25', ls='--', lw=.7, zorder=5)
for lo, hi, c in [(6, 10, '#7b4173'), (30, 48, '#2f7d32'),
                  (60, 100, '#e8701a')]:
    ax.axhline(lo, color=c, lw=.5, alpha=.6); ax.axhline(hi, color=c, lw=.5, alpha=.6)
ax.set_yscale('log'); ax.set_yticks([2, 8, 20, 50, 100])
ax.set_yticklabels(['2', '8', '20', '50', '100'], fontsize=6.5)
ax.set_xticks([]); ax.set_xlim(-.5, len(x) - .5)
ax.set_ylabel('frequency (Hz)', fontsize=7)
cb = fig.colorbar(im, cax=fig.add_subplot(GE[1, 1]))
cb.set_label('power (z per frequency)', fontsize=5.6)
cb.ax.tick_params(labelsize=5.2); cb.outline.set_visible(False)
lp(ax, 'B', x=-.20)

ax = fig.add_subplot(GE[2, 0]); shade(ax)
for nm, lo, hi, c in BANDS:
    ax.plot(x, band_trace(lo, hi), color=c, lw=1.0, label=nm)
ax.axhline(0, color='0.6', lw=.7, ls=':')
ax.set_xlim(-.5, len(x) - .5); ax.set_xticks([])
ax.set_ylabel('band power (z)', fontsize=7)
# above the axes at the right: inside, the three entries sat on the traces in
# whichever block happened to be low at that point
ax.legend(fontsize=5.6, frameon=False, ncol=3, loc='lower right',
          bbox_to_anchor=(1.0, 1.01), handlelength=1.1, columnspacing=1.1,
          borderpad=0, handletextpad=.45)
tidy(ax); lp(ax, 'C', x=-.20)

ax = fig.add_subplot(GE[3, 0]); shade(ax)
# Slow gamma on its own axis: its MI is ~5x smaller than fast gamma's, so on a
# shared axis it is a flat line at the bottom and the reader cannot see that it
# does NOT fall -- which is the comparison the panel exists to make.
PAC_C = {'slow_gamma': '#2f7d32', 'fast_gamma': '#e8701a'}
ax.plot(pac_t, pac_v['fast_gamma'] * 1e3, color=PAC_C['fast_gamma'], lw=1.1)
ax.set_xlim(-.5, len(x) - .5)
ax.set_xlabel('trial', fontsize=7.5)
ax.set_ylabel('theta–fast-gamma\nMI $\\times 10^{3}$', fontsize=7,
              color=PAC_C['fast_gamma'])
ax.tick_params(axis='y', colors=PAC_C['fast_gamma'])
tidy(ax); lp(ax, 'D', x=-.20)
axp = ax.twinx()
axp.plot(pac_t, pac_v['slow_gamma'] * 1e3, color=PAC_C['slow_gamma'], lw=1.1,
         alpha=.9)
axp.set_ylabel('theta–slow-gamma\nMI $\\times 10^{3}$', fontsize=6.2,
               color=PAC_C['slow_gamma'])
axp.tick_params(labelsize=5.5, colors=PAC_C['slow_gamma'])
axp.spines[['top', 'left']].set_visible(False)
axp.spines['right'].set_color(PAC_C['slow_gamma'])
# No legend here on purpose: both y-axis labels carry their band in the trace's
# own colour, with matching tick colours, so a legend would repeat that in a
# panel short of vertical room and sitting on top of the traces.

# ---- E-J: the population, two panels per row ---------------------------------
# Reading order stays E F / G H / I J, so A-J still runs left-to-right then down.
GQ = OUTER[1].subgridspec(1, 2, wspace=.42, width_ratios=[.88, 1.62])
# H is two axes under one label: the geometry on the left, and the two
# derived contrasts it is a claim about drawn explicitly on the right.
GH = GQ[1].subgridspec(1, 2, wspace=.46, width_ratios=[1.0, .80])

# E is split in two because the state effect and the spectrum's own shape live
# on different scales, and one axis cannot show both. The raw whitened spectrum
# spans about 1.7 units -- the theta peak, the 1/f tail, the mains line -- while
# the state difference is about 0.1, so on the top axis the two states are
# necessarily superimposed. That is worth showing: it is what excludes a gain
# change, a referencing difference or a missing peak. The lower axis plots the
# same two states after session normalisation, which removes the shape and
# leaves the deviations, and there the same 0.1 is a third of the axis. Both
# panels are the two STATES, not their difference, so neither duplicates J.
axs = fig.add_subplot(GE2[0])
for st, col, lab in (('non', NONANCH_COLOR, 'non-anchored'),
                     ('anch', ANCH_COLOR, 'anchored')):
    g = R.groupby('freq')[st].agg(['mean', 'sem'])
    axs.fill_between(g.index, g['mean'] - g['sem'], g['mean'] + g['sem'],
                     color=col, alpha=.30, linewidth=0)
    axs.plot(g.index, g['mean'], color=col, lw=1.1, label=lab)
axs.set_xscale('log'); axs.set_xlim(1.9, 260)
axs.tick_params(labelbottom=False, labelsize=6.5)
# both labels kept short: the lower axis is the taller of the two, so its
# centred label runs the full height of the pair and collided with this one
axs.set_ylabel('spectrum', fontsize=6.8, labelpad=2)
axs.set_title(f'the two states, as spectra and as deviations\n'
              f'({NS} switching sessions)', fontsize=8)
# the whitened spectrum falls away above ~100 Hz, so the top right of this
# short axis is the only reliably empty corner in it
axs.legend(fontsize=5.6, frameon=False, loc='upper right', ncol=1,
           handlelength=1.1, handletextpad=.4, borderaxespad=.2,
           labelspacing=.25)
axs.text(.015, .05, 'same shape — no gain change', transform=axs.transAxes,
         fontsize=5.6, color='0.45', ha='left', va='bottom')
tidy(axs); lp(axs, 'E', x=-.17)

ax = fig.add_subplot(GE2[1], sharex=axs)
for st, col, lab in (('non', NONANCH_COLOR, 'non-anchored'),
                     ('anch', ANCH_COLOR, 'anchored')):
    g = W.groupby('freq')[st].agg(['mean', 'sem'])
    ax.fill_between(g.index, g['mean'] - g['sem'], g['mean'] + g['sem'],
                    color=col, alpha=.30, linewidth=0)
    ax.plot(g.index, g['mean'], color=col, lw=1.2)
ax.axhline(0, color='0.6', lw=.7, ls=':', zorder=0)
ax.axvspan(6, 10, color='0.88', alpha=.6, linewidth=0, zorder=0)
ax.axvspan(30, 48, color='0.88', alpha=.6, linewidth=0, zorder=0)
ax.axvspan(60, 100, color='0.88', alpha=.6, linewidth=0, zorder=0)
ax.set_xscale('log'); ax.set_xticks([2, 8, 20, 50, 100, 200])
ax.set_xticklabels(['2', '8', '20', '50', '100', '200'], fontsize=6.5)
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel('frequency (Hz)', fontsize=7.5)
ax.set_ylabel('deviation (z)', fontsize=7.5, labelpad=2)
ax.text(.985, .06, 'the bands F quantifies', transform=ax.transAxes,
        fontsize=5.8, color='0.45', ha='right')
tidy(ax)

ax = fig.add_subplot(GEF[1])
for j, (nm, lo, hi) in enumerate(PBANDS):
    g = pband(lo, hi).dropna()
    p = wilcoxon(g).pvalue
    c = SIG_C if p < .05 else '0.6'
    ax.bar(j, g.mean(), .62, color=c, linewidth=0)
    se = g.std() / np.sqrt(len(g))
    ax.plot([j, j], [g.mean() - se, g.mean() + se], color='0.3', lw=1)
    ax.annotate(f'{p:.0e}'.replace('e-0', 'e-') if p < .001 else f'{p:.3f}',
                (j, .99), xycoords=('data', 'axes fraction'), ha='center',
                va='top', fontsize=5.2, color=c, rotation=90)
ax.axhline(0, color='0.4', lw=1)
ax.set_xticks(range(len(PBANDS)))
ax.set_xticklabels([b[0].replace('\n', ' ') for b in PBANDS], fontsize=5.6,
                   rotation=38, ha='right')
ax.set_ylabel('anchored $-$ non (z)', fontsize=7.5)
ax.set_ylim(top=ax.get_ylim()[1] * 1.55)
ax.set_title('the difference changes SIGN — not a gain change', fontsize=7.6)
tidy(ax); lp(ax, 'F', x=-.17)

ax = fig.add_subplot(GQ[0])
w = .34
for j, b in enumerate(['slow_gamma', 'fast_gamma']):
    g = P[P.band == b].groupby(['mouse', 'day', 'state']).mi_excess.mean() \
        .unstack().dropna()
    p = wilcoxon(g.anch, g.non).pvalue
    for k, (st, col) in enumerate((('non', NONANCH_COLOR), ('anch', ANCH_COLOR))):
        v = g[st]
        ax.bar(j + (k - .5) * w, v.mean(), w, color=col, linewidth=0)
        se = v.std() / np.sqrt(len(v))
        ax.plot([j + (k - .5) * w] * 2, [v.mean() - se, v.mean() + se],
                color='0.3', lw=1)
    ax.annotate(f'{p:.0e}'.replace('e-0', 'e-') if p < .001 else f'p={p:.2f}',
                (j, max(g.anch.mean(), g.non.mean()) * 1.10), ha='center',
                fontsize=5.6, color=SIG_C if p < .05 else '0.5')
ax.set_xticks([0, 1]); ax.set_xticklabels(['slow\n30–55*', 'fast\n60–100'],
                                          fontsize=6.2)
ax.set_ylim(0, .0034)
ax.set_ylabel('theta–gamma MI\nover shuffle null', fontsize=7.5)
ax.set_title('only FAST gamma\nuncouples', fontsize=8)
ax.text(.5, -.34, '* contains mains', transform=ax.transAxes, fontsize=5,
        color='0.5', ha='center')
tidy(ax); lp(ax, 'G', x=-.17)

ax = fig.add_subplot(GH[0])
# What this panel used to ask -- are the two changes CORRELATED across
# sessions? -- cannot answer the question it was named after, and with n = 26
# its null had a 95% CI of roughly -0.33 to +0.44 behind it. The quadrant
# count that titled it was a dichotomised restatement of the same null, so two
# reports of one weak test read as two pieces of evidence. The claim is
# instead tested pairwise, within session: equal and opposite means the SUM is
# zero while the DIFFERENCE is not, which is what the line and the centroid
# show.
_lim = max(abs(np.r_[TR.slow, TR.fast]).max() * 1.12, .08)
ax.plot([-_lim, _lim], [_lim, -_lim], ls='--', lw=.9, color='0.55', zorder=1)
ax.axhline(0, color='0.85', lw=.8, zorder=0)
ax.axvline(0, color='0.85', lw=.8, zorder=0)
# No per-point residual segments: the sum is zero on AVERAGE, not session by
# session (SD about 0.3), and drawing 26 long perpendiculars made that spread
# -- most of which is measurement noise in two independent band estimates --
# look like a failure of the line. The cloud and the centroid carry it.
ax.scatter(TR.slow, TR.fast, s=15, color=SIG_C, alpha=.85, lw=0, zorder=3)
_mx, _my = TR.slow.mean(), TR.fast.mean()
ax.errorbar(_mx, _my, xerr=TR.slow.std() / np.sqrt(len(TR)),
            yerr=TR.fast.std() / np.sqrt(len(TR)), fmt='D', ms=4.5,
            color='0.15', ecolor='0.15', elinewidth=1.1, capsize=2, zorder=5)
ax.set_xlim(-_lim, _lim); ax.set_ylim(-_lim, _lim)
ax.set_xlabel('slow gamma change (z)', fontsize=7.5)
ax.set_ylabel('fast gamma change (z)', fontsize=7.5)
ax.set_title('power moves from fast to slow,\nwith no net change', fontsize=8)
ax.text(.045, .76, 'equal and\nopposite', transform=ax.transAxes,
        fontsize=5.4, color='0.5', ha='left', va='top', linespacing=1.25)
tidy(ax); lp(ax, 'H', x=-.22)

# ---- the same claim, drawn as the two contrasts it rests on -----------------
# The numbers used to sit in the corner of the scatter, which left the reader
# to take the central result on trust. Each session contributes one point to
# each column, so the spread that the scatter shows along and across the line
# is here as the spread of the quantities actually tested.
ax = fig.add_subplot(GH[1])
_jr = np.random.default_rng(7)
COLS_H = [('slow\n$-$ fast', TR['shift'], P_SHIFT, SIG_C),
          ('slow\n$+$ fast', TR['sum'], P_SUM, '0.55'),
          ('2\u2013200\nHz', TR['broad'], P_BROAD, '0.55')]
ax.axhline(0, color='0.4', lw=.9, zorder=1)
for j, (nm, v, pv, col) in enumerate(COLS_H):
    ax.scatter(j + _jr.uniform(-.17, .17, len(v)), v, s=7, color=col,
               alpha=.55, lw=0, zorder=2)
    se = v.std() / np.sqrt(len(v))
    ax.plot([j - .30, j + .30], [v.mean()] * 2, color='0.12', lw=1.6,
            zorder=4, solid_capstyle='butt')
    ax.plot([j, j], [v.mean() - se, v.mean() + se], color='0.12', lw=1.6,
            zorder=4)
    ax.annotate(f'{pv:.0e}'.replace('e-0', 'e-') if pv < .001 else f'{pv:.2f}',
                (j, .995), xycoords=('data', 'axes fraction'), ha='center',
                va='top', fontsize=5.6,
                color=SIG_C if pv < .05 else '0.5')
ax.set_xticks(range(3))
ax.set_xticklabels([c[0] for c in COLS_H], fontsize=5.9,
                   linespacing=1.25)
ax.set_xlim(-.6, 2.6)
ax.set_ylim(top=max(v.max() for _, v, _, _ in COLS_H) * 1.42)
ax.set_ylabel('z, anchored $-$ non', fontsize=7.5)
ax.set_title('the difference is reliable;\nthe sum and the broadband are not',
             fontsize=8)
tidy(ax)

# Panels I and J were removed. I ranked the per-session slow-minus-fast
# shift, which the population panel already carries, and J plotted the
# crossover profile -- a restatement of E at lower resolution. The
# crossover number itself is quoted in the text; it is printed above.

fig.savefig(OUT, bbox_inches='tight', dpi=220)
print(f'\nwrote {OUT}')