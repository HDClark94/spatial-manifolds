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
from scipy.stats import fisher_exact, spearmanr, wilcoxon

sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lfp_batch import FS, LFP_ROOT, clip_trials, mec_groups, nap, vr_paths
from lfp_spectrum_pac import GAMMA, _bp, tort_mi
from spatial_manifolds.anchoring import ANCH_COLOR, NONANCH_COLOR

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
GRID_C = '#c04744'
LINE = [(48, 52), (98, 102), (148, 152)]
BANDS = [('theta 6–10', 6, 10, '#7b4173'),
         ('slow gamma 30–48', 30, 48, '#2b6cb0'),
         ('fast gamma 60–100', 60, 100, '#c04744')]
PBANDS = [('delta', 2, 5), ('theta', 6, 10), ('beta', 12, 20),
          ('slow\ngamma', 30, 48), ('mid\ngamma', 60, 100),
          ('high\ngamma', 100, 150), ('HFO', 150, 250)]


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
major = median_filter(anch.astype(float), size=MAJOR_FILT, mode='nearest') > .5
TRANS = np.where(np.diff(major.astype(int)) != 0)[0] + 1

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
print(f'  shift index {TR["shift"].mean():+.3f} p={wilcoxon(TR["shift"]).pvalue:.2g}')
print(f'  slow vs fast across sessions: rho={spearmanr(TR.slow, TR.fast)[0]:+.3f} '
      f'p={spearmanr(TR.slow, TR.fast)[1]:.2g}')
print(f'  crossover median {np.median(CX):.1f} Hz in {len(CX)}/{NS} sessions')

# ============================== figure =========================================
# Portrait, sized to A4 with margins (~7 x 10 in). The previous 3-row layout --
# the example, then E-G across, then H-J across -- trimmed to 7.1 x 8.7 in under
# bbox_inches='tight', an aspect of only 1.2:1, so it sat squat on a portrait
# page with the six population panels squeezed three-to-a-row. Four rows with the
# population panels two-to-a-row fills the page and roughly doubles their width.
fig = plt.figure(figsize=(7.0, 10.0))
# left/right/top/bottom are pinned near the page edges because the save uses
# bbox_inches='tight', which discards the figure margins -- so the OUTER gridspec
# proportions, not figsize, decide the final aspect. With matplotlib's default
# margins (gridspec spanning ~78% of width and ~77% of height) a nominal 7x10
# figure trimmed to 6.6x8.4 in, an aspect of 1.28:1, which is why the first
# attempt at "portrait" barely changed anything.
OUTER = fig.add_gridspec(4, 1, height_ratios=[2.45, 1.0, 1.0, 1.0], hspace=.46,
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


# ---- A-D: the example --------------------------------------------------------
GE = OUTER[0].subgridspec(4, 2, height_ratios=[.42, 1.55, .95, .78],
                          width_ratios=[1, .022], hspace=.26, wspace=.02)
x = np.arange(len(nums))

ax = fig.add_subplot(GE[0, 0])
ax.plot(x, pc, color='0.2', lw=1); ax.axhline(0, color='0.6', ls=':', lw=.8)
shade(ax)
ax.set_xlim(-.5, len(x) - .5); ax.set_xticks([])
ax.set_ylabel('population\naxis (PC1, z)', fontsize=7)
ax.set_title(f'M{MOUSE} D{DAY} — one session, trial by trial', fontsize=8.8,
             loc='left')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=ANCH_COLOR, alpha=.35, label='anchored'),
                   Patch(facecolor=NONANCH_COLOR, alpha=.35,
                         label='non-anchored')],
          fontsize=5.6, frameon=False, ncol=2, loc='lower right',
          bbox_to_anchor=(1.0, 1.02), handlelength=1.0, columnspacing=.9,
          borderpad=.1)
tidy(ax); lp(ax, 'A')
axs = ax.twinx()
axs.plot(x, uniform_filter1d(tspd, SMOOTH, mode='nearest'), color='#b8860b',
         lw=.8, alpha=.85)
axs.set_ylabel('speed\n(cm/s)', fontsize=6.2, color='#b8860b')
axs.tick_params(labelsize=5.5, colors='#b8860b')
axs.spines[['top', 'left']].set_visible(False)
axs.spines['right'].set_color('#b8860b')

ax = fig.add_subplot(GE[1, 0])
im = ax.pcolormesh(x, centres, ZB.T, cmap='RdBu_r', vmin=-1.4, vmax=1.4,
                   shading='gouraud', rasterized=True)
for tr_ in TRANS:
    ax.axvline(tr_ - .5, color='0.25', ls='--', lw=.7, zorder=5)
for lo, hi, c in [(6, 10, '#7b4173'), (30, 48, '#2b6cb0'), (60, 100, '#c04744')]:
    ax.axhline(lo, color=c, lw=.5, alpha=.6); ax.axhline(hi, color=c, lw=.5, alpha=.6)
ax.set_yscale('log'); ax.set_yticks([2, 8, 20, 50, 100])
ax.set_yticklabels(['2', '8', '20', '50', '100'], fontsize=6.5)
ax.set_xticks([]); ax.set_xlim(-.5, len(x) - .5)
ax.set_ylabel('frequency (Hz)', fontsize=7)
cb = fig.colorbar(im, cax=fig.add_subplot(GE[1, 1]))
cb.set_label('power (z per frequency)', fontsize=5.6)
cb.ax.tick_params(labelsize=5.2); cb.outline.set_visible(False)
lp(ax, 'B')

ax = fig.add_subplot(GE[2, 0]); shade(ax)
for nm, lo, hi, c in BANDS:
    ax.plot(x, band_trace(lo, hi), color=c, lw=1.0, label=nm)
ax.axhline(0, color='0.6', lw=.7, ls=':')
ax.set_xlim(-.5, len(x) - .5); ax.set_xticks([])
ax.set_ylabel('band power (z)', fontsize=7)
ax.legend(fontsize=5.6, frameon=False, ncol=3, loc='lower left',
          handlelength=1.1, columnspacing=1.1)
tidy(ax); lp(ax, 'C')

ax = fig.add_subplot(GE[3, 0]); shade(ax)
# Slow gamma on its own axis: its MI is ~5x smaller than fast gamma's, so on a
# shared axis it is a flat line at the bottom and the reader cannot see that it
# does NOT fall -- which is the comparison the panel exists to make.
PAC_C = {'slow_gamma': '#2b6cb0', 'fast_gamma': '#c04744'}
ax.plot(pac_t, pac_v['fast_gamma'] * 1e3, color=PAC_C['fast_gamma'], lw=1.1)
ax.set_xlim(-.5, len(x) - .5)
ax.set_xlabel('trial', fontsize=7.5)
ax.set_ylabel('theta–fast-gamma\nMI $\\times 10^{3}$', fontsize=7,
              color=PAC_C['fast_gamma'])
ax.tick_params(axis='y', colors=PAC_C['fast_gamma'])
tidy(ax); lp(ax, 'D')
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
GP = OUTER[1].subgridspec(1, 2, wspace=.42, width_ratios=[1.25, 1.0])
GQ = OUTER[2].subgridspec(1, 2, wspace=.42, width_ratios=[1.0, 1.15])
GT = OUTER[3].subgridspec(1, 2, wspace=.42, width_ratios=[1.0, 1.0])

ax = fig.add_subplot(GP[0])
for st, col, lab in (('non', NONANCH_COLOR, 'non-anchored'),
                     ('anch', ANCH_COLOR, 'anchored')):
    g = R.groupby('freq')[st].agg(['mean', 'sem'])
    ax.fill_between(g.index, g['mean'] - g['sem'], g['mean'] + g['sem'],
                    color=col, alpha=.30, linewidth=0)
    ax.plot(g.index, g['mean'], color=col, lw=1.2, label=lab)
ax.axvspan(6, 10, color='0.88', alpha=.6, linewidth=0, zorder=0)
ax.axvspan(30, 48, color='0.88', alpha=.6, linewidth=0, zorder=0)
ax.set_xscale('log'); ax.set_xticks([2, 8, 20, 50, 100, 200])
ax.set_xticklabels(['2', '8', '20', '50', '100', '200'], fontsize=6.5)
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel('frequency (Hz)', fontsize=7.5)
ax.set_ylabel('session-normalised\nlog (power $\\times$ freq)', fontsize=7.5)
ax.set_title(f'the two states, as spectra\n({NS} switching sessions)', fontsize=8)
ax.legend(fontsize=6, frameon=False, loc='lower left')
tidy(ax); lp(ax, 'E', x=-.17)

ax = fig.add_subplot(GP[1])
for j, (nm, lo, hi) in enumerate(PBANDS):
    g = pband(lo, hi).dropna()
    p = wilcoxon(g).pvalue
    c = GRID_C if p < .05 else '0.6'
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
ax.set_title('the difference changes SIGN,\nso it is not a gain change', fontsize=8)
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
                fontsize=5.6, color=GRID_C if p < .05 else '0.5')
ax.set_xticks([0, 1]); ax.set_xticklabels(['slow\n30–55*', 'fast\n60–100'],
                                          fontsize=6.2)
ax.set_ylim(0, .0034)
ax.set_ylabel('theta–gamma MI\nover shuffle null', fontsize=7.5)
ax.set_title('only FAST gamma\nuncouples', fontsize=8)
ax.text(.5, -.34, '* contains mains', transform=ax.transAxes, fontsize=5,
        color='0.5', ha='center')
tidy(ax); lp(ax, 'G', x=-.17)

ax = fig.add_subplot(GQ[1])
ax.axhline(0, color='0.6', lw=.8); ax.axvline(0, color='0.6', lw=.8)
ax.axhspan(ax.get_ylim()[0], 0, xmin=.5, color=ANCH_COLOR, alpha=.07, linewidth=0)
ax.scatter(TR.slow, TR.fast, s=16, color=GRID_C, alpha=.8, lw=0, zorder=3)
ax.set_xlabel('slow gamma change (z)', fontsize=7.5)
ax.set_ylabel('fast gamma change (z)', fontsize=7.5)
rho, prho = spearmanr(TR.slow, TR.fast)
# Title the null, not the count. 16/26 in the quadrant is what 21/26 up and
# 19/26 down already imply (expected 15.3, Fisher p = 0.59), so advertising the
# count invites it to be read as independent evidence of a coupled shift.
ax.set_title(f'{BOTH}/{len(TR)} in the slow-up/fast-down\nquadrant — '
             f'{EXP_QUAD:.0f} expected, p = {P_QUAD:.2f}', fontsize=8)
ax.annotate(f'$\\rho$ = {rho:+.2f}, p = {prho:.2f}\n(the two changes are not coupled)',
            (.03, .03), xycoords='axes fraction', fontsize=5.6, color='0.35',
            va='bottom')
tidy(ax); lp(ax, 'H', x=-.17)

ax = fig.add_subplot(GT[0])
v = TR['shift'].sort_values().values
ax.bar(np.arange(len(v)), v, .82,
       color=[GRID_C if y > 0 else '0.65' for y in v], linewidth=0)
ax.axhline(0, color='0.4', lw=1)
ax.set_xlabel('session (sorted)', fontsize=7.5)
ax.set_ylabel('slow $-$ fast change (z)', fontsize=7.5)
ax.set_title(f'positive in {int((TR["shift"] > 0).sum())}/{len(TR)},\n'
             f'p = {wilcoxon(TR["shift"]).pvalue:.0e}'.replace('e-0', 'e-'),
             fontsize=8)
tidy(ax); lp(ax, 'I', x=-.17)

ax = fig.add_subplot(GT[1])
# SEM across sessions, not across frequency bins: the question the panel answers
# is whether the SESSIONS agree on where the difference changes sign, so the
# spread that matters is between sessions at each frequency. Without the band a
# reader cannot tell whether the crossover is resolved or whether the profile is
# one noisy curve that happens to cross zero.
prof = Wok.groupby('freq').d.agg(['mean', 'sem'])
m = (prof.index >= 20) & (prof.index <= 120)
pm, ps = prof.loc[m, 'mean'], prof.loc[m, 'sem']
ax.fill_between(prof.index[m], pm - ps, pm + ps, color=GRID_C, alpha=.25,
                linewidth=0, edgecolor='none')
ax.plot(prof.index[m], pm, color=GRID_C, lw=1.2)
ax.axhline(0, color='0.4', lw=1, ls='--')
ax.set_xscale('log'); ax.set_xticks([20, 40, 60, 100])
ax.set_xticklabels(['20', '40', '60', '100'], fontsize=6.5)
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel('frequency (Hz)', fontsize=7.5)
ax.set_ylabel('anchored $-$ non (z)', fontsize=7.5)
ax.set_title(f'crossover at a median\n{np.median(CX):.0f} Hz, in {len(CX)}/{NS} '
             'sessions', fontsize=8)
tidy(ax); lp(ax, 'J', x=-.17)

fig.savefig(OUT, bbox_inches='tight', dpi=220)
print(f'\nwrote {OUT}')
