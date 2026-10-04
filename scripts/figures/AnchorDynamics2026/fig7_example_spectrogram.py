"""One session, trial by trial: the spectrum and theta-gamma coupling against the state.

The population results in fig7_spectrum_pac are averages over sessions and states,
and they say nothing about whether the spectral changes actually track the state
as it moves. This does: a single session's trial-resolved spectrogram, band
powers and theta-gamma coupling, drawn against the population anchoring state on
one shared trial axis.

M26 D18 is the example. It is chosen because it shows all four population effects
in the same direction -- theta down, slow gamma up, fast gamma down, fast-gamma
coupling down when anchored -- on the largest number of speed-matched trials in
the dataset (124 per state), and because it is already the Figure 2A example, so
a reader meets the same session twice rather than a new one.

That choice is a selection and should be read as one: it is the clearest case,
not a typical one. The population figure beside it is what establishes that the
effects hold across sessions.

The spectrogram is z-scored PER FREQUENCY across trials, which is the only way
the gamma range is visible at all beside theta -- raw power falls by two orders
of magnitude across the plotted range. Coupling is computed in a sliding window
of PAC_WIN trials, because a single trial carries only a few seconds of running
and the modulation index needs far more than that to be stable.

Writes fig7_example_spectrogram.pdf
"""
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

sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lfp_batch import FS, LFP_ROOT, clip_trials, mec_groups, nap, vr_paths
from lfp_spectrum_pac import GAMMA, _bp, tort_mi
from spatial_manifolds.anchoring import ANCH_COLOR, NONANCH_COLOR

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = os.path.dirname(os.path.abspath(__file__))
OUT = f'{FIG}/fig7_example_spectrogram.pdf'

MOUSE, DAY = 26, 18
# 1024 at 1 kHz gives ~0.98 Hz bins. At 512 the resolution is 1.95 Hz, so the
# lowest frequency Welch returns above 2 Hz is 3.9 -- every log band below that
# came back empty and the bottom of the spectrogram was blank. The cost is that
# a segment is now 1.024 s long, so MIN_RUN_S rises to 2 s to keep at least two
# segments per trial; almost every trial clears that, since 200 cm at ~45 cm/s
# is over 4 s of running.
NPERSEG = 1024
# 1.9, not 2.0: with 1.024 s segments Welch returns bins every 0.977 Hz, so the
# lowest one at or above 2 Hz is 2.93 and a floor of 2.0 left the bottom of the
# spectrogram blank. Starting at 1.9 admits the 1.953 Hz bin, which is real data,
# and the axis then reaches 2 Hz with something in it.
FLO, FHI = 1.9, 150.
PAC_WIN = 9                 # trials per coupling estimate
N_LOGBINS = 56              # log-spaced frequency bands for the spectrogram
TRIAL_SIGMA = 1.8           # trials; smoothing along the time axis only
SMOOTH = 5                  # trials, for the band-power traces
MIN_RUN_S = 2.0
LINE = [(48, 52), (98, 102)]
# slow gamma is NOT drawn in a teal: the non-anchored shading is teal, and a
# band trace in the same hue reads as part of the background
BANDS = [('theta 6–10', 6, 10, '#7b4173'),
         ('slow gamma 30–48', 30, 48, '#2b6cb0'),
         ('fast gamma 60–100', 60, 100, '#c04744')]
MAJOR_FILT = 9              # trials, odd; the Figure 1/2 major-transition rule

lf = f'{LFP_ROOT}/M{MOUSE}/D{DAY}/VR/sub-M{MOUSE}_ses-D{DAY}_typ-VR_beh.nwb'
f = h5py.File(lf, 'r')
names = [n.decode() for n in f['general/extracellular_ephys/electrodes/channel_name'][:]]
M = mec_groups(MOUSE, DAY, names)
cols = M.g.values
rc = np.sort(cols); back = np.array([int(np.where(rc == c)[0][0]) for c in cols])
D = f['processing/ecephys/LFP/lfp/data'][:, rc][:, back].astype(np.float32)
PH = f['processing/ecephys/Processed/theta/data'][:, rc][:, back].astype(np.float32)
f.close()

bp, cp = vr_paths(MOUSE, DAY)
beh = nap.load_file(bp); clusters = nap.load_file(cp)
trials, _ = clip_trials(beh['trials'].as_dataframe(), clusters)
T = pd.read_csv(f'{ROOT}/data/population_state/anchoring_trials.csv')
st = T[(T.mouse == MOUSE) & (T.day == DAY)]
frac = dict(zip(st.trial.astype(int), st.frac_anch))
pc1 = dict(zip(st.trial.astype(int), st.pc1))

t = np.arange(D.shape[0]) / FS
S_ = beh['S']
spd = np.asarray(S_.values)[np.searchsorted(np.asarray(S_.index), t)
                            .clip(0, len(S_) - 1)]
run = spd >= 3.0

env = {}
for nm, lo, hi in GAMMA:
    b, a = _bp(lo, hi)
    env[nm] = np.abs(hilbert(filtfilt(b, a, D, axis=0), axis=0))

spec, nums, masks, tspd = [], [], [], []
for _, tr in trials.iterrows():
    u = int(tr.number)
    if u not in frac:
        continue
    m = (t >= tr.start) & (t <= tr.end) & run
    if m.sum() < MIN_RUN_S * FS:
        continue
    fr, P = welch(D[m], fs=FS, nperseg=NPERSEG, axis=0)
    sel = (fr >= FLO) & (fr <= FHI)
    spec.append(P[sel].mean(axis=1))
    nums.append(u); masks.append(m); tspd.append(float(spd[m].mean()))
fr_sel = fr[sel]
A = np.array(spec)                                   # trials x freq
nums = np.array(nums)
fa = np.array([frac[u] for u in nums])
pc = np.array([pc1[u] for u in nums])
pc = (pc - pc.mean()) / (pc.std() or 1)
tspd = np.array(tspd)

# TWO state variables, used for different jobs and not interchangeable.
# `anch` is the per-trial state, and is what the band statistics below use so
# that this example is scored the same way as the population figure. `major` is
# that state after the 9-trial median filter of Figures 1 and 2, and is used for
# the block shading and the transition lines -- the blocks a reader should group
# by, with short flickers removed so they do not cover the panel in lines.
anch = fa > .5
major = median_filter(anch.astype(float), size=MAJOR_FILT, mode='nearest') > .5
TRANS = np.where(np.diff(major.astype(int)) != 0)[0] + 1

L = np.log10(A + 1e-20)
Z = (L - L.mean(0)) / (L.std(0) + 1e-12)
notmains = np.array([not any(lo <= x <= hi for lo, hi in LINE) for x in fr_sel])

# ---- the spectrogram, rebinned onto LOG-SPACED frequency bands ---------------
# Welch returns LINEARLY spaced frequencies, and the panel plots them on a log
# axis. That oversamples the top of the range by an order of magnitude -- above
# 50 Hz dozens of independent, noisy bins are squeezed into a few pixels, which
# is the speckle, while below 10 Hz each bin is stretched over many. Averaging
# into log-spaced bands puts a comparable number of original bins behind every
# plotted row, so the smoothing is uniform in the space actually displayed
# rather than applied blindly.
#
# Mains bins are dropped before averaging, so the 50 Hz line does not paint a
# stripe across the image, and the band containing it is simply made of its
# remaining neighbours.
EDGES = np.geomspace(FLO, FHI, N_LOGBINS + 1)
rows, centres = [], []
for lo, hi in zip(EDGES[:-1], EDGES[1:]):
    k = (fr_sel >= lo) & (fr_sel < hi) & notmains
    if k.sum() == 0:
        continue
    rows.append(L[:, k].mean(1))
    centres.append(np.sqrt(lo * hi))
LB = np.array(rows).T                                  # trials x logband
ZB = (LB - LB.mean(0)) / (LB.std(0) + 1e-12)
# smoothing along TRIALS only: the frequency axis has already been averaged, and
# smoothing it again would blur the band boundaries the figure is drawing
ZB = gaussian_filter1d(ZB, TRIAL_SIGMA, axis=0, mode='nearest')
centres = np.array(centres)


def band_trace(lo, hi):
    k = (fr_sel >= lo) & (fr_sel < hi) & notmains
    return uniform_filter1d(Z[:, k].mean(1), SMOOTH, mode='nearest')


# sliding-window coupling: a trial is a few seconds of running, far too little
# for a stable modulation index, so windows of PAC_WIN consecutive trials are
# concatenated and the window is slid by one trial
pac_t, pac_v = [], []
for i in range(len(nums) - PAC_WIN + 1):
    m = np.zeros(len(t), bool)
    for k in range(i, i + PAC_WIN):
        m |= masks[k]
    if m.sum() < 15 * FS:
        continue
    vals = [tort_mi(PH[m, j], env['fast_gamma'][m, j]) for j in range(D.shape[1])]
    pac_t.append(i + PAC_WIN // 2)
    pac_v.append(np.nanmean(vals))
pac_t, pac_v = np.array(pac_t), np.array(pac_v)

print(f'M{MOUSE} D{DAY}: {len(nums)} trials, {int(anch.sum())} anchored, '
      f'{len(pac_t)} coupling windows')
for nm, lo, hi, _ in BANDS:
    v = band_trace(lo, hi)
    print(f'  {nm:18s} anch {v[anch].mean():+.3f}  non {v[~anch].mean():+.3f}')
print(f'  fast-gamma coupling  anch {np.nanmean(pac_v[anch[pac_t]]):.5f}  '
      f'non {np.nanmean(pac_v[~anch[pac_t]]):.5f}')
print(f'  running speed        anch {tspd[anch].mean():.1f}  '
      f'non {tspd[~anch].mean():.1f} cm/s')
print(f'  {len(TRANS)} major transitions ({MAJOR_FILT}-trial median filter)')

# ============================== figure =========================================
fig = plt.figure(figsize=(7.4, 5.4))
# the colourbar gets its own column rather than being taken out of panel B:
# `fig.colorbar(ax=ax)` steals width from that axes alone, which left B narrower
# than A, C and D and broke the shared trial axis
G = fig.add_gridspec(4, 2, height_ratios=[.42, 1.55, .95, .78],
                     width_ratios=[1, .022], hspace=.26, wspace=.02)
x = np.arange(len(nums))


def shade(ax, lines=True):
    """Major non-anchored blocks, plus the transitions, on every panel."""
    d = np.diff(np.concatenate([[0], (~major).astype(int), [0]]))
    for s_, e_ in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        ax.axvspan(s_ - .5, e_ - .5, color=NONANCH_COLOR, alpha=.17, linewidth=0,
                   zorder=0)
    if lines:
        for tr_ in TRANS:
            ax.axvline(tr_ - .5, color='0.35', ls='--', lw=.7, zorder=5)


# A: the population state, with running speed beside it.
# This example is NOT speed-matched -- the population figure is -- so the speed
# trace is shown rather than asserted: theta and gamma both scale with speed, and
# a reader has to be able to see that the spectral changes below are not simply
# the animal running differently.
ax = fig.add_subplot(G[0, 0])
ax.plot(x, pc, color='0.2', lw=1)
ax.axhline(0, color='0.6', ls=':', lw=.8)
shade(ax)
ax.set_xlim(-.5, len(x) - .5)
ax.set_xticks([]); ax.tick_params(labelsize=6.5)
ax.set_ylabel('population\naxis (PC1, z)', fontsize=7.5)
axs = ax.twinx()
axs.plot(x, uniform_filter1d(tspd, SMOOTH, mode='nearest'), color='#b8860b',
         lw=.8, alpha=.85)
axs.set_ylabel('speed\n(cm/s)', fontsize=6.5, color='#b8860b')
axs.tick_params(labelsize=5.5, colors='#b8860b')
axs.spines[['top', 'left']].set_visible(False)
axs.spines['right'].set_color('#b8860b')
ax.set_title(f'M{MOUSE} D{DAY} — the spectrum follows the population state',
             fontsize=9, loc='left')
ax.spines[['top', 'right']].set_visible(False)
ax.text(-.085, 1.0, 'A', transform=ax.transAxes, fontsize=10, weight='bold',
        va='bottom', ha='right')

# B: the spectrogram
ax = fig.add_subplot(G[1, 0])
im = ax.pcolormesh(x, centres, ZB.T, cmap='RdBu_r', vmin=-1.4, vmax=1.4,
                   shading='gouraud', rasterized=True)
for tr_ in TRANS:
    ax.axvline(tr_ - .5, color='0.25', ls='--', lw=.7, zorder=5)
ax.set_yscale('log')
ax.set_yticks([2, 8, 20, 50, 100])
ax.set_yticklabels(['2', '8', '20', '50', '100'], fontsize=6.5)
for lo, hi, c in [(6, 10, '#7b4173'), (30, 48, '#2E7D6C'), (60, 100, '#c04744')]:
    ax.axhline(lo, color=c, lw=.6, alpha=.65)
    ax.axhline(hi, color=c, lw=.6, alpha=.65)
ax.set_xticks([])
ax.set_xlim(-.5, len(x) - .5)
ax.set_ylabel('frequency (Hz)', fontsize=7.5)
cax = fig.add_subplot(G[1, 1])
cb = fig.colorbar(im, cax=cax)
cb.set_label('power (z per frequency)', fontsize=6)
cb.ax.tick_params(labelsize=5.5)
cb.outline.set_visible(False)
ax.text(-.085, 1.0, 'B', transform=ax.transAxes, fontsize=10, weight='bold',
        va='bottom', ha='right')

# C: the three bands
ax = fig.add_subplot(G[2, 0])
shade(ax)
for nm, lo, hi, c in BANDS:
    ax.plot(x, band_trace(lo, hi), color=c, lw=1.0, label=nm)
ax.axhline(0, color='0.6', lw=.7, ls=':')
ax.set_xlim(-.5, len(x) - .5)
ax.set_xticks([]); ax.tick_params(labelsize=6.5)
ax.set_ylabel('band power (z)', fontsize=7.5)
ax.legend(fontsize=5.8, frameon=False, ncol=3, loc='lower left',
          handlelength=1.1, columnspacing=1.1)
ax.spines[['top', 'right']].set_visible(False)
ax.text(-.085, 1.0, 'C', transform=ax.transAxes, fontsize=10, weight='bold',
        va='bottom', ha='right')

# D: theta-nested fast gamma
ax = fig.add_subplot(G[3, 0])
shade(ax)
ax.plot(pac_t, pac_v * 1e3, color='#c04744', lw=1.1)
ax.set_xlim(-.5, len(x) - .5)
ax.set_xlabel('trial', fontsize=7.5)
ax.tick_params(labelsize=6.5)
ax.set_ylabel('theta–fast-gamma\nMI $\\times 10^{3}$', fontsize=7.5)
ax.spines[['top', 'right']].set_visible(False)
ax.text(-.085, 1.0, 'D', transform=ax.transAxes, fontsize=10, weight='bold',
        va='bottom', ha='right')

fig.savefig(OUT, bbox_inches='tight', dpi=220)
print(f'\nwrote {OUT}')
