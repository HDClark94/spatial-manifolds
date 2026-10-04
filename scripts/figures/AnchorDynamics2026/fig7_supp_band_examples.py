"""Figure 7 supplement: the band changes in individual sessions.

The main figure carries one example (M26 D18) and then the population average.
A single example cannot show whether the population trend is a few strong
sessions or a general tendency, and the average cannot show what a session
actually looks like. This fills that gap: the same band traces, for several
sessions, against the population state.

HOW THE SESSIONS WERE CHOSEN, which matters for how the figure is read.
Panels A-C are sessions that DO show the population pattern -- theta down, slow
gamma up, fast gamma down when anchored -- and they were selected for that. They
are illustrations of the trend, not evidence for it; the evidence is the
population panel in the main figure, where the pattern holds in 16 of 26
sessions for slow-up-and-fast-down jointly and 21 of 26 for the slow-minus-fast
shift index.

PANEL D IS A SESSION THAT DOES NOT SHOW IT, and is included deliberately. A
supplement made only of confirming examples would misrepresent a 16/26 result as
if it were 26/26, and the honest way to show a tendency is to show its
exceptions alongside it. M27 D19 switches 16 times and its slow gamma moves the
WRONG way (-0.17), so it is a fair counterexample rather than a quiet session.

SESSIONS ARE ALSO REQUIRED TO SWITCH REPEATEDLY. An earlier version picked purely
on effect size and drew five sessions that each had a single transition: two
blocks that differ, which is what the population panel already shows, and not
power following the state. Every row here has at least 8 major transitions.

Each example: the population axis with major transitions, the trial-resolved
spectrogram, then theta, slow gamma and fast gamma power across trials, with both
states shaded. Examples are laid out two per column. Band powers are
z-scored within session and per frequency, as in the main figure, and the mains
harmonics are excluded from every band.

Writes fig7_supp_band_examples.pdf
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
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d
from scipy.signal import welch

sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lfp_batch import FS, LFP_ROOT, clip_trials, mec_groups, nap, vr_paths
from spatial_manifolds.anchoring import ANCH_COLOR, NONANCH_COLOR

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = os.path.dirname(os.path.abspath(__file__))
OUT = f'{FIG}/fig7_supp_band_examples.pdf'

# The TRACES are drawn from every trial, because a time series cannot skip the
# unmatched ones and stay contiguous. The NUMBERS printed on each row come from
# the speed-matched population table instead, so a reader comparing a row against
# the main figure is comparing like with like. Recomputing them unmatched here
# gave visibly different values -- M20 D24 theta reads -0.06 matched and +0.02
# unmatched -- which is exactly the sort of quiet inconsistency between a
# supplement and its parent figure that is worth not having.
# Read the WORKER SHARDS, not the merged spectrum_by_anchoring.csv, because that
# is what fig7_theta_gamma.py reads and there is no merge step tying the two
# together: lfp_spectrum_pac.py writes shards when given worker arguments and the
# un-suffixed file when run single-threaded, so the merged file is whatever a
# previous run happened to leave behind. The two are identical today (28
# sessions, 14,280 rows each); nothing enforces that, and a re-run of the shards
# without a matching single-worker run would leave this supplement quietly
# disagreeing with its parent figure.
_shards = sorted(glob.glob(f'{ROOT}/data/lfp/spectrum_by_anchoring_w*.csv'))
if not _shards:
    sys.exit('no spectrum_by_anchoring_w*.csv -- run lfp_spectrum_pac.py first')
_SP = pd.concat([pd.read_csv(f) for f in _shards], ignore_index=True)
_W = _SP.pivot_table(index=['mouse', 'day', 'freq'], columns='state',
                     values='z').dropna().reset_index()
_W['d'] = _W.anch - _W.non

NPERSEG, FLO, FHI = 1024, 1.9, 150.
MIN_RUN_S, SMOOTH, MAJOR_FILT = 2.0, 5, 9
N_LOGBINS, TRIAL_SIGMA = 48, 1.8
LINE = [(48, 52), (98, 102), (148, 152)]
BANDS = [('theta 6–10', 6, 10, '#7b4173'),
         ('slow gamma 30–48', 30, 48, '#2b6cb0'),
         ('fast gamma 60–100', 60, 100, '#c04744')]

# (mouse, day, follows the population pattern?)
#
# Chosen on TWO criteria, not one. A session with a large state difference but a
# single transition shows two blocks that happen to differ; it cannot show power
# TRACKING the state, which is what the figure is for. Every row therefore has at
# least 8 major transitions as well as the population pattern, so the traces can
# be seen turning over repeatedly with the shading.
# The two criteria TRADE OFF and the figure has to compromise. Sessions that
# switch most often have the weakest per-state differences -- short blocks mix
# the states and the means regress together -- so the 24-transition sessions are
# all near zero, while the largest effects come from sessions with one long block
# that cannot show tracking. These six sit in between: >= 4 transitions each and
# a non-trivial effect.
SESSIONS = [(26, 19, True),    # 18 transitions, the strongest of the switchers
            (25, 23, True),    # the largest slow-gamma rise
            (21, 20, True),    # 8 transitions, all three bands in the right direction
            (27, 19, False)]   # 16 transitions, slow gamma goes the WRONG way


def not_mains(f):
    return not any(lo <= f <= hi for lo, hi in LINE)


def compute(mo, dy):
    lf = f'{LFP_ROOT}/M{mo}/D{dy}/VR/sub-M{mo}_ses-D{dy}_typ-VR_beh.nwb'
    if not os.path.exists(lf):
        return None
    f = h5py.File(lf, 'r')
    if 'processing/ecephys/LFP/lfp/data' not in f:
        f.close(); return None
    names = [n.decode() for n in
             f['general/extracellular_ephys/electrodes/channel_name'][:]]
    M = mec_groups(mo, dy, names)
    if len(M) < 4:
        f.close(); return None
    cols = M.g.values
    rc = np.sort(cols); back = np.array([int(np.where(rc == c)[0][0]) for c in cols])
    D = f['processing/ecephys/LFP/lfp/data'][:, rc][:, back].astype(np.float32)
    f.close()

    bp, cp = vr_paths(mo, dy)
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials, _ = clip_trials(beh['trials'].as_dataframe(), clusters)
    T = pd.read_csv(f'{ROOT}/data/population_state/anchoring_trials.csv')
    st = T[(T.mouse == mo) & (T.day == dy)]
    if not len(st):
        return None
    frac = dict(zip(st.trial.astype(int), st.frac_anch))
    pc1d = dict(zip(st.trial.astype(int), st.pc1))

    t = np.arange(D.shape[0]) / FS
    S = beh['S']
    spd = np.asarray(S.values)[np.searchsorted(np.asarray(S.index), t)
                               .clip(0, len(S) - 1)]
    run = spd >= 3.0
    spec, nums = [], []
    for _, tr in trials.iterrows():
        u = int(tr.number)
        if u not in frac:
            continue
        m = (t >= tr.start) & (t <= tr.end) & run
        if m.sum() < MIN_RUN_S * FS:
            continue
        fr, P = welch(D[m], fs=FS, nperseg=NPERSEG, axis=0)
        sel = (fr >= FLO) & (fr <= FHI)
        spec.append(P[sel].mean(axis=1)); nums.append(u)
    if len(nums) < 30:
        return None
    fr_sel = fr[sel]
    A = np.array(spec)
    L = np.log10(A + 1e-20)
    Z = (L - L.mean(0)) / (L.std(0) + 1e-12)
    keepf = np.array([not_mains(x) for x in fr_sel])
    pc = np.array([pc1d[u] for u in nums]); pc = (pc - pc.mean()) / (pc.std() or 1)
    anch = np.array([frac[u] for u in nums]) > .5
    major = median_filter(anch.astype(float), size=MAJOR_FILT, mode='nearest') > .5
    tr_ = np.where(np.diff(major.astype(int)) != 0)[0] + 1

    # the spectrogram, on log-spaced frequency bands. Welch returns LINEARLY
    # spaced bins and the panel plots them on a log axis, so the top of the range
    # is oversampled by an order of magnitude and renders as speckle; averaging
    # into log bands puts a comparable number of original bins behind each row.
    edges = np.geomspace(FLO, FHI, N_LOGBINS + 1)
    rr, centres = [], []
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        k = (fr_sel >= lo_) & (fr_sel < hi_) & keepf
        if k.sum():
            rr.append(L[:, k].mean(1)); centres.append(np.sqrt(lo_ * hi_))
    LB = np.array(rr).T
    ZB = gaussian_filter1d((LB - LB.mean(0)) / (LB.std(0) + 1e-12), TRIAL_SIGMA,
                           axis=0, mode='nearest')

    traces, diffs = {}, {}
    w = _W[(_W.mouse == mo) & (_W.day == dy)]
    for nm, lo, hi, _c in BANDS:
        k = (fr_sel >= lo) & (fr_sel < hi) & keepf
        traces[nm] = uniform_filter1d(Z[:, k].mean(1), SMOOTH, mode='nearest')
        kk = w[(w.freq >= lo) & (w.freq < hi) & w.freq.map(not_mains)]
        diffs[nm] = float(kk.d.mean()) if len(kk) else np.nan
    return dict(pc=pc, major=major, trans=tr_, traces=traces, diffs=diffs,
                n=len(nums), spec=ZB, centres=np.array(centres))


rows = []
for mo, dy, follows in SESSIONS:
    r = compute(mo, dy)
    if r is None:
        print(f'  ! M{mo}D{dy}: no data'); continue
    r.update(mo=mo, dy=dy, follows=follows)
    rows.append(r)
    d = r['diffs']
    print(f'M{mo} D{dy} ({r["n"]} trials): '
          + '  '.join(f'{nm.split()[0]} {d[nm]:+.2f}' for nm, _, _, _ in BANDS))

# Four examples in two columns of two, filled column-wise, so each example gets
# enough width for its spectrogram to be legible and a reader can compare them
# side by side rather than down one tall stack.
fig = plt.figure(figsize=(9.4, 7.0))
OUTER = fig.add_gridspec(2, 2, hspace=.46, wspace=.24)
im = None

for i, r in enumerate(rows):
    cell = OUTER[i % 2, i // 2]
    G = cell.subgridspec(3, 1, height_ratios=[.30, 1.15, .80], hspace=.12)
    x = np.arange(r['n'])

    def shade(ax):
        for stt, col in ((r['major'], ANCH_COLOR), (~r['major'], NONANCH_COLOR)):
            d_ = np.diff(np.concatenate([[0], stt.astype(int), [0]]))
            for s_, e_ in zip(np.where(d_ == 1)[0], np.where(d_ == -1)[0]):
                ax.axvspan(s_ - .5, e_ - .5, color=col, alpha=.15, linewidth=0,
                           zorder=0)

    def lines(ax, c='0.35'):
        for tt in r['trans']:
            ax.axvline(tt - .5, color=c, ls='--', lw=.55, zorder=5)

    d = r['diffs']
    tag = ('follows the population pattern' if r['follows']
           else 'DOES NOT — included for that reason')

    ax = fig.add_subplot(G[0])
    shade(ax); lines(ax); ax.plot(x, r['pc'], color='0.2', lw=.85)
    ax.set_xlim(-.5, r['n'] - .5); ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel('PC1', fontsize=5.8)
    ax.set_title(f'M{r["mo"]} D{r["dy"]} — {tag}\n'
                 + ',  '.join(f'{nm.split()[0]} {d[nm]:+.2f}'
                              for nm, _, _, _ in BANDS),
                 fontsize=6.6, loc='left',
                 color='0.2' if r['follows'] else '#a33')
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(-.10, 1.04, 'ABCDEF'[i], transform=ax.transAxes, fontsize=10,
            weight='bold', va='bottom', ha='right')

    ax = fig.add_subplot(G[1])
    im = ax.pcolormesh(x, r['centres'], r['spec'].T, cmap='RdBu_r',
                       vmin=-1.4, vmax=1.4, shading='gouraud', rasterized=True)
    lines(ax, '0.25')
    for lo_, hi_, c in [(6, 10, '#7b4173'), (30, 48, '#2b6cb0'),
                        (60, 100, '#c04744')]:
        ax.axhline(lo_, color=c, lw=.45, alpha=.6)
        ax.axhline(hi_, color=c, lw=.45, alpha=.6)
    ax.set_yscale('log'); ax.set_yticks([2, 8, 20, 50, 100])
    ax.set_yticklabels(['2', '8', '20', '50', '100'], fontsize=5.8)
    ax.set_xticks([]); ax.set_xlim(-.5, r['n'] - .5)
    ax.set_ylabel('freq (Hz)', fontsize=6)

    ax = fig.add_subplot(G[2])
    shade(ax); lines(ax)
    for nm, lo_, hi_, c in BANDS:
        ax.plot(x, r['traces'][nm], color=c, lw=.85, label=nm)
    ax.axhline(0, color='0.6', lw=.55, ls=':')
    ax.set_xlim(-.5, r['n'] - .5)
    ax.tick_params(labelsize=5.8)
    ax.set_ylabel('band power (z)', fontsize=6)
    ax.set_xlabel('trial', fontsize=6.5)
    ax.spines[['top', 'right']].set_visible(False)
    if i == 0:
        ax.legend(fontsize=5.0, frameon=False, ncol=3, loc='lower left',
                  handlelength=.9, columnspacing=.7)

cax = fig.add_axes([.995, .36, .008, .28])
cb = fig.colorbar(im, cax=cax)
cb.set_label('power (z per frequency)', fontsize=5.8)
cb.ax.tick_params(labelsize=5.2); cb.outline.set_visible(False)

fig.legend(handles=[Patch(facecolor=ANCH_COLOR, alpha=.35, label='anchored'),
                    Patch(facecolor=NONANCH_COLOR, alpha=.35,
                          label='non-anchored')],
           fontsize=6.5, frameon=False, ncol=2, loc='upper right',
           bbox_to_anchor=(.99, 1.02))
fig.savefig(OUT, bbox_inches='tight')
print(f'\nwrote {OUT}')
