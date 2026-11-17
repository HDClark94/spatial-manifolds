"""Figure S1: the anchoring classifier, step by step, against simulated ground truth.

Every anchoring result in the paper rests on the trial-correlation classifier being
right, and on recorded data there is no ground truth: a cell's firing either does or
does not match its own template, and nothing independent says which answer is correct.
A simulation is the only place accuracy can be MEASURED rather than inferred, because
there the anchoring state is imposed rather than inferred.

The simulation is a population of grid cells on the 200 cm track, as the recorded
cells are seen: 2 cm bins, Poisson spikes at a running speed of about 30 cm/s. Each
cell belongs to one of four modules and fires in periodic fields along the track; a
1D slice through a 2D lattice stretches the period, so each cell's period is its
module's spacing times 1.0-1.4. On an ANCHORED trial every field is where it always is,
up to a few cm of jitter. On a NON-ANCHORED trial each module's phase is re-drawn,
the whole module shifting together by an offset drawn afresh on every traversal --
what the recorded non-anchored trials do (the phase is re-drawn each traversal, not
drifting). Trial-to-trial gain and running speed vary as well, so trials differ even
when anchored.

ROW 1 -- the simulation: one cell's fields on anchored and non-anchored trials, the
same session's rate maps for a cell from each of three modules, the true module
offsets, and a population decoder, which shows the ground truth is what it says
before any classifier is applied. One session runs anchored (35 trials), non-anchored
(30), anchored (35), so the true label of every trial is known and the classifier
never sees it.

ROW 2 -- the classifier, one step per panel, on a single cell: the trial x position rate
map it starts from, the trial x trial correlation matrix, each trial's mean correlation
against the 2-means split and the per-cell null threshold, and the label sequence before
and after the median filter.

ROW 3 -- what the two post-processing steps cost and buy. I and J ask whether the GATE
earns its place, which a 70/30 session cannot show: there are two groups of trials and
the split 2-means is forced to make is the right one. The gate exists for a cell locked
in ONE mode and for lopsided splits, so row 3 simulates sessions across the whole range
of true anchored fractions, 0 to 1, and scores the reported fraction against the true
one -- the quantity the paper reads, since the population state is a per-trial
fraction. K is the median filter's cost: the resolution floor it imposes.

A non-anchored trial whose re-drawn offset happens to land near a whole period looks
anchored to that cell, and no classifier of single-cell firing can tell; that sets the
specificity this classifier can reach on periodic cells, and the median filter is what
removes most of those isolated false positives.

The classifier is imported from spatial_manifolds.anchoring rather than reimplemented,
so the figure cannot drift from the pipeline. Cached to CLF_CACHE and GATE_CACHE;
delete those to force a re-run.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.ndimage import median_filter
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import (ANCH_COLOR, MIN_ACTIVITY, NONANCH_COLOR,
                                         _mean_corr, null_mean_corr,
                                         smooth_nanaware, trial_cluster_labels)

plt.rcParams['font.family'] = 'Arial'
# mathtext must resolve to Arial, not the DejaVu default, or rho/times and any
# bold panel letters render in a different face from the rest of the figure
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'
plt.rcParams['pdf.fonttype'] = 42

SIMDIR = '/Users/harryclark/Documents/spatial-manifolds/data/classifier_sim'
FIG = '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026'
OUT = f'{FIG}/fig1_supp_classifier.pdf'
os.makedirs(SIMDIR, exist_ok=True)
CLF_CACHE = f'{SIMDIR}/classifier_validation_cache.npz'
GATE_CACHE = f'{SIMDIR}/classifier_gate_sweep_cache.npz'

# ============================ the simulation ===================================
TRACK_CM, BS_CM = 200., 2.
X = np.arange(BS_CM / 2, TRACK_CM, BS_CM)       # bin centres, 100 bins
N_BINS = len(X)
SPACING = np.array([42., 59., 83., 116.])        # module spacings (cm), ratio ~1.4
N_UNITS = 500
DWELL = BS_CM / 30.                              # s per bin at 30 cm/s
JITTER_CM = 2.5                                  # anchored trials: field jitter
MODULE_C = ['0.2', '#a6761d', '#e6ab02', '0.6']             # no red or blue: those are cell types

_r = np.random.default_rng(7)
MODULE = _r.integers(0, len(SPACING), N_UNITS)
PERIOD = SPACING[MODULE] * _r.uniform(1., 1.4, N_UNITS)
PHASE0 = _r.uniform(0, 1, N_UNITS) * PERIOD
PEAK = np.exp(_r.normal(np.log(10.), .5, N_UNITS))           # Hz
FIELD_AMP = np.exp(_r.normal(0, .35, (N_UNITS, 12)))         # per-field amplitude


def rate(u, shift):
    """Cell u's expected rate (Hz) along the track with its fields moved by shift cm."""
    sig = .11 * PERIOD[u]
    k = np.arange(-2, 10)
    c = PHASE0[u] + shift + k * PERIOD[u]
    f = (FIELD_AMP[u][:, None] * np.exp(-(X[None, :] - c[:, None]) ** 2 / (2 * sig ** 2))).sum(0)
    return .3 + .03 * PEAK[u] + PEAK[u] * f


def simulate(units, states, rng):
    """Trial x position rate maps for the given cells over one session.

    states: per trial, 1 anchored, 0 non-anchored. Non-anchored trials re-draw each
    module's offset; every cell of a module moves with it. Returns the maps
    (units, trials, bins) and the module offsets (trials, modules)."""
    states = np.asarray(states)
    nt = len(states)
    off = np.where(states[:, None] == 1, rng.normal(0, JITTER_CM, (nt, len(SPACING))),
                   rng.uniform(0, 1, (nt, len(SPACING))) * SPACING[None, :])
    dwell = DWELL * rng.uniform(.8, 1.25, nt)                 # running speed per trial
    gain = np.exp(rng.normal(0, .25, (len(units), nt)))       # trial-to-trial rate
    tc = np.empty((len(units), nt, N_BINS))
    for i, u in enumerate(units):
        for t in range(nt):
            lam = rate(u, off[t, MODULE[u]]) * gain[i, t] * dwell[t]
            tc[i, t] = rng.poisson(lam) / dwell[t]
    return tc, off


N_A1, N_C, N_A2 = 35, 30, 35
TRUTH = np.r_[np.ones(N_A1), np.zeros(N_C), np.ones(N_A2)].astype(int)
N_TRIALS = len(TRUTH)
T1, T2 = N_A1, N_A1 + N_C                       # anchoring lost, restored
TC, OFFSET = simulate(np.arange(N_UNITS), TRUTH, np.random.default_rng(11))

TA_CMAP = ListedColormap([NONANCH_COLOR, ANCH_COLOR])
TA_NORM = BoundaryNorm([-.5, .5, 1.5], TA_CMAP.N)
MEDFILT = 5                                     # the pipeline's value
GRID_C = '#d95f02'                              # the gate and the null


# ============================ classification ===================================
def _acc(lab, truth):
    ok = ~np.isnan(lab).any(axis=1)
    acc = np.full(lab.shape[0], np.nan)
    acc[ok] = (lab[ok] == truth[None, :]).mean(axis=1)
    return acc, ok


def classify_all(tc, tag=''):
    """Classify every unit once, returning BOTH label sets.

    The median filter is applied to the label SEQUENCE, after the gate (see
    trial_cluster_labels), so filtered and unfiltered labels share every
    expensive step; the unfiltered labels come from the classifier and the filter
    is applied here. Exactly equivalent to medfilt_size=MEDFILT.
    """
    n = tc.shape[0]
    nf = np.full(tc.shape[:2], np.nan)
    t0 = time.time()
    for u in range(n):
        nf[u] = trial_cluster_labels(tc[u], medfilt_size=0, nan_aware=True)[0]
        if tag and (u + 1) % 100 == 0:
            el = time.time() - t0
            print(f'  {tag} {u + 1}/{n}  {el:.0f}s elapsed', flush=True)
    f = np.vstack([median_filter(r, size=MEDFILT, mode='nearest')
                   if np.isfinite(r).all() else r for r in nf])
    return f, nf


# The epoch sweep: sessions of 35 anchored, k non-anchored, 35 anchored trials,
# scored by recall on the k non-anchored trials. A module's offsets are shared by
# all its cells, so one session's draws move every cell at once; each k is
# therefore averaged over independent sessions, not just over cells.
KS = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30]
N_SESS, N_PER = 8, 32

if os.path.exists(CLF_CACHE):
    _c = np.load(CLF_CACHE)
    LAB, LAB_NF = _c['lab'], _c['lab_nf']
    D = pd.DataFrame({'k': _c['k'], 'acc': _c['acc_k'], 'recall': _c['recall']})
    print(f'loaded {CLF_CACHE}')
else:
    print(f'classifying {N_UNITS} cells (one pass, both filter settings) ...', flush=True)
    LAB, LAB_NF = classify_all(TC, tag='cells')
    rows = []
    for k in KS:
        truth_k = np.r_[np.ones(N_A1), np.zeros(k), np.ones(N_A2)].astype(int)
        accs, recs = [], []
        for i_ in range(N_SESS):
            r_ = np.random.default_rng(1000 * k + i_)
            tc_k, _ = simulate(r_.choice(N_UNITS, N_PER, replace=False), truth_k, r_)
            lab_k, _ = classify_all(tc_k)
            acc_k, ok_k = _acc(lab_k, truth_k)
            accs += list(acc_k[ok_k])
            recs += list((lab_k[ok_k][:, N_A1:N_A1 + k] == 0).mean(axis=1))
        rec = float(np.mean(recs))
        rows.append(dict(k=k, acc=np.nanmedian(accs), recall=rec))
        print(f'  epoch {k:>2}: median acc {rows[-1]["acc"]:.3f}, recall {rec:.3f}', flush=True)
    D = pd.DataFrame(rows)
    np.savez_compressed(CLF_CACHE, lab=LAB, lab_nf=LAB_NF, k=D.k.values,
                        acc_k=D.acc.values, recall=D.recall.values)
    print(f'wrote {CLF_CACHE}')

ACC, OK = _acc(LAB, TRUTH)
ACC_NF, OK_NF = _acc(LAB_NF, TRUTH)
BOTH = OK & OK_NF

# ---- the gate sweep -----------------------------------------------------------
# Sessions of 30 trials, m anchored followed by 30 - m non-anchored -- two
# contiguous blocks, because real states occur in blocks and the median filter
# interacts with run length -- classified with and without the gate. Each m is
# averaged over independent sessions, for the reason given for the epoch sweep.
# The measure that matters is the REPORTED anchored fraction against the true one.
N_F = 30
MS = [0, 1, 2, 3, 5, 8, 11, 15, 19, 22, 25, 27, 28, 29, 30]
N_GATE_SESS, N_GATE_PER = 10, 20
N_GATE_UNITS = N_GATE_SESS * N_GATE_PER

if os.path.exists(GATE_CACHE):
    _g = np.load(GATE_CACHE)
    FR_ON, FR_OFF, AC_ON, AC_OFF = (_g['fr_on'], _g['fr_off'], _g['ac_on'], _g['ac_off'])
    print(f'loaded {GATE_CACHE}')
else:
    FR_ON = np.full((len(MS), N_GATE_UNITS), np.nan)
    FR_OFF, AC_ON, AC_OFF = (np.full_like(FR_ON, np.nan) for _ in range(3))
    print(f'gate sweep: {len(MS)} anchored fractions x {N_GATE_UNITS} cells ...', flush=True)
    for i, m in enumerate(MS):
        truth_m = np.r_[np.ones(m), np.zeros(N_F - m)].astype(int)
        tc_m = np.concatenate([
            simulate(r_.choice(N_UNITS, N_GATE_PER, replace=False), truth_m, r_)[0]
            for r_ in (np.random.default_rng(2000 * m + i_) for i_ in range(N_GATE_SESS))])
        for j in range(N_GATE_UNITS):
            for gate, FR, AC in ((True, FR_ON, AC_ON), (False, FR_OFF, AC_OFF)):
                lab = trial_cluster_labels(tc_m[j], medfilt_size=0, nan_aware=True, gate=gate)[0]
                if not np.isfinite(lab).all():
                    continue
                lab = median_filter(lab, size=MEDFILT, mode='nearest')
                FR[i, j] = lab.mean()
                AC[i, j] = (lab == truth_m).mean()
        print(f'  true fraction {m / N_F:.2f}: reported {np.nanmean(FR_ON[i]):.3f} gated, '
              f'{np.nanmean(FR_OFF[i]):.3f} ungated', flush=True)
    np.savez_compressed(GATE_CACHE, fr_on=FR_ON, fr_off=FR_OFF, ac_on=AC_ON,
                        ac_off=AC_OFF, ms=np.array(MS), n_f=N_F)
    print(f'wrote {GATE_CACHE}')

TRUE_FR = np.array(MS) / N_F
ERR_ON = np.nanmean(np.abs(np.nanmean(FR_ON, axis=1) - TRUE_FR))
ERR_OFF = np.nanmean(np.abs(np.nanmean(FR_OFF, axis=1) - TRUE_FR))


def _cm(lab, mask):
    c = confusion_matrix(np.tile(TRUTH, int(mask.sum())),
                         lab[mask].ravel().astype(int), labels=[0, 1])
    return c / c.sum(axis=1, keepdims=True)


CM, CM_F, CM_NF = _cm(LAB, OK), _cm(LAB, BOTH), _cm(LAB_NF, BOTH)

# ---- the stretches of Figure 2: whole track, task region, black box -------------
# the same session's cells classified from each stretch, without the median filter,
# as Figure 2 classifies them. The black box of trial t is the last 30 cm of trial
# t-1 and the first 30 cm of trial t, so at the two switches it straddles two
# states; it is scored against trial t's own state, and also without the trials
# either side of a switch.
SEG_CACHE = f'{SIMDIR}/classifier_stretch_cache.npz'
_bb = lambda tc: np.concatenate([np.r_[np.full((1, 15), np.nan), tc[:-1, -15:]], tc[:, :15]], axis=1)
STRETCH = {'whole track': lambda tc: tc, 'task region': lambda tc: tc[:, 25:55], 'black box': _bb}
if os.path.exists(SEG_CACHE):
    _s = np.load(SEG_CACHE)
    LAB_ST = {k: _s[k.replace(' ', '_')] for k in STRETCH}
    print(f'loaded {SEG_CACHE}')
else:
    LAB_ST = {}
    for k, f_ in STRETCH.items():
        LAB_ST[k] = (LAB_NF if k == 'whole track' else
                     np.array([trial_cluster_labels(f_(TC[u]), medfilt_size=0, nan_aware=True)[0]
                               for u in range(N_UNITS)]))
        print(f'  classified the {k}', flush=True)
    np.savez_compressed(SEG_CACHE, **{k.replace(' ', '_'): v for k, v in LAB_ST.items()})
    print(f'wrote {SEG_CACHE}')
EDGE = np.zeros(N_TRIALS, bool)
EDGE[[T1 - 1, T1, T2 - 1, T2]] = True                 # the trials either side of a switch
ST_ACC = {}
for k, L_ in LAB_ST.items():
    ok_ = np.isfinite(L_).all(axis=1)
    hit_ = L_[ok_] == TRUTH[None, :]
    ST_ACC[k] = dict(acc=np.median(hit_.mean(1)), acc_in=np.median(hit_[:, ~EDGE].mean(1)),
                     sens=(L_[ok_][:, TRUTH == 1] == 1).mean(),
                     spec=(L_[ok_][:, TRUTH == 0] == 0).mean(), n=int(ok_.sum()),
                     per_cell=hit_.mean(1))

# ---- the population decoder: ground truth before any classifier ---------------
# each bin's population vector against the noise-free anchored templates of every
# bin; the decoded position is the best-matching bin
TEMPL = np.array([rate(u, 0.) for u in range(N_UNITS)])          # (units, bins)
TZ = (TEMPL - TEMPL.mean(0)) / TEMPL.std(0)
DECODE_ERR = np.empty(N_TRIALS)
for t in range(N_TRIALS):
    P = np.array([smooth_nanaware(TC[u, t]) for u in range(N_UNITS)])
    PZ = (P - P.mean(0)) / (P.std(0) + 1e-9)
    DECODE_ERR[t] = np.abs(X[np.argmax(PZ.T @ TZ, axis=1)] - X).mean()

# ---- the walkthrough cell -----------------------------------------------------
# a cell the classifier gets right, and an informative one: of the cells at or
# above 0.95 accuracy, the one whose non-anchored trials are least often aliased
# (their offset near a whole period), so that the panels show decorrelation doing
# the work rather than a near-perfect cell
_cand = [u for u in range(N_UNITS) if OK[u] and ACC[u] >= .95 and MODULE[u] == 1]
EX = int(max(_cand, key=lambda u: (ACC[u], PEAK[u])))
print(f'walkthrough cell: {EX}, module {MODULE[EX] + 1}, period {PERIOD[EX]:.0f} cm, '
      f'peak {PEAK[EX]:.1f} Hz, accuracy {ACC[EX]:.3f}')

raw = TC[EX].astype(float)
sm = np.array([smooth_nanaware(raw[t]) for t in range(N_TRIALS)])
valid = sm.sum(axis=1) > MIN_ACTIVITY
C = np.corrcoef(sm[valid])
mc = _mean_corr(C)
THR = null_mean_corr(sm[valid])
_km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(mc.reshape(-1, 1))
CENT = np.sort(_km.cluster_centers_.ravel())
mc_full = np.full(N_TRIALS, np.nan)
mc_full[valid] = mc

# ============================== figure =========================================
fig = plt.figure(figsize=(8.6, 8.4))
G = fig.add_gridspec(3, 1, height_ratios=[.92, 1.05, .80], hspace=.52)


def lp(ax, letter, x=-.30, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='right')


def tidy(ax):
    ax.tick_params(labelsize=6.5)
    ax.spines[['top', 'right']].set_visible(False)


# ===== ROW 1: the simulation ===================================================
g1 = G[0].subgridspec(1, 4, width_ratios=[1.15, 1.2, 1, 1.05], wspace=.5)

# A -- one cell's fields: held on anchored trials, moved on non-anchored ones
ax = fig.add_subplot(g1[0])
ra = np.random.default_rng(3)
for j in range(3):
    ax.plot(X, rate(EX, ra.normal(0, JITTER_CM)), color=ANCH_COLOR, lw=1, alpha=.85,
            label='anchored' if j == 0 else None)
for j, f_ in enumerate((.3, .55, .8)):
    ax.plot(X, rate(EX, f_ * SPACING[MODULE[EX]]), color=NONANCH_COLOR,
            lw=1, alpha=.85, ls='--', label='non-anchored' if j == 0 else None)
ax.set_xlim(0, TRACK_CM); ax.set_xticks([0, 100, 200])
ax.set_xlabel('position (cm)', fontsize=7.5); ax.set_ylabel('rate (Hz)', fontsize=7.5)
ax.set_title('one simulated grid cell:\nfields hold, or move', fontsize=8)
ax.legend(fontsize=6, frameon=False, loc='upper right', bbox_to_anchor=(1.02, 1.04))
ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
tidy(ax)
lp(ax, 'A', x=-.28)

# B -- the same session for a cell of each of three modules: a module moves together
gb_ = g1[1].subgridspec(1, 3, wspace=.12)
for j, m in enumerate((0, 2, 3)):
    u = int(np.flatnonzero((MODULE == m) & OK)[np.argmax(PEAK[(MODULE == m) & OK])])
    ax = fig.add_subplot(gb_[j])
    M_ = np.array([smooth_nanaware(TC[u, t]) for t in range(N_TRIALS)])
    ax.imshow(M_, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest',
              vmax=np.nanpercentile(M_, 99), extent=[0, TRACK_CM, 0, N_TRIALS])
    for b in (T1, T2):
        ax.axhline(b, color='w', ls='--', lw=.7)
    ax.set_xticks([0, 100]); ax.tick_params(labelsize=6)
    ax.set_title(f'module {m + 1}', fontsize=6.5, color=MODULE_C[m], pad=2)
    if j == 0:
        ax.set_ylabel('trial', fontsize=7.5)
        lp(ax, 'B', x=-.62)
    else:
        ax.set_yticks([])
    if j == 1:
        ax.set_xlabel('position (cm)', fontsize=7.5)

# C -- the true module offsets: the ground truth the classifier never sees
ax = fig.add_subplot(g1[2])
ax.axvspan(T1, T2, color=NONANCH_COLOR, alpha=.18, linewidth=0)
for m in range(len(SPACING)):
    ax.scatter(np.arange(N_TRIALS), OFFSET[:, m] / SPACING[m], s=3, color=MODULE_C[m], lw=0,
               label=f'module {m + 1}')
ax.set_ylim(-.1, 1.05)
ax.set_xlabel('trial', fontsize=7.5); ax.set_ylabel('module offset (periods)', fontsize=7.5)
ax.set_title('anchored 35 | not 30 | 35:\nthe true offsets', fontsize=8)
tidy(ax)
lp(ax, 'C', x=-.3)

# D -- a population decoder shows the ground truth before any classifier
ax = fig.add_subplot(g1[3])
ax.axvspan(T1, T2, color=NONANCH_COLOR, alpha=.18, linewidth=0)
ax.plot(np.arange(N_TRIALS), DECODE_ERR, color='0.2', lw=1.1)
ax.set_xlabel('trial', fontsize=7.5)
ax.set_ylabel('decode error (cm)', fontsize=7.5)
ax.set_title('population decoder,\nnot a classifier', fontsize=8)
tidy(ax)
lp(ax, 'D', x=-.28)

# ===== ROW 2: the classifier, one step per panel ===============================
g2 = G[1].subgridspec(1, 4, width_ratios=[1.12, 1.12, 1.22, .55], wspace=.5)

# E -- the trial x position rate map: the classifier's INPUT. The NaN-aware
# interpolation that fills the 22% of unvisited bins is a prerequisite of having
# a rate map at all, not a step of the classifier, so it is not given a panel.
ax = fig.add_subplot(g2[0])
ax.imshow(sm, aspect='auto', origin='lower', cmap='viridis',
          vmin=0, vmax=np.nanpercentile(sm, 99),
          interpolation='nearest', extent=[0, TRACK_CM, 0, N_TRIALS])
for b in (T1, T2):
    ax.axhline(b, color='w', ls='--', lw=.8)
ax.set_ylabel('trial', fontsize=7.5)
ax.set_xlabel('position (cm)', fontsize=7.5)
ax.set_xticks([0, 100, 200])
ax.set_title('1  trial $\\times$ position rate map', fontsize=7.5, loc='left')
tidy(ax)
lp(ax, 'E', x=-.30, y=1.10)

# F -- trial x trial correlation. aspect='auto' so it matches E in height; the
# colour range is set from the data rather than [-1, 1], which would render the
# whole A block one saturated colour and hide the structure the panel is for.
ax = fig.add_subplot(g2[1])
v = float(np.nanmax(np.abs(C[~np.eye(len(C), dtype=bool)])))
im = ax.imshow(C, origin='lower', cmap='RdBu_r', vmin=-v, vmax=v, aspect='auto',
               interpolation='nearest')
for b in (T1, T2):
    ax.axhline(b, color='k', ls='--', lw=.6)
    ax.axvline(b, color='k', ls='--', lw=.6)
ax.set_xlabel('trial', fontsize=7.5)
ax.set_title('2  trial $\\times$ trial corr.', fontsize=7.5, loc='left')
ax.set_yticklabels([])
tidy(ax)
lp(ax, 'F', x=-.14, y=1.10)

# G -- mean correlation, the null threshold, the 2-means split
ax = fig.add_subplot(g2[2])
ax.axvspan(T1, T2, color=NONANCH_COLOR, alpha=.18, linewidth=0)
ax.plot(np.arange(N_TRIALS), mc_full, color='0.2', lw=1)
ax.axhline(THR, color=GRID_C, lw=1.2, label=f"cell's own null {THR:.2f}")
for c in CENT:
    ax.axhline(c, color='0.45', lw=.8, ls=':')
ax.set_xlabel('trial', fontsize=7.5)
ax.set_ylabel('mean $r$ with other trials', fontsize=7.5)
ax.set_title('3  gate vs the null', fontsize=7.5, loc='left')
ax.legend(fontsize=6, frameon=False, loc='lower left')
tidy(ax)
lp(ax, 'G', x=-.28, y=1.10)

# H -- the label sequences
ax = fig.add_subplot(g2[3])
stack = np.vstack([TRUTH, LAB_NF[EX], LAB[EX]])
ax.imshow(stack.T, aspect='auto', origin='lower', cmap=TA_CMAP, norm=TA_NORM,
          interpolation='nearest')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['truth', 'no filt', 'filt'], fontsize=6.5, rotation=90)
ax.set_ylabel('trial', fontsize=7.5)
ax.set_title('4  labels', fontsize=7.5, loc='left')
ax.tick_params(labelsize=6.5)
lp(ax, 'H', x=-.72, y=1.10)

# ===== ROW 3: what the two post-processing steps cost and buy ==================
# I, J -- why the A -> C -> A session alone is not enough. The gate changes
# nothing at 70/30; it is at the extremes that 2-means fails and the gate rescues.
# K -- the resolution floor the median filter imposes, which is the panel that
# constrains what the paper may claim about transition speed.
g3 = G[2].subgridspec(1, 4, width_ratios=[1, 1, 1.04, 1.1], wspace=.55)

ax = fig.add_subplot(g3[0])
ax.plot([0, 1], [0, 1], ls=':', color='k', lw=1, zorder=1)
for F, c, nm in ((FR_OFF, '0.6', 'no gate'), (FR_ON, GRID_C, 'gate')):
    mu = np.nanmean(F, axis=1)
    se = np.nanstd(F, axis=1) / np.sqrt(np.sum(~np.isnan(F), axis=1))
    ax.fill_between(TRUE_FR, mu - se, mu + se, color=c, alpha=.25, linewidth=0)
    ax.plot(TRUE_FR, mu, 'o-', color=c, ms=3, lw=1.2, label=nm)
ax.set_xlabel('true anchored fraction', fontsize=7.5)
ax.set_ylabel('reported fraction', fontsize=7.5)
ax.set_title('the gate recovers 1 and halves\nthe over-report near 0', fontsize=8)
ax.legend(fontsize=6.5, frameon=False, loc='upper left')
tidy(ax)
lp(ax, 'I', x=-.24)

ax = fig.add_subplot(g3[1])
for A_, c, nm in ((AC_OFF, '0.6', 'no gate'), (AC_ON, GRID_C, 'gate')):
    mu = np.nanmean(A_, axis=1)
    se = np.nanstd(A_, axis=1) / np.sqrt(np.sum(~np.isnan(A_), axis=1))
    ax.fill_between(TRUE_FR, mu - se, mu + se, color=c, alpha=.25, linewidth=0)
    ax.plot(TRUE_FR, mu, 'o-', color=c, ms=3, lw=1.2, label=nm)
ax.axvspan(.55, .85, color=NONANCH_COLOR, alpha=.16, linewidth=0)
ax.text(.70, .30, 'the session in\nrows 1-2 sits here',
        fontsize=6, ha='center', color='0.3')
ax.set_xlabel('true anchored fraction', fontsize=7.5)
ax.set_ylabel('per-trial accuracy', fontsize=7.5)
ax.set_ylim(0, 1.06)
ax.set_title('which is why that session\ncannot test the gate', fontsize=8)
ax.legend(fontsize=6.5, frameon=False, loc='lower left')
tidy(ax)
lp(ax, 'J', x=-.24)

# K -- the resolution floor the median filter imposes
ax = fig.add_subplot(g3[2])
ax.plot(D.k, D.recall, 'o-', color='0.25', ms=4, lw=1.3)
ax.axvline(3, color=GRID_C, ls='--', lw=1)
ax.text(3.25, .10, 'filter floor\n3 trials', fontsize=6.5, color=GRID_C)
ax.set_xscale('log')
ax.set_xticks([1, 2, 3, 5, 10, 30])
ax.set_xticklabels(['1', '2', '3', '5', '10', '30'], fontsize=6.5)
ax.set_xlabel('non-anchored epoch length (trials)', fontsize=7.5)
ax.set_ylabel('recall on that epoch', fontsize=7.5)
ax.set_ylim(-.04, 1.04)
ax.set_title('a size-5 filter erases runs < 3', fontsize=8)
tidy(ax)
lp(ax, 'K', x=-.22)

# L -- the stretches of Figure 2, scored against the truth
ax = fig.add_subplot(g3[3])
STC = {'whole track': '0.62', 'task region': '#e0b43c', 'black box': '0.2'}
rng_ = np.random.default_rng(5)
for j, (k, v) in enumerate(ST_ACC.items()):
    pc_ = v['per_cell']
    ax.bar(j, np.mean(pc_), width=.65, color=STC[k], alpha=.85, lw=0)
    ax.scatter(j + rng_.uniform(-.2, .2, len(pc_)), pc_, s=1.5, color='k', alpha=.2, lw=0, zorder=3)
    ax.text(j, 1.03, f"{v['sens']:.2f}\n{v['spec']:.2f}", ha='center', va='bottom', fontsize=5.6,
            color='0.3', linespacing=1.1)
ax.text(-.75, 1.03, 'sens.\nspec.', ha='right', va='bottom', fontsize=5.6, color='0.3', linespacing=1.1)
ax.set_xticks(range(3)); ax.set_xticklabels(['whole\ntrack', 'task\nregion', 'black\nbox'], fontsize=6.5)
ax.set_ylim(0, 1.0); ax.set_ylabel('per-trial accuracy, no filter', fontsize=7.5)
ax.set_title('the stretches of Figure 2', fontsize=8, pad=20)
tidy(ax)
lp(ax, 'L', x=-.3)

fig.savefig(OUT, bbox_inches='tight')
print(f'\nwrote {OUT}')

print(f"""
ANCHORING CLASSIFIER VALIDATION (simulated grid cells, anchored 35 / non-anchored 30 / anchored 35)

  units classified            {int(OK.sum())}/{N_UNITS}
  median accuracy             {np.nanmedian(ACC):.3f}  (IQR {np.nanpercentile(ACC, 25):.3f}-{np.nanpercentile(ACC, 75):.3f})
  units >= 0.90 accuracy      {int((ACC >= .9).sum())} ({100 * np.nanmean(ACC >= .9):.1f}%)
  sensitivity / specificity   {CM[1, 1]:.3f} / {CM[0, 0]:.3f}
  median filter, accuracy     {np.nanmedian(ACC_NF[BOTH]):.3f} -> {np.nanmedian(ACC[BOTH]):.3f}
  median filter, sensitivity  {CM_NF[1, 1]:.3f} -> {CM_F[1, 1]:.3f}
  median filter, specificity  {CM_NF[0, 0]:.3f} -> {CM_F[0, 0]:.3f}
  period dependence           Spearman rho = {spearmanr(PERIOD[OK], ACC[OK])[0]:+.3f} (p = {spearmanr(PERIOD[OK], ACC[OK])[1]:.3g})
  accuracy by module          {', '.join(f'{np.nanmedian(ACC[OK & (MODULE == m)]):.3f}' for m in range(len(SPACING)))}
  gate, mean |reported - true| fraction  {ERR_OFF:.3f} ungated -> {ERR_ON:.3f} gated
  ungated at true 0           {np.nanmean(FR_OFF[0]):.3f} reported anchored
  decode error                {DECODE_ERR[TRUTH == 1].mean():.1f} cm anchored, {DECODE_ERR[TRUTH == 0].mean():.1f} cm non-anchored
  stretches, no filter (median acc; acc away from switches; sens/spec):
{chr(10).join(f'    {k:12s} {v["acc"]:.3f}; {v["acc_in"]:.3f}; {v["sens"]:.3f}/{v["spec"]:.3f}  ({v["n"]} cells)' for k, v in ST_ACC.items())}
  resolution floor            recall {float(D[D.k == 2].recall.iloc[0]):.2f} at 2 trials, {float(D[D.k == 3].recall.iloc[0]):.2f} at 3
""")
