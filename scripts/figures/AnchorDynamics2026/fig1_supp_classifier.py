"""Figure 1 supplement: the anchoring classifier, step by step, against RNN ground truth.

Every anchoring result in the paper rests on the trial-correlation classifier being
right, and on recorded data there is no ground truth -- a cell's firing either does or
does not match its own template, and nothing independent says which answer is correct.
A path-integrating RNN is the only place accuracy can be MEASURED rather than inferred,
because there the anchoring state is something we impose rather than infer.

ROW 1 -- how the task is recreated in the network. net7 (Ng=1024, trained purely by
path integration on a 2.2 x 2.2 m box) is run along a fixed straight 200 cm chord
through that box, chosen to cross about three firing fields of its best grid units, and
teleported back to the start at the end of every trial -- the linear-track paradigm
with its trial structure, in a network that never saw a track. A grid unit's periodic
firing along the track is therefore a SLICE through its hexagonal field arrangement,
not a separately specified 1D property, which is the panel B point.

The anchoring manipulation is the teleport. In condition A the hidden state is
re-anchored at each teleport by a place-cell initial activation, exactly as at the
start of every training sequence -- the network is told where it is. In condition C the
teleport still happens (position really does reset) but the network is never told: no
re-anchoring, and the teleport step itself is fed as zero velocity, so no self-motion
information at the jump. The hidden state carries over uncorrected and the
representation drifts off the track reference frame. One continuous session runs
A (35 trials) -> C (30) -> A (35), so phase_of_trial is a true per-trial anchored /
non-anchored label, and the classifier never sees it. Panel D shows the manipulation
worked, in decode error, before any classifier is applied.

ROW 2 -- the classifier, one step per panel, on a single cell: the trial x position rate
map it starts from, the trial x trial correlation matrix, each trial's mean correlation
against the 2-means split and the per-cell null threshold, and the label sequence before
and after the median filter. The NaN-aware interpolation over unvisited bins is not
given a panel: it is a prerequisite of having a rate map at all, not a decision the
classifier makes.

ROW 3 -- what the two post-processing steps cost and buy.

I and J ask whether the GATE earns its place, which the A -> C -> A session cannot show.
That session is a 70/30 split, and a 70/30 split is exactly where plain 2-means already
works: there really are two groups of trials and the split it is forced to make is the
right one. Ground-truth accuracy is 0.990 with the gate and 0.990 without it, so that
number says nothing about the gate. The gate exists for the cases the session does not
contain -- a cell locked in ONE mode, and lopsided splits. 2-means always returns two
clusters, so a cell anchored on every trial is still cut roughly in half, and the median
filter then launders the salt-and-pepper into runs that look like state structure.
Row 3 therefore rebuilds sessions across the whole range of true anchored fractions,
0 to 1, and scores the reported fraction against the true one -- the quantity the paper
actually reads, since the population state is a per-trial fraction.

K is the median FILTER's cost: the resolution floor it imposes. Its benefit -- median
accuracy 0.950 -> 0.990, bought entirely in specificity (0.809 -> 0.885) at no cost to
sensitivity (0.993 -> 0.993) -- is reported in the printed summary rather than plotted.

The classifier is imported from spatial_manifolds.anchoring rather than reimplemented,
so the figure cannot drift from the pipeline.

Cached to CLF_CACHE and GATE_CACHE; delete those to force a re-run.
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
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
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

SIM = '/Users/harryclark/Documents/spatial-manifolds/data/rnn_xgboost/net7_1D_track_sim'
FIG = '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026'
OUT = f'{FIG}/fig1_supp_classifier.pdf'
CLF_CACHE = f'{SIM}/classifier_validation_cache.npz'

_sw = np.load(f'{SIM}/switch_cache.npz', allow_pickle=True)
_sim = np.load(f'{SIM}/sim_cache.npz', allow_pickle=True)
_of = np.load(f'{SIM}/of_ratemaps_cache.npz')

TC = _sw['all_unit_tc_switch']                 # (units, trials, bins)
PHASE = _sw['phase_of_trial'].astype(str)      # 'A' cued, 'C' blind
DECODE_ERR = _sw['decode_errs_switch']
GRID_SCORES = _sim['grid_scores']
START, END = _sim['start_pos'], _sim['end_pos']
TRAJ = _sim['example_trial_xy']
BOX = float(_sim['box_width'])
BS_CM, TRACK_CM = float(_sim['bs']), float(_sim['tl'])
OF_RM, OF_UNITS = _of['ratemaps'], _of['units']

TRUTH = (PHASE == 'A').astype(int)
N_UNITS, N_TRIALS, N_BINS = TC.shape
T1 = int(np.argmax(PHASE == 'C'))               # cue lost
T2 = T1 + int((PHASE == 'C').sum())             # cue restored

TA_CMAP = ListedColormap([NONANCH_COLOR, ANCH_COLOR])
TA_NORM = BoundaryNorm([-.5, .5, 1.5], TA_CMAP.N)
SE_CMAP = LinearSegmentedColormap.from_list('se', ['tab:green', 'tab:red'])
MEDFILT = 5                                     # the pipeline's value
GRID_C = '#c04744'
DT = 0.02                                       # s per simulation step


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
    expensive step -- the NaN-aware smoothing, the correlation matrix, the
    2-means fit, and the 200-shift per-cell null. Calling the classifier twice
    would recompute all of that to change one cheap post-step, so the
    unfiltered labels come from the classifier and the filter is applied here.
    Exactly equivalent to medfilt_size=MEDFILT, not an approximation.
    """
    n = tc.shape[0]
    nf = np.full(tc.shape[:2], np.nan)
    t0 = time.time()
    for u in range(n):
        nf[u] = trial_cluster_labels(tc[u], medfilt_size=0, nan_aware=True)[0]
        if tag and (u + 1) % 200 == 0:
            el = time.time() - t0
            print(f'  {tag} {u + 1}/{n}  {el:.0f}s elapsed, '
                  f'{el / (u + 1) * (n - u - 1):.0f}s left', flush=True)
    f = np.vstack([median_filter(r, size=MEDFILT, mode='nearest')
                   if np.isfinite(r).all() else r for r in nf])
    return f, nf


# The epoch sweep splices sessions of varying non-anchored length by putting the
# FIRST k trials of the C block between the two A blocks. Taking the first k is the
# principled choice: the hidden state drifts progressively during C, so the first k
# C trials are exactly what a k-trial cue loss would have produced. Splicing later
# C trials would sample a more-drifted state than a short epoch ever reaches, and
# would flatter the classifier. The null is rebuilt per spliced session rather than
# reused, since it is built from that session's own profiles -- so the sweep is ten
# full passes and runs on a fixed subsample. The quantity is a population mean
# recall, and the floor it localises is set by the filter width, not by n.
KS = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30]
N_SWEEP = 256

if os.path.exists(CLF_CACHE):
    _c = np.load(CLF_CACHE)
    LAB, LAB_NF = _c['lab'], _c['lab_nf']
    D = pd.DataFrame({'k': _c['k'], 'acc': _c['acc_k'], 'recall': _c['recall']})
    print(f'loaded {CLF_CACHE}')
else:
    print(f'classifying {N_UNITS} units (one pass, both filter settings) ...',
          flush=True)
    LAB, LAB_NF = classify_all(TC, tag='units')
    A_pre, C_all, A_post = (np.arange(T1), np.arange(T1, T2),
                            np.arange(T2, N_TRIALS))
    sub = np.random.default_rng(0).choice(N_UNITS, N_SWEEP, replace=False)
    rows = []
    for k in KS:
        idx = np.concatenate([A_pre, C_all[:k], A_post])
        truth_k = np.concatenate([np.ones(len(A_pre)), np.zeros(k),
                                  np.ones(len(A_post))]).astype(int)
        lab_k, _ = classify_all(TC[np.ix_(sub, idx)])
        acc_k, ok_k = _acc(lab_k, truth_k)
        rec = np.nanmean((lab_k[ok_k][:, len(A_pre):len(A_pre) + k] == 0)
                         .mean(axis=1))
        rows.append(dict(k=k, acc=np.nanmedian(acc_k), recall=rec))
        print(f'  epoch {k:>2}: median acc {rows[-1]["acc"]:.3f}, '
              f'recall {rec:.3f}', flush=True)
    D = pd.DataFrame(rows)
    np.savez_compressed(CLF_CACHE, lab=LAB, lab_nf=LAB_NF, k=D.k.values,
                        acc_k=D.acc.values, recall=D.recall.values)
    print(f'wrote {CLF_CACHE}')

ACC, OK = _acc(LAB, TRUTH)
ACC_NF, OK_NF = _acc(LAB_NF, TRUTH)
BOTH = OK & OK_NF

# ---- the gate sweep -----------------------------------------------------------
# The A -> C -> A session CANNOT test the gate. It is a 70/30 split, which is
# exactly where plain 2-means already works: there are two genuinely different
# groups of trials and the split it is forced to make is the right one. Running
# the session with and without the gate leaves ground-truth accuracy at 0.990
# either way, so quoting that number says nothing about whether the gate helps.
#
# The gate exists for the cases this session does not contain: a cell LOCKED in
# one mode, and lopsided splits. 2-means always returns two clusters, so a cell
# that was anchored on every trial still gets cut roughly in half, and the median
# filter then launders the salt-and-pepper into runs that look like real state
# structure. Testing it therefore requires sessions built at a RANGE of true
# anchored fractions, including 0 and 1.
#
# Sessions of 30 trials are spliced from the switch session's own pools (70 A, 30
# C) as m anchored trials followed by 30 - m non-anchored ones -- kept as two
# contiguous blocks rather than interleaved, because real states occur in blocks
# and the median filter interacts with run length. The measure that matters is
# the REPORTED anchored fraction against the true one: the paper reads the
# population state off that fraction, so a classifier that is accurate per trial
# but biased in fraction would still corrupt every population result.
N_F = 30
MS = [0, 1, 2, 3, 5, 8, 11, 15, 19, 22, 25, 27, 28, 29, 30]
N_GATE_UNITS = 200
GATE_CACHE = f'{SIM}/classifier_gate_sweep_cache.npz'

if os.path.exists(GATE_CACHE):
    _g = np.load(GATE_CACHE)
    FR_ON, FR_OFF, AC_ON, AC_OFF = (_g['fr_on'], _g['fr_off'],
                                    _g['ac_on'], _g['ac_off'])
    print(f'loaded {GATE_CACHE}')
else:
    A_POOL = np.where(PHASE == 'A')[0]
    C_POOL = np.where(PHASE == 'C')[0]
    gsub = np.random.default_rng(1).choice(N_UNITS, N_GATE_UNITS, replace=False)
    FR_ON = np.full((len(MS), N_GATE_UNITS), np.nan)
    FR_OFF, AC_ON, AC_OFF = (np.full_like(FR_ON, np.nan) for _ in range(3))
    print(f'gate sweep: {len(MS)} anchored fractions x {N_GATE_UNITS} units ...',
          flush=True)
    for i, m in enumerate(MS):
        idx = np.concatenate([A_POOL[:m], C_POOL[:N_F - m]]).astype(int)
        truth_m = np.concatenate([np.ones(m), np.zeros(N_F - m)]).astype(int)
        for j, u in enumerate(gsub):
            for gate, FR, AC in ((True, FR_ON, AC_ON), (False, FR_OFF, AC_OFF)):
                lab = trial_cluster_labels(TC[u][idx], medfilt_size=0,
                                           nan_aware=True, gate=gate)[0]
                if not np.isfinite(lab).all():
                    continue
                lab = median_filter(lab, size=MEDFILT, mode='nearest')
                FR[i, j] = lab.mean()
                AC[i, j] = (lab == truth_m).mean()
        print(f'  true fraction {m / N_F:.2f}: reported '
              f'{np.nanmean(FR_ON[i]):.3f} gated, {np.nanmean(FR_OFF[i]):.3f} '
              f'ungated', flush=True)
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

# ---- the walkthrough cell -----------------------------------------------------
# Chosen from the units with a cached 2D map, so panels B and E-I are the SAME
# cell and the figure reads as one cell carried through the whole pipeline.
#
# The selection requires the cell to stay ACTIVE through the blind block, not just
# to classify well. The highest grid-score units go nearly silent during C, so
# their C-block trials fall below the classifier's activity floor and are
# force-labelled non-anchored before the correlation step is reached: they score
# 1.000, but the trial x trial correlation, the 2-means split and the null gate
# never decide anything, which makes them useless for a figure whose whole point
# is to show those steps deciding. Requiring C-block activity picks a cell that
# is classified by DECORRELATION, the mechanism the paper actually relies on.
_cblock = np.where(PHASE == 'C')[0]


def _c_active(u):
    s = np.array([smooth_nanaware(TC[u][t].astype(float)) for t in _cblock])
    return float((s.sum(axis=1) > MIN_ACTIVITY).mean())


_cand = [u for u in OF_UNITS if GRID_SCORES[u] > .3 and OK[u] and ACC[u] >= .95]
_cand = [u for u in _cand if _c_active(u) >= .9] or _cand
EX = int(max(_cand, key=lambda u: (ACC[u], GRID_SCORES[u])))
EX_OF = int(np.where(OF_UNITS == EX)[0][0])
print(f'walkthrough cell: unit {EX}, grid score {GRID_SCORES[EX]:+.2f}, '
      f'accuracy {ACC[EX]:.3f}, C-block active {_c_active(EX):.2f}')

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
fig = plt.figure(figsize=(7.4, 8.4))
G = fig.add_gridspec(3, 1, height_ratios=[.92, 1.05, .80], hspace=.52)


def lp(ax, letter, x=-.30, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='right')


def grad_path(ax, xy, lw=1.5, z=2, alpha=1.):
    pts = np.asarray(xy).reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(seg, cmap=SE_CMAP, array=np.linspace(0, 1, len(seg)),
                        linewidth=lw, zorder=z, alpha=alpha)
    ax.add_collection(lc)


def tidy(ax):
    ax.tick_params(labelsize=6.5)
    ax.spines[['top', 'right']].set_visible(False)


# ===== ROW 1: how the task is recreated ========================================
g1 = G[0].subgridspec(1, 4, width_ratios=[1, 1, 1.05, 1.15], wspace=.52)

# A -- the box and the slice
ax = fig.add_subplot(g1[0])
hw = BOX / 2
ax.add_patch(plt.Rectangle((-hw, -hw), BOX, BOX, fill=False, edgecolor='0.5',
                           lw=1))
grad_path(ax, np.linspace(START, END, 60), lw=2.6, z=3)
grad_path(ax, TRAJ, lw=.7, z=2, alpha=.75)
ax.scatter(*START, color='tab:green', s=26, zorder=4)
ax.scatter(*END, color='tab:red', s=26, zorder=4)
ax.annotate('', xy=(START[0], START[1] - .10), xytext=(END[0], END[1] - .10),
            arrowprops=dict(arrowstyle='->', color='0.35', lw=1.1,
                            connectionstyle='arc3,rad=.30'))
ax.text(0, -.74, 'teleport', fontsize=6.5, color='0.35', ha='center')
ax.set_xlim(-hw - .06, hw + .06); ax.set_ylim(-hw - .06, hw + .06)
ax.set_aspect('equal')
ax.set_xlabel('x (m)', fontsize=7.5); ax.set_ylabel('y (m)', fontsize=7.5)
ax.set_title('200 cm chord through\nthe 2.2 m box', fontsize=8)
tidy(ax)
lp(ax, 'A', x=-.26)

# B -- the 1D track is a slice through the 2D grid
ax = fig.add_subplot(g1[1])
ax.imshow(OF_RM[EX_OF], origin='lower', cmap='viridis', extent=[-hw, hw, -hw, hw],
          interpolation='bilinear')
ax.plot([START[0], END[0]], [START[1], END[1]], color='w', lw=1.8)
ax.plot([START[0], END[0]], [START[1], END[1]], color='k', lw=.7, ls='--')
ax.set_aspect('equal')
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f'unit {EX}, grid score {GRID_SCORES[EX]:.2f}\n'
             '1D firing is this slice', fontsize=8)
lp(ax, 'B', x=-.10)

# C -- the trial structure: teleport sawtooth
ax = fig.add_subplot(g1[2])
along = np.linalg.norm(TRAJ - START[None, :], axis=1) * 100
t_tr = np.arange(len(along)) * DT
for tr in range(4):
    off = tr * (t_tr[-1] + .3)
    ax.plot(t_tr + off, along, color='0.25', lw=1)
    if tr:
        ax.plot([off - .3, off], [TRACK_CM, 0], color=GRID_C, lw=.9, ls=':')
ax.set_xlabel('time (s)', fontsize=7.5)
ax.set_ylabel('position on track (cm)', fontsize=7.5)
ax.set_title('4 trials; dotted =\nteleport reset', fontsize=8)
tidy(ax)
lp(ax, 'C', x=-.30)

# D -- the manipulation worked, before any classifier is applied
ax = fig.add_subplot(g1[3])
ax.axvspan(T1, T2, color=NONANCH_COLOR, alpha=.18, linewidth=0)
ax.plot(np.arange(N_TRIALS), DECODE_ERR * 100, color='0.2', lw=1.1)
ax.set_xlabel('trial', fontsize=7.5)
ax.set_ylabel('decode error (cm)', fontsize=7.5)
ax.set_title('A (cued) 35 | C (blind) 30 | A 35\nground truth, not a classifier',
             fontsize=8)
ax.text(T1 + 15, ax.get_ylim()[1] * .9, 'no cue', fontsize=6.5, ha='center',
        color='0.3')
tidy(ax)
lp(ax, 'D', x=-.26)

# ===== ROW 2: the classifier, one step per panel ===============================
g2 = G[1].subgridspec(1, 4, width_ratios=[1.12, 1.12, 1.22, .55], wspace=.5)

# E -- the trial x position rate map: the classifier's INPUT. The NaN-aware
# interpolation that fills the 22% of unvisited bins is a prerequisite of having
# a rate map at all, not a step of the classifier, so it is not given a panel.
ax = fig.add_subplot(g2[0])
ax.imshow(sm, aspect='auto', origin='lower', cmap='viridis',
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
g3 = G[2].subgridspec(1, 3, width_ratios=[1, 1, 1.04], wspace=.52)

ax = fig.add_subplot(g3[0])
ax.plot([0, 1], [0, 1], ls=':', color='k', lw=1, zorder=1)
for F, c, nm in ((FR_OFF, '0.6', 'no gate'), (FR_ON, GRID_C, 'gate')):
    mu = np.nanmean(F, axis=1)
    se = np.nanstd(F, axis=1) / np.sqrt(np.sum(~np.isnan(F), axis=1))
    ax.fill_between(TRUE_FR, mu - se, mu + se, color=c, alpha=.25, linewidth=0)
    ax.plot(TRUE_FR, mu, 'o-', color=c, ms=3, lw=1.2, label=nm)
ax.set_xlabel('true anchored fraction', fontsize=7.5)
ax.set_ylabel('reported fraction', fontsize=7.5)
ax.set_title('2-means cannot report 0 or 1;\nthe gate can', fontsize=8)
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
ax.text(.70, .30, 'the A$\\rightarrow$C$\\rightarrow$A\nsession sits here',
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
ax.plot(D.k, D.recall, 'o-', color='#4a7ba7', ms=4, lw=1.3)
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

fig.savefig(OUT, bbox_inches='tight')
print(f'\nwrote {OUT}')

print(f"""
ANCHORING CLASSIFIER VALIDATION (RNN ground truth, A -> C -> A cue-loss session)

  units classified            {int(OK.sum())}/{N_UNITS}
  median accuracy             {np.nanmedian(ACC):.3f}  (IQR {np.nanpercentile(ACC, 25):.3f}-{np.nanpercentile(ACC, 75):.3f})
  units >= 0.90 accuracy      {int((ACC >= .9).sum())} ({100 * np.nanmean(ACC >= .9):.1f}%)
  sensitivity / specificity   {CM[1, 1]:.3f} / {CM[0, 0]:.3f}
  median filter, accuracy     {np.nanmedian(ACC_NF[BOTH]):.3f} -> {np.nanmedian(ACC[BOTH]):.3f}
  median filter, sensitivity  {CM_NF[1, 1]:.3f} -> {CM_F[1, 1]:.3f}
  median filter, specificity  {CM_NF[0, 0]:.3f} -> {CM_F[0, 0]:.3f}
  gridness dependence         Spearman rho = {spearmanr(GRID_SCORES[OK], ACC[OK])[0]:+.3f} (p = {spearmanr(GRID_SCORES[OK], ACC[OK])[1]:.3g})
  resolution floor            recall {float(D[D.k == 2].recall.iloc[0]):.2f} at 2 trials, {float(D[D.k == 3].recall.iloc[0]):.2f} at 3
""")
