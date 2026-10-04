"""Example sessions: the population anchoring state and theta, on one trial axis.

Five sessions rather than one, because a single example cannot show whether the
effect is typical or whether the example was chosen well. They span the range
deliberately:

  M25 D23   the reference example -- best-balanced states of any session
            (49.3% anchored), median cell-following (0.269), theta -10.6%
  M27 D23   nearly as balanced (48.9%) with a MODEST effect, -3.7%, over 393
            trials: what a typical session looks like
  M26 D18   the largest theta effect in the dataset, -19.6%, but the most
            extreme session on the cell side too (70% of cells lock to PC1
            against a median of 27%), so it overstates everything
  M21 D22   carries the scatter panel of fig6_speed_theta_coupling, so the two
            figures can be read against each other; effect -3.7%
  M20 D14   the largest cell count (274), effect -7.3%

Sessions were required to have >= 30 trials per state, a theta amplitude
difference at p < 0.01, and > 60 cells; within that set they were chosen to
span balance and effect size rather than to maximise either.

Panels share one trial axis so the question "do they move together?" can be read
directly rather than inferred from a correlation. Mean running speed is shown on
the same axis because theta amplitude tracks speed closely -- with speed plotted
alongside, a reader can see whether a theta step coincides with a speed step or
is independent of it, rather than having to trust the matching.
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import median_filter
from scipy.stats import mannwhitneyu
import pynapple as nap
sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
_g = {}
for _c in json.load(open('/Users/harryclark/Documents/spatial-manifolds/scripts/'
                         'figures/AnchorDynamics2026/lick_raster_by_trial_type.ipynb'))['cells']:
    if _c['cell_type'] != 'code':
        continue
    _s = ''.join(_c['source'])
    if _s.startswith('MOUSE, DAY') or 'INV = pd.DataFrame' in _s:
        continue
    exec(compile(_s, '<nb>', 'exec'), _g)
globals().update({k: v for k, v in _g.items() if not k.startswith('__')})
from spatial_manifolds.anchoring import trial_cluster_labels
plt.rcParams['font.family'] = 'Arial'

SESSIONS = [(25, 23), (27, 23), (26, 18), (21, 22), (20, 14)]
FIG = '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026'
SMOOTH = 5            # trials, for the running means only


def make(MO, DY):
    T = pd.read_csv('/Users/harryclark/Documents/spatial-manifolds/data/lfp/theta_frequency.csv')
    t = T[(T.mouse == MO) & (T.day == DY)].sort_values('trial').reset_index(drop=True)
    print(f'M{MO}D{DY}: {len(t)} trials with LFP')

    # per-cell labels, for the raster
    bp, cp = vr_paths(MO, DY)
    beh = nap.load_file(bp); clusters = nap.load_file(cp)
    trials, orig = clip_trials(beh['trials'].as_dataframe(), clusters)
    keep = np.isin(beh['trials'].as_dataframe().number.values.astype(int), orig)
    ids = spatial_cell_ids(MO, DY, clusters)
    tn, trav = beh['trial_number'], beh['travel']
    dt = trav - (float(np.asarray(tn.values)[0]) - 1) * TL
    moving = beh['S'].threshold(3.0, method='above').time_support
    n_tr = len(beh['trials'].as_dataframe())
    tc = nap.compute_1d_tuning_curves(clusters[ids], dt, nb_bins=n_tr * NBIN,
                                      minmax=[0, n_tr * TL], ep=moving)
    L = []
    for c in ids:
        lab, _, _, _ = trial_cluster_labels(np.asarray(tc[c]).reshape(n_tr, NBIN)[keep])
        if not np.isnan(lab).all():
            L.append(lab)
    L = np.array(L)
    frac = np.nanmean(L, axis=0)
    Lc = np.nan_to_num(L - np.nanmean(L, axis=1, keepdims=True))
    U, sv, Vt = np.linalg.svd(Lc, full_matrices=False)
    load, pc1 = U[:, 0], Vt[0]
    if np.corrcoef(pc1, np.nan_to_num(frac))[0, 1] < 0:
        load, pc1 = -load, -pc1
    order = np.argsort(load)[::-1]
    print(f'  {L.shape[0]} MEC cells, PC1 {100*sv[0]**2/np.sum(sv**2):.0f}% of label variance')

    state = median_filter((frac > 0.5).astype(float), size=9, mode='nearest') > 0.5
    # Boundaries are drawn only around runs of at least MINRUN trials. Sessions
    # such as M27 D23 flicker between states for a few trials at a time, and a
    # dashed line at every one of those turns the panel into a picket fence
    # while telling the reader nothing. The raster and PC1 still show every
    # trial, so nothing is hidden by this -- it only thins the annotation.
    MINRUN = 8
    disp = state.copy()
    while True:                      # absorb short runs into the run before them
        e = np.r_[0, np.where(np.diff(disp.astype(int)) != 0)[0] + 1, len(disp)]
        short = [i for i in range(len(e) - 1) if e[i + 1] - e[i] < MINRUN]
        if not short:
            break
        i = short[0]
        disp[e[i]:e[i + 1]] = disp[e[i] - 1] if e[i] > 0 else disp[e[i + 1]]
    bounds = list(np.where(np.diff(disp.astype(int)) != 0)[0] + 1)
    tr_ax = np.arange(1, len(frac) + 1)
    sm = lambda v: pd.Series(v).rolling(SMOOTH, center=True, min_periods=2).mean().values

    TA = ListedColormap([NONANCH_COLOR, ANCH_COLOR])
    NORM = BoundaryNorm([-0.5, 0.5, 1.5], TA.N)
    fig, axes = plt.subplots(5, 1, figsize=(7.6, 8.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.5, .7, .8, .8, .8],
                                              hspace=.18))

    ax = axes[0]
    ax.imshow(L[order], aspect='auto', cmap=TA, norm=NORM, interpolation='nearest',
              extent=[1, len(frac), L.shape[0], 1])
    ax.set_ylabel('Cell (PC1 order)', fontsize=8.5)
    ax.set_title(f'M{MO} D{DY} — anchoring state and theta on one trial axis',
                 fontsize=10, loc='left', pad=6)
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax = axes[1]
    ax.fill_between(tr_ax, 0, pc1, where=pc1 >= 0, color=ANCH_COLOR, lw=0,
                    edgecolor='none', interpolate=True)
    ax.fill_between(tr_ax, 0, pc1, where=pc1 < 0, color=NONANCH_COLOR, lw=0,
                    edgecolor='none', interpolate=True)
    ax.axhline(0, color='0.5', lw=.7)
    ax.set_ylabel('PC1', fontsize=8.5)
    ax.spines[['top', 'right']].set_visible(False)

    for ax, col, lab, c in ((axes[2], 'amp', 'Theta amplitude', '#b5651d'),
                            (axes[3], 'inst_hz', 'Theta freq (Hz)', '#3b6ea5'),
                            (axes[4], 'speed', 'Speed (cm/s)', '#4f8f4f')):
        v = np.full(len(frac), np.nan)
        v[t.trial.values - 1] = t[col].values
        ax.plot(tr_ax, v, color='0.75', lw=.6, zorder=2)
        ax.plot(tr_ax, sm(v), color=c, lw=1.8, zorder=3)
        lo, hi = np.nanpercentile(v, [1, 99])          # a couple of spikes otherwise
        pad = (hi - lo) * .35                          # compress the block structure
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(lab, fontsize=8.5)
        ax.spines[['top', 'right']].set_visible(False)
        a = t[t.anch][col]; b = t[~t.anch][col]
        ax.text(.995, .06, f'anchored {a.mean():.2f} vs {b.mean():.2f}, '
                f'p = {mannwhitneyu(a, b)[1]:.1g}', transform=ax.transAxes,
                fontsize=7, color='0.3', ha='right', zorder=6,
                bbox=dict(facecolor='white', alpha=.75, edgecolor='none', pad=1.5))

    for ax in axes:
        for bd in bounds:
            ax.axvline(bd + 1, color='0.15', lw=.9, ls='--', zorder=5)
    axes[-1].set_xlabel('Trial', fontsize=9)
    for ax in axes:
        ax.tick_params(labelsize=7.5)
    axes[0].set_xlim(1, len(frac))

    out = f'{FIG}/fig6_lfp_example_M{MO}D{DY}.pdf'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print('saved', out)
    for col in ('amp', 'inst_hz', 'peak_hz'):
        a, b = t[t.anch][col], t[~t.anch][col]
        print(f'  {col:8s} anchored {a.mean():8.3f}  non {b.mean():8.3f}  '
              f'{100*(a.mean()-b.mean())/b.mean():+6.2f}%  p={mannwhitneyu(a,b)[1]:.2g}')


if __name__ == '__main__':
    for _mo, _dy in SESSIONS:
        try:
            make(_mo, _dy)
        except Exception as _e:
            print(f'  ! M{_mo}D{_dy}: {type(_e).__name__}: {_e}', flush=True)
