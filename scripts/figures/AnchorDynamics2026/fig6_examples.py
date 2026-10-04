"""Candidate example sessions for Figure 5, side by side.

The population analysis makes four claims about the anchored state:

    theta amplitude   LOWER      (13/19 sessions)
    theta frequency   LOWER      (15/19)
    r(speed, amp)     LOWER      (15/19)   <- the decoupling
    r(speed, freq)    LOWER      (16/19)   <- the decoupling
    running speed     unchanged

An example panel should show all of them, and the two gain effects most of all,
since those carry the decoupling result. Only FIVE of nineteen sessions get all
four directions right, and only one of those is also blocky enough to read as a
raster -- so the choice is genuinely constrained, and the current example is not
the best available: M28 D23 gets 3 of 4, and the one it misses is
r(speed, frequency), the strongest population effect (p = 2.7e-4), which in that
session runs the WRONG way (+0.08).

This renders the candidates on identical axes so the trade-offs are visible
rather than argued. Each column is one session: the anchoring raster, theta
amplitude, theta frequency, running speed, and the two speed-theta scatters
split by state.

Usage:  python3 fig6_examples.py
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
from scipy.stats import pearsonr

sys.path.insert(0, '/Users/harryclark/Documents/spatial-manifolds/src')
from spatial_manifolds.anchoring import (ANCH_COLOR, NONANCH_COLOR,
                                         load_session_labels)

plt.rcParams['font.family'] = 'Arial'
FIG = '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/AnchorDynamics2026'
LFP = '/Users/harryclark/Documents/spatial-manifolds/data/lfp'
SMOOTH = 5

# chosen from the ranking in the docstring; the last is the incumbent, kept so
# the comparison includes what the figure currently uses
CANDIDATES = [(28, 19), (28, 25), (27, 19), (21, 21), (28, 23)]
TRACES = (('amp', 'Theta amp.', '#b5651d'),
          ('inst_hz', 'Theta (Hz)', '#3b6ea5'),
          ('speed', 'Speed (cm/s)', '#4f8f4f'))


def load():
    T = pd.read_csv(f'{LFP}/theta_frequency.csv')
    T['anch'] = T.anch.astype(bool)
    return T


def session_stats(T, mo, dy):
    d = T[(T.mouse == mo) & (T.day == dy)]
    a, n = d[d.anch], d[~d.anch]
    return dict(d_amp=100 * (a.amp.mean() - n.amp.mean()) / d.amp.mean(),
                d_hz=a.inst_hz.mean() - n.inst_hz.mean(),
                d_spd=a.speed.mean() - n.speed.mean(),
                r_amp_a=pearsonr(a.speed, a.amp)[0],
                r_amp_n=pearsonr(n.speed, n.amp)[0],
                r_hz_a=pearsonr(a.speed, a.inst_hz)[0],
                r_hz_n=pearsonr(n.speed, n.inst_hz)[0])


def draw_column(fig, gs, col, T, mo, dy, show_ylab):
    """Raster, three traces and two scatters for one session."""
    z = load_session_labels(mo, dy)
    in_pop = np.isfinite(z['pc1_load'])
    L, load_ = z['labels'][in_pop], z['pc1_load'][in_pop]
    order = np.argsort(load_)[::-1]
    te = T[(T.mouse == mo) & (T.day == dy)].sort_values('trial')
    n_tr = L.shape[1]
    tr = np.arange(1, n_tr + 1)
    st = session_stats(T, mo, dy)
    sm = lambda v: pd.Series(v).rolling(SMOOTH, center=True,
                                        min_periods=2).mean().values

    ax = fig.add_subplot(gs[0, col])
    ax.imshow(L[order], aspect='auto',
              cmap=ListedColormap([NONANCH_COLOR, ANCH_COLOR]),
              norm=BoundaryNorm([-.5, .5, 1.5], 2), interpolation='nearest',
              extent=[1, n_tr, L.shape[0], 1])
    nok = sum(v < 0 for v in (st['d_amp'], st['d_hz'],
                              st['r_amp_a'] - st['r_amp_n'],
                              st['r_hz_a'] - st['r_hz_n']))
    ax.set_title(f'M{mo} D{dy}   {nok}/4 effects\n{int(in_pop.sum())} cells, '
                 f'{n_tr} trials', fontsize=8, loc='left',
                 color='0.15' if nok == 4 else '0.45')
    if show_ylab:
        ax.set_ylabel('Cell (PC1 order)', fontsize=8)
    ax.set_xticklabels([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    for k, (c, lab, colr) in enumerate(TRACES):
        ax = fig.add_subplot(gs[k + 1, col])
        v = np.full(n_tr, np.nan)
        ok = (te.trial.values >= 1) & (te.trial.values <= n_tr)
        v[te.trial.values[ok] - 1] = te[c].values[ok]
        ax.plot(tr, v, color='0.8', lw=.4)
        ax.plot(tr, sm(v), color=colr, lw=1.3)
        lo, hi = np.nanpercentile(v, [1, 99]); pad = (hi - lo) * .35
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(1, n_tr)
        key = {'amp': 'd_amp', 'inst_hz': 'd_hz', 'speed': 'd_spd'}[c]
        unit = '%' if c == 'amp' else (' Hz' if c == 'inst_hz' else ' cm/s')
        ax.annotate(f'Δ {st[key]:+.2f}{unit}', (.98, .04), xycoords='axes fraction',
                    ha='right', fontsize=6.3,
                    color='0.2' if st[key] < 0 else '#b03030')
        if show_ylab:
            ax.set_ylabel(lab, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[['top', 'right']].set_visible(False)
        if k < len(TRACES) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Trial', fontsize=8)

    for k, (c, lab) in enumerate((('amp', 'Theta amplitude'),
                                  ('inst_hz', 'Theta frequency (Hz)'))):
        ax = fig.add_subplot(gs[k + 4, col])
        for m, cc in ((te.anch.values, ANCH_COLOR), (~te.anch.values, NONANCH_COLOR)):
            ax.scatter(te.speed.values[m], te[c].values[m], s=4, color=cc, lw=0,
                       alpha=.45, zorder=2)
            if m.sum() > 5:
                b = np.polyfit(te.speed.values[m], te[c].values[m], 1)
                xx = np.linspace(te.speed.values[m].min(), te.speed.values[m].max(), 20)
                ax.plot(xx, np.polyval(b, xx), color=cc, lw=1.6, zorder=3)
        ra = st[f'r_{"amp" if c == "amp" else "hz"}_a']
        rn = st[f'r_{"amp" if c == "amp" else "hz"}_n']
        ax.annotate(f'r  anch {ra:+.2f}\n     non {rn:+.2f}', (.03, .96),
                    xycoords='axes fraction', va='top', fontsize=6.3,
                    color='0.2' if ra < rn else '#b03030', linespacing=1.2)
        if show_ylab:
            ax.set_ylabel(lab, fontsize=8)
        ax.set_xlabel('Speed (cm/s)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[['top', 'right']].set_visible(False)


if __name__ == '__main__':
    T = load()
    fig = plt.figure(figsize=(3.05 * len(CANDIDATES), 11.2))
    gs = fig.add_gridspec(6, len(CANDIDATES),
                          height_ratios=[1.35, .62, .62, .62, 1.05, 1.05],
                          hspace=.42, wspace=.34)
    for i, (mo, dy) in enumerate(CANDIDATES):
        draw_column(fig, gs, i, T, mo, dy, show_ylab=(i == 0))
    h = [plt.Line2D([], [], marker='s', ls='', ms=6, color=ANCH_COLOR,
                    label='anchored'),
         plt.Line2D([], [], marker='s', ls='', ms=6, color=NONANCH_COLOR,
                    label='non-anchored')]
    fig.legend(handles=h, loc='lower center', ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(.5, -.012))
    out = f'{FIG}/fig6_example_candidates.pdf'
    plt.savefig(out, dpi=180, bbox_inches='tight')
    print('saved', out)
    for mo, dy in CANDIDATES:
        s = session_stats(T, mo, dy)
        print(f'  M{mo}D{dy}: amp {s["d_amp"]:+6.2f}%  hz {s["d_hz"]:+.3f}  '
              f'spd {s["d_spd"]:+5.2f}  Δr_amp {s["r_amp_a"]-s["r_amp_n"]:+.3f}  '
              f'Δr_hz {s["r_hz_a"]-s["r_hz_n"]:+.3f}')
