"""Figure S16: theta-nested spectral components and the anchoring state.

Following Lopes-dos-Santos et al. (2018) and Esparza et al. (2023), every theta
cycle of the entorhinal LFP is described by where in frequency (20-120 Hz) and
theta phase its gamma power sits, and the cycles are decomposed into a few
recurring kinds -- theta-nested spectral components (tSCs). The question for
Figure 8 is whether anchored and non-anchored trials are built from different
kinds of cycle, and whether the kind of cycle tracks what the MEC population
encodes on it.

All maps are the INDUCED maps of tsc_extract.py (theta-locked waveform removed,
so non-sinusoidal theta cannot pose as phase-locked gamma), normalised as in
tsc_analyse.py. Components: tsc_analyse.fit, K = 4 by split-half stability.

(A) The four templates. (B) Speed-matched state difference in each component's
weight and in the cycle's overall gamma level, per session, in units of that
session's SD. (C) How well state can be read from the cycle: cross-validated
AUC of a logistic regression on the four weights plus level, per session, on
single cycles and on trial means. (D) Within each state, partial correlation of
each weight with the cycle's decoding error (tsc_neural.py). (E) One session's
per-cycle population vectors embedded with UMAP and coloured three ways.

Per-session numbers are cached in data/lfp/tsc_figure_cache.npz.
"""
import glob
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore')
FIG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FIG)
import tsc_analyse as A
from spatial_manifolds.anchoring import ANCH_COLOR, NONANCH_COLOR

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
LFP = f'{ROOT}/data/lfp'
OUT = f'{FIG}/fig8_supp_tsc.pdf'
CACHE = f'{LFP}/tsc_figure_cache.npz'
EX = 'M28D25'                      # the manifold example (also in Figure 4)
# Component colours avoid red and blue (cell types) and the state purple/teal.
CCOL = ['#d95f02', '#e6ab02', '#e7298a', '#8c8c8c']
LCOL = '0.15'


def session_stats(files, project):
    """Speed-matched state difference (SD units) and state AUC, per session."""
    rng = np.random.default_rng(A.SEED)
    rows = []
    for fn in files:
        x, level, meta = A.load_session(fn, 'ind')
        if 'anch' not in meta:
            continue
        anch = meta['anch'].astype(bool)
        if anch.all() or (~anch).all():
            continue
        W = np.c_[project(x), level]
        W = (W - W.mean(0)) / W.std(0)
        idx = A.speed_matched_idx(meta['speed'], anch, rng)
        if len(idx) < 40:
            continue
        a, Wm, tr = anch[idx], W[idx], meta['trial'][idx]
        r = dict(session=os.path.basename(fn)[:-4], n=len(idx))
        for c in range(W.shape[1]):
            r[f'd{c}'] = Wm[a, c].mean() - Wm[~a, c].mean()
        # state from the cycle: logistic regression, folds by trial so no
        # trial's cycles are on both sides
        ntr = len(np.unique(tr))
        if ntr >= 10 and a.sum() >= 20 and (~a).sum() >= 20:
            # A fold whose training trials hold one state only cannot be
            # scored; rather than leave its cycles at p = 0 (which scored such
            # sessions at AUC 0), the session is left out of C.
            p = np.full(len(idx), np.nan)
            for trn, tst in GroupKFold(5).split(Wm, a, tr):
                if len(np.unique(a[trn])) < 2:
                    break
                p[tst] = LogisticRegression(max_iter=500).fit(Wm[trn], a[trn]).predict_proba(Wm[tst])[:, 1]
            if np.isnan(p).any():
                rows.append(r); continue
            r['auc_cycle'] = roc_auc_score(a, p)
            # the same predictions averaged over each trial's cycles
            d = pd.DataFrame(dict(t=tr, a=a, p=p)).groupby('t').mean()
            if (d.a > .5).sum() >= 3 and (d.a <= .5).sum() >= 3:
                r['auc_trial'] = roc_auc_score(d.a > .5, d.p)
        rows.append(r)
    return pd.DataFrame(rows)


def manifold(key):
    import umap
    z = np.load(f'{LFP}/tsc_neural_cycles/{key}.npz')
    rng = np.random.default_rng(0)
    i = np.sort(rng.choice(len(z['err']), min(5000, len(z['err'])), replace=False))
    C = np.sqrt(z['counts'][i].astype(float))
    C = C[:, C.std(0) > 0]
    C = (C - C.mean(0)) / C.std(0)
    from sklearn.decomposition import PCA
    Y = umap.UMAP(n_neighbors=30, min_dist=.3, random_state=0).fit_transform(
        PCA(10, random_state=0).fit_transform(C))
    return Y, z['pos'][i], z['anch'][i], z['W'][i].argmax(1)


def build_cache(files):
    rng = np.random.default_rng(A.SEED)
    S = [A.load_session(f, 'ind')[0] for f in files]
    X = np.concatenate([x[rng.choice(len(x), min(A.N_PER_SESSION, len(x)), replace=False)]
                        for x in S])
    del S
    T, project = A.fit(X, A.K, A.SEED)
    D = session_stats(files, project)
    Y, pos, anch, dom = manifold(EX)
    np.savez_compressed(CACHE + '.partial.npz', T=T, D=D.to_records(index=False), Y=Y,
                        pos=pos, anch=anch, dom=dom)
    os.replace(CACHE + '.partial.npz', CACHE)


def _box(ax, data, xs, cols, w=.55):
    for x, v, c in zip(xs, data, cols):
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        ax.scatter(x + np.random.default_rng(int(x * 10) + 100).uniform(-.16, .16, len(v)), v,
                   s=6, color=c, alpha=.55, lw=0, zorder=2)
        ax.plot([x - w / 2, x + w / 2], [np.median(v)] * 2, color='k', lw=1.4, zorder=3)


if __name__ == '__main__':
    files = sorted(f for f in glob.glob(f'{LFP}/tsc/M*D*.npz') if 'partial' not in f)
    if not os.path.exists(CACHE) or '--rebuild' in sys.argv:
        build_cache(files)
    z = np.load(CACHE, allow_pickle=True)
    T, D = z['T'], pd.DataFrame(z['D'])
    freqs = np.load(f'{LFP}/tsc_components.npz')['freqs']
    nph = int(np.load(f'{LFP}/tsc_components.npz')['nph'])
    N = pd.read_csv(f'{LFP}/tsc_neural.csv')
    R = pd.read_csv(f'{LFP}/tsc_components_state.csv')
    K = len(T)
    ph = (np.arange(nph) + .5) * 360 / nph

    fig = plt.figure(figsize=(10.2, 7.4))
    outer = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.05], hspace=.62,
                             left=.085, right=.95, top=.95, bottom=.07)

    # ---- A: templates -----------------------------------------------------
    gA = outer[0].subgridspec(1, K + 1, width_ratios=[1] * K + [.05], wspace=.25)
    vm = np.abs(T).max()
    labels = []
    for c in range(K):
        M = T[c].reshape(len(freqs), nph)
        fi, pi = np.unravel_index(M.argmax(), M.shape)
        labels.append(f'tSC{c + 1}')
        ax = fig.add_subplot(gA[c])
        im = ax.pcolormesh(np.r_[0, ph + 180 / nph], np.geomspace(freqs[0] / (freqs[1] / freqs[0]) ** .5,
                           freqs[-1] * (freqs[1] / freqs[0]) ** .5, len(freqs) + 1),
                           M, cmap='PuOr_r', vmin=-vm, vmax=vm, shading='flat')
        ax.set_yscale('log'); ax.set_yticks([20, 40, 80]); ax.set_yticklabels(['20', '40', '80'])
        ax.minorticks_off()
        ax2 = ax.twinx()
        ax2.plot(np.linspace(0, 360, 100), np.cos(np.deg2rad(np.linspace(0, 360, 100))),
                 color='w', lw=1.2)
        ax2.set_ylim(-4, 1.3); ax2.axis('off')
        ax.set_xticks([0, 180, 360]); ax.tick_params(labelsize=6.5)
        ax.set_xlabel('Theta phase (deg)', fontsize=7.5)
        if c == 0:
            ax.set_ylabel('Frequency (Hz)', fontsize=8)
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(f'{labels[c]}: {freqs[fi]:.0f} Hz at {ph[pi]:.0f} deg\n'
                     f'split-half r = {R.stability[c]:.2f}', fontsize=7, color=CCOL[c],
                     linespacing=1.15, pad=3)
        if c == 0:
            ax.text(-.32, 1.12, 'A', transform=ax.transAxes, fontsize=10, weight='bold')
    cb = fig.colorbar(im, cax=fig.add_subplot(gA[K]))
    cb.set_ticks([]); cb.outline.set_visible(False)
    cb.ax.set_title('weight', fontsize=6, pad=3)

    # ---- B, C, D ----------------------------------------------------------
    gM = outer[1].subgridspec(1, 3, width_ratios=[1.15, .6, 1.4], wspace=.42)
    ax = fig.add_subplot(gM[0])
    names = labels + ['level']
    cols = CCOL + [LCOL]
    dat = [D[f'd{c}'].values for c in range(K + 1)]
    _box(ax, dat, np.arange(K + 1), cols)
    ax.axhline(0, color='0.5', lw=.7)
    ps = [wilcoxon(v[np.isfinite(v)]).pvalue for v in dat]
    for x, p, h in zip(range(K + 1), ps, A.holm(ps)):
        ax.text(x, 1.0, f'{np.median(dat[x]):+.2f}\np={h:.2g}', transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=5.6, linespacing=1.05)
        print(f'B {names[x]}: median {np.median(dat[x]):+.3f} SD, p = {p:.3g}, Holm {h:.3g}, n = {len(dat[x])}')
    ax.set_xticks(range(K + 1)); ax.set_xticklabels(names, fontsize=6.5, rotation=30)
    ax.set_ylabel('anchored - non-anchored\n(SD, speed-matched)', fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.text(-.27, 1.17, 'B', transform=ax.transAxes, fontsize=10, weight='bold')

    ax = fig.add_subplot(gM[1])
    au = [D.auc_cycle.dropna().values, D.auc_trial.dropna().values]
    _box(ax, au, [0, 1], ['0.35', '0.35'])
    ax.axhline(.5, color='0.5', lw=.7, ls='--')
    for x, v in enumerate(au):
        ax.text(x, 1.0, f'{np.median(v):.3f}\np={wilcoxon(v - .5).pvalue:.2g}',
                transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=5.6,
                linespacing=1.05)
        print(f'C AUC {["cycle", "trial"][x]}: median {np.median(v):.3f}, n = {len(v)}, '
              f'p = {wilcoxon(v - .5).pvalue:.3g}')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['cycle', 'trial'], fontsize=6.5)
    ax.set_xlim(-.6, 1.6)
    ax.set_ylabel('State from tSCs (AUC)', fontsize=7.5); ax.tick_params(labelsize=6.5)
    ax.text(-.6, 1.17, 'C', transform=ax.transAxes, fontsize=10, weight='bold')

    ax = fig.add_subplot(gM[2])
    xs, dd, cc = [], [], []
    for si, (st, off) in enumerate((('anch', -.2), ('non', .2))):
        for c, nm in enumerate([0, 1, 2, 3, 'level']):
            v = N[(N.state == st) & (N.component.astype(str) == str(nm))].rho.dropna().values
            xs.append(c + off); dd.append(v)
            cc.append(ANCH_COLOR if st == 'anch' else NONANCH_COLOR)
    _box(ax, dd, xs, cc, w=.3)
    ax.axhline(0, color='0.5', lw=.7)
    for si, st in enumerate(('anch', 'non')):
        ps = [wilcoxon(v).pvalue for v in dd[si * 5:(si + 1) * 5]]
        for c, (p, h) in enumerate(zip(ps, A.holm(ps))):
            print(f'D {st} {names[c]}: median rho {np.median(dd[si * 5 + c]):+.4f}, p = {p:.3g}, Holm {h:.3g}')
            if h < .05:
                ax.text(c + (-.2 if si == 0 else .2), 1.0, '*', transform=ax.get_xaxis_transform(),
                        ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(K + 1)); ax.set_xticklabels(names, fontsize=6.5, rotation=30)
    ax.set_ylabel('partial rho, weight vs\ndecoding error', fontsize=7.5); ax.tick_params(labelsize=6.5)
    ax.scatter([], [], color=ANCH_COLOR, s=8, label='anchored')
    ax.scatter([], [], color=NONANCH_COLOR, s=8, label='non-anchored')
    ax.legend(fontsize=6, frameon=False, loc='lower right', handletextpad=.2)
    ax.text(-.22, 1.17, 'D', transform=ax.transAxes, fontsize=10, weight='bold')

    # ---- E: manifold ------------------------------------------------------
    gE = outer[2].subgridspec(1, 3, wspace=.3)
    Y, pos, anch, dom = z['Y'], z['pos'], z['anch'].astype(bool), z['dom']
    for k, (title, c, kw) in enumerate((
            ('position', pos, dict(cmap='twilight', vmin=0, vmax=200)),
            ('state', np.where(anch, 1, 0), dict(cmap=matplotlib.colors.ListedColormap(
                [NONANCH_COLOR, ANCH_COLOR]), vmin=0, vmax=1)),
            ('largest tSC weight', dom, dict(cmap=matplotlib.colors.ListedColormap(CCOL),
                                              vmin=-.5, vmax=K - .5)))):
        ax = fig.add_subplot(gE[k])
        o = np.random.default_rng(k).permutation(len(Y))
        sc = ax.scatter(Y[o, 0], Y[o, 1], c=c[o], s=1.2, lw=0, rasterized=True, **kw)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal', 'datalim')
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=7.5, pad=2)
        if k == 0:
            ax.text(-.12, 1.08, 'E', transform=ax.transAxes, fontsize=10, weight='bold')
            ax.text(0, -.04, f'{EX[:3]} {EX[3:]}, {len(Y)} theta cycles, UMAP of population vectors',
                    transform=ax.transAxes, fontsize=6, va='top', color='0.35')
            cb = fig.colorbar(sc, ax=ax, shrink=.6, pad=.01, ticks=[0, 100, 200])
            cb.ax.tick_params(labelsize=5.5); cb.outline.set_visible(False)
            cb.set_label('cm', fontsize=6)
        if k == 2:
            for c_ in range(K):
                ax.scatter([], [], color=CCOL[c_], s=8, label=labels[c_])
            ax.legend(fontsize=5.8, frameon=False, loc='center left', bbox_to_anchor=(1, .5),
                      handletextpad=.1)
        if k == 1:
            print(f'E state: {anch.mean():.0%} anchored cycles')
    tmp = OUT + '.partial'
    fig.savefig(tmp, format='pdf', dpi=300); os.replace(tmp, OUT)
    print('wrote', OUT)
