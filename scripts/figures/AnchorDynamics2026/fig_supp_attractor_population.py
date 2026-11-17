"""Out of the paper (tried as S15, dropped 2026-11-02): recruiting non-grid cells to the bump.

Figure S14 reads the bump from the grid module alone. A cross-validated search
(attractor_population_search.py) asks which other simultaneously recorded MEC
cells carry the module's torus coordinate beyond what position gives, and adds
them to the readout. This figure shows how the search works, what it finds in
one session drawn as S14 draws the module (attractor_recruit_examples.py), and
the tests across sessions (attractor_search_eval.py, attractor_search_drift.py).

    A  the method
    B  the search in the example session, against its null
    C  the bump's phase map on held-out trials (as S14 B, C): the grid module,
       the chosen cells alone, and the two together
    D  three chosen cells: position tuning, and tuning to the module's bump
       with position removed, by state; and on the torus
    E  the bump on the sheet in one snippet per state (as S14 F), module alone
       and with the chosen cells, and its speed
    F  cells chosen per session, real against null
    G  which cells are chosen
    H  held-out bump coherence, module alone and with the chosen cells
    I  the strict test: position removed and slow drift high-passed out,
       agreement of the chosen cells alone with an independent grid half
    J  the bump's speed against the animal's, module alone and with them

Writes fig_supp_attractor_population.pdf
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
FIG = os.path.dirname(os.path.abspath(__file__))
PS = '/Users/harryclark/Documents/spatial-manifolds/data/population_state'
OUT = f'{FIG}/fig_supp_attractor_population.pdf'
ANCH, NON = '#a8559e', '#3f9b8f'
TA = ListedColormap([NON, ANCH])
NORM = BoundaryNorm([-0.5, 0.5, 1.5], TA.N)
REC = '#c99a06'                     # the bump read with the chosen cells
ICOL = {'non-grid spatial': '#3171ae', 'non-spatial': '#888888',
        'putative interneuron': '#d95f02', 'grid': '#c04744'}
SHORT = {'non-grid spatial': 'non-grid\nspatial', 'non-spatial': 'non-\nspatial',
         'putative interneuron': 'inter-\nneuron'}
EXAMPLE = (28, 25)
MIN_RELIABLE = 15                   # grid cells; Figure S14 K, L
HP = 50                             # the high-pass of panel I, in 100 ms bins (5 s)
NB, TL, DT = 100, 200.0, 0.1
rng = np.random.default_rng(0)


def load(sfx):
    out = dict(S=pd.read_csv(f'{PS}/attractor_search_sessions{sfx}.csv'),
               C=pd.read_csv(f'{PS}/attractor_search_cells{sfx}.csv'),
               K=pd.read_csv(f'{PS}/attractor_search_curves{sfx}.csv'),
               E=pd.read_csv(f'{PS}/attractor_search_eval{sfx}.csv'))
    f = f'{PS}/attractor_search_drift{sfx}.csv'
    out['D'] = pd.read_csv(f) if os.path.exists(f) else None
    return out


def lp(ax, s, x=-.22, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, weight='bold', va='bottom', ha='right')


def bare(ax):
    ax.tick_params(labelsize=6.5)
    ax.spines[['top', 'right']].set_visible(False)


def star(p):
    return f'p = {p:.1e}'.replace('e-0', 'e-') if p < 1e-3 else f'p = {p:.2g}'


def mixed_p(d):
    """p for a per-session difference with mouse as a random effect."""
    d = d.dropna().reset_index()
    d.columns = ['mouse', 'day', 'y']
    return smf.mixedlm('y ~ 1', d, groups=d.mouse).fit(reml=True).pvalues.iloc[0]


def ttl(ax, s, fs=7.2, **k):
    ax.set_title(s, fontsize=fs, loc='left', color='0.25', **k)


R_, N_ = load(''), load('_null')
mo, dy = EXAMPLE
Zc = np.load(f'{PS}/attractor_cache/M{mo}D{dy}.npz')
Zr = np.load(f'{PS}/attractor_recruit/M{mo}D{dy}.npz')
cells = list(Zc['cells'])
mod_i = np.array([cells.index(c) for c in Zc['module']])
R = Zc['R'].astype(float)
P = Zc['P'].astype(float).reshape(len(cells), -1, NB)
st = Zr['state'].astype(bool)
ok, anch, non = (Zc[k].astype(bool) for k in ('ok', 'anch', 'non'))
# The two display axes: of the three lattice axes, the pair whose phases are
# most independent in time (a dependent pair draws a stripe on the sheet), the
# one of the two that advances more along the track called 'along'.
_adv = [abs(np.nanmedian(np.unwrap(w, axis=1)[:, -1] - np.unwrap(w, axis=1)[:, 0]))
        for w in Zr['p_mod']]
_tm = Zr['t_mod'][:, ok]
_pair = min(((0, 1), (0, 2), (1, 2)),
            key=lambda p: abs(np.mean(np.exp(1j * (_tm[p[0]] - _tm[p[1]])))))
K_ALONG, K_ACROSS = sorted(_pair, key=lambda k: -_adv[k])
# the example cells: chosen in all five folds, the top score of each identity
_cs = pd.read_csv(f'{PS}/attractor_search_cells.csv')
_cs = _cs[(_cs.mouse == EXAMPLE[0]) & (_cs.day == EXAMPLE[1]) & ~_cs.module]
_cs = _cs.groupby(['cluster_id', 'identity']).agg(sel=('selected', 'mean'), z=('z', 'mean')).reset_index()
EX_CELLS = [int(_cs[(_cs.identity == i) & (_cs.sel == 1)].sort_values('z').cluster_id.iloc[-1])
            for i in ('non-grid spatial', 'non-spatial', 'putative interneuron')
            if ((_cs.identity == i) & (_cs.sel == 1)).any()]
kstar_ex = int(np.median([len(Zr[f'sel{f}']) for f in range(5)]))

fig = plt.figure(figsize=(9.6, 12.6))
gs = fig.add_gridspec(5, 20, hspace=.62, wspace=2.6, left=.055, right=.985, top=.96,
                      bottom=.045, height_ratios=[.95, 1.7, 1.0, 2.2, 1.15])

# ---- A: the method -----------------------------------------------------------
ax = fig.add_subplot(gs[0, 0:13]); ax.axis('off'); ax.set_xlim(0, 4); ax.set_ylim(0, 1)
lp(ax, 'A', x=-.02, y=.98)
STEPS = [('1  the reference bump',
          'the grid module\'s phase\non the torus, moment\nby moment:\n'
          r'$\psi_j(t)=\angle\sum_c r_c(t)\,e^{i\phi_{cj}}$'),
         ('2  score every other cell',
          'how much the bump improves\nthe fit of its rate beyond\nposition (per state) and speed;\n'
          'z against the bump shifted\nin time'),
         ('3  add cells in score order',
          'each cell placed at its own\nphase on the torus; keep the\nnumber that makes the bump\n'
          'most coherent on validation\ntrials'),
         ('4  test on held-out trials',
          'five folds of 8-trial blocks,\nboth states in every fold;\nnothing about a test trial\n'
          'chooses or places a cell')]
for i, (h, b) in enumerate(STEPS):
    x0 = i + .04
    ax.add_patch(FancyBboxPatch((x0, .12), .84, .74, boxstyle='round,pad=0.01,rounding_size=0.04',
                                fc='0.96', ec='0.6', lw=.6))
    ax.text(x0 + .05, .80, h, fontsize=7.2, weight='bold', va='top', color='0.15')
    ax.text(x0 + .05, .64, b, fontsize=6.2, va='top', color='0.25', linespacing=1.3)
    if i < 3:
        ax.add_patch(FancyArrowPatch((x0 + .86, .49), (x0 + .98, .49), arrowstyle='-|>',
                                     mutation_scale=7, color='0.4', lw=.8))
ax.text(.04, .0, 'null: the same search with every non-grid cell shifted in time by at least '
        '60 s, which keeps each cell\'s own statistics and breaks its relation to the bump',
        fontsize=6.2, color='0.35', style='italic')

# ---- B: the search in the example session -----------------------------------
ax = fig.add_subplot(gs[0, 14:20]); lp(ax, 'B', x=-.2)
for D, col, lab in ((R_, 'k', 'recorded'), (N_, '0.65', 'null')):
    k = D['K'][(D['K'].mouse == mo) & (D['K'].day == dy)]
    m = k.groupby('k')['obj'].agg(['mean', 'min', 'max'])
    ax.fill_between(m.index, m['min'], m['max'], color=col, alpha=.18, lw=0, edgecolor='none')
    ax.plot(m.index, m['mean'], color=col, lw=1.2, label=lab)
    ks = D['S'][(D['S'].mouse == mo) & (D['S'].day == dy)].k_star.median()
    ax.axvline(ks, color=col, lw=.8, ls=(0, (3, 2)))
ax.set_xlabel('cells added to the grid module', fontsize=7.0)
ax.set_ylabel('validation coherence\nabove random phases', fontsize=7.0)
ttl(ax, f'M{mo} D{dy}, {len(mod_i)} grid cells: step 3\n(band, five folds; dashed, the size kept)')
ax.legend(fontsize=6, frameon=False, loc='lower right')
bare(ax)

# ---- C: the phase maps, held out ---------------------------------------------
GC = gs[1, :].subgridspec(2, 4, wspace=.12, hspace=.42, width_ratios=[.12, 1, 1, 1])
ntr = P.shape[1]
cmaps = {}
for r_, (kax, lab) in enumerate(((K_ALONG, r'$\psi_\parallel$ along'),
                                 (K_ACROSS, r'$\psi_\perp$ across'))):
    a0 = fig.add_subplot(GC[r_, 0])
    a0.imshow(st[:, None].astype(float), aspect='auto', cmap=TA, norm=NORM,
              interpolation='nearest', extent=[0, 1, ntr, 0])
    a0.set_xticks([]); a0.tick_params(labelsize=6.2)
    a0.set_ylabel(f'trial\n{lab}', fontsize=7.0)
    if r_:
        a0.set_xlabel('state', fontsize=6.4)
    if not r_:
        lp(a0, 'C', x=-1.6, y=1.08)
    for j, (key, name) in enumerate((('p_mod', f'grid module ({len(mod_i)} cells)'),
                                     ('p_sel', f'chosen cells alone (~{kstar_ex})'),
                                     ('p_both', 'module + chosen'))):
        ax = fig.add_subplot(GC[r_, j + 1])
        M = Zr[key][kax]
        ax.imshow(M, cmap='hsv', aspect='auto', interpolation='nearest',
                  extent=[0, TL, ntr, 0], vmin=-np.pi, vmax=np.pi)
        ax.set_yticks([]); ax.set_xticks([0, 100, 200]); ax.tick_params(labelsize=6.2)
        if r_:
            ax.set_xlabel('position (cm)', fontsize=7.0)
        if not r_:
            ttl(ax, name, fs=7.6, pad=13)
            if key == 'p_sel':
                ag = [np.cos(Zr[key][k_] - Zr['p_mod'][k_]) for k_ in (K_ALONG, K_ACROSS)]
                ax.text(0, 1.02, 'agreement with the module (along, across): anchored '
                        f'{np.nanmean(ag[0][st]):.2f}, {np.nanmean(ag[1][st]):.2f}; non-anchored '
                        f'{np.nanmean(ag[0][~st]):.2f}, {np.nanmean(ag[1][~st]):.2f}',
                        transform=ax.transAxes, fontsize=5.8, color='0.35', va='bottom')
ax = fig.axes[-1]
bb = ax.get_position()
cax = fig.add_axes([bb.x1 + .006, bb.y0, .006, bb.height])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-np.pi, np.pi), cmap='hsv'), cax=cax)
cb.set_ticks([-np.pi, 0, np.pi]); cb.set_ticklabels([r'$-\pi$', '0', r'$\pi$'])
cb.ax.tick_params(labelsize=5.8, length=2, pad=1.5); cb.outline.set_linewidth(.4)
cb.set_label('bump phase', fontsize=6.2, labelpad=1)

# ---- D: three chosen cells ---------------------------------------------------
# The module's bump in time (the module's own phases; none of these cells is in
# it), and each cell's rate with its per-state position map (5 cm) removed.
psi_t = np.stack([np.angle(np.exp(1j * q) @ R[mod_i]) for q in Zc['phi_all']])
pb = np.clip((Zc['trackpos'] / TL * 40).astype(int), 0, 39)
CS = R_['C'][(R_['C'].mouse == mo) & (R_['C'].day == dy)]
GD = gs[2, :].subgridspec(1, 11, wspace=.75, width_ratios=[1, 1, .8, .35] * 2 + [1, 1, .8])
NBP = 18
eb = np.linspace(-np.pi, np.pi, NBP + 1); ec = (eb[:-1] + eb[1:]) / 2
for i, cid in enumerate(EX_CELLS):
    c = cells.index(cid)
    idn = str(Zc['ident'][c])
    zc = CS[CS.cluster_id == cid].z.mean()
    res = R[c].copy()
    for s_ in (anch, non):
        for b in range(40):
            m = s_ & (pb == b)
            if m.any():
                res[m] -= R[c][m].mean()
    ax = fig.add_subplot(GD[0, 4 * i])
    x = (np.arange(NB) + .5) * TL / NB
    for msk, col in ((st, ANCH), (~st, NON)):
        y = P[c][msk]
        mu, se = y.mean(0), y.std(0) / np.sqrt(len(y))
        ax.fill_between(x, mu - se, mu + se, color=col, alpha=.25, lw=0, edgecolor='none')
        ax.plot(x, mu, color=col, lw=1)
    ax.set_xticks([0, 100, 200]); ax.set_xlabel('position (cm)', fontsize=6.6)
    ax.set_ylabel('rate (z)', fontsize=6.6, labelpad=1)
    bare(ax); ax.tick_params(labelsize=5.8)
    ax.set_title(f'{SHORT[idn].replace(chr(10), " ").replace("- ", "-")} {cid}, score z = {zc:.0f}',
                 fontsize=7.0, loc='left', color=ICOL[idn])
    if not i:
        lp(ax, 'D', x=-.42, y=1.06)
    ax = fig.add_subplot(GD[0, 4 * i + 1])
    for msk, col in ((anch, ANCH), (non, NON)):
        k_ = np.digitize(psi_t[K_ALONG][msk], eb) - 1
        v = res[msk]
        mu = np.array([v[k_ == b].mean() for b in range(NBP)])
        se = np.array([v[k_ == b].std() / np.sqrt((k_ == b).sum()) for b in range(NBP)])
        ax.fill_between(ec, mu - se, mu + se, color=col, alpha=.25, lw=0, edgecolor='none')
        ax.plot(ec, mu, color=col, lw=1)
    ax.axhline(0, color='0.8', lw=.5, zorder=0)
    ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
    ax.set_xlabel(r'module bump, $\psi_\parallel$', fontsize=6.6)
    ax.set_ylabel('rate, pos. removed (z)', fontsize=6.0, labelpad=1)
    bare(ax); ax.tick_params(labelsize=5.8)
    ax = fig.add_subplot(GD[0, 4 * i + 2])
    H = np.zeros((12, 12)); Nn = np.zeros((12, 12))
    e12 = np.linspace(-np.pi, np.pi, 13)
    u = np.clip(np.digitize(psi_t[K_ALONG][ok], e12) - 1, 0, 11)
    w = np.clip(np.digitize(psi_t[K_ACROSS][ok], e12) - 1, 0, 11)
    np.add.at(H, (w, u), res[ok]); np.add.at(Nn, (w, u), 1)
    ax.imshow(H / np.maximum(Nn, 1), origin='lower', cmap='magma', extent=[-np.pi, np.pi] * 2,
              interpolation='bicubic')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(r'$\psi_\parallel$', fontsize=6.4, labelpad=1)
    ax.set_ylabel(r'$\psi_\perp$', fontsize=6.4, labelpad=1)
    ax.set_title('on the torus', fontsize=6.2, color='0.35', pad=2)

# ---- E: the bump on the sheet, module alone and with the chosen cells --------
# Snippets chosen by the MODULE's coherence above random phases (as S14 F), so
# the choice cannot favour the recruited readout.
phi = Zc['phi']
Rm = R[mod_i]
nrm = np.linalg.norm(Rm, axis=0) + 1e-9
coh = np.mean([np.abs(np.exp(1j * q) @ Rm) for q in phi], 0) / (np.sqrt(len(mod_i) / 2) * nrm)
sh = np.mean([np.mean([np.abs(np.exp(1j * rng.uniform(-np.pi, np.pi, len(mod_i))) @ Rm)
                       for _ in range(2)], 0) for _ in range(10)], 0) / (np.sqrt(len(mod_i) / 2) * nrm)
SNIP, NF, STEP = 48, 24, 2
ft = Zr['fold_t']


def best(mask):
    good = np.convolve(mask.astype(float), np.ones(SNIP), 'valid') == SNIP
    sc = np.convolve(coh - sh, np.ones(SNIP) / SNIP, 'valid')
    same = np.array([ft[i] == ft[i + SNIP - 1] for i in range(len(sc))])
    sc[~(good & same)] = -np.inf
    return int(np.argmax(sc))


_g = np.linspace(-np.pi, np.pi, 30, endpoint=False)
_GU, _GV = np.meshgrid(_g, _g, indexing='ij')
_e = np.linspace(-np.pi, np.pi, 31)
_EU, _EV = np.meshgrid(_e, _e, indexing='ij')
_X0, _Y0 = (_EU - _EV / 2) / (2 * np.pi), (np.sqrt(3) / 2 * _EV) / (2 * np.pi)


def kernel(ph):
    d1 = (ph[0][:, None, None] - _GU[None] + np.pi) % (2 * np.pi) - np.pi
    d2 = (ph[1][:, None, None] - _GV[None] + np.pi) % (2 * np.pi) - np.pi
    K = np.exp(-(d1 ** 2 + d2 ** 2) / (2 * .62 ** 2))
    return K / K.sum(0)


t = np.arange(R.shape[1]) * DT
run = ok & np.isfinite(Zr['bs_mod']) & np.isfinite(Zr['bs_both'])
zs = lambda v: (v - np.nanmean(v[run])) / (np.nanstd(v[run]) + 1e-9)
ZA, ZM, ZB = zs(Zc['spd'].astype(float)), zs(Zr['bs_mod']), zs(Zr['bs_both'])
GE = gs[3, :].subgridspec(6, 1, hspace=.55, height_ratios=[.8, .8, 1.0, .8, .8, 1.0])
for r_, (nm, col, msk) in enumerate((('anchored', ANCH, anch), ('non-anchored', NON, non))):
    i0 = best(msk)
    f_ = int(ft[i0])
    sel = Zr[f'sel{f_}']
    ph = Zr[f'ph{f_}'][[K_ALONG, K_ACROSS]]
    for j, (ii, lab) in enumerate(((mod_i, f'grid module ({len(mod_i)})'),
                                   (np.r_[mod_i, sel], f'module + chosen ({len(mod_i) + len(sel)})'))):
        K = kernel(ph[:, ii])
        ax = fig.add_subplot(GE[3 * r_ + j])
        for k in range(NF):
            F = (K * R[ii][:, i0 + STEP * k][:, None, None]).sum(0)
            F = (F - F.min()) / (F.max() - F.min() + 1e-9)
            ax.pcolormesh(_X0 + k * 1.62, _Y0, F, cmap='magma', shading='flat', rasterized=True)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_xlim(-.62, (NF - 1) * 1.62 + 1.0)
        ax.text(-.75, .43, lab, fontsize=6.4, ha='right', va='center',
                color='k' if not j else REC, weight='bold')
        if not j:
            ax.set_title(f'{nm}: every 200 ms for 4.8 s (t = {t[i0]:.0f} s, a held-out trial)',
                         fontsize=7.4, loc='left', color=col, pad=3)
        if not r_ and not j:
            lp(ax, 'E', x=-.105, y=1.05)
    av = fig.add_subplot(GE[3 * r_ + 2])
    c0 = t[i0] + SNIP * DT / 2
    w_ = (t >= c0 - 15) & (t <= c0 + 15)
    for v, cc, lab in ((ZA, '0.6', 'the animal'), (ZM, 'k', 'bump, module'),
                       (ZB, REC, 'bump, module + chosen')):
        av.plot(t[w_] - c0, np.where(ok, v, np.nan)[w_], color=cc, lw=.85, label=lab)
    av.axvspan(t[i0] - c0, t[i0 + SNIP - 1] - c0, color=col, alpha=.18, lw=0)
    av.axhline(0, color='0.8', lw=.5, zorder=0)
    av.set_xlim(-15, 15); av.set_ylim(-3, 3.8)
    av.set_ylabel('speed (z)', fontsize=6.8)
    bare(av); av.tick_params(labelsize=6.0)
    m_ = w_ & run
    av.text(.99, .98, f'r with the animal over these 30 s: module '
            f'{np.corrcoef(ZA[m_], ZM[m_])[0, 1]:+.2f}, + chosen {np.corrcoef(ZA[m_], ZB[m_])[0, 1]:+.2f}',
            transform=av.transAxes, fontsize=5.8, ha='right', va='top', color='0.3')
    if not r_:
        av.tick_params(labelbottom=False)
        av.legend(fontsize=5.8, frameon=False, loc='upper left', ncol=3, handlelength=1.0,
                  columnspacing=.9, borderaxespad=.1)
    else:
        av.set_xlabel('time from the snippet (s)', fontsize=6.8)

# ---- F-J: across sessions ----------------------------------------------------
GS = gs[4, :].subgridspec(1, 5, wspace=.75, width_ratios=[.75, 1, 1.1, 1.5, 1.1])
small = R_['S'].groupby(['mouse', 'day']).n_mod.first() < MIN_RELIABLE

# F: cells chosen per session
ax = fig.add_subplot(GS[0]); lp(ax, 'F', x=-.42)
kr = R_['S'].groupby(['mouse', 'day']).k_star.median()
kn = N_['S'].groupby(['mouse', 'day']).k_star.median().reindex(kr.index)
for a_, b_ in zip(kr, kn):
    ax.plot([0, 1], [a_, b_], color='0.85', lw=.6)
for x, v, c in ((0, kr, 'k'), (1, kn, '0.6')):
    ax.scatter(np.full(len(v), x) + rng.uniform(-.07, .07, len(v)), v, s=9, color=c, lw=0, zorder=3)
    ax.plot([x - .2, x + .2], [np.nanmedian(v)] * 2, color=c, lw=2)
ax.set_xticks([0, 1]); ax.set_xticklabels(['recorded', 'null'], fontsize=6.6); ax.set_xlim(-.5, 1.5)
ax.set_ylabel('cells chosen (median of folds)', fontsize=6.8)
ttl(ax, f'{len(kr)} sessions: {np.median(kr):.0f} vs\n{np.nanmedian(kn):.0f} '
        f'({star(wilcoxon(kr, kn).pvalue)})', fs=6.8)
bare(ax)

# G: which cells
ax = fig.add_subplot(GS[1]); lp(ax, 'G', x=-.3)
order = ['non-grid spatial', 'non-spatial', 'putative interneuron']
for j, (D, a_) in enumerate(((R_, 1.), (N_, .35))):
    c = D['C'][~D['C'].module & D['C'].identity.isin(order)]
    g = c.groupby(['mouse', 'day', 'cluster_id', 'identity']).selected.mean().reset_index()
    fr = g.assign(s=g.selected >= .6).groupby('identity').s.mean().reindex(order)
    ax.bar(np.arange(3) + (j - .5) * .38, 100 * fr.values, width=.38,
           color=[ICOL[o] for o in order], alpha=a_, lw=0)
    if not j:
        nn = g.groupby('identity').size().reindex(order)
        for i_, (v, n_) in enumerate(zip(fr.values, nn.values)):
            ax.text(i_ - .19, 100 * v + 1, f'{n_}', fontsize=5.4, ha='center', color='0.35')
ax.set_xticks(range(3)); ax.set_xticklabels([SHORT[o] for o in order], fontsize=6.2)
ax.set_ylabel('% chosen in at least\n3 of 5 folds', fontsize=6.8)
ttl(ax, 'solid, recorded; faint, null\n(cells tested above the bars)', fs=6.8)
bare(ax)

# H: held-out coherence
ax = fig.add_subplot(GS[2]); lp(ax, 'H', x=-.3)
S_ = R_['S'].groupby(['mouse', 'day']).mean(numeric_only=True)
xs = {'mod_a': 0, 'both_a': 1, 'mod_n': 2.4, 'both_n': 3.4}
for _, r in S_.iterrows():
    ax.plot([0, 1], [r.mod_a, r.both_a], color='0.85', lw=.6)
    ax.plot([2.4, 3.4], [r.mod_n, r.both_n], color='0.85', lw=.6)
for k_, x in xs.items():
    c = ANCH if k_.endswith('_a') else NON
    ax.scatter(np.full(len(S_), x), S_[k_], s=8, color=c, lw=0, zorder=3,
               alpha=.5 if k_.startswith('mod') else 1)
    ax.plot([x - .2, x + .2], [S_[k_].median()] * 2, color=c, lw=2)
ax.axhline(0, color='k', lw=.6, ls=(0, (3, 2)))
ax.set_xticks(list(xs.values()))
ax.set_xticklabels(['module', '+\nchosen', 'module', '+\nchosen'], fontsize=6.0)
ax.set_ylabel('held-out bump\nabove phase shuffle', fontsize=6.8)
pa = mixed_p(S_.both_a - S_.mod_a)
pn = mixed_p(S_.both_n - S_.mod_n)
ttl(ax, f'anchored {S_.mod_a.median():.3f} to {S_.both_a.median():.3f}\n({star(pa)}); '
        f'non-anchored\n{S_.mod_n.median():.3f} to {S_.both_n.median():.3f} ({star(pn)})', fs=6.6)
bare(ax)

# I: the strict test
ax = fig.add_subplot(GS[3]); lp(ax, 'I', x=-.2)
D0 = R_['D'][R_['D'].sig == HP].set_index(['mouse', 'day'])
DN = N_['D'][N_['D'].sig == HP].set_index(['mouse', 'day']) if N_['D'] is not None else None
series = [('half', 'grid half A', '0.3', D0), ('sel', 'chosen', 'k', D0),
          ('un', 'unchosen', '0.6', D0)] + ([('sel', 'null chosen', '0.8', DN)] if DN is not None else [])
txt = []
for s_i, (stt, c_st) in enumerate((('a', ANCH), ('n', NON))):
    base = s_i * (len(series) + .8)
    for j, (k_, lab, c, src) in enumerate(series):
        v = src[f'{k_}_{stt}'].dropna()
        x = base + j
        ax.scatter(np.full(len(v), x) + rng.uniform(-.12, .12, len(v)), v, s=7, color=c, lw=0,
                   zorder=3)
        ax.plot([x - .28, x + .28], [v.median()] * 2, color=c_st, lw=2, zorder=4)
    e = D0.dropna(subset=[f'sel_{stt}', f'half_{stt}', f'un_{stt}'])
    txt.append(f'{"anchored" if stt == "a" else "non-anch."}: chosen {e[f"sel_{stt}"].median():.2f}, '
               f'half A {e[f"half_{stt}"].median():.2f} ({star(mixed_p(e[f"sel_{stt}"] - e[f"half_{stt}"]))});\n'
               f'   vs unchosen {star(mixed_p(e[f"sel_{stt}"] - e[f"un_{stt}"]))}')
ax.axhline(0, color='k', lw=.6, ls=(0, (3, 2)))
lbl = [s[1] for s in series]
ax.set_xticks(list(range(len(lbl))) + [len(lbl) + .8 + i for i in range(len(lbl))])
ax.set_xticklabels(lbl * 2, fontsize=5.8, rotation=90)
ax.set_ylabel('agreement with grid half B,\nposition and drift removed', fontsize=6.8)
ttl(ax, '\n'.join(txt), fs=6.2)
ax.text(.22, -.58, 'anchored', transform=ax.transAxes, fontsize=6.6, color=ANCH, ha='center')
ax.text(.76, -.58, 'non-anchored', transform=ax.transAxes, fontsize=6.6, color=NON, ha='center')
bare(ax)

# J: the bump's speed
ax = fig.add_subplot(GS[4]); lp(ax, 'J', x=-.3)
V = pd.read_csv(f'{PS}/attractor_recruit_speed.csv').set_index(['mouse', 'day'])
sm = (V.n_mod < MIN_RELIABLE).values
pv = []
for s_i, (stt, c_st) in enumerate((('a', ANCH), ('n', NON))):
    a_ = V[f'r_mod_{stt}'] - V[f'rnull_mod_{stt}']
    b_ = V[f'r_both_{stt}'] - V[f'rnull_both_{stt}']
    x0 = s_i * 2.4
    for y0, y1, s0 in zip(a_, b_, sm):
        ax.plot([x0, x0 + 1], [y0, y1], color='0.85', lw=.6)
    for x, v in ((x0, a_), (x0 + 1, b_)):
        ax.scatter(np.full(len(v), x)[~sm], v[~sm], s=8, color=c_st, lw=0, zorder=3)
        ax.scatter(np.full(len(v), x)[sm], v[sm], s=8, facecolor='none', edgecolor=c_st, lw=.6,
                   zorder=3)
        ax.plot([x - .2, x + .2], [np.median(v)] * 2, color=c_st, lw=2)
    pv.append((np.median(a_), np.median(b_), mixed_p(b_ - a_)))
ax.axhline(0, color='k', lw=.6, ls=(0, (3, 2)))
ax.set_xticks([0, 1, 2.4, 3.4])
ax.set_xticklabels(['module', '+\nchosen', 'module', '+\nchosen'], fontsize=6.0)
ax.set_ylabel('bump speed vs animal\'s,\nr above phase shuffle', fontsize=6.8)
ttl(ax, f'anchored {pv[0][0]:.2f} to {pv[0][1]:.2f} ({star(pv[0][2])})\n'
        f'non-anchored {pv[1][0]:.2f} to {pv[1][1]:.2f} ({star(pv[1][2])})', fs=6.4)
bare(ax)

fig.savefig(OUT, dpi=250, bbox_inches='tight')
print('wrote', OUT)
