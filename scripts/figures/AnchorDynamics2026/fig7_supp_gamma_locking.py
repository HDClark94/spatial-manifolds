"""Figure 7 supplement: what spikes do to gamma, and what gamma does to them.

Figure 8's model has fast-spiking interneurons generating the gamma that nests
in theta, and Figure 2 has interneurons among the cells that follow the
anchoring state. Those meet only if the interneurons here are coupled to the
gamma in this field potential. Computed by gamma_locking.py.

    A  the control the whole figure rests on: coupling against distance
    B  coupling by identity and band
    C  where in the gamma cycle each identity fires
    D  the one positive result, per session
    E  the state comparison, which is null

WHY A COMES FIRST. A fast-spiking cell's waveform leaks into the local field at
exactly the frequencies being tested, so a cell measured against its own
channel is partly correlating with itself. Panel A is that artifact drawn to
scale: own-channel coupling runs two to four times the value 120 um away. Every
other panel uses the nearest group at least 120 um from the cell -- nearest
rather than furthest, because gamma is local and the far end of the entorhinal
span would manufacture a null as surely as the own channel manufactures a
result.

PPC, NOT MRL, throughout: interneurons fire about five times more than
principal cells and the mean resultant length rises as spike count falls, so an
MRL comparison between the two would mostly compare firing rates.

Writes fig7_supp_gamma_locking.pdf
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = os.path.dirname(os.path.abspath(__file__))
OUT = f'{FIG}/fig7_supp_gamma_locking.pdf'
ANCH_COLOR, NONANCH_COLOR = '#a8559e', '#3f9b8f'
ICOL = {'grid': '#c04744', 'non-grid spatial': '#3171ae',
        'non-spatial': '#888888', 'putative interneuron': '#d95f02'}
ORDER = ['grid', 'non-grid spatial', 'non-spatial', 'putative interneuron']
SHORT = {'grid': 'grid', 'non-grid spatial': 'NGS', 'non-spatial': 'NS',
         'putative interneuron': 'int'}
BANDS = [('slow', 'slow gamma\n30–48 Hz'), ('fast', 'fast gamma\n60–100 Hz')]


def lp(ax, s, x=-.26, y=1.0):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='right')


def tidy(ax):
    ax.tick_params(labelsize=6.5)
    ax.spines[['top', 'right']].set_visible(False)


D = pd.read_csv(f'{ROOT}/data/lfp/gamma_locking.csv')
D = D[D.identity.isin(ORDER)].copy()
NS_ = D.groupby(['mouse', 'day']).ngroups
print(f'{len(D)} cells, {NS_} sessions')
print(D.identity.value_counts().to_string())

fig = plt.figure(figsize=(10, 5.0))
G = fig.add_gridspec(2, 3, hspace=.72, wspace=.46, left=.075, right=.985,
                     top=.845, bottom=.11)

# ---- A: coupling against distance ------------------------------------------
ax = fig.add_subplot(G[0, 0]); lp(ax, 'A')
STEPS = [('own', 'own\nchannel'), ('ppc', '120 µm'), ('far', '1.2 mm')]
for k in ORDER:
    g = D[D.identity == k]
    v = [g[f'{p}_fast_a'].median() for p, _ in STEPS]
    ax.plot(range(3), v, 'o-', ms=4, lw=1.3, color=ICOL[k], label=SHORT[k])
ax.axhline(0, color='0.7', lw=.7, ls=':')
ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in STEPS], fontsize=6.2,
                                            linespacing=1.2)
ax.set_xlim(-.3, 2.3)
ax.set_ylabel('PPC with fast gamma', fontsize=7.5)
ax.set_title('the leak, drawn to scale:\ncoupling falls off with distance',
             fontsize=7.2, loc='left')
ax.legend(fontsize=5.8, frameon=False, loc='upper right', ncol=2,
          columnspacing=.8, handlelength=1.1)
tidy(ax)

# ---- B: coupling by identity and band --------------------------------------
ax = fig.add_subplot(G[0, 1]); lp(ax, 'B')
w = .19
for j, (bn, lab) in enumerate(BANDS):
    for i, k in enumerate(ORDER):
        s = D[D.identity == k].groupby(['mouse', 'day'])[f'ppc_{bn}_a'].median()
        ax.bar(j + (i - 1.5) * w, s.median(), w * .9, color=ICOL[k], lw=0)
        ax.errorbar(j + (i - 1.5) * w, s.median(),
                    yerr=[[s.median() - s.quantile(.25)],
                          [s.quantile(.75) - s.median()]],
                    fmt='none', ecolor='0.35', lw=.7, capsize=1.5)
ax.set_xticks(range(2)); ax.set_xticklabels([l for _, l in BANDS], fontsize=6.5,
                                            linespacing=1.2)
ax.set_ylabel('PPC, 120 µm away', fontsize=7.5)
ax.set_title('interneurons are not the most\ncoupled class — grid cells are',
             fontsize=7.2, loc='left')
for j, (bn, _) in enumerate(BANDS):
    s = D.assign(i=D.identity == 'putative interneuron').groupby(
        ['mouse', 'day', 'i'])[f'ppc_{bn}_a'].median().unstack().dropna()
    p = wilcoxon(s[True], s[False]).pvalue
    ax.text(j, ax.get_ylim()[1] * .97, f'int vs rest\np = {p:.2f}', ha='center',
            va='top', fontsize=5.6, color='0.3', linespacing=1.2)
    print(f'  {bn}: interneurons vs principal, per session p = {p:.3g}')
tidy(ax)

# ---- C: preferred phase ----------------------------------------------------
ax = fig.add_subplot(G[0, 2], projection='polar'); lp(ax, 'C', x=-.20, y=1.02)
for k in ORDER:
    v = D[D.identity == k]['phi_fast_a'].dropna().values
    z = np.mean(np.exp(1j * v))
    ax.annotate('', xy=(np.angle(z), np.abs(z)), xytext=(0, 0),
                arrowprops=dict(arrowstyle='-|>', color=ICOL[k], lw=1.5,
                                shrinkA=0, shrinkB=0))
    print(f'  {k:22} fast-gamma phase {np.degrees(np.angle(z)) % 360:6.0f} deg, '
          f'R = {np.abs(z):.3f}')
ax.set_rlim(0, .62); ax.set_rticks([.3, .6])
ax.set_yticklabels(['0.3', '0.6'], fontsize=5.4)
ax.set_xticks(np.radians([0, 90, 180, 270]))
ax.set_xticklabels(['0°', '90°', '180°', '270°'], fontsize=6)
ax.grid(lw=.4, color='0.85')
ax.set_title('preferred fast-gamma phase, colours\nas in A (arrow = consistency)',
             fontsize=7.2, loc='left', pad=8)

# ---- D: the interneuron lag, per session -----------------------------------
ax = fig.add_subplot(G[1, 0]); lp(ax, 'D')
lag = {}
for bn, _ in BANDS:
    rows = []
    for (mo, dy), g in D.groupby(['mouse', 'day']):
        a = g[g.identity == 'putative interneuron'][f'phi_{bn}_a'].dropna()
        b = g[g.identity != 'putative interneuron'][f'phi_{bn}_a'].dropna()
        if len(a) < 4 or len(b) < 10:
            continue
        rows.append(np.degrees(np.angle(
            np.exp(1j * (np.angle(np.mean(np.exp(1j * a)))
                         - np.angle(np.mean(np.exp(1j * b)))))))) 
    lag[bn] = np.array(rows)
for j, (bn, _) in enumerate(BANDS):
    v = lag[bn]
    ax.scatter(np.full(len(v), j) + np.random.default_rng(0).uniform(-.11, .11, len(v)),
               v, s=7, color='#d95f02' if bn == 'fast' else '0.6', lw=0, alpha=.8)
    ax.plot([j - .22, j + .22], [np.median(v)] * 2, color='k', lw=1.6)
    p = wilcoxon(v).pvalue
    ax.text(j, 148, f'p = {p:.0e}'.replace('e-0', 'e-') if p < .01
            else f'p = {p:.2f}', ha='center', fontsize=6, color='0.2')
    print(f'  {bn}: interneuron minus principal phase, median {np.median(v):+.1f} deg, '
          f'p = {p:.3g}, later in {int((v > 0).sum())}/{len(v)}')
ax.axhline(0, color='0.5', lw=.8, ls='--')
ax.set_xticks(range(2)); ax.set_xticklabels([l for _, l in BANDS], fontsize=6.5,
                                            linespacing=1.2)
ax.set_xlim(-.5, 1.5); ax.set_ylim(-180, 180); ax.set_yticks([-180, -90, 0, 90, 180])
ax.set_ylabel('interneuron − principal\nphase (deg)', fontsize=7.5)
ax.set_title('interneurons follow the principal\ncells, in fast gamma only',
             fontsize=7.2, loc='left')
tidy(ax)

# ---- E: the state comparison -----------------------------------------------
ax = fig.add_subplot(G[1, 1]); lp(ax, 'E')
labs, ps = [], []
for bn, _ in BANDS:
    for who, m in (('int', D.identity == 'putative interneuron'),
                   ('rest', D.identity != 'putative interneuron')):
        s = D[m].groupby(['mouse', 'day'])[[f'ppc_{bn}_a', f'ppc_{bn}_n']].median().dropna()
        labs.append(f'{bn}\n{who}'); ps.append((s[f'ppc_{bn}_a'], s[f'ppc_{bn}_n']))
xs = np.arange(len(labs))
for i, (a, n) in enumerate(ps):
    ax.bar(i - .18, a.median(), .34, color=ANCH_COLOR, lw=0,
           label='anchored' if i == 0 else None)
    ax.bar(i + .18, n.median(), .34, color=NONANCH_COLOR, lw=0,
           label='non-anchored' if i == 0 else None)
raw = [wilcoxon(a, n).pvalue for a, n in ps]
order = np.argsort(raw); run = 0.0
holm = np.empty(len(raw))
for r_, i_ in enumerate(order):
    run = max(run, raw[i_] * (len(raw) - r_)); holm[i_] = min(run, 1.0)
for i, h in enumerate(holm):
    ax.text(i, max(ps[i][0].median(), ps[i][1].median()) * 1.08,
            f'{h:.2f}', ha='center', fontsize=5.8, color='0.3')
print('  state comparison, Holm: '
      + ', '.join(f'{l.replace(chr(10), " ")} {h:.3f}' for l, h in zip(labs, holm)))
ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=6, linespacing=1.2)
ax.set_ylabel('PPC, 120 µm away', fontsize=7.5)
ax.set_title('nothing moves with the state\n(Holm-corrected p above each pair)',
             fontsize=7.2, loc='left')
ax.legend(fontsize=5.8, frameon=False, loc='upper right')
tidy(ax)

# ---- F: spike counts, so the reader can see why PPC was used ---------------
ax = fig.add_subplot(G[1, 2]); lp(ax, 'F')
for i, k in enumerate(ORDER):
    v = np.log10(D[D.identity == k].n_a.clip(lower=1))
    ax.bar(i, np.median(v), .6, color=ICOL[k], lw=0)
ax.set_xticks(range(4)); ax.set_xticklabels([SHORT[k] for k in ORDER], fontsize=6.5)
ax.set_ylabel('log$_{10}$ spikes, anchored', fontsize=7.5)
ax.set_title('why PPC: interneurons fire five\ntimes more than principal cells',
             fontsize=7.2, loc='left')
tidy(ax)

fig.suptitle('Spike coupling to gamma in entorhinal cortex, by cell identity '
             f'and by anchoring state ({len(D)} cells, {NS_} sessions)',
             fontsize=8.6, x=.075, ha='left', y=.985)
fig.savefig(OUT, dpi=220, bbox_inches='tight')
print(f'\nwrote {OUT}')
