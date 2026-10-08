"""Figure 3 supplement — the laminar sampling bias along the mediolateral axis,
and what it does to the mediolateral results.

WHY THIS EXISTS. MEC is curved and a four-shank probe is flat, so shanks placed
at different mediolateral positions do not enter the tissue at the same depth
relative to the layers. If medial shanks sample superficial layers and lateral
shanks sample deep ones, then any mediolateral gradient in Figure 3 could be a
laminar gradient wearing a mediolateral coat. The manuscript asserted that this
had been checked and was absent, citing an "Extended Figure 2B" that does not
exist in this document. It is checked here, and it is NOT absent.

    A   CONTACTS. Of 4,872 ENTm recording sites, the superficial share falls
        from 90.3% at the most medial quartile to 59.8% at the most lateral.
        Pooled rho = -0.267 (p = 4e-80); within mouse median rho = -0.447,
        negative in 6 of 6 mice (p = 0.031). This is the quantity the
        manuscript's claim was about, and the claim does not hold.

    B   CELLS. The bias carries through to the neurons actually analysed:
        90.4% superficial medially against ~62% laterally, within-session
        median rho = -0.402, p = 0.016.

    C,D THE CONSEQUENCE, which is not the same for the two results Figure 3
        reports. Stratifying by layer:

            identity   all layers  rho -0.078  p = 0.025   <- the reported result
                       superficial rho -0.051  p = 0.24
                       deep        rho -0.032  p = 0.58  (7 sessions)

            following  all layers  rho -0.109  p = 0.034   <- the reported result
                       superficial rho -0.144  p = 0.043
                       deep        rho -0.033  p = 0.84  (6 sessions)

        The IDENTITY gradient does not survive stratification: the effect
        roughly halves and loses significance within superficial layers, which
        is what partial confounding by layer looks like. The FOLLOWING gradient
        does survive, and is if anything stronger within superficial layers.

        That inverts the caveat as the manuscript states it. The text says the
        following gradient may be an expression of the identity gradient; on
        this control the following gradient is the more robust of the two.

HONEST LIMITS. Deep-layer sessions are few -- 6 and 7 of them -- so the deep
columns are underpowered rather than null, and nothing here shows the gradient
is ABSENT deep. Stratification is not mediation: it shows the mediolateral
effect is not carried by layer differences within a stratum, not that layer
plays no part. And the superficial/deep split is the Allen annotation of each
contact, which inherits whatever registration error the atlas fit carries.

Writes fig3_supp_laminar.pdf
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
from scipy.stats import spearmanr, wilcoxon

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.bf'] = 'Arial:bold'
plt.rcParams['pdf.fonttype'] = 42

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = os.path.dirname(os.path.abspath(__file__))
PS = f'{ROOT}/data/population_state'
DC = '/Users/harryclark/Downloads/device_contact_id_annotations.csv'
OUT = f'{FIG}/fig3_supp_laminar.pdf'

MIN_SPAN_UM = 200.0        # as in Figure 3: two shank pitches
SUP_C, DEEP_C = '#1f78b4', '#b2182b'
GC_C, FOL_C = '#c04744', '#2b6cb0'


def _lp(ax, s, dx=-.20, dy=1.0):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='right')


def within_session(df, xv, yv, restrict=True):
    """Per-session Spearman, on sessions that actually span the axis."""
    out = []
    for _, g in df.dropna(subset=[xv, yv]).groupby(['mouse', 'day']):
        if restrict and (g[xv].max() - g[xv].min()) < MIN_SPAN_UM:
            continue
        if len(g) < 20 or g[xv].nunique() < 2 or g[yv].nunique() < 2:
            continue
        r = spearmanr(g[xv], g[yv])[0]
        if np.isfinite(r):
            out.append(r)
    out = np.array(out)
    return out, (wilcoxon(out).pvalue if len(out) >= 6 else np.nan)


def quartile_profile(df, mlcol, supcol):
    q = pd.qcut(df[mlcol], 4, labels=False, duplicates='drop')
    return df.groupby(q).agg(ml=(mlcol, 'median'), sup=(supcol, 'mean'),
                             n=(supcol, 'size')).reset_index(drop=True)


# ── data ─────────────────────────────────────────────────────────────────────
con = pd.read_csv(DC)
con = con[con.brain_region.astype(str).str.startswith('ENTm')].copy()
con['sup'] = con.brain_region.str.extract(r'ENTm(\d)')[0].isin(['1', '2', '3'])
con['ml'] = con.coord_SCs_x.abs()

C = pd.read_csv(f'{ROOT}/data/cell_classifications_v2.csv')
C = C[C.brain_region.astype(str).str.startswith('ENTm')].copy()
C['sup'] = C.brain_region.str.extract(r'ENTm(\d)')[0].isin(['1', '2', '3'])
C['ml'] = C.coord_SCs_x.abs()
C['supf'] = C.sup.astype(float)
U = pd.read_csv(f'{PS}/unit_table.csv')[['mouse', 'day', 'cluster_id', 'r',
                                         'in_population']]
C = C.merge(U, on=['mouse', 'day', 'cluster_id'], how='left')
SP = C[C.cell_class_of1.isin(['GC', 'NGS'])].copy()
SP['gc'] = (SP.cell_class_of1 == 'GC').astype(float)
FO = C[C.in_population.fillna(False) & C.r.notna()].copy()

print(f'{len(con)} ENTm contacts, {len(C)} ENTm cells, '
      f'{len(SP)} spatial, {len(FO)} with an agreement score')

fig = plt.figure(figsize=(10, 3.3))
gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.15, 1.15], wspace=.52,
                      left=.065, right=.985, top=.80, bottom=.20)

# ── A: contacts ──────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0]); _lp(ax, 'A')
p = quartile_profile(con, 'ml', 'sup')
ax.plot(p.ml, p.sup * 100, '-o', color=SUP_C, lw=1.6, ms=5)
for _, r_ in p.iterrows():
    ax.annotate(f'{int(r_.n)}', (r_.ml, r_.sup * 100), textcoords='offset points',
                xytext=(0, 7), ha='center', fontsize=5.6, color='0.45')
rho, pv = spearmanr(con.ml, con.sup.astype(float))
ws = np.array([spearmanr(g.ml, g.sup.astype(float))[0]
               for _, g in con.groupby('mouse')
               if g.ml.nunique() > 1 and g.sup.nunique() > 1])
ws = ws[np.isfinite(ws)]
ax.set_title(f'recording sites\npooled rho = {rho:+.3f} (p = {pv:.0e})',
             fontsize=7.4, loc='left', color='0.25')
ax.text(.03, .06, f'within mouse median {np.median(ws):+.2f}\n'
                  f'negative in {(ws < 0).sum()}/{len(ws)} mice, '
                  f'p = {wilcoxon(ws).pvalue:.3f}',
        transform=ax.transAxes, fontsize=6.0, color='0.4', linespacing=1.4)
ax.set_xlabel('medial–lateral (µm)', fontsize=8)
ax.set_ylabel('superficial contacts (%)', fontsize=8)
ax.set_ylim(40, 100); ax.margins(x=.14)
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

# ── B: cells ─────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1]); _lp(ax, 'B')
p = quartile_profile(C, 'ml', 'sup')
ax.plot(p.ml, p.sup * 100, '-o', color='0.3', lw=1.6, ms=5)
for _, r_ in p.iterrows():
    ax.annotate(f'{int(r_.n)}', (r_.ml, r_.sup * 100), textcoords='offset points',
                xytext=(0, 7), ha='center', fontsize=5.6, color='0.45')
w, pw = within_session(C, 'ml', 'supf')
ax.set_title(f'the cells analysed\nwithin session rho = {np.median(w):+.3f} '
             f'(p = {pw:.3f})', fontsize=7.4, loc='left', color='0.25')
ax.set_xlabel('medial–lateral (µm)', fontsize=8)
ax.set_ylabel('superficial cells (%)', fontsize=8)
ax.set_ylim(40, 100); ax.margins(x=.14)
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

# ── C, D: does the M-L result survive inside a layer? ────────────────────────
for k, (df, yv, nm, col, letter) in enumerate((
        (SP, 'gc', 'grid identity', GC_C, 'C'),
        (FO, 'r', 'following', FOL_C, 'D'))):
    ax = fig.add_subplot(gs[2 + k]); _lp(ax, letter, dx=-.26)
    strata = [('all\nlayers', df), ('superficial', df[df.sup]),
              ('deep', df[~df.sup])]
    for i, (lab, s) in enumerate(strata):
        w, pw = within_session(s, 'ml', yv)
        if not len(w):
            continue
        c = col if i == 0 else (SUP_C if i == 1 else DEEP_C)
        ax.scatter(np.full(len(w), i) + np.random.default_rng(0)
                   .uniform(-.13, .13, len(w)), w, s=11, color=c, alpha=.75,
                   lw=0, zorder=3)
        ax.plot([i - .26, i + .26], [np.median(w)] * 2, color='k', lw=2, zorder=4)
        ax.annotate(f'p = {pw:.3g}\nn = {len(w)}', (i, .97),
                    xycoords=('data', 'axes fraction'), ha='center', va='top',
                    fontsize=5.9, color='#a33' if pw < .05 else '0.4',
                    linespacing=1.3)
    ax.axhline(0, color='0.6', lw=.8, ls=':')
    ax.set_xticks(range(len(strata)))
    ax.set_xticklabels([s[0] for s in strata], fontsize=6.8)
    ax.set_xlim(-.5, len(strata) - .5)
    ax.set_ylabel('within-session rho\n(medial–lateral)', fontsize=8)
    ax.set_title(f'{nm} against position,\nstratified by layer', fontsize=7.4,
                 loc='left', color='0.25')
    ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

fig.suptitle('Recording sites are biased toward superficial layers medially, '
             'and what that does to the mediolateral results',
             fontsize=8.6, y=.99, x=.065, ha='left')
plt.savefig(OUT, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'wrote {OUT}')
