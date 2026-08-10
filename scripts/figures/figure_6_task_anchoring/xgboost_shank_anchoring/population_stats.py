"""
Population statistics for the MEC shank-reconstruction assay.

Aggregates every cached session (data/xgboost_shank_anchoring/M*D*_shank_reconstruction.pkl)
into a tidy dataframe (one row per target x covariate-shank) and asks:

  Q1. Does the mediolateral (ML) position of the covariate shank affect the ability to
      reconstruct a target MEC cell / its anchoring labels?
        - reconstruction matrix (target shank x covariate shank), by pR2 and by label-MCC
        - pR2 / label-MCC vs |ML distance| between shanks
        - directional (medial vs lateral covariates)
  Q2. Is reconstruction different in anchored vs non-anchored epochs?
        - paired pR2(anchored) vs pR2(non-anchored), Delta distribution, Wilcoxon
        - anch/non-anch decay with ML distance

Shanks are ranked medial->lateral within each session by mean |coord_SCs_x| (handles
probes implanted in reverse). Re-run any time; it uses whatever sessions are cached.
"""
import os, glob, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, spearmanr
from sklearn.metrics import matthews_corrcoef

DATA_DIR = '/Users/harryclark/Documents/spatial-manifolds/data/xgboost_shank_anchoring'
FIG_DIR  = '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/figure_6_task_anchoring/xgboost_shank_anchoring/figures'
LOCS_CSV = '/Users/harryclark/Downloads/COHORT12/all_cluster_brain_locations_chris.csv'
os.makedirs(FIG_DIR, exist_ok=True)

C_ANCH, C_NON = '#9d5391', '#c7a3c4'
C_OVER = '#2c3e50'


def shank_ml(mouse, day):
    """Mean |coord_SCs_x| per shank (ENTm+PAR) -> medial rank (0 = most medial)."""
    loc = pd.read_csv(LOCS_CSV)
    loc = loc[(loc['mouse'] == mouse) & (loc['day'] == day)]
    loc = loc[loc['brain_region'].astype(str).str.startswith(('ENTm', 'PAR'))]
    ml = loc.assign(ml=loc['coord_SCs_x'].abs()).groupby('shank_id')['ml'].mean()
    order = ml.sort_values().index.tolist()
    rank = {int(sh): i for i, sh in enumerate(order)}
    return {int(k): float(v) for k, v in ml.items()}, rank


def label_metrics(obs, pred):
    obs = np.asarray(obs, float); pred = np.asarray(pred, float)
    m = np.isfinite(obs) & np.isfinite(pred)
    out = dict(mcc=np.nan, acc=np.nan, anch_recall=np.nan, nonanch_recall=np.nan,
               n_valid=int(m.sum()), obs_anch_frac=np.nan)
    if m.sum() < 8:
        return out
    o = obs[m].astype(int); p = pred[m].astype(int)
    out['obs_anch_frac'] = float(o.mean())
    out['acc'] = float((o == p).mean())
    if len(np.unique(o)) > 1 and len(np.unique(p)) > 1:
        try:
            out['mcc'] = float(matthews_corrcoef(o, p))
        except Exception:
            pass
    if (o == 1).any(): out['anch_recall'] = float((p[o == 1] == 1).mean())
    if (o == 0).any(): out['nonanch_recall'] = float((p[o == 0] == 0).mean())
    return out


# ---------------------------------------------------------------- build tidy df
rows = []
files = sorted(glob.glob(f'{DATA_DIR}/M*D*_shank_reconstruction.pkl'))
print(f'{len(files)} cached sessions')
for f in files:
    d = pickle.load(open(f, 'rb'))
    mouse, day = d['mouse'], d['day']
    ml_abs, medrank = shank_ml(mouse, day)
    for tid, ti in d['targets'].items():
        t_sh = ti['shank']
        obs = ti['observed_labels']
        for sh, s in ti['shanks'].items():
            lm = label_metrics(obs, s['pred_labels'])
            t_ml, c_ml = ml_abs.get(t_sh, np.nan), ml_abs.get(sh, np.nan)
            t_rk, c_rk = medrank.get(t_sh, np.nan), medrank.get(sh, np.nan)
            rows.append(dict(
                session=f'M{mouse}D{day}', mouse=mouse, day=day,
                target=tid, target_shank=t_sh, cov_shank=sh,
                target_medrank=t_rk, cov_medrank=c_rk,
                drank_signed=(c_rk - t_rk), drank_abs=abs(c_rk - t_rk),
                dml_abs=abs(c_ml - t_ml), dml_signed=(c_ml - t_ml),
                is_same_shank=s['is_same_shank'],
                n_ent=s['n_ent_used'], n_par=s['n_par_used'],
                pr2=s['pr2_overall'], pr2_anch=s['pr2_anchored'], pr2_non=s['pr2_nonanchored'],
                pr2_diff=(s['pr2_anchored'] - s['pr2_nonanchored']),
                **lm))
df = pd.DataFrame(rows)
df.to_csv(f'{DATA_DIR}/population_stats.csv', index=False)
print(f'{len(df)} target x cov-shank rows across {df.session.nunique()} sessions '
      f'({df.target.nunique()} targets)')

NR = int(np.nanmax(df[['target_medrank', 'cov_medrank']].values)) + 1  # n medial ranks


def _mat(value, agg='mean'):
    m = np.full((NR, NR), np.nan)
    g = df.groupby(['target_medrank', 'cov_medrank'])[value].mean()
    for (t, c), v in g.items():
        if np.isfinite(t) and np.isfinite(c):
            m[int(t), int(c)] = v
    return m


# ================================================================ FIGURE A: ML effect
fig, ax = plt.subplots(2, 2, figsize=(10, 8.5))
LAB = 'medial → lateral'

for a, (val, title, cmap) in zip(
        [ax[0, 0], ax[0, 1]],
        [('pr2', 'Reconstruction pR²', 'viridis'),
         ('mcc', 'Anchoring-label MCC', 'magma')]):
    M = _mat(val)
    im = a.imshow(M, cmap=cmap, origin='upper')
    for i in range(NR):
        for j in range(NR):
            if np.isfinite(M[i, j]):
                a.text(j, i, f'{M[i, j]:.02f}', ha='center', va='center',
                       color='w' if M[i, j] < np.nanmean(M) else 'k', fontsize=9)
    a.set_xticks(range(NR)); a.set_yticks(range(NR))
    a.set_xlabel(f'covariate shank ({LAB})'); a.set_ylabel(f'target shank ({LAB})')
    a.set_title(title); plt.colorbar(im, ax=a, fraction=0.046)

# pR2 vs |ML distance| (shank-rank distance)
axd = ax[1, 0]
for val, col, lab in [('pr2', C_OVER, 'overall'), ('pr2_anch', C_ANCH, 'anchored'),
                      ('pr2_non', C_NON, 'non-anchored')]:
    g = df.groupby('drank_abs')[val].agg(['mean', 'sem'])
    axd.errorbar(g.index, g['mean'], yerr=g['sem'], marker='o', color=col, label=lab, capsize=3)
axd.axhline(0, color='k', lw=0.5, ls='--')
axd.set_xlabel('|shank distance| (medial-lateral steps)'); axd.set_ylabel('pR²')
axd.set_title('Reconstruction vs ML distance'); axd.legend(fontsize=8, frameon=False)
axd.set_xticks(sorted(df['drank_abs'].dropna().unique()))
for sp in ('top', 'right'): axd.spines[sp].set_visible(False)

# directional: signed ML displacement of covariate shank (pR2 left axis, MCC right axis)
axs = ax[1, 1]
g = df.groupby('drank_signed')['pr2'].agg(['mean', 'sem'])
axs.errorbar(g.index, g['mean'], yerr=g['sem'], marker='o', color=C_OVER, label='pR² (left)', capsize=3)
axs.axvline(0, color='k', lw=0.5, ls='--')
axs.set_xlabel('covariate shank position rel. to target\n(← medial     lateral →)')
axs.set_ylabel('pR²', color=C_OVER); axs.tick_params(axis='y', labelcolor=C_OVER)
axs.set_title('Directional ML effect')
axm = axs.twinx()
gm = df.groupby('drank_signed')['mcc'].agg(['mean', 'sem'])
axm.errorbar(gm.index, gm['mean'], yerr=gm['sem'], marker='s', color='#2E7D6C', label='label MCC (right)', capsize=3)
axm.set_ylabel('label MCC', color='#2E7D6C'); axm.tick_params(axis='y', labelcolor='#2E7D6C')
l1, la1 = axs.get_legend_handles_labels(); l2, la2 = axm.get_legend_handles_labels()
axs.legend(l1 + l2, la1 + la2, fontsize=8, frameon=False, loc='lower center')
axs.spines['top'].set_visible(False); axm.spines['top'].set_visible(False)

fig.suptitle('MEC shank reconstruction — mediolateral position', fontsize=13, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(f'{FIG_DIR}/population_ml_reconstruction.pdf', bbox_inches='tight')
fig.savefig(f'{FIG_DIR}/population_ml_reconstruction.png', dpi=200, bbox_inches='tight')
plt.close(fig)


# ================================================================ FIGURE B: anch vs non-anch
fig, ax = plt.subplots(2, 2, figsize=(10, 8.5))
d2 = df.dropna(subset=['pr2_anch', 'pr2_non'])

# paired scatter
a = ax[0, 0]
same = d2['is_same_shank']
a.scatter(d2.loc[~same, 'pr2_non'], d2.loc[~same, 'pr2_anch'], s=10, alpha=0.4,
          color='#7f8c8d', label='different shank')
a.scatter(d2.loc[same, 'pr2_non'], d2.loc[same, 'pr2_anch'], s=14, alpha=0.7,
          color='#c0392b', label='same shank')
lim = [min(d2['pr2_non'].min(), d2['pr2_anch'].min()), max(d2['pr2_non'].max(), d2['pr2_anch'].max())]
a.plot(lim, lim, 'k--', lw=0.8)
a.set_xlabel('pR² non-anchored'); a.set_ylabel('pR² anchored')
a.set_title('Anchored vs non-anchored (each target×shank)')
a.legend(fontsize=8, frameon=False)
for sp in ('top', 'right'): a.spines[sp].set_visible(False)

# Delta distribution
a = ax[0, 1]
diff = d2['pr2_diff'].dropna()
try:
    w, p = wilcoxon(d2['pr2_anch'], d2['pr2_non'])
except Exception:
    p = np.nan
parts = a.violinplot([diff.values], showmeans=True, showextrema=False)
for pc in parts['bodies']:
    pc.set_facecolor(C_ANCH); pc.set_alpha(0.5)
a.axhline(0, color='k', lw=0.6, ls='--')
a.set_xticks([1]); a.set_xticklabels(['all pairs'])
a.set_ylabel('pR² anchored − non-anchored')
a.set_title(f'Δ pR² (anch−non)\nWilcoxon p={p:.1e}, median={diff.median():.3f}')
for sp in ('top', 'right'): a.spines[sp].set_visible(False)

# Delta by shank relation
a = ax[1, 0]
grp = d2.assign(rel=np.where(d2['is_same_shank'], 'same', 'other')).groupby('rel')['pr2_diff']
labs = ['same', 'other']
a.violinplot([d2[d2.is_same_shank]['pr2_diff'].dropna().values,
             d2[~d2.is_same_shank]['pr2_diff'].dropna().values], showmeans=True, showextrema=False)
a.axhline(0, color='k', lw=0.6, ls='--')
a.set_xticks([1, 2]); a.set_xticklabels(labs)
a.set_ylabel('Δ pR² (anch−non)'); a.set_title('Δ by shank relation')
for sp in ('top', 'right'): a.spines[sp].set_visible(False)

# anch & non-anch decay with ML distance
a = ax[1, 1]
for val, col, lab in [('pr2_anch', C_ANCH, 'anchored'), ('pr2_non', C_NON, 'non-anchored')]:
    g = df.groupby('drank_abs')[val].agg(['mean', 'sem'])
    a.errorbar(g.index, g['mean'], yerr=g['sem'], marker='o', color=col, label=lab, capsize=3)
a.axhline(0, color='k', lw=0.5, ls='--')
a.set_xlabel('|shank distance|'); a.set_ylabel('pR²')
a.set_title('Anch / non-anch vs ML distance'); a.legend(fontsize=8, frameon=False)
a.set_xticks(sorted(df['drank_abs'].dropna().unique()))
for sp in ('top', 'right'): a.spines[sp].set_visible(False)

fig.suptitle('MEC shank reconstruction — anchored vs non-anchored', fontsize=13, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(f'{FIG_DIR}/population_anch_vs_nonanch.pdf', bbox_inches='tight')
fig.savefig(f'{FIG_DIR}/population_anch_vs_nonanch.png', dpi=200, bbox_inches='tight')
plt.close(fig)


# ================================================================ stats printout
lines = []
def _pr(s=''): lines.append(s); print(s)

_pr('=' * 64)
_pr(f'POPULATION STATS  ({df.session.nunique()} sessions, {df.target.nunique()} targets, {len(df)} rows)')
_pr('sessions: ' + ', '.join(sorted(df.session.unique())))
_pr('=' * 64)

_pr('\nQ2  Anchored vs non-anchored pR2')
_pr(f"  median pR2 anchored      = {d2['pr2_anch'].median():.4f}")
_pr(f"  median pR2 non-anchored  = {d2['pr2_non'].median():.4f}")
_pr(f"  median Delta (anch-non)  = {d2['pr2_diff'].median():.4f}")
try:
    w, p = wilcoxon(d2['pr2_anch'], d2['pr2_non'])
    _pr(f"  Wilcoxon (paired, all rows): W={w:.0f}, p={p:.2e}  "
        f"(caveat: rows not independent — pseudoreplication)")
except Exception as e:
    _pr(f'  Wilcoxon failed: {e}')
# per-session paired medians (independent units)
persess = d2.groupby('session')[['pr2_anch', 'pr2_non']].median()
if len(persess) >= 3:
    try:
        w, p = wilcoxon(persess['pr2_anch'], persess['pr2_non'])
        _pr(f"  Wilcoxon on per-session medians (n={len(persess)}): W={w:.0f}, p={p:.3f}")
    except Exception:
        pass
_pr('  per-session median pR2 (anch | non):')
for s, r in persess.iterrows():
    _pr(f'    {s}: {r.pr2_anch:.3f} | {r.pr2_non:.3f}')

_pr('\nQ1  ML distance effect (Spearman across rows)')
for val, lab in [('pr2', 'pR2 overall'), ('pr2_anch', 'pR2 anchored'), ('mcc', 'label MCC')]:
    sub = df.dropna(subset=['drank_abs', val])
    if len(sub) > 10:
        rho, p = spearmanr(sub['drank_abs'], sub[val])
        _pr(f'  {lab:14s} vs |shank distance|: rho={rho:+.3f}, p={p:.2e}')
_pr('\n  mean pR2 by shank relation:')
_pr(f"    same shank : {df[df.is_same_shank]['pr2'].mean():.4f} (n={df.is_same_shank.sum()})")
_pr(f"    other shank: {df[~df.is_same_shank]['pr2'].mean():.4f} (n={(~df.is_same_shank).sum()})")

# optional mixed model (session + target random intercepts)
try:
    import statsmodels.formula.api as smf
    long = pd.concat([
        d2[['session', 'target', 'drank_abs', 'pr2_anch']].rename(columns={'pr2_anch': 'pr2'}).assign(mode='anch'),
        d2[['session', 'target', 'drank_abs', 'pr2_non']].rename(columns={'pr2_non': 'pr2'}).assign(mode='non'),
    ], ignore_index=True).dropna()
    m = smf.mixedlm('pr2 ~ C(mode) + drank_abs', long, groups=long['session']).fit()
    _pr('\n  Mixed model  pr2 ~ mode + ML_distance  (random intercept: session)')
    for name in m.params.index:
        _pr(f'    {name:22s} beta={m.params[name]:+.4f}  p={m.pvalues[name]:.3g}')
except Exception as e:
    _pr(f'\n  (mixed model skipped: {e})')

open(f'{FIG_DIR}/population_stats_report.txt', 'w').write('\n'.join(lines))
print(f'\nWrote figures + report to {FIG_DIR}')
