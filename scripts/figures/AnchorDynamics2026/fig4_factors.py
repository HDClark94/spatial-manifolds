"""What best predicts how closely an MEC cell follows the population axis?

Extends fig4_controls.py's mixed model (position beside identity, layer and
firing rate) to every per-cell property the data hold, and asks which matters
most. Same cells: MEC cells (ENTm*) in the 27 sessions whose shanks span at
least 200 um, complete cases.

OUTCOME. Agreement above chance: |r| with the session PC1 (leave-one-out) minus
the mean |r| of the cell's own circular-shift null (pc1_by_region.csv), the
quantity of Figure 4E-F.

FACTORS, in groups:
  position   mediolateral and dorsoventral position, each centred within
             session (only within-session differences can say anything about
             anatomy once session has its own intercept); layer (superficial
             L1-3 vs deep L5-6)
  cell type  identity: grid, non-grid spatial, non-spatial, putative interneuron
  activity   log VR firing rate; waveform peak-to-trough duration
  tuning     open-field spatial information, grid score, head-direction mean
             vector length, border score, |speed correlation|, theta index
Anteroposterior position is left out: within a session it varies by a median
33 um, against 175-185 um for the other two axes.

MODEL. Linear mixed model, session random intercept with shank nested inside
it as a variance component (as fig4_controls.py; position varies mostly
between shanks, so shank must be a unit). Continuous factors are z-scored, so
coefficients are per SD and comparable.

IMPORTANCE. For each factor, the model refitted without it (cell type: all three
contrasts at once): the likelihood-ratio test (ML fits), Holm across factors,
and the drop in marginal R^2 (Nakagawa: variance of the fixed-effect
prediction over total variance) -- the variance that factor explains and no
other does -- beside the R^2 of a model holding that factor ALONE, so variance
it shares with other factors is visible rather than credited to whichever is
in the model. Session and shank variance components are reported beside them.
Because several factors overlap by construction (putative interneurons are
classed on waveform and rate, grid cells on grid score), the drop-one share
understates factors that duplicate one another. The figure therefore ranks them
by DOMINANCE analysis (general dominance, Budescu 1993): each factor's added R^2
averaged over every subset of the others (2^11 models per factor), so shared
variance is divided fairly rather than discarded. Least squares on session-
centred outcome and factors (session fixed effects), which makes the 4,096 fits
fast. As a check that does not assume linearity, gradient-boosted trees on the
session-centred outcome, cross-validated with whole sessions held out, ranked
by permutation importance.

Writes data/population_state/fig4_factors.json
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2

warnings.filterwarnings('ignore')
ROOT = '/Users/harryclark/Documents/spatial-manifolds'
PS = f'{ROOT}/data/population_state'
OUT = f'{PS}/fig4_factors.json'
KEY = ['mouse', 'day', 'cluster_id']

FACTORS = {          # name -> (columns in the formula, group, label)
    'ml':      (['ml_z'], 'position', 'mediolateral position'),
    'dv':      (['dv_z'], 'position', 'dorsoventral position'),
    'layer':   (['sup'], 'position', 'superficial layer'),
    'type':    (['C(identity, Treatment("non-grid spatial"))'], 'cell type', 'cell type'),
    'rate':    (['rate_z'], 'activity', 'firing rate (log)'),
    'wave':    (['wave_z'], 'activity', 'waveform width'),
    'si':      (['si_z'], 'tuning', 'spatial information'),
    'grid':    (['gs_z'], 'tuning', 'grid score'),
    'hd':      (['hd_z'], 'tuning', 'head-direction tuning'),
    'border':  (['bs_z'], 'tuning', 'border score'),
    'speed':   (['spd_z'], 'tuning', 'speed tuning |r|'),
    'theta':   (['th_z'], 'tuning', 'theta modulation'),
}


def load():
    A = pd.read_csv(f'{PS}/fig4_controls_cells.csv')
    A = A[A.mec].copy()
    C = pd.read_csv(f'{ROOT}/data/cell_classifications_v2.csv')
    U = pd.read_csv(f'{PS}/unit_table.csv')
    T = pd.read_csv(f'{PS}/of_theta_index.csv')
    d = (A.merge(C[KEY + ['coord_SCs_y', 'spatial_information_of1', 'grid_score_of1',
                          'hd_mean_vector_length_of1', 'border_score_of1']], on=KEY, how='left')
          .merge(U[KEY + ['peak_to_trough_duration', 'speed_r']], on=KEY, how='left')
          .merge(T[T.of == 'OF1'][KEY + ['theta_index']], on=KEY, how='left'))
    d = d[d.identity.isin(['grid', 'non-grid spatial', 'non-spatial', 'putative interneuron'])]
    d['sess'] = d.mouse.astype(str) + '_' + d.day.astype(str)
    d['shank_id'] = d.sess + '_' + d.shank.astype(int).astype(str)
    d['sup'] = d.region.eq('MEC sup').astype(float)
    d['ml_c'] = d.ml - d.groupby('sess').ml.transform('mean')
    d['dv_c'] = d.coord_SCs_y - d.groupby('sess').coord_SCs_y.transform('mean')
    d['lrate'] = np.log10(d.firing_rate.clip(lower=.01))
    d['lsi'] = np.log10(d.spatial_information_of1.clip(lower=1e-3))
    d['aspd'] = d.speed_r.abs()
    src = dict(ml_z='ml_c', dv_z='dv_c', rate_z='lrate', wave_z='peak_to_trough_duration',
               si_z='lsi', gs_z='grid_score_of1', hd_z='hd_mean_vector_length_of1',
               bs_z='border_score_of1', spd_z='aspd', th_z='theta_index')
    d = d.dropna(subset=['excess', 'sup'] + list(src.values())).copy()
    for z, c in src.items():
        d[z] = (d[c] - d[c].mean()) / d[c].std()
    d.attrs['sd'] = {z: float(d[c].std()) for z, c in src.items()}
    return d


def fit(d, terms, reml=False):
    f = 'excess ~ ' + (' + '.join(terms) if terms else '1')
    return smf.mixedlm(f, d, groups=d['sess'], re_formula='1',
                       vc_formula={'shank': '0 + C(shank_id)'}).fit(reml=reml, method='lbfgs')


def r2(m):
    """Nakagawa marginal and conditional R^2 for a random-intercept model."""
    fe = np.var(m.model.exog @ m.fe_params.values)
    s_sess = float(m.cov_re.iloc[0, 0]); s_sh = float(m.vcomp[0]); s_e = float(m.scale)
    tot = fe + s_sess + s_sh + s_e
    return dict(r2m=fe / tot, r2c=(fe + s_sess + s_sh) / tot, sess=s_sess / tot,
                shank=s_sh / tot, resid=s_e / tot)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); out = np.empty_like(p); run = 0
    for r, i in enumerate(o):
        run = max(run, p[i] * (len(p) - r)); out[i] = min(run, 1.0)
    return out


def dominance(d):
    """General dominance of each factor on the session-centred outcome."""
    import itertools
    y = (d.excess - d.groupby('sess').excess.transform('mean')).values
    cols = {}
    for k, (terms, _, _) in FACTORS.items():
        if k == 'type':
            v = np.c_[[(d.identity == l).astype(float).values
                       for l in ('grid', 'non-spatial', 'putative interneuron')]].T
        else:
            v = d[terms].values.astype(float)
        cols[k] = v - pd.DataFrame(v).groupby(d.sess.values).transform('mean').values
    keys = list(cols); n = len(keys); r2s = {}
    tss = np.sum((y - y.mean()) ** 2)
    for r in range(n + 1):
        for S in itertools.combinations(range(n), r):
            if not S:
                r2s[S] = 0.; continue
            X = np.hstack([np.ones((len(y), 1))] + [cols[keys[i]] for i in S])
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            r2s[S] = 1 - np.sum((y - X @ b) ** 2) / tss
    dom = {}
    for i, k in enumerate(keys):
        rest = [j for j in range(n) if j != i]
        dom[k] = float(np.mean([np.mean([r2s[tuple(sorted(S + (i,)))] - r2s[S]
                                         for S in itertools.combinations(rest, m)])
                                for m in range(n)]))
    return dom, float(r2s[tuple(range(n))])


def boosted(d):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import GroupKFold
    cols = {k: v[0][0] for k, v in FACTORS.items() if k != 'type'}
    X = d[list(cols.values())].copy()
    for lvl in ('grid', 'non-spatial', 'putative interneuron'):
        X[f'type_{lvl}'] = (d.identity == lvl).astype(float)
    y = (d.excess - d.groupby('sess').excess.transform('mean')).values
    imp = {k: [] for k in FACTORS}
    for trn, tst in GroupKFold(5).split(X, y, d.sess):
        m = HistGradientBoostingRegressor(max_depth=3, learning_rate=.05, max_iter=300,
                                          random_state=0).fit(X.iloc[trn], y[trn])
        Xt = X.iloc[tst]
        base = m.score(Xt, y[tst])
        rng = np.random.default_rng(0)
        for k in FACTORS:
            cs = [c for c in X.columns if c.startswith('type_')] if k == 'type' else [cols[k]]
            drops = []
            for _ in range(20):
                Xp = Xt.copy(); perm = rng.permutation(len(Xp))
                Xp[cs] = Xp[cs].values[perm]
                drops.append(base - m.score(Xp, y[tst]))
            imp[k].append(np.mean(drops))
    return {k: float(np.mean(v)) for k, v in imp.items()}


if __name__ == '__main__':
    d = load()
    print(f'{len(d)} MEC cells, {d.shank_id.nunique()} shanks, {d.sess.nunique()} sessions')
    allt = [t for v in FACTORS.values() for t in v[0]]
    full = fit(d, allt); full_reml = fit(d, allt, reml=True)
    R = r2(full_reml)
    print('full model R2:', {k: round(v, 4) for k, v in R.items()})
    # variance inflation of the continuous and binary terms
    Xc = d[[t for t in allt if not t.startswith('C(')]]
    vif = {c: float(np.linalg.inv(np.corrcoef(Xc.values.T))[i, i]) for i, c in enumerate(Xc)}
    print('VIF:', {k: round(v, 2) for k, v in vif.items()})
    coef = {k: dict(b=float(full_reml.params[k]), se=float(full_reml.bse[k]),
                    p=float(full_reml.pvalues[k]))
            for k in full_reml.params.index if k != 'Group Var' and 'Var' not in k}
    rows = []
    for k, (terms, grp, lab) in FACTORS.items():
        red = fit(d, [t for t in allt if t not in terms])
        lr = 2 * (full.llf - red.llf)
        df = 3 if k == 'type' else 1
        red_reml = fit(d, [t for t in allt if t not in terms], reml=True)
        rows.append(dict(factor=k, group=grp, label=lab, lr=float(lr), df=df,
                         p=float(chi2.sf(lr, df)),
                         d_r2m=R['r2m'] - r2(red_reml)['r2m'],
                         alone_r2m=r2(fit(d, terms, reml=True))['r2m']))
    F = pd.DataFrame(rows)
    F['p_holm'] = holm(F.p)
    gb = boosted(d)
    F['perm_importance'] = F.factor.map(gb)
    dom, r2_within = dominance(d)
    F['dominance'] = F.factor.map(dom)
    F = F.sort_values('dominance', ascending=False)
    print(F.round(5).to_string(index=False))
    print('coefficients (per SD):')
    for k, v in coef.items():
        print(f'   {k:60s} {v["b"]:+.4f} ± {v["se"]:.4f}  p = {v["p"]:.3g}')
    # how much the position effect leans on the tuning factors it covaries with
    base = ['ml_z', 'C(identity, Treatment("non-grid spatial"))', 'sup', 'rate_z']
    ml_steps = {}
    for nm, extra in (('Figure 4 model', []), ('+ spatial information', ['si_z']),
                      ('+ theta modulation', ['th_z']), ('+ both', ['si_z', 'th_z']),
                      ('all factors', [t for t in allt if t not in base])):
        m = fit(d, base + extra, reml=True)
        ml_steps[nm] = dict(b=float(m.params['ml_z']), se=float(m.bse['ml_z']),
                            p=float(m.pvalues['ml_z']))
        print(f'   ML {nm:24s} b {m.params["ml_z"]:+.4f} p {m.pvalues["ml_z"]:.3g}')
    cors = {c: float(np.corrcoef(d.ml_z, d[c])[0, 1]) for c in Xc if c != 'ml_z'}
    json.dump(dict(n_cells=len(d), r2_within=r2_within, ml_steps=ml_steps, ml_cor=cors, n_shanks=int(d.shank_id.nunique()),
                   n_sess=int(d.sess.nunique()), r2=R, vif=vif, coef=coef,
                   sd=d.attrs['sd'], factors=F.to_dict(orient='records')),
              open(OUT + '.partial', 'w'), indent=1)
    os.replace(OUT + '.partial', OUT)
    print('wrote', OUT)
