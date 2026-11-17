"""Group summaries of speed_state_glm.py, the numbers quoted for Figure S17.

Groups are defined entirely outside the task, as in the Results: open-field
speed modulation (speed_p < 0.05, speed_r > 0 in the unit table) crossed with
the waveform interneuron call. Each cell gives its speed gain when not anchored
(b_n, log-rate per 10 cm/s) and the change when anchored (b_int). Per session
the group's cells are averaged with inverse-variance weights; the change is
then expressed as a percentage of the non-anchored gain, with a 95% interval
from a hierarchical bootstrap (mice, then sessions within mice), and tested
with a mixed model on the session values (mouse random intercept).
"""
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

ROOT = '/Users/harryclark/Documents/spatial-manifolds/data'
N_BOOT = 2000
GROUP_ORDER = ['speed interneurons', 'speed principal', 'non-speed interneurons',
               'non-speed principal']


def load():
    C = pd.read_csv(f'{ROOT}/lfp/speed_state_glm_cells.csv')
    U = pd.read_csv(f'{ROOT}/population_state/unit_table.csv')
    U = U.rename(columns={'cluster_id': 'cluster'})[['mouse', 'day', 'cluster', 'identity',
                                                     'speed_r', 'speed_p']]
    D = C.merge(U, on=['mouse', 'day', 'cluster'], how='inner')
    sp = (D.speed_p < .05) & (D.speed_r > 0)
    it = D.identity == 'putative interneuron'
    D['group'] = np.select([sp & it, sp & ~it, ~sp & it, ~sp & ~it], GROUP_ORDER, default='')
    return D.replace([np.inf, -np.inf], np.nan).dropna(subset=['b_n', 'b_int', 'se_int'])


def per_session(D, col_n='b_n', col_i='b_int', col_se='se_int'):
    def agg(g):
        w = 1 / g[col_se].clip(lower=1e-4) ** 2
        return pd.Series(dict(b_n=np.average(g[col_n], weights=w),
                              b_i=np.average(g[col_i], weights=w), n=len(g)))
    return D.groupby(['mouse', 'day']).apply(agg).reset_index()


def hboot(S, rng, n=N_BOOT):
    mice = S.mouse.unique()
    out = []
    for _ in range(n):
        parts = []
        for m in rng.choice(mice, len(mice), replace=True):
            g = S[S.mouse == m]
            parts.append(g.iloc[rng.integers(0, len(g), len(g))])
        B = pd.concat(parts)
        out.append(100 * B.b_i.mean() / B.b_n.mean())
    return np.percentile(out, [2.5, 97.5])


def summarise(D, label='', **cols):
    rng = np.random.default_rng(0)
    res = {}
    for grp in GROUP_ORDER:
        S = per_session(D[D.group == grp], **cols)
        S = S[S.n >= 2]
        pct = 100 * S.b_i.mean() / S.b_n.mean()
        lo, hi = hboot(S, rng)
        m = smf.mixedlm('b_i ~ 1', S, groups=S['mouse']).fit(reml=True)
        res[grp] = dict(pct=pct, lo=lo, hi=hi, p=float(m.pvalues['Intercept']),
                        gain_n=S.b_n.mean(), gain_a=(S.b_n + S.b_i).mean(),
                        n_sess=len(S), n_mice=S.mouse.nunique(),
                        n_cells=int(D[D.group == grp].shape[0]),
                        up=int((S.b_i > 0).sum()))
        r = res[grp]
        print(f'  {label}{grp:24s} gain {r["gain_n"]:+.4f} -> {r["gain_a"]:+.4f} per 10 cm/s; '
              f'change {pct:+.1f}% (95% CI {lo:+.1f} to {hi:+.1f}), mixed p = {r["p"]:.3g}; '
              f'{r["n_cells"]} cells, {r["n_sess"]} sessions, {r["n_mice"]} mice')
    return res


if __name__ == '__main__':
    D = load()
    print(f'{len(D)} MEC cells, {D.groupby(["mouse", "day"]).ngroups} sessions, '
          f'{D.mouse.nunique()} mice')
    print('position held (the model):')
    summarise(D)
    print('without position terms (what position does):')
    summarise(D, 'nopos ', col_n='b_n_nopos', col_i='b_int_nopos', col_se='se_int_nopos')
    S = pd.read_csv(f'{ROOT}/lfp/speed_state_glm_sessions.csv').dropna(subset=['dec_r_a', 'dec_r_n'])
    from scipy.stats import wilcoxon
    for a, n_, lab in (('dec_r_a', 'dec_r_n', 'decoding of unexpected speed r'),
                       ('sres_sd_a', 'sres_sd_n', 'SD of unexpected speed (cm/s)')):
        m = smf.mixedlm('d ~ 1', S.assign(d=S[a] - S[n_]), groups=S['mouse']).fit(reml=True)
        print(f'  {lab}: anchored {S[a].median():.3f} vs {S[n_].median():.3f}; '
              f'{(S[a] > S[n_]).sum()} of {len(S)} higher anchored; Wilcoxon p = '
              f'{wilcoxon(S[a], S[n_]).pvalue:.3g}; mixed p = {m.pvalues["Intercept"]:.3g}')
