"""Does SUMMED population rate track speed less when anchored?

The 80-200 Hz LFP band lost its speed coupling (+0.157 -> -0.024), and that band
is usually read as a proxy for population spiking. If the proxy is good, summed
population rate should lose speed coupling too. If population rate keeps
tracking speed, the high-band result is about synchrony or synaptic current,
not about how much the population fires.
"""
import sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
exec(open('/private/tmp/claude-501/-Users-harryclark-Documents-spatial-manifolds/'
          '35f3d8d1-3ead-4d48-8089-491c15b000b7/scratchpad/speed_tuning.py')
     .read().split('if __name__')[0])
T = pd.read_csv(f'{OUT}/theta_frequency.csv')
rows = []
for (mo, dy), st in T.groupby(['mouse', 'day']):
    try:
        got = binned(int(mo), int(dy), dict(zip(st.trial.astype(int), st.anch.astype(bool))))
    except Exception as e:
        print(f'  ! M{mo}D{dy}: {e}', flush=True); continue
    if got is None: continue
    R, speed, anch, ids = got
    if anch.sum() < MINBIN or (~anch).sum() < MINBIN: continue
    pop = R.sum(1)
    cs = common_support(speed, anch)
    d = dict(mouse=mo, day=dy)
    for tag, m in (('', np.ones(len(speed), bool)), ('cs_', cs)):
        for sel, lab in ((anch & m, 'a'), (~anch & m, 'n')):
            if sel.sum() < MINBIN: continue
            d[f'{tag}r_{lab}'] = np.corrcoef(speed[sel], pop[sel])[0, 1]
            d[f'{tag}sl_{lab}'] = np.polyfit(speed[sel], pop[sel], 1)[0] / pop[sel].mean()
    rows.append(d)
    print(f'M{mo}D{dy}: r {d.get("r_a",np.nan):+.3f} / {d.get("r_n",np.nan):+.3f}', flush=True)
P = pd.DataFrame(rows)
P.to_csv(f'{OUT}/speed_poprate.csv', index=False)
print(f'\n{len(P)} sessions')
for tag, name in (('', 'all running bins'), ('cs_', 'common speed support')):
    d = P.dropna(subset=[f'{tag}r_a', f'{tag}r_n'])
    print(f'  {name}: r(pop rate, speed) anchored {d[f"{tag}r_a"].median():+.3f} '
          f'vs non {d[f"{tag}r_n"].median():+.3f}, p = {wilcoxon(d[f"{tag}r_a"], d[f"{tag}r_n"]).pvalue:.3g}')
    print(f'      normalised slope anchored {d[f"{tag}sl_a"].median():+.5f} '
          f'vs non {d[f"{tag}sl_n"].median():+.5f}, p = {wilcoxon(d[f"{tag}sl_a"], d[f"{tag}sl_n"]).pvalue:.3g}')
