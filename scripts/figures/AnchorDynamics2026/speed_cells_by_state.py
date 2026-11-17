"""Do open-field speed cells change their speed coding between anchoring states?

The numbers quoted in the Results for the four groups defined entirely outside
the task -- open-field speed modulation (speed_p < 0.05, speed_r > 0 in the unit
table) crossed with the waveform interneuron call -- on each state's common
speed support (the cs_ columns of speed_tuning_cells.csv, from speed_tuning.py).
Per session the median over the group's cells, then paired across sessions.
The minimum detectable change is for 80% power at alpha 0.05, from the SD of
the paired per-session log ratio.
"""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, norm

ROOT = '/Users/harryclark/Documents/spatial-manifolds/data'
C = pd.read_csv(f'{ROOT}/lfp/speed_tuning_cells.csv')
U = pd.read_csv(f'{ROOT}/population_state/unit_table.csv')
U = U.rename(columns={'cluster_id': 'cluster'})[['mouse', 'day', 'cluster', 'identity',
                                                 'speed_r', 'speed_p']]
D = C.merge(U, on=['mouse', 'day', 'cluster'])
D['speed'] = (D.speed_p < .05) & (D.speed_r > 0)
D['int'] = D.identity == 'putative interneuron'
GROUPS = {'speed interneurons': D.speed & D.int, 'speed principal': D.speed & ~D.int,
          'non-speed interneurons': ~D.speed & D.int, 'non-speed principal': ~D.speed & ~D.int}
print(f'{len(D)} MEC cells with speed tuning in both states, {D.groupby(["mouse", "day"]).ngroups} sessions')
for name, m in GROUPS.items():
    for v in ('slope', 'gain'):
        g = (D[m].groupby(['mouse', 'day'])[[f'cs_{v}_a', f'cs_{v}_n']].median().dropna())
        a, n = g[f'cs_{v}_a'], g[f'cs_{v}_n']
        lr = np.log(np.clip(a, 1e-9, None) / np.clip(n, 1e-9, None))
        ok = np.isfinite(lr) & (a > 0) & (n > 0)
        mde = (np.exp((norm.ppf(.975) + norm.ppf(.8)) * lr[ok].std() / np.sqrt(ok.sum())) - 1) * 100
        print(f'  {name:24s} {v:5s} anchored {a.mean():.4f} vs {n.mean():.4f} '
              f'({100 * (a.mean() / n.mean() - 1):+.1f}%), p = {wilcoxon(a, n).pvalue:.3g}, '
              f'{len(g)} sessions; MDE {mde:.0f}%')
