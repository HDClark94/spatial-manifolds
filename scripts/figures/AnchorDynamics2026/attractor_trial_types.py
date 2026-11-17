"""Trial type (cued / uncued) for each trial row of the attractor cache.

The cache's position maps have one row per trial inside the ephys window
(attractor_coherence.session_maps keeps those rows, in order). This lists the
VR trial number and type of each row: 'b' beaconed (cued), 'nb' non-beaconed
(uncued), and any other type the task used.

Reads the VR behaviour files (attractor_coherence.vr_paths) for the sessions in
data/population_state/attractor_cache/
Writes data/population_state/attractor_trial_types.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import pynapple as nap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attractor_coherence as AC                                  # noqa: E402

rows = []
for f in sorted(glob.glob(f'{AC.PS}/attractor_cache/M*D*.npz')):
    z = np.load(f)
    mo, dy = int(z['mouse']), int(z['day'])
    bp, cp = AC.vr_paths(mo, dy)
    ta = nap.load_file(bp)['trials'].as_dataframe()
    _, orig = AC.clip_trials(ta, nap.load_file(cp))
    t = ta[np.isin(ta.number.values.astype(int), orig)].reset_index(drop=True)
    assert len(t) == len(z['state']), (mo, dy, len(t), len(z['state']))
    for i, r in t.iterrows():
        rows.append(dict(mouse=mo, day=dy, row=i, trial=int(r.number), type=r['type'],
                         anchored=bool(z['state'][i])))
    print(f'  M{mo}D{dy}: {len(t)} trials, {(t["type"] == "nb").mean():.0%} uncued', flush=True)
pd.DataFrame(rows).to_csv(f'{AC.PS}/attractor_trial_types.csv', index=False)
print('wrote attractor_trial_types.csv')
