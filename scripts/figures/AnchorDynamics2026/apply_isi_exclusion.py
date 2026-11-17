"""Exclude units whose refractory-period violations rule out a neural origin,
and isolated groups of units at the dorsal end of a shank.

spikeinterface's isi_violations_ratio is the rate of inter-spike intervals under
1.5 ms relative to the unit's firing rate, scaled so that a Poisson spike train
scores ~0.5. Merging any number of independent neurons cannot take it much above
that, so a ratio above ISI_MAX (20x a Poisson train) means the events come in
sub-1.5 ms clusters: duplicated detections or non-neural transients, not a
neuron or a multi-unit of neurons. Across bombcell good and mua units the log
ratio is bimodal with a dip at ~10; no bombcell 'good' unit exceeds 5. Flagged
units have small templates (median ~15 uV), smeared across channels, and fire in
bursts (waveform cards: view_waveforms.py).

A unit is excluded if the ratio exceeds ISI_MAX in its VR or its OF1 session
(the same sorted unit; metrics from the analyzers, via pack_waveforms.py). Its
cell_class_of1 and cell_class_of2 become 'EXCL', as for bombcell noise; the
classes before this step are kept in cell_class_of1_pre_isi / _of2_pre_isi, and
running again re-derives from those, so the step is idempotent. Run after
apply_rule_anatomy.py, which rewrites the table from its atlas backup.

SECOND RULE (2026-10-31): dorsal islands. On each shank of each session the
curated units are sorted by depth along the probe and scanned from the most
dorsal. If the first gap of at least ISLAND_GAP_UM between neighbouring units
has at most ISLAND_MAX_N units above it, and at most ISLAND_MAX_FRAC of the
shank's units, those units form an island. It is excluded when it also lies
above every other (non-island) curated unit recorded on that shank in any
session, and the atlas places all of it in visual cortex or outside the brain
('root'): units at the brain surface, above the cortex every session records. Gaps between neighbouring
curated units are rarely large (99% under 75 um; 41 of 16,736 at >= 300 um),
and islands are small (median amplitude ~0.3-0.9 of the units just below the
gap, SNR ~0.3-0.8), carry three times the usual share of VR bombcell noise
labels, and the atlas puts most of them outside the brain (M26 D12's nine
are the survivors of ~80 clusters at the top of the probe that bombcell and the
ISI rule already reject) or in layer 1. They also set the shank's
"top" (brain entry) in anatomy_rules.py, so this runs before it. Islands deep in
the recording (M21 D15/D17, whose recordings start 720 um up the probe) or
placed in MEC by the atlas (M20 D22, D24) are kept.

NOT APPLIED: bombcell's VR labels (2026-10-31). Curation uses bombcell's OF1
labels; also excluding units bombcell calls noise/non_soma in VR was tried and
reversed. 662 units (3.9%) would go, with only slightly flatter VR footprints
(amplitude 60 um out, 0.32 of the peak against 0.22). M22 D34 and D37 have
failed VR templates (~1 uV), so VR bombcell calls every unit there noise.

Reads data/waveforms/clark2025_waveforms.npz; writes data/cell_classifications_v2.csv
data/unit_quality/isi_excluded_units.csv and data/unit_quality/dorsal_island_units.csv.
"""
import numpy as np
import pandas as pd

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
CSV = f'{ROOT}/data/cell_classifications_v2.csv'
WAVE = f'{ROOT}/data/waveforms/clark2025_waveforms.npz'
OUT = f'{ROOT}/data/unit_quality/isi_excluded_units.csv'
ISI_MAX = 10
OUT_ISL = f'{ROOT}/data/unit_quality/dorsal_island_units.csv'
ISLAND_GAP_UM = 300      # 4x the 99th percentile of gaps between neighbouring units
ISLAND_MAX_N = 10
ISLAND_MAX_FRAC = 0.10
TYPES = ('VR', 'OF1')

W = np.load(WAVE, allow_pickle=True)
rows = []
for k in W.files:
    s, f = k.split('/')
    if f == 'qm_isi_violations_ratio' and s.split('_')[1] in TYPES:
        rows.append(pd.DataFrame(dict(mouse=int(s[1:3]), day=int(s.split('D')[1].split('_')[0]),
                                      typ=s.split('_')[1], cluster_id=W[f'{s}/unit_ids'], isi=W[k])))
R = pd.concat(rows).pivot_table(index=['mouse', 'day', 'cluster_id'], columns='typ',
                                 values='isi').add_prefix('isi_ratio_').reset_index()

C = pd.read_csv(CSV)
for c in ('cell_class_of1', 'cell_class_of2'):
    if f'{c}_pre_isi' not in C:
        C[f'{c}_pre_isi'] = C[c]
    C[c] = C[f'{c}_pre_isi']
C = C.drop(columns=[c for c in C if c.startswith('isi_ratio_') or c == 'bombcell_VR']).merge(
    R, on=['mouse', 'day', 'cluster_id'], how='left')
bad = (C[[f'isi_ratio_{t}' for t in TYPES]] > ISI_MAX).any(axis=1) & (C.cell_class_of1 != 'EXCL')
C.loc[bad, ['cell_class_of1', 'cell_class_of2']] = 'EXCL'

# dorsal islands, among the units still curated
K = C[C.cell_class_of1 != 'EXCL'].assign(shank=lambda d: (d.coord_probe_x // 250).astype(int))
island = {}
for (mo, dy, sh), g in K.groupby(['mouse', 'day', 'shank']):
    g = g.sort_values('coord_probe_y', ascending=False)
    gaps = -np.diff(g.coord_probe_y.values)
    for j in np.where(gaps >= ISLAND_GAP_UM)[0]:
        if j + 1 <= min(ISLAND_MAX_N, ISLAND_MAX_FRAC * len(g)):
            island[(mo, dy, sh)] = g.index[:j + 1]
        break
flagged = np.concatenate([v for v in island.values()]) if island else []
body_top = K[~K.index.isin(flagged)].groupby(['mouse', 'shank']).coord_probe_y.max()
isl = []
for (mo, dy, sh), ix in island.items():
    above = K.loc[ix, 'coord_probe_y'].min() > body_top.get((mo, sh), -np.inf)
    # the atlas puts these in visual cortex or outside the brain ('root')
    surface = K.loc[ix, 'brain_region_atlas'].astype(str).str.match(r'VIS|root$').all()
    if above and surface:
        isl += list(ix)
bad_isl = C.index.isin(isl)
C.loc[bad_isl, ['cell_class_of1', 'cell_class_of2']] = 'EXCL'
C.to_csv(CSV, index=False)
C[bad_isl][['mouse', 'day', 'cluster_id', 'brain_region', 'brain_region_atlas',
            'cell_class_of1_pre_isi', 'coord_probe_x', 'coord_probe_y']].to_csv(OUT_ISL, index=False)

C[bad][['mouse', 'day', 'cluster_id', 'brain_region', 'cell_class_of1_pre_isi',
        'isi_ratio_VR', 'isi_ratio_OF1']].round(3).to_csv(OUT, index=False)
print(f'{bad.sum()} of {(C.cell_class_of1_pre_isi != "EXCL").sum()} curated units excluded '
      f'(ISI violation ratio > {ISI_MAX} in VR or OF1)')
print(C[bad].groupby('cell_class_of1_pre_isi').size().to_dict())
print(f'{bad_isl.sum()} more excluded as dorsal islands '
      f'({C[bad_isl].groupby(["mouse", "day"]).ngroups} sessions)')
print(C[bad_isl].groupby('cell_class_of1_pre_isi').size().to_dict())
print(f'{(C.cell_class_of1 != "EXCL").sum()} curated units remain')
