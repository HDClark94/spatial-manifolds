"""Relabel brain regions with the physiological anatomy of Figure S15.

Every cluster with coordinates in a mouse whose shanks have landmarks
(anatomy_rules.py: rule_shanks.csv, the mean of each shank's single-shank
sessions) is placed by its dorsoventral coordinate on its shank (nearest shank
mediolateral position):

  above the boundary           visual cortex
  boundary to exit (+30 um)    MEC / parasubiculum
  below the exit               cerebellum ('CB')

The LFP boundary is then corrected by cell type, since grid cells are expected
in MEC: on each shank (pooling every session), the boundary is raised to the
most dorsal grid cell of the run that continues up from MEC across it, the run
ending at the first gap between grid cells of more than GRID_GAP_UM. The
boundary only moves dorsally, and by at most GRID_MAX_UM (the disagreement
between repeat recordings of one shank, Figure S15), so grid cells further into
visual cortex, beyond a gap, keep their visual label. The LFP and corrected
boundaries are written to rule_shanks_cells.csv.

Where this agrees with the atlas structure (VIS*; ENTm*, ENTl*, PAR*, PRE*,
POST*) the atlas label, with its layer, is kept. Where it does not, the cluster
takes the atlas label of the nearest cluster (in depth, same session and shank)
that the atlas places in the structure the rule assigns, so layer information
survives; failing that 'VISp' or 'ENTm'. Clusters above the most dorsal curated
cell keep their atlas label.

The atlas label is kept in brain_region_atlas, and the original file is backed
up to cell_classifications_v2_atlas.csv whenever the table arrives without it.
Running again re-derives from that backup, never from a relabelled file.

Writes data/cell_classifications_v2.csv (brain_region relabelled)
"""
import os, shutil
import numpy as np, pandas as pd

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
CSV = f'{ROOT}/data/cell_classifications_v2.csv'
BAK = f'{ROOT}/data/cell_classifications_v2_atlas.csv'
PS = f'{ROOT}/data/population_state'
GRID_GAP_UM = 100
GRID_MAX_UM = 150


def grid_boundary(b, y):
    """Raise boundary b to the top of the grid-cell run (depths y) continuing up from below it."""
    y = np.sort(np.asarray(y))
    below = y[y >= b]
    if not len(below):
        return b
    cur = below[0]
    for v in y[y < b][::-1]:
        if cur - v > GRID_GAP_UM or b - v > GRID_MAX_UM:
            break
        cur = v
    return min(b, cur)


def family(a):
    a = str(a)
    if a.startswith('VIS'):
        return 'VIS'
    if a.startswith(('ENTm', 'ENTl', 'PAR', 'PRE', 'POST')):
        return 'MEC/PARA'
    return 'other'


if __name__ == '__main__':
    # a classification table without brain_region_atlas was written fresh (with
    # atlas labels) by the classification step: it becomes the atlas backup
    if not os.path.exists(BAK) or 'brain_region_atlas' not in pd.read_csv(CSV, nrows=1).columns:
        shutil.copy(CSV, BAK)
    C = pd.read_csv(BAK)
    C['brain_region_atlas'] = C.brain_region
    S = pd.read_csv(f'{PS}/rule_shanks.csv').set_index(['mouse', 'shank'])
    DC = pd.read_csv(f'{ROOT}/data/device_contact_id_annotations.csv')
    SH = DC.groupby(['mouse', 'shank_id']).coord_SCs_x.mean().abs()
    has = C.coord_SCs_x.notna() & C.coord_SCs_y.notna() & C.mouse.isin(S.index.get_level_values(0))
    C['shank'] = np.nan
    C.loc[has, 'shank'] = [int(np.argmin([abs(SH[(m, k)] - abs(x)) for k in range(4)]))
                           for m, x in zip(C.mouse[has], C.coord_SCs_x[has])]
    gc = has & (C.cell_class_of1 == 'GC')
    S['boundary_lfp'] = S.boundary
    for (m, k), g in C[gc].groupby(['mouse', 'shank']):
        if (m, k) in S.index:
            S.loc[(m, k), 'boundary'] = grid_boundary(S.loc[(m, k)].boundary_lfp, g.coord_SCs_y)
    S['grid_shift'] = S.boundary_lfp - S.boundary
    S.to_csv(f'{PS}/rule_shanks_cells.csv')
    print('boundary raised by grid cells (um):')
    print(S.grid_shift[S.grid_shift > 0].round(0).to_string())
    rule = pd.Series(index=C.index, dtype=object)
    for i in C.index[has]:
        m, k, y = int(C.mouse[i]), int(C.shank[i]), C.coord_SCs_y[i]
        if (m, k) not in S.index:
            continue
        s = S.loc[(m, k)]
        if y < s.top - 30:
            continue
        rule[i] = 'VIS' if y < s.boundary else 'MEC/PARA' if y <= s.exit + 30 else 'CB'
    fam = C.brain_region_atlas.map(family)
    new = C.brain_region_atlas.astype(object).copy()
    changed = 0
    for (m, d, k), g in C[rule.notna()].groupby(['mouse', 'day', 'shank']):
        for i in g.index:
            r = rule[i]
            if r == 'CB':
                new[i] = 'CB'; changed += fam[i] != 'other' or C.brain_region_atlas[i] != 'CB'
                continue
            if fam[i] == r:
                continue
            pool = g[(fam[g.index] == r)]
            if len(pool):
                j = (pool.coord_SCs_y - C.coord_SCs_y[i]).abs().idxmin()
                new[i] = C.brain_region_atlas[j]
            else:
                new[i] = 'VISp' if r == 'VIS' else 'ENTm'
            changed += 1
    C['brain_region'] = new
    C['anatomy_rule'] = rule
    C.drop(columns='shank').to_csv(CSV, index=False)
    cur = C.cell_class_of1.isin(['GC', 'NGS', 'NS'])
    print(f'{int(rule.notna().sum())} clusters placed, {changed} relabelled')
    print(pd.crosstab(C.brain_region_atlas[cur].map(family), C.brain_region[cur].map(family), margins=True))
