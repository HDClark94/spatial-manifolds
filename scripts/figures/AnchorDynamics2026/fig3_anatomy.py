"""Figure 3 — where MEC spatial cells sit, and what position predicts.

Usage:  python3 fig3_anatomy.py

    A     the gradient shown rather than summarised: open-field rate maps from
          four M25 sessions whose probes sat at different mediolateral
          positions, sampled PROPORTIONALLY so the grid:non-grid ratio drawn is
          the ratio actually recorded
    B, C  best-fit slices through the recorded cells -- one mouse (M25) and all
          mice pooled -- each under a small 3D view of the brain showing where
          the probe sat. Cells coloured by open-field identity over the Allen
          region annotation.
    D-F   identity across the anatomical axes: the grid fraction against
          mediolateral and dorsoventral position, and superficial against deep.
          Anterior-posterior is shown as superficial vs deep rather than as a
          coordinate, because in a structure as curved as MEC the AP coordinate
          is mostly a proxy for layer, and layer is the quantity with a
          mechanism attached to it.
    G-I   the same axes against whether a cell FOLLOWS the population anchoring
          state, pooled and within session.

SCOPE: THIS FIGURE IS ABOUT ANATOMY WITHIN MEC. The coarser question -- whether
a cell has to be in MEC at all, scored against each cell's own circular-shift
null -- is a between-region comparison and lives with the other regional panels
in the Figure 1 supplement (fig1_supp_nonmec.py).

ONE AXIS DOES EVERYTHING. Mediolateral position predicts both what a cell is and
whether it follows; dorsoventral position and probe depth predict neither; layer
predicts identity only.

    axis      predicts            within session (correct restriction)
    M-L       grid identity       rho -0.078, 18/27 sessions, p = 0.025
    M-L       following           rho -0.109, 19/27 sessions, p = 0.034
    layer     grid identity       8.9% sup vs 3.9% deep, 21/31, p = 0.002
    layer     following           rho +0.028, 25/39 sessions, p = 0.154  (NOT a panel)
    D-V       grid identity       p = 0.90
    D-V       following           p = 0.99
    A-P       grid identity       p = 0.10
    depth     following           p = 0.92

LAYER AGAINST FOLLOWING IS NOT PLOTTED, AND THE ROW ABOVE IS WHY IT IS STILL
RECORDED HERE. The panels test M-L, D-V and probe DEPTH against following; depth
is not layer, so for a while nothing in this file tested the quantity the text
was claiming when it called the responsive subnetwork "superficial". It does not
hold up at the standard this figure applies everywhere else: pooled, superficial
cells agree with the population axis more than deep ones (+0.162 vs +0.134,
p = 0.004), but pooling is unsafe here for exactly the reason it is unsafe
mediolaterally, and within session the effect is a non-significant trend
(+0.158 vs +0.139 paired, 24/36 sessions, p = 0.077). The superficial bias in
following is therefore the one carried by IDENTITY -- layer predicts what a cell
is, and what a cell is predicts following -- not a gradient in following of its
own. Run the test again before anyone restores the word to a heading.

WHICH SESSIONS CAN ANSWER THE QUESTION. Mediolateral position is assigned PER
SHANK, so a single-shank session holds no mediolateral variation at all and
contributes only noise -- 33 of 61 sessions span under 50 um. The within-session
test therefore runs on the 27 sessions spanning at least 200 um, two shank
pitches or more.

This matters more than it sounds. An earlier version selected sessions by
`nunique > 3`, the count of distinct mediolateral values, which EXCLUDES the
cleanly sampled 2-, 3- and 4-shank sessions (a 3-shank session holds three
discrete values while spanning 510 um) and ADMITS single-shank sessions whose
coordinates differ only by sub-micron numerical jitter. It ran the test almost
entirely on sessions with no mediolateral variation and reported p = 1.00 for an
effect that is present at p = 0.034.

THE CONTROL THAT MAKES THE RESTRICTION HONEST: for dorsoventral position and
probe depth the criterion changes nothing (p = 0.99 and 0.92 either way),
because every session varies along the shank. Only the axis sampled by shanks is
affected, so the restriction is not manufacturing the effect.

WHAT CANNOT BE SETTLED HERE. Mediolateral position predicts identity, and
identity predicts following, so the mediolateral gradient in following may be
nothing more than the identity gradient seen through it. Restricting to non-grid
cells leaves the effect size almost unchanged (rho -0.100 against -0.109) but
loses significance at 26 sessions (p = 0.111). The two accounts are not
separable at this sample size and the figure does not claim otherwise.
"""
import math
import os
import re
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle
from scipy.stats import spearmanr, wilcoxon
from tifffile import imread

plt.rcParams['font.family'] = 'Arial'
# panel letters are set as mathtext \bf inside the axes titles, so mathtext
# must resolve to Arial Bold rather than the default DejaVu
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'


def _lp(ax, s, dx=-.16, dy=1.0):
    """Bold panel letter, drawn separately from the title (see fig2_single_units.py
    for why: inline mathtext ties the letter's size to that panel's own title
    fontsize, which varies panel to panel)."""
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='right')
ROOT = '/Users/harryclark/Documents/spatial-manifolds'
FIG = f'{ROOT}/scripts/figures/AnchorDynamics2026'
PS = f'{ROOT}/data/population_state'
RENDERS = f'{ROOT}/data/anatomy_renders'
BG = '/Users/harryclark/.brainglobe/allen_mouse_10um_v1.2'
DC_PATH = '/Users/harryclark/Downloads/device_contact_id_annotations.csv'

COL_HIPPO, COL_PREPARA = '#9C6A38', '#ABDBD2'
COL_MEC, COL_VIS = '#2E7D6C', '#8C6BB1'
COL_GC, COL_NGS, COL_OTHER = '#c04744', '#3171ae', '#888888'
COL_CONTACT = '#ff6500'
CELL_ALPHA = .75
ML_JITTER_STD = 40.0
# a session must span two shank pitches before a within-session correlation
# along the mediolateral axis means anything -- see the module docstring
MIN_SPAN_UM = 200.0
# ML_BOUNDARY removed 2026-10-03. The 3400 um medial/lateral split was drawn in
# the slice and binned panels but no criterion for it was ever stated, and every
# within-session test in this figure runs on the CONTINUOUS coordinate, so the
# line asserted a threshold the analysis does not use.

# ── CCF transform and region shading, as in grid_cell_anatomy_v2 ─────────────
annotations_set = imread(f'{BG}/annotation.tiff')
structure_set = pd.read_csv(f'{BG}/structures.csv')
id_to_acr = dict(zip(structure_set['id'], structure_set['acronym']))


def stereo_to_ccf(SC, angle=-0.0873):
    SC = np.asarray(SC, float)
    stretch = SC / np.array([1.0, 0.9434, 1.0])
    rotate = np.array([stretch[0] * math.cos(angle) - stretch[1] * math.sin(angle),
                       stretch[0] * math.sin(angle) + stretch[1] * math.cos(angle),
                       stretch[2]])
    return rotate + np.array([5400.0, 440.0, 5700.0])


def _df_to_ccf(d):
    return np.array([stereo_to_ccf(r) for r in d[['SC_z', 'SC_y', 'SC_x']].values])


def _dc_to_ccf(d):
    return np.array([stereo_to_ccf(r) for r in
                     d[['coord_SCs_z', 'coord_SCs_y', 'coord_SCs_x']].values])


_RE = {1: re.compile(r'^(CA[1-4]?|DG)(-|$)'), 2: re.compile(r'^(PRE|PAR)\d*$'),
       3: re.compile(r'^ENTm'), 4: re.compile(r'^VIS')}


def _ids_to_numeric(aid_flat):
    numeric = np.zeros(len(aid_flat), dtype=int)
    for sid in np.unique(aid_flat):
        if sid == 0:
            continue
        acr = id_to_acr.get(sid, 'root')
        for k, rx in _RE.items():
            if rx.match(acr):
                numeric[aid_flat == sid] = k
                break
    return numeric


cmap_list = [(1, 1, 1, 0)]
for hexcol in (COL_HIPPO, COL_PREPARA, COL_MEC, COL_VIS):
    r, g, b = (int(hexcol.lstrip('#')[i:i + 2], 16) / 255 for i in (0, 2, 4))
    cmap_list.append((r, g, b, .55))
region_cmap = ListedColormap(cmap_list)
region_norm = BoundaryNorm(np.arange(len(cmap_list) + 1) - .5, region_cmap.N)

_df_regions = pd.read_csv(f'{ROOT}/data/cell_classifications_no_regions.csv')
_df_regions = _df_regions[_df_regions['mouse'] != 22]
_df_regions['SC_x'] = -np.abs(_df_regions['SC_x'])
_jit = np.random.default_rng(1)
_dc_all = pd.read_csv(DC_PATH)
_dc_all = _dc_all[_dc_all['mouse'] != 22]


def _apply_ml_jitter(d):
    d = d.copy()
    if ML_JITTER_STD > 0:
        d['SC_x'] = d['SC_x'] + _jit.normal(0, ML_JITTER_STD, size=len(d))
    return d


def get_mouse_pts(mouse=None):
    d = _df_regions if mouse is None else _df_regions[_df_regions['mouse'] == mouse]
    out = []
    for sel in (d['cell_type'] == 'GC', d['cell_type'] == 'NG',
                ~d['cell_type'].isin(['GC', 'NG'])):
        k = _apply_ml_jitter(d[sel].dropna(subset=['SC_x', 'SC_y', 'SC_z']))
        out.append(_df_to_ccf(k) if len(k) else np.empty((0, 3)))
    return out


def get_mouse_contacts(mouse=None):
    d = _dc_all if mouse is None else _dc_all[_dc_all['mouse'] == mouse]
    return _dc_to_ccf(d)


def fit_best_slice(fit_pts, extent_pts=None):
    """AP-DV-tilt-constrained best fit; the ML axis stays fixed in-plane.

    Fit to DEVICE CONTACTS, not to cell locations: per-cell ML is a per-shank
    nominal plus synthetic jitter, so fitting a plane to it would be fitting to
    the approximation rather than to the probe geometry that produced it.
    """
    if extent_pts is None:
        extent_pts = fit_pts
    centroid = fit_pts.mean(axis=0)
    ay_c = fit_pts[:, [0, 1]] - fit_pts[:, [0, 1]].mean(axis=0)
    _, _, Vt = np.linalg.svd(ay_c, full_matrices=False)
    major = Vt[0] if Vt[0][1] >= 0 else -Vt[0]
    minor = Vt[1] if Vt[1][0] >= 0 else -Vt[1]
    basis2 = np.array([major[0], major[1], 0.0])

    def project(pts):
        return 5700.0 - pts[..., 2], (pts - centroid) @ basis2

    def reconstruct(u, v):
        p = np.empty(np.broadcast(np.asarray(u), np.asarray(v)).shape + (3,))
        p[..., 0] = centroid[0] + v * basis2[0]
        p[..., 1] = centroid[1] + v * basis2[1]
        p[..., 2] = 5700.0 - u
        return p

    u_e, v_e = project(extent_pts)
    pu, pv = (u_e.max() - u_e.min()) * .08, (v_e.max() - v_e.min()) * .08
    return dict(u_lim=(u_e.min() - pu, u_e.max() + pu),
                v_lim=(v_e.min() - pv, v_e.max() + pv),
                tilt_deg=np.degrees(np.arctan2(minor[1], minor[0])),
                project=project, reconstruct=reconstruct)


def sample_region_grid(fit, step=10):
    u_vals = np.arange(*fit['u_lim'], step)
    v_vals = np.arange(*fit['v_lim'], step)
    V, U = np.meshgrid(v_vals, u_vals, indexing='ij')
    g = fit['reconstruct'](U, V)
    idx = [np.round(g[..., k] / 10).astype(int) for k in range(3)]
    sh = annotations_set.shape
    ok = np.ones(idx[0].shape, bool)
    for k in range(3):
        ok &= (idx[k] >= 0) & (idx[k] < sh[k])
    aid = np.zeros(idx[0].shape, int)
    aid[ok] = annotations_set[idx[0][ok], idx[1][ok], idx[2][ok]]
    return u_vals, v_vals, _ids_to_numeric(aid.ravel()).reshape(idx[0].shape)


def render_brain(mouse, path):
    """Small 3D view of the brain with MEC and this mouse's contacts.

    Rendered in a SUBPROCESS: brainrender opens a VTK window and loading the
    10 um atlas leaves state behind that interferes with matplotlib's Agg
    backend in the same interpreter. Cached, because the render takes minutes.
    """
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    code = f'''
import numpy as np, pandas as pd, math
from brainrender import Scene, settings as s
from brainrender.actors import Points
s.SHOW_AXES = False; s.BACKGROUND_COLOR = [1, 1, 1]; s.OFFSCREEN = True
# 'cartoon', the default, draws a silhouette line on every internal mesh edge,
# which at thumbnail size reads as scribble over the brain
s.SHADER_STYLE = 'plastic'
s.ROOT_ALPHA = 0.10
def t(SC, a=-0.0873):
    SC = np.asarray(SC, float); st = SC / np.array([1.0, 0.9434, 1.0])
    return np.array([st[0]*math.cos(a)-st[1]*math.sin(a),
                     st[0]*math.sin(a)+st[1]*math.cos(a), st[2]]) + np.array([5400.,440.,5700.])
dc = pd.read_csv({DC_PATH!r}); dc = dc[dc.mouse != 22]
{'dc = dc[dc.mouse == %d]' % mouse if mouse is not None else ''}
pts = np.array([t(r) for r in dc[['coord_SCs_z','coord_SCs_y','coord_SCs_x']].values])
sc = Scene(root=True, atlas_name='allen_mouse_10um', title='')
for nm, col, al in (('ENTm', {COL_MEC!r}, .5), ('HPF', {COL_HIPPO!r}, .18)):
    try: sc.add_brain_region(nm, alpha=al, color=col)
    except Exception: pass
sc.add(Points(pts, name='c', colors={COL_CONTACT!r}, radius=45, alpha=.9))
cen = pts.mean(0); F = (float(cen[0]), float(cen[1]), float(-cen[2]))
a = np.radians(320)
sc.render(camera={{'pos': (F[0]+30000*np.cos(a), F[1]-9000, F[2]-30000*np.sin(a)),
                  'focal_point': (6500., 4000., -5700.), 'viewup': (0,-1,0),
                  'clipping_range': (12000, 70000)}}, interactive=False, zoom=0.95)
sc.screenshot(name={path[:-4]!r}, scale=2)
sc.close()
'''
    # the first render loads the 10 um atlas and can take several minutes;
    # a timeout keeps a stalled VTK context from blocking the whole figure
    subprocess.run([sys.executable, '-c', code], check=True,
                   capture_output=True, timeout=1500)
    if not os.path.exists(path):
        raise RuntimeError('brainrender produced no screenshot')
    return path


def trim_white(img, pad=6):
    """Crop the white margin brainrender leaves around the brain."""
    a = img[..., :3] if img.ndim == 3 else img
    ink = (a.min(axis=2) < .97) if a.ndim == 3 else (a < .97)
    if not ink.any():
        return img
    ys, xs = np.where(ink)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, img.shape[1])
    return img[y0:y1, x0:x1]


# ── quantitative data ───────────────────────────────────────────────────────
C = pd.read_csv(f'{ROOT}/data/cell_classifications_v2.csv')
U = pd.read_csv(f'{PS}/unit_table.csv')[['mouse', 'day', 'cluster_id', 'r', 'identity']]
D = C.merge(U, on=['mouse', 'day', 'cluster_id'], how='left')
mec = D[D.brain_region.astype(str).str.startswith('ENTm')].copy()
mec['ml'] = mec.coord_SCs_x.abs()
mec['dv'] = mec.coord_SCs_y
mec['depth'] = mec.coord_probe_y
mec['layer'] = mec.brain_region.str.extract(r'ENTm(\d)')[0]
mec['sup'] = mec.layer.isin(['1', '2', '3'])
sp = mec[mec.cell_class_of1.isin(['GC', 'NGS'])].copy()
sp['gc'] = (sp.cell_class_of1 == 'GC').astype(float)
print(f'{len(mec)} MEC cells, {len(sp)} spatial, '
      f'{sp.groupby(["mouse", "day"]).ngroups} sessions, {sp.mouse.nunique()} mice')


def multi_shank(df, axis):
    span = df.groupby(['mouse', 'day'])[axis].agg(lambda v: v.max() - v.min())
    return set(span[span >= MIN_SPAN_UM].index)


def within_session(df, axis, yvar, restrict=True):
    sel = multi_shank(df, axis) if restrict else None
    ws = []
    for (mo, dy), g in df.dropna(subset=[yvar]).groupby(['mouse', 'day']):
        if sel is not None and (mo, dy) not in sel:
            continue
        if len(g) < 20 or g[axis].nunique() < 2 or g[yvar].nunique() < 2:
            continue
        r = spearmanr(g[axis], g[yvar])[0]
        if np.isfinite(r):
            ws.append(r)
    ws = np.array(ws)
    p = wilcoxon(ws).pvalue if len(ws) >= 6 else np.nan
    return ws, p


def binned(df, axis, yvar, nbin=8):
    q = df.dropna(subset=[axis, yvar])
    e = np.quantile(q[axis], np.linspace(0, 1, nbin + 1))
    e[-1] += 1e-6
    k = np.digitize(q[axis], e[1:-1])
    g = q.groupby(k)
    return (g[axis].mean().values, g[yvar].mean().values,
            (g[yvar].std(ddof=1) / np.sqrt(g[yvar].size())).values)



# ── M25 rate maps: the mediolateral gradient, shown rather than summarised ───
# Four M25 sessions whose probes sat at different mediolateral positions, from
# ~3000 um (medial) to ~3720 um (lateral). Cells are sampled PROPORTIONALLY --
# the grid:non-grid ratio drawn in each block is the ratio actually recorded in
# that session -- so the panel reports the gradient directly rather than
# illustrating it with hand-picked examples. Grid cells are taken in grid-score
# order and non-grid cells spread evenly across the dorsoventral extent so the
# block is not all one depth.
import pynapple as nap
from spatial_manifolds.detect_grids import (_fill_nans_from_neighbors,
                                            compute_travel_projected,
                                            gaussian_filter_nan)

SRC = '/Users/harryclark/Downloads/clark2025/'
RM_SESSIONS = [(24, 3000), (19, 3270), (22, 3540), (23, 3720)]
RM_N, RM_COLS = 24, 3          # 24 cells per session, 8 rows of 3
_cls = pd.read_csv(f'{ROOT}/data/cell_classifications.csv')


_OF_CACHE = {}


def _of_session(mouse, day):
    """Loaded once per session: 24 cells x 4 sessions is 96 maps, and reopening
    both NWB files for each of them dominates the figure's runtime."""
    k = (mouse, day)
    if k not in _OF_CACHE:
        stem = f'{SRC}M{mouse}/D{day}/OF1/sub-M{mouse}_ses-D{day}_typ-OF1'
        _OF_CACHE[k] = (nap.load_file(f'{stem}_beh.nwb'),
                        nap.load_file(f'{stem}_srt-kilosort4_clusters.npz'))
    return _OF_CACHE[k]


def of_rate_map(cid, mouse, day, df):
    beh, clusters = _of_session(mouse, day)
    lag = df[df.cluster_id == cid].travel.values[0]
    pos = np.stack([beh['P_x'], beh['P_y']], axis=1)
    bl = compute_travel_projected(['P_x', 'P_y'], pos, pos, lag)
    pl = np.stack([bl['P_x'], bl['P_y']], axis=1)
    tc = nap.compute_2d_tuning_curves(nap.TsGroup([clusters[cid]]), pl,
                                      nb_bins=(40, 40))[0][0]
    tc = gaussian_filter_nan(_fill_nans_from_neighbors(tc), sigma=(2.5, 2.5))
    return np.clip(tc, 0, np.nanpercentile(tc, 99))


def pick_cells(day):
    sub = _cls[(_cls.mouse == 25) & (_cls.day == day)]
    gc = sub[sub.cell_type == 'GC'].copy()
    ngs = sub[sub.cell_type == 'NG'].copy()
    tot = len(gc) + len(ngs)
    n_gc = max(round(len(gc) / tot * RM_N), min(1, len(gc))) if tot else 0
    gc = gc.sort_values('grid_score_best', ascending=False).head(n_gc)
    ngs = ngs.sort_values('SC_y')
    if len(ngs) > RM_N - n_gc:
        ngs = ngs.iloc[np.linspace(0, len(ngs) - 1, RM_N - n_gc).astype(int)]
    ngs = ngs.sort_values('spatial_information_score_best', ascending=False)
    gc = gc.assign(_t='GC'); ngs = ngs.assign(_t='NGS')
    return pd.concat([gc, ngs]).reset_index(drop=True), len(gc), len(ngs)


# ── figure ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14.2, 13.4))
G = fig.add_gridspec(4, 1, height_ratios=[2.45, 1, 1.25, 1], hspace=.38)
# the rate-map block and the equal-aspect slices are both TALL, so they sit
# side by side rather than stacked -- stacked, each wastes the other's width
# No legend column. A vertical key down the right-hand side cost ~10% of the
# width of the widest row in the figure, squeezing both the rate maps and the
# slices; the same eight entries fit in a single horizontal strip under the
# row, where they cost height that was whitespace anyway.
TOP = G[0].subgridspec(1, 5, width_ratios=[.26, 2.48, .66, .66, .66],
                       wspace=.26)

# A: the gradient shown as rate maps, medial -> lateral
g0 = TOP[1].subgridspec(1, len(RM_SESSIONS), wspace=.12)
for bi, (day, ml) in enumerate(RM_SESSIONS):
    cells, n_gc, n_ngs = pick_cells(day)
    gb = g0[bi].subgridspec(RM_N // RM_COLS, RM_COLS, wspace=.08, hspace=.08)
    for ci, (_, row) in enumerate(cells.iterrows()):
        axm = fig.add_subplot(gb[ci])
        try:
            axm.imshow(of_rate_map(int(row.cluster_id), 25, day, cells), cmap='viridis')
        except Exception:
            axm.set_facecolor('0.92')
        axm.set_xticks([]); axm.set_yticks([])
        for _spine in axm.spines.values():
            _spine.set_color(COL_GC if row._t == 'GC' else COL_NGS)
            _spine.set_linewidth(1.1)
        if ci == 1:
            axm.set_title(f'D{day}  ML≈{ml} µm\n{n_gc} grid + {n_ngs} non-grid',
                          fontsize=6.6, pad=4)
    print(f'  M25 D{day} (ML≈{ml}): {n_gc} grid + {n_ngs} non-grid drawn')
# the recording sequence, so that the rate maps are not read as task data: every
# map in A comes from OF1, the arena session that PRECEDES the virtual-reality
# task, which is what makes cell identity independent of the behaviour it is
# later used to explain
axseq = fig.add_subplot(TOP[0]); axseq.axis('off')
axseq.set_xlim(0, 1); axseq.set_ylim(0, 1)
_boxes = [('OF 1', .78, True), ('VR', .50, False), ('OF 2', .22, False)]
for _lab, _yc, _hot in _boxes:
    axseq.add_patch(Rectangle((.14, _yc - .055), .72, .11,
                              facecolor=COL_GC if _hot else '0.92',
                              edgecolor='0.45' if not _hot else COL_GC, lw=.8,
                              zorder=2))
    axseq.text(.50, _yc, _lab, ha='center', va='center', fontsize=6.4, zorder=3,
               color='white' if _hot else '0.25',
               weight='bold' if _hot else 'normal')
for _y0, _y1 in ((.725, .615), (.445, .335)):
    axseq.annotate('', xy=(.50, _y1), xytext=(.50, _y0),
                   arrowprops=dict(arrowstyle='-|>', color='0.45', lw=.8))
axseq.text(.50, .92, 'session order', ha='center', va='center', fontsize=6.2,
           style='italic', color='0.35')
axseq.text(.50, .08, 'maps in A are\nfrom OF 1', ha='center', va='center',
           fontsize=5.8, color=COL_GC)

fig.text(g0[0].get_position(fig).x0 - .028, G[0].get_position(fig).y1,
         'A', fontsize=10, weight='bold', va='top')
_p0, _p1 = g0[0].get_position(fig), g0[len(RM_SESSIONS) - 1].get_position(fig)
fig.text((_p0.x0 + _p1.x1) / 2, _p0.y0 - .012,
         'medial  ←' + '—' * 46 + '→  lateral', fontsize=6.4, ha='center',
         va='top', style='italic', color='0.3')

# B, C: a small 3D view over each best-fit slice

# Three individual mice rather than one mouse plus a pooled fit. The pooled
# slice averaged over animals whose probes sat at different mediolateral
# positions, which is exactly the variable the panel is about; three mice that
# each reach the medial end show the grid enrichment there directly. M29 and
# M28 have the highest grid fractions after M25 (6.0% and 4.4%) and both
# sample in to 3000-3120 um.
SLICES = [('B', 'M25', 25), ('C', 'M29', 29), ('D', 'M28', 28)]
for k, (letter, name, mouse) in enumerate(SLICES):
    gg = TOP[k + 2].subgridspec(2, 1, height_ratios=[.34, 1], hspace=.04)
    axb = fig.add_subplot(gg[0])
    try:
        png = render_brain(mouse, f'{RENDERS}/brain_{"all" if mouse is None else f"m{mouse}"}.png')
        axb.imshow(trim_white(plt.imread(png)))
    except Exception as e:
        axb.text(.5, .5, f'render unavailable\n{type(e).__name__}', ha='center',
                 va='center', transform=axb.transAxes, fontsize=6.5, color='0.5')
    axb.axis('off')
    _lp(axb, letter)
    axb.set_title(f'{name}', fontsize=8.5, loc='left')

    ax = fig.add_subplot(gg[1])
    gc_p, ngs_p, oth_p = get_mouse_pts(mouse)
    contacts = get_mouse_contacts(mouse)
    allp = np.concatenate([p for p in (gc_p, ngs_p, oth_p) if len(p)])
    fit = fit_best_slice(contacts, extent_pts=allp)
    u_vals, v_vals, grid = sample_region_grid(fit)
    # both axes are micrometres, so the panel is drawn to scale rather than
    # stretched to fill its cell -- a mediolateral micron and a dorsoventral
    # micron are the same distance and the slice should look like the tissue
    ax.imshow(grid, origin='lower', aspect='equal', cmap=region_cmap,
              norm=region_norm,
              extent=[u_vals[0], u_vals[-1], v_vals[0], v_vals[-1]], zorder=1)
    ax.set_aspect('equal', adjustable='box')
    rng = np.random.default_rng(0)
    xs, ys, cs = [], [], []
    for p, col in ((oth_p, COL_OTHER), (ngs_p, COL_NGS), (gc_p, COL_GC)):
        if not len(p):
            continue
        u, v = fit['project'](p)
        xs.append(u); ys.append(v); cs.append(np.full(len(u), col))
    xs, ys, cs = (np.concatenate(a) for a in (xs, ys, cs))
    o = rng.permutation(len(xs))
    ax.scatter(xs[o], ys[o], color=cs[o], s=3.0, alpha=CELL_ALPHA, lw=0, zorder=3)
    ax.set_xlabel('medial–lateral (µm)', fontsize=8)
    if k == 0:
        ax.set_ylabel('dorsal–ventral (µm)', fontsize=8)
    ax.invert_yaxis()
    # equal aspect makes the panel narrow, so the default tick density collides
    ax.xaxis.set_major_locator(plt.MaxNLocator(2))
    ax.tick_params(labelsize=6.2)
    ax.spines[['top', 'right']].set_visible(False)
    n_gc, n_ngs = len(gc_p), len(ngs_p)
    ax.set_title(f'{n_gc} grid, {n_ngs} non-grid\n{fit["tilt_deg"]:+.1f}° from coronal',
                 fontsize=6.4, loc='left')

_handles = ([Patch(facecolor=c, label=l) for l, c in
             (('Hippocampus', COL_HIPPO), ('Pre/parasubiculum', COL_PREPARA),
              ('MEC', COL_MEC), ('Visual cortex', COL_VIS))]
            + [Patch(facecolor=c, label=l) for l, c in
               (('grid cell', COL_GC), ('non-grid spatial', COL_NGS),
                ('other', COL_OTHER))]
            + [Patch(facecolor=COL_CONTACT, label='probe contacts')])
# Centred in the GAP between the first and second rows, not hung directly under
# row 0: the slice panels' x tick labels and axis title extend below their axes
# box, and anchoring to G[0].y0 printed the legend straight through them.
_y_mid = (G[0].get_position(fig).y0 + G[1].get_position(fig).y1) / 2
fig.legend(handles=_handles, loc='center', bbox_to_anchor=(.56, _y_mid),
           ncol=8, fontsize=6.8, frameon=False, handlelength=1.1,
           handleheight=1.1, columnspacing=1.3, borderpad=0)

# C-E: identity across the axes
g2 = G[1].subgridspec(1, 3, wspace=.42)
for k, (axis, lab, letter) in enumerate((('ml', 'medial–lateral (µm)', 'E'),
                                         ('dv', 'dorsal–ventral (µm)', 'F'))):
    ax = fig.add_subplot(g2[k])
    x, y, e = binned(sp, axis, 'gc')
    ax.fill_between(x, y - e, y + e, color=COL_GC, alpha=.25, lw=0, edgecolor='none')
    ax.plot(x, y, color=COL_GC, lw=1.8)
    ws, p = within_session(sp, axis, 'gc')
    ax.set_xlabel(lab, fontsize=8)
    if k == 0:
        ax.set_ylabel('fraction grid\n(of spatial cells)', fontsize=8)
    ax.set_ylim(0, max(ax.get_ylim()[1], .02) * 1.22)
    _lp(ax, letter)
    ax.set_title(f'within session rho = {np.median(ws):+.3f}\n'
                 f'p = {p:.3g}  ({len(ws)} sessions)', fontsize=7, loc='left')
    ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

# E: layer, which is what the AP axis is really reporting in a curved structure
ax = fig.add_subplot(g2[2])
rows = []
for (mo, dy), g in sp.groupby(['mouse', 'day']):
    a, b = g[g.sup], g[~g.sup]
    if len(a) >= 10 and len(b) >= 10:
        rows.append((a.gc.mean(), b.gc.mean()))
R = np.array(rows)
pl = wilcoxon(R[:, 0] - R[:, 1]).pvalue
for i in range(len(R)):
    ax.plot([0, 1], R[i], color='0.75', lw=.7, zorder=2)
for j, (col, nm) in enumerate(((COL_GC, 'superficial'), ('0.45', 'deep'))):
    ax.errorbar(j, R[:, j].mean(), yerr=R[:, j].std(ddof=1) / np.sqrt(len(R)),
                color=col, marker='o', ms=6, lw=1.6, capsize=4, zorder=4)
ax.set_xticks([0, 1]); ax.set_xticklabels(['superficial\n(L1–3)', 'deep\n(L5–6)'],
                                          fontsize=7)
ax.set_xlim(-.4, 1.4)
ax.set_ylabel('fraction grid', fontsize=8)
_lp(ax, 'G')
ax.set_title(f'higher in {int((R[:, 0] > R[:, 1]).sum())}/{len(R)} sessions\n'
             f'p = {pl:.3g}', fontsize=7, loc='left')
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

# ── G: the raster itself, ordered mediolaterally ─────────────────────────────
# Panels H and I report the mediolateral gradient in following as a correlation.
# This shows the thing the correlation is computed on. Cells are ordered by
# mediolateral position rather than by PC1 loading, so if position organises
# following at all, the left of the raster should track the population state
# more tightly than the right.
#
# M25 D25, chosen for continuity with the rate maps in panel A, which are all
# M25. READ THE ANCHORED FRACTION IN THE TITLE BEFORE READING THE RASTER: this
# session is anchored on most of its trials, so most of the raster is one colour
# and the panel shows how the cells are sampled across shanks rather than a
# strong state alternation. M28 D25 is the best-balanced multi-shank session in
# the dataset (49% anchored, 480 um, 4 shanks) and is the one to switch to if
# the panel is meant to carry the state dynamics as well as the sampling.
RAST_MO, RAST_DY = 25, 25
from scipy.ndimage import median_filter
from spatial_manifolds.anchoring import (ANCH_COLOR as ANCH_C,
                                         NONANCH_COLOR as NONANCH_C,
                                         load_session_labels)

# wspace is wider than the rows above because this one now carries a colorbar
# on the slice panel: the bar's tick labels and the raster's rotated y-label
# both float outward by a fixed point padding that no gridspec setting accounts
# for, so a gap that looks ample between the axes boxes still collides in ink.
gR = G[2].subgridspec(1, 5, width_ratios=[.88, 2.25, .30, .18, 1.55],
                      wspace=.46)
_z = load_session_labels(RAST_MO, RAST_DY)
_ids = [int(c) for c in _z['cluster_id']]
_co = (C[(C.mouse == RAST_MO) & (C.day == RAST_DY)]
       .assign(ml=lambda d: d.coord_SCs_x.abs())
       .dropna(subset=['ml']).set_index('cluster_id').ml.to_dict())
_keep = [(i, c) for i, c in enumerate(_ids) if c in _co]
_keep.sort(key=lambda t: _co[t[1]])              # medial -> lateral
_rows = [i for i, _ in _keep]
_mls = np.array([_co[c] for _, c in _keep])
_L = _z['labels'][_rows]
_frac = np.asarray(_z['frac_anch'], float)
_n_tr = _L.shape[1]
_st = median_filter((_frac > .5).astype(float), size=9, mode='nearest') > .5
_tr = list(np.where(np.diff(_st.astype(int)) != 0)[0] + 1)

# ---- H: the same session on its own best-fit slice -----------------------
# The raster beside this orders cells medial -> lateral as a number; this shows
# where those cells actually sit, in the style of the region-example rows, so
# the gradient can be read off the tissue rather than off an axis label.
_rr0 = (pd.read_csv(f'{PS}/pc1_by_region.csv')
        .query('mouse == @RAST_MO and day == @RAST_DY')
        .set_index('cluster_id'))
axsl = fig.add_subplot(gR[0])
_cells = (C[(C.mouse == RAST_MO) & (C.day == RAST_DY)]
          .dropna(subset=['coord_SCs_x', 'coord_SCs_y', 'coord_SCs_z']))
_cpts = np.array([stereo_to_ccf(r) for r in
                  _cells[['coord_SCs_z', 'coord_SCs_y',
                          'coord_SCs_x']].values])
_contacts = get_mouse_contacts(RAST_MO)
_fitH = fit_best_slice(_contacts, extent_pts=_cpts)
_uv, _vv, _grid = sample_region_grid(_fitH)
axsl.imshow(_grid, origin='lower', aspect='equal', cmap=region_cmap,
            norm=region_norm, zorder=1,
            extent=[_uv[0], _uv[-1], _vv[0], _vv[-1]])
_pu, _pv = _fitH['project'](_cpts)
_cr = np.array([_rr0.r.get(c, np.nan) for c in _cells.cluster_id])
_okc = np.isfinite(_cr)
axsl.scatter(_pu[~_okc], _pv[~_okc], s=2.0, color='0.80', alpha=.5, lw=0, zorder=2)
_sc = axsl.scatter(_pu[_okc], _pv[_okc], c=_cr[_okc], s=7.0, cmap='viridis',
                   vmin=np.nanpercentile(_cr, 5), vmax=np.nanpercentile(_cr, 95),
                   alpha=.9, lw=0, zorder=3)
axsl.set_aspect('equal', adjustable='box')
axsl.invert_yaxis()
axsl.set_xlabel('medial–lateral (µm)', fontsize=7)
axsl.set_ylabel('dorsal–ventral (µm)', fontsize=7)
axsl.xaxis.set_major_locator(plt.MaxNLocator(2))
axsl.yaxis.set_major_locator(plt.MaxNLocator(3))
axsl.tick_params(labelsize=6)
axsl.spines[['top', 'right']].set_visible(False)
_cb = fig.colorbar(_sc, ax=axsl, fraction=.045, pad=.03)
# label ABOVE the bar, not rotated beside it: the rotated label floats outward
# by a fixed point padding and ran into the raster's y-axis label next door
_cb.ax.set_title('agreement\n(r)', fontsize=5.5, pad=3, linespacing=1.15)
_cb.ax.tick_params(labelsize=5.5); _cb.outline.set_visible(False)
_lp(axsl, 'H', dx=-.42, dy=1.02)
axsl.set_title('where these cells sit', fontsize=7.4, loc='left')

TA_ = ListedColormap([NONANCH_C, ANCH_C])
NORM_ = BoundaryNorm([-.5, .5, 1.5], TA_.N)
ax = fig.add_subplot(gR[1])
ax.imshow(_L.T, aspect='auto', cmap=TA_, norm=NORM_, interpolation='nearest',
          extent=[0, len(_rows), _n_tr, 1])
for tb in _tr:
    ax.axhline(tb, color='0.1', lw=.8, ls='--', zorder=4)
# shank boundaries: ML is assigned per shank, so these are the sampling units
for b in np.where(np.diff(_mls) > 1)[0]:
    ax.axvline(b + 1, color='k', lw=1.1, zorder=5)
for u in np.unique(_mls):
    m = _mls == u
    ax.annotate(f'{u:.0f}', ((np.where(m)[0].mean()), 1.012),
                xycoords=('data', 'axes fraction'), ha='center', va='bottom',
                fontsize=6, color='0.25')
ax.set_xlabel('MEC cells, ordered medial → lateral  (µm, per shank)', fontsize=8)
ax.set_ylabel('Trial', fontsize=8)
ax.set_title(f'M{RAST_MO} D{RAST_DY} — the raster behind I, ordered by position '
             f'rather than by PC1\n{len(_rows)} MEC cells across '
             f'{len(np.unique(_mls))} shanks, {(_frac > .5).mean() * 100:.0f}% of '
             f'trials anchored', fontsize=7.4, loc='left', pad=14)
ax.tick_params(labelsize=7)
for sp_ in ax.spines.values():
    sp_.set_visible(False)

axp = fig.add_subplot(gR[2])
_pc = np.nan_to_num(np.asarray(_z['pc1'], float))
_y = np.arange(1, len(_pc) + 1)
axp.fill_betweenx(_y, 0, _pc, where=_pc >= 0, color=ANCH_C, lw=0, edgecolor='none',
                  interpolate=True)
axp.fill_betweenx(_y, 0, _pc, where=_pc < 0, color=NONANCH_C, lw=0, edgecolor='none',
                  interpolate=True)
axp.axvline(0, color='0.4', lw=.7)
for tb in _tr:
    axp.axhline(tb, color='0.1', lw=.8, ls='--', zorder=4)
axp.set_ylim(_n_tr, 1); axp.set_xticks([]); axp.tick_params(labelleft=False, left=False)
axp.set_title('PC1', fontsize=7.5, pad=4)
for sp_ in axp.spines.values():
    sp_.set_visible(False)

# the same cells' agreement, on the same ordering -- the link to panel H
axa = fig.add_subplot(gR[4])
_rr = (pd.read_csv(f'{PS}/pc1_by_region.csv')
       .query('mouse == @RAST_MO and day == @RAST_DY')
       .set_index('cluster_id'))
_cid = [c for _, c in _keep]
_rv = np.array([_rr.r.get(c, np.nan) for c in _cid])
_ok = np.isfinite(_rv)
axa.scatter(_mls[_ok], _rv[_ok], s=14, color='0.45', alpha=.8, lw=0, zorder=3)
for u in np.unique(_mls):
    m = (_mls == u) & _ok
    if m.sum() >= 3:
        axa.errorbar(u, _rv[m].mean(), yerr=_rv[m].std(ddof=1) / np.sqrt(m.sum()),
                     color='k', marker='o', ms=6, lw=1.6, capsize=4, zorder=4)
_rho = spearmanr(_mls[_ok], _rv[_ok])
axa.set_xlabel('medial–lateral (µm)', fontsize=8)
axa.set_ylabel('agreement with\npopulation axis (r)', fontsize=8)
axa.set_title(f'this session: rho = {_rho[0]:+.3f}, p = {_rho[1]:.3g}\n'
              f'(one point of the 27 in panel J)', fontsize=7, loc='left')
axa.tick_params(labelsize=7); axa.spines[['top', 'right']].set_visible(False)
print(f'  raster M{RAST_MO}D{RAST_DY}: {len(_rows)} cells, '
      f'{len(np.unique(_mls))} shanks, rho={_rho[0]:+.3f} p={_rho[1]:.3g}')

# H-J: the same axes against FOLLOWING
g3 = G[3].subgridspec(1, 3, wspace=.42)
foll = mec.dropna(subset=['r'])
for k, (axis, lab, letter) in enumerate((('ml', 'medial–lateral (µm)', 'I'),
                                         ('dv', 'dorsal–ventral (µm)', 'J'))):
    ax = fig.add_subplot(g3[k])
    x, y, e = binned(foll, axis, 'r')
    ax.fill_between(x, y - e, y + e, color='0.35', alpha=.28, lw=0, edgecolor='none')
    ax.plot(x, y, color='k', lw=1.8)
    rho_p = spearmanr(foll[axis], foll.r)
    ws, p = within_session(foll, axis, 'r')
    ax.set_xlabel(lab, fontsize=8)
    if k == 0:
        ax.set_ylabel('agreement with\npopulation axis (r)', fontsize=8)
    _lp(ax, letter)
    ax.set_title(f'pooled rho = {rho_p[0]:+.2f} (p = {rho_p[1]:.0e})\n'
                 f'WITHIN SESSION rho = {np.median(ws):+.3f}, p = {p:.3g} '
                 f'({len(ws)} sess.)', fontsize=6.6, loc='left')
    ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

# H: the within-session distributions side by side, which is the actual test
ax = fig.add_subplot(g3[2])
rng = np.random.default_rng(0)
SUMM = [('M–L', 'ml', foll, 'r', True), ('D–V', 'dv', foll, 'r', True),
        ('depth', 'depth', foll, 'r', True)]
for i, (nm, axis, df, yv, rs) in enumerate(SUMM):
    ws, p = within_session(df, axis, yv, restrict=rs)
    ax.scatter(i + rng.uniform(-.16, .16, len(ws)), ws, s=13, color='0.45',
               alpha=.8, lw=0, zorder=3)
    ax.plot([i - .28, i + .28], [np.median(ws)] * 2, color='k', lw=2, zorder=4)
    ax.annotate(f'p = {p:.3g}' if p >= .001 else f'p = {p:.1e}',
                (i, .98), xycoords=('data', 'axes fraction'), ha='center', va='top',
                fontsize=6.2, color='#a33' if p < .05 else '0.35')
ax.axhline(0, color='0.6', lw=.8, ls=':')
ax.set_xticks(range(len(SUMM)))
ax.set_xticklabels([s[0] for s in SUMM], fontsize=7.5)
ax.set_xlim(-.5, len(SUMM) - .5)
ax.set_ylabel('within-session rho', fontsize=8)
_lp(ax, 'K')
ax.set_title(f'only M–L survives\n(M–L: {len(within_session(foll, "ml", "r")[0])} '
             f'multi-shank sessions)', fontsize=7, loc='left')
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)


out = f'{FIG}/fig3_anatomy.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'\nsaved {out}')
