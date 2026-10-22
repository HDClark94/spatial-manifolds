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
from matplotlib.colors import TwoSlopeNorm, BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
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
DC_PATH = '/Users/harryclark/Documents/spatial-manifolds/data/device_contact_id_annotations.csv'  # build_device_contacts.py, from clark2025

# Region shading. The previous palette gave MEC a teal-green and visual cortex
# a purple, which read as the non-anchored teal and the anchored magenta on the
# rasters in H and I. These avoid both state colours, the red and blue that
# Figure 2 reserves for cell types, the interneuron orange and the contact
# orange. MEC is split into superficial (L1-3) and deep (L5-6), the paper's own
# layer split; there are no layer-4 cells in the data.
COL_HIPPO, COL_PREPARA = '#BDBDBD', '#A3A85E'
COL_MEC_SUP, COL_MEC_DEEP = '#E3BC4F', '#8C5A2B'
COL_VIS, COL_REGION_OTHER = '#7F8C99', '#E8E8E8'
COL_MEC = COL_MEC_SUP                     # kept for any older reference
# GREY KEY, as Figure S7: the cells carry the colour, the tissue only says where
# it is. One grey per class, used for the slices, the raster strips, the legend
# and the 3D renders alike. S7's own levels put visual cortex and pre/para at
# 0.90 and 0.87, indistinguishable on a strip, so these are spread a little
# further; deep MEC is a step darker than superficial to keep the layer split.
REGION_GREY = {1: .96, 2: .86, 3: .80, 4: .69, 5: .91}     # by region class
GREY_OTHER = .975
REGION_ALPHA = .55
REGION_COLS = (COL_HIPPO, COL_PREPARA, COL_MEC_SUP, COL_MEC_DEEP, COL_VIS)
REGION_NAMES = ('Hippocampus', 'Pre/parasubiculum', 'MEC superficial (L1–3)',
                'MEC deep (L5–6)', 'Visual cortex')


def _blend(hexcol, a=REGION_ALPHA):
    """The colour a region patch shows on white paper at alpha a."""
    rgb = np.array([int(hexcol.lstrip('#')[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    return tuple(a * rgb + (1 - a))
COL_GC, COL_NGS, COL_OTHER = '#c04744', '#3171ae', '#888888'
COL_CONTACT = '#ff6500'
CELL_ALPHA = .75
ML_JITTER_STD = 40.0
ML_JITTER_FIG4 = 15.0         # um, display jitter on the Figure 4 slices
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
       3: re.compile(r'^ENTm[123]'), 4: re.compile(r'^ENTm[456]'),
       5: re.compile(r'^VIS')}


def region_class(acr):
    """0 other, 1 hippocampus, 2 pre/para, 3 MEC sup, 4 MEC deep, 5 visual."""
    for k, rx in _RE.items():
        if rx.match(str(acr)):
            return k
    return 0


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


cmap_list = [(1, 1, 1, 0)] + [(REGION_GREY[k],) * 3 + (1.0,) for k in (1, 2, 3, 4, 5)]
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
for nm, col, al in ([(f'ENTm{{k}}', {str(REGION_GREY[3])!r}, .6) for k in (1, 2, 3)]
                   + [(f'ENTm{{k}}', {str(REGION_GREY[4])!r}, .6) for k in (5, 6)]
                   + [('HPF', {str(REGION_GREY[1])!r}, .25)]):
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
fig = plt.figure(figsize=(14.2, 10.0))
# row 1 is empty and holds only the legend: with the region key split by MEC
# layer the legend is too wide for the gap under the slices, whose x labels hang
# below their axes and collided with it wherever it was centred.
G = fig.add_gridspec(3, 1, height_ratios=[2.45, .16, 1], hspace=.38)
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

_handles = ([Patch(facecolor=str(REGION_GREY[k]), edgecolor='0.6', lw=.4, label=l)
              for k, l in zip((1, 2, 3, 4, 5), REGION_NAMES)]
            + [Patch(facecolor=str(GREY_OTHER), edgecolor='0.6', lw=.4,
                     label='other / unassigned')]
            + [Patch(facecolor=c, label=l) for l, c in
               (('grid cell', COL_GC), ('non-grid spatial', COL_NGS),
                ('other', COL_OTHER))]
            + [Patch(facecolor=COL_CONTACT, label='probe contacts')])
# Centred in the GAP between the first and second rows, not hung directly under
# row 0: the slice panels' x tick labels and axis title extend below their axes
# box, and anchoring to G[0].y0 printed the legend straight through them.
_y_mid = (G[1].get_position(fig).y0 + G[1].get_position(fig).y1) / 2
fig.legend(handles=_handles, loc='center', bbox_to_anchor=(.56, _y_mid),
           ncol=10, fontsize=6.6, frameon=False, handlelength=1.0,
           handleheight=1.1, columnspacing=1.3, borderpad=0)

# C-E: identity across the axes
g2 = G[2].subgridspec(1, 3, wspace=.42)
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

# ── H, I: two example sessions, each as slice + raster + PC1 + agreement ─────
# J and L report the mediolateral gradient in following as a correlation. These
# rows show the thing the correlation is computed on, in two sessions. Cells are
# ordered by mediolateral position rather than by PC1 loading, so if position
# organises following at all, the medial (left) end of each raster should track
# the population state more tightly than the lateral end.
#
# M25 D25 is kept for continuity with the rate maps in panel A, which are all
# M25; it is anchored on most trials, so its raster is mostly one colour. M26 D19
# was added because it has what M25 D25 lacks: balanced states (PC1 positive on
# half the trials), 13 sustained transitions, and four times the cells -- with
# the following concentrated on the medial shank (mean agreement 0.44 medial
# against 0.08 lateral, rho = -0.55). It was chosen by ranking every clean
# per-shank session (>= 3 shanks; M20/M21 positions carry per-cell jitter and
# were excluded) on sustained transitions, state balance and the within-session
# mediolateral rho; it is the second session of M26, the day after the Figure 1
# example. M28 D25 was added as a third because, with every region counted, its
# most medial shank looks unresponsive -- the region strip shows why.
#
# TRANSITIONS are where z-scored PC1 changes sign and holds the new sign for at
# least three trials, as in Figure 8A-D. They used to come from a 9-trial median
# filter on the fraction of cells called anchored, which is a different quantity
# from the PC1 strip the dashed lines are drawn across.
from scipy.ndimage import median_filter  # noqa: F401  (kept for other callers)
from spatial_manifolds.anchoring import (ANCH_COLOR as ANCH_C,
                                         NONANCH_COLOR as NONANCH_C,
                                         load_session_labels)

# Two per row. D is M29 D23, chosen to add a fourth animal (rank 8 of 27 by
# MEC-only rho, -0.19). The stronger within-MEC alternative was M25 D18 (all four
# shanks in MEC, rank 5). M29's other sessions are flat or reversed, which the
# caption says.
EXAMPLES = [('A', 25, 25), ('B', 26, 19), ('C', 28, 25), ('D', 29, 25)]
TA_ = ListedColormap([NONANCH_C, ANCH_C])
NORM_ = BoundaryNorm([-.5, .5, 1.5], TA_.N)
_rr_all = pd.read_csv(f'{PS}/pc1_by_region.csv')


def pc1_transitions(pc1, min_run=3):
    """z-scored PC1 and the trials (1-based) where it changes sign for >= min_run."""
    z = (pc1 - np.nanmean(pc1)) / np.nanstd(pc1)
    s = np.where(np.nan_to_num(z) >= 0, 1, -1)
    runs = []
    for k, v in enumerate(s):
        if runs and runs[-1][0] == v:
            runs[-1][2] = k
        else:
            runs.append([v, k, k])
    merged = []
    for v, a_, b_ in runs:
        if merged and (b_ - a_ + 1 < min_run or merged[-1][0] == v):
            merged[-1][2] = b_
        else:
            merged.append([v, a_, b_])
    return z, [m[1] + 1 for m in merged[1:]]


def example_row(cell, letter, mo, dy):
    # wspace is wider than the rows above because this one carries a colorbar
    # on the slice panel: the bar's tick labels and the raster's rotated y-label
    # both float outward by a fixed point padding that no gridspec setting
    # accounts for, so a gap that looks ample between the boxes collides in ink.
    # [raster | PC1] tight together on the left, the slice (+ boxplot strip) on
    # the right. The slice is drawn at equal aspect, so it is height-limited
    # and would float in the middle of its slot; it is anchored to the right
    # edge instead, which closes the gap that otherwise opens after PC1.
    gR = cell.subgridspec(1, 2, width_ratios=[2.55, 1.15], wspace=.30)
    gLft = gR[0].subgridspec(1, 2, width_ratios=[2.25, .30], wspace=.05)
    z = load_session_labels(mo, dy)
    ids = [int(c) for c in z['cluster_id']]
    cs = (C[(C.mouse == mo) & (C.day == dy)].dropna(subset=['coord_SCs_x'])
          .assign(ml=lambda d: d.coord_SCs_x.abs()).set_index('cluster_id'))
    # medial -> lateral by shank, then DORSAL -> VENTRAL within the shank
    # (coord_SCs_y grows ventrally: visual cortex sits near 2240, MEC 2550-2815),
    # so each shank's region strip reads like the shank itself
    # cells whose labels are NaN on every trial carry no anchoring call at all;
    # they are dropped from every panel of the row (raster, strip, slice, boxes)
    _all_nan = np.isnan(z['labels']).all(axis=1)
    nolab = {c for i, c in enumerate(ids) if _all_nan[i]}
    keep = sorted([(i, c) for i, c in enumerate(ids)
                   if c in cs.index and not _all_nan[i]],
                  key=lambda t: (cs.ml[t[1]], cs.coord_SCs_y[t[1]]))
    rows = [i for i, _ in keep]
    cid = [c for _, c in keep]
    mls = np.array([cs.ml[c] for c in cid])
    rcls = np.array([region_class(cs.brain_region[c]) for c in cid])
    L = z['labels'][rows]
    frac = np.asarray(z['frac_anch'], float)
    n_tr = L.shape[1]
    pcz, tr = pc1_transitions(np.asarray(z['pc1'], float))
    rr = _rr_all.query('mouse == @mo and day == @dy').set_index('cluster_id')

    # slice: where these cells sit, coloured by agreement ABOVE CHANCE -- the
    # same quantity as Figure 2D: |r| minus the mean |r| of the cell's own
    # circular-shift null, so absolute is compared with absolute (subtracting
    # an absolute null from a signed r is the error that retired two earlier
    # claims). Centred on chance, so cells below their own null show too.
    # the slice under a strip of per-shank boxplots that shares its mediolateral
    # axis, so each shank's distribution sits directly over the shank
    gSl = gR[1].subgridspec(2, 1, height_ratios=[.45, 1], hspace=.10)
    axbx = fig.add_subplot(gSl[0])
    axsl = fig.add_subplot(gSl[1])
    cells = (C[(C.mouse == mo) & (C.day == dy)]
             .dropna(subset=['coord_SCs_x', 'coord_SCs_y', 'coord_SCs_z']))
    cells = cells[~cells.cluster_id.isin(nolab)]
    cpts = np.array([stereo_to_ccf(r) for r in
                     cells[['coord_SCs_z', 'coord_SCs_y', 'coord_SCs_x']].values])
    fitH = fit_best_slice(get_mouse_contacts(mo), extent_pts=cpts)
    uv, vv, grid = sample_region_grid(fitH)
    axsl.imshow(grid, origin='lower', aspect='equal', cmap=region_cmap,
                norm=region_norm, zorder=1, extent=[uv[0], uv[-1], vv[0], vv[-1]])
    pu, pv = fitH['project'](cpts)
    # a little Gaussian jitter on the mediolateral coordinate, for display only:
    # every cell on a shank is assigned the shank's one nominal position, so
    # unjittered they stack into a single column. SD 15 um against shank
    # spacings of 210-270 um keeps the shanks distinct. Shank positions for
    # labels and boxes use the unjittered values.
    pu_true = pu.copy()
    pu = pu + np.random.default_rng(mo * 100 + dy).normal(0, ML_JITTER_FIG4, len(pu))
    ex = np.array([rr.excess.get(c, np.nan) for c in cells.cluster_id])
    okc = np.isfinite(ex)
    axsl.scatter(pu[~okc], pv[~okc], s=2.0, color='0.80', alpha=.5, lw=0, zorder=2)
    hi = max(abs(np.nanpercentile(ex, 2)), abs(np.nanpercentile(ex, 98)))
    # DIVERGING, white at chance. PRGn because the obvious alternatives each
    # collide with a colour this figure already uses: BrBG ends in the
    # non-anchored teal, PiYG in near-magenta, RdBu in the grid/non-grid red
    # and blue of B-D, and PuOr's orange vanishes on the gold MEC shading
    # where the above-chance cells sit.
    sc = axsl.scatter(pu[okc], pv[okc], c=ex[okc], s=8.0, cmap='PRGn',
                      norm=TwoSlopeNorm(vcenter=0.0, vmin=-hi, vmax=hi),
                      alpha=.95, lw=0, edgecolors='none', zorder=3)
    axsl.set_aspect('equal', adjustable='box')
    axsl.set_anchor('E')
    axsl.invert_yaxis()
    axsl.set_xlabel('medial–lateral (µm)', fontsize=7)
    axsl.set_ylabel('dorsal–ventral (µm)', fontsize=7)
    axsl.xaxis.set_major_locator(plt.MaxNLocator(2))
    axsl.yaxis.set_major_locator(plt.MaxNLocator(3))
    axsl.tick_params(labelsize=6)
    axsl.spines[['top', 'right']].set_visible(False)
    cb = fig.colorbar(sc, ax=axsl, fraction=.045, pad=.03)
    # label ABOVE the bar, not rotated beside it: the rotated label floats
    # outward by a fixed point padding and ran into the raster's y-label
    # loc='left' so the title grows rightwards, away from the S labels that
    # now sit over the most lateral shank
    cb.ax.tick_params(labelsize=5.5); cb.outline.set_visible(False)
    # shanks are numbered medial -> lateral (S1 most medial): the cell table
    # carries each cell's per-shank mediolateral position but not the device's
    # own shank id, so this is the order the raster beside it uses
    _cml = cells.coord_SCs_x.abs().round(0).values
    bx_vals, bx_pos = [], []
    for j_, u_ in enumerate(np.sort(np.unique(_cml))):
        m_ = _cml == u_
        top = np.argmin(pv[m_])                      # most dorsal cell
        xs_ = float(np.median(pu_true[m_]))
        axsl.annotate(f'S{j_ + 1}', (xs_, pv[m_][top]),
                      xytext=(0, 3), textcoords='offset points', ha='center',
                      va='bottom', fontsize=6.5, weight='bold', color='0.15',
                      annotation_clip=False, zorder=6)
        bx_vals.append(ex[m_ & okc]); bx_pos.append(xs_)

    # agreement above chance, one box per shank at the shank's own position
    xl = axsl.get_xlim()
    w_ = .28 * (np.min(np.diff(bx_pos)) if len(bx_pos) > 1 else 200.0)
    bp = axbx.boxplot(bx_vals, positions=bx_pos, widths=w_, showfliers=False,
                      patch_artist=True, manage_ticks=False,
                      medianprops=dict(color='k', lw=1.3),
                      boxprops=dict(facecolor='0.85', edgecolor='0.35', lw=.7),
                      whiskerprops=dict(color='0.35', lw=.7),
                      capprops=dict(color='0.35', lw=.7))
    axbx.axhline(0, color='0.5', lw=.7, ls=':', zorder=1)
    axbx.set_xlim(xl)
    axbx.set_xticks([])
    axbx.set_ylabel('above\nchance', fontsize=6, labelpad=1)
    axbx.tick_params(labelsize=5.5)
    axbx.spines[['top', 'right', 'bottom']].set_visible(False)
    # the slice is drawn at equal aspect, so its box shrinks inside its slot;
    # match the strip above to the slice's DRAWN width so the shanks line up
    axsl.apply_aspect()
    _ps, _pb = axsl.get_position(), axbx.get_position()
    axbx.set_position([_ps.x0, _pb.y0, _ps.width, _pb.height])

    # raster, one block per shank, each under a strip giving the cell's region
    shanks = np.unique(mls)
    counts = [int((mls == u).sum()) for u in shanks]
    HR = [.055, 1]
    gS = gLft[0].subgridspec(2, len(shanks), height_ratios=HR,
                           width_ratios=counts, wspace=.035, hspace=.025)
    strip_rgb = np.array([(GREY_OTHER,) * 3] +
                         [(REGION_GREY[k],) * 3 for k in (1, 2, 3, 4, 5)])
    for j, u in enumerate(shanks):
        m = mls == u
        axs = fig.add_subplot(gS[0, j])
        axs.imshow(strip_rgb[rcls[m]][None, :, :], aspect='auto',
                   interpolation='nearest')
        axs.set_xticks([]); axs.set_yticks([])
        for sp_ in axs.spines.values():
            sp_.set_visible(False)
        # a shank with very few cells is too narrow for its label, which then
        # runs into the next one; the S label below still identifies it
        if m.sum() >= .06 * len(mls):
            axs.set_title(f'{u:.0f} µm', fontsize=6, color='0.25', pad=2)
        if j == 0:
            axs.set_ylabel('region', fontsize=6, rotation=0, ha='right',
                           va='center', labelpad=2)
            axs.annotate(f'M{mo} D{dy}, {len(rows)} cells',
                         (0, 1), xycoords='axes fraction', xytext=(0, 13),
                         textcoords='offset points', fontsize=7.4, ha='left',
                         va='bottom')
        ax = fig.add_subplot(gS[1, j])
        ax.imshow(L[m].T, aspect='auto', cmap=TA_, norm=NORM_,
                  interpolation='nearest', extent=[0, m.sum(), n_tr, 1])
        for tb in tr:
            ax.axhline(tb, color='0.1', lw=.7, ls='--', zorder=4)
        ax.set_xticks([])
        ax.set_xlabel(f'S{j + 1}', fontsize=7, weight='bold', labelpad=2)
        if j == 0:
            ax.set_ylabel('Trial', fontsize=8)
            ax.tick_params(labelsize=7)
        else:
            ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_visible(False)

    # z-scored PC1, on the same rows as the raster (same height split)
    # the row letter, at the raster's top-left now that the raster leads
    _pos = gLft[0].get_position(fig)
    fig.text(_pos.x0 - .045, _pos.y1 + .012, letter, fontsize=10, weight='bold',
             va='bottom', ha='right')
    gP = gLft[1].subgridspec(2, 1, height_ratios=HR, hspace=.025)
    axp = fig.add_subplot(gP[1])
    pc = np.nan_to_num(pcz)
    y = np.arange(1, len(pc) + 1)
    axp.fill_betweenx(y, 0, pc, where=pc >= 0, color=ANCH_C, lw=0,
                      edgecolor='none', interpolate=True)
    axp.fill_betweenx(y, 0, pc, where=pc < 0, color=NONANCH_C, lw=0,
                      edgecolor='none', interpolate=True)
    axp.axvline(0, color='0.4', lw=.7)
    for tb in tr:
        axp.axhline(tb, color='0.1', lw=.8, ls='--', zorder=4)
    axp.set_ylim(n_tr, 1); axp.set_xticks([])
    axp.tick_params(labelleft=False, left=False)
    # labelled underneath: above, it collides with the last shank's µm label
    # now that PC1 sits tight against the raster
    axp.set_xlabel('PC1\n(z)', fontsize=7, labelpad=2)
    for sp_ in axp.spines.values():
        sp_.set_visible(False)

    # The per-session statistics the caption quotes. The panel that drew them
    # (agreement against position) was removed once the boxplot strip above
    # the slice showed the same per-shank distributions on the slice's own axis.
    rv = np.array([rr.r.get(c, np.nan) for c in cid])
    ok = np.isfinite(rv)
    per_shank = []
    for u in shanks:
        m = (mls == u) & ok
        if m.sum() >= 3:
            per_shank.append((u, rv[m].mean(), int(m.sum())))
    rho = spearmanr(mls[ok], rv[ok])
    mec_ = ok & np.isin(rcls, (3, 4))
    rho_m = spearmanr(mls[mec_], rv[mec_])
    print(f'  example {letter} M{mo}D{dy}: {len(rows)} cells in raster, '
          f'{int(ok.sum())} with agreement, {len(shanks)} shanks, '
          f'span {mls.max() - mls.min():.0f} um, {(frac > .5).mean() * 100:.0f}% '
          f'anchored, {len(tr)} PC1 transitions, rho={rho[0]:+.3f} p={rho[1]:.3g}; '
          'per-shank agreement medial->lateral: '
          + ', '.join(f'{v:.2f} (n={n})' for _, v, n in per_shank)
          + f' | MEC only: rho={rho_m[0]:+.3f} p={rho_m[1]:.3g}')
    print('     region classes by shank: ' + '; '.join(
        f'{u:.0f}:' + ','.join(f'{k}={int(((mls == u) & (rcls == k)).sum())}'
                               for k in range(6) if ((mls == u) & (rcls == k)).any())
        for u in shanks))


# ══ FIGURE 3 ENDS HERE. Everything below is the following figure (Figure 5):
# the mediolateral organisation of FOLLOWING, split out of what was Figure 3H-L (now Figure 4A-E).
out3 = f'{FIG}/fig3_anatomy.pdf'
plt.savefig(out3, dpi=200, bbox_inches='tight')
print(f'\nsaved {out3}')
plt.close(fig)

fig = plt.figure(figsize=(14.2, 16.2))
G = fig.add_gridspec(6, 1, height_ratios=[.13, 1.30, 1.30, 1, 1.05, 1.25], hspace=.42)
_rh = ([Patch(facecolor=str(REGION_GREY[k]), edgecolor='0.6', lw=.4, label=l)
        for k, l in zip((1, 2, 3, 4, 5), REGION_NAMES)]
       + [Patch(facecolor=str(GREY_OTHER), edgecolor='0.6', lw=.4,
                label='other / unassigned')])
_yl = (G[0].get_position(fig).y0 + G[0].get_position(fig).y1) / 2
fig.legend(handles=_rh, loc='center', bbox_to_anchor=(.5, _yl), ncol=6,
           fontsize=6.8, frameon=False, handlelength=1.1,
           title='region (slices, raster strips and points)', title_fontsize=6.8)
# E, F on one row; G (within-session summary) joins the controls H, I below
g3 = G[3].subgridspec(1, 2, wspace=.30)
foll = mec.dropna(subset=['r'])
N_ML_SESS = len(within_session(foll, 'ml', 'r')[0])
for (_letter, _mo, _dy), _cell in zip(EXAMPLES, [c for r_ in (G[1], G[2]) for c in r_.subgridspec(1, 2, wspace=.24)]):
    example_row(_cell, _letter, _mo, _dy)
# E: the pooled mediolateral gradient, as agreement ABOVE CHANCE (the quantity
# the example slices and boxplots show), not raw agreement with the axis
_EX = pd.read_csv(f'{PS}/pc1_by_region.csv')[['mouse', 'day', 'cluster_id', 'excess']]
follx = foll.merge(_EX, on=['mouse', 'day', 'cluster_id'], how='left')
ax = fig.add_subplot(g3[0])
x, y, e = binned(follx, 'ml', 'excess')
ax.fill_between(x, y - e, y + e, color='0.35', alpha=.28, lw=0, edgecolor='none')
ax.plot(x, y, color='k', lw=1.8)
ax.axhline(0, color='0.6', lw=.7, ls=':')
_rp = spearmanr(follx.dropna(subset=['excess']).ml, follx.dropna(subset=['excess']).excess)
_ws, _p = within_session(follx, 'ml', 'excess')
ax.set_xlabel('medial–lateral (µm)', fontsize=8)
ax.set_ylabel('agreement above chance', fontsize=8)
_lp(ax, 'E')
ax.set_title(f'MEC cells, all sessions: pooled rho = {_rp[0]:+.2f} (p = {_rp[1]:.0e})\n'
             f'WITHIN SESSION rho = {np.median(_ws):+.3f}, p = {_p:.3g} ({len(_ws)} sess.)',
             fontsize=6.6, loc='left')
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)
print(f'  E: pooled rho {_rp[0]:+.3f} p {_rp[1]:.2g}; within {np.median(_ws):+.3f} p {_p:.3g} n {len(_ws)}')

# F: distance from the MEC-parasubiculum border. Probe placement differs a lot
# between animals, which is why the pooled mediolateral coordinate is unsafe;
# distance from an anatomical landmark aligns every session to the same one.
# The border is taken from each animal's best-fit slice through the atlas: every
# parasubiculum pixel with an MEC neighbour is a border point, and a cell's
# distance is the SHORTEST distance from its projected position to any border
# point, signed by the cell's own atlas label (MEC +, parasubiculum -). Cells
# are at per-shank nominal mediolateral positions, so this is no finer than the
# shank pitch.
from scipy.spatial import cKDTree
CB = (C.merge(_EX, on=['mouse', 'day', 'cluster_id'], how='left')
      .dropna(subset=['coord_SCs_x', 'coord_SCs_y', 'coord_SCs_z']))
CB = CB[CB.brain_region.astype(str).str.match(r'^(ENTm|PAR|PRE)')].copy()
CB['dist'] = np.nan
for _mo, _g in CB.groupby('mouse'):
    try:
        _pts = np.array([stereo_to_ccf(r_) for r_ in
                         _g[['coord_SCs_z', 'coord_SCs_y', 'coord_SCs_x']].values])
        _fit = fit_best_slice(get_mouse_contacts(_mo), extent_pts=_pts)
        _uv, _vv, _grid = sample_region_grid(_fit)
    except Exception as _e:
        print(f'  border: M{_mo} skipped ({type(_e).__name__})'); continue
    _uv, _vv = np.asarray(_uv), np.asarray(_vv)
    assert _grid.shape == (len(_vv), len(_uv)), (_grid.shape, len(_vv), len(_uv))
    _par, _mc = _grid == 2, np.isin(_grid, (3, 4))
    _nb = np.zeros_like(_par)
    _nb[1:, :] |= _mc[:-1, :]; _nb[:-1, :] |= _mc[1:, :]
    _nb[:, 1:] |= _mc[:, :-1]; _nb[:, :-1] |= _mc[:, 1:]
    _bi, _bj = np.where(_par & _nb)
    # fewer than ~20 border pixels means the slice only grazes the boundary
    # (M20 gives 4), which is too little to measure distance from
    if len(_bi) < 20:
        print(f'  border: M{_mo} has too little MEC-parasubiculum border in its slice '
              f'({len(_bi)} points), skipped'); continue
    _tree = cKDTree(np.c_[_uv[_bj], _vv[_bi]])
    _pu, _pv = _fit['project'](_pts)
    _d, _ = _tree.query(np.c_[_pu, _pv])
    _sg = np.where(_g.brain_region.astype(str).str.startswith('ENTm'), 1.0, -1.0)
    CB.loc[_g.index, 'dist'] = _sg * _d
    print(f'  border: M{_mo} {len(_bi)} border points, {len(_g)} cells')
CBv = CB.dropna(subset=['dist', 'excess'])
ax = fig.add_subplot(g3[1])
for side, col in ((CBv.dist < 0, '0.55'), (CBv.dist >= 0, 'k')):
    x, y, e = binned(CBv[side], 'dist', 'excess', nbin=6 if col == '0.55' else 8)
    ax.fill_between(x, y - e, y + e, color=col, alpha=.25, lw=0, edgecolor='none')
    ax.plot(x, y, color=col, lw=1.8)
ax.axvline(0, color='0.3', lw=.8, ls='--')
ax.axhline(0, color='0.6', lw=.7, ls=':')
ax.annotate('parasubiculum', (.02, .97), xycoords='axes fraction', fontsize=6.5,
            color='0.45', va='top')
ax.annotate('MEC', (.98, .97), xycoords='axes fraction', fontsize=6.5, ha='right', va='top')
_m = CBv[CBv.dist >= 0]
_rb = spearmanr(_m.dist, _m.excess)
_wb, _pb = within_session(_m.assign(ml=_m.dist), 'ml', 'excess')
_near = CBv[CBv.dist.abs() <= 250]
_pn = mannwhitneyu(_near[_near.dist < 0].excess, _near[_near.dist >= 0].excess).pvalue
ax.set_xlabel('distance from the MEC–parasubiculum border (µm)', fontsize=8)
ax.set_ylabel('agreement above chance', fontsize=8)
_lp(ax, 'F')
ax.set_title(f'MEC side, pooled: rho = {_rb[0]:+.2f} (p = {_rb[1]:.0e}); within session '
             f'{np.median(_wb):+.3f}, p = {_pb:.3g}\nwithin 250 µm of the border: '
             f'parasubiculum {_near[_near.dist < 0].excess.mean():.3f} vs MEC '
             f'{_near[_near.dist >= 0].excess.mean():.3f} (p = {_pn:.2g})',
             fontsize=6.4, loc='left')
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)
print(f'  F: MEC side pooled rho {_rb[0]:+.3f} p {_rb[1]:.2g}; within {np.median(_wb):+.3f} '
      f'p {_pb:.3g} n {len(_wb)}; near border PAR {_near[_near.dist < 0].excess.mean():.3f} '
      f'(n={int((_near.dist < 0).sum())}) MEC {_near[_near.dist >= 0].excess.mean():.3f} '
      f'(n={int((_near.dist >= 0).sum())}) p {_pn:.3g}; cells with distance: {len(CBv)}')

# G: the within-session distributions side by side, which is the actual test
gC = G[4].subgridspec(1, 3, wspace=.48, width_ratios=[1.1, .8, 1.25])
ax = fig.add_subplot(gC[0])
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
_lp(ax, 'G')
ax.set_title(f'only M–L survives\n(M–L: {len(within_session(foll, "ml", "r")[0])} '
             f'multi-shank sessions)', fontsize=7, loc='left')
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)


# ── H-K: controls and extensions, computed by fig4_controls.py ────────────────
import json as _json
CTL = pd.read_csv(f'{PS}/fig4_controls_cells.csv')
SUMM_ = _json.load(open(f'{PS}/fig4_controls_summary.json'))
CM = CTL[CTL.mec]


def _ws(df, y, min_cells=20):
    out = {}
    for k, g in df.dropna(subset=['ml', y]).groupby(['mouse', 'day']):
        if len(g) >= min_cells and g.ml.nunique() > 1 and g[y].nunique() > 1:
            out[k] = spearmanr(g.ml, g[y])[0]
    return pd.Series(out)


# H: the gradient against an axis built without the cell's own shank
ax = fig.add_subplot(gC[1]); _lp(ax, 'H')
a_, b_ = _ws(CM, 'r'), _ws(CM, 'r_lso')
k_ = a_.index.intersection(b_.index)
for kk in k_:
    ax.plot([0, 1], [a_[kk], b_[kk]], color='0.75', lw=.6, zorder=1)
for i, v in enumerate((a_[k_], b_[k_])):
    ax.scatter(np.full(len(v), i), v, s=9, color='0.35', lw=0, zorder=2)
    ax.plot([i - .22, i + .22], [np.median(v)] * 2, color='k', lw=2, zorder=3)
    pv_ = wilcoxon(v).pvalue
    ax.annotate(f'p = {pv_:.3f}', (i, .98), xycoords=('data', 'axes fraction'),
                ha='center', va='top', fontsize=6.2, color='#a33' if pv_ < .05 else '0.35')
ax.axhline(0, color='0.6', lw=.8, ls=':')
ax.set_xticks([0, 1]); ax.set_xticklabels(['all\nshanks', 'other\nshanks'], fontsize=7)
ax.set_xlim(-.5, 1.5)
ax.set_ylabel('within-session rho\n(position vs agreement)', fontsize=8)
ax.set_title('PC1 built without the\ncell\'s own shank', fontsize=7, loc='left')
ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)

# I: how much of the position effect survives the factors it covaries with
# (fig4_factors.py). Lateral cells are less theta-modulated and carry less
# spatial information, so part of the gradient is shared with those.
FAC = _json.load(open(f'{PS}/fig4_factors.json'))
ax = fig.add_subplot(gC[2]); _lp(ax, 'I')
_st = list(FAC['ml_steps'].items())
for i, (nm, v) in enumerate(_st):
    ax.errorbar(v['b'], i, xerr=1.96 * v['se'], fmt='o', color='k' if v['p'] < .05 else '0.55',
                ms=4, lw=1, capsize=0)
    ax.annotate(f'p = {v["p"]:.2g}', (1.0, i), xycoords=('axes fraction', 'data'),
                ha='right', va='center', fontsize=6.2, color='#a33' if v['p'] < .05 else '0.35')
ax.axvline(0, color='0.6', lw=.8, ls=':')
ax.set_yticks(range(len(_st))); ax.set_yticklabels([k for k, _ in _st], fontsize=6.5)
ax.invert_yaxis()
ax.set_xlabel('mediolateral coefficient\n(per SD lateral, 95% CI)', fontsize=7.5)
ax.set_title(f"position in a mixed model, {FAC['n_cells']} MEC cells on shanks\n"
             f"nested in {FAC['n_sess']} sessions, as more factors are added", fontsize=7, loc='left')
ax.tick_params(labelsize=6.5); ax.spines[['top', 'right']].set_visible(False)
_xl = ax.get_xlim(); ax.set_xlim(_xl[0], _xl[1] + .55 * (_xl[1] - _xl[0]))

# J, K: every factor in one model (fig4_factors.py)
gD = G[5].subgridspec(1, 2, wspace=.62, width_ratios=[1.0, 1.25])
ax = fig.add_subplot(gD[0]); _lp(ax, 'J', dx=-.42)
_lab = {r['factor']: r['label'] for r in FAC['factors']}
_grp = {r['factor']: r['group'] for r in FAC['factors']}
_ph = {r['factor']: r['p_holm'] for r in FAC['factors']}
_TERMS = [('ml_z', 'ml', 'mediolateral'), ('dv_z', 'dv', 'dorsoventral'),
          ('sup', 'layer', 'superficial (vs deep)'),
          ('C(identity, Treatment("non-grid spatial"))[T.grid]', 'type', 'grid (vs non-grid spatial)'),
          ('C(identity, Treatment("non-grid spatial"))[T.non-spatial]', 'type', 'non-spatial'),
          ('C(identity, Treatment("non-grid spatial"))[T.putative interneuron]', 'type', 'interneuron'),
          ('rate_z', 'rate', 'firing rate (log)'), ('wave_z', 'wave', 'waveform width'),
          ('si_z', 'si', 'spatial information'), ('th_z', 'theta', 'theta modulation'),
          ('gs_z', 'grid', 'grid score'), ('bs_z', 'border', 'border score'),
          ('hd_z', 'hd', 'head-direction tuning'), ('spd_z', 'speed', 'speed tuning |r|')]
_y, _ticks, _prev = 0, [], None
for t, f_, lab in _TERMS:
    if _prev is not None and _grp[f_] != _prev:
        _y += .6
    c = FAC['coef'][t]
    col = 'k' if _ph[f_] < .05 else '0.55'
    ax.errorbar(c['b'], _y, xerr=1.96 * c['se'], fmt='o', color=col, ms=3.8, lw=1, capsize=0)
    _ticks.append((_y, lab)); _prev = _grp[f_]; _y += 1
ax.axvline(0, color='0.6', lw=.8, ls=':')
ax.set_yticks([a for a, _ in _ticks]); ax.set_yticklabels([b for _, b in _ticks], fontsize=6.5)
ax.invert_yaxis()
_g0 = {}
for (yy, _), (t, f_, _) in zip(_ticks, _TERMS):
    _g0.setdefault(_grp[f_], []).append(yy)
for g_, ys in _g0.items():
    ax.annotate(g_, (1.02, np.mean(ys)), xycoords=('axes fraction', 'data'), fontsize=6.5,
                color='0.35', va='center', rotation=-90)
ax.set_xlabel('coefficient on agreement above chance\n(per SD, or vs reference; 95% CI)', fontsize=7.5)
ax.set_title('all factors in one model; black = factor significant\n(likelihood ratio, Holm)',
             fontsize=7, loc='left')
ax.tick_params(labelsize=6.5); ax.spines[['top', 'right']].set_visible(False)

ax = fig.add_subplot(gD[1]); _lp(ax, 'K', dx=-.36)
# Ranked by dominance: each factor's added R^2 averaged over every subset of
# the others, so variance that overlapping factors share (interneurons are
# classed on rate and waveform, grid cells on grid score) is divided between
# them rather than dropped, as a drop-one share would.
_F = pd.DataFrame(FAC['factors']).sort_values('dominance', ascending=False).reset_index(drop=True)
_tot = FAC['r2_within']
_yy = np.arange(len(_F))
ax.barh(_yy, 100 * _F.dominance / _tot, height=.62, color='0.3', lw=0)
for i, r_ in _F.iterrows():
    ax.annotate(('p < 1e-4' if r_.p_holm < 1e-4 else f'p = {r_.p_holm:.2g}'),
                (100 * r_.dominance / _tot, i), xytext=(4, 0), textcoords='offset points',
                va='center', fontsize=6, color='#a33' if r_.p_holm < .05 else '0.45')
ax.set_yticks(_yy); ax.set_yticklabels(_F.label, fontsize=6.5); ax.invert_yaxis()
ax.set_xlabel('share of explained variance (%, dominance analysis)', fontsize=7.5)
_r2 = FAC['r2']
ax.set_title(f"which factor matters (all {len(_F)}: {100 * _tot:.1f}% of within-session variance)",
             fontsize=7, loc='left')
ax.text(.98, .04, 'p: unique effect\n(likelihood ratio, Holm)', transform=ax.transAxes,
        ha='right', va='bottom', fontsize=6, color='0.4')
ax.set_xlim(0, 100 * _F.dominance.max() / _tot * 1.3)
ax.tick_params(labelsize=6.5); ax.spines[['top', 'right']].set_visible(False)

# Switch timing against position and agreement by region (former J, K) were
# removed from the figure; both are still computed by fig4_controls.py.

out = f'{FIG}/fig4_ml_following.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'\nsaved {out}')
