"""Best-fit recording slice through the atlas: the shared implementation.

The same slice geometry is wanted by more than one figure (Figure 3's anatomy
panels, Figure 6's region examples), and it is not the kind of thing to have two
copies of -- the CCF transform, the plane fit and the region shading all have to
agree or the figures disagree about where a cell is. This module holds it once.

THE PLANE IS FITTED TO DEVICE CONTACTS, NOT TO CELL LOCATIONS. Per-cell
mediolateral position is a per-shank nominal value plus synthetic jitter, so
fitting a plane to cell positions fits the approximation rather than the probe
geometry that produced it. The contacts are the geometry.

The 4.8 GB annotation volume is memory-mapped and loaded ONLY when a region grid
is actually requested, so importing this module is cheap. Figure 6's examples
need the slice outline and the cells; nothing forces them to pay for the atlas
unless they ask to shade it.

NOTE: fig3_anatomy.py still carries its own copy of these functions. It predates
this module and works; it should be switched over, but not silently and not in
the same change as a figure rebuild. Until then, any edit here must be mirrored
there -- see the open items in PAPER.md.
"""
import math
import re

import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

ROOT = '/Users/harryclark/Documents/spatial-manifolds'
BG = '/Users/harryclark/.brainglobe/allen_mouse_10um_v1.2'
DC_PATH = '/Users/harryclark/Downloads/device_contact_id_annotations.csv'

COL_HIPPO, COL_PREPARA = '#9C6A38', '#ABDBD2'
COL_MEC, COL_VIS = '#2E7D6C', '#8C6BB1'
COL_CONTACT = '#ff6500'

_REGION_RE = {1: re.compile(r'^(CA[1-4]?|DG)(-|$)'),
              2: re.compile(r'^(PRE|PAR)\d*$'),
              3: re.compile(r'^ENTm'),
              4: re.compile(r'^VIS')}

_cmap_list = [(1, 1, 1, 0)]
for _hex in (COL_HIPPO, COL_PREPARA, COL_MEC, COL_VIS):
    _cmap_list.append(tuple(int(_hex.lstrip('#')[i:i + 2], 16) / 255
                            for i in (0, 2, 4)) + (.55,))
region_cmap = ListedColormap(_cmap_list)
region_norm = BoundaryNorm(np.arange(len(_cmap_list) + 1) - .5, region_cmap.N)

# A NEUTRAL alternative, for figures where the CELL colours carry the meaning.
# The palette above gives MEC a green and visual cortex a purple; a figure that
# colours its cells by structure on a different key would then be saying "green"
# and "purple" twice with opposite referents, which is worse than no shading at
# all. These greys say where the tissue is and nothing else.
_neutral = [(1, 1, 1, 0)] + [(0, 0, 0, a) for a in (.07, .13, .22, .10)]
region_cmap_neutral = ListedColormap(_neutral)
region_norm_neutral = BoundaryNorm(np.arange(len(_neutral) + 1) - .5,
                                   region_cmap_neutral.N)
REGION_LABEL = {1: 'hippocampus', 2: 'PRE/PAR', 3: 'MEC', 4: 'VIS'}

_ann = None
_id_to_acr = None


def _atlas():
    """Memory-map the annotation volume on first use.

    memmap rather than imread: the volume is 4.8 GB and the slice sampling
    touches a few hundred thousand scattered voxels, so reading it whole is
    pure cost. imread is the fallback if the file is not a plain strip TIFF.
    """
    global _ann, _id_to_acr
    if _ann is None:
        import tifffile
        try:
            _ann = tifffile.memmap(f'{BG}/annotation.tiff', mode='r')
        except (ValueError, OSError):
            _ann = tifffile.imread(f'{BG}/annotation.tiff')
        s = pd.read_csv(f'{BG}/structures.csv')
        _id_to_acr = dict(zip(s['id'], s['acronym']))
    return _ann, _id_to_acr


def stereo_to_ccf(SC, angle=-0.0873):
    """Stereotaxic (z, y, x) to CCF, matching grid_cell_anatomy_v2."""
    SC = np.asarray(SC, float)
    stretch = SC / np.array([1.0, 0.9434, 1.0])
    rotate = np.array([stretch[0] * math.cos(angle) - stretch[1] * math.sin(angle),
                       stretch[0] * math.sin(angle) + stretch[1] * math.cos(angle),
                       stretch[2]])
    return rotate + np.array([5400.0, 440.0, 5700.0])


def to_ccf(d, cols=('SC_z', 'SC_y', 'SC_x')):
    """Rows of a frame to CCF coordinates."""
    if not len(d):
        return np.empty((0, 3))
    return np.array([stereo_to_ccf(r) for r in d[list(cols)].values])


def contacts_ccf(mouse=None):
    """Device contacts in CCF, for one mouse or all."""
    dc = pd.read_csv(DC_PATH)
    dc = dc[dc['mouse'] != 22]
    if mouse is not None:
        dc = dc[dc['mouse'] == mouse]
    return to_ccf(dc, ('coord_SCs_z', 'coord_SCs_y', 'coord_SCs_x'))


def fit_best_slice(fit_pts, extent_pts=None):
    """AP-DV-tilt-constrained best fit; the ML axis stays fixed in-plane.

    Returns u_lim/v_lim, the tilt, and project/reconstruct maps between CCF and
    the slice's own (u, v) coordinates. u is depth below the fixed ML reference
    plane; v runs along the fitted in-plane direction.
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


def ids_to_numeric(aid_flat):
    """Atlas ids to the five shading classes (0 = other)."""
    _, id_to_acr = _atlas()
    numeric = np.zeros(len(aid_flat), dtype=int)
    for sid in np.unique(aid_flat):
        if sid == 0:
            continue
        acr = id_to_acr.get(sid, 'root')
        for k, rx in _REGION_RE.items():
            if rx.match(acr):
                numeric[aid_flat == sid] = k
                break
    return numeric


def sample_region_grid(fit, step=10):
    """Shading classes on the fitted plane, as (u_vals, v_vals, grid)."""
    ann, _ = _atlas()
    u_vals = np.arange(*fit['u_lim'], step)
    v_vals = np.arange(*fit['v_lim'], step)
    V, U = np.meshgrid(v_vals, u_vals, indexing='ij')
    g = fit['reconstruct'](U, V)
    idx = [np.round(g[..., k] / 10).astype(int) for k in range(3)]
    ok = np.ones(idx[0].shape, bool)
    for k in range(3):
        ok &= (idx[k] >= 0) & (idx[k] < ann.shape[k])
    aid = np.zeros(idx[0].shape, int)
    aid[ok] = np.asarray(ann[idx[0][ok], idx[1][ok], idx[2][ok]])
    return u_vals, v_vals, ids_to_numeric(aid.ravel()).reshape(idx[0].shape)
