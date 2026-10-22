"""Grid module identification from spatial autocorrelogram structure.

Single source of truth for the module-identification rules described in the
Methods (`scripts/figures/AnchorDynamics2026/PAPER.md`, "Grid module
identification"). Module labels are deliberately NOT stored in the
classification datasheet: they depend on which cells are classified as grid
cells, which changes whenever the shuffle count changes. Recompute them from
this module instead, so the rules and the labels can never drift apart.

Everything here is per session. Modules are defined over simultaneously
recorded cells; a cluster pooled across sessions would at best be a spacing
band shared by different animals.

Typical use:

    from spatial_manifolds.grid_modules import (
        sac_fft, sac_features, identify_session_modules, MODULE_PARAMS)

    lab, info = identify_session_modules(sac_matrix, cell_class, spacing, orientation)
"""
from dataclasses import dataclass, asdict

import numpy as np
from scipy.signal import fftconvolve
from skimage.feature import peak_local_max
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class ModuleParams:
    """Validated defaults. See Methods for the sweep that selected them."""
    umap_metric: str = 'manhattan'
    umap_min_dist: float = 0.05
    umap_init: str = 'spectral'
    random_state: int = 42
    db_min_samples: int = 3        # swept 3-6
    min_gc_cluster: int = 5
    min_enrich: float = 2.0        # swept 1.5-2.5; below 2.0 reproducibility degrades
    max_spacing_iqr_cm: float = 10.0
    max_orient_sd_deg: float = 10.0
    min_cells_session: int = 60
    min_gc_session: int = 12


MODULE_PARAMS = ModuleParams()


# ── autocorrelogram geometry ────────────────────────────────────────────────
def sac_fft(rmap, min_overlap=3):
    """2-D spatial autocorrelogram by FFT, with low-overlap/zero-variance guards."""
    a = np.nan_to_num(np.asarray(rmap, float)); ones = np.ones_like(a)
    f = lambda b, x: fftconvolve(x, np.rot90(b, 2), mode='full')
    s1s2 = f(a, a); s1 = f(a, ones); s2 = f(ones, a)
    s1sq = f(a ** 2, ones); s2sq = f(ones, a ** 2)
    n = np.maximum(f(ones, ones), 1e-12); nsq = n ** 2
    v1 = np.maximum(s1sq / n - (s1 ** 2) / nsq, 0.0)
    v2 = np.maximum(s2sq / n - (s2 ** 2) / nsq, 0.0)
    denom = np.sqrt(v1 * v2); cov = s1s2 / n - (s1 * s2) / nsq
    out = np.zeros_like(cov)
    ok = (denom > 1e-12) & (np.round(f(ones, ones)) >= min_overlap)
    out[ok] = cov[ok] / denom[ok]
    return np.clip(out, -1.0, 1.0)


def _peaks(sac, centre, min_distance=2, n=6):
    pk = peak_local_max(np.nan_to_num(sac), min_distance=min_distance, exclude_border=False)
    if len(pk) < 2:
        return None
    dy, dx = pk[:, 0] - centre, pk[:, 1] - centre
    d = np.hypot(dy, dx); keep = d > min_distance
    dy, dx, d = dy[keep], dx[keep], d[keep]
    if not len(d):
        return None
    o = np.argsort(d)[:n]
    return dy[o], dx[o], d[o]


def sac_spacing(sac, centre, bin_cm, min_distance=2):
    """Median centre-to-peak distance over the 6 nearest peaks, in cm."""
    pk = _peaks(sac, centre, min_distance)
    return float(np.median(pk[2]) * bin_cm) if pk is not None else np.nan


def sac_orientation(sac, centre, min_distance=2):
    """Grid orientation in degrees, modulo 60.

    A hexagonal lattice is invariant under 60-degree rotation, so peak angles
    are wrapped to [0, 60) and averaged on a circle: mapped x6 onto the full
    circle so the circular mean is well defined, then mapped back. Never treat
    this as a linear variable -- 59 and 1 degrees are 2 degrees apart, not 58.
    """
    pk = _peaks(sac, centre, min_distance)
    if pk is None:
        return np.nan
    dy, dx, _ = pk
    a = np.radians((np.degrees(np.arctan2(dy, dx)) % 60.0) * 6.0)
    return float((np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360.0) / 6.0)


def circ_spread_60(vals):
    """Circular SD (degrees) for orientations on the 60-degree circle."""
    v = np.asarray(vals, float); v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.nan
    a = np.radians(v * 6.0)
    R = min(max(np.hypot(np.sin(a).mean(), np.cos(a).mean()), 1e-12), 1.0)
    return float(np.degrees(np.sqrt(-2 * np.log(R))) / 6.0)


def sac_vector(sac, disc_mask):
    """Central disc of the SAC, flattened and z-scored: shape, not magnitude."""
    v = sac[disc_mask].astype(float)
    s = v.std()
    return (v - v.mean()) / s if s > 1e-9 else v - v.mean()


# ── clustering ──────────────────────────────────────────────────────────────
def knee_eps(emb, k):
    """DBSCAN eps at the k-distance knee.

    The point of the sorted k-nearest-neighbour distance curve furthest from
    the chord joining its endpoints -- where within-cluster spacing gives way
    to between-cluster gaps. Preferred over a silhouette sweep, which rewards
    many tight clusters and collapses to the smallest radius offered.
    """
    k = int(min(k, len(emb) - 1))
    d, _ = NearestNeighbors(n_neighbors=k).fit(emb).kneighbors(emb)
    kd = np.sort(d[:, -1]); x = np.arange(len(kd), dtype=float)
    x1, y1, x2, y2 = 0.0, kd[0], float(len(kd) - 1), kd[-1]
    num = np.abs((y2 - y1) * x - (x2 - x1) * kd + x2 * y1 - y2 * x1)
    return float(kd[np.argmax(num / np.hypot(y2 - y1, x2 - x1))])


def identify_session_modules(embedding, cell_class, spacing_cm, orientation_deg,
                             params=MODULE_PARAMS):
    """Assign module labels to one session's cells.

    Parameters
    ----------
    embedding : (n_cells, 2) UMAP embedding of the session's SAC vectors.
    cell_class : (n_cells,) 'GC' / 'NGS' / 'NS'.
    spacing_cm, orientation_deg : (n_cells,) per-cell SAC geometry.

    Returns
    -------
    module : (n_cells,) float array; module index for grid cells in a qualifying
        cluster, NaN otherwise. NaN means "grid cell with no module assignment",
        not "not a grid cell" -- coverage is reported, never maximised.
    info : dict with the DBSCAN labels, chosen eps, and a per-cluster record of
        enrichment and coherence including clusters that were rejected.
    """
    cell_class = np.asarray(cell_class)
    spacing_cm = np.asarray(spacing_cm, float)
    orientation_deg = np.asarray(orientation_deg, float)

    eps = knee_eps(embedding, params.db_min_samples)
    lab = DBSCAN(eps=eps, min_samples=params.db_min_samples).fit_predict(embedding)

    module = np.full(len(cell_class), np.nan)
    base = max((cell_class == 'GC').mean(), 1e-9)
    clusters, mid = [], 0
    for cl in sorted(c for c in set(lab) if c >= 0):
        m = lab == cl
        gc = m & (cell_class == 'GC') & np.isfinite(spacing_cm) & np.isfinite(orientation_deg)
        enrich = (cell_class[m] == 'GC').mean() / base
        sp, orr = spacing_cm[gc], orientation_deg[gc]
        iqr = (np.percentile(sp, 75) - np.percentile(sp, 25)) if gc.sum() > 1 else np.nan
        osd = circ_spread_60(orr) if gc.sum() > 1 else np.nan
        # Enrichment alone is not enough: a cluster can be grid-dominated while
        # mixing several modules. Coherence is what makes it a module.
        ok = (gc.sum() >= params.min_gc_cluster and enrich >= params.min_enrich
              and np.isfinite(iqr) and iqr <= params.max_spacing_iqr_cm
              and np.isfinite(osd) and osd <= params.max_orient_sd_deg)
        if ok:
            module[gc] = mid
        clusters.append(dict(cluster=int(cl), n=int(m.sum()), n_gc=int(gc.sum()),
                             enrichment=float(enrich), spacing_iqr=float(iqr) if np.isfinite(iqr) else np.nan,
                             orient_sd=float(osd) if np.isfinite(osd) else np.nan,
                             spacing=float(np.median(sp)) if gc.sum() else np.nan,
                             orientation=float(np.median(orr)) if gc.sum() else np.nan,
                             is_module=bool(ok), module=mid if ok else None))
        mid += int(ok)
    return module, dict(dbscan_labels=lab, eps=eps, clusters=clusters,
                        n_modules=mid, params=asdict(params))
