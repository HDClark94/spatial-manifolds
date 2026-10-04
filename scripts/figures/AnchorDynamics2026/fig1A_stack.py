#!/usr/bin/env python
"""Figure 1A - "A cognitive map is composed from feature representations,
then registered to the world."

Builds two variants of the same schematic:

    fig1A_stack.pdf     open arena (2D)
    fig1A_stack_1D.pdf  virtual linear track (1D, 200 cm task)

Both panels are *schematics*: the rate maps are analytic idealisations of
each tuning type, not data. Rate maps are rendered in viridis.

Run:  python fig1A_stack.py
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import Bbox

HERE = __file__.rsplit("/", 1)[0]

RATE_CMAP = plt.get_cmap("viridis")

# feature colours, shared between the world objects and the composed map
COL = {
    "objA": "#a83232",    # cylinder
    "objB": "#6b6bab",    # cone
    "cueA": "#bb63bb",    # magenta cue bar
    "cueB": "#15a06e",    # green cue bar
    "border": "#e08a17",  # arena wall / track boundary
    "reward": "#a83232",  # reward location (1D)
}
PALE = np.array([0.93, 0.93, 0.93])
EDGE = "#555555"
LABEL = "#333333"
ARROW = "#999999"

# ---------------------------------------------------------------- arena layout
OBJ_A = (0.17, 0.79)                      # cylinder
OBJ_B = (0.82, 0.60)                      # cone
CUE_A = ((0.31, 0.85), (0.47, 0.90))      # magenta bar
CUE_B = ((0.63, 0.11), (0.79, 0.17))      # green bar
ANIMAL = (0.47, 0.52)
OV_VEC = (0.15, -0.14)                    # object-vector cell offset


# =============================================================== 2D rate maps
def _grid_map(X, Y, spacing=0.29, orient=0.22, phase=(0.13, 0.07)):
    k = 4 * np.pi / (np.sqrt(3) * spacing)
    g = np.zeros_like(X)
    for i in range(3):
        a = orient + i * np.pi / 3
        g += np.cos(k * (np.cos(a) * (X - phase[0]) + np.sin(a) * (Y - phase[1])))
    return np.clip((g + 1.5) / 4.5, 0, 1) ** 1.7


def _seg_dist(X, Y, p0, p1):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    t = np.clip(((X - p0[0]) * d[0] + (Y - p0[1]) * d[1]) / (d @ d), 0, 1)
    return np.hypot(X - (p0[0] + t * d[0]), Y - (p0[1] + t * d[1]))


def _bump(X, Y, c, s):
    return np.exp(-((X - c[0]) ** 2 + (Y - c[1]) ** 2) / (2 * s ** 2))


def _cue_map(X, Y, s=0.075):
    a = np.exp(-_seg_dist(X, Y, *CUE_A) ** 2 / (2 * s ** 2))
    b = np.exp(-_seg_dist(X, Y, *CUE_B) ** 2 / (2 * s ** 2))
    return np.maximum(a, b)


def _ov_map(X, Y, s=0.095):
    a = _bump(X, Y, (OBJ_A[0] + OV_VEC[0], OBJ_A[1] + OV_VEC[1]), s)
    b = _bump(X, Y, (OBJ_B[0] + OV_VEC[0], OBJ_B[1] + OV_VEC[1]), s)
    return np.maximum(a, b)


def _border_map(X, Y, s=0.055):
    """A single border cell: one long wall plus a stretch of the adjacent one."""
    d = np.minimum(Y, np.where(Y < 0.55, X, np.inf))
    return np.exp(-d ** 2 / (2 * s ** 2))


def _hetero_map(X, Y, seed=7):
    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.normal(size=X.shape), sigma=X.shape[0] / 11.0, mode="wrap")
    f = (f - f.min()) / np.ptp(f)
    return f ** 1.6


LAYERS_2D = [
    ("grid cells", _grid_map),
    ("cue cells", _cue_map),
    ("object-vector cells", _ov_map),
    ("border cells", _border_map),
    ("heterogeneous spatial cells", _hetero_map),
]


def _composed_2d(X, Y):
    """Height field plus per-feature weights for the composed map surface."""
    w = {
        "objA": 1.00 * _bump(X, Y, OBJ_A, 0.100),
        "objB": 0.94 * _bump(X, Y, OBJ_B, 0.100),
        "cueA": 0.86 * np.exp(-_seg_dist(X, Y, *CUE_A) ** 2 / (2 * 0.058 ** 2)),
        "cueB": 0.84 * np.exp(-_seg_dist(X, Y, *CUE_B) ** 2 / (2 * 0.058 ** 2)),
    }
    dwall = np.minimum(np.minimum(X, 1 - X), np.minimum(Y, 1 - Y))
    w["border"] = 0.52 * np.exp(-dwall ** 2 / (2 * 0.050 ** 2))
    return w


def _blend(w, ref=0.52):
    """Weighted feature colours over a pale base."""
    keys = list(w)
    tot = sum(w[k] for k in keys)
    num = sum(w[k][..., None] * np.array(to_rgb(COL[k])) for k in keys)
    hue = num / np.maximum(tot, 1e-9)[..., None]
    a = np.clip(tot / ref, 0, 1)[..., None]
    return PALE * (1 - a) + hue * a


# ========================================================= projection helpers
def _disp(ax, p):
    """3D data point -> display pixels."""
    x2, y2, _ = proj3d.proj_transform(p[0], p[1], p[2], ax.get_proj())
    return np.asarray(ax.transData.transform((x2, y2)), float)


def _corners(x0, x1, y0, y1, z):
    return [(x0, y0, z), (x1, y0, z), (x0, y1, z), (x1, y1, z)]


def _side_label(fig, ax, pts, text, side="left", pad=14, **kw):
    """Right-align text just outside the projected extent of `pts`."""
    d = np.array([_disp(ax, p) for p in pts])
    y = d[:, 1].mean()
    if side == "left":
        x, ha = d[:, 0].min() - pad, "right"
    else:
        x, ha = d[:, 0].max() + pad, "left"
    fx, fy = fig.transFigure.inverted().transform((x, y))
    return fig.text(fx, fy, text, ha=ha, va="center", **kw)


def _side_arrow(fig, ax, x_ref_pts, p0, p1, text, pad=22, fs=8):
    """Vertical arrow to the right of the content, between two 3D z levels."""
    d = np.array([_disp(ax, p) for p in x_ref_pts])
    x = d[:, 0].max() + pad
    y0 = _disp(ax, p0)[1]
    y1 = _disp(ax, p1)[1]
    inv = fig.transFigure.inverted()
    fx, fy0 = inv.transform((x, y0))
    _, fy1 = inv.transform((x, y1))
    fig.patches.append(FancyArrowPatch(
        (fx, fy0), (fx, fy1), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=11, color=ARROW, lw=1.2,
        shrinkA=0, shrinkB=0, zorder=5))
    return fig.text(fx + 0.012, (fy0 + fy1) / 2, text, ha="left", va="center",
                    fontsize=fs, color=ARROW)


def _pin(fig, ax, p, text, dx=0, dy=0, **kw):
    """Figure-space text anchored to a 3D point, offset in pixels."""
    x, y = _disp(ax, p)
    fx, fy = fig.transFigure.inverted().transform((x + dx, y + dy))
    return fig.text(fx, fy, text, **kw)


def _content_bbox(fig, ax, pts, texts, pad_in=0.12):
    """Tight bbox (inches) around projected 3D content plus figure-level text."""
    d = np.array([_disp(ax, p) for p in pts])
    x0, y0 = d[:, 0].min(), d[:, 1].min()
    x1, y1 = d[:, 0].max(), d[:, 1].max()
    rend = fig.canvas.get_renderer()
    for t in texts:
        bb = t.get_window_extent(rend)
        x0, y0 = min(x0, bb.x0), min(y0, bb.y0)
        x1, y1 = max(x1, bb.x1), max(y1, bb.y1)
    for pt in fig.patches:
        bb = pt.get_window_extent(rend)
        x0, y0 = min(x0, bb.x0), min(y0, bb.y0)
        x1, y1 = max(x1, bb.x1), max(y1, bb.y1)
    bb = Bbox([[x0, y0], [x1, y1]])
    return bb.transformed(fig.dpi_scale_trans.inverted()).expanded(1.0, 1.0).padded(pad_in)


# ================================================================== 3D helpers
def _plane(ax, X, Y, z, colors, lw=0.08):
    ax.plot_surface(
        X, Y, np.full_like(X, z), facecolors=colors, shade=False,
        rstride=1, cstride=1, linewidth=0, antialiased=True,
    )


def _outline(ax, z, color=EDGE, lw=0.8, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    xs = [x0, x1, x1, x0, x0]
    ys = [y0, y0, y1, y1, y0]
    ax.plot(xs, ys, [z] * 5, color=color, lw=lw, zorder=10)


def _cylinder(ax, c, z0, r=0.075, h=0.16, color=COL["objA"]):
    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(0, 1, 8)
    U, V = np.meshgrid(u, v)
    ax.plot_surface(c[0] + r * np.cos(U), c[1] + r * np.sin(U), z0 + h * V,
                    color=color, shade=True, linewidth=0.1,
                    edgecolors=(1, 1, 1, 0.25), rstride=1, cstride=1)
    rr, uu = np.meshgrid(np.linspace(0, r, 5), u)
    ax.plot_surface(c[0] + rr * np.cos(uu), c[1] + rr * np.sin(uu),
                    np.full_like(rr, z0 + h), color=color, shade=True,
                    linewidth=0, rstride=1, cstride=1)


def _cone(ax, c, z0, r=0.085, h=0.17, color=COL["objB"]):
    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(0, 1, 10)
    U, V = np.meshgrid(u, v)
    ax.plot_surface(c[0] + r * (1 - V) * np.cos(U), c[1] + r * (1 - V) * np.sin(U),
                    z0 + h * V, color=color, shade=True, linewidth=0.1,
                    edgecolors=(1, 1, 1, 0.25), rstride=1, cstride=1)


def _finish(fig, ax, out, box, labels, arrows, zs, title, title_fs=11.5,
            pins=()):
    """Place side labels/arrows in figure space, then crop to the content."""
    x0, x1, y0, y1 = box
    fig.canvas.draw()

    texts = [_side_label(fig, ax, _corners(x0, x1, y0, y1, z), name,
                         fontsize=fs, color=LABEL)
             for name, z, fs in labels]
    texts += [_pin(fig, ax, pt, txt, dx, dy, **kw) for pt, txt, dx, dy, kw in pins]

    ref = _corners(x0, x1, y0, y1, min(zs)) + _corners(x0, x1, y0, y1, max(zs))
    ymid = 0.5 * (y0 + y1)
    for (za, zb), txt in arrows:
        texts.append(_side_arrow(fig, ax, ref, (x1, ymid, za), (x1, ymid, zb), txt))

    pts = [p for z in zs for p in _corners(x0, x1, y0, y1, z)]
    fig.canvas.draw()
    bb = _content_bbox(fig, ax, pts, texts)
    cx = 0.5 * (bb.x0 + bb.x1) / fig.get_figwidth()
    cy = min(bb.y1 / fig.get_figheight() + 0.010, 0.99)
    t = fig.text(cx, cy, title, ha="center", va="bottom", fontsize=title_fs)
    texts.append(t)
    fig.canvas.draw()
    fig.savefig(out, bbox_inches=_content_bbox(fig, ax, pts, texts))
    plt.close(fig)
    print("wrote", out)


# =================================================================== figure 2D
def build_2d(out):
    n = 76
    g = np.linspace(0, 1, n)
    X, Y = np.meshgrid(g, g)

    z_layers = [0.0, 0.44, 0.88, 1.32, 1.76]
    z_map, z_world = 2.95, 4.70
    bump_h = 0.62

    fig = plt.figure(figsize=(7.0, 8.2))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)

    # --- feature layers (bottom -> top), viridis
    for (name, fn), z in zip(LAYERS_2D, z_layers):
        r = fn(X, Y)
        _plane(ax, X, Y, z, RATE_CMAP(r))
        _outline(ax, z)

    # --- composed map
    w = _composed_2d(X, Y)
    Zc = sum(w.values())
    Zc = Zc / Zc.max()
    _ = ax.plot_surface(
        X, Y, z_map + bump_h * Zc, facecolors=_blend(w), shade=True,
        rstride=1, cstride=1, linewidth=0, antialiased=True,
    )

    # --- the world
    _outline(ax, z_world, color=COL["border"], lw=1.8)
    _cylinder(ax, OBJ_A, z_world)
    _cone(ax, OBJ_B, z_world)
    for seg, key in ((CUE_A, "cueA"), (CUE_B, "cueB")):
        ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]],
                [z_world + 0.012] * 2, color=COL[key], lw=4.5,
                solid_capstyle="round")
    ax.plot([ANIMAL[0]], [ANIMAL[1]], [z_world + 0.02], marker="o", ms=7,
            color="black")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(-0.15, z_world + 0.35)
    ax.set_box_aspect((1, 1, 2.05), zoom=1.0)
    ax.view_init(elev=22, azim=-62)
    ax.set_axis_off()
    fig.subplots_adjust(left=0.24, right=0.86, top=0.94, bottom=0.02)

    names = [("the world", z_world, 9.5),
             ("cognitive map\n(composed representation)", z_map + 0.30, 8.5)]
    names += [(n, z, 8.5) for (n, _), z in zip(LAYERS_2D, z_layers)]
    _finish(fig, ax, out,
            box=(0, 1, 0, 1),
            labels=names,
            arrows=[((z_layers[-1] + 0.15, z_map - 0.35), "compose"),
                    ((z_map + bump_h + 0.15, z_world - 0.15),
                     "register\n(anchoring)")],
            zs=z_layers + [z_map, z_map + bump_h, z_world],
            title="A cognitive map is composed from feature representations,\n"
                  "then registered to the world")


# =========================================================== 1D track (200 cm)
TRACK_LEN = 200.0
BB = 30.0            # black box at each end
RZ = (90.0, 110.0)   # reward zone
RZ_C = 0.5 * (RZ[0] + RZ[1])


def _t_grid(x, spacing=55.0, phase=12.0):
    return np.clip(np.cos(2 * np.pi * (x - phase) / spacing), 0, 1) ** 1.4


def _t_cue(x, s=7.0):
    return np.exp(-(x - RZ_C) ** 2 / (2 * s ** 2))


def _t_rewardvec(x, s=11.0, d=26.0):
    return np.exp(-(x - (RZ[0] - d)) ** 2 / (2 * s ** 2))


def _t_boundary(x, s=11.0):
    d = np.minimum(np.abs(x - BB), np.abs(x - (TRACK_LEN - BB)))
    return np.exp(-d ** 2 / (2 * s ** 2))


def _t_ramp(x):
    r = np.clip((x - BB) / (RZ[0] - BB), 0, None)
    r = np.where(x <= RZ[1], np.minimum(r, 1.0), np.clip(1 - (x - RZ[1]) / 25.0, 0.08, 1))
    return np.clip(r, 0, 1) ** 1.3


LAYERS_1D = [
    ("grid cells", _t_grid),
    ("cue cells", _t_cue),
    ("reward-vector cells", _t_rewardvec),
    ("track-boundary cells", _t_boundary),
    ("ramping cells", _t_ramp),
]


def _world_track_rgb(x, ny):
    """RGB texture for the track: black boxes at both ends, patterned walls,
    green/black cue bars over the reward zone."""
    rgb = np.tile(np.array([0.88, 0.88, 0.88]), (ny, len(x), 1))
    dark = (x < BB) | (x > TRACK_LEN - BB)
    rgb[:, dark] = np.array([0.10, 0.10, 0.10])
    # wall pattern (optic flow): faint regular stripes on the track proper
    stripe = (~dark) & (np.cos(2 * np.pi * x / 10.0) > 0.55)
    rgb[:, stripe] = np.array([0.72, 0.72, 0.72])
    # cue bars marking the reward zone on cued trials
    inrz = (x >= RZ[0]) & (x <= RZ[1])
    bars = inrz & (np.cos(2 * np.pi * x / 5.0) > 0)
    rgb[:, inrz] = np.array([0.06, 0.06, 0.06])
    rgb[:, bars] = np.array(to_rgb(COL["cueB"]))
    return rgb


def _composed_1d(x):
    w = {
        "border": 0.85 * (np.exp(-(x - BB) ** 2 / (2 * 9.0 ** 2))
                          + np.exp(-(x - (TRACK_LEN - BB)) ** 2 / (2 * 9.0 ** 2))),
        "cueB": 1.00 * np.exp(-(x - RZ_C) ** 2 / (2 * 10.0 ** 2)),
        "objB": 0.30 * _t_ramp(x) * (x < RZ[1]),
    }
    return w


def build_1d(out):
    nx, ny = 420, 26
    x = np.linspace(0, TRACK_LEN, nx)
    y = np.linspace(0, 1, ny)
    Xc, Yc = np.meshgrid(x / TRACK_LEN, y)          # plot coords, x in [0,1]

    W = 0.30                                         # ribbon half-width in plot units
    Yp = (Yc - 0.5) * 2 * W

    z_layers = [0.0, 0.115, 0.230, 0.345, 0.460]
    z_map, z_world = 0.90, 1.42
    bump_h = 0.20

    fig = plt.figure(figsize=(8.0, 6.9))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)

    for (name, fn), z in zip(LAYERS_1D, z_layers):
        r = np.tile(fn(x), (ny, 1))
        ax.plot_surface(Xc, Yp, np.full_like(Xc, z), facecolors=RATE_CMAP(r),
                        shade=False, rstride=1, cstride=6, linewidth=0,
                        antialiased=True)
        _outline(ax, z, y0=-W, y1=W)

    # composed map
    w = _composed_1d(x)
    prof = sum(w.values())
    prof = prof / prof.max()
    Zc = np.tile(prof, (ny, 1))
    cols = _blend({k: np.tile(v, (ny, 1)) for k, v in w.items()}, ref=0.75)
    ax.plot_surface(Xc, Yp, z_map + bump_h * Zc, facecolors=cols, shade=True,
                    rstride=1, cstride=3, linewidth=0, antialiased=True)

    # the world
    ax.plot_surface(Xc, Yp, np.full_like(Xc, z_world),
                    facecolors=_world_track_rgb(x, ny), shade=False,
                    rstride=1, cstride=3, linewidth=0, antialiased=True)
    _outline(ax, z_world, color=COL["border"], lw=1.8, y0=-W, y1=W)
    ax.plot([ANIMAL_X := 0.34], [0.0], [z_world + 0.012], marker="o", ms=7,
            color="black")

    ax.set_xlim(0, 1); ax.set_ylim(-0.34, 0.34); ax.set_zlim(-0.05, z_world + 0.08)
    ax.set_box_aspect((1.9, 0.62, 1.40), zoom=1.0)
    ax.view_init(elev=26, azim=-62)
    ax.set_axis_off()
    fig.subplots_adjust(left=0.26, right=0.84, top=0.94, bottom=0.02)

    names = [("the world\n(virtual linear track)", z_world, 9.5),
             ("cognitive map\n(composed representation)", z_map + 0.08, 8.5)]
    names += [(n, z, 8.5) for (n, _), z in zip(LAYERS_1D, z_layers)]
    _finish(fig, ax, out,
            box=(0, 1, -W, W),
            labels=names,
            arrows=[((z_layers[-1] + 0.05, z_map - 0.08), "compose"),
                    ((z_map + bump_h + 0.03, z_world - 0.04),
                     "register\n(anchoring)")],
            zs=z_layers + [z_map, z_map + bump_h, z_world],
            title="A cognitive map is composed from feature representations, then\n"
                  "registered to the world \u2014 virtual linear track",
            title_fs=11.0,
            pins=[((RZ_C / TRACK_LEN, W, z_world), "reward zone", 0, 11,
                   dict(ha="center", va="bottom", fontsize=7.5, color=LABEL)),
                  ((BB / TRACK_LEN, -W, z_world), "black box", 0, -22,
                   dict(ha="center", va="top", fontsize=6.5, color="#777777")),
                  ((1 - BB / TRACK_LEN, -W, z_world), "black box", 0, -22,
                   dict(ha="center", va="top", fontsize=6.5, color="#777777"))])


if __name__ == "__main__":
    build_2d(f"{HERE}/fig1A_stack.pdf")
    build_1d(f"{HERE}/fig1A_stack_1D.pdf")
