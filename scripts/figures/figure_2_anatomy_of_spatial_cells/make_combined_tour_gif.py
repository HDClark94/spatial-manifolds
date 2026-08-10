import os, math
import numpy as np
import pandas as pd
import imageio.v2 as imageio
from PIL import Image
from brainrender import Scene, settings as br_settings
from brainrender.actors import Points
from vedo import Text2D

br_settings.SHOW_AXES = False
br_settings.BACKGROUND_COLOR = [1, 1, 1]
br_settings.SHADER_STYLE = 'cartoon'   # cartoon look, but silhouette outlines are softened below

SIL_ALPHA = float(os.environ.get('SIL_ALPHA', '0'))     # edge outline opacity; 0 = no outlines
SIL_COLOR = os.environ.get('SIL_COLOR', 'k')            # outline colour (kept black, just translucent)

FRAMES_PER_TURN = int(os.environ.get('FRAMES_PER_TURN', '90'))   # angular resolution (4 deg/frame)
TURNS_PER_PHASE = int(os.environ.get('TURNS_PER_PHASE', '2'))
SCALE    = int(os.environ.get('SCALE', '2'))
FPS      = float(os.environ.get('FPS', '20'))
TARGET_W = int(os.environ.get('TARGET_W', '760'))
OUT_GIF  = os.environ.get('OUT_GIF', '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/figure_2_anatomy_of_spatial_cells/mec_combined_tour.gif')
FRAME_DIR = '/private/tmp/claude-501/-Users-harryclark-Documents-spatial-manifolds/162578db-bd09-49f6-8d07-ce927cc74234/scratchpad/frames_tour'
os.makedirs(FRAME_DIR, exist_ok=True)

def stereo_to_ccf(SC, angle=-0.0873):
    SC = np.asarray(SC, float)
    stretch = SC / np.array([1.0, 0.9434, 1.0])
    rotate = np.array([
        stretch[0]*math.cos(angle) - stretch[1]*math.sin(angle),
        stretch[0]*math.sin(angle) + stretch[1]*math.cos(angle),
        stretch[2]])
    return rotate + np.array([5400.0, 440.0, 5700.0])

# ---------------- cells ----------------
_df = pd.read_csv('/Users/harryclark/Documents/spatial-manifolds/data/cell_classifications_no_regions.csv')
_df = _df[_df['mouse'] != 22]
_df['SC_x'] = -np.abs(_df['SC_x'])
_gc_all = _df[_df['cell_type'] == 'GC'].dropna(subset=['SC_x', 'SC_y', 'SC_z'])
_ngs    = _df[_df['cell_type'] == 'NG'].dropna(subset=['SC_x', 'SC_y', 'SC_z'])
_ns     = _df[~_df['cell_type'].isin(['GC', 'NG'])].dropna(subset=['SC_x', 'SC_y', 'SC_z'])
_gc_either = _gc_all[_gc_all['classified_by'].isin(['OF1', 'OF2'])]
_gc_both   = _gc_all[~_gc_all['classified_by'].isin(['OF1', 'OF2'])]
def _cells_ccf(d):
    return np.empty((0, 3)) if len(d) == 0 else np.array([stereo_to_ccf(r) for r in d[['SC_z', 'SC_y', 'SC_x']].values])
gc_both_pts, gc_either_pts = _cells_ccf(_gc_both), _cells_ccf(_gc_either)
ngs_pts, ns_pts = _cells_ccf(_ngs), _cells_ccf(_ns)
COL_BOTH, COL_EITHER, COL_NGS, COL_NS = '#c04744', '#e8963e', '#3171ae', '#cccccc'

# ---------------- contacts ----------------
dc = pd.read_csv('/Users/harryclark/Downloads/device_contact_id_annotations.csv')
dc = dc[dc['mouse'] != 22].dropna(subset=['coord_SCs_z', 'coord_SCs_y', 'coord_SCs_x']).copy()
dc['coord_SCs_x'] = -np.abs(dc['coord_SCs_x'])
def _contacts_ccf(d):
    return np.array([stereo_to_ccf(r) for r in d[['coord_SCs_z', 'coord_SCs_y', 'coord_SCs_x']].values])
mice = sorted(dc['mouse'].unique())
PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf']
mouse_color = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(mice)}
contact_pts = {m: _contacts_ccf(dc[dc['mouse'] == m]) for m in mice}

# hippocampal-entorhinal circuit colour scheme
COL_HIPPO   = '#9C6A38'   # hippocampus (brown)
COL_PREPARA = '#ABDBD2'   # pre/parasubiculum (light teal)
COL_MEC     = '#2E7D6C'   # medial entorhinal cortex (dark teal)
STRUCT_REGIONS = [
    ('CA',   COL_HIPPO,   0.40),   # hippocampus = CA + DG only (excludes midline IG/FC)
    ('DG',   COL_HIPPO,   0.40),
    ('PRE',  COL_PREPARA, 0.40),   # PRE + PAR share the pre/parasubiculum colour
    ('PAR',  COL_PREPARA, 0.40),
    ('ENTm', COL_MEC,     0.40),
]
# the region colour scheme is used consistently in every phase; PRE+PAR collapse to one
# legend entry ("Pre/parasubiculum") since they share a colour.
REGION_LEGEND = [
    ('Hippocampus', COL_HIPPO),
    ('Pre/parasubiculum', COL_PREPARA),
    ('Medial entorhinal cortex', COL_MEC),
]

# ---------------- shared camera ----------------
cloud = np.vstack([gc_both_pts, gc_either_pts, ngs_pts])
centroid = cloud.mean(axis=0)
FOCAL = (float(centroid[0]), float(centroid[1]), float(-centroid[2]))
R    = float(os.environ.get('R', '17000'))
ELEV = float(os.environ.get('ELEV', '-3500'))

def cam_at(az_deg):
    a = math.radians(az_deg)
    return {'pos': (FOCAL[0] + R*math.cos(a), FOCAL[1] + ELEV, FOCAL[2] - R*math.sin(a)),
            'focal_point': FOCAL, 'viewup': (0, -1, 0), 'clipping_range': (500, 80000)}

def add_regions(scene, region_list):
    for rname, color, alpha in region_list:
        try:
            scene.add_brain_region(rname, alpha=alpha, color=color)
        except Exception as e:
            print(f'  skipped {rname}: {e}')

# ---------------- per-phase scene builders (legends are drawn later via PIL) ----------------
def build_regions():
    s = Scene(root=True, atlas_name='allen_mouse_10um', title='')
    add_regions(s, STRUCT_REGIONS)
    return s

def build_contacts():
    s = Scene(root=True, atlas_name='allen_mouse_10um', title='')
    add_regions(s, STRUCT_REGIONS)
    for m in mice:
        s.add(Points(contact_pts[m], name=f'm{m}', colors=mouse_color[m], radius=55, alpha=0.75))
    return s

def build_cells():
    s = Scene(root=True, atlas_name='allen_mouse_10um', title='')
    add_regions(s, STRUCT_REGIONS)
    if len(ns_pts):        s.add(Points(ns_pts, name='ns', colors=COL_NS, radius=45, alpha=0.35))
    if len(ngs_pts):       s.add(Points(ngs_pts, name='ngs', colors=COL_NGS, radius=70, alpha=0.9))
    if len(gc_either_pts): s.add(Points(gc_either_pts, name='gce', colors=COL_EITHER, radius=85, alpha=0.95))
    if len(gc_both_pts):   s.add(Points(gc_both_pts, name='gcb', colors=COL_BOTH, radius=95, alpha=1.0))
    return s

def build_grid_both():
    s = Scene(root=True, atlas_name='allen_mouse_10um', title='')
    add_regions(s, STRUCT_REGIONS)
    if len(gc_both_pts):   s.add(Points(gc_both_pts, name='gcb', colors=COL_BOTH, radius=95, alpha=1.0))
    return s

# (phase name, scene builder, number of rotations to play)
PHASES = [
    ('regions',   build_regions,   1),
    ('contacts',  build_contacts,  2),
    ('cells',     build_cells,     1),
    ('grid_both', build_grid_both, 1),
]
PHASE_LEGENDS = {
    'regions':  ('Brain regions', REGION_LEGEND),
    'contacts': ('Probe contacts by mouse', [(f'M{int(m)}', mouse_color[m]) for m in mice]),
    'cells':    ('Spatial cell types',
                 [('Grid cell (both OF)', COL_BOTH), ('Grid cell (one OF)', COL_EITHER),
                  ('Non-grid spatial', COL_NGS), ('Non-spatial', COL_NS)]),
    'grid_both': ('Grid cells (both OF)', [('Grid cell (both OF)', COL_BOTH)]),
}
angles = np.linspace(200, 200 + 360, FRAMES_PER_TURN, endpoint=False)

# ---------------- render one rotation per phase (or reuse cached frames) ----------------
phase_frames = {}
if os.environ.get('REASSEMBLE_ONLY') == '1':
    for pname, _b, _t in PHASES:
        phase_frames[pname] = [os.path.join(FRAME_DIR, f)
                               for f in sorted(os.listdir(FRAME_DIR)) if f.startswith(pname + '_')]
    print('REASSEMBLE_ONLY: cached frames', {k: len(v) for k, v in phase_frames.items()})
else:
  for pname, builder, _turns in PHASES:
    print(f'=== phase: {pname} ===')
    scene = builder()
    scene.render(camera=cam_at(angles[0]), interactive=False, zoom=1.9)
    # cartoon edge outlines live on each brain-region actor's (and root's) .silhouette;
    # drop them entirely if SIL_ALPHA<=0, else make them see-through
    sil_meshes, seen = [], set()
    for a in list(scene.actors) + ([scene.root] if getattr(scene, 'root', None) is not None else []):
        sil = getattr(a, 'silhouette', None)
        if sil is not None and id(sil.mesh) not in seen:
            seen.add(id(sil.mesh))
            sil_meshes.append(sil.mesh)
    if SIL_ALPHA <= 0:
        for m in sil_meshes:
            scene.plotter.remove(m)
    else:
        for m in sil_meshes:
            m.alpha(SIL_ALPHA)
            if SIL_COLOR:
                m.c(SIL_COLOR)
    fps_list = []
    for i, az in enumerate(angles):
        scene.plotter.show(camera=cam_at(az), interactive=False, resetcam=False)
        fp = os.path.join(FRAME_DIR, f'{pname}_{i:03d}.png')
        scene.plotter.screenshot(fp, scale=SCALE)
        fps_list.append(fp)
        if i % 30 == 0:
            print(f'  frame {i+1}/{FRAMES_PER_TURN}')
    try:
        scene.close()
    except Exception:
        pass
    phase_frames[pname] = fps_list

# optionally thin each phase's rotation to a target frame count (keeps total under a cap)
_sub = os.environ.get('SUBSAMPLE_PER_TURN')
if _sub:
    n = int(_sub)
    for p in phase_frames:
        fs = phase_frames[p]
        idx = sorted(set(int(round(i)) for i in np.linspace(0, len(fs) - 1, n)))
        phase_frames[p] = [fs[i] for i in idx]
    total = sum(len(phase_frames[p]) * t for p, _b, t in PHASES)
    print('subsampled per turn ->', {k: len(v) for k, v in phase_frames.items()}, '| total frames', total)

# ---------------- assemble: each phase spun TURNS_PER_PHASE times ----------------
from PIL import ImageDraw, ImageFont, ImageColor

def _font(sz):
    for p in ['/System/Library/Fonts/Supplemental/Arial.ttf', '/Library/Fonts/Arial.ttf',
              '/System/Library/Fonts/Helvetica.ttc']:
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()

def draw_legend(im, phase):
    """Bordered-swatch legend (top-left) + title (bottom-left), readable for any fill colour."""
    title, entries = PHASE_LEGENDS[phase]
    d = ImageDraw.Draw(im)
    sw = max(11, im.width // 36)          # swatch size scales with frame width
    lh = int(sw * 1.55)
    pad = sw
    f, ft = _font(int(sw * 0.95)), _font(int(sw * 1.15))
    x, y = pad, pad
    for lab, col in entries:
        rgb = ImageColor.getrgb(col)
        d.rectangle([x, y, x + sw, y + sw], fill=rgb, outline=(60, 60, 60), width=1)
        d.text((x + sw + sw // 2, y + sw / 2), lab, fill=(20, 20, 20), font=f, anchor='lm')
        y += lh
    d.text((pad, im.height - pad), title, fill=(0, 0, 0), font=ft, anchor='lb')
    return im

_cache = {}
def load_composited(fp, phase):
    if fp not in _cache:
        im = Image.open(fp).convert('RGB')
        if im.width > TARGET_W:
            im = im.resize((TARGET_W, round(im.height * TARGET_W / im.width)), Image.LANCZOS)
        im = draw_legend(im, phase)
        _cache[fp] = np.asarray(im)
    return _cache[fp]

sequence = []
for pname, _b, turns in PHASES:
    for _turn in range(turns):
        sequence.extend((fp, pname) for fp in phase_frames[pname])
imgs = [load_composited(fp, ph) for fp, ph in sequence]
print('sequence:', [(p, t) for p, _b, t in PHASES], '| total output frames', len(sequence))

# Quantise every frame to ONE shared global palette (no dithering) so static elements
# like the legend don't flicker frame-to-frame from per-frame palette changes.
pil_frames = [Image.fromarray(a) for a in imgs]
W, H = pil_frames[0].size
sample_idx = sorted(set(int(i) for i in np.linspace(0, len(pil_frames) - 1, 9)))
montage = Image.new('RGB', (W, H * len(sample_idx)))
for k, si in enumerate(sample_idx):
    montage.paste(pil_frames[si], (0, k * H))
# force the exact legend/UI colours into the palette source so saturated swatches
# (e.g. the green M25) aren't collapsed to a nearby grey by median-cut quantization
force_cols, _seen = [], set()
for _title, _entries in PHASE_LEGENDS.values():
    for _lab, _c in _entries:
        if _c not in _seen:
            _seen.add(_c); force_cols.append(_c)
for _c in ('#000000', '#ffffff'):
    if _c not in _seen:
        _seen.add(_c); force_cols.append(_c)
blk = max(24, H // 12)
bar = Image.new('RGB', (W, blk * len(force_cols)))
_bd = ImageDraw.Draw(bar)
for i, c in enumerate(force_cols):
    _bd.rectangle([0, i * blk, W, (i + 1) * blk], fill=ImageColor.getrgb(c))
pal_src = Image.new('RGB', (W, montage.height + bar.height))
pal_src.paste(montage, (0, 0))
pal_src.paste(bar, (0, montage.height))

# 255 colours (not 256) leaves a palette slot so PIL can use transparency for
# inter-frame optimisation (unchanged pixels -> transparent -> only the moving region is stored)
PAL_COLORS = int(os.environ.get('PAL_COLORS', '255'))
OPTIMIZE = os.environ.get('OPTIMIZE', '1') == '1'
pal_img = pal_src.convert('P', palette=Image.ADAPTIVE, colors=PAL_COLORS)
q_frames = [f.quantize(palette=pal_img, dither=Image.Dither.NONE) for f in pil_frames]
q_frames[0].save(OUT_GIF, save_all=True, append_images=q_frames[1:],
                 duration=1000/FPS, loop=0, disposal=1, optimize=OPTIMIZE)
print('WROTE_GIF', OUT_GIF, round(os.path.getsize(OUT_GIF)/1e6, 1), 'MB,', len(q_frames),
      'frames @', FPS, 'fps', imgs[0].shape, '| optimize', OPTIMIZE, '| palcolors', PAL_COLORS)
