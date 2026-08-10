import os, sys, math
import numpy as np
import pandas as pd
import imageio.v2 as imageio
from PIL import Image
from brainrender import Scene, settings as br_settings
from brainrender.actors import Points
from brainrender.actor import Actor
from vedo import Text2D

br_settings.SHOW_AXES = False
br_settings.BACKGROUND_COLOR = [1, 1, 1]

N_FRAMES = int(os.environ.get('N_FRAMES', '72'))
SCALE    = int(os.environ.get('SCALE', '2'))
OUT_GIF  = os.environ.get('OUT_GIF', '/private/tmp/claude-501/-Users-harryclark-Documents-spatial-manifolds/162578db-bd09-49f6-8d07-ce927cc74234/scratchpad/mec_celltypes.gif')
FRAME_DIR = '/private/tmp/claude-501/-Users-harryclark-Documents-spatial-manifolds/162578db-bd09-49f6-8d07-ce927cc74234/scratchpad/frames'
os.makedirs(FRAME_DIR, exist_ok=True)

# ---------------- transform + data (from grid_cell_anatomy_v2 cell 1) ----------------
def stereo_to_ccf(SC, angle=-0.0873):
    SC = np.asarray(SC, float)
    stretch = SC / np.array([1.0, 0.9434, 1.0])
    rotate = np.array([
        stretch[0]*math.cos(angle) - stretch[1]*math.sin(angle),
        stretch[0]*math.sin(angle) + stretch[1]*math.cos(angle),
        stretch[2]])
    return rotate + np.array([5400.0, 440.0, 5700.0])

_df = pd.read_csv('/Users/harryclark/Documents/spatial-manifolds/data/cell_classifications_no_regions.csv')
_df = _df[_df['mouse'] != 22]
_df['SC_x'] = -np.abs(_df['SC_x'])

_gc_all = _df[_df['cell_type'] == 'GC'].dropna(subset=['SC_x', 'SC_y', 'SC_z'])
_ngs    = _df[_df['cell_type'] == 'NG'].dropna(subset=['SC_x', 'SC_y', 'SC_z'])
_ns     = _df[~_df['cell_type'].isin(['GC', 'NG'])].dropna(subset=['SC_x', 'SC_y', 'SC_z'])
_gc_either = _gc_all[_gc_all['classified_by'].isin(['OF1', 'OF2'])]
_gc_both   = _gc_all[~_gc_all['classified_by'].isin(['OF1', 'OF2'])]

def _to_ccf(d):
    if len(d) == 0:
        return np.empty((0, 3))
    return np.array([stereo_to_ccf(r) for r in d[['SC_z', 'SC_y', 'SC_x']].values])

gc_both_pts   = _to_ccf(_gc_both)
gc_either_pts = _to_ccf(_gc_either)
ngs_pts       = _to_ccf(_ngs)
ns_pts        = _to_ccf(_ns)
print(f'GC both {len(gc_both_pts)}, GC either {len(gc_either_pts)}, NGS {len(ngs_pts)}, NS {len(ns_pts)}')

COL_BOTH   = '#c04744'
COL_EITHER = '#e8963e'
COL_NGS    = '#3171ae'
COL_NS     = '#cccccc'

STRUCT_REGIONS = [
    ('ENT', 'silver',         0.35),
    ('PAR', 'grey',           0.30),
    ('PRE', 'slategrey',      0.30),
    ('VIS', 'black',          0.18),
    ('SUB', 'lightsteelblue', 0.30),
    ('RSP', 'tan',            0.22),
]

# ---------------- camera framing on the cell cloud ----------------
cloud = np.vstack([p for p in [gc_both_pts, gc_either_pts, ngs_pts] if len(p) > 0])
centroid = cloud.mean(axis=0)
FOCAL = (float(centroid[0]), float(centroid[1]), float(-centroid[2]))  # rendered z is flipped
R = float(os.environ.get('R', '17000'))
ELEV = float(os.environ.get('ELEV', '-3500'))
print('focal', FOCAL)

# ---------------- build scene ----------------
scene = Scene(root=True, atlas_name='allen_mouse_10um', title='')
for rname, color, alpha in STRUCT_REGIONS:
    try:
        scene.add_brain_region(rname, alpha=alpha, color=color)
    except Exception as e:
        print(f'  skipped {rname}: {e}')

# points: draw faint NS first, then NGS, then grid cells on top
if len(ns_pts):       scene.add(Points(ns_pts, name='ns', colors=COL_NS, radius=45, alpha=0.35))
if len(ngs_pts):      scene.add(Points(ngs_pts, name='ngs', colors=COL_NGS, radius=70, alpha=0.9))
if len(gc_either_pts):scene.add(Points(gc_either_pts, name='gce', colors=COL_EITHER, radius=85, alpha=0.95))
if len(gc_both_pts):  scene.add(Points(gc_both_pts, name='gcb', colors=COL_BOTH, radius=95, alpha=1.0))

# legend as fixed 2D overlay
legend = [('Grid cell (both OF)', COL_BOTH), ('Grid cell (one OF)', COL_EITHER),
          ('Non-grid spatial', COL_NGS), ('Non-spatial', COL_NS)]
for i, (lab, col) in enumerate(legend):
    scene.add(Text2D('● ' + lab, pos=(0.02, 0.95 - 0.045*i), c=col, s=1.1, font='Arial'))
scene.add(Text2D('MEC spatial cell types', pos=(0.02, 0.02), c='k', s=1.2, font='Arial'))

def cam_at(az_deg):
    a = math.radians(az_deg)
    return {'pos': (FOCAL[0] + R*math.cos(a), FOCAL[1] + ELEV, FOCAL[2] - R*math.sin(a)),
            'focal_point': FOCAL, 'viewup': (0, -1, 0), 'clipping_range': (500, 80000)}

# initial render
angles = np.linspace(200, 200 + 360, N_FRAMES, endpoint=False)
scene.render(camera=cam_at(angles[0]), interactive=False, zoom=1.9)

frames = []
for i, az in enumerate(angles):
    scene.plotter.show(camera=cam_at(az), interactive=False, resetcam=False)
    fp = os.path.join(FRAME_DIR, f'f{i:03d}.png')
    scene.plotter.screenshot(fp, scale=SCALE)
    frames.append(fp)
    if i % 10 == 0:
        print(f'frame {i+1}/{N_FRAMES}')

try:
    scene.close()
except Exception:
    pass

TARGET_W = int(os.environ.get('TARGET_W', '950'))
imgs = []
for f in frames:
    im = Image.open(f).convert('RGB')
    if im.width > TARGET_W:
        h = round(im.height * TARGET_W / im.width)
        im = im.resize((TARGET_W, h), Image.LANCZOS)
    imgs.append(np.asarray(im))
FPS = float(os.environ.get('FPS', '10'))
imageio.mimsave(OUT_GIF, imgs, duration=1000/FPS, loop=0)
print('WROTE_GIF', OUT_GIF, os.path.getsize(OUT_GIF), 'bytes,', len(imgs), 'frames', imgs[0].shape)
