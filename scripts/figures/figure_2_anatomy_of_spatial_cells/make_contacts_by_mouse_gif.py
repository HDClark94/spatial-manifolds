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

N_FRAMES = int(os.environ.get('N_FRAMES', '60'))
SCALE    = int(os.environ.get('SCALE', '2'))
FPS      = float(os.environ.get('FPS', '10'))
TARGET_W = int(os.environ.get('TARGET_W', '950'))
EXCLUDE_22 = os.environ.get('EXCLUDE_22', '1') == '1'   # match the cell-type analysis (drops mouse 22)
OUT_GIF  = os.environ.get('OUT_GIF', '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/figure_2_anatomy_of_spatial_cells/mec_contacts_by_mouse.gif')
FRAME_DIR = '/private/tmp/claude-501/-Users-harryclark-Documents-spatial-manifolds/162578db-bd09-49f6-8d07-ce927cc74234/scratchpad/frames_contacts'
os.makedirs(FRAME_DIR, exist_ok=True)

def stereo_to_ccf(SC, angle=-0.0873):
    SC = np.asarray(SC, float)
    stretch = SC / np.array([1.0, 0.9434, 1.0])
    rotate = np.array([
        stretch[0]*math.cos(angle) - stretch[1]*math.sin(angle),
        stretch[0]*math.sin(angle) + stretch[1]*math.cos(angle),
        stretch[2]])
    return rotate + np.array([5400.0, 440.0, 5700.0])

# ---------------- contacts, folded onto one hemisphere like the cells ----------------
dc = pd.read_csv('/Users/harryclark/Downloads/device_contact_id_annotations.csv')
if EXCLUDE_22:
    dc = dc[dc['mouse'] != 22]
dc = dc.dropna(subset=['coord_SCs_z', 'coord_SCs_y', 'coord_SCs_x']).copy()
dc['coord_SCs_x'] = -np.abs(dc['coord_SCs_x'])   # same fold as SC_x = -abs for cells

def to_ccf(d):
    return np.array([stereo_to_ccf(r) for r in d[['coord_SCs_z', 'coord_SCs_y', 'coord_SCs_x']].values])

mice = sorted(dc['mouse'].unique())
PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf']
mouse_color = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(mice)}
pts_by_mouse = {m: to_ccf(dc[dc['mouse'] == m]) for m in mice}
print('mice:', mice, '| contacts each:', {int(m): len(p) for m, p in pts_by_mouse.items()})

STRUCT_REGIONS = [
    ('ENT', 'silver',         0.35),
    ('PAR', 'grey',           0.30),
    ('PRE', 'slategrey',      0.30),
    ('VIS', 'black',          0.18),
    ('SUB', 'lightsteelblue', 0.30),
    ('RSP', 'tan',            0.22),
]

# ---------------- camera framing (same view family as the cell-type gif) ----------------
allpts = np.vstack(list(pts_by_mouse.values()))
centroid = allpts.mean(axis=0)
FOCAL = (float(centroid[0]), float(centroid[1]), float(-centroid[2]))
R    = float(os.environ.get('R', '17000'))
ELEV = float(os.environ.get('ELEV', '-3500'))
print('focal', FOCAL)

scene = Scene(root=True, atlas_name='allen_mouse_10um', title='')
for rname, color, alpha in STRUCT_REGIONS:
    try:
        scene.add_brain_region(rname, alpha=alpha, color=color)
    except Exception as e:
        print(f'  skipped {rname}: {e}')

for m in mice:
    scene.add(Points(pts_by_mouse[m], name=f'm{m}', colors=mouse_color[m], radius=55, alpha=0.75))

for i, m in enumerate(mice):
    scene.add(Text2D(f'● M{int(m)}', pos=(0.02, 0.95 - 0.045*i), c=mouse_color[m], s=1.1, font='Arial'))
scene.add(Text2D('Probe contacts by mouse', pos=(0.02, 0.02), c='k', s=1.2, font='Arial'))

def cam_at(az_deg):
    a = math.radians(az_deg)
    return {'pos': (FOCAL[0] + R*math.cos(a), FOCAL[1] + ELEV, FOCAL[2] - R*math.sin(a)),
            'focal_point': FOCAL, 'viewup': (0, -1, 0), 'clipping_range': (500, 80000)}

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

imgs = []
for f in frames:
    im = Image.open(f).convert('RGB')
    if im.width > TARGET_W:
        im = im.resize((TARGET_W, round(im.height * TARGET_W / im.width)), Image.LANCZOS)
    imgs.append(np.asarray(im))
imageio.mimsave(OUT_GIF, imgs, duration=1000/FPS, loop=0)
print('WROTE_GIF', OUT_GIF, os.path.getsize(OUT_GIF), 'bytes,', len(imgs), 'frames @', FPS, 'fps', imgs[0].shape)
