"""Orbiting brainrender GIF of the Neuropixel 2.0 multishank probe in MEC,
using the hippocampal-entorhinal region colour scheme."""
import os, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
from brainrender import Scene, settings as br_settings
from brainrender.actors import Cylinder
from brainrender.actor import Actor
from vedo import shapes

br_settings.SHOW_AXES = False
br_settings.BACKGROUND_COLOR = [1, 1, 1]
br_settings.SHADER_STYLE = 'cartoon'
br_settings.ROOT_ALPHA = float(os.environ.get('ROOT_ALPHA', '0.12'))

FRAMES_PER_TURN = int(os.environ.get('FRAMES_PER_TURN', '180'))
TURNS           = int(os.environ.get('TURNS', '2'))
SCALE           = int(os.environ.get('SCALE', '2'))
FPS             = float(os.environ.get('FPS', '20'))
TARGET_W        = int(os.environ.get('TARGET_W', '950'))
PAL_COLORS      = int(os.environ.get('PAL_COLORS', '255'))
SIL_ALPHA       = float(os.environ.get('SIL_ALPHA', '0'))
OUT_GIF = os.environ.get('OUT_GIF', '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/figure_2_anatomy_of_spatial_cells/mec_probe_tour.gif')
FRAME_DIR = '/private/tmp/claude-501/-Users-harryclark-Documents-spatial-manifolds/162578db-bd09-49f6-8d07-ce927cc74234/scratchpad/frames_probe'
os.makedirs(FRAME_DIR, exist_ok=True)

# ---- region colour scheme (same as the region/cell tours) ----
COL_HIPPO, COL_PREPARA, COL_MEC = '#9C6A38', '#ABDBD2', '#2E7D6C'
COL_PROBE = '#1c1c1c'
REGIONS = [('CA', COL_HIPPO, 0.40), ('DG', COL_HIPPO, 0.40),
           ('PRE', COL_PREPARA, 0.40), ('PAR', COL_PREPARA, 0.40),
           ('ENTm', COL_MEC, 0.45)]
LEGEND = [('Hippocampus', COL_HIPPO), ('Pre/parasubiculum', COL_PREPARA),
          ('Medial entorhinal cortex', COL_MEC), ('Neuropixel probe', COL_PROBE)]


# ---- stereotaxic -> CCF + probe geometry (from the provided example) ----
def StereoToCCF(SC=np.array([1, 1, 1]), angle=-0.0873):
    stretch = np.asarray(SC, float) / np.array([1, 0.9434, 1])
    rotate = np.array([stretch[0]*math.cos(angle) - stretch[1]*math.sin(angle),
                       stretch[0]*math.sin(angle) + stretch[1]*math.cos(angle),
                       stretch[2]])
    return rotate + np.array([5400, 440, 5700])


class Cylinder2(Actor):
    def __init__(self, pos_from, pos_to, color='powderblue', alpha=1, radius=350):
        mesh = shapes.Cylinder(pos=[pos_from, pos_to], c=color, r=radius, alpha=alpha)
        Actor.__init__(self, mesh, name='Cylinder', br_class='Cylinder')


def add_probes_in_x(scene, cranial_site_xyz, depth=1000, attack_angle_x=-15,
                    color='red', distance_between_probes=250, radius=20):
    a = np.deg2rad(-attack_angle_x)
    target = cranial_site_xyz + np.array([depth*np.sin(a), depth*np.cos(a), 0])
    for i in range(4):
        off = np.array([0, 0, -i*distance_between_probes])
        scene.add(Cylinder2(StereoToCCF(cranial_site_xyz+off), StereoToCCF(target+off),
                            color=color, radius=radius))
    return target


def add_neuropixel2_multishank(scene, cranial_site_xyz, attack_angle_x, depth, radius=35):
    add_probes_in_x(scene, cranial_site_xyz, attack_angle_x=attack_angle_x, color=COL_PROBE, depth=depth, radius=radius)
    add_probes_in_x(scene, cranial_site_xyz, attack_angle_x=attack_angle_x, color=(75, 75, 75), depth=depth-3000, radius=radius+1)


MEC_CRANIAL = np.array([3600, -1000, -3200])
MEC_DEPTH   = 5000
MEC_ANGLE   = -10


def build_scene():
    s = Scene(root=True, inset=False, atlas_name='allen_mouse_10um', title='')
    for rname, color, alpha in REGIONS:
        try:
            s.add_brain_region(rname, alpha=alpha, color=color)
        except Exception as e:
            print(f'  skipped {rname}: {e}')
    add_neuropixel2_multishank(s, MEC_CRANIAL, attack_angle_x=MEC_ANGLE, depth=MEC_DEPTH,
                               radius=int(os.environ.get('PROBE_RADIUS', '45')))
    return s


# focal = middle of the probe (CCF), rendered z is flipped
a = np.deg2rad(-MEC_ANGLE)
_mid = MEC_CRANIAL + 0.5 * np.array([MEC_DEPTH*np.sin(a), MEC_DEPTH*np.cos(a), 0])
_f = StereoToCCF(_mid)
FOCAL = (float(_f[0]), float(_f[1]), float(-_f[2]))
R    = float(os.environ.get('R', '15000'))
ELEV = float(os.environ.get('ELEV', '-3000'))
ZOOM = float(os.environ.get('ZOOM', '1.7'))
print('focal', FOCAL)


def cam_at(az_deg):
    ang = math.radians(az_deg)
    return {'pos': (FOCAL[0] + R*math.cos(ang), FOCAL[1] + ELEV, FOCAL[2] - R*math.sin(ang)),
            'focal_point': FOCAL, 'viewup': (0, -1, 0), 'clipping_range': (500, 80000)}


scene = build_scene()
angles = np.linspace(200, 200 + 360, FRAMES_PER_TURN, endpoint=False)
scene.render(camera=cam_at(angles[0]), interactive=False, zoom=ZOOM)

# drop cartoon silhouette outlines (as in the region/cell tours)
sil, seen = [], set()
for act in list(scene.actors) + ([scene.root] if getattr(scene, 'root', None) is not None else []):
    sh = getattr(act, 'silhouette', None)
    if sh is not None and id(sh.mesh) not in seen:
        seen.add(id(sh.mesh)); sil.append(sh.mesh)
if SIL_ALPHA <= 0:
    for m in sil:
        scene.plotter.remove(m)
else:
    for m in sil:
        m.alpha(SIL_ALPHA)

frames = []
for i, az in enumerate(angles):
    scene.plotter.show(camera=cam_at(az), interactive=False, resetcam=False)
    fp = os.path.join(FRAME_DIR, f'f{i:03d}.png')
    scene.plotter.screenshot(fp, scale=SCALE)
    frames.append(fp)
    if i % 30 == 0:
        print(f'frame {i+1}/{FRAMES_PER_TURN}')
try:
    scene.close()
except Exception:
    pass

# ---- assemble GIF: shared palette (forced legend colours) + swatch legend ----
def _font(sz):
    for p in ['/System/Library/Fonts/Supplemental/Arial.ttf', '/Library/Fonts/Arial.ttf',
              '/System/Library/Fonts/Helvetica.ttc']:
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()

def draw_legend(im):
    d = ImageDraw.Draw(im)
    sw = max(11, im.width // 36); lh = int(sw * 1.55); pad = sw
    f, ft = _font(int(sw*0.95)), _font(int(sw*1.15))
    x, y = pad, pad
    for lab, col in LEGEND:
        d.rectangle([x, y, x+sw, y+sw], fill=ImageColor.getrgb(col), outline=(60, 60, 60), width=1)
        d.text((x+sw+sw//2, y+sw/2), lab, fill=(20, 20, 20), font=f, anchor='lm')
        y += lh
    d.text((pad, im.height-pad), 'MEC Neuropixel implant', fill=(0, 0, 0), font=ft, anchor='lb')
    return im

_cache = {}
def load(fp):
    if fp not in _cache:
        im = Image.open(fp).convert('RGB')
        if im.width > TARGET_W:
            im = im.resize((TARGET_W, round(im.height*TARGET_W/im.width)), Image.LANCZOS)
        _cache[fp] = draw_legend(im)
    return _cache[fp]

pil_frames = [load(fp) for fp in frames] * TURNS
W, H = pil_frames[0].size
sample = [pil_frames[int(i)] for i in np.linspace(0, len(pil_frames)-1, 9)]
montage = Image.new('RGB', (W, H*len(sample)))
for k, im in enumerate(sample):
    montage.paste(im, (0, k*H))
# force legend colours into the palette so swatches stay exact
force = [c for _, c in LEGEND] + ['#000000', '#ffffff']
blk = max(24, H // 12)
bar = Image.new('RGB', (W, blk*len(force)))
bd = ImageDraw.Draw(bar)
for i, c in enumerate(force):
    bd.rectangle([0, i*blk, W, (i+1)*blk], fill=ImageColor.getrgb(c))
pal_src = Image.new('RGB', (W, montage.height + bar.height))
pal_src.paste(montage, (0, 0)); pal_src.paste(bar, (0, montage.height))
pal_img = pal_src.convert('P', palette=Image.ADAPTIVE, colors=PAL_COLORS)
q = [f.quantize(palette=pal_img, dither=Image.Dither.NONE) for f in pil_frames]
q[0].save(OUT_GIF, save_all=True, append_images=q[1:], duration=1000/FPS, loop=0, disposal=1, optimize=False)
print('WROTE_GIF', OUT_GIF, round(os.path.getsize(OUT_GIF)/1e6, 1), 'MB,', len(q), 'frames', (H, W))
