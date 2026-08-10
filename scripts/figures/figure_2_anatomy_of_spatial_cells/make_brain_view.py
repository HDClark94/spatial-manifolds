"""Still of the whole (translucent) brain with the hippocampal-entorhinal regions and
the Neuropixel multishank probe in MEC, viewed from a clock position in the horizontal
plane (anterior = 12 o'clock). Default CLOCK=2.7 -> near-sagittal, rotated slightly
toward anterior. Same colour scheme as the region/probe tours."""
import os, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
from brainrender import Scene, settings as br_settings
from brainrender.actor import Actor
from vedo import shapes

br_settings.SHOW_AXES = False
br_settings.BACKGROUND_COLOR = [1, 1, 1]
br_settings.SHADER_STYLE = 'cartoon'
ALPHA   = float(os.environ.get('ALPHA', '0.12'))   # brain transparency
br_settings.ROOT_ALPHA = ALPHA

CLOCK   = float(os.environ.get('CLOCK', '2.7'))    # 12 = anterior, 3 = full sagittal
HANDED  = float(os.environ.get('HANDED', '1'))
SCALE   = int(os.environ.get('SCALE', '3'))
ELEV    = float(os.environ.get('ELEV', '-1200'))
R       = float(os.environ.get('R', '26000'))
ZOOM    = float(os.environ.get('ZOOM', '1.45'))
SIL     = os.environ.get('SIL', '0') == '1'
SHOW_REGIONS = os.environ.get('SHOW_REGIONS', '1') == '1'
SHOW_PROBE   = os.environ.get('SHOW_PROBE', '1') == '1'
LEGEND       = os.environ.get('LEGEND', '1') == '1'
OUT_PNG = os.environ.get('OUT_PNG', '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/figure_2_anatomy_of_spatial_cells/brain_view_2oclock.png')
TMP = '/private/tmp/claude-501/-Users-harryclark-Documents-spatial-manifolds/162578db-bd09-49f6-8d07-ce927cc74234/scratchpad/brainview_raw'

# ---- region colour scheme (same as the tours) ----
COL_HIPPO, COL_PREPARA, COL_MEC = '#9C6A38', '#ABDBD2', '#2E7D6C'
COL_PROBE = '#1c1c1c'
REGIONS = [('CA', COL_HIPPO, 0.55), ('DG', COL_HIPPO, 0.55),
           ('PRE', COL_PREPARA, 0.55), ('PAR', COL_PREPARA, 0.55),
           ('ENTm', COL_MEC, 0.6)]
LEGEND_ITEMS = [('Hippocampus', COL_HIPPO), ('Pre/parasubiculum', COL_PREPARA),
                ('Medial entorhinal cortex', COL_MEC), ('Neuropixel probe', COL_PROBE)]


# ---- stereotaxic -> CCF + probe (from the provided example) ----
def StereoToCCF(SC, angle=-0.0873):
    stretch = np.asarray(SC, float) / np.array([1, 0.9434, 1])
    rotate = np.array([stretch[0]*math.cos(angle) - stretch[1]*math.sin(angle),
                       stretch[0]*math.sin(angle) + stretch[1]*math.cos(angle), stretch[2]])
    return rotate + np.array([5400, 440, 5700])

class Cylinder2(Actor):
    def __init__(self, pos_from, pos_to, color='powderblue', alpha=1, radius=350):
        Actor.__init__(self, shapes.Cylinder(pos=[pos_from, pos_to], c=color, r=radius, alpha=alpha),
                       name='Cylinder', br_class='Cylinder')

def add_probes_in_x(scene, cranial, depth, attack_angle_x, color, radius, dbp=250):
    a = np.deg2rad(-attack_angle_x)
    target = cranial + np.array([depth*np.sin(a), depth*np.cos(a), 0])
    for i in range(4):
        off = np.array([0, 0, -i*dbp])
        scene.add(Cylinder2(StereoToCCF(cranial+off), StereoToCCF(target+off), color=color, radius=radius))

def add_probe(scene, cranial, attack_angle_x, depth, radius=45):
    add_probes_in_x(scene, cranial, depth, attack_angle_x, COL_PROBE, radius)
    add_probes_in_x(scene, cranial, depth-3000, attack_angle_x, (75, 75, 75), radius+1)

MEC_CRANIAL, MEC_DEPTH, MEC_ANGLE = np.array([3600, -1000, -3200]), 5000, -10

# ---- build scene ----
scene = Scene(root=True, inset=False, atlas_name='allen_mouse_10um', title='')
if ALPHA >= 1.0:
    scene.root.mesh.color('#d9dcd6').alpha(1.0)
if SHOW_REGIONS:
    for rname, color, alpha in REGIONS:
        try:
            scene.add_brain_region(rname, alpha=alpha, color=color)
        except Exception as e:
            print(f'  skipped {rname}: {e}')
if SHOW_PROBE:
    add_probe(scene, MEC_CRANIAL, attack_angle_x=MEC_ANGLE, depth=MEC_DEPTH,
              radius=int(os.environ.get('PROBE_RADIUS', '45')))

# ---- clock camera (x=AP, y=DV, z=-ML) ----
b = scene.root.mesh.bounds()
cx, cy, cz = (b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2
theta = math.radians(CLOCK * 30.0)
offset = math.cos(theta)*np.array([-1.0, 0.0, 0.0]) + math.sin(theta)*np.array([0.0, 0.0, HANDED])
cam = {'pos': (cx + R*offset[0], cy + ELEV, cz + R*offset[2]),
       'focal_point': (cx, cy, cz), 'viewup': (0, -1, 0), 'clipping_range': (500, 90000)}

scene.render(camera=cam, interactive=False, zoom=ZOOM)
if not SIL:
    seen = set()
    for act in list(scene.actors) + ([scene.root] if getattr(scene, 'root', None) is not None else []):
        s = getattr(act, 'silhouette', None)
        if s is not None and id(s.mesh) not in seen:
            seen.add(id(s.mesh)); scene.plotter.remove(s.mesh)
    scene.plotter.render()

scene.screenshot(name=TMP, scale=SCALE)
try:
    scene.close()
except Exception:
    pass

# ---- legend + autocrop ----
im = Image.open(TMP + '.png').convert('RGB')
def _font(sz):
    for p in ['/System/Library/Fonts/Supplemental/Arial.ttf', '/Library/Fonts/Arial.ttf',
              '/System/Library/Fonts/Helvetica.ttc']:
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()

from PIL import ImageChops
diff = ImageChops.difference(im, Image.new('RGB', im.size, (255, 255, 255)))
bbox = diff.getbbox()
if bbox:
    pad = int(0.05 * max(im.size)); l, t, r, bo = bbox
    im = im.crop((max(0, l-pad), max(0, t-pad), min(im.width, r+pad), min(im.height, bo+pad)))

if LEGEND and (SHOW_REGIONS or SHOW_PROBE):
    d = ImageDraw.Draw(im)
    sw = max(16, im.width // 42); lh = int(sw*1.6); pad = sw
    f = _font(int(sw*1.05))
    x, y = pad, pad
    for lab, col in LEGEND_ITEMS:
        if lab == 'Neuropixel probe' and not SHOW_PROBE:
            continue
        if lab != 'Neuropixel probe' and not SHOW_REGIONS:
            continue
        d.rectangle([x, y, x+sw, y+sw], fill=ImageColor.getrgb(col), outline=(60, 60, 60), width=2)
        d.text((x+sw+sw//2, y+sw/2), lab, fill=(20, 20, 20), font=f, anchor='lm')
        y += lh

im.save(OUT_PNG)
print('WROTE', OUT_PNG, im.size)
