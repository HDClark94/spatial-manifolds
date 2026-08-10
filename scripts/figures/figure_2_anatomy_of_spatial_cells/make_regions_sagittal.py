"""Static sagittal view of the hippocampal-entorhinal region meshes inside a translucent brain."""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
from brainrender import Scene, settings as br_settings

br_settings.SHOW_AXES = False
br_settings.BACKGROUND_COLOR = [1, 1, 1]
br_settings.SHADER_STYLE = 'cartoon'
br_settings.ROOT_ALPHA = float(os.environ.get('ROOT_ALPHA', '0.12'))  # translucent brain so regions show through

SCALE   = int(os.environ.get('SCALE', '3'))
CAMERA  = os.environ.get('CAMERA', 'sagittal')
SIL_ALPHA = float(os.environ.get('SIL_ALPHA', '0'))
OUT_PNG = os.environ.get('OUT_PNG', '/Users/harryclark/Documents/spatial-manifolds/scripts/figures/figure_2_anatomy_of_spatial_cells/mec_regions_sagittal.png')
TMP = '/private/tmp/claude-501/-Users-harryclark-Documents-spatial-manifolds/162578db-bd09-49f6-8d07-ce927cc74234/scratchpad/sagittal_raw'

COL_HIPPO, COL_PREPARA, COL_MEC = '#9C6A38', '#ABDBD2', '#2E7D6C'
REGIONS = [
    ('CA',   COL_HIPPO,   0.55),
    ('DG',   COL_HIPPO,   0.55),
    ('PRE',  COL_PREPARA, 0.55),
    ('PAR',  COL_PREPARA, 0.55),
    ('ENTm', COL_MEC,     0.60),
]
LEGEND = [('Hippocampus', COL_HIPPO), ('Pre/parasubiculum', COL_PREPARA),
          ('Medial entorhinal cortex', COL_MEC)]

scene = Scene(root=True, atlas_name='allen_mouse_10um', title='')
for rname, color, alpha in REGIONS:
    try:
        scene.add_brain_region(rname, alpha=alpha, color=color)
    except Exception as e:
        print(f'  skipped {rname}: {e}')

scene.render(camera=CAMERA, interactive=False, zoom=float(os.environ.get('ZOOM', '1.15')))

# clean cartoon look: drop the silhouette outlines
sil, seen = [], set()
for a in list(scene.actors) + ([scene.root] if getattr(scene, 'root', None) is not None else []):
    s = getattr(a, 'silhouette', None)
    if s is not None and id(s.mesh) not in seen:
        seen.add(id(s.mesh)); sil.append(s.mesh)
if SIL_ALPHA <= 0:
    for m in sil:
        scene.plotter.remove(m)
else:
    for m in sil:
        m.alpha(SIL_ALPHA)

scene.plotter.render()
scene.screenshot(name=TMP, scale=SCALE)
try:
    scene.close()
except Exception:
    pass

# add a swatch legend (bordered so light/dark colours stay readable)
im = Image.open(TMP + '.png').convert('RGB')
def _font(sz):
    for p in ['/System/Library/Fonts/Supplemental/Arial.ttf', '/Library/Fonts/Arial.ttf',
              '/System/Library/Fonts/Helvetica.ttc']:
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()
if os.environ.get('LEGEND', '1') == '1':
    d = ImageDraw.Draw(im)
    sw = max(16, im.width // 42); lh = int(sw * 1.6); pad = sw
    f = _font(int(sw * 1.05))
    x, y = pad, pad
    for lab, col in LEGEND:
        d.rectangle([x, y, x + sw, y + sw], fill=ImageColor.getrgb(col), outline=(60, 60, 60), width=2)
        d.text((x + sw + sw // 2, y + sw / 2), lab, fill=(20, 20, 20), font=f, anchor='lm')
        y += lh
im.save(OUT_PNG)
print('WROTE_PNG', OUT_PNG, im.size)
