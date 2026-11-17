"""Make the mock-up PDFs smaller without touching their vector content.

The figures are embedded as vector PDFs, but many carry raster images
(rasterised heatmaps, sheet frames, rate maps) saved at 250-1400 dpi as
placed on the page. Each image is resampled down to `dpi` at its largest
placement (Lanczos, alpha premultiplied, so masked edges keep their colour)
and stored losslessly again; JPEG is never used, since MuPDF's own
rewrite_images turned some colour scatter rasters black. Each page's many
content streams are merged into one (a third of the file was their overhead),
and objects are packed into compressed object streams. Text, lines and
markers stay vector.

  python shrink_pdf.py IN.pdf OUT.pdf [dpi]      (default 300)

build_cell_pdf.py and build_nature_pdf.py call fit() on their output: the PDF
in place at 300 dpi (250 or 200 if that is still over 25 MB), and a
*_small.pdf at 150 dpi beside it.
"""
import io
import os
import sys

import pymupdf
from PIL import Image


def _placements(doc):
    """Largest on-page dpi of each image xref."""
    best = {}
    for page in doc:
        for info in page.get_image_info(xrefs=True):
            x, b = info.get('xref', 0), pymupdf.Rect(info['bbox'])
            if not x or b.width < 1 or b.height < 1:
                continue
            dpi = max(info['width'] / (b.width / 72), info['height'] / (b.height / 72))
            best[x] = max(best.get(x, 0), dpi)
    return best


def _smask(doc, xref):
    sm = doc.xref_get_key(xref, 'SMask')
    return int(sm[1].split()[0]) if sm[0] == 'xref' else 0


def _image(doc, xref):
    """The image as PIL, in L or RGB, and its soft mask as L (or None)."""
    base = pymupdf.Pixmap(doc, xref)
    if base.alpha:
        base = pymupdf.Pixmap(base, 0)
    if base.colorspace is None or base.colorspace.n not in (1, 3):
        base = pymupdf.Pixmap(pymupdf.csRGB, base)
    im = Image.frombytes('L' if base.n == 1 else 'RGB', (base.width, base.height), base.samples)
    m = _smask(doc, xref)
    mask = None
    if m:
        mp = pymupdf.Pixmap(doc, m)
        if (mp.width, mp.height) == im.size and mp.n == 1:
            mask = Image.frombytes('L', im.size, mp.samples)
    return im, mask


def _idat(im):
    """PNG-predicted zlib data of an image, as a PDF FlateDecode stream takes it."""
    buf = io.BytesIO()
    im.save(buf, 'PNG', optimize=True)
    b, out, k = buf.getvalue(), [], 8
    while k < len(b):
        n = int.from_bytes(b[k:k + 4], 'big')
        if b[k + 4:k + 8] == b'IDAT':
            out.append(b[k + 8:k + 8 + n])
        k += 12 + n
    return b''.join(out)


def _put(doc, xref, im):
    data = _idat(im)
    doc.update_stream(xref, data, compress=False)
    n = 1 if im.mode == 'L' else 3
    for k, v in (('Width', str(im.width)), ('Height', str(im.height)),
                 ('ColorSpace', '/DeviceGray' if n == 1 else '/DeviceRGB'),
                 ('BitsPerComponent', '8'), ('Filter', '/FlateDecode'),
                 ('DecodeParms', f'<</Predictor 15/Colors {n}/BitsPerComponent 8/Columns {im.width}>>'),
                 ('Decode', 'null'), ('Length', str(len(data)))):
        doc.xref_set_key(xref, k, v)


def shrink(src, dst, dpi=300, slack=1.15):
    doc = pymupdf.open(src)
    best = _placements(doc)
    n = saved = 0
    for x, d in best.items():
        if d <= dpi * slack:
            continue
        try:
            im, mask = _image(doc, x)
            w, h = max(1, round(im.width * dpi / d)), max(1, round(im.height * dpi / d))
            if mask is not None:               # premultiplied, so masked edges keep colour
                rgba = im.convert('RGB').copy()
                rgba.putalpha(mask)
                small = rgba.convert('RGBa').resize((w, h), Image.LANCZOS).convert('RGBA')
                ni = small.convert('RGB') if im.mode == 'RGB' else small.convert('L')
                nm = small.getchannel('A')
            else:
                ni, nm = im.resize((w, h), Image.LANCZOS), None
            m = _smask(doc, x)
            old = len(doc.xref_stream_raw(x)) + (len(doc.xref_stream_raw(m)) if m else 0)
            new = len(_idat(ni)) + (len(_idat(nm)) if nm is not None else 0)
            if new > .85 * old:                # not worth it (palette images grow as RGB)
                continue
            _put(doc, x, ni)
            if nm is not None:
                _put(doc, m, nm)
            n += 1
            saved += old - new
        except Exception as e:                 # leave any image it cannot handle as it is
            print(f'    image {x} left as is: {e}', flush=True)
    # the builders add one content stream per text run (tens of thousands);
    # merged into one per page they compress far better (renders identically)
    for page in doc:
        page.clean_contents(sanitize=False)
    doc.save(dst, garbage=3, deflate=True, use_objstms=1)
    doc.close()
    print(f'  {os.path.basename(dst)}: {n} images resampled to {dpi} dpi ({saved / 1e6:.1f} MB '
          f'saved in images), {os.path.getsize(dst) / 1e6:.1f} MB', flush=True)
    return dst


LIMIT = 25e6                                   # bytes


def fit(path, small_dpi=150):
    """The PDF in place at 300 dpi, or 250 or 200 if still over LIMIT, and a
    *_small.pdf beside it at small_dpi."""
    tmp = path[:-4] + '_tmp.pdf'
    for dpi in (300, 250, 200):
        shrink(path, tmp, dpi)
        if os.path.getsize(tmp) < LIMIT:
            break
    os.replace(tmp, path)
    shrink(path, path[:-4] + '_small.pdf', small_dpi)


if __name__ == '__main__':
    shrink(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 300)
