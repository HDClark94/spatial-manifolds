"""Figure 1 = the task schematic (A) beside the session/units/behaviour panels.

The two halves are built by different things -- the schematic is drawn as vectors
in figure1_schematics.ipynb and saved as fig1_schem.pdf, while the data panels
come from the v2 composite cell of the same notebook -- so this script joins them
at the PDF level rather than re-running either. Both stay VECTOR: show_pdf_page
places the source page as embedded content, it does not rasterise.

PANEL LETTERS. The data half carries no panel letters at all (checked: no single
uppercase words in its text layer), so they are added here in BOLD ARIAL, the
same face the matplotlib figures are set in. Their positions are anchored to text
already present in the source -- the axis titles 'cued', 'unit', 'MEC
population', 'median', 'behaviour', 'ADI' -- rather than to hardcoded
coordinates, so a rebuild of the data half that shifts a panel carries its letter
with it instead of silently leaving it behind.

Writes fig1_composite.pdf
"""
import os

import pymupdf

FIG = os.path.dirname(os.path.abspath(__file__))
SCHEM = f'{FIG}/fig1_schem.pdf'
DATA = f'{FIG}/fig1_v2_M26D18.pdf'
OUT = f'{FIG}/fig1_composite.pdf'

ARIAL_BOLD = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
LETTER_PT = 10.0       # matches the matplotlib fontsize=10 used for every panel
                        # letter elsewhere -- both are PDF points, so this is a
                        # genuine like-for-like match, not just the same number
GAP = 16.0            # between schematic and data half
PAD_L = 6.0
# The schematic is portrait (1.61 x 2.12 in) beside a landscape data half, so a
# fraction much below 1 leaves dead space above and below it. At ~0.97 it fills
# the height with a hairline margin; the cost is overall width, which is the
# unavoidable trade when a tall panel sits next to a wide one.
SCHEM_FRAC = 0.97     # schematic height as a fraction of the data half's height

# Panel letters for the data half. Each is placed relative to an anchor word in
# the source PDF: (letter, anchor text, which occurrence, dx, dy) in source
# coordinates, dy measured UP from the anchor's top edge.
ANCHORS = [
    ('B', 'cued', 0, -58, 10),          # cued stop raster
    ('C', 'uncued', 0, -58, 10),        # uncued stop raster
    ('D', 'unit', 0, -26, 10),          # the three example units
    ('E', 'MEC', 1, -54, 10),           # population anchoring raster + PC1
    ('F', 'median', 0, -34, 10),        # anchored fraction across sessions
    ('G', 'behaviour', 0, -26, 10),     # hit rate by state
    ('H', 'ADI', 0, -26, 10),           # anchoring dependence index
]


def main():
    for f in (SCHEM, DATA):
        if not os.path.exists(f):
            raise SystemExit(f'missing {f}')
    sch = pymupdf.open(SCHEM)
    dat = pymupdf.open(DATA)
    sr, dr = sch[0].rect, dat[0].rect

    sh = dr.height * SCHEM_FRAC
    sw = sh * (sr.width / sr.height)
    W = PAD_L + sw + GAP + dr.width
    H = dr.height

    out = pymupdf.open()
    page = out.new_page(width=W, height=H)
    # schematic, vertically centred on the left
    sy = (H - sh) / 2
    page.show_pdf_page(pymupdf.Rect(PAD_L, sy, PAD_L + sw, sy + sh), sch, 0)
    x0 = PAD_L + sw + GAP
    page.show_pdf_page(pymupdf.Rect(x0, 0, x0 + dr.width, H), dat, 0)

    page.insert_font(fontname='ab', fontfile=ARIAL_BOLD)
    # A sits at the top-left of the schematic
    page.insert_text(pymupdf.Point(PAD_L - 2, sy + 2), 'A',
                     fontname='ab', fontsize=LETTER_PT)

    words = dat[0].get_text('words')
    placed = []
    for letter, anchor, occ, dx, dy in ANCHORS:
        hits = [w for w in words if w[4] == anchor]
        if len(hits) <= occ:
            print(f'  ! anchor "{anchor}"[{occ}] not found - {letter} skipped')
            continue
        w = hits[occ]
        px = x0 + w[0] + dx
        py = w[1] - dy + LETTER_PT
        page.insert_text(pymupdf.Point(max(px, x0 + 1), max(py, LETTER_PT)),
                         letter, fontname='ab', fontsize=LETTER_PT)
        placed.append(letter)

    out.save(OUT, deflate=True)
    print(f'wrote {OUT}  ({W/72:.2f} x {H/72:.2f} in)')
    print(f'  schematic {sw/72:.2f} x {sh/72:.2f} in as panel A')
    print(f'  letters placed: A + {", ".join(placed)}')


if __name__ == '__main__':
    main()
