"""One Figure 2 supplement covering both questions about a non-anchored trial.

These were two separate supplements and are now one, because they answer two
halves of a single question -- what IS a non-anchored trial? -- and a reader
who has the first half without the second is left with the obvious follow-up
unanswered:

    A-D   has the field MOVED somewhere else, or has it GONE?
          (from fig2_supp_nonanchored.py)
    E-K   and does the cell's firing RATE change with the state, or only where
          it fires?  (from fig2_supp_rate.py)

Stacked at the PDF level rather than rebuilt as one script, the same way
fig1_composite.py joins the two halves of Figure 1: show_pdf_page embeds the
source pages as content, so both halves stay VECTOR and each source script
remains runnable and testable on its own. fig2_supp_rate.py letters its panels
from E so the combined figure reads A-K continuously.

Writes fig2_supp_nonanchored_rate.pdf
"""
import os

import pymupdf

FIG = os.path.dirname(os.path.abspath(__file__))
TOP = f'{FIG}/fig2_supp_nonanchored.pdf'
BOT = f'{FIG}/fig2_supp_rate.pdf'
OUT = f'{FIG}/fig2_supp_nonanchored_rate.pdf'
GAP = 14.0          # points between the two halves


def main():
    for f in (TOP, BOT):
        if not os.path.exists(f):
            raise SystemExit(f'missing {f} -- run its own script first')
    a = pymupdf.open(TOP)
    b = pymupdf.open(BOT)
    ra, rb = a[0].rect, b[0].rect

    # match widths: scale the narrower half up so the two align on the page
    W = max(ra.width, rb.width)
    ha = ra.height * (W / ra.width)
    hb = rb.height * (W / rb.width)
    H = ha + GAP + hb

    out = pymupdf.open()
    page = out.new_page(width=W, height=H)
    page.show_pdf_page(pymupdf.Rect(0, 0, W, ha), a, 0)
    page.show_pdf_page(pymupdf.Rect(0, ha + GAP, W, H), b, 0)
    out.save(OUT, deflate=True)
    print(f'wrote {OUT}  ({W/72:.2f} x {H/72:.2f} in)')
    print(f'  top    {TOP.split("/")[-1]}  panels A-D')
    print(f'  bottom {BOT.split("/")[-1]}  panels E-K')


if __name__ == '__main__':
    main()
