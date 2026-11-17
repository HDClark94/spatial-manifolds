"""One Figure 2 supplement covering both questions about a non-anchored trial.

These were two separate supplements and are now one, because they answer two
halves of a single question -- what IS a non-anchored trial? -- and a reader
who has the first half without the second is left with the obvious follow-up
unanswered:

    A-F   how a trial gets its label, step by step, on one session
          (fig2_supp_classifier_steps.py)
    G-J   has the field MOVED somewhere else, or has it GONE?
          (from fig2_supp_nonanchored.py)

The firing-rate panels (fig2_supp_rate.py, formerly K-Q) were removed from the
figure on 2026-10-24; the script still runs on its own.

Stacked at the PDF level rather than rebuilt as one script, the same way
fig1_composite.py joins the two halves of Figure 1: show_pdf_page embeds the
source pages as content, so both halves stay VECTOR and each source script
remains runnable and testable on its own. fig2_supp_nonanchored.py letters its panels
G-J, so the combined figure reads A-J continuously.

Writes fig2_supp_nonanchored_rate.pdf
"""
import os

import pymupdf

FIG = os.path.dirname(os.path.abspath(__file__))
STEPS = f'{FIG}/fig2_supp_classifier_steps.pdf'
TOP = f'{FIG}/fig2_supp_nonanchored.pdf'
OUT = f'{FIG}/fig2_supp_nonanchored_rate.pdf'
GAP = 14.0          # points between the two halves


def main():
    for f in (STEPS, TOP):
        if not os.path.exists(f):
            raise SystemExit(f'missing {f} -- run its own script first')
    docs = [pymupdf.open(f) for f in (STEPS, TOP)]
    # match widths: scale each part to the widest so they align on the page
    W = max(d[0].rect.width for d in docs)
    hs = [d[0].rect.height * (W / d[0].rect.width) for d in docs]
    H = sum(hs) + GAP * (len(hs) - 1)
    out = pymupdf.open()
    page = out.new_page(width=W, height=H)
    y = 0.0
    for d, h in zip(docs, hs):
        page.show_pdf_page(pymupdf.Rect(0, y, W, y + h), d, 0)
        y += h + GAP
    out.save(OUT, deflate=True)
    print(f'wrote {OUT}  ({W/72:.2f} x {H/72:.2f} in)')
    print(f'  steps  {STEPS.split("/")[-1]}  panels A-F')
    print(f'  middle {TOP.split("/")[-1]}  panels G-J')


if __name__ == '__main__':
    main()
