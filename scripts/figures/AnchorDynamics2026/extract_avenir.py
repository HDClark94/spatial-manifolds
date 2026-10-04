"""Extract the Avenir Next faces the paper mock-up sets its display type in.

macOS ships Avenir Next as a TrueType Collection, and PyMuPDF's Font/insert_font
load a single face from a file -- given a .ttc they take face 0 and ignore the
other eleven, so asking for Regular silently yields Bold. fontTools can split the
collection, so the four faces actually used are written out as standalone .ttf
files next to the figures.

Run once; build_cell_pdf.py reads fonts/ and falls back to the base-14 fonts if
the directory is absent.
"""
import os

from fontTools.ttLib import TTCollection

SRC = '/System/Library/Fonts/Avenir Next.ttc'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
# face index -> filename. Demi Bold rather than Bold for headings and caption
# titles: at caption sizes full Bold is heavier than Cell Press display type.
WANT = {7: 'AvenirNext-Regular.ttf',
        2: 'AvenirNext-DemiBold.ttf',
        0: 'AvenirNext-Bold.ttf',
        4: 'AvenirNext-Italic.ttf'}


def main():
    if not os.path.exists(SRC):
        raise SystemExit(f'{SRC} not found -- Avenir Next is a macOS system font')
    os.makedirs(OUT, exist_ok=True)
    coll = TTCollection(SRC)
    for idx, name in WANT.items():
        path = os.path.join(OUT, name)
        coll.fonts[idx].save(path)
        got = coll.fonts[idx]['name'].getDebugName(4)
        print(f'  [{idx:2d}] {got:26s} -> {path} ({os.path.getsize(path)//1024} KB)')
    print(f'\n{len(WANT)} faces written to {OUT}')


if __name__ == '__main__':
    main()
