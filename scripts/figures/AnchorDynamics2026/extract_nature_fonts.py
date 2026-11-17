"""Extract the faces the Nature-style mock-up is set in.

Nature sets running text in Harding Text and display matter (author line,
article details, references, running foot) in Graphik. Both are commercial and
not on this machine. The nearest system faces are Charter for Harding Text (a
sturdy transitional text serif of the same kind, with a black weight for the
title) and Helvetica Neue for Graphik (a neutral grotesque).

macOS ships both as TrueType Collections, and PyMuPDF loads one face from a
file -- given a .ttc it takes face 0 -- so the faces used are split out with
fontTools, as extract_avenir.py does for the Cell mock-up.

Run once; build_nature_pdf.py reads fonts/.
"""
import os

from fontTools.ttLib import TTCollection

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
WANT = {
    '/System/Library/Fonts/Supplemental/Charter.ttc': {
        0: 'Charter-Roman.ttf', 1: 'Charter-Italic.ttf', 3: 'Charter-Bold.ttf',
        2: 'Charter-BoldItalic.ttf', 5: 'Charter-Black.ttf'},
    '/System/Library/Fonts/HelveticaNeue.ttc': {
        0: 'HelveticaNeue-Regular.ttf', 10: 'HelveticaNeue-Medium.ttf',
        1: 'HelveticaNeue-Bold.ttf', 2: 'HelveticaNeue-Italic.ttf'},
}


def main():
    os.makedirs(OUT, exist_ok=True)
    for src, faces in WANT.items():
        if not os.path.exists(src):
            raise SystemExit(f'{src} not found -- a macOS system font')
        coll = TTCollection(src)
        for idx, name in faces.items():
            path = os.path.join(OUT, name)
            coll.fonts[idx].save(path)
            print(f'  [{idx:2d}] {coll.fonts[idx]["name"].getDebugName(4):26s} -> {name}')


if __name__ == '__main__':
    main()
