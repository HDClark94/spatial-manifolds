"""The manuscript as an editable Word file, for co-authors to edit in Google Docs.

Same content as the Cell mock-up (build_cell_pdf.py), taken from the same
sources so the two cannot drift: title matter, Highlights and In brief, the
MANUSCRIPT.md body, STAR Methods from PAPER.md, the AI declaration, references,
and every main and supplemental figure with its caption. Layout is left to the
word processor: one column, plain styles, so it converts cleanly on upload.

Google Docs cannot hold a vector figure, so each figure is rendered to PNG at
DPI. A main figure is placed after the paragraph that first cites it (at the
end of the Results if nothing does); supplemental figures follow the references.
The PNGs and the vector PDFs are also written to google_doc/figures, for a
Drive folder beside the document.

  python build_docx.py

Writes google_doc/AnchorDynamics2026_manuscript.docx and google_doc/figures/.
Keep it under Google's 50 MB conversion limit.
"""
import os
import re
import shutil

import pymupdf
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Inches, Pt

import build_cell_pdf as B

OUTD = f'{B.FIG}/google_doc'
OUT = f'{OUTD}/AnchorDynamics2026_manuscript.docx'
DPI = 200
MAXW, MAXH = 6.5, 8.6            # inches, inside A4 at 2 cm margins with a caption
# the PDF builders spell the address with S for @ (base-14 font workaround)
EMAIL = 'mattnolan@ed.ac.uk'

INLINE = re.compile(r'(\*\*.+?\*\*|(?<!\*)\*(?!\s).+?(?<!\s)\*(?!\*)|`[^`]+`'
                    r'|\^\{?-?[\d,*]+\}?)')


def add_runs(par, text, size=None, bold=False, italic=False):
    """Markdown bold/italic/code and ^superscripts as Word runs."""
    for tok in INLINE.split(text):
        if not tok:
            continue
        b, i, sup = bold, italic, False
        if tok.startswith('**') and tok.endswith('**') and len(tok) > 4:
            tok, b = tok[2:-2], True
        elif tok.startswith('*') and tok.endswith('*') and len(tok) > 2:
            tok, i = tok[1:-1], True
        elif tok.startswith('`') and tok.endswith('`'):
            tok = tok[1:-1]
        elif tok.startswith('^'):
            tok, sup = tok.lstrip('^').strip('{}'), True
        r = par.add_run(tok)
        r.bold, r.italic = b or None, i or None
        r.font.superscript = sup or None
        if size:
            r.font.size = Pt(size)
    return par


def render(path, label):
    """Figure PDF -> PNG in figures/ (and the PDF beside it); returns the PNG."""
    stem = re.sub(r'\W+', '_', label).strip('_')
    png = f'{OUTD}/figures/{stem}.png'
    shutil.copyfile(path, f'{OUTD}/figures/{stem}.pdf')
    with pymupdf.open(path) as d:
        d[0].get_pixmap(dpi=DPI, alpha=False).save(png)
    return png


def figure(doc, label, path, title, caption):
    src = f'{B.FIG}/{path}'
    if not os.path.exists(src):
        print(f'  ! {path} missing, skipped')
        return
    png = render(src, label)
    with pymupdf.open(src) as d:
        r = d[0].rect
    w = min(MAXW, MAXH * r.width / r.height)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(png, width=Inches(w))
    cap = doc.add_paragraph()
    add_runs(cap, f'{label}. {title}', size=9, bold=True)
    add_runs(doc.add_paragraph(), caption, size=9)


def main():
    os.makedirs(f'{OUTD}/figures', exist_ok=True)
    blocks = B.parse(open(B.SRC).read())
    meth = B.parse_methods(f'{B.FIG}/PAPER.md')
    cited = B.check_citations([t for _, t in blocks] + [t for _, t in meth] +
                              [' '.join(map(str, f)) for f in B.FIGURES + B.SUPPLEMENTS])

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for side in ('left_margin', 'right_margin', 'top_margin', 'bottom_margin'):
        setattr(sec, side, Cm(2.0))
    doc.styles['Normal'].font.name = 'Arial'
    doc.styles['Normal'].font.size = Pt(11)

    doc.add_heading(B.TITLE, 0)
    add_runs(doc.add_paragraph(), B.AUTHORS)
    for a in B.AFFIL:
        add_runs(doc.add_paragraph(), a.replace('mattnolanSed.ac.uk', EMAIL), size=9)
    doc.add_heading('Highlights', 1)
    for h in B.HIGHLIGHTS:
        add_runs(doc.add_paragraph(style='List Bullet'), h)
    doc.add_heading('In brief', 1)
    add_runs(doc.add_paragraph(), B.IN_BRIEF)

    # the Summary is the first body paragraph; MANUSCRIPT.md gives it no heading
    summary = next(t for k, t in blocks if k == 'p')
    doc.add_heading('Summary', 1)
    add_runs(doc.add_paragraph(), summary)

    figs = list(B.FIGURES)
    placed = set()
    held, hold = [], False          # MANUSCRIPT.md's AI declaration, set after Methods
    for kind, text in blocks:
        if text == summary or (kind == 'h1' and text.lower().startswith('summary')):
            continue
        if kind == 'h1' and text.lower().startswith('declaration'):
            hold = True
        if hold:
            held.append((kind, text))
            continue
        if kind == 'h1':
            doc.add_heading(text, 1)
        elif kind == 'h2':
            doc.add_heading(text, 2)
        else:
            add_runs(doc.add_paragraph(), text)
            # a main figure after the paragraph that first cites it
            for f in figs:
                n = f[0].split()[1]
                if f[0] not in placed and re.search(
                        rf'Figures? (?:[\dS]+[A-Z\-]*,? (?:and )?)*{n}(?![\d])', text):
                    figure(doc, *f)
                    placed.add(f[0])
    # uncited ones (the model, Figure 10) close the Discussion
    for f in figs:
        if f[0] not in placed:
            figure(doc, *f)

    for kind, text in meth + held:
        if kind == 'h1':
            doc.add_heading(text, 1)
        elif kind == 'h2':
            doc.add_heading(text, 2)
        else:
            add_runs(doc.add_paragraph(), text)

    doc.add_heading('References', 1)
    for k in sorted(cited, key=lambda k: B.REFS[k].lower()):
        add_runs(doc.add_paragraph(), B.REFS[k], size=10)

    doc.add_heading('Supplemental figures', 1)
    for f in B.SUPPLEMENTS:
        figure(doc, *f)

    doc.save(OUT)
    n = len(os.listdir(f'{OUTD}/figures')) // 2
    print(f'wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB), {n} figures '
          f'({len(placed)} placed at first citation)')


if __name__ == '__main__':
    main()
