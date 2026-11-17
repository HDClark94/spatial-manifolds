"""Compile the manuscript as a Nature Article mock-up.

The text, figures, captions and reference list are those of the Cell mock-up
(build_cell_pdf.py), which this script imports rather than copies, so the two
documents cannot drift apart. Only the layout differs. It follows a published
Nature Article (Chandra et al., Nature 638, 739-751, 2025), measured from its
PDF:

  page        595 x 791 pt; hairline vertical rule at x = 34 down every page,
              header rule at y = 45 under a running "Article" set in the title
              face; folio at the foot, outside edge, page number in bold
  opener      "Article", the title in a black display serif across the page;
              article details stacked in a narrow left column between hairlines;
              authors and the summary paragraph (9 pt on 13) to their right;
              affiliations as a 6 pt footnote above a hairline
  body        two columns of 255 pt, 10 pt gutter; text serif 8.3 pt on 10.8,
              justified, paragraphs marked by a first-line indent and no space;
              unnumbered section heads in bold 9.8 pt; no "Introduction" or
              "Results" heads
  figures     across the page at the top, legend beneath in two columns of
              7 pt: "Fig. 1 | Title." in bold, panels as bold lowercase "a,"
  end matter  "Online content", numbered references in a 6 pt sans in two
              columns, then Methods on a new page, then each Extended Data
              figure on a page of its own with its legend beneath

Nature's faces, Harding Text and Graphik, are commercial; Charter and Helvetica
Neue stand in for them (extract_nature_fonts.py). Figure and panel references
are rewritten to Nature's form on the way in (Figure 2G -> Fig. 2g, Figure S5 ->
Extended Data Fig. 5); the sources keep Cell's form. Panel letters inside the
figure PDFs are still Cell's capitals.

Writes AnchorDynamics2026_Nature_mockup.pdf
"""
import datetime
import os
import re
import sys

import pymupdf
import pyphen

FIG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FIG)
import build_cell_pdf as C  # noqa: E402

OUT = f'{FIG}/AnchorDynamics2026_Nature_mockup.pdf'
PW, PH = 595.28, 790.87
LX, RX = 40.0, 561.0
GUT = 10.0
COLW = (RX - LX - GUT) / 2
TOP, BOT = 64.0, 742.0                     # first baseline, last baseline
BODY, LEAD = 8.3, 10.8
INDENT = 8.0
CAP, CAP_LEAD = 7.0, 9.3
HEAD, HEAD_LEAD = 9.8, 12.5
HAIR = .35
HYPH = pyphen.Pyphen(lang='en_GB')         # Nature hyphenates; justified text needs it

_FD = os.path.join(FIG, 'fonts')
FONTS = {'cr': 'Charter-Roman.ttf', 'ci': 'Charter-Italic.ttf', 'cb': 'Charter-Bold.ttf',
         'cbi': 'Charter-BoldItalic.ttf', 'ck': 'Charter-Black.ttf',
         'hr': 'HelveticaNeue-Regular.ttf', 'hm': 'HelveticaNeue-Medium.ttf',
         'hb': 'HelveticaNeue-Bold.ttf', 'hi': 'HelveticaNeue-Italic.ttf'}
FONTS = {k: os.path.join(_FD, v) for k, v in FONTS.items()}
if not all(os.path.exists(f) for f in FONTS.values()):
    raise SystemExit('fonts missing: run extract_nature_fonts.py first')
_M = {k: pymupdf.Font(fontfile=f) for k, f in FONTS.items()}
FAMILY = {'serif': {(0, 0): 'cr', (0, 1): 'ci', (1, 0): 'cb', (1, 1): 'cbi'},
          'sans': {(0, 0): 'hr', (0, 1): 'hi', (1, 0): 'hb', (1, 1): 'hb'}}

# inline style markers, carried inside words through wrapping
SUP_ON, SUP_OFF = C.SUP_ON, C.SUP_OFF
B_ON, B_OFF, I_ON, I_OFF = '\x03', '\x04', '\x05', '\x06'
SUP_SCALE, SUP_RISE = .66, .32

FOOT = 'Nature  |  Manuscript draft  |  2026'
DETAILS = ['Clark et al., manuscript draft', 'Nature-style mock-up',
           f'Compiled: {datetime.date.today():%-d %B %Y}', 'Not a published article']
AUTHORS_N = ('Harry Clark\x011\x02, Wolf de Wulf\x011\x02, Chris Halcrow\x011\x02 & '
             'Matthew F. Nolan\x011,2\x02')
AFFIL_N = ('\x011\x02Centre for Discovery Brain Sciences, University of Edinburgh, Edinburgh, '
           'UK. \x012\x02Simons Initiative for the Developing Brain, University of Edinburgh, '
           'Edinburgh, UK. e-mail: mattnolan@ed.ac.uk')
ONLINE = ('Any methods, additional references, Nature Portfolio reporting summaries, source '
          'data, extended data, supplementary information, acknowledgements, peer review '
          'information; details of author contributions and competing interests; and '
          'statements of data and code availability are available with the online version '
          'of this article.')
AI_DECL = C.__dict__.get('AI_DECLARATION') or (
    "During the preparation of this work the authors used Claude (Anthropic) to write and "
    "debug analysis code, to compute the statistics reported here from the authors' own "
    "recordings, to generate the figures, and to draft and revise manuscript text. The "
    "scientific questions, experimental design, data collection and analysis decisions were "
    "the authors'. Every reported result was produced by code that the authors reviewed, and "
    "the authors reviewed and edited all generated text. The authors take full "
    "responsibility for the content of this publication.")


# ── text: Cell's conventions to Nature's ────────────────────────────────────
def _panels(s, sep=','):
    """'D-F' -> 'd-f' (en dash), 'B, C' -> 'b,c'."""
    s = re.sub(r'\s*[-–]\s*', '–', s.strip())
    return re.sub(r'\s*,\s*', sep, s).lower()


def nature_refs(t):
    """Figure references in Nature's form: Fig. 2g, Figs. 2 and 3, Extended Data
    Fig. 5e. A trailing run of single capitals (', F') belongs to the same figure."""
    tail = r'((?:[A-Z](?![a-z])(?:\s*[-–]\s*[A-Z](?![a-z]))?)(?:\s*,\s*[A-Z](?![a-zA-Z])(?:\s*[-–]\s*[A-Z](?![a-z]))?)*)?'

    def one(m):
        supp, num, pan = m.group(1), m.group(2), m.group(3)
        lab = f'Extended Data Fig. {num}' if supp else f'Fig. {num}'
        return lab + (_panels(pan) if pan else '')
    t = re.sub(r'\bFigures\b', 'Figs.', t)
    t = re.sub(r'\bFigure\s+(S?)(\d+)' + tail, one, t)
    # the bare items of a list: "Figs. 2, S3 and S4"
    t = re.sub(r'(?<=, )(S)(\d+)' + tail, one, t)
    t = re.sub(r'(?<=and )(S)(\d+)' + tail, one, t)
    t = re.sub(r'(?<=Figs\. )(S)(\d+)' + tail, one, t)
    return t


def bold_words(t):
    return ' '.join(B_ON + w + B_OFF for w in t.split())


def caption_panels(t):
    """'(A) text (B, C) text' -> bold 'a, text b,c, text', Nature's legend form."""
    return re.sub(r'(^|(?<=[.;:)] ))\(([A-Z](?:\s*[-–,]\s*[A-Z])*)\)\s+',
                  lambda m: m.group(1) + B_ON + _panels(m.group(2)) + B_OFF + ', ', t)


# Reference numbers, in order of first citation: the main text and main-figure
# legends first, then the Methods and Extended Data legends, as Nature numbers
# them. Filled by number_references() before anything is set.
NUM = {}


def number_references(main_texts, methods_texts):
    order = []
    for t in main_texts:
        order += [k for k in C.cite_keys(t) if k not in order]
    n_main = len(order)
    for t in methods_texts:
        order += [k for k in C.cite_keys(t) if k not in order]
    C.check_citations(main_texts + methods_texts)
    NUM.clear(); NUM.update({k: i for i, k in enumerate(order, 1)})
    return order[:n_main], order[n_main:]


def _sup_numbers(keys):
    """[3, 4, 5, 9] -> '3–5,9' as a superscript run."""
    ns = sorted({NUM[k] for k in keys})
    parts, i = [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        parts.append(f'{ns[i]}–{ns[j]}' if j - i >= 2 else
                     ','.join(map(str, ns[i:j + 1])))
        i = j + 1
    return SUP_ON + ','.join(parts) + SUP_OFF


def cite_numbers(t):
    """Author-year citations to Nature's superscript numbers: ' (Kropff et al.,
    2015; Ye et al., 2018)' becomes a superscript on the word before, and
    'Hardcastle et al. (2017)' becomes 'Hardcastle et al.' with one."""
    t = re.sub(r'\s*' + C.CITE_PAREN.pattern,
               lambda m: _sup_numbers([C.cite_key(a, y) for a, y in
                                       C.CITE_ONE.findall(m.group(1))]), t)
    return C.CITE_NARR.sub(lambda m: m.group(1) + _sup_numbers([C.cite_key(m.group(1),
                                                                            m.group(2))]), t)


def clean(t, cites=True):
    """Markdown to style markers; unicode kept where Charter has the glyph."""
    if cites:
        t = cite_numbers(t)
    t = re.sub(r'\*\*(.+?)\*\*', lambda m: bold_words(m.group(1)), t)
    t = re.sub(r'(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)',
               lambda m: ' '.join(I_ON + w + I_OFF for w in m.group(1).split()), t)
    t = t.replace('`', '')
    # Charter has no subscript digits; set them as plain digits (θ₁ -> θ1)
    t = t.translate(str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789'))
    t = re.sub(r'([A-Za-z0-9])([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)',
               lambda m: m.group(1) + SUP_ON + ''.join(C.SUPS.get(c, c) for c in m.group(2))
               + SUP_OFF, t)
    t = re.sub(r'\^\{?(-?\d+)\}?', lambda m: SUP_ON + m.group(1).replace('-', '−') + SUP_OFF, t)
    for a, b in {'→': '->', '↔': '<->', '∼': '~'}.items():
        t = t.replace(a, b)
    t = re.sub(r'(?<=\d)-(?=\d)', '–', t)              # ranges take an en dash
    t = t.replace('STAR Methods', 'Methods')
    t = re.sub(r'(?<![\w.])(\d+(?:\.\d+)?)e-0*(\d+)\b',
               lambda m: f'{m.group(1)} × 10{SUP_ON}−{m.group(2)}{SUP_OFF}', t)
    t = t.replace('<=', '≤').replace('>=', '≥').replace('+/-', '±')
    # Nature sets statistics as italic P and n
    t = re.sub(r'(?<![\w\x05])p(?=\s*[=<>≤≥]\s*[\d.])', I_ON + 'P' + I_OFF, t)
    t = re.sub(r'(?<![\w\x05])n(?=\s*=\s*\d)', I_ON + 'n' + I_OFF, t)
    return nature_refs(t)


def word_runs(words):
    """Runs for a line of words, with bold/italic state carried across words."""
    state = [0, 0, 0]
    out = []
    for w in words:
        rr, buf = [], ''
        for ch in w:
            if ch in (SUP_ON, SUP_OFF, B_ON, B_OFF, I_ON, I_OFF):
                if buf:
                    rr.append((buf, *state)); buf = ''
                if ch == SUP_ON: state[0] = 1
                elif ch == SUP_OFF: state[0] = 0
                elif ch == B_ON: state[1] = 1
                elif ch == B_OFF: state[1] = 0
                elif ch == I_ON: state[2] = 1
                elif ch == I_OFF: state[2] = 0
            else:
                buf += ch
        if buf:
            rr.append((buf, *state))
        out.append(rr)
    return out


def face(family, bold, ital, base=None):
    if base and not bold and not ital:
        return base
    return FAMILY[family][(bold, ital)]


def rlen(rr, family, size, base=None):
    return sum(_M[face(family, b, i, base)].text_length(t, size * (SUP_SCALE if s else 1))
               for t, s, b, i in rr)


def nature_ref_entry(r):
    """'Authors (2017). Title. Journal 94, 375-387.' in Nature's order, with the
    journal italic and the volume bold."""
    m = re.match(r'(?P<a>.+?) \((?P<y>\d{4}[a-z]?)\)\. (?P<rest>.+)$', r)
    if not m:
        return r
    a = m.group('a').replace(', and ', ' & ').replace(' and ', ' & ')
    a = re.sub(r'([A-Z])\.([A-Z])', r'\1. \2', a)
    rest = m.group('rest').rstrip('.')
    v = re.match(r'(?P<tj>.+) (?P<v>\d+), (?P<p>[\w–-]+)$', rest)
    if v:
        tj = v.group('tj')
        k = tj.find('. ')
        title, jour = (tj[:k + 1], tj[k + 2:]) if k > 0 else (tj, '')
        jour = ' '.join(I_ON + w + I_OFF for w in jour.split())
        return (f'{a} {title} {jour} {B_ON}{v.group("v")}{B_OFF}, '
                f'{v.group("p").replace("-", "–")} ({m.group("y")}).')
    return f'{a} {rest} ({m.group("y")}).'


# ── layout ──────────────────────────────────────────────────────────────────
class Doc:
    def __init__(self):
        self.d = pymupdf.open()
        self.pending = []
        self.full = False
        self.new_page()

    # pages
    def new_page(self):
        self.page = self.d.new_page(width=PW, height=PH)
        for n, f in FONTS.items():
            self.page.insert_font(fontname=n, fontfile=f)
        n = self.d.page_count
        p = self.page
        p.draw_line((34, 28), (34, 744), color=(0, 0, 0), width=HAIR)
        p.insert_text((LX, 39), 'Article', fontname='ck', fontsize=14)
        recto = n % 2 == 1
        p.draw_line((0 if recto else LX, 45), (RX if recto else PW, 45), color=(0, 0, 0), width=HAIR)
        num = str(n)
        if recto:
            w = _M['hr'].text_length(FOOT + '  |  ', 8) + _M['hb'].text_length(num, 8)
            p.insert_text((RX - w, 768), FOOT + '  |  ', fontname='hr', fontsize=8)
            p.insert_text((RX - _M['hb'].text_length(num, 8), 768), num, fontname='hb', fontsize=8)
        else:
            p.insert_text((LX, 768), num, fontname='hb', fontsize=8)
            p.insert_text((LX + _M['hb'].text_length(num, 8), 768), '  |  ' + FOOT,
                          fontname='hr', fontsize=8)
        self.col, self.col_top, self.y, self.bot = 0, TOP, TOP, BOT
        self.first_para = True

    def x_of(self):
        return LX + self.col * (COLW + GUT)

    def next_col(self):
        if self.col == 0:
            self.col, self.y = 1, self.col_top
        else:
            self.new_page()
            self.flush_pending()

    # text
    def wrap(self, text, family, size, width, base=None, first_indent=0.0):
        words = text.split()
        rrs = word_runs(words)
        sp = _M[FAMILY[family][(0, 0)]].text_length(' ', size)
        lines, cur, cw = [], [], 0.0
        queue = list(zip(words, rrs))
        while queue:
            w, rr = queue.pop(0)
            ww = rlen(rr, family, size, base)
            lim = width - (first_indent if not lines else 0)
            if cur and cw + sp + ww > lim:
                # hyphenate a plain word across the break if a piece of it fits
                if len(rr) == 1 and not any(rr[0][1:]) and re.fullmatch(r'[A-Za-z]{6,}[,.;:)]?', w):
                    core = w.rstrip(',.;:)')
                    for a, b in HYPH.iterate(core):
                        if len(a) < 3 or len(b) < 3:
                            continue
                        head = [(a + '-', 0, 0, 0)]
                        if cw + sp + rlen(head, family, size, base) <= lim:
                            cur.append((a + '-', head)); lines.append(cur)
                            rest = b + w[len(core):]
                            cur, cw = [], 0.0
                            queue.insert(0, (rest, [(rest, 0, 0, 0)]))
                            break
                    else:
                        lines.append(cur); cur, cw = [(w, rr)], ww
                    continue
                lines.append(cur); cur, cw = [(w, rr)], ww
            else:
                cw += (sp if cur else 0) + ww
                cur.append((w, rr))
        if cur:
            lines.append(cur)
        return lines

    def draw(self, line, x, y, width, family, size, base=None, justify=False, color=(0, 0, 0)):
        sp = _M[FAMILY[family][(0, 0)]].text_length(' ', size)
        nat = sum(rlen(rr, family, size, base) for _, rr in line) + sp * (len(line) - 1)
        extra = 0.0
        if justify and len(line) > 1 and 0 < width - nat < width / 3:
            extra = (width - nat) / (len(line) - 1)
        for _, rr in line:
            for t, s, b, i in rr:
                f = face(family, b, i, base)
                fs = size * (SUP_SCALE if s else 1)
                self.page.insert_text((x, y - (size * SUP_RISE if s else 0)), t, fontname=f,
                                      fontsize=fs, color=color)
                x += _M[f].text_length(t, fs)
            x += sp + extra

    def para(self, text, family='serif', size=BODY, lead=LEAD, indent=None, justify=True,
             base=None, gap=0.0, color=(0, 0, 0)):
        if indent is None:
            indent = 0.0 if self.first_para else INDENT
        lines = self.wrap(clean(text), family, size, COLW, base, indent)
        for k, ln in enumerate(lines):
            if self.y > self.bot:
                self.next_col()
            ind = indent if k == 0 else 0.0
            self.draw(ln, self.x_of() + ind, self.y, COLW - ind, family, size, base,
                      justify and k < len(lines) - 1, color)
            self.y += lead
        self.y += gap
        self.first_para = False

    def heading(self, text, size=HEAD, lead=HEAD_LEAD, pre=7.0):
        if self.y + pre + lead + 3 * LEAD > self.bot:
            self.next_col()
        elif self.y > self.col_top + 1:
            self.y += pre
        for ln in self.wrap(clean(text), 'serif', size, COLW, base='cb'):
            self.draw(ln, self.x_of(), self.y + 2, COLW, 'serif', size, base='cb')
            self.y += lead
        self.y += 1.5
        self.first_para = True

    # figures
    def legend_lines(self, label, title, caption, width):
        title = re.sub(r',\s*related to .*$', '', title)
        txt = (bold_words(f'{label} |') + ' ' + bold_words(clean(title).rstrip('.') + '.')
               + ' ' + clean(caption_panels(caption)))
        return self.wrap(txt, 'serif', CAP, width)

    def show_full_width(self, src, y0):
        """The figure at the full text width, shrunk only if taller than the page.
        Returns its bottom."""
        r = src[0].rect
        w = RX - LX
        h = w * r.height / r.width
        if h > BOT - y0:
            h = BOT - y0
            w = h * r.width / r.height
        x = LX + (RX - LX - w) / 2
        self.page.show_pdf_page(pymupdf.Rect(x, y0, x + w, y0 + h), src, 0)
        return y0 + h

    def flow_legend(self, lines, y, ncol):
        """Set legend lines in ncol columns from y, balanced on the page where it
        ends and continued on fresh pages when it runs past the foot. Returns the
        y under the last line."""
        k0, n = 0, len(lines)
        while k0 < n:
            per = int((BOT - y) // CAP_LEAD) + 1
            left = n - k0
            if per < 2:                             # no room left here: new page
                self.new_page()
                y = TOP
                continue
            rows = -(-left // ncol) if left <= ncol * per else per
            for c in range(ncol):
                for j in range(rows):
                    k = k0 + c * rows + j
                    if k >= n:
                        break
                    self.draw(lines[k], LX + c * (COLW + GUT) if ncol > 1 else LX,
                              y + j * CAP_LEAD, COLW if ncol > 1 else RX - LX, 'serif', CAP,
                              justify=k < n - 1)
            k0 += ncol * rows
            if k0 < n:
                self.new_page()
                y = TOP
            else:
                y += rows * CAP_LEAD
        return y

    def place_figure(self, label, src, title, caption, top=None):
        """Figure across the full text width at the top of the page, legend
        beneath in two columns and on to the next page if it must; the body text
        resumes under the legend."""
        y0 = TOP - 8 if top is None else top
        y = self.show_full_width(src, y0) + 12
        y = self.flow_legend(self.legend_lines(label, title, caption, COLW), y, 2)
        self.col_top = self.y = y + 10
        self.col = 0
        self.first_para = False
        if self.y + 3 * LEAD > BOT:                 # too little room for text here
            self.new_page()

    def flush_pending(self):
        if self.pending and self.col == 0 and self.y <= self.col_top + 1:
            lab, path, ttl, cap = self.pending.pop(0)
            src = pymupdf.open(path)
            self.place_figure(lab, src, ttl, cap)
            src.close()

    def extended_data(self, label, path, title, caption):
        """One Extended Data figure per page at the full text width, its legend
        beneath across the page and on to the next page if it must."""
        self.new_page()
        src = pymupdf.open(path)
        y = self.show_full_width(src, TOP - 8) + 12
        self.flow_legend(self.legend_lines(label, title, caption, RX - LX), y, 1)
        src.close()


def ref_list(doc, keys):
    """Numbered references in 6 pt sans, as Nature sets them."""
    for key in keys:
        lines = doc.wrap(clean(nature_ref_entry(C.REFS[key]), cites=False), 'sans', 6,
                         COLW - 11)
        if doc.y + 8 * len(lines) > doc.bot:
            doc.next_col()
        doc.page.insert_text((doc.x_of(), doc.y), f'{NUM[key]}.', fontname='hr', fontsize=6)
        for ln in lines:
            doc.draw(ln, doc.x_of() + 11, doc.y, COLW - 11, 'sans', 6)
            doc.y += 8
        doc.y += 1


def opener(doc, summary):
    """Page 1: Article, title, details, authors, summary, affiliations."""
    p = doc.page
    y = 74.0
    for ln in doc.wrap(C.TITLE, 'serif', 27, RX - LX, base='ck'):
        doc.draw(ln, LX, y, RX - LX, 'serif', 27, base='ck')
        y += 28.5
    rule = max(141.0, y + 18)
    p.draw_line((LX, rule), (RX, rule), color=(0, 0, 0), width=HAIR)
    for k, t in enumerate(DETAILS):
        yy = rule + 11 + 16 * k
        p.insert_text((LX, yy), t, fontname='hm', fontsize=8)
        p.draw_line((LX, yy + 5), (206, yy + 5), color=(0, 0, 0), width=HAIR)
    ax = 217.0
    for ln in doc.wrap(AUTHORS_N, 'sans', 8, RX - ax, base='hb'):
        doc.draw(ln, ax, rule + 11, RX - ax, 'sans', 8, base='hb')
    p.draw_line((ax, rule + 32), (RX, rule + 32), color=(0, 0, 0), width=HAIR)
    y = rule + 46
    lines = doc.wrap(clean(summary), 'serif', 9, RX - ax)
    for k, ln in enumerate(lines):
        doc.draw(ln, ax, y, RX - ax, 'serif', 9, justify=False)
        y += 13
    y = max(y, rule + 16 * len(DETAILS)) + 2
    p.draw_line((LX, y), (RX, y), color=(0, 0, 0), width=HAIR)
    # affiliations footnote
    fl = doc.wrap(AFFIL_N, 'sans', 6, RX - LX)
    fy = 740 - 8 * (len(fl) - 1)
    p.draw_line((LX, fy - 9), (RX, fy - 9), color=(0, 0, 0), width=HAIR)
    for ln in fl:
        doc.draw(ln, LX, fy, RX - LX, 'sans', 6)
        fy += 8
    doc.col_top = doc.y = y + 14
    doc.bot = 740 - 8 * len(fl) - 14
    doc.col = 0
    doc.first_para = True


def main():
    blocks = C.parse(open(C.SRC).read())
    summary = next(t for k, t in blocks if k == 'p')
    doc = Doc()
    opener(doc, summary)

    figs = list(C.FIGURES)
    meth = C.parse_methods(f'{FIG}/PAPER.md')
    main_refs, meth_refs = number_references(
        [t for _, t in blocks] + [' '.join(map(str, f[2:])) for f in figs],
        [t for _, t in meth] + [' '.join(map(str, f[2:])) for f in C.SUPPLEMENTS])
    pinned = {'Figure 10': 'ideas and speculation'}
    skip_h1 = ('summary', 'introduction', 'results')
    seen_summary, in_decl = False, False
    for kind, text in blocks:
        if kind == 'p' and not seen_summary and text == summary:
            seen_summary = True
            continue
        if kind == 'h1':
            in_decl = text.lower().startswith('declaration')
            if in_decl or text.lower() in skip_h1:
                continue
            doc.heading(text)
            continue
        if in_decl:
            continue
        if kind == 'h2':
            doc.heading(text)
            for i, (lab, path, ttl, cap) in enumerate(figs):
                want = pinned.get(lab)
                if want and want not in text.lower():
                    continue
                if os.path.exists(f'{FIG}/{path}'):
                    doc.pending.append((lab.replace('Figure', 'Fig.'), f'{FIG}/{path}', ttl, cap))
                    figs.pop(i)
                break
        else:
            doc.para(text)
    for lab, path, ttl, cap in figs:
        if os.path.exists(f'{FIG}/{path}'):
            doc.pending.append((lab.replace('Figure', 'Fig.'), f'{FIG}/{path}', ttl, cap))
    while doc.pending:
        doc.new_page()
        doc.flush_pending()

    # online content and references
    doc.heading('Online content')
    doc.para(ONLINE)
    doc.heading('References')
    ref_list(doc, main_refs)

    # Methods
    doc.new_page()
    doc.y = TOP + 6
    doc.draw(doc.wrap('Methods', 'serif', 12, COLW, base='cb')[0], LX, TOP + 4, COLW,
             'serif', 12, base='cb')
    doc.col_top = doc.y = TOP + 22
    doc.first_para = True
    for kind, text in meth:
        if kind == 'h1':
            continue
        if kind == 'h2':
            doc.heading(text, size=8.5, lead=11, pre=6)
        else:
            doc.para(text)
    for h, t in (('Data availability', '[To be completed.]'),
                 ('Code availability', '[To be completed.]'),
                 ('Use of generative AI', AI_DECL),
                 ('Author contributions', '[To be completed.]'),
                 ('Competing interests', 'The authors declare no competing interests. '
                  '[To be confirmed.]')):
        doc.heading(h, size=8.5, lead=11, pre=6)
        doc.para(t)
    if meth_refs:                     # numbering continues from the main list
        doc.heading('Methods references', size=8.5, lead=11, pre=6)
        doc.y += 2
        ref_list(doc, meth_refs)
    print(f'  Methods: {sum(1 for k, _ in meth if k == "h2")} subsections')

    # Extended Data
    n = 0
    for lab, path, ttl, cap in C.SUPPLEMENTS:
        if not os.path.exists(f'{FIG}/{path}'):
            print(f'  ! {path} missing, skipped')
            continue
        doc.extended_data(lab.replace('Figure S', 'Extended Data Fig. '), f'{FIG}/{path}', ttl, cap)
        n += 1
    print(f'  Extended Data figures: {n} (Nature allows 10)')
    doc.d.save(OUT, deflate=True, garbage=3)
    print(f'wrote {OUT}  ({doc.d.page_count} pages)')
    import shrink_pdf                     # under 25 MB, and a smaller copy
    shrink_pdf.fit(OUT)


if __name__ == '__main__':
    main()
