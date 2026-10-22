#!/usr/bin/env python3
"""Render the AnchorDynamics markdown documents to PDF, one file each.

Markdown -> styled HTML (python-markdown) -> PDF (headless Chrome). Chrome is
used because it is already on this machine and lays out CSS properly; pandoc and
a LaTeX toolchain are not installed, and the macOS `cupsfilter` route renders
tables and code blocks badly.

Filenames are STABLE by design. Add each PDF to Zotero once as a linked file and
every later export overwrites the same path, so Zotero always opens the current
version without re-importing. A date-stamped copy is also kept under `history/`
so earlier drafts are not lost.

Run daily via the launchd agent in scripts/launchd/, or by hand:
    python3 scripts/export_docs_pdf.py
"""
import datetime as _dt
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO, 'scripts', 'figures', 'AnchorDynamics2026')
DOCS = ['PAPER.md', 'FIGURES.md', 'HYPOTHESES.md']
OUT_DIR = os.path.expanduser('~/Documents/AnchorDynamics_PDFs')
CHROME = ('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')

CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
body { font: 10.5pt/1.5 -apple-system, "Helvetica Neue", Arial, sans-serif;
       color: #1a1a1a; max-width: none; }
h1 { font-size: 19pt; margin: 0 0 2pt; page-break-after: avoid; }
h2 { font-size: 14pt; margin: 20pt 0 6pt; padding-bottom: 3pt;
     border-bottom: 1.5px solid #444; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 15pt 0 4pt; color: #222; page-break-after: avoid; }
h4 { font-size: 10.5pt; margin: 12pt 0 3pt; color: #333; page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }
code { font: 9pt/1.4 "SF Mono", Menlo, monospace; background: #f2f2f4;
       padding: 1px 3px; border-radius: 3px; }
pre { background: #f7f7f9; border: 1px solid #e2e2e6; border-radius: 4px;
      padding: 7pt 9pt; overflow-x: auto; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.5pt; }
blockquote { margin: 8pt 0; padding: 6pt 10pt; background: #fbf7ed;
             border-left: 3px solid #e0a030; page-break-inside: avoid; }
blockquote p { margin: 3pt 0; }
table { border-collapse: collapse; width: 100%; margin: 9pt 0; font-size: 9pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 4pt 6pt; text-align: left;
         vertical-align: top; }
th { background: #eee; font-weight: 600; }
hr { border: none; border-top: 1px solid #ddd; margin: 16pt 0; }
a { color: #1a5490; text-decoration: none; }
.meta { color: #777; font-size: 8.5pt; margin: 0 0 14pt;
        border-bottom: 1px solid #ddd; padding-bottom: 8pt; }
.toc { background: #f7f7f9; border: 1px solid #e2e2e6; border-radius: 4px;
       padding: 8pt 12pt; margin-bottom: 14pt; font-size: 9pt; }
.toc ul { margin: 3pt 0; padding-left: 16pt; }
.toc > ul { padding-left: 0; list-style: none; }
"""


def render(md_path, out_pdf):
    import markdown
    text = open(md_path, encoding='utf-8').read()
    # strip a leading H1 so it is not duplicated by the header we add
    body_md = re.sub(r'\A#\s+(.+?)\n', '', text)
    title = (re.match(r'\A#\s+(.+)', text).group(1)
             if re.match(r'\A#\s+', text) else os.path.basename(md_path))
    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc',
                                       'sane_lists', 'attr_list', 'footnotes'])
    body = md.convert(body_md)
    stamp = _dt.datetime.now().strftime('%Y-%m-%d %H:%M')
    mtime = _dt.datetime.fromtimestamp(os.path.getmtime(md_path)).strftime('%Y-%m-%d %H:%M')
    words = len(re.findall(r'\b\w+\b', body_md))
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>{CSS}</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="meta">{html.escape(os.path.basename(md_path))} &middot; last edited {mtime}
 &middot; {words:,} words &middot; exported {stamp}</p>
<div class="toc">{md.toc}</div>
{body}</body></html>"""

    tmp = tempfile.NamedTemporaryFile('w', suffix='.html', delete=False, encoding='utf-8')
    tmp.write(doc); tmp.close()
    profile = tempfile.mkdtemp(prefix='chrome-pdf-')
    if os.path.exists(out_pdf):
        os.unlink(out_pdf)
    # Chrome writes the PDF and then does NOT exit on this machine (it sits open
    # emitting CVDisplayLink errors), so subprocess.run would block until timeout
    # even though the file is already complete. Poll for the output instead and
    # terminate once its size has stopped changing.
    proc = subprocess.Popen(
        [CHROME, '--headless=new', '--disable-gpu', '--no-sandbox',
         f'--user-data-dir={profile}', '--no-pdf-header-footer',
         '--disable-extensions', '--disable-background-networking',
         '--virtual-time-budget=10000',
         f'--print-to-pdf={out_pdf}', f'file://{tmp.name}'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        size, stable, waited = -1, 0, 0.0
        while waited < 120:
            if proc.poll() is not None and os.path.exists(out_pdf):
                break
            cur = os.path.getsize(out_pdf) if os.path.exists(out_pdf) else -1
            stable = stable + 1 if (cur == size and cur > 0) else 0
            size = cur
            if stable >= 3:                    # unchanged for ~1.5 s: finished
                break
            time.sleep(0.5); waited += 0.5
        if not os.path.exists(out_pdf) or os.path.getsize(out_pdf) < 1000:
            raise RuntimeError(f'chrome produced no usable PDF after {waited:.0f}s')
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        os.unlink(tmp.name)
        shutil.rmtree(profile, ignore_errors=True)
    return words


def main():
    if not os.path.exists(CHROME):
        sys.exit(f'Chrome not found at {CHROME}')
    hist = os.path.join(OUT_DIR, 'history')
    os.makedirs(hist, exist_ok=True)
    today = _dt.date.today().isoformat()
    ok = True
    for name in DOCS:
        src = os.path.join(SRC_DIR, name)
        if not os.path.exists(src):
            print(f'  SKIP {name}: not found'); ok = False; continue
        stem = os.path.splitext(name)[0]
        out = os.path.join(OUT_DIR, f'AnchorDynamics_{stem}.pdf')
        try:
            words = render(src, out)
        except Exception as e:
            print(f'  FAIL {name}: {type(e).__name__}: {e}'); ok = False; continue
        shutil.copy2(out, os.path.join(hist, f'{stem}_{today}.pdf'))
        print(f'  {name:16s} -> {out}  ({words:,} words, '
              f'{os.path.getsize(out)/1024:.0f} KB)')
    print(('done' if ok else 'completed with errors'), _dt.datetime.now().isoformat(timespec='seconds'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
