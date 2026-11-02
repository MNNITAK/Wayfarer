"""Build abstract/Technical_Abstract.pdf from abstract/abstract.md.

  python3 abstract/build.py

Markdown -> HTML (Python-Markdown) -> PDF (headless Chrome print, A4, 2 cm margins). Figures are embedded as SVG,
so they stay vector in the PDF. Headings are numbered and figures captioned "Fig. N." by this script.
"""
import html
import re
import subprocess
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

CSS = '''
@page { size: A4; margin: 2cm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font: 11pt/1.36 Charter, "Bitstream Charter", Georgia, serif; color: #111; margin: 0; text-align: justify; hyphens: auto; }
header { text-align: center; margin-bottom: 6pt; }
header h1 { font: bold 15.5pt/1.2 Charter, Georgia, serif; margin: 0 0 3pt; }
header .sub { font-size: 10pt; color: #333; margin: 0 0 3pt; }
header .auth { font-size: 10.5pt; margin: 0; }
.links { text-align: center; font-size: 9pt; margin: 3pt 0 8pt; padding: 4pt 0; border-top: 0.6pt solid #999; border-bottom: 0.6pt solid #999; }
.links p { margin: 0; }
a { color: #1f3f73; text-decoration: none; }
h1, h2 { font-family: Helvetica, Arial, sans-serif; break-after: avoid; text-align: left; }
h1 { font-size: 12.5pt; margin: 9pt 0 3pt; }
h2 { font-size: 11pt; margin: 6pt 0 2pt; }
p { margin: 0 0 4pt; orphans: 3; widows: 3; }
ul, ol { margin: 0 0 4pt; padding-left: 1.5em; }
li { margin: 0 0 1.5pt; }
code { font: 9pt Menlo, "DejaVu Sans Mono", monospace; overflow-wrap: anywhere; font-variant-ligatures: none; }
figure { margin: 6pt 0 8pt; text-align: center; break-inside: avoid; }
figure img { display: block; margin: 0 auto; }
figcaption { font-size: 9pt; line-height: 1.3; text-align: center; margin-top: 3pt; color: #222; }
figcaption b { font-family: Helvetica, Arial, sans-serif; }
.credits code { font-size: 8pt; }
.credits ul { columns: 2; column-gap: 16pt; font-size: 8.6pt; line-height: 1.3; text-align: left; padding-left: 1.1em; }
.credits li { break-inside: avoid; }
'''


def front_matter(text):
    m = re.match(r'---\n(.*?)\n---\n', text, re.S)
    meta = dict(re.findall(r'^(\w+):\s*"(.*)"\s*$', m.group(1), re.M))
    return meta, text[m.end():]


def number_headings(body):
    sec, sub = 0, 0

    def h(m):
        nonlocal sec, sub
        if m[1] == '1':
            sec, sub = sec + 1, 0
            num = f'{sec}'
        else:
            sub += 1
            num = f'{sec}.{sub}'
        return f'<h{m[1]}>{num}&nbsp;&nbsp;{m[2]}</h{m[1]}>'
    return re.sub(r'<h([12])>(.*?)</h\1>', h, body)


def figures(body):
    n = 0

    def f(m):
        nonlocal n
        n += 1
        tag = m[1]
        alt = html.unescape(re.search(r'alt="([^"]*)"', tag)[1])
        cap = re.sub(r'`([^`]+)`', r'<code>\1</code>', html.escape(alt))
        return f'<figure>{tag}<figcaption><b>Fig. {n}.</b> {cap}</figcaption></figure>'
    return re.sub(r'<p>(<img [^>]*>)</p>', f, body)


def main():
    meta, md = front_matter((HERE / 'abstract.md').read_text())
    body = markdown.markdown(md, extensions=['extra', 'attr_list', 'md_in_html', 'sane_lists'])
    body = figures(number_headings(body))
    # let long paths/flags break after separators instead of stretching justified lines
    body = re.sub(r'<code>(.*?)</code>', lambda m: '<code>' + re.sub(r'([/_.](?!>))', r'\1<wbr>', m[1]) + '</code>', body)
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{meta["title"]}</title><style>{CSS}</style></head>'
            f'<body><header><h1>{meta["title"]}</h1><p class="sub">{meta["subtitle"]}</p>'
            f'<p class="auth">{meta["author"]}</p></header>{body}</body></html>')
    out_html = HERE / 'Technical_Abstract.html'
    out_html.write_text(page)
    subprocess.run([CHROME, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                    f'--print-to-pdf={HERE / "Technical_Abstract.pdf"}', out_html.as_uri()],
                   check=True, stderr=subprocess.DEVNULL)
    print('wrote', HERE / 'Technical_Abstract.pdf')


if __name__ == '__main__':
    main()
