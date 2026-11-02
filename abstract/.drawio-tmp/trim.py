"""Crop each figure SVG's viewBox to its drawn content (bbox from the 300 dpi PNG) plus a small margin."""
import re
from pathlib import Path
from PIL import Image, ImageChops
FIG = Path(__file__).resolve().parent.parent / 'figures'
for n in ['fig1-architecture', 'fig2-pipeline', 'fig3-sop', 'fig4-navgrid']:
    im = Image.open(FIG / f'{n}.png').convert('RGB')
    bb = ImageChops.difference(im, Image.new('RGB', im.size, (255, 255, 255))).getbbox()
    svg = (FIG / f'{n}.svg').read_text()
    vb = re.search(r'viewBox="([-\d.]+) ([-\d.]+) [-\d.]+ [-\d.]+"', svg)
    ox, oy = float(vb[1]), float(vb[2])
    k, pad = 3.125, 6
    x0, y0, x1, y1 = [v / k for v in bb]
    x0, y0, w, h = ox + x0 - pad, oy + y0 - pad, round(x1 - x0 + 2 * pad), round(y1 - y0 + 2 * pad)
    svg = re.sub(r'width="\d+" height="\d+" viewBox="[^"]*"', f'width="{w}" height="{h}" viewBox="{x0:.1f} {y0:.1f} {w} {h}"', svg, count=1)
    (FIG / f'{n}.svg').write_text(svg)
    print(n, w, h)
