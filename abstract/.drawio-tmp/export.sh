#!/bin/bash
# Export each trimmed SVG to a vector PDF (Chrome print) and a 300 dpi PNG (3.125x screenshot of 96 dpi CSS px).
cd "$(dirname "$0")/../figures"
C="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"; T=$(mktemp -d)
for n in fig1-architecture fig2-pipeline fig3-sop fig4-navgrid; do
  read W H < <(python3 -c "import re;s=open('$n.svg').read(800);m=re.search(r'width=\"(\d+)\" height=\"(\d+)\"',s);print(m[1],m[2])")
  printf '<!doctype html><html><head><style>@page{size:%spx %spx;margin:0}html,body{margin:0}img{display:block;width:%spx;height:%spx}</style></head><body><img src="%s"></body></html>' $W $H $W $H "file://$PWD/$n.svg" > $T/$n.html
  "$C" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf=$n.pdf "file://$T/$n.html" 2>/dev/null
  "$C" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=3.125 --window-size=$W,$H --screenshot=$n.png "file://$T/$n.html" 2>/dev/null
done
