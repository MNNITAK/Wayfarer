# Verification checklist: `abstract/Technical_Abstract.pdf`

## Guideline compliance

| Requirement | Where it is met | Status |
|---|---|---|
| Single PDF, ≤ 5 pages incl. figures and credits | `Technical_Abstract.pdf`: 5 pages, A4 (595 × 842 pt), checked with PyMuPDF | ✅ |
| System Architecture | §1 + Fig. 1 (p. 1) | ✅ |
| Pipeline Design | §2 + Fig. 2 (p. 2) | ✅ |
| SOP details: how it is set up, run and operated, as numbered steps | §3, steps 1–8 + Fig. 3 (pp. 2–3) | ✅ |
| Technical Approach | §4.1–4.5 + Fig. 4 (pp. 3–5) | ✅ |
| Credits for every library, tool, API, model, service, dataset | §5, two columns; matches the `notes.md` §8 inventory, including the abstract's own tooling | ✅ |
| Title + team/author line at the top | Title, Team Simpletons, ID TM-A30B5FE384F, 4 members, repo / viewer / video links | ✅ |
| A4, 2 cm margins, 11 pt, numbered headings | `build.py` CSS: `@page { size: A4; margin: 2cm }`, body 11 pt Charter, headings numbered 1–5 / 4.1–4.5 | ✅ |
| Vector figures | 0 raster images in the PDF; figures embedded as SVG | ✅ |
| Figures numbered and referenced in the text | Fig. 1–4, each cited ("Fig. 1", "Fig. 2 shows…") | ✅ |
| Editable diagram sources + vector + 300 dpi PNG exports | `figures/*.drawio`, `*.svg`, `*.pdf`, `*.png` (3.125× = 300 dpi) | ✅ |
| Diagram style: flat, one sans font, muted palette, no gradients/shadows/emoji, real component names, labelled edges, no crossings | C4 Blue palette, Helvetica; CLI reports `edge-crossings=0` on all four | ✅ |
| Built with pandoc + xelatex; draw.io Desktop exports | **Deviation**: these could not run on this Mac (see below); Chrome headless was used | ⚠️ |

## Spot-checked claims

| Claim in the abstract | Source | Result |
|---|---|---|
| Frames sampled at 8 fps | `pipeline/wayfarer.py:224` (`--fps … default=8`) | ✅ |
| Exhaustive matching up to 150 frames | `pipeline/wayfarer.py:30` (`EXHAUSTIVE_MAX = 150`) | ✅ |
| Route grid inflated by 0.3 m | `web/src/main.js:44` (`inflate(grid, 0.3)`) | ✅ |
| Walker radius 0.22 m, eye height 1.6 m | `web/src/player.js:6` | ✅ |
| 3,276,779 splats; 22.3 MB SOG vs 183.5 MB PLY (8.2×) | `scenes/apartment-fast/scene.json` `"count": 3276779`; `scene.sog` 22,251,974 B; `work/apartment-fast_splats.ply` 183,499,987 B | ✅ |
| Scale estimates > 12 % apart are flagged | `pipeline/navgrid.py:293` | ✅ |

## Toolchain deviation

pandoc + xelatex and draw.io Desktop were not usable here:
- Homebrew is not installed.
- The user-local installers (micromamba, TinyTeX, the draw.io dmg) were blocked by the permission classifier.
- The PyPI `pypandoc_binary` pandoc is x86-only, and the Mac has no Rosetta.

The same `abstract.md` is therefore built by `abstract/build.py`: Python-Markdown → HTML → headless Chrome print-to-PDF. Figures use the drawio skill's documented no-Desktop fallback: YAML → `.drawio` + SVG with the skill's CLI, then vector PDF / 300 dpi PNG via Chrome (`.drawio-tmp/export.sh`).

Rebuild: `python3 abstract/.drawio-tmp/gen.py`, the drawio CLI per figure, `export.sh`, `trim.py`, `export.sh`, `python3 abstract/build.py`.
