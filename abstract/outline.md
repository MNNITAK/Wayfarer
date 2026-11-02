# Outline: Wayfarer Technical Abstract (≤ 5 pages, A4, 2 cm margins, 11 pt)

Evidence for every bullet is in `abstract/notes.md` (section number in brackets).

## Title block (~0.3 page)
- Title: *Wayfarer: Smartphone Video to a Walkable Indoor Digital Twin*
- Line: Team name · Team ID · members · institute (placeholders), repository URL, viewer URL (placeholder)
- Overview paragraph, 4–5 sentences: problem (2D floor plans → web 3D twin); input (one phone video); output (photoreal splat + metric occupancy grid in a zero-install viewer); what Round 1 covers (capture SOP, reconstruction, free roaming) and what the grid enables later (POIs, A*). [1]

## 1. System Architecture (~1 page) → Fig. 1
- Three tiers: Capture (phone: `capture.html`, Coach), Processing (`server.py` → `preflight.py` → `wayfarer.py` + stage modules), Viewing (static Vite app: `main.js`, `player.js`, `nav.js`, `minimap.js`, Spark). [1, 2]
- Interfaces: HTTP API on 8765 proxied at `/api`; the scene package (`scene.json` + `scene.sog`) is the only contract between processing and viewer. [1, 3]
- Design choices: stdlib server; every stage writes to disk and resumes (`--force STEP`); the grid stored as plain-text rows. [3]
- Alternative inputs: photo folders, multiple room clips, RGB-D scans via `import_rgbd.py`; optional Kaggle training. [2]

## 2. Pipeline Design (~1 page) → Fig. 2
- Five steps with their tools, parameters and gates: frames → sfm → train → export/eval → navgrid, then post-steps (autopoi, surfaces, floorplan; optional mask-people, ai-repair). [3]
- Artifacts at each edge (frames/, COLMAP sparse/0, transforms.json, splat.ply, metrics.json, scene.json/sog).
- Gates: < 50 % registration stops the run; multi-model warning; scale disagreement > 12 % flagged. [3]
- Stats travel with the twin (`stats.json` → `scene.json`). [3]

## 3. SOP Details (~0.75 page) → Fig. 3
Numbered steps covering setup, run and operation:
1. Install (SETUP.md: uv venv, torch/nerfstudio/gsplat wheel, COLMAP 4.2, FFmpeg, Node 20; optional Ollama + qwen2.5vl). [2, 8]
2. Prepare the space (lights, fans off, doors open, mirrors/screens covered, marker printed at 150 mm, ceiling taped). [5]
3. Phone settings (1× lens, 4K/30 or 1080p/60, landscape, AE/AF lock, stabilisation on, HDR off). [5]
4. Film (perimeter loop facing in, corner sweeps, 10–15° tilt down, ceiling sweep, along-the-wall pass, marker 5 s from 3+ angles, doorway overlap, end at start; ≤ 0.5 m/s, ≤ 30°/s). [5]
5. Check the take (Coach review / `preflight.py`; retakes). [5]
6. Build (`start.bat` → Studio → *Build twin*, or the CLI command). [2]
7. Inspect gates in the log (sharp frames, registration %, scale cross-check, walkable area). [5]
8. Publish / open the viewer (`?scene=<name>`; static build). [4]
- Compact table of quality-gate thresholds. [5]

## 4. Technical Approach (~1.25 pages) → Fig. 4 (optional)
- 4.1 Blur-aware frame selection: local-relative sharpness criterion and why. [3]
- 4.2 Robust SfM: COLMAP driven directly; exhaustive vs sequential + loop closure; global mapper; stray-camera removal; with the recorded home-capture comparison (only if OPEN QUESTION 3 approved). [3]
- 4.3 Gaussian splatting configured for 8 GB: bilateral grid, scale regularisation, pose refinement, resume; MCMC fallback. [3]
- 4.4 Navigation-grid builder: floater filter → floor RANSAC → three-way metric scale → Manhattan yaw → obstacle band → depth carving → camera-path walkability → SOG export. [3]
- 4.5 Web viewer: Spark LoD with a fixed splat budget, √5σ cutoff, widened foveation cone; per-axis collision with a 0.22 m body; grid-based A* (octile, no corner cutting, 0.3 m margin with raw-grid fallback, line-of-sight smoothing); ranked search. [4]
- 4.6 Verification: automated tests and what they assert (nav 6, capture metrics 8, synthetic navgrid within 3 %, marker, preflight, coach). [6]

## 5. Credits (~0.5 page)
- Compact two-column table: every item in notes §8 (name, purpose, licence), including the abstract tooling (draw.io, pandoc, XeLaTeX, agent skills).

## Proposed diagrams (draw.io, academic theme, one consistent palette)
1. **Fig. 1: System architecture.** Three swimlanes (Capture / Processing / Viewing) with real component names; labelled edges (video upload, `/api` jobs, scene package, HTTP fetch). About 14 nodes.
2. **Fig. 2: Pipeline / data flow.** Left-to-right: video → frames → sfm → train → export → navgrid → scene package, with artifact labels on the edges and gates as diamonds; post-steps as a lower branch. About 12 nodes.
3. **Fig. 3: SOP flowchart.** Top-to-bottom: Install → Prepare → Configure phone → Film → Pre-flight check (decision: pass? → retake loop) → Build → Inspect gates (decision) → Open in viewer. About 10 nodes.
4. **Fig. 4 (optional): Navigation-grid builder.** Seven-stage chain from splat to grid with the three scale estimates feeding a "scale fusion" node. About 10 nodes.

Palette proposal: muted blue (processing), muted teal (capture/viewer), grey (data artifacts), one accent for decision/gate shapes; sans-serif (Helvetica/Arial) throughout.
