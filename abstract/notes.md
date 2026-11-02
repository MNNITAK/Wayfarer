# Wayfarer: repository notes for the technical abstract

Every claim below points to the file it was read from. Items not verifiable from the code are under **OPEN QUESTIONS**.

## 1. What the system is

- Purpose: turn a smartphone video of an indoor space into a photorealistic, walkable 3D web twin with first-person roaming, POI search and obstacle-aware routes. (`README.md` intro; Namma Space, Techfest 2026-27)
- Three runtime parts:
  1. **Capture**: phone-side Capture Coach page (`web/capture.html`, `web/src/capture.js`, `web/src/capture-metrics.js`) and the capture SOP (`docs/SOP.md`).
  2. **Processing**: local job server (`pipeline/server.py`, stdlib HTTP), pre-flight check (`pipeline/preflight.py`), orchestrator (`pipeline/wayfarer.py`) and its stage modules in `pipeline/`.
  3. **Viewing**: static Vite web app (`web/index.html`, `web/src/main.js`, `player.js`, `nav.js`, `minimap.js`), served with `web/vite.config.js`.
- Launch on Windows: `start.bat` starts `pipeline/server.py` (port 8765, proxied at `/api` by `web/vite.config.js`), the Vite dev server on 5173 and an HTTPS phone server on 5174, then opens `upload.html`.

## 2. Entry points

| Entry point | Role | Source |
|---|---|---|
| `python pipeline/wayfarer.py <videos/photo dirs> --name <id> [--ceiling H]` | full pipeline, 5 resumable steps | `pipeline/wayfarer.py:1-13, 31, 216-244` |
| `pipeline/server.py` | upload, jobs, Capture Coach sessions API | `pipeline/server.py` docstring |
| `pipeline/preflight.py` | fast capture check before the build | `pipeline/preflight.py` docstring |
| `pipeline/coach.py plan / verify / review` | AI capture-planning agents | `pipeline/coach.py` docstring |
| `pipeline/import_rgbd.py` | finished RGB-D scan → twin, no COLMAP or training | `pipeline/import_rgbd.py` docstring |
| `pipeline/rgbd_to_nerfstudio.py`, `kaggle_train.py`, `kaggle_job.py` | train a splat from an RGB-D scan on a local or Kaggle GPU | docstrings; `README.md` "Finished RGB-D scans" |
| `web/` `npm run dev / build / test / phone` | viewer, tests, HTTPS phone mode | `web/package.json` scripts |

## 3. Data flow (traced through `pipeline/wayfarer.py main()`)

1. **frames** (`step_frames`, `wayfarer.py:69-109`): ffmpeg extracts candidates at `--fps` (default 8, `:224`), scaled to max side 1280 px (1600 for `--quality high/max`, `:262`). Per clip, the sharpest frame per window (variance of Laplacian on a 480 px grey thumbnail, `sharpness()` `:59-66`) is kept if ≥ 0.55× the sharpest frame within ±2 s (`:96-102`). Cap `--max-frames` 500 (`:225`). Photo folders keep every photo (`:97`).
2. **sfm** (`colmap_sparse`, `step_sfm`, `:140-206`): COLMAP run directly: `feature_extractor` with single shared OPENCV camera, GPU SIFT, 8192 features (`:152-155`); `exhaustive_matcher` when ≤ 150 frames (`EXHAUSTIVE_MAX`, `:29`), otherwise `sequential_matcher` with overlap 20, quadratic overlap and loop detection (`:160-163`); `view_graph_calibrator` + `global_mapper` by default (`--mapper global`, `:173-175`, `:229`). The largest model becomes `sparse/0` (`:177-185`); stray cameras > 10× the median distance are deleted (`stray_cameras`, `drop_stray_cameras`, `:118-137`). Gate: < 50 % registered stops the run; a split into several models prints a warning (`:191-199`). Then `ns-process-data --skip-colmap` (`:200-201`). Optional `--mask-people` (`pipeline/masks.py`, Faster R-CNN person masks → ignored by COLMAP and training).
   - Code comment, `:170-171`: global mapping on "myhome" registered 284/292 frames in one model, 0.67 px, 6.5 min vs incremental 282 frames split in two, 0.71 px, 17 min.
3. **train** (`:285-316`): `ns-train splatfacto` with `--max-num-iterations` 7k/30k/40k/50k by `--quality` (`:290`), CPU uint8 image cache, bilateral grid, scale regularisation, SO3xR3 camera optimiser (`:307-312`). Resumes from the last checkpoint (`:292-298`). Alternative `--trainer mcmc` → `pipeline/train_mcmc.py` (gsplat MCMCStrategy, hard Gaussian cap, default 1M, per its docstring).
4. **export** (`:318-337`): `ns-export gaussian-splat` → `splat.ply`; `ns-eval` → PSNR/SSIM/LPIPS on held-out frames, stored in `stats.json` (`:329-337`).
5. **navgrid** (`:339-345` → `pipeline/navgrid.py`):
   - filter splats with opacity < 0.3 (`navgrid.py:191-193`);
   - RANSAC floor plane (`ransac_plane`, `:92`; 500 iterations, tilt ≤ 25°) on the low layer (`:236-244`), ceiling RANSAC with tilt ≤ 8° (`:259`);
   - metric scale priority: `--scale`, ArUco marker (`pipeline/marker_scale.py`, DICT_4X4_50, `:12-19`), ceiling + `--ceiling`, camera height (default 1.4 m); estimates more than 12 % apart are flagged (`navgrid.py:268-296`);
   - Manhattan yaw alignment (`manhattan_yaw`, `:132`, `:303-304`);
   - obstacle band 0.12–1.9 m, "tall" from 1.2 m (`build()` signature `:187-188`); optional depth carving (`pipeline/depthcarve.py`, `:332-349`); blobs < 0.03 m² dropped (`:350`);
   - camera path cells forced walkable and floor kept only where connected to it (`:355-387`);
   - spawn from the camera path (`:396-406`); writes `scene.json` (`:409`) and SOG via PlayCanvas splat-transform (`:468`).
6. **post-steps** (`wayfarer.py:346-372`): optional `--ai-repair` (`pipeline/airepair.py`, NVIDIA Difix); `autopoi.py` (torchvision Faster R-CNN MobileNet-V3 COCO detector, min score 0.6, `autopoi.py:41,149`) unless `--no-autopoi`; `surfaces.py` (floor/ceiling texture from frames + `simple_lama_inpainting`, `surfaces.py:116-124`) unless `--no-surfaces`; `stats` merged into `scene.json`; `floorplan.py` writes an SVG plan.
- Output: `web/public/scenes/<name>/{scene.json, scene.sog, grid.png, floorplan.svg}` (`README.md` "Capture → twin", scene folders on disk).
- `scene.json` schema: `README.md` "scene.json" section (name, cellSize, origin, rows with `.`/`#`/`o`/space, floorY, wallHeight, eyeHeight, spawn, splat TRS, pois).

## 4. Web viewer (`web/src/`)

- Scene chosen by `?scene=`, else `VITE_DEFAULT_SCENE`, else `demo` (`main.js:8`); loads `scene.json` by fetch (`main.js:41`).
- Spark `SparkRenderer` with `maxStdDev: √5`, `coneFov0: 130`, `coneFov: 160`; `SplatMesh` with `lod: true` (LoD tree in a worker) (`main.js:55-62`).
- Player: eye height 1.6 m, radius 0.22 m (`player.js:6`); exponential accel/decel (`player.js:106`); per-axis collision against the grid (`player.js:117-118`); fly mode; touch stick + drag-to-look when pointer lock is missing (commits `adb2488`, `55426c7`).
- Navigation (`nav.js`): `parseGrid`, `inflate`, `nearestFree`, 8-connected `astar` with octile heuristic and no corner cutting (`nav.js:70-71`), `lineOfSight`, `smooth`, `findRoute` with fallback from inflated to raw grid (`nav.js:130-137`), `nextInstruction` (`:148`). Margin 0.3 m (`main.js:44`). Route drawn as instanced arrow meshes (`main.js:280-317`); ETA at 1.3 m/s (`main.js:10`); arrival after < 0.6 m (`main.js:328`).
- Search: ranked score exact 100 > prefix 80 > word-prefix 60 > substring 40 > description/category 20 > subsequence 10, top 8 (`main.js:244-257`).
- Minimap (`minimap.js`); blueprint fallback when no splat (`README.md` table, demo scene has `splat: null`).
- Keys: `README.md` key table.

## 5. Capture SOP and checks

- SOP steps: `docs/SOP.md` sections 0–6 (kit, prepare, phone settings, capture path, single take vs room-by-room, quality gates, privacy).
- Single source of thresholds: `pipeline/capture_requirements.md` §8 table (blur < 0.55× local best; turn > 30°/s; dark < 35/255; blank wall < 1.5 grey levels/px; exposure jump > 18 in 1/8 s; ≥ 90 % registered on 120 frames at 640 px; median ray spread ≥ 15°).
- Capture Coach flow Survey → Prepare → Record → Review → Build, with retakes: `README.md` "Capture Coach"; server endpoints in `pipeline/server.py` docstring; agents in `pipeline/coach.py`.

## 6. Tests

- `web/test/nav.test.js` (6 tests), `web/test/capture-metrics.test.js` (8 tests), run by `npm test` (`web/package.json`).
- `pipeline/test_navgrid.py`: synthetic room in a tilted/rotated/scaled frame; asserts scale within 3 %, wall height within 0.1 m, walkable area 18–24 m², table 0.6–1.8 m², door walkable, plus a wrong-up-hint case (`test_navgrid.py:56-83`).
- `pipeline/test_marker_scale.py`, `test_preflight.py`, `test_coach.py` (Ollama and Claude stubbed), `test_wayfarer.py`.

## 7. Reconstructed scenes on disk (not in git)

- `web/public/scenes/apartment*/scene.json`: Redwood indoor apartment imported via `import_rgbd.py`; 4,680,595 splats (`apartment`), 98×79 grid at 0.1 m, about 41.6 m² walkable, 18 POIs (4 rooms + objects). No `stats` block (no held-out metrics).

## 8. Credits inventory

| Name | Used for | License | Source of evidence |
|---|---|---|---|
| three.js 0.186.1 | WebGL scene, camera, CSS2D labels | MIT | `web/package.json`, `package-lock.json` |
| Spark (@sparkjsdev/spark) 2.3.1 | Gaussian splat rendering with LoD | MIT | same |
| PlayCanvas splat-transform 3.10.1 (+ playcanvas 2.23.3) | PLY → SOG compression, floater filter | MIT | same; `navgrid.py:468` |
| @adobe/spz 0.2.3 | dependency of splat-transform | ISC | `package-lock.json` |
| Vite 8.3.4 (rolldown, lightningcss MPL-2.0, postcss) | dev server and build | MIT (lightningcss MPL-2.0) | same |
| @vitejs/plugin-basic-ssl 2.3.0 | self-signed HTTPS for phone mode | MIT | same; `vite.config.js` |
| NumPy | array maths throughout | BSD-3-Clause | `pipeline/requirements.txt` |
| Pillow | image I/O, sharpness thumbnails | MIT-CMU (HPND) | `requirements.txt`; `wayfarer.py:61` |
| SciPy | rotations, KD-trees, ndimage | BSD-3-Clause | imports in `navgrid.py`, `import_rgbd.py`, `autopoi.py` |
| OpenCV (cv2, ArUco) | marker detection, image ops | Apache-2.0 | `marker_scale.py:18` |
| PyTorch 2.1.2 / torchvision 0.16.2 | GPU tensors; Faster R-CNN MobileNet-V3 COCO detector | BSD-3-Clause | `docs/SETUP.md`; `autopoi.py:41`, `masks.py:24` |
| COCO-pretrained detector weights (torchvision) | object and person detection | per torchvision weights (BSD-3 code; COCO dataset CC BY 4.0) | `autopoi.py`, `masks.py` |
| nerfstudio 1.1.5 (splatfacto, ns-process-data/train/export/eval) | splat training, export, evaluation | Apache-2.0 | `docs/SETUP.md`; `wayfarer.py:200-327` |
| gsplat 1.4.0 | CUDA rasteriser; MCMC trainer; depth rendering | Apache-2.0 | `docs/SETUP.md`; `train_mcmc.py`, `depthcarve.py` |
| COLMAP 4.2 (incl. global mapper / GLOMAP) | features, matching, SfM | BSD-3-Clause | `docs/SETUP.md`; `wayfarer.py:152-175` |
| FFmpeg | frame extraction | LGPL-2.1+ | `docs/SETUP.md`; `wayfarer.py:79` |
| scikit-image | marching cubes for mesh export | BSD-3-Clause | `mesh.py:73` |
| PyMeshLab | mesh cleanup | GPL-3.0 | `mesh.py:71` |
| simple-lama-inpainting (LaMa model) | inpaint unseen floor/ceiling texels | Apache-2.0 (LaMa) | `surfaces.py:124` |
| NVIDIA Difix3D / `nvidia/difix_ref` model | optional AI view repair | NVIDIA license, non-commercial research | `airepair.py:10, 161-162` |
| Ollama + Qwen2.5-VL 7B | local vision model for the Capture Coach | Ollama MIT; Qwen2.5-VL-7B Apache-2.0 | `coach.py:8-10, 30`; `docs/SETUP.md` §4 |
| Anthropic Claude API + `anthropic` SDK (optional) | alternative coach provider | commercial API; SDK MIT | `coach.py:11, 155-175` |
| Kaggle (kernels, `kaggle` CLI) | optional free cloud GPU training | Kaggle ToS; CLI Apache-2.0 | `kaggle_train.py`, `kaggle_job.py` |
| Vercel | static hosting of the viewer (https://wayfarer-ten-kappa.vercel.app/) | commercial service | team answer; `main.js:8` (`VITE_DEFAULT_SCENE` for hosted builds) |
| uv | Python env setup | MIT / Apache-2.0 | `docs/SETUP.md` |
| Node.js | runs viewer tooling and `node --test` | MIT | `docs/SETUP.md`, `web/package.json` |
| Redwood indoor RGB-D dataset | benchmark scene (apartment) | to confirm | `import_rgbd.py` docstring; `README.md` |
| draw.io format via drawio-skills CLI (`.claude/skills/drawio/scripts/cli.js`) | figure sources (.drawio) and SVG | MIT | this abstract |
| Python-Markdown 3.9 | abstract.md → HTML | BSD-3-Clause | `abstract/build.py` |
| Google Chrome (headless) | HTML/SVG → vector PDF, 300 dpi PNG | proprietary (Chromium BSD) | `abstract/build.py`, `.drawio-tmp/export.sh` |
| PyMuPDF 1.26 | page count, page renders for checks | AGPL-3.0 | Phase 6 checks |
| Pillow 11.3 | figure whitespace trimming | MIT-CMU | figure export |

Note: pandoc + xelatex and draw.io Desktop could not run here (Homebrew missing; the installers were blocked by the permission classifier; the PyPI pandoc wheel is x86-only and there is no Rosetta), so the Chrome-based build replaces them.
| drawio-skills, scientific-writing agent skills | figure and writing workflow for this abstract | MIT | `.claude/skills/*/SKILL.md` |

Licences not stated in the repo itself come from the upstream projects and are marked for a final check.

## ANSWERS FROM THE TEAM (2026-11-01)

- Team: **Simpletons**, Team ID **TM-A30B5FE384F**. Members: Ayush Jadaun (leader), Abhay Agarwal, Arnav Raj, Akshay Yadav.
- Live viewer: https://wayfarer-ten-kappa.vercel.app/ · Demo video: https://drive.google.com/file/d/1haPZGPQqfS8M_v1reZNNYYvk4NMd-9Gs/view?usp=sharing
- Report the numbers of the current working **"fast apartment" demo** (uncommitted files are fine).
- Training GPU: **Kaggle T4** (team statement).

### Fast apartment demo: measured from files on disk
- `web/public/scenes/apartment-fast/scene.json`: name "Apartment (sharp + fast)"; **3,276,779 splats**; grid 98 × 79 cells at 0.1 m; **38.4 m² walkable** (count of `.` cells × 0.01 m²); wall height 2.67 m; eye height 1.6 m; **18 POIs** (4 rooms + 14 items, all from `autopoi.py`); no `stats` block.
- `web/public/scenes/apartment-fast/scene.sog`: 22,251,974 B (**22.3 MB**). Source splat PLY `work/apartment-fast_splats.ply`: 183,499,987 B (**183.5 MB**), so about **8.2×** smaller after SOG.
- Built with `import_rgbd.py` from the Redwood apartment scan: mesh `work/datasets/apartment/ours/apartment.ply`, trajectory `work/datasets/apartment/pose/apartment.log`, **31,919 RGB-D frames** (`work/datasets/apartment/rgbd/apartment/image`), and the LIDAR scan `work/datasets/apartment/lidar/merged/apt.ply`, registered automatically (`lidar/align_debug.json`). The flags `--voxel 0.005 --adaptive 0.02` come from `README.md` ("5 mm detail, 2 cm on plain walls: sharp + 60 fps") and commit `d683b5a` ("3x fewer splats, 60 fps"); the exact command used for this folder is not logged.
- The only Kaggle log on disk (`work/kaggle/apartment/output/wayfarer-apartment-train.log`) records "NO GPU in this session". So the fast demo does **not** come from Kaggle splat training; the T4 is reported as the team's training setup.

## OPEN QUESTIONS

1. **Team name, team ID, member names, institute**: to be supplied.
2. **Live web-viewer URL and video link**: not in the repo. `main.js:8` mentions hosted builds (`VITE_DEFAULT_SCENE`), but no deployment URL is recorded.
3. **Home-capture results**: the only real-capture numbers are in a code comment (`wayfarer.py:170-171`). There are no PSNR/SSIM/LPIPS, splat count, size or FPS for the home twin (the scenes on disk are the Redwood apartment, with no `stats`). Should the abstract report the comment's numbers, or should you supply measured ones?
4. **Earlier-draft numbers not backed by code**: the old `docs/abstract/abstract.html` states 0.04 % synthetic scale error, 0.3 % marker error, 90 MB → 8 MB SOG, and a 27.72 vs 23.09 dB trainer benchmark. The tests only assert looser bounds (scale within 3 %). These are excluded unless you confirm them.
5. **Sample data in a `test/` directory**: the brief requires one; the repo has none (`git ls-files`).
6. **Hosting**: the README says "any desktop browser"; Vercel is mentioned only in the old draft. Where is it actually deployed?
7. **Redwood dataset licence**: not stated in the repo.
8. **Platform**: `docs/SETUP.md` says tested on Windows 11 + RTX 4070 Laptop (8 GB). Same hardware for the submission?
