# What the Wayfarer pipeline needs from a capture

Single source of truth for the Capture Coach: the planner/verifier/reviewer agents (`coach.py`) read this file as
their brief, and the numeric checks in `preflight.py` and `web/src/capture-metrics.js` use the numbers below.
When a pipeline threshold changes, change it here too.

The pipeline: video -> sharp frames -> COLMAP camera poses -> Gaussian splatting -> metric floor + walkable grid ->
a first-person web twin. A capture is good when every stage can do its job.

## 1. Sharp frames (`step_frames`, `wayfarer.py`)
- Frames are sampled at 8 fps; in each ~2 s window a frame is kept only if its sharpness (variance of the Laplacian
  on a 480 px greyscale thumbnail) is at least **0.55x** the sharpest frame of that moment. Motion blur drops frames.
- Target: **> 70 %** of windows keep a sharp frame.
- Causes of blur: walking faster than **0.5 m/s**, turning faster than **30 deg/s**, low light (long exposure).

## 2. Camera poses (`step_sfm`, COLMAP)
- Every kept frame must share features with its neighbours. Target: **>= 90 %** of frames registered in **one**
  model. Below **50 %** the build stops. A split into several models means some rooms are missing.
- Up to 150 frames everything is matched with everything; beyond that, neighbours + loop closure, so **returning to
  the start** matters on long takes.
- Feature killers: blank walls filmed closer than ~1 m, fast pans, mirrors and screens (a fake room behind the wall),
  moving things (ceiling fans, people, pets, curtains in wind), exposure jumping between frames (lock AE/AF; HDR,
  night and beauty modes off), the 0.5x ultra-wide lens.
- Separate rooms only join through **doorways walked slowly** with both rooms in view for ~3 s.

## 3. Gaussian splatting (training)
- Each surface is fitted only from the viewpoints that saw it. A surface filmed from one direction becomes needle-like
  "fibrous" splats when seen from another angle in the viewer.
- **Viewpoint rule: film from where people will stand in the viewer, looking where they will look.** Every wall needs
  views both head-on and at a grazing angle while walking along it; the middle of each room needs views looking out.
- Rays observing the same wall point should span **>= 15 deg** (checked on the quick SfM model).
- Even lighting: all lights on, curtains half-drawn, no moving sun patches.

## 4. Metric scale and floor (`navgrid.py`, `marker_scale.py`)
- The floor plane is found from splats; the floor must be **visible in most frames** (camera tilted 10-15 deg down).
- Scale comes from a printed ArUco marker (DICT_4X4_50, 150 mm black square, flat on a wall or floor), seen from 3+
  angles at 0.5-2 m, filling >= 1/10 of the frame; cross-checked with the tape-measured ceiling (one ceiling sweep per
  room) and a camera held at ~1.4 m. Estimates disagreeing by > 12 % trigger a warning.

## 5. Walkable grid
- Obstacles are taken from 0.12-1.9 m above the floor. Furniture needs to be seen all round (a low loop at ~0.8 m around
  beds and tables) so its footprint is closed. Doorways the camera walked through always stay open.

## 6. Capture path (per room)
1. Scale marker first (5 s, 3 angles).
2. Perimeter loop 0.5-1 m from the walls, facing in, chest height (~1.4 m), tilted slightly down.
3. **Along-the-wall pass:** walk along each wall 0.5-1 m away, camera pointing along the walking direction and angled
   30-45 deg toward the wall.
4. A few slow walks centre -> wall -> centre; a slow 360 deg turn in the middle.
5. Corner sweeps (2 s stop, 90 deg sweep), one ceiling sweep (~3 s), low loop around big furniture.
6. Doorway to the next room: walk through slowly, both rooms in view.
7. Finish where you started.

## 7. Room preparation
Lights on; fans and moving objects off/out; interior doors fully open; screens off; big mirrors covered; personal
documents and photos removed (the twin is photorealistic); marker placed; ceiling height measured.

## 8. Numbers the automatic checks use
| Check | Value | Where |
|---|---|---|
| Motion blur | sharpness < 0.55x the sharpest frame within 2 s | preflight, live guide |
| Turn speed | > 30 deg/s (image motion, 65 deg lens; gyro on the phone) | preflight, live guide |
| Too dark | mean grey < 35 of 255 | preflight, live guide |
| Blank wall | mean gradient < 1.5 grey levels/px at 240 px wide | preflight, live guide |
| Exposure jump | grey level jumps > 18 in 1/8 s while the view is still | preflight |
| Camera poses | >= 90 % registered, one model (120 frames, 640 px) | preflight |
| View angles | median ray spread >= 15 deg per floor-plan cell | preflight |

The blur, dark, blank and exposure numbers are first estimates from synthetic footage; tune them on real takes
(a take that builds well should pass, a take that failed in the main build should not).
