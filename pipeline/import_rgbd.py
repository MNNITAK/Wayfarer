"""RGB-D dataset (mesh + camera trajectory + colour/depth frames) -> walkable web twin, without training.

  python pipeline/import_rgbd.py --mesh apartment.ply --log apartment.log --frames rgbd/apartment --name apartment

For finished scans such as the Redwood indoor scenes: the geometry and the metric camera path are already known, so
there is no COLMAP and no GPU training. Each surface point is coloured from the real photos (averaged over many
views, occlusion-checked against the depth images, sharp frames weighted up), turned into a flat splat facing along
the surface normal, and the walkable grid comes from navgrid.build with the real camera path.

Trajectory: Redwood .log (camera-to-world 4x4 per frame, OpenCV axes: x right, y down, z forward).
Frames: <frames>/image/NNNNNN.jpg and <frames>/depth/NNNNNN.png (16-bit millimetres).
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

import navgrid

SH_C0 = 0.28209479177387814


def read_mesh(path):
    """(vertices Nx3 float64, triangles Mx3 uint32) from a binary PLY with only x/y/z and a triangle list."""
    with open(path, 'rb') as f:
        header = []
        while (line := f.readline().decode('ascii').strip()) != 'end_header':
            header.append(line)
        n = int(next(l.split()[2] for l in header if l.startswith('element vertex')))
        m = int(next((l.split()[2] for l in header if l.startswith('element face')), 0))
        vtype = next(l.split()[1] for l in header if l.startswith('property') and l.endswith(' x'))
        v = np.fromfile(f, dtype=navgrid.PLY_TYPES[vtype], count=3 * n).reshape(n, 3).astype(np.float64)
        faces = np.fromfile(f, dtype=[('n', 'u1'), ('i', '<u4', 3)], count=m)
    if m and not (faces['n'] == 3).all():
        sys.exit('only triangle meshes are supported')
    return v, faces['i']


def read_log(path):
    """Redwood .log -> camera-to-world matrices (N, 4, 4)."""
    rows = [l.split() for l in Path(path).read_text().splitlines() if l.strip()]
    return np.array([[list(map(float, r)) for r in rows[i + 1:i + 5]] for i in range(0, len(rows), 5)])


def vertex_normals(v, tri):
    fn = np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]])  # area-weighted face normals
    n = np.zeros_like(v)
    for k in range(3):
        for a in range(3):
            n[:, a] += np.bincount(tri[:, k], weights=fn[:, a], minlength=len(v))
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def voxel_pick(v, size):
    """Index of one vertex per voxel of `size` metres."""
    key = np.floor((v - v.min(0)) / size).astype(np.int64)
    key = (key[:, 0] * 73856093) ^ (key[:, 1] * 19349663) ^ (key[:, 2] * 83492791)
    _, first = np.unique(key, return_index=True)
    return np.sort(first)


def frame_paths(frames, i):
    return frames / 'image' / f'{i:06d}.jpg', frames / 'depth' / f'{i:06d}.png'


def load_frame(frames, i):
    from PIL import Image
    c, d = frame_paths(frames, i)
    return np.asarray(Image.open(c).convert('RGB')), np.asarray(Image.open(d)).astype(np.float32) / 1000


def project(P, c2w, K):
    """World points -> (u, v, z) in a camera with OpenCV axes."""
    R, t = c2w[:3, :3], c2w[:3, 3]
    X = (P - t) @ R  # = R^T (P - t)
    z = X[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = K[0] * X[:, 0] / z + K[2]
        w = K[1] * X[:, 1] / z + K[3]
    return u, w, z


def alignment(P, poses, frames, K, ids):
    """Median |mesh depth - sensor depth| (m) over a few frames: checks poses, intrinsics and axes together."""
    errs = []
    for i in ids:
        _, depth = load_frame(frames, i)
        h, w = depth.shape
        u, v, z = project(P, poses[i], K)
        ok = (z > 0.2) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ui, vi, zi = u[ok].astype(int), v[ok].astype(int), z[ok]
        zbuf = np.full(h * w, np.inf)
        np.minimum.at(zbuf, vi * w + ui, zi)
        zbuf = zbuf.reshape(h, w)
        m = np.isfinite(zbuf) & (depth > 0)
        if m.sum() > 1000:
            errs.append(float(np.median(np.abs(zbuf[m] - depth[m]))))
    return float(np.median(errs)) if errs else float('inf')


MIN_AGREE = 0.3  # share of near-surface points within 3 cm of the sensor depth; below it the frame's pose is off


def colorize(P, N, poses, frames, K, ids, log=print):
    """Average colour per point over frames where it is visible (depth test), weighted by view angle, distance
    and frame sharpness; frames whose pose disagrees with the mesh are skipped. Returns (rgb Nx3 0..1, weight N)."""
    acc, wsum = np.zeros((len(P), 3), np.float32), np.zeros(len(P), np.float32)
    skipped = 0
    P32, N32 = P.astype(np.float32), N.astype(np.float32)
    t0 = time.time()
    for k, i in enumerate(ids):
        rgb, depth = load_frame(frames, i)
        h, w = depth.shape
        g = rgb.mean(2)
        lap = g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1] - 4 * g[1:-1, 1:-1]
        sharp = float(lap.var())
        c2w = poses[i].astype(np.float32)
        u, v, z = project(P32, c2w, K)
        ok = (z > 0.2) & (z < 5) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        idx = np.flatnonzero(ok)
        ui, vi, zi = u[idx].astype(int), v[idx].astype(int), z[idx]
        d = depth[vi, ui]
        vis = (d > 0) & (np.abs(zi - d) < 0.03 + 0.02 * zi)  # the sensor saw this surface, not something in front
        # judge the pose only on points near the surface the sensor saw: points hidden behind it fail the depth
        # test legitimately, while a shifted pose moves every nearby point off by a few cm
        near = (d > 0) & (np.abs(zi - d) < 0.3)
        if vis.sum() < MIN_AGREE * max(1, np.count_nonzero(near)):
            skipped += 1
            continue
        idx, ui, vi, zi = idx[vis], ui[vis], vi[vis], zi[vis]
        ray = P32[idx] - c2w[:3, 3]
        ray /= np.linalg.norm(ray, axis=1, keepdims=True)
        cos = np.abs((ray * N32[idx]).sum(1))
        wgt = (np.clip(cos, 0.15, 1) ** 2 / np.maximum(zi, 0.5) ** 2) * sharp
        acc[idx] += rgb[vi, ui].astype(np.float32) * wgt[:, None]
        wsum[idx] += wgt
        if k % 200 == 0:
            log(f'  colour: frame {k + 1}/{len(ids)}, {np.count_nonzero(wsum) / len(P):.0%} of points seen, '
                f'{skipped} frames skipped for a bad pose ({time.time() - t0:.0f} s)')
    log(f'  {skipped} of {len(ids)} frames skipped: their pose disagrees with the mesh')
    seen = wsum > 0
    rgb = np.zeros_like(acc)
    rgb[seen] = acc[seen] / wsum[seen, None] / 255
    return rgb, wsum


def splats(P, N, rgb, size):
    """Flat Gaussians: wide along the surface, thin along the normal (scale_2 is the normal axis)."""
    n = N / np.linalg.norm(N, axis=1, keepdims=True)
    q = np.c_[1 + n[:, 2], -n[:, 1], n[:, 0], np.zeros(len(n))]  # rotates +z onto n: (w, x, y, z)
    flip = q[:, 0] < 1e-6  # n = -z
    q[flip] = [0, 1, 0, 0]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    names = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2',
             'rot_0', 'rot_1', 'rot_2', 'rot_3']
    out = np.zeros(len(P), dtype=[(k, '<f4') for k in names])
    out['x'], out['y'], out['z'] = P.T
    dc = (rgb - 0.5) / SH_C0
    out['f_dc_0'], out['f_dc_1'], out['f_dc_2'] = dc.T
    out['opacity'] = math.log(0.97 / 0.03)
    out['scale_0'] = out['scale_1'] = math.log(size * 0.7)
    out['scale_2'] = math.log(size * 0.08)
    out['rot_0'], out['rot_1'], out['rot_2'], out['rot_3'] = q.T
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mesh', type=Path, required=True)
    ap.add_argument('--log', type=Path, required=True, help='camera trajectory (.log)')
    ap.add_argument('--frames', type=Path, required=True, help='folder with image/ and depth/')
    ap.add_argument('--name', required=True, help='scene id: web/public/scenes/<name>')
    ap.add_argument('--title', help='display name')
    ap.add_argument('--intrinsics', default='525,525,319.5,239.5', help='fx,fy,cx,cy (default: PrimeSense)')
    ap.add_argument('--voxel', type=float, default=0.012, help='splat spacing in metres')
    ap.add_argument('--step', type=int, default=15, help='use every Nth frame for colour')
    ap.add_argument('--work', type=Path, default=Path(__file__).resolve().parent.parent / 'work')
    a = ap.parse_args()
    K = np.array([float(x) for x in a.intrinsics.split(',')])
    out = Path(__file__).resolve().parent.parent / 'web' / 'public' / 'scenes' / a.name

    t = time.time()
    print(f'[1/5] reading {a.mesh.name}')
    V, tri = read_mesh(a.mesh)
    poses = read_log(a.log)
    n_frames = len(list((a.frames / 'image').glob('*.jpg')))
    poses = poses[:n_frames]
    print(f'  {len(V):,} vertices, {len(tri):,} triangles, {len(poses):,} camera poses ({time.time() - t:.0f} s)')

    print('[2/5] checking poses + intrinsics against the depth images')
    # every vertex: a sparse sample lets pixels see through gaps to the wall behind and inflates the error
    err = alignment(V, poses, a.frames, K, np.linspace(0, len(poses) - 1, 12).astype(int))
    print(f'  mesh vs sensor depth: median {err * 100:.1f} cm')
    # wrong axes, inverted poses or a wrong focal length are off by tens of cm to metres; a few badly posed frames
    # only nudge this, and colorize() skips those frames one by one
    if err > 0.15:
        sys.exit('poses, intrinsics or axes do not match the mesh (> 15 cm); check --intrinsics and the trajectory')
    if err > 0.05:
        print('  WARNING: some frames are poorly aligned; they will be skipped when colouring')

    print(f'[3/5] surface points every {a.voxel * 100:.1f} cm + normals')
    N = vertex_normals(V, tri)
    sel = voxel_pick(V, a.voxel)
    P, N = V[sel], N[sel]
    del V, tri
    print(f'  {len(P):,} points')

    print(f'[4/5] colour from every {a.step}th frame')
    ids = np.arange(0, len(poses), a.step)
    rgb, wsum = colorize(P, N, poses, a.frames, K, ids)
    keep = wsum > 0
    print(f'  {keep.mean():.0%} of points seen in at least one frame; dropping the rest')
    v = splats(P[keep], N[keep], rgb[keep], a.voxel)

    print('[5/5] floor, walls, walkable grid')
    cams = (poses[:, :3, 3], poses[:, :3, 2], -poses[:, :3, 1])  # centre, forward, image-up
    up_hint = np.median(cams[2], 0)
    scene = navgrid.build(v, cams, up=up_hint, scale=1.0, cell=0.1)  # the scan is metric already
    # look from where the scanner looked: a hand-held scan rarely sees above ~2 m, so a 1.6 m eye peers over the walls
    from scipy.spatial.transform import Rotation
    tr = scene['transform']
    cam_y = cams[0] @ Rotation.from_quat(tr['rotation']).as_matrix()[1] * tr['scale'][1] + tr['position'][1]
    scene['eyeHeight'] = round(float(np.clip(np.median(cam_y) - scene['floorY'], 1.1, 1.7)), 2)
    print(f"  eye height {scene['eyeHeight']} m (median scanner height)")
    a.work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    ply = a.work / f'{a.name}_splats.ply'
    navgrid.write_ply(v, ply)
    navgrid.export_scene(v, scene, out, a.title or a.name, str(ply))
    print(f'done in {(time.time() - t) / 60:.1f} min: open the viewer with ?scene={a.name}')


if __name__ == '__main__':
    main()
