"""RGB-D dataset (mesh + camera trajectory + colour/depth frames) -> walkable web twin, without training.

  python pipeline/import_rgbd.py --mesh apartment.ply --log apartment.log --frames rgbd/apartment --name apartment

For finished scans such as the Redwood indoor scenes: the geometry and the metric camera path are already known, so
there is no COLMAP and no GPU training. Each surface point is coloured from the real photos (averaged over many
views, occlusion-checked against the depth images, sharp frames weighted up), turned into a flat splat facing along
the surface normal, and the walkable grid comes from navgrid.build with the real camera path.

Trajectory: Redwood .log (camera-to-world 4x4 per frame, OpenCV axes: x right, y down, z forward).
Frames: <frames>/image/NNNNNN.jpg and <frames>/depth/NNNNNN.png (16-bit millimetres).
With --lidar (a coloured laser scan of the same space, e.g. Redwood's merged & resampled scan), the scan replaces the
mesh: it is registered to the mesh automatically and keeps its own colours, so ceilings and upper walls are complete.
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


def depth_agree(d, z):
    """Per point: does the sensor depth d confirm projected depth z? Plus whether the frame's pose is trustworthy."""
    vis = (d > 0) & (np.abs(z - d) < 0.03 + 0.02 * z)  # the sensor saw this surface, not something in front
    # judge the pose only on points near the surface the sensor saw: points hidden behind it fail the depth
    # test legitimately, while a shifted pose moves every nearby point off by a few cm
    near = (d > 0) & (np.abs(z - d) < 0.3)
    return vis, vis.sum() >= MIN_AGREE * max(1, np.count_nonzero(near))


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
        vis, good = depth_agree(depth[vi, ui], zi)
        if not good:
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


def splats(P, N, rgb, size, spread=0.7):
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
    out['scale_0'] = out['scale_1'] = math.log(size * spread)
    out['scale_2'] = math.log(size * 0.08)
    out['rot_0'], out['rot_1'], out['rot_2'], out['rot_3'] = q.T
    return out


def fill_above(P, N, rgb, scene, size=0.03, ceiling=None):
    """Splats the scan never captured: walls continued up to the ceiling, a ceiling and a backing floor.

    A hand-held scan at chest height rarely sees above ~2 m. Each wall column (vertical surface reaching above 1.1 m,
    higher than tables, beds and sofa backs)
    is extended from its highest point to the ceiling in the colour of its top 30 cm; the ceiling is a flat plane in
    the median wall-top colour. Returns (P, N, rgb) in the scan's frame, the ceiling height, and the wall and ceiling + floor counts.
    """
    from scipy import ndimage
    from scipy.spatial.transform import Rotation
    tr = scene['transform']
    R, sc, t = Rotation.from_quat(tr['rotation']).as_matrix(), tr['scale'][0], np.array(tr['position'])
    m, n = sc * P @ R.T + t, N @ R.T  # scene frame: y up, floor at 0, metres
    y = m[:, 1]
    if ceiling is None:  # the highest surfaces the scan did reach are usually the ceiling's edge
        ceiling = float(np.clip(np.quantile(y, 0.998), 2.3, 3.2))

    wall = (y > 1.1) & (y < ceiling - size) & (np.abs(n[:, 1]) < 0.3)
    cell = np.floor(m[wall][:, [0, 2]] / size).astype(np.int64)
    # only wall-like runs: a thin pole (floor lamp, door stop) has few neighbouring columns, a wall has a line of them
    occ = np.zeros(tuple(cell.max(0) - cell.min(0) + 1), bool)
    occ[tuple((cell - cell.min(0)).T)] = True
    nb = ndimage.uniform_filter(occ.astype(np.float32), size=7, mode='constant') * 49
    wide = nb[tuple((cell - cell.min(0)).T)] > 6.5
    wall[np.flatnonzero(wall)[~wide]] = False
    cell = cell[wide]
    _, col = np.unique(cell, axis=0, return_inverse=True)
    col = col.ravel()
    top = np.full(col.max() + 1, -np.inf)
    np.maximum.at(top, col, y[wall])
    near = y[wall] > top[col] - 0.3
    cnt = np.bincount(col[near], minlength=len(top))[:, None]
    mean = lambda a: np.stack([np.bincount(col[near], weights=a[near, k], minlength=len(top)) for k in range(a.shape[1])], 1) / cnt
    xz, cn, cc = mean(m[wall][:, [0, 2]]), mean(n[wall]), mean(rgb[wall])
    steps = np.maximum(0, np.floor((ceiling - top) / size)).astype(int)
    rep = np.repeat(np.arange(len(top)), steps)
    k = np.arange(len(rep)) - np.repeat(np.cumsum(steps) - steps, steps)
    wp = np.c_[xz[rep, 0], top[rep] + (k + 1) * size, xz[rep, 1]]
    wn, wc = cn[rep], cc[rep]

    rows, cs = scene['rows'], scene['cellSize']
    known = np.array([[c != ' ' for c in r] for r in rows])
    known = ndimage.binary_fill_holes(ndimage.binary_closing(known, iterations=3))  # no gaps in the ceiling
    iz, ix = np.nonzero(known)
    g = np.arange(0, cs, size) + size / 2
    gx, gz = np.meshgrid(g, g)
    cx = (scene['origin'][0] + ix[:, None] * cs + gx.ravel()).ravel()
    cz = (scene['origin'][1] + iz[:, None] * cs + gz.ravel()).ravel()
    cp = np.c_[cx, np.full(len(cx), ceiling), cz]
    shade = np.median(cc, 0) if len(cc) else np.array([0.85, 0.85, 0.85])
    # floor under furniture and in corners the scanner never looked at: 1 cm below the scanned floor so it only shows
    # through holes, in the floor's median colour
    fl = (np.abs(y) < 0.05) & (n[:, 1] > 0.8)
    fp = np.c_[cx, np.full(len(cx), -0.01), cz]
    fshade = np.median(rgb[fl], 0) if fl.any() else shade
    out_m = np.r_[wp, cp, fp]
    out_n = np.r_[wn, np.tile([0.0, -1.0, 0.0], (len(cp), 1)), np.tile([0.0, 1.0, 0.0], (len(fp), 1))]
    out_c = np.r_[wc, np.tile(shade, (len(cp), 1)), np.tile(fshade, (len(fp), 1))]
    back = ((out_m - t) / sc) @ R  # scene frame -> scan frame
    return back, out_n @ R, np.clip(out_c, 0, 1), ceiling, len(wp), len(cp) + len(fp)


def read_points(path):
    """(xyz Nx3 float64, rgb Nx3 0..1 or None) from a binary PLY point cloud (e.g. a LIDAR scan)."""
    with open(path, 'rb') as f:
        header = []
        while (line := f.readline().decode('ascii').strip()) != 'end_header':
            header.append(line)
        props = [(l.split()[2], navgrid.PLY_TYPES[l.split()[1]]) for l in header if l.startswith('property')]
        n = int(next(l.split()[2] for l in header if l.startswith('element vertex')))
        a = np.fromfile(f, dtype=props, count=n)
    rgb = np.stack([a['red'], a['green'], a['blue']], 1) / 255 if 'red' in a.dtype.names else None
    return np.stack([a['x'], a['y'], a['z']], 1).astype(np.float64), rgb


def level(up):
    """Rotation taking `up` to +y (Rodrigues)."""
    u = up / np.linalg.norm(up)
    v, c = np.cross(u, [0, 1, 0]), u[1]
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c) if c > -0.999 else np.diag([1.0, -1, -1])


def floor_height(y):
    """The floor is the densest height band in the lower half of the scan."""
    h, e = np.histogram(y[y < np.median(y)], bins=200)
    return e[np.argmax(h)]


def register_lidar(L, V, up, log=print):
    """4x4 taking LIDAR points L into the frame of mesh V (both metric): level both, find the yaw and shift that best
    overlay their floor plans (wall slice 1.0-1.8 m, FFT cross-correlation per 1 deg), then refine with ICP."""
    from scipy.spatial import cKDTree
    Lp, Vp = L[voxel_pick(L, 0.03)], V[voxel_pick(V, 0.03)]
    Rm = level(up)
    M = Vp @ Rm.T
    M[:, 1] -= floor_height(M[:, 1])
    c, size = 0.05, 800

    def raster(P):
        g = np.zeros((size, size), np.float32)
        ij = np.floor((P - P.mean(0)) / c).astype(int) + size // 2
        ok = (ij >= 0).all(1) & (ij < size).all(1)
        np.add.at(g, (ij[ok, 0], ij[ok, 1]), 1)
        return np.minimum(g, 3)

    wall = lambda P: P[(P[:, 1] > 1.0) & (P[:, 1] < 1.8)][:, [0, 2]]
    A = wall(M)
    Fa = np.fft.rfft2(raster(A))
    best = (-1,)
    for axis in range(3):  # the scanner's up axis is unknown: try each axis and sign
        for sign in (1, -1):
            up_l = np.zeros(3)
            up_l[axis] = sign
            Rl = level(up_l)
            X = Lp @ Rl.T
            X[:, 1] -= floor_height(X[:, 1])
            B = wall(X)
            if len(B) < 1000:
                continue
            for deg in range(360):
                th = np.radians(deg)
                Br = B @ np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]).T
                cc = np.fft.irfft2(Fa * np.conj(np.fft.rfft2(raster(Br))), s=(size, size))
                k = np.unravel_index(np.argmax(cc), cc.shape)
                if cc[k] > best[0]:
                    best = (cc[k], Rl, X, th, (np.array(k) + size // 2) % size - size // 2, Br.mean(0))
    _, Rl, X, th, k, bm = best
    R2 = np.eye(3)
    R2[np.ix_([0, 2], [0, 2])] = [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
    T2 = np.zeros(3)
    T2[[0, 2]] = A.mean(0) - bm + k * c
    lift = np.array([0, -floor_height((Lp @ Rl.T)[:, 1]), 0])
    R, T = R2 @ Rl, R2 @ lift + T2  # lidar -> levelled mesh frame, coarse
    Y = Lp @ R.T + T
    tree = cKDTree(M)
    for it in range(30):  # point-to-point ICP, tightening the match radius
        d, i = tree.query(Y, distance_upper_bound=0.3 if it < 10 else 0.1)
        ok = np.isfinite(d)
        p, q = Y[ok], M[i[ok]]
        U, _, Wt = np.linalg.svd((p - p.mean(0)).T @ (q - q.mean(0)))
        Ri = (U @ Wt).T
        if np.linalg.det(Ri) < 0:
            Wt[-1] *= -1
            Ri = (U @ Wt).T
        Ti = q.mean(0) - Ri @ p.mean(0)
        Y = Y @ Ri.T + Ti
        R, T = Ri @ R, Ri @ T + Ti
    d, _ = tree.query(Y)
    log(f'  LIDAR registered: {np.mean(d < 0.05):.0%} of its points within 5 cm of the mesh (median {np.median(d[d < 0.1]) * 100:.1f} cm '
        'where they overlap; the rest is what the RGB-D scan missed)')
    floor_m = floor_height((Vp @ Rm.T)[:, 1])
    out = np.eye(4)  # undo the mesh levelling: mesh = Rm^T (levelled + floor)
    out[:3, :3] = Rm.T @ R
    out[:3, 3] = Rm.T @ (T + [0, floor_m, 0])
    return out


def point_normals(P, k=12):
    """Normal per point: the least-variance direction of its k nearest neighbours."""
    from scipy.spatial import cKDTree
    N = np.empty_like(P)
    tree = cKDTree(P)
    for s in range(0, len(P), 500_000):
        _, i = tree.query(P[s:s + 500_000], k)
        Q = P[i] - P[i].mean(1, keepdims=True)
        N[s:s + 500_000] = np.linalg.eigh(np.einsum('nki,nkj->nij', Q, Q))[1][:, :, 0]
    return N


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mesh', type=Path, required=True)
    ap.add_argument('--log', type=Path, required=True, help='camera trajectory (.log)')
    ap.add_argument('--frames', type=Path, required=True, help='folder with image/ and depth/')
    ap.add_argument('--name', required=True, help='scene id: web/public/scenes/<name>')
    ap.add_argument('--title', help='display name')
    ap.add_argument('--intrinsics', default='525,525,319.5,239.5', help='fx,fy,cx,cy (default: PrimeSense)')
    ap.add_argument('--voxel', type=float, default=0.012, help='splat spacing in metres')
    ap.add_argument('--lidar', type=Path, help='coloured LIDAR point cloud of the same space: replaces the mesh geometry')
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

    cams = (poses[:, :3, 3], poses[:, :3, 2], -poses[:, :3, 1])  # centre, forward, image-up
    up_hint = np.median(cams[2], 0)
    if a.lidar:
        # a tripod laser scan sees ceilings and every wall, in its own colours; the camera path only places it
        print(f'[3/5] LIDAR scan {a.lidar.name}: registering it to the RGB-D mesh')
        L, Lrgb = read_points(a.lidar)
        if Lrgb is None:
            sys.exit('the LIDAR scan has no colours')
        G = register_lidar(L, V, up_hint)
        del V, tri
        sel = voxel_pick(L, a.voxel)
        P, rgb = L[sel] @ G[:3, :3].T + G[:3, 3], Lrgb[sel]
        del L, Lrgb
        print(f'[4/5] normals for {len(P):,} points')
        N = point_normals(P)
        v = splats(P, N, rgb, a.voxel, spread=1.1)  # scan points are unevenly spaced: overlap more, no see-through specks
    else:
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
        P, N, rgb = P[keep], N[keep], rgb[keep]
        v = splats(P, N, rgb, a.voxel)

    print('[5/5] floor, walls, walkable grid')
    scene = navgrid.build(v, cams, up=up_hint, scale=1.0, cell=0.1)  # the scan is metric already
    if not a.lidar:
        # look from where the scanner looked: a hand-held scan rarely sees above ~2 m, so a 1.6 m eye peers over walls
        from scipy.spatial.transform import Rotation
        tr = scene['transform']
        cam_y = cams[0] @ Rotation.from_quat(tr['rotation']).as_matrix()[1] * tr['scale'][1] + tr['position'][1]
        scene['eyeHeight'] = round(float(np.clip(np.median(cam_y) - scene['floorY'], 1.1, 1.7)), 2)
        print(f"  eye height {scene['eyeHeight']} m (median scanner height)")
        fill_p, fill_n, fill_c, ceiling, nw, nc = fill_above(P, N, rgb, scene)
        v = np.concatenate([v, splats(fill_p, fill_n, fill_c, 0.03)])
        scene['wallHeight'] = round(ceiling, 2)
        print(f'  ceiling at {ceiling:.2f} m: added {nw:,} wall splats above the scan and {nc:,} ceiling + floor splats')
    a.work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    ply = a.work / f'{a.name}_splats.ply'
    navgrid.write_ply(v, ply)
    navgrid.export_scene(v, scene, out, a.title or a.name, str(ply))
    print(f'done in {(time.time() - t) / 60:.1f} min: open the viewer with ?scene={a.name}')


if __name__ == '__main__':
    main()
