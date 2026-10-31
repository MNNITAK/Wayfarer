"""RGB-D scan (mesh + trajectory + frames) -> nerfstudio dataset, to train a photoreal splat with the known poses.

  python pipeline/rgbd_to_nerfstudio.py --mesh apartment.ply --log apartment.log --frames rgbd/apartment --out work/apartment_ns

No COLMAP: the scan's metric camera path becomes transforms.json directly. From every window of --step frames the
sharpest one whose pose agrees with the mesh (same depth test as import_rgbd) is kept. The mesh, coloured from those
frames, becomes sparse_pc.ply, which splatfacto uses as its starting point cloud instead of random points.
Train with the same ns-train flags as wayfarer.py (locally on a GPU, or on Kaggle: see kaggle_train.py).
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

import import_rgbd as ir

OPENCV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])  # nerfstudio poses use OpenGL axes (y up, z backwards)


def sharpness(path):
    im = Image.open(path)
    im.draft('L', (im.width // 2, im.height // 2))  # JPEG decodes at half size: ~4x faster, same ranking
    g = np.asarray(im.convert('L'), np.float32)
    lap = g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1] - 4 * g[1:-1, 1:-1]
    return float(lap.var())


def pose_ok(P, c2w, frames, i, K):
    depth = ir.load_frame(frames, i)[1]
    h, w = depth.shape
    u, v, z = ir.project(P, c2w, K)
    ok = (z > 0.2) & (z < 5) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
    return ir.depth_agree(depth[v[ok].astype(int), u[ok].astype(int)], z[ok])[1]


def pick_frames(poses, frames, K, P, step, log=print):
    """Sharpest frame per window of `step` whose pose agrees with the mesh (tries the 3 sharpest)."""
    t0, picked, bad = time.time(), [], 0
    for s in range(0, len(poses), step):
        window = list(range(s, min(s + step, len(poses))))
        for i in sorted(window, key=lambda i: -sharpness(ir.frame_paths(frames, i)[0]))[:3]:
            if pose_ok(P, poses[i].astype(np.float32), frames, i, K):
                picked.append(i)
                break
        else:
            bad += 1
        if (s // step) % 300 == 0:
            log(f'  window {s // step + 1}/{-(-len(poses) // step)}: {len(picked)} kept, {bad} without a good pose '
                f'({time.time() - t0:.0f} s)')
    log(f'  {len(picked)} frames kept, {bad} windows dropped for a bad pose')
    return picked


def write_points(path, P, rgb):
    """PLY with float xyz + uchar rgb: the format nerfstudio reads for ply_file_path."""
    out = np.zeros(len(P), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    out['x'], out['y'], out['z'] = P.T
    c = np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8)
    out['red'], out['green'], out['blue'] = c.T
    with open(path, 'wb') as f:
        f.write((f'ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\n'
                 'property float x\nproperty float y\nproperty float z\n'
                 'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode())
        out.tofile(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mesh', type=Path, required=True)
    ap.add_argument('--log', type=Path, required=True, help='camera trajectory (.log)')
    ap.add_argument('--frames', type=Path, required=True, help='folder with image/ and depth/')
    ap.add_argument('--out', type=Path, required=True, help='nerfstudio dataset folder to write')
    ap.add_argument('--intrinsics', default='525,525,319.5,239.5', help='fx,fy,cx,cy (default: PrimeSense)')
    ap.add_argument('--step', type=int, default=10, help='keep one frame per window of this many')
    ap.add_argument('--voxel', type=float, default=0.02, help='spacing of the starting point cloud in metres')
    a = ap.parse_args()
    K = np.array([float(x) for x in a.intrinsics.split(',')])

    t = time.time()
    print(f'[1/3] reading {a.mesh.name}')
    V, tri = ir.read_mesh(a.mesh)
    poses = ir.read_log(a.log)[:len(list((a.frames / 'image').glob('*.jpg')))]
    N = ir.vertex_normals(V, tri)
    sel = ir.voxel_pick(V, a.voxel)
    P, N = V[sel], N[sel]
    del V, tri
    print(f'  {len(poses):,} poses, {len(P):,} starting points ({time.time() - t:.0f} s)')

    print(f'[2/3] picking the sharpest well-posed frame of every {a.step}')
    check = P[::max(1, len(P) // 150_000)].astype(np.float32)  # enough points to judge a pose, cheap to project
    ids = pick_frames(poses, a.frames, K, check, a.step)

    print('[3/3] writing images, transforms.json and the coloured starting points')
    img_dir = a.out / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    w, h = Image.open(ir.frame_paths(a.frames, ids[0])[0]).size
    out_frames = []
    for i in ids:
        name = f'frame_{i:06d}.jpg'  # capture order sorts by name, which navgrid relies on
        shutil.copyfile(ir.frame_paths(a.frames, i)[0], img_dir / name)
        out_frames.append({'file_path': f'images/{name}', 'transform_matrix': (poses[i] @ OPENCV_TO_GL).tolist()})
    rgb, wsum = ir.colorize(P, N, poses, a.frames, K, ids[::3])
    keep = wsum > 0
    write_points(a.out / 'sparse_pc.ply', P[keep], rgb[keep])
    meta = {'camera_model': 'OPENCV', 'fl_x': K[0], 'fl_y': K[1], 'cx': K[2], 'cy': K[3], 'w': w, 'h': h,
            'k1': 0.0, 'k2': 0.0, 'p1': 0.0, 'p2': 0.0, 'ply_file_path': 'sparse_pc.ply', 'frames': out_frames}
    (a.out / 'transforms.json').write_text(json.dumps(meta, indent=1))
    print(f'done in {(time.time() - t) / 60:.1f} min: {len(ids)} frames, {keep.sum():,} points -> {a.out}')


if __name__ == '__main__':
    main()
