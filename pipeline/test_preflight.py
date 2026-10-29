"""Self-check for the capture pre-flight. Run: python pipeline/test_preflight.py"""
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from preflight import (FPS, clip_issues, frame_metrics, narrow_regions, read_images_bin, read_points_bin,
                       review, stretches)

rng = np.random.default_rng(0)


def box_blur(a, k):
    for axis in (0, 1):
        c = np.cumsum(np.pad(a, [(k, k) if i == axis else (0, 0) for i in (0, 1)], mode='edge'), axis=axis)
        a = (np.take(c, range(2 * k, c.shape[axis]), axis) - np.take(c, range(0, c.shape[axis] - 2 * k), axis)) / (2 * k)
    return a


# a textured "room" panorama; the camera pans across it
pano = box_blur(rng.uniform(0, 255, (300, 6000)), 2)
pano = (pano - pano.min()) / np.ptp(pano) * 200 + 30


def clip(speeds, blur_at=(), dark_at=(), blank_at=()):
    """One frame per 1/FPS s; speeds = px per frame at 480 px wide."""
    x, frames = 0.0, []
    for i, v in enumerate(speeds):
        a = pano[15:285, int(x):int(x) + 480].copy()
        if i in blur_at:
            a = box_blur(a, 6)
        if i in dark_at:
            a *= 0.1
        if i in blank_at:
            a = np.full_like(a, 128) + rng.normal(0, 0.3, a.shape)
        frames.append(a.astype(np.float32))
        x += v
    return frames


assert stretches([0, 1, 1, 1, 0, 0, 0, 0, 1], fps=1, min_s=2) == [(1.0, 4.0)]
assert stretches([1, 0, 1, 1], fps=1, min_s=1, gap=1) == [(0.0, 4.0)]
print('ok: stretches')

# 10 s calm walk with a 3.5 s blurred stretch at 4 s
n = 10 * FPS
blur = set(range(4 * FPS, 4 * FPS + 28))
issues, frac = clip_issues(frame_metrics(clip([5] * n, blur_at=blur)), 'a.mp4')
kinds = {i['kind'] for i in issues}
b = [i for i in issues if i['kind'] == 'blur']
assert b and b[0]['severity'] == 'fail' and abs(b[0]['t0'] - 4) <= 0.5 and abs(b[0]['t1'] - 7.5) <= 0.5, issues
assert 'turn' not in kinds, issues
print(f'ok: blurred stretch found at {b[0]["t0"]:.1f}-{b[0]["t1"]:.1f} s, sharp fraction {frac:.0%}')

# calm, then 3 s of a fast pan (~65 deg/s), then calm
m = frame_metrics(clip([5] * 3 * FPS + [60] * 3 * FPS + [5] * 3 * FPS))
calm = np.nanmedian(m['turn'][1:3 * FPS])
assert 3 < calm < 9, calm  # 5 px/frame at 480 px, 65 deg FOV, 8 fps -> ~5.4 deg/s
issues, _ = clip_issues(m, 'b.mp4')
t = [i for i in issues if i['kind'] == 'turn']
assert t and t[0]['severity'] == 'fail' and abs(t[0]['t0'] - 3) <= 0.6, issues
print(f'ok: fast turn found at {t[0]["t0"]:.1f}-{t[0]["t1"]:.1f} s (calm walk measured {calm:.1f} deg/s)')

issues, _ = clip_issues(frame_metrics(clip([5] * n, dark_at=set(range(16, 40)), blank_at=set(range(48, 76)))), 'c')
assert {'dark', 'blank'} <= {i['kind'] for i in issues if i['severity'] == 'fail'}, issues
issues, frac = clip_issues(frame_metrics(clip([5] * n)), 'd.mp4')
assert not [i for i in issues if i['severity'] == 'fail'] and frac > 0.9, (issues, frac)
print('ok: dark and blank-wall stretches fail, a clean walk passes')


# view angles: a wall at x = 5 seen head-on from one spot vs walked along
wall = np.c_[np.full(3000, 5.0), rng.uniform(-3, 3, 3000), rng.uniform(0, 2.5, 3000)]
up = np.array([0, 0, 1.0])
def cams(pts):
    return {i + 1: np.array(p) for i, p in enumerate(pts)}
head_on = cams([(0, y, 1.4) for y in np.linspace(-0.3, 0.3, 8)])
along = cams([(4, y, 1.4) for y in np.linspace(-3, 3, 13)] + [(0, y, 1.4) for y in (-0.3, 0.3)])
all_seen = lambda c: [np.array(list(c), dtype=np.int32)] * len(wall)
bad = narrow_regions(wall, all_seen(head_on), head_on, up)
good = narrow_regions(wall, all_seen(along), along, up)
assert bad and not good, (len(bad), len(good))
print(f'ok: head-on-only wall flagged ({bad[0]["spread"]:.0f} deg), walked-along wall passes')


# COLMAP binary readers round-trip
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    with open(d / 'images.bin', 'wb') as f:
        f.write(struct.pack('<Q', 2))
        for iid, name in ((3, 'c00_00001.jpg'), (7, 'c01_00042.jpg')):
            f.write(struct.pack('<i7di', iid, 1, 0, 0, 0, 1, 2, 3, 1) + name.encode() + b'\0')
            f.write(struct.pack('<Q', 1) + struct.pack('<ddq', 5.0, 6.0, -1))
    with open(d / 'points3D.bin', 'wb') as f:
        f.write(struct.pack('<Q', 1) + struct.pack('<Q3d3BdQ', 9, 1, 2, 3, 0, 0, 0, 0.5, 2) + struct.pack('<4i', 3, 0, 7, 0))
    ims = read_images_bin(d / 'images.bin')
    xyz, tracks = read_points_bin(d / 'points3D.bin')
    assert ims[7][0] == 'c01_00042.jpg' and np.allclose(ims[3][2], [1, 2, 3]) and np.allclose(ims[3][1], np.eye(3))
    assert np.allclose(xyz, [[1, 2, 3]]) and tracks[0].tolist() == [3, 7]
print('ok: COLMAP images.bin / points3D.bin readers')


# end to end on an encoded video (metrics only): needs ffmpeg
if shutil.which('ffmpeg'):
    from PIL import Image
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for i, a in enumerate(clip([5] * 3 * FPS + [5] * 4 * FPS + [5] * 3 * FPS, blur_at=set(range(24, 56)))):
            Image.fromarray(a.clip(0, 255).astype(np.uint8)).save(d / f'{i:05d}.png')
        subprocess.run(['ffmpeg', '-v', 'error', '-framerate', str(FPS), '-i', d / '%05d.png', '-pix_fmt', 'yuv420p',
                        '-crf', '12', d / 'walk.mp4'], check=True)
        r = review([d / 'walk.mp4'], d / 'check', log=lambda *_: None, quick_sfm=False)
        assert not r['pass'] and r['retakes'] and r['checks'][1]['status'] == 'skipped', r
        assert (d / 'check' / 'report.json').exists()
        shutil.copy(d / 'walk.mp4', d / 'retake.mp4')  # stands in for a clean retake
        r = review([d / 'walk.mp4', d / 'retake.mp4'], d / 'check2', log=lambda *_: None, quick_sfm=False,
                   retake_from=1)
        warned = [i for i in r['issues'] if i['clip'] == 'walk.mp4' and 'covered by a retake' in i['message']]
        assert warned and all(i['severity'] == 'warn' for i in warned), r['issues']
    print('ok: review() fails the blurred take, suggests a retake, and treats the stretch as covered once it is filmed')
else:
    print('skip: ffmpeg not on PATH, end-to-end video check not run')
