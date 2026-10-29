"""Capture pre-flight: check phone video(s) against pipeline/capture_requirements.md before the long build.

  python pipeline/preflight.py walk.mp4 [room2.mp4 ...] --out work/check/report.json

Checks run in order and stop at the first failure, so a bad take is caught in minutes, not after training:
  metrics  motion blur, turn speed, exposure jumps, dark and blank-wall stretches, with timestamps  (< 1 min, CPU)
  sfm      COLMAP on ~120 small frames: >= 90 % registered in one model, like the main pipeline's gate   (minutes)
  angles   every wall seen from a spread of viewing directions; head-on-only walls turn fibrous in the twin
Each failure comes with a short retake to film (a clip that starts and ends on views already filmed), not a re-shoot.
"""
import argparse
import json
import math
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

from wayfarer import colmap_sparse, run, step_frames

# numbers from capture_requirements.md
FPS = 8                 # candidate frames per second, as step_frames samples them
SHARP_RATIO = 0.55      # a frame below this fraction of the sharpest frame within +-2 s is motion-blurred
MIN_SHARP = 0.70        # fraction of sharp frames needed
HFOV = 65.0             # horizontal field of view (deg) of a phone's 1x main camera along the long side
MAX_TURN = 30.0         # deg/s
DARK = 35               # mean grey level (0..255) below which the phone exposes long and blurs
EXPOSURE_JUMP = 18      # grey-level jump between two frames 1/8 s apart while the view barely moved: AE not locked
BLANK = 2.5             # mean |gradient| (grey levels/px at 240 px wide) of a featureless wall
MIN_REGISTERED = 0.90   # quick SfM gate
MIN_SPREAD = 15.0       # deg between the rays that saw a wall point
QUICK_FRAMES, QUICK_SIDE = 120, 640


# ---------- per-frame measurements ----------
def _small(a, width=240):
    """Box-downsample a greyscale frame to about `width` px wide."""
    k = max(1, a.shape[1] // width)
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean((1, 3))


def image_shift(a, b):
    """Global translation (dx, dy) in px from frame a to b by phase correlation, and the peak strength (0..1)."""
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    A, B = np.fft.fft2((a - a.mean()) * win), np.fft.fft2((b - b.mean()) * win)
    R = A.conj() * B
    r = np.fft.ifft2(R / (np.abs(R) + 1e-9)).real
    y, x = np.unravel_index(np.argmax(r), r.shape)
    dy = y - r.shape[0] if y > r.shape[0] // 2 else y
    dx = x - r.shape[1] if x > r.shape[1] // 2 else x
    return float(dx), float(dy), float(r.max())


def frame_metrics(frames, fps=FPS):
    """frames: greyscale float arrays of one clip, `fps` apart. Returns per-frame sharpness, brightness, texture and
    rotation speed (deg/s; nan where the image motion could not be measured)."""
    sharp, bright, texture, turn = [], [], [], [np.nan]
    prev = None
    for a in frames:
        lap = a[1:-1, :-2] + a[1:-1, 2:] + a[:-2, 1:-1] + a[2:, 1:-1] - 4 * a[1:-1, 1:-1]
        sharp.append(float(lap.var()))
        s = _small(a)
        bright.append(float(s.mean()))
        texture.append(float(np.abs(np.diff(s, axis=0)).mean() + np.abs(np.diff(s, axis=1)).mean()) / 2)
        if prev is not None:
            dx, dy, peak = image_shift(prev, s)
            turn.append(math.hypot(dx, dy) / max(s.shape) * HFOV * fps if peak > 0.03 else np.nan)
        prev = s
    return {k: np.array(v) for k, v in
            (('sharpness', sharp), ('brightness', bright), ('texture', texture), ('turn', turn))}


# ---------- issues from measurements ----------
def stretches(mask, fps=FPS, min_s=1.0, gap=2):
    """(start, end) times in s of runs of True at least min_s long; gaps of <= `gap` frames are bridged."""
    out, start, last = [], None, None
    for i, on in enumerate(list(mask) + [False]):
        if on:
            if start is None or i - last > gap + 1:
                if start is not None and (last - start + 1) / fps >= min_s:
                    out.append((start / fps, (last + 1) / fps))
                start = i
            last = i
    if start is not None and (last - start + 1) / fps >= min_s:
        out.append((start / fps, (last + 1) / fps))
    return out


def fmt(t):
    t = max(0, t)
    return f'{int(t // 60)}:{int(t % 60):02d}'


RETAKE_FIX = {
    'blur': 'move slower, with more light',
    'turn': 'turn slowly: about 3 s per quarter turn',
    'dark': 'switch on the lights in that area first',
    'blank': 'stay at least 1 m from the wall and keep furniture, an edge or a corner in frame',
    'exposure': 'lock exposure and focus (long-press on the screen) before you start',
}


def clip_issues(m, clip, fps=FPS):
    """Issues for one clip from frame_metrics output."""
    issues = []
    s = m['sharpness']
    r = max(2, int(2 * fps))
    local = np.array([s[max(0, i - r):i + r + 1].max() for i in range(len(s))])
    blurred = (s < SHARP_RATIO * local) | (s < 0.3 * np.median(s))
    sharp_frac = 1 - blurred.mean() if len(s) else 0.0
    for t0, t1 in stretches(blurred, fps, 1.0):
        issues.append(dict(kind='blur', severity='fail' if t1 - t0 >= 3 else 'warn', clip=clip, t0=t0, t1=t1,
                           message=f'{fmt(t0)}–{fmt(t1)} motion-blurred'))
    turn = np.convolve(np.nan_to_num(m['turn']), np.ones(3) / 3, 'same')
    for t0, t1 in stretches(turn > MAX_TURN, fps, 0.5):
        issues.append(dict(kind='turn', severity='fail' if t1 - t0 >= 2 else 'warn', clip=clip, t0=t0, t1=t1,
                           message=f'{fmt(t0)}–{fmt(t1)} turned faster than {MAX_TURN:.0f}°/s'))
    for t0, t1 in stretches(m['brightness'] < DARK, fps, 1.0):
        issues.append(dict(kind='dark', severity='fail' if t1 - t0 >= 2 else 'warn', clip=clip, t0=t0, t1=t1,
                           message=f'{fmt(t0)}–{fmt(t1)} too dark'))
    for t0, t1 in stretches(m['texture'] < BLANK, fps, 1.5):
        issues.append(dict(kind='blank', severity='fail' if t1 - t0 >= 3 else 'warn', clip=clip, t0=t0, t1=t1,
                           message=f'{fmt(t0)}–{fmt(t1)} only a blank surface in view'))
    jumps = np.abs(np.diff(m['brightness'])) > EXPOSURE_JUMP
    still = np.nan_to_num(m['turn'][1:], nan=99) < 10
    for i in np.flatnonzero(jumps & still)[:5]:
        t = (i + 1) / fps
        issues.append(dict(kind='exposure', severity='warn', clip=clip, t0=t, t1=t + 1 / fps,
                           message=f'{fmt(t)} brightness jumped while the view stayed put'))
    return issues, float(sharp_frac)


def retake(issue):
    fix = RETAKE_FIX.get(issue['kind'], issue.get('fix', 'move slowly'))
    return dict(clip=issue['clip'], t0=issue['t0'], t1=issue['t1'],
                instruction=f"Film a short extra clip covering what you saw at {fmt(issue['t0'])}–{fmt(issue['t1'])} "
                            f"of {issue['clip']}: start on a view you already filmed, {fix}, and end on a view you "
                            'already filmed.')


def read_gray(paths):
    from PIL import Image
    return [np.asarray(Image.open(p).convert('L'), dtype=np.float32) for p in paths]


def check_metrics(clips, tmp, log=print, retake_from=None):
    """Clips before `retake_from` already have retakes filmed for their problems: those stretches only warn now."""
    issues, sharp, seconds = [], [], 0.0
    for i, src in enumerate(clips):
        d = tmp / f'm{i:02d}'
        d.mkdir(parents=True, exist_ok=True)
        run(['ffmpeg', '-v', 'error', '-i', src, '-vf', f'fps={FPS},scale=480:-2,format=gray', '-q:v', '3',
             d / '%05d.jpg'])
        frames = sorted(d.glob('*.jpg'))
        m = frame_metrics(read_gray(frames))
        found, frac = clip_issues(m, Path(src).name)
        if retake_from is not None and i < retake_from:
            for f in found:
                if f['severity'] == 'fail':
                    f['severity'], f['message'] = 'warn', f['message'] + ' (covered by a retake)'
        issues += found
        sharp.append(frac * len(frames))
        seconds += len(frames) / FPS
        log(f'  {Path(src).name}: {len(frames) / FPS:.0f} s, {frac:.0%} sharp, {len(found)} issue(s)')
    sharp_frac = sum(sharp) / max(1, seconds * FPS)
    if sharp_frac < MIN_SHARP:
        issues.insert(0, dict(kind='blur', severity='fail', clip=None, t0=None, t1=None,
                              message=f'only {sharp_frac:.0%} of frames are sharp (need {MIN_SHARP:.0%}): '
                                      'walk slower and add light'))
    return issues, {'sharpFraction': round(sharp_frac, 3), 'seconds': round(seconds, 1)}


# ---------- COLMAP model ----------
def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_images_bin(path):
    """{image_id: (name, R world->cam, t)} from a COLMAP images.bin."""
    out = {}
    with open(path, 'rb') as f:
        for _ in range(struct.unpack('<Q', f.read(8))[0]):
            iid, *qt, _cam = struct.unpack('<i7di', f.read(64))
            name = b''
            while (c := f.read(1)) != b'\0':
                name += c
            n2d = struct.unpack('<Q', f.read(8))[0]
            f.seek(24 * n2d, 1)
            out[iid] = (name.decode(), qvec2rotmat(qt[:4]), np.array(qt[4:]))
    return out


def read_points_bin(path):
    """(xyz Nx3, list of image-id arrays) from a COLMAP points3D.bin."""
    xyz, tracks = [], []
    with open(path, 'rb') as f:
        for _ in range(struct.unpack('<Q', f.read(8))[0]):
            _pid, x, y, z, _r, _g, _b, _err, n = struct.unpack('<Q3d3BdQ', f.read(51))
            xyz.append((x, y, z))
            tracks.append(np.frombuffer(f.read(8 * n), dtype='<i4')[::2].copy())
    return np.array(xyz).reshape(-1, 3), tracks


def frame_time(name, fps=FPS):
    """'c01_00042.jpg' -> (clip index 1, seconds into that clip)."""
    clip, idx = Path(name).stem.split('_')
    return int(clip[1:]), (int(idx) - 1) / fps


# ---------- view-angle coverage ----------
def ray_spread(point, centers):
    """Largest angle (deg) between the rays from a point to the cameras that saw it."""
    d = centers - point
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    c = np.clip(d @ d.T, -1, 1)
    return float(np.degrees(np.arccos(c.min())))


def narrow_regions(xyz, tracks, centers, up, cell_frac=1 / 12, min_points=30, min_spread=MIN_SPREAD):
    """Floor-plan cells whose points were seen from too narrow a range of directions (head-on only).
    centers: {image_id: camera centre}. Returns [{cell, points, spread, frame}] widest first, `frame` = the image
    id that saw most of the cell's points (where to show the user)."""
    ids = np.array(list(centers))
    C = np.array([centers[i] for i in ids])
    idx = {iid: k for k, iid in enumerate(ids)}
    up = up / np.linalg.norm(up)
    e1 = np.cross(up, [1, 0, 0] if abs(up[0]) < 0.9 else [0, 1, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    P2 = np.c_[xyz @ e1, xyz @ e2]
    lo, hi = np.percentile(np.r_[P2, np.c_[C @ e1, C @ e2]], [2, 98], axis=0)
    size = max(hi - lo) * cell_frac
    cells = {}
    for k, (p, tr) in enumerate(zip(xyz, tracks)):
        tr = [t for t in set(tr.tolist()) if t in idx]
        if len(tr) < 2:
            continue
        key = tuple(((P2[k] - lo) // size).astype(int))
        cells.setdefault(key, []).append((ray_spread(p, C[[idx[t] for t in tr]]), tr))
    out = []
    for key, pts in cells.items():
        if len(pts) < min_points:
            continue
        spread = float(np.median([s for s, _ in pts]))
        if spread < min_spread:
            seen = np.bincount(np.concatenate([np.array(tr) for _, tr in pts]))
            out.append(dict(cell=key, points=len(pts), spread=round(spread, 1), frame=int(np.argmax(seen))))
    return sorted(out, key=lambda r: -r['points'])


def camera_centers(images):
    """{image_id: centre}, and the world up direction (cameras are held upright: their image-down axis
    averages to world down)."""
    centers = {i: -R.T @ t for i, (_, R, t) in images.items()}
    down = np.mean([R.T @ np.array([0, 1, 0]) for _, R, _ in images.values()], axis=0)
    return centers, -down


# ---------- the review ----------
def review(clips, work, log=print, quick_sfm=None, retake_from=None):
    """Run the checks; returns the report dict (also written to work/report.json)."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    thumbs = work / 'thumbs'
    shutil.rmtree(thumbs, ignore_errors=True)
    thumbs.mkdir()
    names = [Path(c).name for c in clips]
    report = {'pass': False, 'clips': names, 'checks': [], 'issues': [], 'retakes': [], 'stats': {}}

    def done(name, status, summary, issues=()):
        report['checks'].append(dict(name=name, status=status, summary=summary))
        report['issues'] += list(issues)
        report['retakes'] += [retake(i) for i in issues if i['severity'] == 'fail' and i.get('clip')]
        log(f'  {name}: {status} - {summary}')
        return status != 'fail'

    def finish(ok):
        for name in ('metrics', 'sfm', 'angles'):
            if not any(c['name'] == name for c in report['checks']):
                report['checks'].append(dict(name=name, status='skipped', summary='after an earlier failure'))
        report['pass'] = ok
        (work / 'report.json').write_text(json.dumps(report, indent=1))
        return report

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        log('[1/3] metrics')
        try:
            issues, stats = check_metrics(clips, tmp, log, retake_from)
        except SystemExit as e:
            done('metrics', 'fail', f'could not read the video ({e})')
            return finish(False)
        report['stats'].update(stats)
        fails = [i for i in issues if i['severity'] == 'fail']
        summary = f"{stats['sharpFraction']:.0%} sharp, {len(fails)} problem(s), {len(issues) - len(fails)} warning(s)"
        if not done('metrics', 'fail' if fails else 'warn' if issues else 'pass', summary, issues):
            return finish(False)

    if quick_sfm is None:
        quick_sfm = shutil.which('colmap') is not None
    if not quick_sfm:
        done('sfm', 'skipped', 'COLMAP not found; the main build will check camera poses')
        return finish(True)

    log('[2/3] sfm (quick COLMAP)')
    frames, ns = work / 'frames', work / 'colmap'
    try:
        shutil.rmtree(frames, ignore_errors=True)
        step_frames([Path(c) for c in clips], frames, FPS, QUICK_FRAMES, QUICK_SIDE)
        n_reg, n_frames, n_models = colmap_sparse(frames, ns, 'sequential', 'global')
    except SystemExit as e:
        done('sfm', 'fail', f'COLMAP failed ({e})')
        return finish(False)
    report['stats'].update(framesRegistered=n_reg, framesChecked=n_frames, models=n_models)
    model = ns / 'colmap' / 'sparse' / '0'
    images = read_images_bin(model / 'images.bin') if n_models else {}
    ratio = n_reg / max(1, n_frames)
    issues = []
    if ratio < MIN_REGISTERED or n_models > 1:
        placed = {name for name, _, _ in images.values()}
        lost = [frame_time(p.name) for p in sorted(frames.iterdir()) if p.name not in placed]
        for c in sorted({c for c, _ in lost}):
            ts = np.array(sorted(t for k, t in lost if k == c))
            mask = np.zeros(int(ts.max() * FPS) + 2, bool)
            mask[(ts * FPS).astype(int)] = True
            for t0, t1 in stretches(mask, FPS, 0.1, gap=int(4 * FPS)):
                issues.append(dict(kind='tracking', severity='fail', clip=names[c], t0=t0, t1=t1,
                                   fix='walk through doorways slowly with both rooms in view, never film a blank wall '
                                       'up close',
                                   message=f'{fmt(t0)}–{fmt(t1)} of {names[c]} could not be placed with the rest'))
        summary = (f'{ratio:.0%} of frames placed (need {MIN_REGISTERED:.0%})' +
                   (f', split into {n_models} pieces' if n_models > 1 else ''))
        done('sfm', 'fail', summary, issues)
        return finish(False)
    done('sfm', 'pass', f'{ratio:.0%} of frames placed in one model')

    log('[3/3] view angles')
    xyz, tracks = read_points_bin(model / 'points3D.bin')
    centers, up = camera_centers(images)
    regions = narrow_regions(xyz, tracks, centers, up)[:5]
    issues = []
    for k, r in enumerate(regions):
        name = images[r['frame']][0]
        c, t = frame_time(name)
        thumb = f'thumbs/angle{k}.jpg'
        shutil.copy(frames / name, work / thumb)
        issues.append(dict(kind='angles', severity='fail', clip=names[c], t0=t, t1=t + 1, thumb=thumb,
                           fix='walk along this wall 0.5–1 m away with the camera pointing ahead and angled toward it',
                           message=f'the surface seen at {fmt(t)} of {names[c]} was only filmed from one direction '
                                   f'({r["spread"]:.0f}° of view angles, need {MIN_SPREAD:.0f}°)'))
    report['stats']['narrowRegions'] = len(regions)
    if not done('angles', 'fail' if issues else 'pass',
                f'{len(issues)} surface(s) seen from one direction only' if issues else 'walls seen from several angles',
                issues):
        return finish(False)
    return finish(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('clips', nargs='+', type=Path, help='capture video(s), in order')
    ap.add_argument('--work', type=Path, required=True, help='scratch dir; report.json and thumbs/ are written here')
    ap.add_argument('--no-sfm', action='store_true', help='skip the quick COLMAP and view-angle checks')
    ap.add_argument('--retake-from', type=int, help='index of the first retake clip; earlier clips\' local problems '
                                                    'count as covered')
    a = ap.parse_args()
    for c in a.clips:
        if not c.exists():
            sys.exit(f'input not found: {c}')
    r = review(a.clips, a.work, quick_sfm=False if a.no_sfm else None, retake_from=a.retake_from)
    print('PASS: ready to build' if r['pass'] else f"FAIL: {len(r['retakes'])} retake(s) suggested")
    for t in r['retakes']:
        print('  -', t['instruction'])
    sys.exit(0 if r['pass'] else 1)


if __name__ == '__main__':
    main()
