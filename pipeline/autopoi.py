"""Auto-label the twin: detect furniture/appliances in the training frames and place them as POIs.

COCO object detector (torchvision Faster R-CNN, BSD) on the frames -> 3D position of each box from the
trained splat's rendered depth -> cluster across views -> keep objects seen from >= 3 frames.

  python pipeline/autopoi.py --scene web/public/scenes/home --splat work/home/export/splat.ply \
      --transforms work/home/colmap/transforms.json --dataparser <run>/dataparser_transforms.json

RGB-D scans (import_rgbd.py) have real depth and known poses, so no splat rendering is needed:

  python pipeline/autopoi.py --scene web/public/scenes/apartment --log apartment.log --frames rgbd/apartment

Rooms are labelled too, from what was found in them (a bed makes a bedroom, a sofa a living room, ...).
"""
import argparse
import json
from pathlib import Path

import numpy as np

# COCO class -> POI name (what someone would search for indoors)
LABELS = {'bed': 'Bed', 'couch': 'Sofa', 'chair': 'Chair', 'tv': 'TV', 'refrigerator': 'Fridge', 'oven': 'Oven',
          'microwave': 'Microwave', 'sink': 'Sink', 'toilet': 'Toilet', 'dining table': 'Table', 'potted plant': 'Plant',
          'laptop': 'Laptop', 'clock': 'Clock', 'bench': 'Bench', 'toaster': 'Toaster', 'suitcase': 'Suitcase'}
# objects that name the room they stand in
ROOM_OF = {'bed': 'Bedroom', 'couch': 'Living Room', 'refrigerator': 'Kitchen', 'oven': 'Kitchen',
           'microwave': 'Kitchen', 'toilet': 'Bathroom', 'dining table': 'Dining Area'}
# merge radius ~ half the object's size: one 2 m bed seen from both ends is still one bed
RADIUS = {'bed': 2.0, 'couch': 1.6, 'dining table': 1.0, 'bench': 1.0, 'refrigerator': 0.8}
# floor furniture cannot float: centre height must be below this (metres)
MAX_HEIGHT = {'bed': 1.0, 'couch': 1.0, 'chair': 1.1, 'dining table': 1.1, 'bench': 0.9, 'toilet': 0.9,
              'refrigerator': 1.6, 'potted plant': 1.6, 'suitcase': 1.0}


def detect(paths, min_score):
    import torch
    from PIL import Image
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    dev = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, box_score_thresh=min_score).eval().to(dev)
    names = weights.meta['categories']
    tfm = weights.transforms()
    out = []
    with torch.no_grad():
        for p in paths:
            img = tfm(Image.open(p).convert('RGB')).to(dev)
            r = model([img])[0]
            dets = [(names[l], float(s), b.tolist()) for l, s, b in zip(r['labels'], r['scores'], r['boxes'])
                    if names[l] in LABELS]
            out.append(dets)
    return out


def cluster(points, radius):
    """Greedy clustering on xz; returns list of index arrays."""
    left, groups = list(range(len(points))), []
    while left:
        seed = left.pop(0)
        members = [seed]
        centre = points[seed]
        for _ in range(3):  # re-centre a few times
            members = [i for i in [seed] + left if np.linalg.norm(points[i][[0, 2]] - centre[[0, 2]]) < radius]
            centre = np.median(points[members], 0)
        left = [i for i in left if i not in members]
        groups.append(np.array(members))
    return groups


def splat_hits(a, scene):
    """Detections placed with depth rendered from the trained splat (video captures)."""
    import depthcarve
    from navgrid import read_ply
    tf = json.loads(Path(a.transforms).read_text())
    frames = sorted(tf['frames'], key=lambda f: f['file_path'])[::a.every]
    root = Path(a.transforms).parent
    views, K, w, h = depthcarve.load_views(a.transforms, a.dataparser, max_views=10 ** 9)
    views = views[::a.every]
    print(f'detecting objects in {len(frames)} frames')
    dets = detect([root / f['file_path'] for f in frames], a.min_score)
    if not any(dets):
        return {}
    v = read_ply(a.splat)
    width = 320
    depths, alphas, Ks = depthcarve.render_depths(v, views, K, w, h, width=width)
    return locate(dets, views, Ks, width / w, scene, lambda fi, box: depths[fi][box][alphas[fi][box] > 0.6])


def rgbd_hits(a, scene):
    """Detections placed with the scan's own depth images and camera path (import_rgbd.py scenes)."""
    import import_rgbd as ir
    poses = ir.read_log(Path(a.log))
    frames = Path(a.frames)
    ids = np.arange(0, min(len(poses), len(list((frames / 'image').glob('*.jpg')))), a.every)
    print(f'detecting objects in {len(ids)} frames')
    dets = detect([ir.frame_paths(frames, i)[0] for i in ids], a.min_score)
    if not any(dets):
        return {}
    k = np.array([float(x) for x in a.intrinsics.split(',')])
    Ks = np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]])
    depth = lambda fi, box: ir.load_frame(frames, ids[fi])[1][box]
    return locate(dets, poses[ids], Ks, 1.0, scene, depth)


def locate(dets, views, Ks, sc, scene, depth_at):
    """label -> [(scene-frame point, score, frame)] from the median depth over each box's centre."""
    n_det = sum(len(d) for d in dets)
    print(f'  {n_det} detections of {len({x[0] for d in dets for x in d})} kinds')
    tr = scene['splat']
    from scipy.spatial.transform import Rotation  # noqa: E402  (scipy ships with nerfstudio)
    R = Rotation.from_quat(tr['rotation']).as_matrix()
    s, t = tr['scale'][0], np.array(tr['position'])
    hits = {}  # label -> list of (metric point, score, frame)
    for fi, frame_dets in enumerate(dets):
        for label, score, (x0, y0, x1, y1) in frame_dets:
            # median depth over the box centre (inner 40%), where the object itself is most likely
            cx0, cx1 = int((x0 + 0.3 * (x1 - x0)) * sc), int((x1 - 0.3 * (x1 - x0)) * sc) + 1
            cy0, cy1 = int((y0 + 0.3 * (y1 - y0)) * sc), int((y1 - 0.3 * (y1 - y0)) * sc) + 1
            d = depth_at(fi, np.s_[cy0:cy1, cx0:cx1])
            d = d[d > 0]
            if len(d) < 3:
                continue
            u, vv = (x0 + x1) / 2 * sc, (y0 + y1) / 2 * sc
            ray = np.array([(u - Ks[0, 2]) / Ks[0, 0], (vv - Ks[1, 2]) / Ks[1, 1], 1.0])
            p = views[fi][:3, :3] @ (ray * np.median(d)) + views[fi][:3, 3]
            m = s * (R @ p) + t
            if -0.2 < m[1] < MAX_HEIGHT.get(label, 3.2):  # reject depth outliers (box centre hit the wall behind)
                hits.setdefault(label, []).append((m, score, fi))
    return hits


def nearest_walkable(scene, x, z):
    rows, cell, (ox, oz) = scene['rows'], scene['cellSize'], scene['origin']
    free = np.array([[c == '.' for c in r] for r in rows])
    iz, ix = np.nonzero(free)
    i = np.argmin((ox + (ix + 0.5) * cell - x) ** 2 + (oz + (iz + 0.5) * cell - z) ** 2)
    return round(float(ox + (ix[i] + 0.5) * cell), 2), round(float(oz + (iz[i] + 0.5) * cell), 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene', required=True, help='scene folder with scene.json')
    ap.add_argument('--splat')
    ap.add_argument('--transforms')
    ap.add_argument('--dataparser')
    ap.add_argument('--log', help='RGB-D: camera trajectory (.log)')
    ap.add_argument('--frames', help='RGB-D: folder with image/ and depth/')
    ap.add_argument('--intrinsics', default='525,525,319.5,239.5', help='RGB-D: fx,fy,cx,cy')
    ap.add_argument('--min-score', type=float, default=0.6)
    ap.add_argument('--min-views', type=int, default=3)
    ap.add_argument('--min-share', type=float, default=0.0,
                    help='also require being seen in this share of the frames (RGB-D default 0.015: long scans '
                         'glimpse a false "toilet" a few times)')
    ap.add_argument('--every', type=int, default=2, help='use every n-th frame')
    a = ap.parse_args()

    scene_file = Path(a.scene) / 'scene.json'
    scene = json.loads(scene_file.read_text())
    if a.log:
        hits = rgbd_hits(a, scene)
    elif a.splat and a.transforms and a.dataparser:
        hits = splat_hits(a, scene)
    else:
        ap.error('pass --splat/--transforms/--dataparser, or --log/--frames for an RGB-D scan')

    pois = [p for p in scene.get('pois', []) if not str(p.get('id', '')).startswith('auto-')]  # re-runs replace
    n_frames = 1 + max((x[2] for hl in hits.values() for x in hl), default=0)
    min_views = max(a.min_views, round((a.min_share or (0.015 if a.log else 0)) * n_frames))
    found = []  # (label, centre, views, confidence)
    for label, hl in hits.items():
        pts = np.array([x[0] for x in hl])
        for g in cluster(pts, radius=RADIUS.get(label, 0.6)):
            frames_seen = {hl[i][2] for i in g}
            if len(frames_seen) >= min_views:
                found.append((label, np.median(pts[g], 0), len(frames_seen), float(np.mean([hl[i][1] for i in g]))))
    # detectors call a sofa a bed from some angles: a sofa in the same spot, seen at least half as often, wins
    near = lambda f, k: np.hypot(k[1][0] - f[1][0], k[1][2] - f[1][2]) < 0.7
    found = [f for f in found if not (f[0] == 'bed' and any(k[0] == 'couch' and near(f, k) and k[2] >= f[2] / 2 for k in found))]
    # most-seen first; drop a weaker one of the same kind within its merge radius (a bed split in two) or any other
    # kind within 0.7 m (one sofa called "bed" from some angles)
    kept = []
    for f in sorted(found, key=lambda f: -f[2]):
        d = lambda k: np.hypot(k[1][0] - f[1][0], k[1][2] - f[1][2])
        if not any(d(k) < (RADIUS.get(f[0], 0.6) if k[0] == f[0] else 0.7) for k in kept):
            kept.append(f)
    added, anchors = [], []
    for label, c, views, conf in kept:
        added.append({'name': LABELS[label], 'category': 'item', 'x': round(float(c[0]), 2), 'z': round(float(c[2]), 2),
                      'y': round(float(min(max(c[1], 0.4), 2.0)), 2),
                      'desc': f'auto-detected · seen in {views} views · confidence {conf:.0%}', 'views': views})
        if label in ROOM_OF:
            anchors.append((ROOM_OF[label], c, views))
    # rooms: one label per room kind and place (a second bed 3+ m away is a second bedroom), on a walkable spot
    rooms = []
    for name, c, views in sorted(anchors, key=lambda r: -r[2]):
        if any(r[0] == name and np.hypot(r[1][0] - c[0], r[1][2] - c[2]) < 3 for r in rooms):
            continue
        rooms.append((name, c, views))
    for name, c, views in rooms:
        x, z = nearest_walkable(scene, c[0], c[2])
        added.append({'name': name, 'category': 'room', 'x': x, 'z': z, 'desc': 'auto-labelled from what is in it',
                      'views': 10 ** 6 + views})
    # number duplicates ("Chair 2") most-seen first
    added.sort(key=lambda p: -p['views'])
    count = {}
    for p in added:
        count[p['name']] = count.get(p['name'], 0) + 1
        if count[p['name']] > 1:
            p['name'] = f"{p['name']} {count[p['name']]}"
        p['id'] = 'auto-' + p['name'].lower().replace(' ', '-')
        del p['views']
    scene['pois'] = pois + added
    scene_file.write_text(json.dumps(scene, indent=1))
    print(f'  placed {len(added)} POIs: ' + ', '.join(p['name'] for p in added))

if __name__ == '__main__':
    main()
