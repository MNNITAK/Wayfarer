"""Capture Coach agents: turn the generic capture rules into instructions for *this* room, verify fixes from photos,
and check that a finished take covered the plan. Their brief is pipeline/capture_requirements.md.

  python pipeline/coach.py plan photo1.jpg photo2.jpg ... --out plan.json
  python pipeline/coach.py verify --plan plan.json --item fan photo.jpg --out verdict.json
  python pipeline/coach.py review --plan plan.json take1.mp4 [take2.mp4 ...] --out coverage.json

Needs the `anthropic` package and an API key (ANTHROPIC_API_KEY, or an `ant auth login` profile).
WAYFARER_COACH_MODEL overrides the model.
"""
import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = os.environ.get('WAYFARER_COACH_MODEL', 'claude-opus-5-5')
IMAGE_SIDE = 1024        # px; enough to spot a fan or a mirror, small enough to keep requests cheap
REVIEW_FRAMES = 24


class CoachError(RuntimeError):
    pass


def requirements():
    return (ROOT / 'capture_requirements.md').read_text(encoding='utf-8')


def _obj(props, required=None):
    return {'type': 'object', 'properties': props, 'required': required or list(props), 'additionalProperties': False}


_str = {'type': 'string'}
_strs = {'type': 'array', 'items': _str}

PLAN_SCHEMA = _obj({
    'summary': _str,
    'rooms': _strs,
    'hazards': {'type': 'array', 'items': _obj({
        'id': _str, 'what': _str, 'where': _str, 'fix': _str,
        'photo_check': {'type': 'boolean'},
        'severity': {'type': 'string', 'enum': ['must', 'should']},
    })},
    'marker': _obj({'where': _str, 'why': _str}),
    'route': {'type': 'array', 'items': _obj({'id': _str, 'room': _str, 'title': _str, 'detail': _str, 'landmarks': _strs})},
    'settings': _strs,
    'notes': _strs,
})

VERDICT_SCHEMA = _obj({'passed': {'type': 'boolean'}, 'reason': _str})

COVERAGE_SCHEMA = _obj({
    'ok': {'type': 'boolean'},
    'missing': {'type': 'array', 'items': _obj({'step_id': _str, 'what': _str, 'retake': _str})},
})

PLAN_TASK = """These photos show the indoor space the user is about to film for a Wayfarer digital twin (one photo per
corner facing in, the doorways, the ceiling). Using the requirements above, write a capture plan for exactly this space:

- hazards: everything in these photos that would hurt the reconstruction (ceiling fans, mirrors, TVs/screens, glass,
  windows with direct sun, moving things, people or pets, dark areas, doors that are closed, personal documents or
  photos). Give each a short stable id (e.g. "fan-bedroom"), say where it is, and a concrete fix. photo_check is true
  when a new photo can show the fix was done (fan stopped and lights on can, "hold AE lock" cannot). severity "must"
  when it would break the build (moving fan, big mirror, closed door between rooms), otherwise "should".
- marker: where to stick the printed scale marker in the first room, flat, passed within 2 m.
- route: the filming route as ordered steps, naming real landmarks from the photos. Per room include the perimeter
  loop facing in, an along-the-wall pass for each wall (say which landmarks keep a plain wall from being blank), slow
  walks centre to wall and back, a low loop around big furniture, the ceiling sweep, doorways to the next room, and
  ending where the take started.
- settings: the phone settings to set before recording.
- notes: anything specific to this space (glossy floor, very plain walls, tight corridor, low light).
Only describe what you can see; if a photo is unclear, say so in notes instead of guessing."""


def image_block(path):
    """A downscaled JPEG of the photo/frame as a base64 image content block."""
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    im.thumbnail((IMAGE_SIDE, IMAGE_SIDE))
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=85)
    return {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg',
                                        'data': base64.standard_b64encode(buf.getvalue()).decode()}}


def ask(content, schema, effort='medium', client=None):
    """One structured request with the requirements as the system prompt; returns the parsed JSON."""
    import anthropic
    client = client or anthropic.Anthropic()
    try:
        r = client.beta.messages.create(
            model=MODEL, max_tokens=16000,
            system=requirements(),
            messages=[{'role': 'user', 'content': content}],
            output_config={'effort': effort, 'format': {'type': 'json_schema', 'schema': schema}},
            betas=['server-side-fallback-2026-07-01'], fallbacks='default',  # re-run on a fallback model if declined
        )
    except anthropic.AuthenticationError:
        raise CoachError('no valid Anthropic API key: set ANTHROPIC_API_KEY or run `ant auth login`')
    except anthropic.RateLimitError:
        raise CoachError('the AI coach is rate-limited right now; try again in a minute')
    except anthropic.APIStatusError as e:
        raise CoachError(f'the AI coach request failed ({e.status_code}): {e.message}')
    except anthropic.APIConnectionError:
        raise CoachError('cannot reach the Anthropic API; check the internet connection')
    if r.stop_reason == 'refusal':
        raise CoachError('the AI coach declined these images')
    if r.stop_reason == 'max_tokens':
        raise CoachError('the AI coach answer was cut off')
    text = next((b.text for b in r.content if b.type == 'text'), None)
    if text is None:
        raise CoachError('the AI coach returned no answer')
    return json.loads(text)


def labelled(images):
    """[(label, path)] -> content blocks with a text label before each image."""
    out = []
    for label, path in images:
        out += [{'type': 'text', 'text': label}, image_block(path)]
    return out


def plan(photos, client=None):
    content = labelled([(f'Photo {i + 1}: {Path(p).name}', p) for i, p in enumerate(photos)])
    p = ask(content + [{'type': 'text', 'text': PLAN_TASK}], PLAN_SCHEMA, effort='high', client=client)
    p['hazards'].append({'id': 'marker', 'what': 'Printed scale marker', 'where': p['marker']['where'],
                         'fix': f"Print docs/marker.svg at 100 % and stick it flat: {p['marker']['where']}",
                         'photo_check': True, 'severity': 'must'})
    return p


def verify_item(capture_plan, item_id, photo, client=None):
    item = next((h for h in capture_plan['hazards'] if h['id'] == item_id), None)
    if item is None:
        raise CoachError(f'no item {item_id} in the plan')
    task = (f"The user was asked to fix this before filming:\n{json.dumps(item, indent=1)}\n\nThe photo shows the place "
            'now. passed is true only if the photo clearly shows the fix was done (a ceiling fan looks stopped with '
            'sharp, unblurred blades; a mirror is covered; the marker is flat and fully visible). If the photo does '
            'not show the right place or is unclear, passed is false and reason says what photo to take. reason is '
            'one or two short sentences addressed to the user.')
    return ask([image_block(photo), {'type': 'text', 'text': task}], VERDICT_SCHEMA, client=client)


def sample_frames(clips, out, n=REVIEW_FRAMES):
    """About n frames spread over all clips: [(label, path)]."""
    raw = []
    for i, clip in enumerate(clips):
        d = out / f'c{i}'
        d.mkdir(parents=True, exist_ok=True)
        subprocess.run([shutil.which('ffmpeg') or 'ffmpeg', '-v', 'error', '-i', str(clip), '-vf',
                        'fps=1,scale=640:-2', '-q:v', '4', str(d / '%05d.jpg')], check=True)
        raw += [(f'take {i + 1} at {int(t) // 60}:{int(t) % 60:02d}', f)
                for t, f in enumerate(sorted(d.glob('*.jpg')))]
    step = max(1, len(raw) / n)
    return [raw[int(k * step)] for k in range(min(n, len(raw)))]


def review_coverage(capture_plan, clips, client=None):
    with tempfile.TemporaryDirectory() as tmp:
        frames = sample_frames([Path(c) for c in clips], Path(tmp))
        task = (f"These frames are spread evenly over the finished capture. The plan was:\n"
                f"{json.dumps(capture_plan['route'], indent=1)}\n\nFor each route step, decide whether the frames show "
                'it was filmed (its landmarks visible from the angle the step asks for). Only list a step as missing '
                'when you are fairly sure: frames are sparse, so a step seen in at least one frame counts as done. For '
                'each missing step, retake is a short instruction for an extra clip that starts and ends on views '
                'already filmed. ok is true when nothing important is missing.')
        return ask(labelled(frames) + [{'type': 'text', 'text': task}], COVERAGE_SCHEMA, client=client)


def available():
    """Whether the coach can run: the SDK is installed and some credential is configured."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get('ANTHROPIC_API_KEY') or os.environ.get('ANTHROPIC_AUTH_TOKEN') or
                os.environ.get('ANTHROPIC_PROFILE') or (Path.home() / '.config' / 'anthropic').exists())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan', help='survey photos -> capture plan')
    p.add_argument('photos', nargs='+', type=Path)
    v = sub.add_parser('verify', help='check one fix from a new photo')
    v.add_argument('photo', type=Path)
    v.add_argument('--item', required=True)
    r = sub.add_parser('review', help='did the take cover the plan')
    r.add_argument('clips', nargs='+', type=Path)
    for s in (v, r):
        s.add_argument('--plan', type=Path, required=True)
    for s in (p, v, r):
        s.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    try:
        if a.cmd == 'plan':
            result = plan(a.photos)
        else:
            capture_plan = json.loads(a.plan.read_text(encoding='utf-8'))
            result = (verify_item(capture_plan, a.item, a.photo) if a.cmd == 'verify'
                      else review_coverage(capture_plan, a.clips))
    except CoachError as e:
        a.out.write_text(json.dumps({'error': str(e)}), encoding='utf-8')
        sys.exit(f'coach: {e}')
    a.out.write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
