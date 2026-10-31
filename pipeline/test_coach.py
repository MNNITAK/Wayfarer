"""Self-check for the Capture Coach agents, with the Claude API and Ollama stubbed out. Run: python pipeline/test_coach.py"""
import base64
import io
import json
import shutil
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

import coach


class StubClient:
    """Records each request and answers with the next canned JSON (or a refusal)."""
    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        a = self.answers.pop(0)
        if a == 'refusal':
            return SimpleNamespace(stop_reason='refusal', content=[])
        return SimpleNamespace(stop_reason='end_turn', content=[SimpleNamespace(type='text', text=json.dumps(a))])


PLAN = {
    'summary': 'Two-room flat', 'rooms': ['living room', 'bedroom'],
    'hazards': [{'id': 'fan-living', 'what': 'Ceiling fan', 'where': 'living room centre', 'fix': 'Switch it off',
                 'photo_check': True, 'severity': 'must'}],
    'marker': {'where': 'left of the sofa, on the wall', 'why': 'passed within 1 m on the perimeter loop'},
    'route': [{'id': 'r1', 'room': 'living room', 'title': 'Perimeter loop', 'detail': 'start at the door',
               'landmarks': ['sofa', 'bookshelf']}],
    'settings': ['1x lens'], 'notes': [],
}

coach.PROVIDER = 'claude'  # the stubbed-client tests below; Ollama is checked at the end
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    photos = []
    for i in range(3):
        p = d / f'corner{i}.jpg'
        Image.fromarray(np.random.default_rng(i).integers(0, 255, (3000, 4000, 3), dtype=np.uint8)).save(p)
        photos.append(p)

    stub = StubClient(PLAN)
    plan = coach.plan(photos, client=stub)
    req = stub.calls[0]
    images = [b for b in req['messages'][0]['content'] if b['type'] == 'image']
    assert len(images) == 3 and req['model'] == coach.MODEL
    assert max(Image.open(io.BytesIO(base64.b64decode(images[0]['source']['data']))).size) == 1024
    assert 'capture' in req['system'].lower() and req['fallbacks'] == 'default'
    assert req['output_config']['format']['schema'] == coach.PLAN_SCHEMA
    assert [h['id'] for h in plan['hazards']] == ['fan-living', 'marker']  # the marker becomes a checked item
    print('ok: plan sends downscaled photos with the requirements brief and adds the marker check')

    stub = StubClient({'passed': False, 'reason': 'The blades are blurred: the fan is still turning.'})
    v = coach.verify_item(plan, 'fan-living', photos[0], client=stub)
    assert v['passed'] is False and 'Ceiling fan' in stub.calls[0]['messages'][0]['content'][-1]['text']
    try:
        coach.verify_item(plan, 'nope', photos[0], client=StubClient())
        raise AssertionError('unknown item accepted')
    except coach.CoachError:
        pass
    print('ok: verify sends the item and the new photo, rejects unknown items')

    try:
        coach.verify_item(plan, 'fan-living', photos[0], client=StubClient('refusal'))
        raise AssertionError('refusal not raised')
    except coach.CoachError as e:
        assert 'declined' in str(e)
    print('ok: a refusal becomes a readable error, not a crash')

    if shutil.which('ffmpeg'):
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=640x360:rate=10', '-t', '40',
                        '-pix_fmt', 'yuv420p', str(d / 'take.mp4')], check=True)
        stub = StubClient({'ok': True, 'missing': []})
        r = coach.review_coverage(plan, [d / 'take.mp4'], client=stub)
        sent = [b for b in stub.calls[0]['messages'][0]['content'] if b['type'] == 'image']
        labels = [b['text'] for b in stub.calls[0]['messages'][0]['content'] if b['type'] == 'text']
        assert r['ok'] and len(sent) == coach.review_frames() and labels[0] == 'take 1 at 0:00', labels[:2]
        print(f'ok: review sends {len(sent)} labelled frames spread over the take')
    else:
        print('skip: ffmpeg not on PATH, review frame sampling not run')


# Ollama: a fake server answers /api/chat and /api/tags like the real one
seen = []


class FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply({'models': [{'name': 'qwen2.5vl:7b'}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        seen.append(req)
        if req['model'] != 'qwen2.5vl:7b':
            return self.reply({'error': f"model '{req['model']}' not found"}, 404)
        self.reply({'message': {'role': 'assistant', 'content': json.dumps(PLAN)}})


srv = HTTPServer(('127.0.0.1', 0), FakeOllama)
threading.Thread(target=srv.serve_forever, daemon=True).start()
coach.PROVIDER, coach.OLLAMA_URL = 'ollama', f'http://127.0.0.1:{srv.server_port}'
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    photos = []
    for i in range(15):
        p = d / f'p{i:02d}.jpg'
        Image.fromarray(np.random.default_rng(i).integers(0, 255, (1500, 2000, 3), dtype=np.uint8)).save(p)
        photos.append(p)
    assert coach.available()
    plan = coach.plan(photos)
    req = seen[-1]
    user = req['messages'][1]
    assert req['format'] == coach.PLAN_SCHEMA and req['options']['num_ctx'] == coach.OLLAMA_CTX
    assert req['messages'][0]['role'] == 'system' and 'capture' in req['messages'][0]['content'].lower()
    assert len(user['images']) == coach.MAX_PLAN_PHOTOS and '[image 12]' in user['content'] and 'p14.jpg' in user['content']
    assert max(Image.open(io.BytesIO(base64.b64decode(user['images'][0]))).size) == 640
    assert plan['hazards'][-1]['id'] == 'marker'
    print('ok: Ollama gets the schema as format, 12 evenly picked 640 px photos and a big enough context')

    coach.OLLAMA_MODEL = 'llava:missing'
    try:
        coach.plan(photos[:2])
        raise AssertionError('missing model not reported')
    except coach.CoachError as e:
        assert 'ollama pull llava:missing' in str(e)
    assert not coach.available()
    coach.OLLAMA_URL = 'http://127.0.0.1:9'  # nothing listens here
    try:
        coach.plan(photos[:2])
        raise AssertionError('Ollama down not reported')
    except coach.CoachError as e:
        assert 'not running' in str(e)
    print('ok: a missing model or a stopped Ollama gives a clear next step')
srv.shutdown()
