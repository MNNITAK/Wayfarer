"""Local job server: upload a video from the browser, run the pipeline, poll progress.

  .venv-ns/Scripts/python.exe pipeline/server.py      (start.bat does this for you)

  POST /api/uploads?filename=room1.mp4                             body = raw video bytes -> {upload}
  POST /api/jobs   {"uploads": [ids...], "title", "ceiling", "quality"}  (several clips -> one connected space)
  GET  /api/jobs/<id>                                              -> status, step, log tail
  GET  /api/scenes                                                 -> reconstructed scenes

Capture Coach sessions (a capture is checked and re-filmed until it meets the pipeline's needs, then built):
  POST /api/sessions   {"title", "ceiling"}                        -> {id}
  GET  /api/sessions/<id>                                          -> stage, clips, reviews (+ log tail)
  GET  /api/sessions/<id>/files/<path>                             -> review thumbnails, photos
  POST /api/sessions/<id>/clips?filename=walk.mp4  body = video   -> runs pipeline/preflight.py on all clips
  POST /api/sessions/<id>/build {"quality", "force"}               -> starts the normal job once a review passed
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / 'work' / 'uploads'
SCENES = ROOT / 'web' / 'public' / 'scenes'
SESSIONS = ROOT / 'work' / 'sessions'
PORT = int(os.environ.get('WAYFARER_PORT', 8765))
STEP_RE = re.compile(r'^\[(\d)/5\] (\w+)')
VIDEO_EXT = {'.mp4', '.mov', '.m4v', '.mkv', '.avi', '.webm'}
jobs, uploads = {}, {}  # ponytail: in-memory, one machine, one user; persist to disk if jobs must survive restarts
lock = threading.Lock()  # one GPU -> one training job (or quick-SfM review) at a time
sessions = {}
FILE_TYPES = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png'}


def slug(s):
    return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')[:40] or 'scene'


def pipeline_env():
    env = os.environ.copy()
    scripts = Path(sys.executable).parent  # venv Scripts/bin: ns-process-data, ns-train, ns-export
    extra = [str(scripts)]
    if os.environ.get('WAYFARER_COLMAP'):
        extra.append(os.environ['WAYFARER_COLMAP'])
    if sys.platform == 'win32':  # PATH edits made after this process started live only in the registry
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as k:
                extra.append(os.path.expandvars(winreg.QueryValueEx(k, 'Path')[0]))
        except OSError:
            pass
    env['PATH'] = os.pathsep.join(extra + [env.get('PATH', '')])
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    return env


def run_job(job):
    job['status'] = 'queued'
    with lock:
        job['status'] = 'running'
        cmd = [sys.executable, str(ROOT / 'pipeline' / 'wayfarer.py'), *job['video'], '--name', job['scene'], '--title', job['title']]
        if job['ceiling']:
            cmd += ['--ceiling', str(job['ceiling'])]
        if job['quality']:
            cmd += ['--quality', job['quality']]
        if job.get('aiRepair'):
            cmd += ['--ai-repair']
        if job.get('maskPeople'):
            cmd += ['--mask-people']
        log = open(job['log'], 'w', encoding='utf-8', errors='replace')
        p = subprocess.Popen(cmd, cwd=ROOT, env=pipeline_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding='utf-8', errors='replace', bufsize=1)
        for line in p.stdout:
            log.write(line)
            log.flush()
            if m := STEP_RE.match(line):
                job['step'], job['stepName'] = int(m[1]), m[2]
        p.wait()
        log.close()
        job['status'] = 'done' if p.returncode == 0 else 'failed'
        job['finished'] = time.time()


def session_dir(s):
    return SESSIONS / s['id']


def save_session(s):
    session_dir(s).mkdir(parents=True, exist_ok=True)
    (session_dir(s) / 'session.json').write_text(json.dumps(s, indent=1), encoding='utf-8')


def load_sessions():
    for f in SESSIONS.glob('*/session.json') if SESSIONS.exists() else []:
        s = json.loads(f.read_text(encoding='utf-8'))
        for rev in s['reviews']:
            if rev['status'] in ('queued', 'running'):  # the server stopped mid-review
                rev['status'] = 'interrupted'
        sessions[s['id']] = s


def run_review(s, rev):
    """pipeline/preflight.py over every clip of the session; its report decides the next stage."""
    rev['status'] = 'queued'
    save_session(s)
    with lock:  # quick SfM shares the GPU with builds
        rev['status'] = 'running'
        save_session(s)
        cmd = [sys.executable, str(ROOT / 'pipeline' / 'preflight.py'), *rev['clips'], '--work', rev['work']]
        if s.get('retakeFrom') is not None:
            cmd += ['--retake-from', str(s['retakeFrom'])]
        cmd += ['--names', *[f"take {k + 1}" for k in range(len(rev['clips']))]]
        with open(rev['log'], 'w', encoding='utf-8', errors='replace') as log:
            p = subprocess.run(cmd, cwd=ROOT, env=pipeline_env(), stdout=log, stderr=subprocess.STDOUT)
    report_file = Path(rev['work']) / 'report.json'
    if report_file.exists():
        report = json.loads(report_file.read_text(encoding='utf-8'))
    else:
        report = {'pass': False, 'checks': [{'name': 'preflight', 'status': 'fail',
                                             'summary': f'the check crashed (exit {p.returncode}); see the log'}],
                  'issues': [], 'retakes': []}
    rel = Path(rev['work']).relative_to(session_dir(s)).as_posix()
    for i in report['issues']:
        if i.get('thumb'):
            i['thumbUrl'] = f"/api/sessions/{s['id']}/files/{rel}/{i['thumb']}"
    rev['report'], rev['status'] = report, 'done'
    s['stage'] = 'ready' if report['pass'] else 'retake'
    save_session(s)


def tail(path, n=40):
    try:
        text = Path(path).read_text(encoding='utf-8', errors='replace')
    except FileNotFoundError:
        return []
    lines = [l.split('\r')[-1] for l in text.splitlines()]  # progress bars redraw with \r: keep the last frame
    return [l for l in lines if l.strip()][-n:]


class Handler(BaseHTTPRequestHandler):
    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # keep the console for pipeline output
        pass

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == '/api/scenes':
            scenes = []
            for d in sorted(SCENES.iterdir()) if SCENES.exists() else []:
                f = d / 'scene.json'
                if f.exists():
                    s = json.loads(f.read_text(encoding='utf-8'))
                    scenes.append({'id': d.name, 'name': s.get('name', d.name), 'splat': bool(s.get('splat')), 'pois': len(s.get('pois', [])), 'psnr': s.get('stats', {}).get('psnr'),
                                   'updated': f.stat().st_mtime})
            return self.send_json(scenes)
        if url.path.startswith('/api/jobs/'):
            job = jobs.get(url.path.rsplit('/', 1)[-1])
            if not job:
                return self.send_json({'error': 'no such job'}, 404)
            return self.send_json({k: v for k, v in job.items() if k not in ('video', 'log')} | {'log': tail(job['log'])})
        if url.path == '/api/jobs':
            return self.send_json([{k: v for k, v in j.items() if k not in ('video', 'log')} for j in jobs.values()])
        if url.path.startswith('/api/sessions/'):
            return self.get_session(url.path.split('/')[3:])
        self.send_json({'error': 'not found'}, 404)

    def get_session(self, parts):
        s = sessions.get(parts[0])
        if not s:
            return self.send_json({'error': 'no such session'}, 404)
        if len(parts) == 1:
            last = s['reviews'][-1] if s['reviews'] else None
            return self.send_json(s | {'reviewLog': tail(last['log'], 12) if last else []})
        if parts[1] == 'files' and len(parts) > 2:
            f = (session_dir(s) / '/'.join(parts[2:])).resolve()
            if f.suffix.lower() not in FILE_TYPES or not f.is_relative_to(session_dir(s).resolve()) or not f.is_file():
                return self.send_json({'error': 'not found'}, 404)
            body = f.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', FILE_TYPES[f.suffix.lower()])
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            return self.wfile.write(body)
        self.send_json({'error': 'not found'}, 404)

    def reject(self, msg, code=400):
        # drain the upload first: replying mid-body makes proxies drop the response
        left = int(self.headers.get('Content-Length', 0))
        while left > 0 and (chunk := self.rfile.read(min(left, 1 << 20))):
            left -= len(chunk)
        self.send_json({'error': msg}, code)

    def do_POST(self):
        url = urlparse(self.path)
        if url.path == '/api/uploads':
            return self.upload(url)
        if url.path == '/api/jobs':
            return self.create_job()
        if url.path == '/api/sessions':
            return self.create_session()
        if url.path.startswith('/api/sessions/'):
            parts = url.path.split('/')[3:]
            s = sessions.get(parts[0])
            if not s:
                return self.reject('no such session', 404)
            if parts[1:] == ['clips']:
                return self.add_clip(s, url)
            if parts[1:] == ['build']:
                return self.build_session(s)
        self.reject('not found', 404)

    def read_json(self):
        try:
            return json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
        except ValueError:
            return None

    def create_session(self):
        q = self.read_json()
        if q is None:
            return self.send_json({'error': 'body must be JSON'}, 400)
        sid = uuid.uuid4().hex[:8]
        s = {'id': sid, 'title': (q.get('title') or '').strip()[:40], 'ceiling': q.get('ceiling') or None,
             'created': time.time(), 'stage': 'record', 'clips': [], 'reviews': [], 'retakeFrom': None, 'job': None}
        sessions[sid] = s
        save_session(s)
        self.send_json({'id': sid}, 201)

    def add_clip(self, s, url):
        """A recorded take or a retake; every clip so far is reviewed together (they merge into one space)."""
        filename = Path(parse_qs(url.query).get('filename', ['capture.mp4'])[0]).name
        ext = Path(filename).suffix.lower()
        if ext not in VIDEO_EXT:
            return self.reject(f'unsupported file type {ext or "(none)"} in {filename}')
        if int(self.headers.get('Content-Length', 0)) <= 0:
            return self.reject('empty upload')
        n = len(s['clips'])
        path = session_dir(s) / 'clips' / f'{n:02d}_{slug(Path(filename).stem)}{ext}'
        if not self.save_body(path):
            return self.send_json({'error': 'upload interrupted'}, 400)
        last = s['reviews'][-1] if s['reviews'] else None
        if last and last.get('report') and not last['report']['pass'] and s.get('retakeFrom') is None:
            s['retakeFrom'] = n  # clips from here on are retakes for the problems found so far
        s['clips'].append({'name': filename, 'path': str(path), 'sizeMB': round(path.stat().st_size / 1e6, 1)})
        work = session_dir(s) / 'reviews' / str(len(s['reviews']))
        rev = {'n': len(s['reviews']), 'clips': [c['path'] for c in s['clips']], 'work': str(work),
               'log': str(work.parent / f'{work.name}.log'), 'status': 'queued', 'report': None}
        work.parent.mkdir(parents=True, exist_ok=True)
        s['reviews'].append(rev)
        s['stage'] = 'review'
        threading.Thread(target=run_review, args=(s, rev), daemon=True).start()
        self.send_json({'clip': n, 'review': rev['n']}, 201)

    def build_session(self, s):
        q = self.read_json() or {}
        last = s['reviews'][-1] if s['reviews'] else None
        passed = bool(last and last.get('report') and last['report']['pass'])
        if not s['clips'] or not (passed or q.get('force')):
            return self.send_json({'error': 'the capture has not passed its review yet'}, 400)
        quality = q.get('quality') if q.get('quality') in ('fast', 'default', 'high', 'max') else None
        try:
            ceiling = float(s['ceiling']) if s.get('ceiling') else None
        except (TypeError, ValueError):
            ceiling = None
        job = start_job(s['clips'], s['title'], ceiling, quality, q)
        s['job'], s['stage'], s['forced'] = job['id'], 'built', not passed
        save_session(s)
        self.send_json({'id': job['id'], 'scene': job['scene']}, 201)

    def upload(self, url):
        """One video per request (raw body); a job then references one or more uploads."""
        filename = parse_qs(url.query).get('filename', ['capture.mp4'])[0]
        ext = Path(filename).suffix.lower()
        if ext not in VIDEO_EXT:
            return self.reject(f'unsupported file type {ext or "(none)"} in {filename}')
        size = int(self.headers.get('Content-Length', 0))
        if size <= 0:
            return self.reject('empty upload')
        uid = uuid.uuid4().hex[:10]
        video = UPLOADS / f'{uid}{ext}'
        if not self.save_body(video):
            return self.send_json({'error': 'upload interrupted'}, 400)
        uploads[uid] = {'path': str(video), 'name': filename, 'sizeMB': round(size / 1e6, 1)}
        self.send_json({'upload': uid}, 201)

    def save_body(self, path):
        """Stream the raw request body to `path` (phone videos are often > 1 GB). False if the client hung up."""
        path.parent.mkdir(parents=True, exist_ok=True)
        left = int(self.headers.get('Content-Length', 0))
        with open(path, 'wb') as f:
            while left > 0:
                chunk = self.rfile.read(min(left, 1 << 20))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        if left > 0:
            path.unlink(missing_ok=True)
            return False
        return True

    def create_job(self):
        try:
            q = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
        except ValueError:
            return self.send_json({'error': 'body must be JSON'}, 400)
        ids = q.get('uploads') or []
        if not ids or any(i not in uploads for i in ids):
            return self.send_json({'error': 'upload at least one video first'}, 400)
        try:
            ceiling = float(q['ceiling']) if q.get('ceiling') not in (None, '') else None
        except (TypeError, ValueError):
            return self.send_json({'error': 'ceiling must be a number in metres'}, 400)
        if ceiling is not None and not 1.8 <= ceiling <= 12:
            return self.send_json({'error': 'ceiling should be between 1.8 and 12 m'}, 400)
        quality = q.get('quality') if q.get('quality') in ('fast', 'default', 'high', 'max') else None

        job = start_job([uploads[i] for i in ids], q.get('title'), ceiling, quality, q)
        self.send_json({'id': job['id'], 'scene': job['scene']}, 201)


def start_job(files, title, ceiling, quality, q):
    """files: [{path, name, sizeMB}]; q: the request options (aiRepair, maskPeople)."""
    jid = uuid.uuid4().hex[:8]
    title = (title or '').strip()[:40] or Path(files[0]['name']).stem
    scene = f'{slug(title)}-{jid[:4]}'
    UPLOADS.mkdir(parents=True, exist_ok=True)
    job = {'id': jid, 'scene': scene, 'title': title, 'ceiling': ceiling, 'quality': quality, 'aiRepair': bool(q.get('aiRepair')), 'maskPeople': bool(q.get('maskPeople')),
           'video': [f['path'] for f in files], 'clips': [f['name'] for f in files],
           'log': str(UPLOADS / f'{scene}.log'), 'status': 'queued', 'step': 0, 'stepName': 'upload',
           'created': time.time(), 'sizeMB': round(sum(f['sizeMB'] for f in files), 1)}
    jobs[jid] = job
    threading.Thread(target=run_job, args=(job,), daemon=True).start()
    return job


if __name__ == '__main__':
    if not shutil.which('colmap', path=pipeline_env()['PATH']):
        print('WARNING: colmap not found on PATH (set WAYFARER_COLMAP to its folder). Uploads will fail at the sfm step.')
    load_sessions()
    print(f'Wayfarer job server on http://127.0.0.1:{PORT}')
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
