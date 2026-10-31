// Wayfarer Capture Coach: record with a live guide, then the job server checks the take (pipeline/preflight.py)
// and asks for short retakes until the capture is good enough to build.
import { uploadFile, postJson } from './api.js';
import { LIVE_WIDTH, SECTORS, grayFromRGBA, sharpness, brightness, texture, dHash, cameraDirection, Guide } from './capture-metrics.js';

const $ = (id) => document.getElementById(id);
const KEY = 'wayfarer:coach';
let session = null;

const store = {
  get() { try { return localStorage.getItem(KEY); } catch { return null; } },
  set(v) { try { v ? localStorage.setItem(KEY, v) : localStorage.removeItem(KEY); } catch { /* storage blocked */ } },
};

function show(stage) {
  for (const s of ['start', 'record', 'review', 'built']) $(`stage-${s}`).hidden = s !== stage;
}

function fail(err) {
  if (err.down) { $('server-down').hidden = false; return; }
  alert(err.message);
}

async function ensureSession() {
  if (session) return session;
  const { id } = await postJson('/api/sessions', { title: $('title').value.trim(), ceiling: $('ceiling').value || null });
  store.set(id);
  session = { id, clips: [], reviews: [] };
  return session;
}

async function sendClip(file, bar) {
  await ensureSession();
  if (bar) bar.hidden = false;
  try {
    await uploadFile(`/api/sessions/${session.id}/clips?${new URLSearchParams({ filename: file.name })}`, file,
      (loaded) => { if (bar) bar.firstElementChild.style.width = `${(loaded / file.size) * 100}%`; });
  } finally {
    if (bar) bar.hidden = true;
  }
  show('review');
  poll();
}

// ---------- recording with the live guide ----------
const rec = { stream: null, recorder: null, chunks: [], guide: null, timer: null, started: 0, sensors: false, ori: null, rot: null, acc: [] };
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });

async function openCamera() {
  show('record');
  $('rec-toggle').textContent = 'Start recording';
  $('next-room').disabled = true;
  try {
    rec.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 } },
    });
  } catch (e) {
    show('start');
    return alert(window.isSecureContext ? `Camera unavailable: ${e.message}` :
      'The camera needs a secure page. On the phone, open the https:// address that start.bat prints.');
  }
  $('preview').srcObject = rec.stream;
  await $('preview').play();
}

async function startSensors() {
  if (rec.listening) return;
  rec.listening = true;
  // iOS asks for motion permission, and only from a tap
  for (const E of [window.DeviceMotionEvent, window.DeviceOrientationEvent]) {
    if (E?.requestPermission) { try { await E.requestPermission(); } catch { /* denied */ } }
  }
  addEventListener('deviceorientation', (e) => {
    if (e.beta == null) return;
    rec.ori = cameraDirection(e.alpha ?? 0, e.beta, e.gamma ?? 0);
    rec.sensors = true;
  });
  addEventListener('devicemotion', (e) => {
    const r = e.rotationRate;
    if (r && r.alpha != null) rec.rot = Math.hypot(r.alpha, r.beta, r.gamma);
    const a = e.accelerationIncludingGravity;
    if (a && a.x != null) {
      rec.acc.push([performance.now(), Math.hypot(a.x, a.y, a.z)]);
      while (rec.acc.length && performance.now() - rec.acc[0][0] > 1000) rec.acc.shift();
    }
  });
}

// footsteps shake the phone: the spread of |acceleration| over the last second
function walking() {
  if (rec.acc.length < 10) return false;
  const v = rec.acc.map((x) => x[1]), m = v.reduce((a, b) => a + b) / v.length;
  return Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / v.length) > 0.8;
}

function pickMime() {
  for (const t of ['video/mp4;codecs=avc1', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm']) {
    if (window.MediaRecorder?.isTypeSupported?.(t)) return t;
  }
  return '';
}

async function startRecording() {
  await startSensors();
  const mimeType = pickMime();
  // a high bitrate keeps fine texture: COLMAP and the splats need detail, not a small file
  rec.recorder = new MediaRecorder(rec.stream, { mimeType, videoBitsPerSecond: 16e6 });
  rec.chunks = [];
  rec.recorder.ondataavailable = (e) => e.data.size && rec.chunks.push(e.data);
  rec.recorder.start(1000);
  rec.guide = new Guide();
  rec.guide.checks.marker = $('marker-done').checked = false;
  rec.started = performance.now();
  rec.lastStop = null;
  try { rec.wake = await navigator.wakeLock?.request('screen'); } catch { /* not allowed */ }
  $('rec-dot').classList.add('on');
  $('rec-toggle').textContent = 'Stop and check';
  $('next-room').disabled = false;
  rec.timer = setInterval(tick, 160);
}

function tick() {
  const v = $('preview');
  if (!v.videoWidth) return;
  const w = LIVE_WIDTH, h = Math.round((v.videoHeight / v.videoWidth) * w);
  canvas.width = w; canvas.height = h;
  ctx.drawImage(v, 0, 0, w, h);
  const g = grayFromRGBA(ctx.getImageData(0, 0, w, h).data, w, h);
  const t = (performance.now() - rec.started) / 1000;
  const r = rec.guide.push({
    t, sharp: sharpness(g, w, h), bright: brightness(g), texture: texture(g, w, h), hash: dHash(g, w, h),
    rot: rec.rot, elevation: rec.ori?.elevation, heading: rec.ori?.heading, walking: walking(),
  });
  $('rec-time').textContent = `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
  $('sensor-note').hidden = rec.sensors || t < 3;
  const hint = $('hint');
  hint.hidden = !r.hint;
  if (r.hint) {
    hint.textContent = r.hint.text;
    hint.className = `hint ${r.hint.level}`;
    if (r.hint.level === 'stop' && rec.lastStop !== r.hint.key) navigator.vibrate?.(200);
  }
  rec.lastStop = r.hint?.level === 'stop' ? r.hint.key : null;
  $('todo').textContent = rec.guide.todo();
  for (const li of document.querySelectorAll('#checklist li')) li.classList.toggle('ok', !!r.checks[li.dataset.k]);
  drawRing(r.coverage, rec.ori?.heading);
}

function wedge(i, r0, r1, cls) {
  const a0 = ((i * 360) / SECTORS - 90 + 1) * (Math.PI / 180), a1 = (((i + 1) * 360) / SECTORS - 90 - 1) * (Math.PI / 180);
  const p = (r, a) => `${(r * Math.cos(a)).toFixed(2)} ${(r * Math.sin(a)).toFixed(2)}`;
  return `<path class="${cls}" d="M${p(r0, a0)} L${p(r1, a0)} A${r1} ${r1} 0 0 1 ${p(r1, a1)} L${p(r0, a1)} A${r0} ${r0} 0 0 0 ${p(r0, a0)}Z"/>`;
}

// inner ring: filmed from the middle / edge; outer ring: filmed while walking along
function drawRing(coverage, heading) {
  let svg = '';
  coverage.forEach((c, i) => {
    svg += wedge(i, 16, 31, c.seen >= 1.5 ? 'seen' : 'empty');
    svg += wedge(i, 34, 46, c.walked >= 1.5 ? 'walked' : 'empty');
  });
  if (heading != null) {
    const a = (heading - 90) * (Math.PI / 180);
    svg += `<circle class="now" r="5" cx="${(8 * Math.cos(a)).toFixed(2)}" cy="${(8 * Math.sin(a)).toFixed(2)}"/>`;
  }
  $('ring').innerHTML = svg;
}

function stopCamera() {
  clearInterval(rec.timer);
  rec.wake?.release?.();
  rec.stream?.getTracks().forEach((t) => t.stop());
  rec.stream = null;
  $('rec-dot').classList.remove('on');
  $('hint').hidden = true;
}

async function stopRecording() {
  const done = new Promise((res) => { rec.recorder.onstop = res; });
  rec.recorder.stop();
  await done;
  stopCamera();
  const type = rec.recorder.mimeType || 'video/webm';
  const n = (session?.clips?.length ?? 0) + 1;
  const file = new File(rec.chunks, `take-${n}.${type.includes('mp4') ? 'mp4' : 'webm'}`, { type });
  $('rec-toggle').disabled = true;
  try { await sendClip(file, $('rec-upload')); } catch (e) { fail(e); show('record'); } finally { $('rec-toggle').disabled = false; }
}

$('go-record').onclick = openCamera;
$('retake-record').onclick = openCamera;
$('rec-toggle').onclick = () => (rec.recorder?.state === 'recording' ? stopRecording() : startRecording());
$('next-room').onclick = () => { rec.guide?.newRoom(); $('marker-done').checked = false; };
$('marker-done').onchange = (e) => { if (rec.guide) rec.guide.checks.marker = e.target.checked; };
$('rec-cancel').onclick = () => {
  if (rec.recorder?.state === 'recording' && !confirm('Discard this recording?')) return;
  if (rec.recorder?.state === 'recording') { rec.recorder.onstop = null; rec.recorder.stop(); }
  stopCamera();
  show(session?.reviews?.length ? 'review' : 'start');
};
for (const id of ['pick-video', 'pick-retake']) {
  $(id).addEventListener('change', (e) => {
    const f = e.target.files[0];
    e.target.value = '';
    if (f) sendClip(f, null).catch(fail);
  });
}

// ---------- review loop ----------
let pollTimer = null;
async function poll() {
  clearTimeout(pollTimer);
  if (!session) return;
  try {
    const r = await fetch(`/api/sessions/${session.id}`);
    if (r.status === 404) { store.set(null); session = null; return show('start'); }
    session = await r.json();
    $('server-down').hidden = true;
  } catch { $('server-down').hidden = false; pollTimer = setTimeout(poll, 4000); return; }
  render();
  const last = session.reviews.at(-1);
  if (last && ['queued', 'running'].includes(last.status)) pollTimer = setTimeout(poll, 2000);
}

const CHECK_NAMES = { metrics: 'Video', sfm: 'Camera poses', angles: 'View angles', preflight: 'Check' };

function render() {
  if (session.stage === 'built') return show('built');
  const last = session.reviews.at(-1);
  if (!last) return show('start');
  show('review');
  const running = ['queued', 'running'].includes(last.status);
  const rep = last.report;
  $('review-status').className = `pill ${running ? 'running' : rep?.pass ? 'done' : 'failed'}`;
  $('review-status').textContent = running ? (last.status === 'queued' ? 'waiting for the GPU' : 'checking…')
    : last.status === 'interrupted' ? 'interrupted' : rep?.pass ? 'ready to build' : 'needs a retake';
  $('review-log').textContent = running ? session.reviewLog.join('\n') : '';

  $('checks').innerHTML = '';
  for (const c of rep?.checks ?? []) {
    const li = document.createElement('li');
    li.innerHTML = '<b></b><span></span>';
    li.className = c.status;
    li.firstChild.textContent = CHECK_NAMES[c.name] ?? c.name;
    li.lastChild.textContent = `${c.status} · ${c.summary}`;
    $('checks').appendChild(li);
  }
  $('retakes').innerHTML = '';
  for (const t of rep?.retakes ?? []) {
    const d = document.createElement('div');
    d.className = 'retake';
    d.textContent = t.instruction;
    $('retakes').appendChild(d);
  }
  $('issues').innerHTML = '';
  for (const i of rep?.issues ?? []) {
    const li = document.createElement('li');
    li.className = i.severity;
    if (i.thumbUrl) { const img = new Image(); img.src = i.thumbUrl; img.alt = ''; li.appendChild(img); }
    li.appendChild(document.createTextNode(i.message));
    $('issues').appendChild(li);
  }
  const failed = !running && rep && !rep.pass;
  $('retake-record').hidden = $('retake-upload').hidden = !(failed || last.status === 'interrupted');
  $('build-box').hidden = !(rep?.pass && !running);
  $('build-anyway').hidden = !failed;
}

async function build(force) {
  try {
    const res = await postJson(`/api/sessions/${session.id}/build`, { quality: $('quality').value, force });
    try { localStorage.setItem('wayfarer:job', res.id); } catch { /* storage blocked */ }
    session.stage = 'built';
    show('built');
  } catch (e) { fail(e); }
}
$('build').onclick = () => build(false);
$('build-anyway').onclick = (e) => {
  e.preventDefault();
  if (confirm('The capture did not pass. The twin will likely have holes, blur or fibrous walls. Build anyway?')) build(true);
};
for (const id of ['start-over', 'start-over-2']) {
  $(id).onclick = (e) => { e.preventDefault(); store.set(null); session = null; show('start'); };
}

// resume a capture in progress
const sid = new URLSearchParams(location.search).get('session') || store.get(); // ?session= continues on another device
if (sid) { session = { id: sid, clips: [], reviews: [] }; poll(); } else show('start');
