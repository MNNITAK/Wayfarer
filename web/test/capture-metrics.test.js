import test from 'node:test';
import assert from 'node:assert/strict';
import { sharpness, brightness, texture, dHash, hamming, cameraDirection, Guide, SECTORS, DARK } from '../src/capture-metrics.js';

const W = 120, H = 80;
// deterministic textured frame, and a box-blurred copy
function textured(seed = 1, shift = 0) {
  const g = new Float32Array(W * H);
  let s = seed;
  const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  const base = Array.from({ length: (W + 40) * H }, rnd);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) g[y * W + x] = 40 + 180 * base[y * (W + 40) + x + shift];
  return g;
}
function blur(g, k = 3) {
  const out = new Float32Array(g.length);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let s = 0, n = 0;
    for (let dy = -k; dy <= k; dy++) for (let dx = -k; dx <= k; dx++) {
      const yy = y + dy, xx = x + dx;
      if (yy >= 0 && yy < H && xx >= 0 && xx < W) { s += g[yy * W + xx]; n++; }
    }
    out[y * W + x] = s / n;
  }
  return out;
}

test('sharpness drops for a blurred frame', () => {
  const g = textured();
  assert.ok(sharpness(blur(g), W, H) < 0.2 * sharpness(g, W, H));
});

test('texture separates a blank wall from a textured view', () => {
  const flat = new Float32Array(W * H).fill(150);
  assert.ok(texture(flat, W, H) < 0.01);
  assert.ok(texture(textured(), W, H) > 20);
  assert.equal(Math.round(brightness(flat)), 150);
});

test('dHash: same view close, different view far', () => {
  const a = blur(textured(1), 2), b = blur(textured(1), 2), c = blur(textured(7), 2);
  assert.equal(hamming(dHash(a, W, H), dHash(b, W, H)), 0);
  assert.ok(hamming(dHash(a, W, H), dHash(c, W, H)) > 15);
});

test('cameraDirection: flat on a table looks down, upright looks level', () => {
  assert.equal(Math.round(cameraDirection(0, 0, 0).elevation), -90);
  assert.equal(Math.round(cameraDirection(0, 90, 0).elevation) + 0, 0); // +0: no -0
  assert.equal(Math.round(cameraDirection(0, 80, 0).elevation), -10); // tilted 10 deg down
  // landscape (gamma = -90 with the phone on its side, beta = 0): level too
  assert.equal(Math.round(Math.abs(cameraDirection(0, 0, -90).elevation)), 0);
  const a = cameraDirection(0, 90, 0).heading, b = cameraDirection(90, 90, 0).heading;
  assert.equal(Math.round((b - a + 360) % 360), 270); // alpha grows counter-clockwise
});

const good = { sharp: 100, bright: 120, texture: 10, rot: 5, elevation: -12, walking: false };

test('guide: fast turn and darkness raise hints, a calm take does not', () => {
  const g = new Guide();
  let r;
  for (let i = 0; i < 12; i++) r = g.push({ ...good, t: i / 6, heading: 0 });
  assert.equal(r.hint, null);
  for (let i = 12; i < 20; i++) r = g.push({ ...good, t: i / 6, heading: 0, rot: 80 });
  assert.equal(r.hint.key, 'turn');
  const d = new Guide();
  for (let i = 0; i < 12; i++) r = d.push({ ...good, t: i / 6, heading: 0, bright: DARK - 10 });
  assert.equal(r.hint.key, 'dark');
});

test('guide: blur is relative to the last seconds', () => {
  const g = new Guide();
  let r;
  for (let i = 0; i < 12; i++) r = g.push({ ...good, t: i / 6, heading: 0 });
  for (let i = 12; i < 20; i++) r = g.push({ ...good, t: i / 6, heading: 0, sharp: 30 });
  assert.equal(r.hint.key, 'blur');
});

test('guide: coverage ring, along-wall pass and ceiling sweep tick the room checklist', () => {
  const g = new Guide();
  let r, t = 0;
  for (let s = 0; s < SECTORS; s++) for (let i = 0; i < 12; i++) r = g.push({ ...good, t: (t += 1 / 6), heading: s * 30 + 10 });
  assert.ok(r.checks.perimeter && !r.checks.alongWalls);
  for (let s = 0; s < SECTORS; s++) for (let i = 0; i < 12; i++) r = g.push({ ...good, t: (t += 1 / 6), heading: s * 30 + 10, walking: true });
  assert.ok(r.checks.alongWalls && !r.checks.ceiling);
  for (let i = 0; i < 20; i++) r = g.push({ ...good, t: (t += 1 / 6), heading: 0, elevation: 70 });
  assert.ok(r.checks.ceiling);
  g.newRoom();
  assert.equal(g.checks.perimeter, false);
});

test('guide: returning to the first view of the room ticks back-at-start', () => {
  const g = new Guide();
  const start = dHash(blur(textured(1), 2), W, H), other = dHash(blur(textured(9), 2), W, H);
  let r, t = 0;
  for (let i = 0; i < 12; i++) r = g.push({ ...good, t: (t += 1 / 6), heading: 0, hash: start });
  for (let i = 0; i < 300; i++) r = g.push({ ...good, t: (t += 1 / 6), heading: 90, hash: other });
  assert.equal(r.checks.backAtStart, false);
  for (let i = 0; i < 4; i++) r = g.push({ ...good, t: (t += 1 / 6), heading: 0, hash: start });
  assert.equal(r.checks.backAtStart, true);
});
