// Capture Coach live checks: pure functions over small greyscale frames and phone sensor readings.
// Thresholds mirror pipeline/capture_requirements.md (and pipeline/preflight.py); change them together.

export const LIVE_WIDTH = 240;   // analysis frame width; texture threshold is calibrated at this size
export const SHARP_RATIO = 0.55; // blurred if below this fraction of the sharpest frame in the last 2 s
export const MAX_TURN = 30;      // deg/s
export const DARK = 35;          // mean grey level
export const BLANK = 2.5;        // mean |gradient| at 240 px wide
export const SECTORS = 12;       // coverage ring: 30 deg each
export const SECTOR_SECONDS = 1.5; // good footage needed per sector

export function grayFromRGBA(rgba, w, h) {
  const g = new Float32Array(w * h);
  for (let i = 0, j = 0; i < g.length; i++, j += 4) g[i] = 0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2];
  return g;
}

// variance of the 4-neighbour Laplacian, as pipeline/wayfarer.py sharpness()
export function sharpness(g, w, h) {
  let n = 0, sum = 0, sum2 = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const l = g[i - 1] + g[i + 1] + g[i - w] + g[i + w] - 4 * g[i];
      sum += l; sum2 += l * l; n++;
    }
  }
  const m = sum / n;
  return sum2 / n - m * m;
}

export function brightness(g) {
  let s = 0;
  for (let i = 0; i < g.length; i++) s += g[i];
  return s / g.length;
}

// mean absolute gradient: low on blank walls
export function texture(g, w, h) {
  let sx = 0, sy = 0;
  for (let y = 0; y < h - 1; y++) {
    for (let x = 0; x < w - 1; x++) {
      const i = y * w + x;
      sx += Math.abs(g[i + 1] - g[i]); sy += Math.abs(g[i + w] - g[i]);
    }
  }
  return (sx + sy) / (2 * (w - 1) * (h - 1));
}

// 64-bit difference hash (9x8 box-averaged grid), as an array of 64 0/1 values
export function dHash(g, w, h) {
  const cell = (cx, cy) => {
    const x0 = Math.floor((cx * w) / 9), x1 = Math.floor(((cx + 1) * w) / 9);
    const y0 = Math.floor((cy * h) / 8), y1 = Math.floor(((cy + 1) * h) / 8);
    let s = 0;
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) s += g[y * w + x];
    return s / Math.max(1, (x1 - x0) * (y1 - y0));
  };
  const bits = [];
  for (let y = 0; y < 8; y++) {
    const row = Array.from({ length: 9 }, (_, x) => cell(x, y));
    for (let x = 0; x < 8; x++) bits.push(row[x] < row[x + 1] ? 1 : 0);
  }
  return bits;
}

export const hamming = (a, b) => a.reduce((d, v, i) => d + (v !== b[i]), 0);

// Rear-camera direction from DeviceOrientation (alpha, beta, gamma in deg; W3C Z-X'-Y'' order), any screen
// orientation. elevation: deg above the horizon; heading: deg around the vertical (origin is device-dependent).
export function cameraDirection(alpha, beta, gamma) {
  const r = Math.PI / 180;
  const [sA, cA, sB, cB, sG, cG] = [Math.sin(alpha * r), Math.cos(alpha * r), Math.sin(beta * r), Math.cos(beta * r), Math.sin(gamma * r), Math.cos(gamma * r)];
  const x = -(cA * sG + sA * sB * cG), y = -(sA * sG - cA * sB * cG), z = -(cB * cG); // R * (0, 0, -1)
  return { elevation: Math.asin(Math.max(-1, Math.min(1, z))) / r, heading: ((Math.atan2(x, y) / r) + 360) % 360 };
}

const HINTS = {
  dark: 'Too dark: switch on the lights here',
  turn: 'Turn slower',
  blur: 'Blurry: move slower and hold steady',
  blank: 'Blank wall: step back or keep furniture or an edge in view',
  tiltDown: 'Tilt down a little so the floor is in view',
  tiltUp: 'Raise the camera: too much floor',
};

// Live guide for one take. push() a sample about 6x per second:
//   { t (s), sharp, bright, texture, hash, rot (deg/s), elevation, heading, walking }
// and get back { hint, coverage, checks } to draw.
export class Guide {
  constructor() { this.hist = []; this.startHashes = []; this.t0 = null; this.newRoom(); }

  newRoom() {
    this.coverage = Array.from({ length: SECTORS }, () => ({ seen: 0, walked: 0 }));
    this.ceiling = 0;
    this.roomStart = null;
    this.startHashes = [];
    this.loopHits = 0;
    this.checks = { marker: false, perimeter: false, alongWalls: false, ceiling: false, backAtStart: false };
  }

  // seconds within the last `span` s for which pred(sample) held (samples weighted by their spacing)
  held(span, pred) {
    const now = this.hist.at(-1).t;
    let s = 0;
    for (let i = this.hist.length - 1; i > 0 && now - this.hist[i].t < span; i--) {
      if (pred(this.hist[i])) s += this.hist[i].t - this.hist[i - 1].t;
    }
    return s;
  }

  push(s) {
    if (this.t0 === null) this.t0 = s.t;
    if (this.roomStart === null) this.roomStart = s.t;
    const dt = this.hist.length ? Math.min(0.5, s.t - this.hist.at(-1).t) : 0;
    this.hist.push(s);
    while (this.hist.length && s.t - this.hist[0].t > 3) this.hist.shift();

    const peak = Math.max(...this.hist.filter((h) => s.t - h.t <= 2).map((h) => h.sharp));
    s.blurred = s.sharp < SHARP_RATIO * peak;
    const fast = Math.abs(s.rot ?? 0) > MAX_TURN;
    const good = !s.blurred && !fast && s.bright >= DARK && s.texture >= BLANK;

    // coverage: footage that the pipeline can use, per heading sector, standing/turning vs walking
    if (good && s.heading != null && Math.abs(s.elevation ?? 0) < 45) {
      const c = this.coverage[Math.floor(s.heading / (360 / SECTORS)) % SECTORS];
      c[s.walking ? 'walked' : 'seen'] += dt;
    }
    if ((s.elevation ?? 0) > 45 && !fast) this.ceiling += dt;

    // back at start: the first seconds of the room seen again after a while
    const inRoom = s.t - this.roomStart;
    if (s.hash && good && inRoom < 2.5 && this.startHashes.length < 8) this.startHashes.push(s.hash);
    if (s.hash && good && inRoom > 45) {
      this.loopHits = this.startHashes.some((h) => hamming(h, s.hash) <= 10) ? this.loopHits + 1 : 0;
      if (this.loopHits >= 3) this.checks.backAtStart = true;
    }

    const covered = (k) => this.coverage.filter((c) => c[k] >= SECTOR_SECONDS).length;
    this.checks.perimeter = covered('seen') >= SECTORS - 2;
    this.checks.alongWalls = covered('walked') >= SECTORS - 4;
    this.checks.ceiling = this.ceiling >= 2.5;

    return { hint: this.hint(), coverage: this.coverage, checks: this.checks };
  }

  hint() {
    const h = (key, level = 'warn') => ({ key, level, text: HINTS[key] });
    if (this.held(1, (x) => x.bright < DARK) > 0.7) return h('dark', 'stop');
    if (this.held(0.6, (x) => Math.abs(x.rot ?? 0) > MAX_TURN) > 0.4) return h('turn', 'stop');
    if (this.held(1, (x) => x.blurred) > 0.6) return h('blur');
    if (this.held(2, (x) => x.texture < BLANK) > 1.5) return h('blank');
    if (this.held(4, (x) => (x.elevation ?? -12) > 0 && x.elevation < 45) > 3.5) return h('tiltDown', 'info');
    if (this.held(4, (x) => (x.elevation ?? -12) < -50) > 3.5) return h('tiltUp', 'info');
    return null;
  }

  // the next thing to film in this room, once nothing is wrong
  todo() {
    const c = this.checks, empty = this.coverage.findIndex((x) => x.seen < SECTOR_SECONDS);
    if (!c.marker) return 'Film the printed marker for 5 s from three angles, then tick it';
    if (!c.perimeter) return `Walk the room's edge facing in: the ${['N', 'NNE', 'ENE', 'E', 'ESE', 'SSE', 'S', 'SSW', 'WSW', 'W', 'WNW', 'NNW'][empty]} part of the ring is still empty`;
    if (!c.alongWalls) return 'Walk along each wall 0.5–1 m away, camera pointing ahead and angled toward the wall';
    if (!c.ceiling) return 'Point up at the ceiling for 3 s';
    if (!c.backAtStart) return 'Finish where you started this room';
    return 'Room done: walk slowly through the doorway into the next room, then tap Next room';
  }
}
