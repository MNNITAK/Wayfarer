import { isFree } from './nav.js';

// First-person walker: pointer-lock mouse look, WASD with acceleration, grid collision.
// On touch screens there is no pointer lock: an on-screen stick moves, dragging anywhere else looks.
export class Player {
  constructor(camera, dom, grid, { eyeHeight = 1.6, floorY = 0, radius = 0.22 } = {}) {
    Object.assign(this, { camera, dom, grid, eyeHeight, floorY, radius });
    this.x = 0; this.z = 0; this.y = floorY + eyeHeight;
    this.yaw = 0; this.pitch = 0;
    this.vx = 0; this.vz = 0; this.vy = 0;
    this.fly = false;
    this.keys = new Set();
    this.onLockChange = () => {};
    // no pointer lock at all (iPhone Safari, some in-app browsers) also means touch controls
    this.touch = matchMedia('(pointer: coarse)').matches || !('requestPointerLock' in Element.prototype);
    this.started = false; // touch mode: stays "locked" once the visitor taps Start
    this.stick = { f: 0, s: 0 };

    addEventListener('keydown', (e) => { if (!isTyping(e)) this.keys.add(e.code); });
    addEventListener('keyup', (e) => this.keys.delete(e.code));
    addEventListener('blur', () => this.keys.clear());
    document.addEventListener('pointerlockchange', () => this.onLockChange(this.locked));
    document.addEventListener('mousemove', (e) => {
      if (!this.locked) return;
      this.yaw -= e.movementX * 0.0022;
      this.pitch = Math.max(-1.45, Math.min(1.45, this.pitch - e.movementY * 0.0022));
    });
    if (this.touch) this.dragToLook();
  }

  get locked() { return this.touch ? this.started : document.pointerLockElement === this.dom; }
  lock() {
    if (!this.touch) {
      // browser refused the lock (iframe, permissions): still close the overlay; clicking the scene retries
      const refused = () => this.onLockChange(true);
      document.addEventListener('pointerlockerror', refused, { once: true });
      return this.dom.requestPointerLock()?.catch?.(refused);
    }
    if (!this.started) { this.started = true; this.onLockChange(true); }
  }
  unlock() { if (!this.touch && this.locked) document.exitPointerLock(); }

  dragToLook() {
    let id = null, lx = 0, ly = 0;
    this.dom.addEventListener('pointerdown', (e) => {
      if (id !== null || e.pointerType === 'mouse') return;
      id = e.pointerId; lx = e.clientX; ly = e.clientY;
    });
    addEventListener('pointermove', (e) => {
      if (e.pointerId !== id || !this.started) return;
      const k = 4 / Math.max(innerWidth, innerHeight); // a full-width swipe turns ~200°
      this.yaw += (e.clientX - lx) * k;
      this.pitch = Math.max(-1.45, Math.min(1.45, this.pitch + (e.clientY - ly) * k));
      lx = e.clientX; ly = e.clientY;
    });
    const end = (e) => { if (e.pointerId === id) id = null; };
    addEventListener('pointerup', end); addEventListener('pointercancel', end);
  }

  // on-screen joystick: el is the base, its first child the knob
  bindStick(el) {
    const knob = el.firstElementChild;
    let id = null, cx = 0, cy = 0;
    const set = (x, y) => {
      const r = el.clientWidth / 2, d = Math.hypot(x, y), k = d > r ? r / d : 1;
      x *= k; y *= k;
      knob.style.transform = `translate(${x}px, ${y}px)`;
      this.stick = { f: -y / r, s: x / r };
    };
    el.addEventListener('pointerdown', (e) => {
      if (id !== null) return;
      id = e.pointerId; el.setPointerCapture(id);
      const b = el.getBoundingClientRect(); cx = b.left + b.width / 2; cy = b.top + b.height / 2;
      set(e.clientX - cx, e.clientY - cy);
    });
    el.addEventListener('pointermove', (e) => { if (e.pointerId === id) set(e.clientX - cx, e.clientY - cy); });
    const end = (e) => { if (e.pointerId === id) { id = null; set(0, 0); } };
    el.addEventListener('pointerup', end); el.addEventListener('pointercancel', end);
  }

  teleport(x, z, yaw = this.yaw) {
    this.x = x; this.z = z; this.yaw = yaw; this.vx = this.vz = 0;
    this.y = this.floorY + this.eyeHeight;
  }

  toggleFly() {
    this.fly = !this.fly;
    if (!this.fly) {
      this.y = this.floorY + this.eyeHeight;
      // landed inside furniture? stay where you took off from instead
      if (!isFree(this.grid, this.x, this.z, this.radius) && this.flyStart) this.teleport(...this.flyStart);
    } else this.flyStart = [this.x, this.z];
  }

  update(dt) {
    const k = this.keys, run = k.has('ShiftLeft') || k.has('ShiftRight');
    const speed = (this.fly ? 4 : 1.6) * (run ? 2.2 : 1);
    let f = (k.has('KeyW') || k.has('ArrowUp') ? 1 : 0) - (k.has('KeyS') || k.has('ArrowDown') ? 1 : 0);
    let s = (k.has('KeyD') || k.has('ArrowRight') ? 1 : 0) - (k.has('KeyA') || k.has('ArrowLeft') ? 1 : 0);
    f += this.stick.f; s += this.stick.s; // analog: a half-pushed stick walks at half speed
    const len = Math.hypot(f, s);
    if (len > 1) { f /= len; s /= len; }
    const sin = Math.sin(this.yaw), cos = Math.cos(this.yaw);
    // forward is -z at yaw 0
    const tx = (-sin * f + cos * s) * speed, tz = (-cos * f - sin * s) * speed;
    const a = 1 - Math.exp(-dt * 10); // smooth accel/decel
    this.vx += (tx - this.vx) * a; this.vz += (tz - this.vz) * a;

    const nx = this.x + this.vx * dt, nz = this.z + this.vz * dt;
    if (this.fly) {
      this.x = nx; this.z = nz;
      const up = (k.has('Space') ? 1 : 0) - (k.has('KeyC') || k.has('ControlLeft') ? 1 : 0);
      this.vy += (up * speed - this.vy) * a;
      this.y = Math.max(this.floorY + 0.3, this.y + this.vy * dt);
    } else {
      // axis-separated so you slide along walls instead of sticking
      if (isFree(this.grid, nx, this.z, this.radius)) this.x = nx; else this.vx = 0;
      if (isFree(this.grid, this.x, nz, this.radius)) this.z = nz; else this.vz = 0;
      const moving = Math.hypot(this.vx, this.vz);
      this.bob = (this.bob || 0) + moving * dt * 6;
      this.y = this.floorY + this.eyeHeight + Math.sin(this.bob) * 0.015 * Math.min(1, moving);
    }
    this.camera.position.set(this.x, this.y, this.z);
    this.camera.rotation.set(this.pitch, this.yaw, 0, 'YXZ');
  }
}

export const isTyping = (e) => /INPUT|SELECT|TEXTAREA/.test(e.target?.tagName);
