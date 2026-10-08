import { BRICK_VOXELS, GRAVITY, VOXEL_SIZE } from '../config';
import { Renderer } from '../render/Renderer';
import { materials } from '../world/materials';
import { UNIFORM, VoxelWorld } from '../world/VoxelWorld';
import { LightSystem } from './LightSystem';

// Particle flags (must match particle_common.wgsl).
const DEPOSIT = 0x20000;
const DUST = 0x40000;

export interface Tool {
  name: string;
  key: string;
  /** Carve radius in metres. */
  radius: number;
  /** Radius within which fragile materials (glass, porcelain) break. */
  fragileRadius: number;
  /** Debris launch speed, m/s. */
  impulse: number;
  /** Shots per second. */
  rate: number;
  pellets: number;
  spread: number;
  auto: boolean;
}

export const TOOLS: Tool[] = [
  { name: 'Pistol', key: '1', radius: 0.035, fragileRadius: 0.09, impulse: 2.5, rate: 7, pellets: 1, spread: 0, auto: true },
  { name: 'Shotgun', key: '2', radius: 0.045, fragileRadius: 0.1, impulse: 3.5, rate: 1.4, pellets: 10, spread: 0.055, auto: false },
  { name: 'Blaster', key: '3', radius: 0.3, fragileRadius: 0.45, impulse: 5, rate: 1.6, pellets: 1, spread: 0, auto: true },
  { name: 'Dynamite', key: '4', radius: 0.7, fragileRadius: 0.85, impulse: 8, rate: 0.6, pellets: 1, spread: 0, auto: false },
];

interface Chunk {
  xs: Int32Array;
  ys: Int32Array;
  zs: Int32Array;
  vals: Uint16Array;
  /** Indices of voxels with no chunk voxel directly below them. */
  bottom: Int32Array;
  vy: number; // voxels / s (positive = down)
  frac: number;
  fragile: boolean;
  lightAnchors: { light: number; anchor: number }[];
}

const PRICE = new Float32Array(256);
const SOFT = new Float32Array(256);
const FRAGILE = new Uint8Array(256); // 0 none, 1 shatter, 2 pane
const STRUCT = new Uint8Array(256);
const DEBRIS = new Float32Array(256);
const LIQUID = new Uint8Array(256);

function initTables() {
  for (const m of materials) {
    PRICE[m.id] = m.price;
    SOFT[m.id] = m.softness;
    FRAGILE[m.id] = m.fragile === 'pane' ? 2 : m.fragile === 'shatter' ? 1 : 0;
    STRUCT[m.id] = m.structural ? 1 : 0;
    DEBRIS[m.id] = m.debris;
    LIQUID[m.id] = m.liquid ? 1 : 0;
  }
}

function hash3(x: number, y: number, z: number): number {
  let h = (x * 374761393 + y * 668265263 + z * 2147483647) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

export class Destruction {
  readonly world: VoxelWorld;
  readonly renderer: Renderer;
  readonly lights: LightSystem;
  chunks: Chunk[] = [];
  /** Voxel keys currently owned by falling chunks (excluded from island searches). */
  private dynamicKeys = new Set<number>();
  bill = 0;
  destroyed = 0;
  /** Max particles spawned per event. */
  particleBudget = 30000;
  private removedScratch: number[] = [];

  constructor(world: VoxelWorld, renderer: Renderer, lights: LightSystem) {
    this.world = world;
    this.renderer = renderer;
    this.lights = lights;
    initTables();
    renderer.onDeposits = (items, count) => this.deposit(items, count);
  }

  /** Milliseconds spent in the last fire() call (carve + island search). */
  lastFireMs = 0;

  /** Fire a tool from `eye` (metres) along unit `dir`. */
  fire(tool: Tool, eye: [number, number, number], dir: [number, number, number]): boolean {
    const t0 = performance.now();
    const any = this.fireInner(tool, eye, dir);
    this.lastFireMs = performance.now() - t0;
    return any;
  }

  private fireInner(tool: Tool, eye: [number, number, number], dir: [number, number, number]): boolean {
    let any = false;
    for (let p = 0; p < tool.pellets; p++) {
      let d = dir;
      if (tool.spread > 0) {
        const s = tool.spread;
        const nd: [number, number, number] = [dir[0] + (Math.random() - 0.5) * 2 * s, dir[1] + (Math.random() - 0.5) * 2 * s, dir[2] + (Math.random() - 0.5) * 2 * s];
        const l = Math.hypot(nd[0], nd[1], nd[2]);
        d = [nd[0] / l, nd[1] / l, nd[2] / l];
      }
      const hit = this.world.raycast(eye[0] / VOXEL_SIZE, eye[1] / VOXEL_SIZE, eye[2] / VOXEL_SIZE, d[0], d[1], d[2], 80 / VOXEL_SIZE);
      if (!hit) continue;
      any = true;
      const n: [number, number, number] = [hit.nx, hit.ny, hit.nz];
      this.impact(hit.x, hit.y, hit.z, hit.v, n, d, tool);
    }
    return any;
  }

  /** Apply a tool's effect at voxel (x, y, z) whose value is v. */
  impact(x: number, y: number, z: number, v: number, n: [number, number, number], dir: [number, number, number], tool: Tool) {
    const id = v & 0xff;
    const removed = this.removedScratch;
    removed.length = 0;
    const r = tool.radius / VOXEL_SIZE;
    // Centre the blast slightly inside the surface for big tools.
    const cx = x + 0.5 - n[0] * Math.min(r * 0.3, 2), cy = y + 0.5 - n[1] * Math.min(r * 0.3, 2), cz = z + 0.5 - n[2] * Math.min(r * 0.3, 2);
    if (FRAGILE[id] === 2) {
      this.floodRemove(x, y, z, id, 600000, removed);
    }
    if (FRAGILE[id] !== 0 || tool.fragileRadius > tool.radius) {
      this.carve(cx, cy, cz, tool.fragileRadius / VOXEL_SIZE, true, removed);
    }
    const kept = this.carve(cx, cy, cz, r, false, removed);
    if (removed.length === 0) return;
    this.spawnDebris(removed, cx, cy, cz, n, dir, tool.impulse / VOXEL_SIZE);
    this.lights.flash([cx * VOXEL_SIZE + n[0] * 0.05, cy * VOXEL_SIZE + n[1] * 0.05, cz * VOXEL_SIZE + n[2] * 0.05], tool.radius);
    this.detachIslands(removed, { cx, cy, cz, r: kept - 1.01 });
  }

  /**
   * Remove voxels inside a jittered sphere. fragileOnly limits to breakable materials.
   * Returns the distance from the centre to the nearest surviving solid voxel
   * (capped at the search radius): removed voxels deeper than that can't border
   * anything, which lets island detection skip them.
   */
  private carve(cx: number, cy: number, cz: number, r: number, fragileOnly: boolean, out: number[]): number {
    const w = this.world;
    const R = Math.ceil(r * 1.2);
    const x0 = Math.max(0, Math.floor(cx - R)), x1 = Math.min(w.dx - 1, Math.ceil(cx + R));
    const y0 = Math.max(0, Math.floor(cy - R)), y1 = Math.min(w.dy - 1, Math.ceil(cy + R));
    const z0 = Math.max(0, Math.floor(cz - R)), z1 = Math.min(w.dz - 1, Math.ceil(cz + R));
    let minKept2 = R * R;
    // Walk bricks first so empty space (most of a blast) costs one lookup per 512 voxels,
    // then edit the brick's voxels in place.
    for (let bz = z0 >> 3; bz <= z1 >> 3; bz++)
      for (let by = y0 >> 3; by <= y1 >> 3; by++)
        for (let bx = x0 >> 3; bx <= x1 >> 3; bx++) {
          const ci = w.cellIndex(bx, by, bz);
          const g = w.grid[ci];
          if (g === 0) continue;
          const ex = Math.max(bx * 8 - cx, 0, cx - bx * 8 - 8);
          const ey = Math.max(by * 8 - cy, 0, cy - by * 8 - 8);
          const ez = Math.max(bz * 8 - cz, 0, cz - bz * 8 - 8);
          const near2 = ex * ex + ey * ey + ez * ez;
          if (near2 > R * R) continue;
          const uni = (g & UNIFORM) !== 0;
          if (uni) {
            // One material: skip the brick if even its largest jittered radius can't reach it.
            const id = g & 0xff;
            const soft = fragileOnly ? (FRAGILE[id] ? 1 : 0) : SOFT[id];
            const reach = soft <= 0 ? -1 : Math.max(r * Math.min(1.4, soft), Math.min(r, 2.5)) * 1.2;
            if (near2 > reach * reach) {
              minKept2 = Math.min(minKept2, near2);
              continue;
            }
          }
          let base = uni ? -1 : (g - 1) * BRICK_VOXELS;
          let edited = false;
          const vx1 = Math.min(x1, bx * 8 + 7), vy1 = Math.min(y1, by * 8 + 7), vz1 = Math.min(z1, bz * 8 + 7);
          for (let z = Math.max(z0, bz * 8); z <= vz1; z++)
            for (let y = Math.max(y0, by * 8); y <= vy1; y++)
              for (let x = Math.max(x0, bx * 8); x <= vx1; x++) {
                const dx = x + 0.5 - cx, dy = y + 0.5 - cy, dz = z + 0.5 - cz;
                const d2 = dx * dx + dy * dy + dz * dz;
                if (d2 > R * R) continue;
                const li = (x & 7) | ((y & 7) << 3) | ((z & 7) << 6);
                const v = base < 0 ? g & 0xffff : w.pool[base + li];
                if (v === 0) continue;
                const id = v & 0xff;
                const soft = fragileOnly ? (FRAGILE[id] ? 1 : 0) : SOFT[id];
                // Hard materials shrink the hole, but a hit always punches through ~5 voxels
                // so rods, chains and table legs can be shot through.
                const rr = soft <= 0 ? -1 : Math.max(r * Math.min(1.4, soft), Math.min(r, 2.5)) * (0.8 + 0.4 * hash3(x, y, z));
                if (d2 > rr * rr) {
                  if (d2 < minKept2) minKept2 = d2;
                  continue;
                }
                if (base < 0) base = w.editableBrick(ci) * BRICK_VOXELS;
                w.pool[base + li] = 0;
                edited = true;
                out.push(x, y, z, v);
              }
          if (edited) {
            w.dirtyBricks.add(base / BRICK_VOXELS);
            w.dirtyCells.add(ci);
          }
        }
    return Math.sqrt(minKept2);
  }

  /** Remove the 6-connected component of material `id` containing (x,y,z). */
  private floodRemove(x: number, y: number, z: number, id: number, budget: number, out: number[]) {
    const w = this.world;
    const stack = [x, y, z];
    const seen = new Set<number>();
    const key = (a: number, b: number, c: number) => a + w.dx * (b + w.dy * c);
    seen.add(key(x, y, z));
    let n = 0;
    while (stack.length && n < budget) {
      const cz = stack.pop()!, cy = stack.pop()!, cx = stack.pop()!;
      const v = w.get(cx, cy, cz);
      if ((v & 0xff) !== id) continue;
      w.set(cx, cy, cz, 0);
      out.push(cx, cy, cz, v);
      n++;
      for (let k = 0; k < 6; k++) {
        const nx = cx + (k === 0 ? 1 : k === 1 ? -1 : 0);
        const ny = cy + (k === 2 ? 1 : k === 3 ? -1 : 0);
        const nz = cz + (k === 4 ? 1 : k === 5 ? -1 : 0);
        if (!w.inBounds(nx, ny, nz)) continue;
        const kk = key(nx, ny, nz);
        if (seen.has(kk)) continue;
        seen.add(kk);
        if ((w.get(nx, ny, nz) & 0xff) === id) stack.push(nx, ny, nz);
      }
    }
  }

  private spawnDebris(removed: number[], cx: number, cy: number, cz: number, n: [number, number, number], dir: [number, number, number], impulse: number) {
    const count = removed.length / 4;
    const keep = Math.min(1, this.particleBudget / Math.max(1, count));
    for (let i = 0; i < removed.length; i += 4) {
      const x = removed[i], y = removed[i + 1], z = removed[i + 2], v = removed[i + 3];
      const id = v & 0xff;
      this.bill += PRICE[id];
      this.destroyed++;
      if (Math.random() > DEBRIS[id] * keep) continue;
      let dx = x + 0.5 - cx, dy = y + 0.5 - cy, dz = z + 0.5 - cz;
      const l = Math.hypot(dx, dy, dz) || 1;
      dx /= l; dy /= l; dz /= l;
      const s = impulse * (0.3 + Math.random() * 0.9);
      const glassy = FRAGILE[id] !== 0;
      const vx = (dx * 0.7 + n[0] * 0.6 - dir[0] * 0.2) * s + (Math.random() - 0.5) * impulse * 0.4;
      const vy = (dy * 0.7 + n[1] * 0.6 - dir[1] * 0.2) * s + (Math.random() - 0.2) * impulse * 0.5;
      const vz = (dz * 0.7 + n[2] * 0.6 - dir[2] * 0.2) * s + (Math.random() - 0.5) * impulse * 0.4;
      let flags = 0;
      if (LIQUID[id]) flags = 0;
      else if (Math.random() < (glassy ? 0.55 : 0.75)) flags = DEPOSIT;
      else if (Math.random() < 0.5) flags = DUST;
      this.renderer.spawn(x + 0.5, y + 0.5, z + 0.5, vx, vy, vz, v | flags, 5 + Math.random() * 4);
    }
  }

  /**
   * After removing voxels, find solid neighbours whose connected component no
   * longer touches anything structural, and turn those into falling chunks.
   */
  private detachIslands(removed: number[], core?: { cx: number; cy: number; cz: number; r: number }) {
    const w = this.world;
    const key = (a: number, b: number, c: number) => a + w.dx * (b + w.dy * c);
    const seeds: number[] = [];
    const seedSet = new Set<number>();
    const core2 = core && core.r > 0 ? core.r * core.r : -1;
    for (let i = 0; i < removed.length; i += 4) {
      const x = removed[i], y = removed[i + 1], z = removed[i + 2];
      if (core2 > 0) {
        const dx = x + 0.5 - core!.cx, dy = y + 0.5 - core!.cy, dz = z + 0.5 - core!.cz;
        if (dx * dx + dy * dy + dz * dz < core2) continue; // all neighbours are air
      }
      for (let k = 0; k < 6; k++) {
        const nx = x + (k === 0 ? 1 : k === 1 ? -1 : 0);
        const ny = y + (k === 2 ? 1 : k === 3 ? -1 : 0);
        const nz = z + (k === 4 ? 1 : k === 5 ? -1 : 0);
        const v = w.get(nx, ny, nz);
        if (v === 0 || STRUCT[v & 0xff]) continue;
        const kk = key(nx, ny, nz);
        if (seedSet.has(kk) || this.dynamicKeys.has(kk)) continue;
        seedSet.add(kk);
        seeds.push(nx, ny, nz);
      }
    }
    if (!seeds.length) return;
    const supported = new Set<number>();
    const budget = 400000;
    for (let s = 0; s < seeds.length; s += 3) {
      const sk = key(seeds[s], seeds[s + 1], seeds[s + 2]);
      if (supported.has(sk)) continue;
      const comp = this.component(seeds[s], seeds[s + 1], seeds[s + 2], supported, budget);
      if (comp === null) continue; // anchored (or too big to care)
      if (comp.length < 48) this.crumble(comp);
      else this.makeChunk(comp);
    }
  }

  /** Tiny floating fragments just become debris. */
  private crumble(keys: number[]) {
    const w = this.world;
    const DX = w.dx, DXY = w.dx * w.dy;
    const removed: number[] = [];
    let mx = 0, my = 0, mz = 0;
    for (const k of keys) {
      const z = Math.floor(k / DXY);
      const rem = k - z * DXY;
      const y = Math.floor(rem / DX);
      const x = rem - y * DX;
      const v = w.get(x, y, z);
      if (!v) continue;
      w.set(x, y, z, 0);
      removed.push(x, y, z, v);
      mx += x; my += y; mz += z;
    }
    if (!removed.length) return;
    const n = removed.length / 4;
    this.spawnDebris(removed, mx / n, my / n + 1, mz / n, [0, 1, 0], [0, -1, 0], 0.6 / VOXEL_SIZE);
  }

  /**
   * DFS (down-first) over non-air voxels. Returns the component's voxel keys if it
   * is floating, or null if it reaches a structural voxel / the floor / the budget.
   * Anchored components are added to `supported` so later seeds stop early.
   */
  private component(x: number, y: number, z: number, supported: Set<number>, budget: number): number[] | null {
    const w = this.world;
    const DX = w.dx, DXY = w.dx * w.dy;
    const visited = new Set<number>();
    const stack: number[] = [x + DX * (y + w.dy * z)];
    visited.add(stack[0]);
    const order = [2, 0, 1, 4, 5, 3]; // push +y first so -y pops first
    let anchored = false;
    while (stack.length) {
      const k = stack.pop()!;
      const cz = Math.floor(k / DXY);
      const rem = k - cz * DXY;
      const cy = Math.floor(rem / DX);
      const cx = rem - cy * DX;
      if (cy === 0) { anchored = true; break; }
      for (const o of order) {
        const nx = cx + (o === 0 ? 1 : o === 1 ? -1 : 0);
        const ny = cy + (o === 2 ? 1 : o === 3 ? -1 : 0);
        const nz = cz + (o === 4 ? 1 : o === 5 ? -1 : 0);
        if (nx < 0 || ny < 0 || nz < 0 || nx >= w.dx || ny >= w.dy || nz >= w.dz) continue;
        const nk = nx + DX * (ny + w.dy * nz);
        if (visited.has(nk) || this.dynamicKeys.has(nk)) continue;
        const v = w.get(nx, ny, nz);
        if (v === 0) continue;
        if (STRUCT[v & 0xff] || supported.has(nk)) { anchored = true; break; }
        visited.add(nk);
        stack.push(nk);
      }
      if (anchored || visited.size > budget) { anchored = true; break; }
    }
    if (anchored) {
      for (const k of visited) supported.add(k);
      return null;
    }
    return Array.from(visited);
  }

  private makeChunk(keys: number[]) {
    const w = this.world;
    const DX = w.dx, DXY = w.dx * w.dy;
    const n = keys.length;
    const xs = new Int32Array(n), ys = new Int32Array(n), zs = new Int32Array(n);
    const vals = new Uint16Array(n);
    const set = new Set(keys);
    let fragile = false;
    const bottom: number[] = [];
    for (let i = 0; i < n; i++) {
      const k = keys[i];
      const z = Math.floor(k / DXY);
      const rem = k - z * DXY;
      const y = Math.floor(rem / DX);
      const x = rem - y * DX;
      xs[i] = x; ys[i] = y; zs[i] = z;
      const v = w.get(x, y, z);
      vals[i] = v;
      if (FRAGILE[v & 0xff]) fragile = true;
      if (!set.has(k - DX)) bottom.push(i);
    }
    const lightAnchors = this.lights.anchorsIn(set);
    for (const k of keys) this.dynamicKeys.add(k);
    this.chunks.push({ xs, ys, zs, vals, bottom: Int32Array.from(bottom), vy: 0, frac: 0, fragile, lightAnchors });
  }

  update(dt: number) {
    if (!this.chunks.length) return;
    const g = GRAVITY / VOXEL_SIZE;
    const w = this.world;
    this.chunks.sort((a, b) => minY(a) - minY(b));
    const next: Chunk[] = [];
    for (const c of this.chunks) {
      c.vy = Math.min(c.vy + g * dt, 25 / VOXEL_SIZE);
      const travel = c.vy * dt + c.frac;
      let steps = Math.floor(travel);
      c.frac = travel - steps;
      let drop = 0;
      let landed = false;
      let blocker = -1;
      while (steps-- > 0) {
        for (let b = 0; b < c.bottom.length; b++) {
          const i = c.bottom[b];
          if (!c.vals[i]) continue;
          const by = c.ys[i] - drop - 1;
          if (by < 0) { blocker = i; break; }
          // Cells still holding (this or another) falling chunk's voxels don't block:
          // they move out of the way this frame.
          if (w.get(c.xs[i], by, c.zs[i]) !== 0 && !this.dynamicKeys.has(c.xs[i] + w.dx * (by + w.dy * c.zs[i]))) { blocker = i; break; }
        }
        if (blocker >= 0) { landed = true; break; }
        drop++;
      }
      if (drop > 0) {
        const DX = w.dx, DY = w.dy;
        for (let i = 0; i < c.xs.length; i++) {
          if (!c.vals[i]) continue;
          // Voxels shot off mid-fall are dropped from the chunk.
          if (w.get(c.xs[i], c.ys[i], c.zs[i]) !== c.vals[i]) { c.vals[i] = 0; continue; }
          w.set(c.xs[i], c.ys[i], c.zs[i], 0);
          this.dynamicKeys.delete(c.xs[i] + DX * (c.ys[i] + DY * c.zs[i]));
        }
        for (let i = 0; i < c.xs.length; i++) {
          c.ys[i] -= drop;
          if (!c.vals[i]) continue;
          w.set(c.xs[i], c.ys[i], c.zs[i], c.vals[i]);
          this.dynamicKeys.add(c.xs[i] + DX * (c.ys[i] + DY * c.zs[i]));
        }
        this.lights.moveAnchors(c.lightAnchors, -drop);
      }
      if (landed) {
        const speed = c.vy * VOXEL_SIZE;
        if (this.land(c, speed)) next.push(c);
        else {
          this.release(c);
          // If we landed on something that is itself floating, both keep falling.
          const bx = c.xs[blocker], by = c.ys[blocker] - 1, bz = c.zs[blocker];
          if (by >= 0 && w.get(bx, by, bz) !== 0) this.detachIslands([bx, by + 1, bz, 0]);
        }
      } else {
        next.push(c);
      }
    }
    this.chunks = next;
  }

  private release(c: Chunk) {
    const w = this.world;
    for (let i = 0; i < c.xs.length; i++) this.dynamicKeys.delete(c.xs[i] + w.dx * (c.ys[i] + w.dy * c.zs[i]));
  }

  get fallingCount(): number {
    return this.chunks.length;
  }

  /** Returns true if the chunk keeps falling (it broke through what it hit). */
  private land(c: Chunk, speed: number): boolean {
    const w = this.world;
    // Smash fragile things underneath.
    let broke = 0;
    if (speed > 1.5) {
      const tool: Tool = { ...TOOLS[0], radius: 0.02, fragileRadius: 0.12, impulse: speed * 0.4 };
      for (let b = 0; b < c.bottom.length && broke < 6; b += Math.max(1, Math.floor(c.bottom.length / 24))) {
        const i = c.bottom[b];
        const v = w.get(c.xs[i], c.ys[i] - 1, c.zs[i]);
        if (v && FRAGILE[v & 0xff]) {
          this.impact(c.xs[i], c.ys[i] - 1, c.zs[i], v, [0, 1, 0], [0, -1, 0], tool);
          broke++;
        }
      }
    }
    if (broke > 0) {
      c.vy *= 0.6;
      return true;
    }
    if (c.fragile && speed > 2.0) {
      // Shatter the breakable parts; sturdy parts stay where they fell.
      const removed: number[] = [];
      for (let i = 0; i < c.xs.length; i++) {
        if (!c.vals[i] || !FRAGILE[c.vals[i] & 0xff]) continue;
        if (w.get(c.xs[i], c.ys[i], c.zs[i]) !== c.vals[i]) continue;
        w.set(c.xs[i], c.ys[i], c.zs[i], 0);
        removed.push(c.xs[i], c.ys[i], c.zs[i], c.vals[i]);
      }
      if (removed.length) {
        let mx = 0, my = 0, mz = 0;
        for (let i = 0; i < removed.length; i += 4) { mx += removed[i]; my += removed[i + 1]; mz += removed[i + 2]; }
        const k = 4 / removed.length;
        this.spawnDebris(removed, mx * k, my * k - 4, mz * k, [0, 1, 0], [0, -1, 0], (speed * 0.45) / VOXEL_SIZE);
        this.lights.flash([mx * k * VOXEL_SIZE, my * k * VOXEL_SIZE, mz * k * VOXEL_SIZE], 0.2);
        // Whatever sturdy bits remain may now be floating fragments.
        this.detachIslands(removed);
      }
    }
    return false;
  }

  /** Settled debris from the GPU becomes real voxels again. */
  private deposit(items: Uint32Array, count: number) {
    const w = this.world;
    const DX = w.dx, DXY = w.dx * w.dy;
    for (let i = 0; i < count; i++) {
      const k = items[i * 2];
      const v = items[i * 2 + 1] & 0xffff;
      if (!v) continue;
      const z = Math.floor(k / DXY);
      const rem = k - z * DXY;
      const y = Math.floor(rem / DX);
      const x = rem - y * DX;
      for (let up = 0; up < 3; up++) {
        if (w.get(x, y + up, z) === 0) {
          if (up === 0 || w.get(x, y + up - 1, z) !== 0) w.set(x, y + up, z, v);
          break;
        }
      }
    }
  }
}

function minY(c: Chunk): number {
  let m = Infinity;
  for (let b = 0; b < c.bottom.length; b++) m = Math.min(m, c.ys[c.bottom[b]]);
  return m;
}

