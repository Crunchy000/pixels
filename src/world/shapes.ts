// Signed-distance primitives (in metres) and a brick-aware rasteriser that
// writes them into a VoxelWorld. Whole bricks inside a shape become uniform
// cells without touching individual voxels, so big slabs are nearly free.

import { BRICK, BRICK_VOXELS, VOXEL_SIZE } from '../config';
import { UNIFORM, VoxelWorld } from './VoxelWorld';

export type Vec3 = [number, number, number];
export type SDF = (x: number, y: number, z: number) => number;

export interface Shape {
  sdf: SDF;
  min: Vec3;
  max: Vec3;
  /** Lipschitz slack: >1 for SDFs that underestimate poorly (ellipsoids, scaled shapes). */
  slack?: number;
  /** Exact AABB test (axis-aligned boxes): fully contains / fully misses the given box. */
  contains?: (x0: number, y0: number, z0: number, x1: number, y1: number, z1: number) => boolean;
}

/** A voxel value, or a function from a voxel centre (metres) to a value. */
export type Paint = number | ((x: number, y: number, z: number) => number);

const len2 = (x: number, y: number) => Math.sqrt(x * x + y * y);
const len3 = (x: number, y: number, z: number) => Math.sqrt(x * x + y * y + z * z);

// ---------------------------------------------------------------------------
// Primitives

export function boxMinMax(x0: number, y0: number, z0: number, x1: number, y1: number, z1: number): Shape {
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, cz = (z0 + z1) / 2;
  const hx = Math.abs(x1 - x0) / 2, hy = Math.abs(y1 - y0) / 2, hz = Math.abs(z1 - z0) / 2;
  return box([cx, cy, cz], [hx, hy, hz]);
}

export function box(c: Vec3, h: Vec3): Shape {
  const [cx, cy, cz] = c;
  const [hx, hy, hz] = h;
  return {
    sdf: (x, y, z) => {
      const qx = Math.abs(x - cx) - hx, qy = Math.abs(y - cy) - hy, qz = Math.abs(z - cz) - hz;
      const out = len3(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0));
      return out + Math.min(Math.max(qx, qy, qz), 0);
    },
    min: [cx - hx, cy - hy, cz - hz],
    max: [cx + hx, cy + hy, cz + hz],
    contains: (x0, y0, z0, x1, y1, z1) =>
      x0 >= cx - hx && x1 <= cx + hx && y0 >= cy - hy && y1 <= cy + hy && z0 >= cz - hz && z1 <= cz + hz,
  };
}

export function roundBox(c: Vec3, h: Vec3, r: number): Shape {
  const [cx, cy, cz] = c;
  const hx = h[0] - r, hy = h[1] - r, hz = h[2] - r;
  return {
    sdf: (x, y, z) => {
      const qx = Math.abs(x - cx) - hx, qy = Math.abs(y - cy) - hy, qz = Math.abs(z - cz) - hz;
      return len3(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0)) + Math.min(Math.max(qx, qy, qz), 0) - r;
    },
    min: [cx - h[0], cy - h[1], cz - h[2]],
    max: [cx + h[0], cy + h[1], cz + h[2]],
  };
}

export function sphere(c: Vec3, r: number): Shape {
  const [cx, cy, cz] = c;
  return {
    sdf: (x, y, z) => len3(x - cx, y - cy, z - cz) - r,
    min: [cx - r, cy - r, cz - r],
    max: [cx + r, cy + r, cz + r],
  };
}

export function ellipsoid(c: Vec3, r: Vec3): Shape {
  const [cx, cy, cz] = c;
  const [rx, ry, rz] = r;
  const rmin = Math.min(rx, ry, rz);
  return {
    sdf: (x, y, z) => (len3((x - cx) / rx, (y - cy) / ry, (z - cz) / rz) - 1) * rmin,
    min: [cx - rx, cy - ry, cz - rz],
    max: [cx + rx, cy + ry, cz + rz],
    slack: Math.max(rx, ry, rz) / rmin,
  };
}

type Axis = 'x' | 'y' | 'z';

/** Capped cylinder centred at c. */
export function cylinder(c: Vec3, r: number, halfHeight: number, axis: Axis = 'y'): Shape {
  r = Math.max(r, MIN_COLUMN_R);
  halfHeight = Math.max(halfHeight, VOXEL_SIZE * 0.5);
  const [cx, cy, cz] = c;
  const sdf: SDF = (x, y, z) => {
    let a: number, b: number, h: number;
    if (axis === 'y') { a = x - cx; b = z - cz; h = y - cy; }
    else if (axis === 'x') { a = y - cy; b = z - cz; h = x - cx; }
    else { a = x - cx; b = y - cy; h = z - cz; }
    const dr = len2(a, b) - r, dh = Math.abs(h) - halfHeight;
    return Math.min(Math.max(dr, dh), 0) + len2(Math.max(dr, 0), Math.max(dh, 0));
  };
  const ext: Vec3 = axis === 'y' ? [r, halfHeight, r] : axis === 'x' ? [halfHeight, r, r] : [r, r, halfHeight];
  return { sdf, min: [cx - ext[0], cy - ext[1], cz - ext[2]], max: [cx + ext[0], cy + ext[1], cz + ext[2]] };
}

/**
 * Thin features must stay 6-connected after voxelisation (or the destruction
 * code sees them as floating). A line needs a radius of at least ~sqrt(3)/2
 * voxels for that; axis-aligned columns are fine at half a voxel.
 */
const MIN_LINE_R = 0.9 * VOXEL_SIZE;
const MIN_COLUMN_R = 0.6 * VOXEL_SIZE;
const MIN_SHELL = 1.75 * VOXEL_SIZE;

export function capsule(a: Vec3, b: Vec3, r: number): Shape {
  r = Math.max(r, MIN_LINE_R);
  const [ax, ay, az] = a;
  const bax = b[0] - ax, bay = b[1] - ay, baz = b[2] - az;
  const bb = bax * bax + bay * bay + baz * baz || 1e-12;
  return {
    sdf: (x, y, z) => {
      const px = x - ax, py = y - ay, pz = z - az;
      const h = Math.min(1, Math.max(0, (px * bax + py * bay + pz * baz) / bb));
      return len3(px - bax * h, py - bay * h, pz - baz * h) - r;
    },
    min: [Math.min(ax, b[0]) - r, Math.min(ay, b[1]) - r, Math.min(az, b[2]) - r],
    max: [Math.max(ax, b[0]) + r, Math.max(ay, b[1]) + r, Math.max(az, b[2]) + r],
  };
}

export function torus(c: Vec3, R: number, r: number, axis: Axis = 'y'): Shape {
  r = Math.max(r, MIN_LINE_R);
  const [cx, cy, cz] = c;
  const sdf: SDF = (x, y, z) => {
    let a: number, b: number, h: number;
    if (axis === 'y') { a = x - cx; b = z - cz; h = y - cy; }
    else if (axis === 'x') { a = y - cy; b = z - cz; h = x - cx; }
    else { a = x - cx; b = y - cy; h = z - cz; }
    return len2(len2(a, b) - R, h) - r;
  };
  const e = R + r;
  const ext: Vec3 = axis === 'y' ? [e, r, e] : axis === 'x' ? [r, e, e] : [e, e, r];
  return { sdf, min: [cx - ext[0], cy - ext[1], cz - ext[2]], max: [cx + ext[0], cy + ext[1], cz + ext[2]] };
}

/** 2D signed distance to a closed polygon (Inigo Quilez). */
function sdPolygon(px: number, py: number, pts: number[]): number {
  const n = pts.length / 2;
  let d = (px - pts[0]) ** 2 + (py - pts[1]) ** 2;
  let s = 1;
  for (let i = 0, j = n - 1; i < n; j = i, i++) {
    const vix = pts[i * 2], viy = pts[i * 2 + 1], vjx = pts[j * 2], vjy = pts[j * 2 + 1];
    const ex = vjx - vix, ey = vjy - viy;
    const wx = px - vix, wy = py - viy;
    const t = Math.min(1, Math.max(0, (wx * ex + wy * ey) / (ex * ex + ey * ey || 1e-12)));
    const bx = wx - ex * t, by = wy - ey * t;
    d = Math.min(d, bx * bx + by * by);
    const c1 = py >= viy, c2 = py < vjy, c3 = ex * wy > ey * wx;
    if ((c1 && c2 && c3) || (!c1 && !c2 && !c3)) s = -s;
  }
  return s * Math.sqrt(d);
}

/** 2D unsigned distance to an open polyline. */
function sdPolyline(px: number, py: number, pts: number[]): number {
  let d = Infinity;
  for (let i = 0; i + 3 < pts.length; i += 2) {
    const ax = pts[i], ay = pts[i + 1], ex = pts[i + 2] - ax, ey = pts[i + 3] - ay;
    const wx = px - ax, wy = py - ay;
    const t = Math.min(1, Math.max(0, (wx * ex + wy * ey) / (ex * ex + ey * ey || 1e-12)));
    d = Math.min(d, (wx - ex * t) ** 2 + (wy - ey * t) ** 2);
  }
  return Math.sqrt(d);
}

/**
 * Surface of revolution around a vertical axis through c.
 * profile: [r0, y0, r1, y1, ...] relative to c. Solid lathes close the profile
 * against the axis; shell lathes are a wall of the given thickness around it.
 */
export function lathe(c: Vec3, profile: number[], shell = 0): Shape {
  if (shell > 0) shell = Math.max(shell, MIN_SHELL);
  const [cx, cy, cz] = c;
  let rmax = 0, ymin = Infinity, ymax = -Infinity;
  for (let i = 0; i < profile.length; i += 2) {
    rmax = Math.max(rmax, profile[i]);
    ymin = Math.min(ymin, profile[i + 1]);
    ymax = Math.max(ymax, profile[i + 1]);
  }
  const pad = shell / 2;
  let sdf: SDF;
  if (shell > 0) {
    sdf = (x, y, z) => sdPolyline(len2(x - cx, z - cz), y - cy, profile) - shell / 2;
  } else {
    const poly = [0, profile[1], ...profile, 0, profile[profile.length - 1]];
    sdf = (x, y, z) => sdPolygon(len2(x - cx, z - cz), y - cy, poly);
  }
  return {
    sdf,
    min: [cx - rmax - pad, cy + ymin - pad, cz - rmax - pad],
    max: [cx + rmax + pad, cy + ymax + pad, cz + rmax + pad],
  };
}

/** Extrude a closed xz polygon (relative to c) between y0..y1 (absolute). */
export function prism(c: Vec3, pts: number[], y0: number, y1: number): Shape {
  const [cx, , cz] = c;
  let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity;
  for (let i = 0; i < pts.length; i += 2) {
    x0 = Math.min(x0, pts[i]); x1 = Math.max(x1, pts[i]);
    z0 = Math.min(z0, pts[i + 1]); z1 = Math.max(z1, pts[i + 1]);
  }
  const ym = (y0 + y1) / 2, hy = (y1 - y0) / 2;
  return {
    sdf: (x, y, z) => {
      const d2 = sdPolygon(x - cx, z - cz, pts);
      const dy = Math.abs(y - ym) - hy;
      return Math.min(Math.max(d2, dy), 0) + len2(Math.max(d2, 0), Math.max(dy, 0));
    },
    min: [cx + x0, y0, cz + z0],
    max: [cx + x1, y1, cz + z1],
  };
}

// ---------------------------------------------------------------------------
// Combinators

export function union(...shapes: Shape[]): Shape {
  return {
    sdf: (x, y, z) => {
      let d = Infinity;
      for (const s of shapes) d = Math.min(d, s.sdf(x, y, z));
      return d;
    },
    min: [0, 1, 2].map((i) => Math.min(...shapes.map((s) => s.min[i]))) as Vec3,
    max: [0, 1, 2].map((i) => Math.max(...shapes.map((s) => s.max[i]))) as Vec3,
    slack: Math.max(1, ...shapes.map((s) => s.slack ?? 1)),
  };
}

export function subtract(a: Shape, ...b: Shape[]): Shape {
  return {
    sdf: (x, y, z) => {
      let d = a.sdf(x, y, z);
      for (const s of b) d = Math.max(d, -s.sdf(x, y, z));
      return d;
    },
    min: a.min,
    max: a.max,
    slack: Math.max(a.slack ?? 1, ...b.map((s) => s.slack ?? 1)),
  };
}

export function intersect(a: Shape, b: Shape): Shape {
  return {
    sdf: (x, y, z) => Math.max(a.sdf(x, y, z), b.sdf(x, y, z)),
    min: [Math.max(a.min[0], b.min[0]), Math.max(a.min[1], b.min[1]), Math.max(a.min[2], b.min[2])],
    max: [Math.min(a.max[0], b.max[0]), Math.min(a.max[1], b.max[1]), Math.min(a.max[2], b.max[2])],
    slack: Math.max(a.slack ?? 1, b.slack ?? 1),
  };
}

/** Rotate a shape by `angle` radians around the vertical axis through `pivot`. */
export function rotateY(s: Shape, angle: number, pivot: Vec3): Shape {
  const c = Math.cos(angle), sn = Math.sin(angle);
  const [px, , pz] = pivot;
  const corners: Vec3[] = [];
  for (const x of [s.min[0], s.max[0]]) for (const z of [s.min[2], s.max[2]]) {
    const dx = x - px, dz = z - pz;
    corners.push([px + c * dx - sn * dz, 0, pz + sn * dx + c * dz]);
  }
  return {
    sdf: (x, y, z) => {
      const dx = x - px, dz = z - pz;
      return s.sdf(px + c * dx + sn * dz, y, pz - sn * dx + c * dz);
    },
    min: [Math.min(...corners.map((k) => k[0])), s.min[1], Math.min(...corners.map((k) => k[2]))],
    max: [Math.max(...corners.map((k) => k[0])), s.max[1], Math.max(...corners.map((k) => k[2]))],
    slack: s.slack,
  };
}

/** Rotate around the x axis through `pivot` (tilt things forward/back). */
export function rotateX(s: Shape, angle: number, pivot: Vec3): Shape {
  const c = Math.cos(angle), sn = Math.sin(angle);
  const [, py, pz] = pivot;
  const corners: Vec3[] = [];
  for (const y of [s.min[1], s.max[1]]) for (const z of [s.min[2], s.max[2]]) {
    const dy = y - py, dz = z - pz;
    corners.push([0, py + c * dy - sn * dz, pz + sn * dy + c * dz]);
  }
  return {
    sdf: (x, y, z) => {
      const dy = y - py, dz = z - pz;
      return s.sdf(x, py + c * dy + sn * dz, pz - sn * dy + c * dz);
    },
    min: [s.min[0], Math.min(...corners.map((k) => k[1])), Math.min(...corners.map((k) => k[2]))],
    max: [s.max[0], Math.max(...corners.map((k) => k[1])), Math.max(...corners.map((k) => k[2]))],
    slack: s.slack,
  };
}

/** Rotate around the z axis through `pivot`. */
export function rotateZ(s: Shape, angle: number, pivot: Vec3): Shape {
  const c = Math.cos(angle), sn = Math.sin(angle);
  const [px, py] = pivot;
  const corners: Vec3[] = [];
  for (const x of [s.min[0], s.max[0]]) for (const y of [s.min[1], s.max[1]]) {
    const dx = x - px, dy = y - py;
    corners.push([px + c * dx - sn * dy, py + sn * dx + c * dy, 0]);
  }
  return {
    sdf: (x, y, z) => {
      const dx = x - px, dy = y - py;
      return s.sdf(px + c * dx + sn * dy, py - sn * dx + c * dy, z);
    },
    min: [Math.min(...corners.map((k) => k[0])), Math.min(...corners.map((k) => k[1])), s.min[2]],
    max: [Math.max(...corners.map((k) => k[0])), Math.max(...corners.map((k) => k[1])), s.max[2]],
    slack: s.slack,
  };
}

// ---------------------------------------------------------------------------
// Rasteriser

export enum Op {
  Add,
  Carve,
  /** Recolour voxels that are already solid. */
  Paint,
}

const scratch = new Uint16Array(BRICK_VOXELS);

export function apply(world: VoxelWorld, shape: Shape, op: Op, paint: Paint = 0): void {
  const vs = VOXEL_SIZE;
  const slack = shape.slack ?? 1;
  const constPaint = typeof paint === 'number';
  const paintFn = typeof paint === 'function' ? paint : null;
  const pv = typeof paint === 'number' ? paint : 0;

  const vx0 = Math.max(0, Math.floor(shape.min[0] / vs - 0.5));
  const vy0 = Math.max(0, Math.floor(shape.min[1] / vs - 0.5));
  const vz0 = Math.max(0, Math.floor(shape.min[2] / vs - 0.5));
  const vx1 = Math.min(world.dx - 1, Math.ceil(shape.max[0] / vs + 0.5));
  const vy1 = Math.min(world.dy - 1, Math.ceil(shape.max[1] / vs + 0.5));
  const vz1 = Math.min(world.dz - 1, Math.ceil(shape.max[2] / vs + 0.5));
  if (vx0 > vx1 || vy0 > vy1 || vz0 > vz1) return;

  const brickHalfDiag = 4 * Math.sqrt(3) * vs * slack + vs * 0.01;
  const subHalfDiag = 2 * Math.sqrt(3) * vs * slack + vs * 0.01;
  const sdf = shape.sdf;

  for (let bz = vz0 >> 3; bz <= vz1 >> 3; bz++)
    for (let by = vy0 >> 3; by <= vy1 >> 3; by++)
      for (let bx = vx0 >> 3; bx <= vx1 >> 3; bx++) {
        const ci = world.cellIndex(bx, by, bz);
        const ox = bx * BRICK, oy = by * BRICK, oz = bz * BRICK;
        let full = false;
        if (shape.contains) {
          full = shape.contains((ox + 0.5) * vs, (oy + 0.5) * vs, (oz + 0.5) * vs, (ox + 7.5) * vs, (oy + 7.5) * vs, (oz + 7.5) * vs);
        }
        if (!full) {
          const d = sdf((ox + 4) * vs, (oy + 4) * vs, (oz + 4) * vs);
          if (d > brickHalfDiag) continue;
          full = d < -brickHalfDiag;
        }
        const g = world.grid[ci];
        if (full) {
          if (op === Op.Add && constPaint) { world.setBrickUniform(ci, pv); continue; }
          if (op === Op.Carve) { world.setBrickUniform(ci, 0); continue; }
          if (op === Op.Paint && g === 0) continue;
          if (op === Op.Paint && constPaint && g & UNIFORM) { world.setBrickUniform(ci, pv); continue; }
        }
        if (op === Op.Paint && g === 0) continue;

        // Load current brick contents.
        if (g === 0) scratch.fill(0);
        else if (g & UNIFORM) scratch.fill(g & 0xffff);
        else scratch.set(world.pool.subarray((g - 1) * BRICK_VOXELS, g * BRICK_VOXELS));

        let changed = false;
        for (let s = 0; s < 8; s++) {
          const sx = ox + (s & 1) * 4, sy = oy + ((s >> 1) & 1) * 4, sz = oz + ((s >> 2) & 1) * 4;
          let subFull = full;
          if (!full) {
            const d = sdf((sx + 2) * vs, (sy + 2) * vs, (sz + 2) * vs);
            if (d > subHalfDiag) continue;
            subFull = d < -subHalfDiag;
          }
          for (let z = sz; z < sz + 4; z++)
            for (let y = sy; y < sy + 4; y++)
              for (let x = sx; x < sx + 4; x++) {
                const px = (x + 0.5) * vs, py = (y + 0.5) * vs, pz = (z + 0.5) * vs;
                if (!subFull && sdf(px, py, pz) > 0) continue;
                const li = (x & 7) | ((y & 7) << 3) | ((z & 7) << 6);
                const cur = scratch[li];
                let next: number;
                if (op === Op.Carve) next = 0;
                else if (op === Op.Paint && cur === 0) continue;
                else next = paintFn ? paintFn(px, py, pz) : pv;
                if (next !== cur) {
                  scratch[li] = next;
                  changed = true;
                }
              }
        }
        if (!changed) continue;
        const p = world.editableBrick(ci);
        world.pool.set(scratch, p * BRICK_VOXELS);
        world.dirtyBricks.add(p);
        world.dirtyCells.add(ci);
        world.compactCell(ci);
      }
}

/** Convenience wrapper that collects scene operations. */
export class Builder {
  constructor(readonly world: VoxelWorld) {}
  add(shape: Shape, paint: Paint): this {
    apply(this.world, shape, Op.Add, paint);
    return this;
  }
  carve(shape: Shape): this {
    apply(this.world, shape, Op.Carve);
    return this;
  }
  paint(shape: Shape, paint: Paint): this {
    apply(this.world, shape, Op.Paint, paint);
    return this;
  }
}
