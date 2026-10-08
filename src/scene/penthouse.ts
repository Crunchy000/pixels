// A Las Vegas penthouse suite, modelled from SDF primitives and rasterised
// into the voxel world. Coordinates are metres: x = east, y = up, z = north.
// The north and east walls are floor-to-ceiling glass looking out over the Strip.

import { BRICK, VOXELS_PER_METER, VOXEL_SIZE as VS } from '../config';
import { M, vox } from '../world/materials';
import {
  Builder,
  Shape,
  Vec3,
  box,
  boxMinMax,
  capsule,
  cylinder,
  ellipsoid,
  intersect,
  lathe,
  prism,
  rotateX,
  rotateY,
  rotateZ,
  roundBox,
  sphere,
  subtract,
  torus,
} from '../world/shapes';
import type { Paint } from '../world/shapes';
import { VoxelWorld } from '../world/VoxelWorld';
import { layoutText } from './font';

export interface SceneLight {
  pos: Vec3;
  radius: number;
  color: Vec3;
  shadow: boolean;
  /** Emissive voxels (metres) powering this light; it dims as they are destroyed. */
  anchors: Vec3[];
}

export interface Scene {
  lights: SceneLight[];
  spawn: { pos: Vec3; yaw: number; pitch: number };
}

// Big architectural planes sit on brick boundaries, so floor, ceiling and wall
// slabs compress to uniform bricks at any voxel density.
const snap = (m: number) => (Math.round((m * VOXELS_PER_METER) / BRICK) * BRICK) / VOXELS_PER_METER;
const BRICK_M = BRICK / VOXELS_PER_METER;
const FLOOR = snap(0.2);
const CEIL = snap(4.1);
const WALL = snap(0.25);
const RW = 16; // room width (x)
const RD = 12; // room depth (z)

function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const tick = () => new Promise<void>((r) => setTimeout(r, 0));

/** Stadium (rounded rectangle) slab: segment along x of half-length hl, radius r. */
function stadium(c: Vec3, hl: number, r: number, y0: number, y1: number): Shape {
  const [cx, , cz] = c;
  const ym = (y0 + y1) / 2, hy = (y1 - y0) / 2;
  return {
    sdf: (x, y, z) => {
      const dx = Math.max(Math.abs(x - cx) - hl, 0);
      const d2 = Math.hypot(dx, z - cz) - r;
      const dy = Math.abs(y - ym) - hy;
      return Math.min(Math.max(d2, dy), 0) + Math.hypot(Math.max(d2, 0), Math.max(dy, 0));
    },
    min: [cx - hl - r, y0, cz - r],
    max: [cx + hl + r, y1, cz + r],
  };
}

/** Wrap a paint function so it sees coordinates in the frame of a shape rotated by rotateY(angle, pivot). */
function localPaint(fn: (x: number, y: number, z: number) => number, angle: number, pivot: Vec3): Paint {
  const c = Math.cos(angle), sn = Math.sin(angle);
  const [px, , pz] = pivot;
  return (x, y, z) => {
    const dx = x - px, dz = z - pz;
    return fn(px + c * dx + sn * dz, y, pz - sn * dx + c * dz);
  };
}

export async function buildPenthouse(world: VoxelWorld, progress: (msg: string, f: number) => void): Promise<Scene> {
  const B = new Builder(world);
  const lights: SceneLight[] = [];
  const R = rng(1337);
  const steps = 16;
  let step = 0;
  const stage = async (msg: string) => {
    progress(msg, step++ / steps);
    await tick();
  };

  // -------------------------------------------------------------------------
  await stage('Pouring the floor slab');
  shell(B);

  await stage('Glazing the Strip view');
  windows(B);

  await stage('Rolling out the casino carpet');
  B.add(boxMinMax(4.3, FLOOR - VS * 0.5, 6.3, 11.6, FLOOR + 2 * VS, 11.3), vox(M.carpet));
  column(B, 4.1, 5.6);
  column(B, 12.1, 5.9);

  await stage('Upholstering the sectional');
  sofa(B);

  await stage('Setting the coffee table');
  coffeeTable(B, 7.9, 8.85, R);

  await stage('Hanging the chandelier');
  lights.push(chandelier(B, 7.9, 8.85, R));
  lights.push({ pos: [7.9, CEIL + 0.05, 8.85], radius: 4, color: [1.0, 0.7, 0.4], shadow: false, anchors: [] });

  await stage('Plugging in the lamps');
  lights.push(floorLamp(B, 10.85, 6.95));
  lights.push(floorLamp(B, 5.25, 10.55));

  await stage('Stocking the bar');
  lights.push(...bar(B, R));

  await stage('Shuffling the cards');
  lights.push(...pokerTable(B, 11.4, 3.5, R));

  await stage('Loading the slots');
  for (let i = 0; i < 4; i++) lights.push(slotMachine(B, 12.75 + i * 0.78, 0.25, i, R));

  await stage('Tuning the piano');
  piano(B, 13.45, 8.7, -0.55);

  await stage('Building the champagne tower');
  champagneTower(B, 10.75, 10.45);

  await stage('Mounting the TV');
  lights.push(tv(B));

  await stage('Lighting the neon');
  lights.push(...neonSigns(B));

  await stage('Placing priceless art');
  decor(B, R);

  await stage('Watering the palms');
  palm(B, 0.85, 11.05, R);
  palm(B, 15.1, 5.0, R);
  palm(B, 0.85, 0.85, R);

  // Ceiling downlights (no fixtures: hidden in the coffers).
  lights.push({ pos: [3.0, CEIL - 0.2, 10.0], radius: 5, color: [0.9, 0.7, 0.5], shadow: false, anchors: [] });
  lights.push({ pos: [6.0, CEIL - 0.2, 2.5], radius: 5, color: [0.9, 0.7, 0.5], shadow: false, anchors: [] });
  lights.push({ pos: [14.0, CEIL - 0.2, 11.0], radius: 4.5, color: [0.8, 0.65, 0.5], shadow: false, anchors: [] });

  progress('Done', 1);
  return {
    lights,
    spawn: { pos: [7.6, FLOOR, 4.6], yaw: Math.PI * 0.82, pitch: -0.08 },
  };
}

// ---------------------------------------------------------------------------

function shell(B: Builder) {
  // Floor slab (structural marble tiles).
  B.add(boxMinMax(0, 0, 0, RW, FLOOR, RD), vox(M.floor));
  // Ceiling slab with a recessed coffer over the lounge.
  B.add(boxMinMax(0, CEIL, 0, RW, 4.4, RD), vox(M.ceiling));
  B.carve(cylinder([7.9, CEIL + 0.07, 8.85], 2.4, 0.08));
  B.add(torus([7.9, CEIL + 0.005, 8.85], 2.42, 0.03), vox(M.gold));
  B.add(torus([7.9, CEIL + 0.11, 8.85], 2.32, 0.012), vox(M.lampshade));
  // Gold trim around the room's ceiling line.
  B.add(boxMinMax(WALL, CEIL - 0.06, WALL, RW - WALL, CEIL, WALL + 0.03), vox(M.gold));
  B.add(boxMinMax(WALL, CEIL - 0.06, WALL, WALL + 0.03, CEIL, RD - WALL), vox(M.gold));

  // West + south walls: concrete core, plaster skin.
  B.add(boxMinMax(0, FLOOR, 0, WALL, CEIL, RD), vox(M.concrete));
  B.add(boxMinMax(0, FLOOR, 0, RW, CEIL, WALL), vox(M.concrete));
  B.add(boxMinMax(WALL - BRICK_M, FLOOR, WALL, WALL, CEIL, RD), vox(M.wall));
  B.add(boxMinMax(WALL, FLOOR, WALL - BRICK_M, RW, CEIL, WALL), vox(M.wall));
  // Dark accent wall behind the bar.
  B.add(boxMinMax(WALL - BRICK_M, FLOOR, snap(2.1), WALL, CEIL, snap(7.9)), vox(M.wallDark));
  // Baseboards.
  B.add(boxMinMax(WALL, FLOOR, WALL, WALL + 0.02, FLOOR + 0.12, RD - WALL), vox(M.marbleBlack));
  B.add(boxMinMax(WALL, FLOOR, WALL, RW - WALL, FLOOR + 0.12, WALL + 0.02), vox(M.marbleBlack));

  // Entrance double door in the south wall.
  B.add(boxMinMax(1.2, FLOOR, WALL - 0.02, 3.0, FLOOR + 2.55, WALL + 0.04), vox(M.marbleBlack));
  B.add(boxMinMax(1.3, FLOOR, WALL, 2.9, FLOOR + 2.45, WALL + 0.05), vox(M.walnutY));
  B.carve(boxMinMax(2.095, FLOOR, WALL + 0.035, 2.105, FLOOR + 2.45, WALL + 0.06));
  for (const x of [1.98, 2.22]) B.add(capsule([x, FLOOR + 0.9, WALL + 0.09], [x, FLOOR + 1.3, WALL + 0.09], 0.012), vox(M.gold));
  for (const x of [1.98, 2.22]) {
    B.add(capsule([x, FLOOR + 0.95, WALL + 0.04], [x, FLOOR + 0.95, WALL + 0.09], 0.008), vox(M.gold));
    B.add(capsule([x, FLOOR + 1.25, WALL + 0.04], [x, FLOOR + 1.25, WALL + 0.09], 0.008), vox(M.gold));
  }
}

function windows(B: Builder) {
  const g = 2 * VS;
  // North glazing.
  B.add(boxMinMax(WALL, FLOOR, 11.8, RW - WALL, CEIL, 11.8 + g), vox(M.windowGlass));
  // East glazing.
  B.add(boxMinMax(15.8 - g, FLOOR, WALL, 15.8, CEIL, 11.8 + g), vox(M.windowGlass));
  const frame = vox(M.frame);
  // Sills and headers.
  B.add(boxMinMax(WALL, FLOOR, 11.72, RW - WALL + 0.2, FLOOR + 0.06, 11.92), frame);
  B.add(boxMinMax(WALL, CEIL - 0.14, 11.72, RW - WALL + 0.2, CEIL, 11.92), frame);
  B.add(boxMinMax(15.65, FLOOR, WALL, 15.88, FLOOR + 0.06, 11.92), frame);
  B.add(boxMinMax(15.65, CEIL - 0.14, WALL, 15.88, CEIL, 11.92), frame);
  // Mullions.
  const n = 8;
  for (let i = 0; i <= n; i++) {
    const x = WALL + ((RW - 2 * WALL) * i) / n;
    B.add(boxMinMax(x - 0.04, FLOOR, 11.72, x + 0.04, CEIL, 11.92), frame);
  }
  const m = 6;
  for (let i = 0; i <= m; i++) {
    const z = WALL + ((11.8 - WALL) * i) / m;
    B.add(boxMinMax(15.65, FLOOR, z - 0.04, 15.88, CEIL, z + 0.04), frame);
  }
  // Corner post.
  B.add(boxMinMax(15.62, FLOOR, 11.7, 15.9, CEIL, 11.95), frame);
}

function column(B: Builder, x: number, z: number) {
  B.add(box([x, FLOOR + 0.08, z], [0.32, 0.08, 0.32]), vox(M.marbleBlack));
  B.add(cylinder([x, (FLOOR + CEIL) / 2, z], 0.24, (CEIL - FLOOR) / 2 - 0.1), vox(M.marble));
  B.add(torus([x, FLOOR + 0.2, z], 0.25, 0.035), vox(M.gold));
  B.add(torus([x, CEIL - 0.3, z], 0.25, 0.03), vox(M.gold));
  B.add(lathe([x, CEIL - 0.28, z], [0.24, 0, 0.3, 0.12, 0.34, 0.18], 0), vox(M.gold));
  B.add(box([x, CEIL - 0.05, z], [0.36, 0.05, 0.36]), vox(M.marbleBlack));
}

// ---------------------------------------------------------------------------
// Lounge

function upholstered(B: Builder, s: Shape, cover: number, inset: number, inner = M.foam) {
  B.add(s, vox(cover));
  // Foam core so blasting a cushion exposes stuffing.
  const shrunk: Shape = { sdf: (x, y, z) => s.sdf(x, y, z) + inset, min: s.min, max: s.max, slack: s.slack };
  B.add(shrunk, vox(inner));
}

function sofa(B: Builder) {
  const y0 = FLOOR + 2 * VS;
  const legH = 0.07;
  const v = M.velvet;
  // Long section along x (back to the south, facing the windows).
  const lx0 = 4.75, lx1 = 10.45, lz0 = 6.45, lz1 = 7.4;
  // Short section along z (back to the west).
  const sx0 = 4.75, sx1 = 5.7, sz1 = 10.1;
  for (const [x, z] of [[lx0 + 0.06, lz0 + 0.06], [lx1 - 0.06, lz0 + 0.06], [lx1 - 0.06, lz1 - 0.06], [sx0 + 0.06, sz1 - 0.06], [sx1 - 0.06, sz1 - 0.06], [sx1 - 0.06, lz1 + 0.3]]) {
    B.add(cylinder([x, y0 + legH / 2, z], 0.025, legH / 2 + VS), vox(M.gold));
  }
  const base = y0 + legH;
  upholstered(B, roundBox([(lx0 + lx1) / 2, base + 0.16, (lz0 + lz1) / 2], [(lx1 - lx0) / 2, 0.16, (lz1 - lz0) / 2], 0.03), v, 0.03);
  upholstered(B, roundBox([(sx0 + sx1) / 2, base + 0.16, (lz1 + sz1) / 2], [(sx1 - sx0) / 2, 0.16, (sz1 - lz1) / 2 + 0.02], 0.03), v, 0.03);
  // Backs.
  upholstered(B, roundBox([(lx0 + lx1) / 2, base + 0.42, lz0 + 0.12], [(lx1 - lx0) / 2, 0.42, 0.12], 0.06), v, 0.04);
  upholstered(B, roundBox([sx0 + 0.12, base + 0.42, (lz0 + sz1) / 2], [0.12, 0.42, (sz1 - lz0) / 2], 0.06), v, 0.04);
  // Arms.
  upholstered(B, roundBox([lx1 - 0.11, base + 0.3, (lz0 + lz1) / 2], [0.11, 0.3, (lz1 - lz0) / 2], 0.05), v, 0.04);
  upholstered(B, roundBox([(sx0 + sx1) / 2, base + 0.3, sz1 - 0.11], [(sx1 - sx0) / 2, 0.3, 0.11], 0.05), v, 0.04);
  // Seat cushions.
  const seatY = base + 0.32 + 0.07;
  const n = 4;
  const cx0 = lx0 + 0.24, cx1 = lx1 - 0.22;
  for (let i = 0; i < n; i++) {
    const a = cx0 + ((cx1 - cx0) * i) / n + 0.008, b = cx0 + ((cx1 - cx0) * (i + 1)) / n - 0.008;
    upholstered(B, roundBox([(a + b) / 2, seatY, (lz0 + 0.24 + lz1) / 2], [(b - a) / 2, 0.075, (lz1 - lz0 - 0.24) / 2], 0.05), v, 0.035);
  }
  for (let i = 0; i < 2; i++) {
    const a = lz1 + (i * (sz1 - 0.22 - lz1)) / 2 + 0.008, b = lz1 + ((i + 1) * (sz1 - 0.22 - lz1)) / 2 - 0.008;
    upholstered(B, roundBox([(sx0 + 0.24 + sx1) / 2, seatY, (a + b) / 2], [(sx1 - sx0 - 0.24) / 2, 0.075, (b - a) / 2], 0.05), v, 0.035);
  }
  // Corner seat.
  upholstered(B, roundBox([(sx0 + 0.24 + sx1) / 2, seatY, (lz0 + 0.24 + lz1) / 2], [(sx1 - sx0 - 0.24) / 2 - 0.008, 0.075, (lz1 - lz0 - 0.24) / 2], 0.05), v, 0.035);
  // Throw pillows.
  const pil = (c: Vec3, yaw: number, mat: number) =>
    upholstered(B, rotateY(rotateX(ellipsoid(c, [0.2, 0.18, 0.07]), -0.35, c), yaw, c), mat, 0.03, M.foam);
  pil([6.2, seatY + 0.22, lz0 + 0.33], 0, M.velvetGold);
  pil([8.9, seatY + 0.22, lz0 + 0.33], 0.15, M.velvetRed);
  pil([sx0 + 0.33, seatY + 0.22, 9.2], Math.PI / 2, M.velvetGold);
}

function coupe(B: Builder, x: number, y: number, z: number, fill: boolean) {
  const t = 1.4 * VS;
  B.add(lathe([x, y, z], [0, t / 2, 0.034, t / 2], t), vox(M.glassClear));
  B.add(cylinder([x, y + 0.045, z], Math.max(0.006, VS * 0.6), 0.04), vox(M.glassClear));
  B.add(lathe([x, y, z], [0.006, 0.085, 0.03, 0.09, 0.048, 0.105, 0.056, 0.125], t), vox(M.glassClear));
  if (fill) B.add(intersect(lathe([x, y, z], [0.006, 0.088, 0.028, 0.093, 0.045, 0.106, 0.05, 0.116], 0), box([x, y + 0.1, z], [0.06, 0.016, 0.06])), vox(M.champagne));
}

function bottle(B: Builder, x: number, y: number, z: number, R: () => number, kind?: number) {
  const k = kind ?? Math.floor(R() * 4);
  const t = 1.5 * VS;
  const glasses = [M.glassGreen, M.glassAmber, M.glassSmoke, M.glassBlue, M.glassRed, M.glassClear];
  const liquids = [M.wine, M.whiskey, M.absinthe, M.champagne, M.water];
  const g = glasses[Math.floor(R() * glasses.length)];
  const liq = liquids[Math.floor(R() * liquids.length)];
  const fill = 0.5 + R() * 0.45;
  const c: Vec3 = [x, y, z];
  if (k === 0) {
    // Wine / spirits.
    const r = 0.034;
    B.add(lathe(c, [0, t / 2, r, t / 2, r, 0.2, 0.016, 0.25, 0.013, 0.31], t), vox(g));
    B.add(intersect(cylinder([x, y + 0.11, z], r - t, 0.1), box([x, y + t + 0.1 * fill, z], [r, 0.1 * fill, r])), vox(liq));
    B.paint(subtract(cylinder([x, y + 0.11, z], r + VS, 0.045), cylinder([x, y + 0.11, z], r - t * 0.7, 0.05)), vox(M.label, 1 + Math.floor(R() * 200)));
    B.add(cylinder([x, y + 0.305, z], 0.015, 0.012), vox(R() < 0.5 ? M.gold : M.cork));
  } else if (k === 1) {
    // Square whiskey decanter.
    const h = 0.17;
    B.add(subtract(roundBox([x, y + h / 2, z], [0.04, h / 2, 0.04], 0.01), roundBox([x, y + h / 2 + t, z], [0.04 - t, h / 2, 0.04 - t], 0.008)), vox(g));
    B.add(box([x, y + t + (h * fill) / 2, z], [0.04 - t - VS * 0.2, (h * fill) / 2, 0.04 - t - VS * 0.2]), vox(M.whiskey));
    B.add(cylinder([x, y + h + 0.02, z], 0.014, 0.025), vox(g));
    B.add(sphere([x, y + h + 0.06, z], 0.022), vox(M.glassClear));
  } else if (k === 2) {
    // Round liqueur flask.
    B.add(lathe(c, [0, t / 2, 0.03, t / 2, 0.052, 0.06, 0.05, 0.12, 0.02, 0.18, 0.012, 0.24], t), vox(g));
    B.add(intersect(lathe(c, [0, t, 0.026, t, 0.046, 0.06, 0.044, 0.12, 0.015, 0.18], 0), box([x, y + 0.1 * fill, z], [0.06, 0.1 * fill, 0.06])), vox(liq));
    B.add(cylinder([x, y + 0.245, z], 0.013, 0.012), vox(M.cork));
  } else {
    // Champagne magnum with gold foil.
    const r = 0.042;
    B.add(lathe(c, [0, t / 2, r, t / 2, r, 0.2, 0.02, 0.27, 0.016, 0.33], t), vox(M.glassGreen));
    B.add(intersect(cylinder([x, y + 0.11, z], r - t, 0.1), box([x, y + t + 0.1 * fill, z], [r, 0.1 * fill, r])), vox(M.champagne));
    B.add(lathe(c, [0.022, 0.26, 0.018, 0.3, 0.018, 0.34], 1.2 * VS), vox(M.gold));
    B.paint(subtract(cylinder([x, y + 0.1, z], r + VS, 0.04), cylinder([x, y + 0.1, z], r - t * 0.7, 0.05)), vox(M.label, 1));
  }
}

function dice(B: Builder, c: Vec3, s: number, yaw: number) {
  const h = s / 2;
  const die = roundBox(c, [h, h, h], s * 0.12);
  B.add(rotateY(die, yaw, c), vox(M.dieRed));
  const pr = s * 0.09;
  const off = s * 0.27;
  const pipsOn = (n: number, face: (u: number, v: number) => Vec3) => {
    const layouts: Record<number, [number, number][]> = {
      1: [[0, 0]],
      2: [[-1, -1], [1, 1]],
      3: [[-1, -1], [0, 0], [1, 1]],
      4: [[-1, -1], [1, 1], [-1, 1], [1, -1]],
      5: [[-1, -1], [1, 1], [-1, 1], [1, -1], [0, 0]],
      6: [[-1, -1], [1, 1], [-1, 1], [1, -1], [-1, 0], [1, 0]],
    };
    for (const [u, v] of layouts[n]) B.add(rotateY(sphere(face(u * off, v * off), pr), yaw, c), vox(M.pip));
  };
  const [x, y, z] = c;
  const e = h - pr * 0.35;
  pipsOn(5, (u, v) => [x + u, y + e, z + v]);
  pipsOn(3, (u, v) => [x + u, y + v, z + e]);
  pipsOn(4, (u, v) => [x + u, y + v, z - e]);
  pipsOn(2, (u, v) => [x + e, y + v, z + u]);
  pipsOn(6, (u, v) => [x - e, y + v, z + u]);
}

function chipStack(B: Builder, x: number, y: number, z: number, n: number, mat: number) {
  const h = n * 0.0045;
  const r = 0.022;
  B.add(cylinder([x, y + h / 2, z], r, h / 2), (px, py, pz) => {
    const a = Math.atan2(pz - z, px - x);
    const edge = Math.hypot(px - x, pz - z) > r - VS * 1.2;
    const stripe = Math.floor(((a + Math.PI) / (2 * Math.PI)) * 8) % 2 === 0;
    const layer = Math.floor((py - y) / 0.0045) % 2;
    return vox(mat, edge && stripe && layer === 0 ? 255 : 0);
  });
}

function coffeeTable(B: Builder, cx: number, cz: number, R: () => number) {
  const y0 = FLOOR + 2 * VS;
  const top = FLOOR + 0.44;
  const hx = 0.8, hz = 0.45;
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
    B.add(box([cx + sx * (hx - 0.03), (y0 + top) / 2, cz + sz * (hz - 0.03)], [0.02, (top - y0) / 2, 0.02]), vox(M.gold));
  }
  B.add(subtract(box([cx, top - 0.035, cz], [hx, 0.015, hz]), box([cx, top - 0.035, cz], [hx - 0.04, 0.02, hz - 0.04])), vox(M.gold));
  B.add(box([cx, top - 0.01, cz], [hx - 0.01, 0.012, hz - 0.01]), vox(M.glassTable));
  B.add(box([cx, y0 + 0.13, cz], [hx - 0.04, 0.015, hz - 0.04]), vox(M.marbleBlack));

  const ty = top + 0.002;
  // Ice bucket with a champagne bottle.
  const bx = cx - 0.42, bz = cz - 0.05;
  B.add(lathe([bx, ty, bz], [0, VS, 0.085, VS, 0.105, 0.2, 0.112, 0.21], 1.6 * VS), vox(M.chrome));
  for (let i = 0; i < 22; i++) {
    const a = R() * Math.PI * 2, r = R() * 0.06;
    B.add(roundBox([bx + Math.cos(a) * r, ty + 0.03 + R() * 0.13, bz + Math.sin(a) * r], [0.018, 0.016, 0.018], 0.005), vox(M.ice));
  }
  const bc: Vec3 = [bx, ty + 0.03, bz];
  const r = 0.04, t = 1.5 * VS;
  B.add(rotateZ(lathe(bc, [0, t / 2, r, t / 2, r, 0.2, 0.02, 0.27, 0.016, 0.33], t), -0.32, bc), vox(M.glassGreen));
  B.add(rotateZ(cylinder([bx, ty + 0.03 + 0.1, bz], r - t, 0.08), -0.32, bc), vox(M.champagne));
  B.add(rotateZ(lathe(bc, [0.022, 0.26, 0.018, 0.3, 0.018, 0.34], 1.2 * VS), -0.32, bc), vox(M.gold));

  coupe(B, cx - 0.12, ty, cz + 0.18, true);
  coupe(B, cx + 0.02, ty, cz + 0.24, true);
  dice(B, [cx + 0.25, ty + 0.06, cz - 0.12], 0.12, 0.3);
  dice(B, [cx + 0.43, ty + 0.06, cz + 0.02], 0.12, -0.6);
  chipStack(B, cx + 0.55, ty, cz + 0.28, 12, M.chipBlack);
  chipStack(B, cx + 0.6, ty, cz + 0.22, 8, M.chipPurple);
  chipStack(B, cx + 0.5, ty, cz + 0.2, 15, M.chipRed);
  // Bowl of something expensive.
  B.add(lathe([cx + 0.15, ty, cz + 0.22], [0, VS, 0.05, VS, 0.1, 0.06, 0.11, 0.075], 1.5 * VS), vox(M.crystal));
  for (let i = 0; i < 7; i++) {
    const a = R() * Math.PI * 2, rr = R() * 0.05;
    B.add(sphere([cx + 0.15 + Math.cos(a) * rr, ty + 0.05 + R() * 0.02, cz + 0.22 + Math.sin(a) * rr], 0.025), vox(R() < 0.5 ? M.porcelainRed : M.gold));
  }
}

function chandelier(B: Builder, cx: number, cz: number, R: () => number): SceneLight {
  const top = CEIL + 0.15;
  const bodyY = 3.05;
  const gold = vox(M.gold);
  // Canopy + rod.
  B.add(lathe([cx, top, cz], [0.14, 0, 0.1, -0.06, 0.02, -0.09], 0), gold);
  B.add(cylinder([cx, (top + bodyY + 0.3) / 2, cz], 0.012, (top - bodyY - 0.3) / 2), gold);
  for (let y = bodyY + 0.4; y < top - 0.1; y += 0.12) B.add(torus([cx, y, cz], 0.022, 0.007), gold);
  // Body.
  B.add(lathe([cx, bodyY, cz], [0, -0.12, 0.06, -0.1, 0.1, 0.0, 0.07, 0.12, 0.04, 0.25, 0.02, 0.32], 0), gold);
  B.add(sphere([cx, bodyY - 0.17, cz], 0.075), vox(M.crystal));
  // Rings + spokes.
  const rings = [
    { r: 0.72, y: bodyY - 0.02, drops: 40, arms: 12 },
    { r: 0.46, y: bodyY + 0.2, drops: 24, arms: 8 },
  ];
  const anchors: Vec3[] = [];
  for (const ring of rings) {
    B.add(torus([cx, ring.y, cz], ring.r, 0.016), gold);
    for (let i = 0; i < ring.arms; i++) {
      const a = (i / ring.arms) * Math.PI * 2;
      const ex = cx + Math.cos(a) * ring.r, ez = cz + Math.sin(a) * ring.r;
      B.add(capsule([cx + Math.cos(a) * 0.06, ring.y + 0.06, cz + Math.sin(a) * 0.06], [ex, ring.y, ez], 0.009), gold);
      // Candle cup + bulb.
      B.add(cylinder([ex, ring.y + 0.035, ez], 0.022, 0.025), gold);
      B.add(ellipsoid([ex, ring.y + 0.09, ez], [0.018, 0.035, 0.018]), vox(M.bulb));
      anchors.push([ex, ring.y + 0.09, ez]);
    }
    for (let i = 0; i < ring.drops; i++) {
      const a = ((i + 0.5) / ring.drops) * Math.PI * 2;
      const ex = cx + Math.cos(a) * ring.r, ez = cz + Math.sin(a) * ring.r;
      const len = 0.05 + R() * 0.08;
      B.add(capsule([ex, ring.y - 0.01, ez], [ex, ring.y - len, ez], VS * 0.55), vox(M.crystal));
      B.add(ellipsoid([ex, ring.y - len - 0.03, ez], [0.016, 0.035, 0.016]), vox(M.crystal));
    }
  }
  // Swags of crystal beads between the rings.
  for (let i = 0; i < 16; i++) {
    const a = (i / 16) * Math.PI * 2;
    let prev: Vec3 | null = null;
    for (let k = 0; k <= 6; k++) {
      const f = k / 6;
      const r = rings[1].r + (rings[0].r - rings[1].r) * f;
      const y = rings[1].y + (rings[0].y - rings[1].y) * f - Math.sin(f * Math.PI) * 0.08;
      const p: Vec3 = [cx + Math.cos(a) * r, y, cz + Math.sin(a) * r];
      B.add(sphere(p, 0.011), vox(M.crystal));
      if (prev) B.add(capsule(prev, p, VS * 0.55), vox(M.crystal));
      prev = p;
    }
  }
  return { pos: [cx, bodyY - 0.34, cz], radius: 11, color: [3.2, 2.2, 1.25], shadow: true, anchors };
}

function floorLamp(B: Builder, x: number, z: number): SceneLight {
  const y0 = FLOOR + 2 * VS;
  B.add(lathe([x, y0, z], [0.18, 0, 0.18, 0.025, 0.04, 0.05, 0.02, 0.06], 0), vox(M.gold));
  B.add(cylinder([x, y0 + 0.8, z], 0.014, 0.78), vox(M.gold));
  B.add(lathe([x, y0, z], [0.23, 1.42, 0.17, 1.75], 1.6 * VS), vox(M.lampshade));
  for (let i = 0; i < 3; i++) {
    const a = (i / 3) * Math.PI * 2;
    B.add(capsule([x, y0 + 1.6, z], [x + Math.cos(a) * 0.18, y0 + 1.72, z + Math.sin(a) * 0.18], 0.006), vox(M.gold));
  }
  B.add(ellipsoid([x, y0 + 1.6, z], [0.035, 0.05, 0.035]), vox(M.bulb));
  return { pos: [x, y0 + 1.5, z], radius: 5.5, color: [1.6, 1.1, 0.6], shadow: false, anchors: [[x, y0 + 1.6, z]] };
}

// ---------------------------------------------------------------------------
// Bar

function bar(B: Builder, R: () => number): SceneLight[] {
  const z0 = 2.4, z1 = 7.6;
  const lights: SceneLight[] = [];
  // Counter.
  B.add(boxMinMax(2.35, FLOOR, z0, 2.86, FLOOR + 1.02, z1), vox(M.walnutZ));
  B.add(boxMinMax(2.33, FLOOR, z0 - 0.02, 2.9, FLOOR + 0.12, z1 + 0.02), vox(M.marbleBlack));
  B.add(boxMinMax(2.86, FLOOR + 0.13, z0, 2.9, FLOOR + 0.94, z1), (_x, y, z) => {
    // Tufted leather front: darker buttons on a diamond grid.
    const u = (z - z0) / 0.16, v = (y - FLOOR) / 0.16;
    const du = u - Math.round(u), dv = v - Math.round(v);
    const odd = (Math.round(u) + Math.round(v)) % 2 === 0;
    return vox(M.leather, odd && Math.hypot(du, dv) < 0.12 ? 200 : 0);
  });
  B.add(boxMinMax(2.86, FLOOR + 0.94, z0, 2.92, FLOOR + 0.97, z1), vox(M.gold));
  B.add(boxMinMax(2.86, FLOOR + 0.12, z0, 2.92, FLOOR + 0.14, z1), vox(M.gold));
  B.add(boxMinMax(2.22, FLOOR + 1.02, z0 - 0.08, 3.0, FLOOR + 1.08, z1 + 0.08), vox(M.marbleBlack));
  B.add(capsule([2.97, FLOOR + 1.008, z0], [2.97, FLOOR + 1.008, z1], 0.008), vox(M.neonPink));
  // Foot rail.
  B.add(capsule([3.06, FLOOR + 0.2, z0 + 0.05], [3.06, FLOOR + 0.2, z1 - 0.05], 0.022), vox(M.gold));
  for (let z = z0 + 0.3; z < z1; z += 1.2) B.add(capsule([2.9, FLOOR + 0.2, z], [3.06, FLOOR + 0.2, z], 0.015), vox(M.gold));
  lights.push({ pos: [3.25, FLOOR + 0.75, 3.5], radius: 1.8, color: [1.6, 0.25, 0.9], shadow: false, anchors: [[2.97, FLOOR + 1.008, 3.5]] });
  lights.push({ pos: [3.25, FLOOR + 0.75, 6.4], radius: 1.8, color: [1.6, 0.25, 0.9], shadow: false, anchors: [[2.97, FLOOR + 1.008, 6.4]] });

  // Back bar: cabinet, mirror, glass shelves, bottles.
  B.add(boxMinMax(WALL, FLOOR, z0, 0.72, FLOOR + 0.9, z1), vox(M.walnutZ));
  B.add(boxMinMax(WALL, FLOOR + 0.9, z0 - 0.02, 0.76, FLOOR + 0.94, z1 + 0.02), vox(M.marbleBlack));
  B.add(boxMinMax(WALL, FLOOR + 0.94, z0 + 0.05, WALL + 0.02, FLOOR + 2.65, z1 - 0.05), vox(M.mirror));
  B.add(boxMinMax(WALL, FLOOR + 2.65, z0 + 0.03, WALL + 0.04, FLOOR + 2.69, z1 - 0.03), vox(M.gold));
  const shelves = [FLOOR + 1.36, FLOOR + 1.8, FLOOR + 2.24];
  for (const sy of shelves) {
    B.add(boxMinMax(WALL + 0.02, sy - 2 * VS, z0 + 0.1, 0.62, sy, z1 - 0.1), vox(M.glassTable));
    for (let z = z0 + 0.3; z < z1 - 0.1; z += 1.15) {
      B.add(capsule([WALL + 0.01, sy - 2.5 * VS, z], [0.58, sy - 2.5 * VS, z], VS * 0.8), vox(M.gold));
    }
    B.add(capsule([WALL + 0.035, sy + VS, z0 + 0.12], [WALL + 0.035, sy + VS, z1 - 0.12], VS * 0.7), vox(M.neonCyan));
  }
  lights.push({ pos: [0.75, FLOOR + 1.85, 3.6], radius: 2.6, color: [0.2, 1.1, 1.3], shadow: false, anchors: [[WALL + 0.035, shelves[1] + VS, 3.6]] });
  lights.push({ pos: [0.75, FLOOR + 1.85, 6.4], radius: 2.6, color: [0.2, 1.1, 1.3], shadow: false, anchors: [[WALL + 0.035, shelves[1] + VS, 6.4]] });
  for (const sy of [FLOOR + 0.94, ...shelves]) {
    let z = z0 + 0.18;
    while (z < z1 - 0.15) {
      bottle(B, 0.45 + (R() - 0.5) * 0.08, sy, z, R);
      z += 0.1 + R() * 0.06;
    }
  }
  // Stuff on the counter.
  for (let i = 0; i < 6; i++) {
    const z = z0 + 0.4 + i * 0.85 + R() * 0.2;
    if (i % 2 === 0) coupe(B, 2.62, FLOOR + 1.08, z, true);
    else bottle(B, 2.55, FLOOR + 1.08, z, R);
  }
  // Rocks glasses with whiskey.
  for (const z of [3.1, 5.15, 6.9]) {
    B.add(lathe([2.7, FLOOR + 1.08, z], [0, VS, 0.035, VS, 0.037, 0.085], 1.4 * VS), vox(M.glassClear));
    B.add(cylinder([2.7, FLOOR + 1.08 + 0.03, z], 0.03, 0.02), vox(M.whiskey));
    B.add(roundBox([2.7, FLOOR + 1.08 + 0.05, z], [0.015, 0.015, 0.015], 0.004), vox(M.ice));
  }
  // Cocktail shaker.
  B.add(lathe([2.5, FLOOR + 1.08, 4.0], [0, 0, 0.04, 0, 0.045, 0.14, 0.035, 0.18, 0.02, 0.22, 0.02, 0.24], 0), vox(M.chrome));

  // Stools.
  for (const z of [3.25, 4.45, 5.65, 6.85]) {
    const x = 3.5;
    B.add(lathe([x, FLOOR, z], [0.2, 0, 0.2, 0.02, 0.05, 0.04, 0.03, 0.05], 0), vox(M.chrome));
    B.add(cylinder([x, FLOOR + 0.4, z], 0.024, 0.36), vox(M.chrome));
    B.add(torus([x, FLOOR + 0.32, z], 0.17, 0.012), vox(M.chrome));
    for (let i = 0; i < 4; i++) {
      const a = (i / 4) * Math.PI * 2 + 0.4;
      B.add(capsule([x, FLOOR + 0.32, z], [x + Math.cos(a) * 0.17, FLOOR + 0.32, z + Math.sin(a) * 0.17], 0.009), vox(M.chrome));
    }
    B.add(cylinder([x, FLOOR + 0.77, z], 0.075, 0.015), vox(M.chrome));
    upholstered(B, roundBox([x, FLOOR + 0.82, z], [0.2, 0.045, 0.2], 0.04), M.leather, 0.025);
    B.add(torus([x, FLOOR + 0.8, z], 0.19, 0.012), vox(M.gold));
  }
  return lights;
}

// ---------------------------------------------------------------------------
// Gaming

function pokerTable(B: Builder, cx: number, cz: number, R: () => number): SceneLight[] {
  const hl = 0.62, rr = 0.68;
  const top = FLOOR + 0.76;
  for (const sx of [-1, 1]) {
    B.add(lathe([cx + sx * 0.55, FLOOR, cz], [0.42, 0, 0.42, 0.04, 0.16, 0.1, 0.12, 0.2, 0.12, 0.6], 0), vox(M.walnutY));
  }
  B.add(stadium([cx, 0, cz], hl, rr - 0.04, top - 0.16, top - 0.02), vox(M.walnut));
  B.add(stadium([cx, 0, cz], hl, rr - 0.17, top - 0.04, top), vox(M.felt));
  B.add(subtract(stadium([cx, 0, cz], hl, rr, top - 0.04, top + 0.06), stadium([cx, 0, cz], hl, rr - 0.17, top - 0.1, top + 0.1)), vox(M.leather));
  B.add(subtract(stadium([cx, 0, cz], hl, rr - 0.165, top - 0.01, top + 0.012), stadium([cx, 0, cz], hl, rr - 0.18, top - 0.1, top + 0.1)), vox(M.gold));
  // Dealer tray.
  B.add(box([cx, top + 0.012, cz - rr + 0.27], [0.25, 0.012, 0.06]), vox(M.chrome));
  const chipMats = [M.chipRed, M.chipBlue, M.chipGreen, M.chipBlack, M.chipPurple];
  for (let i = 0; i < 10; i++) chipStack(B, cx - 0.2 + i * 0.045, top + 0.006, cz - rr + 0.27, 3 + Math.floor(R() * 4), chipMats[i % 5]);

  // Seats around the far side + ends.
  const seats: { x: number; z: number; yaw: number }[] = [];
  for (const sx of [-1, 0, 1]) seats.push({ x: cx + sx * 0.6, z: cz + rr + 0.38, yaw: Math.PI });
  seats.push({ x: cx - hl - rr - 0.38, z: cz, yaw: -Math.PI / 2 });
  seats.push({ x: cx + hl + rr + 0.38, z: cz, yaw: Math.PI / 2 });
  for (const s of seats) {
    // Cards + chips in front of each seat (towards the table centre).
    const dx = cx - s.x, dz = cz - s.z;
    const dl = Math.hypot(dx, dz);
    const px = s.x + (dx / dl) * 0.62, pz = s.z + (dz / dl) * 0.62;
    for (let c = 0; c < 2; c++) {
      const ang = s.yaw + c * 0.25;
      const card = rotateY(box([px + c * 0.03, top + VS * 0.5, pz], [0.032, VS * 0.5, 0.045]), ang, [px, top, pz]);
      B.add(card, localPaint((x, _y, z) => {
        const edge = Math.abs(x - px - c * 0.03) > 0.022 || Math.abs(z - pz) > 0.034;
        return vox(M.card, edge ? 0 : 255);
      }, ang, [px, top, pz]));
    }
    for (let k = 0; k < 3; k++) {
      const a = Math.atan2(dz, dx) + Math.PI / 2;
      chipStack(B, px + Math.cos(a) * (0.12 + k * 0.05), top, pz + Math.sin(a) * (0.12 + k * 0.05), 4 + Math.floor(R() * 10), chipMats[Math.floor(R() * 5)]);
    }
    chair(B, s.x, s.z, s.yaw);
  }

  // Pendant lamp over the table.
  const ly = 2.15;
  B.add(cylinder([cx, (ly + CEIL) / 2 + 0.1, cz], 0.006, (CEIL - ly) / 2 - 0.08), vox(M.steelDark));
  B.add(subtract(stadium([cx, 0, cz], 0.5, 0.2, ly, ly + 0.16), stadium([cx, 0, cz], 0.5, 0.2 - 1.6 * VS, ly - 0.01, ly + 0.16 - 1.6 * VS)), vox(M.glassGreen));
  B.add(stadium([cx, 0, cz], 0.5, 0.21, ly + 0.14, ly + 0.18), vox(M.gold));
  const anchors: Vec3[] = [];
  for (const sx of [-0.4, 0, 0.4]) {
    B.add(sphere([cx + sx, ly + 0.08, cz], 0.035), vox(M.bulb));
    anchors.push([cx + sx, ly + 0.08, cz]);
  }
  return [{ pos: [cx, ly - 0.02, cz], radius: 6, color: [2.2, 1.9, 1.3], shadow: true, anchors }];
}

function chair(B: Builder, x: number, z: number, yaw: number) {
  const c: Vec3 = [x, 0, z];
  const rot = (s: Shape) => rotateY(s, yaw, c);
  for (const [sx, sz] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
    B.add(rot(cylinder([x + sx * 0.19, FLOOR + 0.22, z + sz * 0.19], 0.018, 0.22)), vox(M.gold));
  }
  upholstered(B, rot(roundBox([x, FLOOR + 0.48, z], [0.24, 0.05, 0.24], 0.04)), M.velvetRed, 0.025);
  // Back faces away from the table (local -z after the yaw).
  upholstered(B, rot(roundBox([x, FLOOR + 0.8, z - 0.21], [0.23, 0.3, 0.04], 0.035)), M.velvetRed, 0.02);
  B.add(rot(torus([x, FLOOR + 1.1, z - 0.21], 0.2, 0.012, 'z')), vox(M.gold));
}

function slotMachine(B: Builder, x: number, zBack: number, i: number, R: () => number): SceneLight {
  const z0 = zBack + 0.02, z1 = zBack + 0.62;
  const zc = (z0 + z1) / 2;
  const w = 0.32;
  const cab = vox(M.slotRed, 1 + ((i * 37) % 60));
  B.add(roundBox([x, FLOOR + 0.66, zc], [w, 0.66, (z1 - z0) / 2], 0.03), cab);
  B.add(cylinder([x, FLOOR + 1.32, zc], w, (z1 - z0) / 2 - 0.02, 'z'), cab);
  B.add(boxMinMax(x - w - 0.01, FLOOR, z0, x + w + 0.01, FLOOR + 0.1, z1 + 0.01), vox(M.chrome));
  // Screen recess.
  B.carve(boxMinMax(x - 0.24, FLOOR + 0.78, z1 - 0.05, x + 0.24, FLOOR + 1.16, z1 + 0.01));
  B.add(boxMinMax(x - 0.24, FLOOR + 0.78, z1 - 0.05, x + 0.24, FLOOR + 1.16, z1 - 0.05 + 2 * VS), vox(M.screen));
  B.add(subtract(boxMinMax(x - 0.27, FLOOR + 0.75, z1 - 0.01, x + 0.27, FLOOR + 1.19, z1 + 0.02), boxMinMax(x - 0.24, FLOOR + 0.78, z1 - 0.05, x + 0.24, FLOOR + 1.16, z1 + 0.05)), vox(M.chrome));
  // Button deck.
  const deck = rotateX(box([x, FLOOR + 0.66, z1 + 0.06], [w - 0.02, 0.025, 0.08]), -0.25, [x, FLOOR + 0.66, z1]);
  B.add(deck, vox(M.chrome));
  for (let b = 0; b < 4; b++) B.add(cylinder([x - 0.18 + b * 0.12, FLOOR + 0.7, z1 + 0.07], 0.022, 0.012), vox(b % 2 ? M.neonCyan : M.neonPink));
  // Coin tray.
  B.add(subtract(box([x, FLOOR + 0.26, z1 + 0.04], [0.2, 0.04, 0.05]), box([x, FLOOR + 0.29, z1 + 0.04], [0.18, 0.04, 0.04])), vox(M.chrome));
  // Topper sign: 777.
  neonText(B, '777', [x, FLOOR + 1.33, z1 - 0.01], 0.13, 'south', M.neonGold, 0.008);
  B.add(torus([x, FLOOR + 1.32, z1 - 0.03], w - 0.02, 0.012, 'z'), vox(M.neonPink));
  // Lever.
  B.add(box([x + w + 0.03, FLOOR + 0.9, zc + 0.1], [0.03, 0.06, 0.06]), vox(M.chrome));
  B.add(capsule([x + w + 0.07, FLOOR + 0.9, zc + 0.1], [x + w + 0.07, FLOOR + 1.3, zc + 0.12], 0.012), vox(M.chrome));
  B.add(sphere([x + w + 0.07, FLOOR + 1.33, zc + 0.12], 0.035), vox(M.glassRed));
  // Stool.
  const sz = z1 + 0.42;
  B.add(lathe([x, FLOOR, sz], [0.17, 0, 0.17, 0.02, 0.03, 0.04], 0), vox(M.chrome));
  B.add(cylinder([x, FLOOR + 0.33, sz], 0.022, 0.3), vox(M.chrome));
  upholstered(B, cylinder([x, FLOOR + 0.66, sz], 0.18, 0.045), M.leather, 0.025);
  const hue = R();
  const color: Vec3 = hue < 0.33 ? [1.5, 0.3, 0.8] : hue < 0.66 ? [0.3, 1.0, 1.5] : [1.6, 1.0, 0.2];
  return { pos: [x, FLOOR + 1.0, z1 + 0.4], radius: 2.0, color, shadow: false, anchors: [[x, FLOOR + 0.97, z1 - 0.05 + VS]] };
}

type Facing = 'south' | 'west';

/**
 * Neon text. 'south' signs hang on a wall facing +z (text runs +x);
 * 'west' signs hang on a wall facing +x (text runs -z). `at` is the centre.
 */
function neonText(B: Builder, text: string, at: Vec3, height: number, facing: Facing, mat: number, tube = 0.016): Vec3[] {
  const { strokes, width } = layoutText(text);
  const s = height / 6;
  const [ax, ay, az] = at;
  const map = (u: number, v: number): Vec3 =>
    facing === 'south' ? [ax + (u - width / 2) * s, ay + (v - 3) * s, az] : [ax, ay + (v - 3) * s, az - (u - width / 2) * s];
  const pts: Vec3[] = [];
  for (const st of strokes) {
    const a = map(st.a[0], st.a[1]);
    const b = map(st.b[0], st.b[1]);
    B.add(capsule(a, b, tube), vox(mat));
    pts.push(a);
  }
  // Standoff pins back to the wall.
  for (let i = 0; i < pts.length; i += 3) {
    const p = pts[i];
    const back: Vec3 = facing === 'south' ? [p[0], p[1], p[2] - 0.05] : [p[0] - 0.05, p[1], p[2]];
    B.add(capsule(p, back, VS * 0.6), vox(M.steelDark));
  }
  return pts;
}

function neonSigns(B: Builder): SceneLight[] {
  const lights: SceneLight[] = [];
  // VEGAS over the gaming area, flanked by stars.
  const vy = 2.95, vz = WALL + 0.06;
  const p = neonText(B, 'VEGAS', [9.6, vy, vz], 0.62, 'south', M.neonPink, 0.02);
  neonText(B, '*', [7.85, vy, vz], 0.5, 'south', M.neonGold, 0.016);
  neonText(B, '*', [11.35, vy, vz], 0.5, 'south', M.neonGold, 0.016);
  B.add(boxMinMax(7.5, vy - 0.48, WALL, 11.7, vy - 0.45, WALL + 0.03), vox(M.neonCyan));
  lights.push({ pos: [8.8, vy, 0.8], radius: 3.6, color: [2.2, 0.35, 1.2], shadow: false, anchors: [p[0]] });
  lights.push({ pos: [10.4, vy, 0.8], radius: 3.6, color: [2.2, 0.35, 1.2], shadow: false, anchors: [p[p.length - 1]] });
  // BAR above the back bar.
  const b = neonText(B, 'BAR', [WALL + 0.06, 3.2, 5.0], 0.5, 'west', M.neonCyan, 0.018);
  lights.push({ pos: [0.9, 3.2, 5.0], radius: 3.0, color: [0.3, 1.6, 2.0], shadow: false, anchors: [b[0]] });
  // Martini glass.
  const mx = WALL + 0.06, my = 3.2, mz = 6.6;
  const ms: [number, number, number, number][] = [[-0.18, 0.2, 0, 0], [0.18, 0.2, 0, 0], [-0.18, 0.2, 0.18, 0.2], [0, 0, 0, -0.22], [-0.1, -0.22, 0.1, -0.22]];
  for (const [u0, v0, u1, v1] of ms) B.add(capsule([mx, my + v0, mz - u0], [mx, my + v1, mz - u1], 0.014), vox(M.neonPink));
  B.add(sphere([mx, my + 0.12, mz - 0.05], 0.03), vox(M.absinthe));
  B.add(capsule([mx, my + 0.08, mz], [mx, my + 0.24, mz - 0.12], 0.006), vox(M.neonGold));
  return lights;
}

// ---------------------------------------------------------------------------

function piano(B: Builder, cx: number, cz: number, yaw: number) {
  const c: Vec3 = [cx, 0, cz];
  const rot = (s: Shape) => rotateY(s, yaw, c);
  const outline = [-0.75, 0, 0.75, 0, 0.75, 0.5, 0.68, 0.75, 0.5, 0.95, 0.25, 1.15, 0.1, 1.35, 0.0, 1.6, -0.15, 1.8, -0.35, 1.92, -0.6, 1.95, -0.75, 1.9];
  const inner = [-0.71, 0.04, 0.71, 0.04, 0.71, 0.5, 0.64, 0.73, 0.47, 0.92, 0.22, 1.12, 0.06, 1.33, -0.04, 1.58, -0.18, 1.76, -0.36, 1.87, -0.6, 1.9, -0.71, 1.86];
  const zOff = -0.9; // so the body is centred on c
  const at = (pts: number[]) => pts.map((v, i) => (i % 2 ? v + zOff : v));
  const y0 = FLOOR + 0.62, y1 = FLOOR + 0.95;
  B.add(rot(subtract(prism(c, at(outline), y0, y1), prism(c, at(inner), y0 + 0.14, y1 + 0.01))), vox(M.lacquer));
  B.add(rot(prism(c, at(inner), y0 + 0.14, y0 + 0.17)), vox(M.gold));
  // Strings.
  for (let i = -6; i <= 6; i++) {
    const x = cx + i * 0.09;
    B.add(rot(capsule([x, y0 + 0.18, cz + zOff + 0.1], [x - 0.04, y0 + 0.18, cz + zOff + 1.55 - Math.abs(i) * 0.06], VS * 0.45)), vox(M.chrome));
  }
  // Lid, propped open on the straight (bass) side hinge.
  const hinge: Vec3 = [cx - 0.75, y1 + 0.01, cz];
  B.add(rot(rotateZ(prism(c, at(outline), y1, y1 + 0.022), 0.62, hinge)), vox(M.lacquer));
  const lx = cx - 0.75 + 1.15 * Math.cos(0.62), ly = y1 + 1.15 * Math.sin(0.62);
  B.add(rot(capsule([cx + 0.35, y1 - 0.02, cz + zOff + 0.9], [lx - 0.02, ly - 0.03, cz + zOff + 0.9], 0.012)), vox(M.lacquer));
  // Keyboard.
  const kz0 = cz + zOff - 0.24;
  B.add(rot(boxMinMax(cx - 0.75, y0 - 0.04, kz0, cx + 0.75, y0 + 0.07, cz + zOff + 0.02)), vox(M.lacquer));
  B.add(rot(boxMinMax(cx - 0.66, y0 + 0.07, kz0 + 0.02, cx + 0.66, y0 + 0.09, cz + zOff - 0.02)), localPaint((x) => {
    const k = Math.floor((x - (cx - 0.66)) / 0.0254);
    const f = (x - (cx - 0.66)) / 0.0254 - k;
    return vox(M.ivory, f < 0.12 ? 120 : 0);
  }, yaw, c));
  B.add(rot(boxMinMax(cx - 0.66, y0 + 0.09, kz0 + 0.1, cx + 0.66, y0 + 0.105, cz + zOff - 0.02)), localPaint((x) => {
    const u = (x - (cx - 0.66)) / 0.0254;
    const k = Math.floor(u + 0.5);
    const f = u + 0.5 - k;
    const n = ((k - 1) % 7 + 7) % 7;
    const has = n === 0 || n === 1 || n === 3 || n === 4 || n === 5;
    return has && Math.abs(f - 0.5) < 0.3 ? vox(M.lacquer) : 0;
  }, yaw, c));
  B.add(rot(boxMinMax(cx - 0.75, y0 + 0.07, cz + zOff - 0.02, cx + 0.75, y0 + 0.2, cz + zOff + 0.03)), vox(M.lacquer));
  B.add(rot(box([cx, y0 + 0.24, cz + zOff + 0.06], [0.25, 0.1, 0.01])), vox(M.lacquer));
  // Legs + casters.
  for (const [lx2, lz] of [[-0.62, 0.15], [0.62, 0.15], [-0.45, 1.72]]) {
    B.add(rot(cylinder([cx + lx2, FLOOR + 0.33, cz + zOff + lz], 0.05, 0.3)), vox(M.lacquer));
    B.add(rot(sphere([cx + lx2, FLOOR + 0.03, cz + zOff + lz], 0.032)), vox(M.gold));
  }
  // Pedal lyre.
  B.add(rot(box([cx, FLOOR + 0.3, cz + zOff + 0.25], [0.08, 0.3, 0.03])), vox(M.lacquer));
  for (const px of [-0.05, 0, 0.05]) B.add(rot(box([cx + px, FLOOR + 0.06, cz + zOff + 0.18], [0.012, 0.008, 0.05])), vox(M.gold));
  // Bench.
  const bz = cz + zOff - 0.62;
  for (const [sx, sz] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) B.add(rot(cylinder([cx + sx * 0.32, FLOOR + 0.24, bz + sz * 0.12], 0.02, 0.24)), vox(M.lacquer));
  B.add(rot(box([cx, FLOOR + 0.5, bz], [0.38, 0.03, 0.17])), vox(M.lacquer));
  upholstered(B, rot(roundBox([cx, FLOOR + 0.55, bz], [0.36, 0.035, 0.15], 0.025)), M.leather, 0.02);
}

function champagneTower(B: Builder, cx: number, cz: number) {
  const y0 = FLOOR + 2 * VS;
  B.add(lathe([cx, y0, cz], [0.32, 0, 0.32, 0.03, 0.09, 0.07, 0.07, 0.66, 0.42, 0.7, 0.42, 0.75], 0), vox(M.marble));
  B.add(torus([cx, y0 + 0.725, cz], 0.42, 0.012), vox(M.gold));
  const top = y0 + 0.75;
  const sp = 0.112;
  const levels = 5;
  for (let L = 0; L < levels; L++) {
    const n = levels - L;
    const y = top + L * 0.125 - L * VS * 0.5;
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        const x = cx + (i - (n - 1) / 2) * sp;
        const z = cz + (j - (n - 1) / 2) * sp;
        coupe(B, x, y, z, true);
      }
  }
  bottle(B, cx + 0.33, top, cz - 0.2, rng(5), 3);
}

function tv(B: Builder): SceneLight {
  const z0 = 8.55, z1 = 11.05, y0 = FLOOR + 1.15, y1 = FLOOR + 2.55;
  B.add(boxMinMax(WALL, FLOOR, z0 - 0.15, 0.7, FLOOR + 0.48, z1 + 0.15), vox(M.walnutZ));
  B.add(boxMinMax(WALL, FLOOR + 0.48, z0 - 0.17, 0.73, FLOOR + 0.52, z1 + 0.17), vox(M.marbleBlack));
  B.add(boxMinMax(WALL, y0, z0, WALL + 0.06, y1, z1), vox(M.plasticBlack));
  B.add(boxMinMax(WALL + 0.06, y0 + 0.03, z0 + 0.03, WALL + 0.06 + VS, y1 - 0.03, z1 - 0.03), vox(M.tv));
  // A pair of porcelain vases on the console.
  vase(B, 0.48, FLOOR + 0.52, z0 + 0.1, 0.6, M.porcelain);
  vase(B, 0.48, FLOOR + 0.52, z1 - 0.1, 0.6, M.porcelainRed);
  return { pos: [1.2, (y0 + y1) / 2, (z0 + z1) / 2], radius: 4.2, color: [0.5, 0.45, 1.6], shadow: false, anchors: [[WALL + 0.06 + VS * 0.5, (y0 + y1) / 2, (z0 + z1) / 2]] };
}

function vase(B: Builder, x: number, y: number, z: number, scale: number, mat: number) {
  const s = scale;
  const prof = [0, 0, 0.07, 0, 0.12, 0.05, 0.16, 0.17, 0.14, 0.3, 0.07, 0.4, 0.055, 0.46, 0.075, 0.5].map((v) => v * s);
  for (let i = 1; i < prof.length; i += 2) prof[i] += VS * 0.4;
  B.add(lathe([x, y, z], prof, 1.7 * VS), (px, py, pz) => {
    const a = Math.atan2(pz - z, px - x);
    const h = (py - y) / s;
    const band = (h > 0.04 && h < 0.06) || (h > 0.36 && h < 0.38) || (h > 0.47 && h < 0.49);
    const motif = h > 0.08 && h < 0.33 && Math.sin(a * 5 + Math.sin(h * 30) * 2) > 0.55;
    return vox(mat, band || motif ? 255 : 0);
  });
}

function decor(B: Builder, R: () => number) {
  // Pedestals with vases + a gold trophy along the south wall.
  const pz = 0.62;
  const items: [number, (x: number, y: number) => void][] = [
    [4.4, (x, y) => vase(B, x, y, pz, 1.0, M.porcelain)],
    [5.8, (x, y) => vase(B, x, y, pz, 1.15, M.porcelainRed)],
    [7.0, (x, y) => {
      B.add(lathe([x, y, pz], [0.09, 0, 0.09, 0.03, 0.03, 0.06, 0.02, 0.14, 0.05, 0.16, 0.14, 0.24, 0.16, 0.4], 2 * VS), vox(M.gold));
      B.add(torus([x - 0.17, y + 0.3, pz], 0.06, 0.012, 'z'), vox(M.gold));
      B.add(torus([x + 0.17, y + 0.3, pz], 0.06, 0.012, 'z'), vox(M.gold));
    }],
  ];
  for (const [x, place] of items) {
    B.add(box([x, FLOOR + 0.5, pz], [0.2, 0.5, 0.2]), vox(M.marbleBlack));
    B.add(box([x, FLOOR + 1.01, pz], [0.23, 0.02, 0.23]), vox(M.gold));
    place(x, FLOOR + 1.03);
  }
  // Abstract painting in a gold frame above the pedestals.
  const ax0 = 3.9, ax1 = 7.5, ay0 = 1.75, ay1 = 3.25;
  B.add(boxMinMax(ax0 - 0.08, ay0 - 0.08, WALL, ax1 + 0.08, ay1 + 0.08, WALL + 0.05), vox(M.gold));
  const blobs = Array.from({ length: 9 }, () => ({ x: ax0 + R() * (ax1 - ax0), y: ay0 + R() * (ay1 - ay0), r: 0.15 + R() * 0.4, s: 1 + Math.floor(R() * 254) }));
  B.add(boxMinMax(ax0, ay0, WALL, ax1, ay1, WALL + 0.06), (x, y) => {
    let shade = 0;
    for (const b of blobs) if (Math.hypot(x - b.x, (y - b.y) * 1.3) < b.r) shade = b.s;
    const stripe = Math.abs(Math.sin(x * 9 + y * 3)) < 0.06;
    return vox(M.paperArt, stripe ? 255 : shade);
  });
}

function palm(B: Builder, x: number, z: number, R: () => number) {
  const y0 = FLOOR;
  B.add(lathe([x, y0, z], [0, 0, 0.16, 0, 0.21, 0.42, 0.23, 0.46], 2 * VS), vox(M.terracotta));
  B.add(cylinder([x, y0 + 0.38, z], 0.19, 0.04), vox(M.soil));
  // Slightly curved segmented trunk.
  let px = x, py = y0 + 0.4, pz = z;
  const lean = R() * Math.PI * 2;
  for (let i = 0; i < 8; i++) {
    const nx = px + Math.cos(lean) * 0.025, ny = py + 0.17, nz = pz + Math.sin(lean) * 0.025;
    B.add(capsule([px, py, pz], [nx, ny, nz], 0.045 - i * 0.002), vox(M.trunk));
    B.add(torus([nx, ny - 0.02, nz], 0.04 - i * 0.002, 0.008), vox(M.trunk, 200));
    px = nx; py = ny; pz = nz;
  }
  // Fronds: chains of flattened ellipsoids arcing out and drooping.
  const n = 9;
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2 + R() * 0.3;
    const lift = 0.5 + R() * 0.5;
    let fx = px, fy = py, fz = pz;
    for (let k = 1; k <= 9; k++) {
      const t = k / 9;
      const dist = t * 0.85;
      const nx = px + Math.cos(a) * dist;
      const nz = pz + Math.sin(a) * dist;
      const ny = py + Math.sin(t * Math.PI * 0.9) * 0.28 * lift - t * t * 0.45;
      const w = 0.07 * Math.sin(t * Math.PI) + 0.015;
      const seg = capsule([fx, fy, fz], [nx, ny, nz], 0.012);
      B.add(seg, vox(M.leaf, 1));
      const c: Vec3 = [(fx + nx) / 2, (fy + ny) / 2, (fz + nz) / 2];
      B.add(rotateY(ellipsoid(c, [0.05, 0.012, w]), a, c), vox(M.leaf));
      fx = nx; fy = ny; fz = nz;
    }
  }
}
