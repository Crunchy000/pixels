import { BRICK, BRICK_VOXELS } from '../config';
import { mat } from './materials';

// Sparse two-level voxel store mirrored on the GPU.
//
//   grid[brick] : 0                  -> empty brick (all air)
//                 0x80000000 | value -> uniform brick: all 512 voxels == value
//                 n > 0              -> pool brick n-1 holds 512 explicit u16 voxels
//
// Most of a room is air or big homogeneous slabs, so only bricks along
// material boundaries cost real memory.

export const UNIFORM = 0x80000000;

export class VoxelWorld {
  readonly dx: number;
  readonly dy: number;
  readonly dz: number;
  readonly bx: number;
  readonly by: number;
  readonly bz: number;
  readonly grid: Uint32Array<ArrayBuffer>;
  pool: Uint16Array<ArrayBuffer>;
  poolCapacity: number;
  poolUsed = 0;
  private freeList: number[] = [];

  /** Grid cells whose entry changed since the last GPU sync. */
  readonly dirtyCells = new Set<number>();
  /** Pool bricks whose voxels changed since the last GPU sync. */
  readonly dirtyBricks = new Set<number>();
  /** Bumped when the pool is reallocated (GPU must re-create its buffer). */
  poolGeneration = 0;

  constructor(dx: number, dy: number, dz: number, initialPoolBricks = 1 << 14) {
    this.dx = dx;
    this.dy = dy;
    this.dz = dz;
    this.bx = dx / BRICK;
    this.by = dy / BRICK;
    this.bz = dz / BRICK;
    this.grid = new Uint32Array(this.bx * this.by * this.bz);
    this.poolCapacity = initialPoolBricks;
    this.pool = new Uint16Array(initialPoolBricks * BRICK_VOXELS);
  }

  cellIndex(bx: number, by: number, bz: number): number {
    return bx + this.bx * (by + this.by * bz);
  }

  inBounds(x: number, y: number, z: number): boolean {
    return x >= 0 && y >= 0 && z >= 0 && x < this.dx && y < this.dy && z < this.dz;
  }

  get(x: number, y: number, z: number): number {
    if (x < 0 || y < 0 || z < 0 || x >= this.dx || y >= this.dy || z >= this.dz) return 0;
    const g = this.grid[(x >> 3) + this.bx * ((y >> 3) + this.by * (z >> 3))];
    if (g === 0) return 0;
    if (g & UNIFORM) return g & 0xffff;
    return this.pool[(g - 1) * BRICK_VOXELS + ((x & 7) | ((y & 7) << 3) | ((z & 7) << 6))];
  }

  set(x: number, y: number, z: number, v: number): void {
    if (x < 0 || y < 0 || z < 0 || x >= this.dx || y >= this.dy || z >= this.dz) return;
    const ci = (x >> 3) + this.bx * ((y >> 3) + this.by * (z >> 3));
    const li = (x & 7) | ((y & 7) << 3) | ((z & 7) << 6);
    let g = this.grid[ci];
    if (g === 0) {
      if (v === 0) return;
      g = this.allocBrick(0) + 1;
      this.grid[ci] = g;
      this.dirtyCells.add(ci);
    } else if (g & UNIFORM) {
      const u = g & 0xffff;
      if (u === v) return;
      g = this.allocBrick(u) + 1;
      this.grid[ci] = g;
      this.dirtyCells.add(ci);
    }
    const p = g - 1;
    const off = p * BRICK_VOXELS + li;
    if (this.pool[off] === v) return;
    this.pool[off] = v;
    this.dirtyBricks.add(p);
    this.dirtyCells.add(ci); // density changes too
  }

  /** Fill a whole brick with one value (used by the scene rasteriser). */
  setBrickUniform(ci: number, v: number): void {
    const g = this.grid[ci];
    if (g !== 0 && !(g & UNIFORM)) this.freeBrick(g - 1);
    const next = v === 0 ? 0 : (UNIFORM | v) >>> 0;
    if (this.grid[ci] !== next) {
      this.grid[ci] = next;
      this.dirtyCells.add(ci);
    }
  }

  /**
   * Returns a pool index the caller may write to for cell `ci`, expanding
   * empty/uniform cells as needed. Used for bulk per-voxel writes.
   */
  editableBrick(ci: number): number {
    const g = this.grid[ci];
    if (g !== 0 && !(g & UNIFORM)) return g - 1;
    const p = this.allocBrick(g === 0 ? 0 : g & 0xffff);
    this.grid[ci] = p + 1;
    this.dirtyCells.add(ci);
    return p;
  }

  private allocBrick(fill: number): number {
    let p: number;
    if (this.freeList.length) p = this.freeList.pop()!;
    else {
      if (this.poolUsed >= this.poolCapacity) this.growPool();
      p = this.poolUsed++;
    }
    this.pool.fill(fill, p * BRICK_VOXELS, (p + 1) * BRICK_VOXELS);
    this.dirtyBricks.add(p);
    return p;
  }

  private freeBrick(p: number): void {
    this.freeList.push(p);
    this.dirtyBricks.delete(p);
  }

  private growPool(): void {
    const cap = Math.ceil(this.poolCapacity * 1.5);
    const next = new Uint16Array(cap * BRICK_VOXELS);
    next.set(this.pool);
    this.pool = next;
    this.poolCapacity = cap;
    this.poolGeneration++;
  }

  /** Pool bricks that are actually live (capacity minus free list). */
  get liveBricks(): number {
    return this.poolUsed - this.freeList.length;
  }

  /**
   * Collapse pool bricks that became all-air or all-one-value back into
   * empty/uniform cells. Cheap enough to run on every dirty cell per frame.
   */
  compactCell(ci: number): void {
    const g = this.grid[ci];
    if (g === 0 || g & UNIFORM) return;
    const p = g - 1;
    const base = p * BRICK_VOXELS;
    const first = this.pool[base];
    for (let i = 1; i < BRICK_VOXELS; i++) if (this.pool[base + i] !== first) return;
    this.freeBrick(p);
    this.grid[ci] = first === 0 ? 0 : (UNIFORM | first) >>> 0;
  }

  /** Occupancy of a brick in 0..255 (opaque voxels only), used for coarse AO. */
  cellDensity(ci: number): number {
    const g = this.grid[ci];
    if (g === 0) return 0;
    if (g & UNIFORM) return mat(g & 0xff).glass > 0 ? 0 : 255;
    const base = (g - 1) * BRICK_VOXELS;
    let n = 0;
    for (let i = 0; i < BRICK_VOXELS; i++) {
      const v = this.pool[base + i];
      if (v !== 0 && OPAQUE[v & 0xff]) n++;
    }
    return Math.round((n / BRICK_VOXELS) * 255);
  }

  /** Voxel-level DDA for gameplay ray casts (in voxel units). */
  raycast(ox: number, oy: number, oz: number, dx: number, dy: number, dz: number, maxDist: number, skipGlass = false): RayHit | null {
    let x = Math.floor(ox);
    let y = Math.floor(oy);
    let z = Math.floor(oz);
    const sx = dx > 0 ? 1 : -1;
    const sy = dy > 0 ? 1 : -1;
    const sz = dz > 0 ? 1 : -1;
    const idx = Math.abs(1 / (dx || 1e-9));
    const idy = Math.abs(1 / (dy || 1e-9));
    const idz = Math.abs(1 / (dz || 1e-9));
    let tx = (dx > 0 ? x + 1 - ox : ox - x) * idx;
    let ty = (dy > 0 ? y + 1 - oy : oy - y) * idy;
    let tz = (dz > 0 ? z + 1 - oz : oz - z) * idz;
    let t = 0;
    let nx = 0;
    let ny = 0;
    let nz = 0;
    while (t <= maxDist) {
      const v = this.get(x, y, z);
      if (v !== 0 && !(skipGlass && !OPAQUE[v & 0xff])) return { x, y, z, nx, ny, nz, t, v };
      if (tx < ty && tx < tz) {
        x += sx;
        t = tx;
        tx += idx;
        nx = -sx;
        ny = 0;
        nz = 0;
      } else if (ty < tz) {
        y += sy;
        t = ty;
        ty += idy;
        nx = 0;
        ny = -sy;
        nz = 0;
      } else {
        z += sz;
        t = tz;
        tz += idz;
        nx = 0;
        ny = 0;
        nz = -sz;
      }
      if (x < -1 || y < -1 || z < -1 || x > this.dx || y > this.dy || z > this.dz) return null;
    }
    return null;
  }
}

export interface RayHit {
  x: number;
  y: number;
  z: number;
  nx: number;
  ny: number;
  nz: number;
  t: number;
  v: number;
}

/** OPAQUE[materialId] = 1 unless the material is see-through. Filled lazily by initOpaqueTable(). */
export const OPAQUE = new Uint8Array(256);
export function initOpaqueTable(): void {
  for (let i = 1; i < 256; i++) {
    const m = mat(i);
    OPAQUE[i] = !m || m.glass === 0 ? 1 : 0;
  }
}
