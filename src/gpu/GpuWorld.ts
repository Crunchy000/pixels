import { BRICK_VOXELS, MAX_PARTICLES, VOXEL_SIZE } from '../config';
import type { Scene, SceneLight } from '../scene/penthouse';
import { mat, packMaterials } from '../world/materials';
import { VoxelWorld } from '../world/VoxelWorld';
import { ARGS, ARGS_WORDS, CTRL, CTRL_WORDS, HeapLayout, LIMITS, heapLayout } from './layout';

/** Bytes of the CPU-written uniform block (13 vec4, see `Uniforms` in shared.wgsl). */
export const UNIFORM_FLOATS = 52;
/** Bytes of the GPU-owned camera block (see `Camera` in shared.wgsl). */
const CAMERA_BYTES = 256;
const PARTICLE_BYTES = 32;

/**
 * Every GPU resource the simulation and the renderer share. The voxel world is
 * uploaded exactly once here; from then on only the GPU touches it.
 */
export class GpuWorld {
  readonly device: GPUDevice;
  readonly dims: { x: number; y: number; z: number };
  readonly bricks: { x: number; y: number; z: number };
  readonly supers: { x: number; y: number; z: number };
  readonly numCells: number;
  readonly poolCapacity: number;
  readonly heap: HeapLayout;
  readonly numLights: number;
  readonly numAnchors: number;

  readonly uniformBuf: GPUBuffer;
  readonly uniform = new Float32Array(UNIFORM_FLOATS);
  readonly cameraBuf: GPUBuffer;
  readonly gridBuf: GPUBuffer;
  readonly poolBuf: GPUBuffer;
  readonly superBuf: GPUBuffer;
  readonly matBuf: GPUBuffer;
  readonly ctrlBuf: GPUBuffer;
  /** Indirect dispatch + draw args (written by the GPU, consumed by indirect calls). */
  readonly argsBuf: GPUBuffer;
  readonly heapBuf: GPUBuffer;
  readonly particleBuf: GPUBuffer;
  readonly densityTex: GPUTexture;
  /** The heap range holding the per-frame light list (bound by the renderer as `lights`). */
  readonly lightsRange: { offset: number; size: number };
  /** Live brick count right after the upload (for the HUD). */
  readonly initialBricks: number;

  constructor(device: GPUDevice, world: VoxelWorld, scene: Scene) {
    this.device = device;
    const d = device;
    const q = d.queue;
    this.dims = { x: world.dx, y: world.dy, z: world.dz };
    this.bricks = { x: world.bx, y: world.by, z: world.bz };
    this.supers = { x: Math.ceil(world.bx / 4), y: Math.ceil(world.by / 4), z: Math.ceil(world.bz / 4) };
    this.numCells = world.grid.length;

    // Pool: everything the scene uses plus generous headroom for destruction
    // (blasting expands compressed uniform bricks into real ones).
    const maxBricks = Math.floor(Math.min(d.limits.maxStorageBufferBindingSize, d.limits.maxBufferSize) / (BRICK_VOXELS * 2));
    const used = world.poolUsed;
    if (used > maxBricks) throw new Error(`This GPU can hold ${maxBricks} voxel bricks but the scene needs ${used}. Try a lower ?vpm=`);
    this.poolCapacity = Math.min(maxBricks, Math.max(used + 98304, used * 2));
    this.heap = heapLayout(this.numCells, this.poolCapacity, MAX_PARTICLES);
    if (this.heap.words * 4 > d.limits.maxStorageBufferBindingSize) throw new Error('Simulation heap exceeds the GPU storage buffer limit.');

    const S = GPUBufferUsage.STORAGE;
    const DST = GPUBufferUsage.COPY_DST;
    this.uniformBuf = d.createBuffer({ label: 'uniforms', size: 256, usage: GPUBufferUsage.UNIFORM | DST });
    this.cameraBuf = d.createBuffer({ label: 'camera', size: CAMERA_BYTES, usage: GPUBufferUsage.UNIFORM | S | DST | GPUBufferUsage.COPY_SRC });
    this.gridBuf = d.createBuffer({ label: 'grid', size: world.grid.byteLength, usage: S | DST });
    this.poolBuf = d.createBuffer({ label: 'pool', size: this.poolCapacity * BRICK_VOXELS * 2, usage: S | DST });
    this.superBuf = d.createBuffer({ label: 'super', size: this.supers.x * this.supers.y * this.supers.z * 4, usage: S | DST });
    const mats = packMaterials();
    this.matBuf = d.createBuffer({ label: 'materials', size: mats.byteLength, usage: S | DST });
    this.ctrlBuf = d.createBuffer({ label: 'ctrl', size: CTRL_WORDS * 4, usage: S | DST | GPUBufferUsage.COPY_SRC });
    this.argsBuf = d.createBuffer({ label: 'args', size: ARGS_WORDS * 4, usage: S | DST | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_SRC });
    this.heapBuf = d.createBuffer({ label: 'heap', size: this.heap.words * 4, usage: S | DST });
    this.particleBuf = d.createBuffer({ label: 'particles', size: MAX_PARTICLES * PARTICLE_BYTES, usage: S | DST });
    this.densityTex = d.createTexture({
      label: 'density',
      size: [world.bx, world.by, world.bz],
      dimension: '3d',
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
    });

    // ---- one-time upload -------------------------------------------------
    q.writeBuffer(this.matBuf, 0, mats);
    q.writeBuffer(this.gridBuf, 0, world.grid);
    if (used > 0) q.writeBuffer(this.poolBuf, 0, world.pool, 0, used * BRICK_VOXELS);

    const superCounts = new Uint32Array(this.supers.x * this.supers.y * this.supers.z);
    const density = new Uint8Array(this.numCells * 4);
    for (let z = 0; z < world.bz; z++)
      for (let y = 0; y < world.by; y++)
        for (let x = 0; x < world.bx; x++) {
          const ci = world.cellIndex(x, y, z);
          if (world.grid[ci] !== 0) superCounts[(x >> 2) + this.supers.x * ((y >> 2) + this.supers.y * (z >> 2))]++;
          density[ci * 4] = world.cellDensity(ci);
          density[ci * 4 + 3] = 255;
        }
    q.writeBuffer(this.superBuf, 0, superCounts);
    q.writeTexture({ texture: this.densityTex }, density, { bytesPerRow: world.bx * 4, rowsPerImage: world.by }, [world.bx, world.by, world.bz]);

    // Free list: holes inside the used range, then everything above it.
    const holes = world.freeBricks();
    const free = new Uint32Array(holes.length + (this.poolCapacity - used));
    free.set(holes);
    for (let i = 0; i < this.poolCapacity - used; i++) free[holes.length + i] = this.poolCapacity - 1 - i;
    q.writeBuffer(this.heapBuf, this.heap.offsets.FREE_LIST * 4, free);
    this.initialBricks = used - holes.length;

    const ctrl = new Uint32Array(CTRL_WORDS);
    ctrl[CTRL.C_FREE_TOP] = free.length;
    q.writeBuffer(this.ctrlBuf, 0, ctrl);
    const args = new Uint32Array(ARGS_WORDS);
    args[ARGS.A_DRAWP] = 36;
    args[ARGS.A_DRAWC] = 36;
    q.writeBuffer(this.argsBuf, 0, args);

    // Lights + the emissive voxels that power them.
    const { lights, anchors } = packLights(world, scene.lights);
    this.numLights = lights.length / 16;
    this.numAnchors = Math.floor(anchors.length / 8);
    q.writeBuffer(this.heapBuf, this.heap.offsets.LIGHTS * 4, lights);
    q.writeBuffer(this.heapBuf, this.heap.offsets.ANCHORS * 4, anchors);

    // Player starts at the scene's spawn point (the GPU owns it from here).
    const cam = new Float32Array(CAMERA_BYTES / 4);
    const [px, py, pz] = scene.spawn.pos;
    cam.set([px, py + 1.62, pz, 0], 32); // eye
    cam.set([px, py, pz, scene.spawn.yaw], 40); // feet + yaw
    cam.set([0, 0, 0, scene.spawn.pitch], 44); // vel + pitch
    q.writeBuffer(this.cameraBuf, 0, cam);

    // The GPU owns the world now.
    this.lightsRange = { offset: this.heap.offsets.LIGHTS_OUT * 4, size: (1 + 2 * (LIMITS.maxLights + LIMITS.maxFlashes)) * 16 };
    world.release();
  }
}

/** Light state (16 words) + anchors (8 words) in the layout sim.wgsl's `lights` pass expects. */
function packLights(world: VoxelWorld, scene: SceneLight[]): { lights: Uint32Array<ArrayBuffer>; anchors: Uint32Array<ArrayBuffer> } {
  const n = Math.min(scene.length, LIMITS.maxLights);
  const lights = new Uint32Array(n * 16);
  const lf = new Float32Array(lights.buffer);
  const anchorList: number[] = [];
  for (let i = 0; i < n; i++) {
    const l = scene[i];
    const start = anchorList.length / 8;
    for (const a of l.anchors) {
      if (anchorList.length / 8 >= LIMITS.maxAnchors) break;
      const v = snapToEmissive(world, Math.floor(a[0] / VOXEL_SIZE), Math.floor(a[1] / VOXEL_SIZE), Math.floor(a[2] / VOXEL_SIZE));
      anchorList.push(v[0], v[1], v[2], 0, v[1], 0, 0, 0);
    }
    const count = anchorList.length / 8 - start;
    const o = i * 16;
    lf.set([l.pos[0], l.pos[1], l.pos[2], l.radius, l.color[0], l.color[1], l.color[2], l.shadow ? 1 : 0, 1], o);
    lights[o + 9] = start;
    lights[o + 10] = count;
    lf[o + 11] = i * 7.31;
  }
  return { lights, anchors: new Uint32Array(anchorList.length ? anchorList : [0]) };
}

/** Find an emissive voxel at or near the anchor point (sample points can land just off a thin tube). */
function snapToEmissive(world: VoxelWorld, x: number, y: number, z: number): [number, number, number] {
  for (let r = 0; r <= 2; r++)
    for (let dz = -r; dz <= r; dz++)
      for (let dy = -r; dy <= r; dy++)
        for (let dx = -r; dx <= r; dx++) {
          const v = world.get(x + dx, y + dy, z + dz);
          if (v && mat(v & 0xff).emission > 0) return [x + dx, y + dy, z + dz];
        }
  return [x, y, z];
}

