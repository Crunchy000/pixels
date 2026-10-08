import { BRICK_VOXELS, DIMS, MAX_LIGHTS, MAX_PARTICLES, VOXEL_SIZE } from '../config';
import { packMaterials } from '../world/materials';
import { VoxelWorld } from '../world/VoxelWorld';
import commonWGSL from './shaders/common.wgsl?raw';
import raymarchWGSL from './shaders/raymarch.wgsl?raw';
import particleCommonWGSL from './shaders/particle_common.wgsl?raw';
import particleSimWGSL from './shaders/particle_sim.wgsl?raw';
import particleDrawWGSL from './shaders/particle_draw.wgsl?raw';
import postWGSL from './shaders/post.wgsl?raw';

export interface Light {
  pos: [number, number, number]; // metres
  radius: number; // metres
  color: [number, number, number]; // linear HDR
  shadow: boolean;
}

export interface FrameParams {
  viewProj: Float32Array<ArrayBuffer>;
  invViewProj: Float32Array<ArrayBuffer>;
  camPos: [number, number, number];
  time: number;
  dt: number;
  sunDir: [number, number, number];
  sunIntensity: number;
  sunColor: [number, number, number];
  ambient: [number, number, number];
  night: number;
  lights: Light[];
  reflections: boolean;
  lightShadows: boolean;
  exposure: number;
  bloom: number;
}

const PARTICLE_BYTES = 32;
const DEPOSIT_CAP = 1 << 15;
const DEPOSIT_BYTES = 16 + DEPOSIT_CAP * 8;
const BLOOM_LEVELS = 6;
/** ?debug=1 normals, 2 albedo, 3 distance, 4 material id. */
const DEBUG_VIEW = Number(new URLSearchParams(location.search).get('debug') ?? 0) || 0;

export class Renderer {
  device!: GPUDevice;
  context!: GPUCanvasContext;
  format!: GPUTextureFormat;
  canvas: HTMLCanvasElement;
  world: VoxelWorld;

  private uniformBuf!: GPUBuffer;
  private uniformData = new Float32Array(64);
  private gridBuf!: GPUBuffer;
  private poolBuf!: GPUBuffer;
  private poolGeneration = -1;
  private poolCapacityBricks = 0;
  private matBuf!: GPUBuffer;
  private lightBuf!: GPUBuffer;
  private lightData = new Float32Array(MAX_LIGHTS * 8);
  private densityTex!: GPUTexture;
  private density: Uint8Array<ArrayBuffer>;
  private linearSampler!: GPUSampler;
  private nearestSampler!: GPUSampler;

  private group0Layout!: GPUBindGroupLayout;
  private group0!: GPUBindGroup;

  private raymarchPipeline!: GPURenderPipeline;
  private simPipeline!: GPUComputePipeline;
  private drawPipeline!: GPURenderPipeline;
  private brightPipeline!: GPURenderPipeline;
  private downPipeline!: GPURenderPipeline;
  private compositePipeline!: GPURenderPipeline;

  private particleBuf!: GPUBuffer;
  private aliveBuf!: GPUBuffer;
  private drawArgsBuf!: GPUBuffer;
  private depositBuf!: GPUBuffer;
  private readbackBuf!: GPUBuffer;
  private readbackPending = false;
  private simGroup!: GPUBindGroup;
  private drawGroup!: GPUBindGroup;
  private particleCursor = 0;
  private spawnStaging = new ArrayBuffer(PARTICLE_BYTES * 8192);
  private spawnF32 = new Float32Array(this.spawnStaging);
  private spawnU32 = new Uint32Array(this.spawnStaging);
  private spawnCount = 0;

  // Size-dependent resources.
  private width = 0;
  private height = 0;
  private sceneColor!: GPUTexture;
  private sceneDepth!: GPUTexture;
  private bloomTex!: GPUTexture;
  private bloomPasses: { view: GPUTextureView; group: GPUBindGroup; buf: GPUBuffer; w: number; h: number; bright: boolean }[] = [];
  private compositeGroup!: GPUBindGroup;
  private compositeBuf!: GPUBuffer;
  pixelated = false;
  private pixelatedBound = false;

  onDeposits: ((items: Uint32Array, count: number) => void) | null = null;
  lastError: string | null = null;

  constructor(canvas: HTMLCanvasElement, world: VoxelWorld) {
    this.canvas = canvas;
    this.world = world;
    this.density = new Uint8Array(world.grid.length);
  }

  async init(): Promise<void> {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser.');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter found.');
    const want = Math.max(this.world.poolCapacity * BRICK_VOXELS * 2 * 2, 256 << 20);
    this.device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, want),
        maxBufferSize: Math.min(adapter.limits.maxBufferSize, want),
      },
    });
    this.device.addEventListener('uncapturederror', (e) => {
      const msg = (e as GPUUncapturedErrorEvent).error.message;
      console.error(msg);
      this.lastError = msg;
    });
    this.device.lost.then((info) => {
      this.lastError = `GPU device lost: ${info.message}`;
    });
    this.context = this.canvas.getContext('webgpu')!;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device: this.device, format: this.format, alphaMode: 'opaque' });

    this.createStaticResources();
    await this.createPipelines();
    this.syncWorld(true);
  }

  get maxPoolBricks(): number {
    return Math.floor(this.device.limits.maxStorageBufferBindingSize / (BRICK_VOXELS * 2));
  }

  private createStaticResources(): void {
    const d = this.device;
    const w = this.world;
    this.uniformBuf = d.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.gridBuf = d.createBuffer({ size: w.grid.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const mats = packMaterials();
    this.matBuf = d.createBuffer({ size: mats.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.matBuf, 0, mats);
    this.lightBuf = d.createBuffer({ size: this.lightData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.densityTex = d.createTexture({
      size: [w.bx, w.by, w.bz],
      dimension: '3d',
      format: 'r8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.linearSampler = d.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge' });
    this.nearestSampler = d.createSampler({ magFilter: 'nearest', minFilter: 'nearest', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });

    const all = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE;
    this.group0Layout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: all, buffer: { type: 'uniform' } },
        { binding: 1, visibility: all, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: all, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: all, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: all, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: all, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 6, visibility: all, sampler: { type: 'filtering' } },
      ],
    });

    // Particles.
    this.particleBuf = d.createBuffer({ size: MAX_PARTICLES * PARTICLE_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.aliveBuf = d.createBuffer({ size: MAX_PARTICLES * 4, usage: GPUBufferUsage.STORAGE });
    this.drawArgsBuf = d.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.drawArgsBuf, 0, new Uint32Array([36, 0, 0, 0]));
    this.depositBuf = d.createBuffer({ size: DEPOSIT_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.depositBuf, 0, new Uint32Array([0, DEPOSIT_CAP, 0, 0]));
    this.readbackBuf = d.createBuffer({ size: DEPOSIT_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  }

  private async createPipelines(): Promise<void> {
    const d = this.device;
    const module = async (label: string, code: string) => {
      const m = d.createShaderModule({ label, code });
      const info = await m.getCompilationInfo();
      const errs = info.messages.filter((x) => x.type === 'error');
      if (errs.length) {
        const text = errs.map((e) => `${label}:${e.lineNum}:${e.linePos} ${e.message}`).join('\n');
        throw new Error(`Shader compile failed\n${text}`);
      }
      return m;
    };
    const rayMod = await module('raymarch', commonWGSL + raymarchWGSL);
    const simMod = await module('particle-sim', commonWGSL + particleCommonWGSL + particleSimWGSL);
    const drawMod = await module('particle-draw', commonWGSL + particleCommonWGSL + particleDrawWGSL);
    const postMod = await module('post', postWGSL);

    const simLayout = d.createBindGroupLayout({
      entries: [0, 1, 2, 3].map((b) => ({ binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' as const } })),
    });
    const drawLayout = d.createBindGroupLayout({
      entries: [0, 1].map((b) => ({ binding: b, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' as const } })),
    });

    this.raymarchPipeline = await d.createRenderPipelineAsync({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.group0Layout] }),
      vertex: { module: rayMod, entryPoint: 'vsMain' },
      fragment: { module: rayMod, entryPoint: 'fsMain', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'always' },
    });
    this.simPipeline = await d.createComputePipelineAsync({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.group0Layout, simLayout] }),
      compute: { module: simMod, entryPoint: 'simulate' },
    });
    this.drawPipeline = await d.createRenderPipelineAsync({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.group0Layout, drawLayout] }),
      vertex: { module: drawMod, entryPoint: 'vsParticle' },
      fragment: { module: drawMod, entryPoint: 'fsParticle', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    });
    this.brightPipeline = await d.createRenderPipelineAsync({
      layout: 'auto',
      vertex: { module: postMod, entryPoint: 'vsFull' },
      fragment: { module: postMod, entryPoint: 'fsBright', targets: [{ format: 'rgba16float' }] },
    });
    this.downPipeline = await d.createRenderPipelineAsync({
      layout: 'auto',
      vertex: { module: postMod, entryPoint: 'vsFull' },
      fragment: { module: postMod, entryPoint: 'fsDown', targets: [{ format: 'rgba16float' }] },
    });
    this.compositePipeline = await d.createRenderPipelineAsync({
      layout: 'auto',
      vertex: { module: postMod, entryPoint: 'vsFull' },
      fragment: { module: postMod, entryPoint: 'fsComposite', targets: [{ format: this.format }] },
    });

    this.simGroup = d.createBindGroup({
      layout: simLayout,
      entries: [
        { binding: 0, resource: { buffer: this.particleBuf } },
        { binding: 1, resource: { buffer: this.aliveBuf } },
        { binding: 2, resource: { buffer: this.drawArgsBuf } },
        { binding: 3, resource: { buffer: this.depositBuf } },
      ],
    });
    this.drawGroup = d.createBindGroup({
      layout: drawLayout,
      entries: [
        { binding: 0, resource: { buffer: this.particleBuf } },
        { binding: 1, resource: { buffer: this.aliveBuf } },
      ],
    });
  }

  private rebuildGroup0(): void {
    this.group0 = this.device.createBindGroup({
      layout: this.group0Layout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: { buffer: this.gridBuf } },
        { binding: 2, resource: { buffer: this.poolBuf } },
        { binding: 3, resource: { buffer: this.matBuf } },
        { binding: 4, resource: { buffer: this.lightBuf } },
        { binding: 5, resource: this.densityTex.createView() },
        { binding: 6, resource: this.linearSampler },
      ],
    });
  }

  /** Push CPU-side voxel edits to the GPU. */
  syncWorld(full = false): void {
    const w = this.world;
    const q = this.device.queue;

    // Collapse edited bricks first so freed pool slots aren't uploaded.
    for (const ci of w.dirtyCells) w.compactCell(ci);

    if (full || w.poolGeneration !== this.poolGeneration || w.poolCapacity > this.poolCapacityBricks) {
      const cap = w.poolCapacity;
      if (cap > this.maxPoolBricks) {
        this.lastError = `Voxel pool exceeds this GPU's storage limit (${cap} > ${this.maxPoolBricks} bricks). Try a lower ?vpm=`;
      }
      this.poolBuf?.destroy();
      this.poolBuf = this.device.createBuffer({
        size: Math.min(cap, this.maxPoolBricks) * BRICK_VOXELS * 2,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.poolCapacityBricks = cap;
      this.poolGeneration = w.poolGeneration;
      const used = Math.min(w.poolUsed, this.maxPoolBricks);
      if (used > 0) q.writeBuffer(this.poolBuf, 0, w.pool, 0, used * BRICK_VOXELS);
      q.writeBuffer(this.gridBuf, 0, w.grid);
      for (let i = 0; i < w.grid.length; i++) this.density[i] = w.cellDensity(i);
      q.writeTexture({ texture: this.densityTex }, this.density, { bytesPerRow: w.bx, rowsPerImage: w.by }, [w.bx, w.by, w.bz]);
      w.dirtyCells.clear();
      w.dirtyBricks.clear();
      this.rebuildGroup0();
      return;
    }

    if (w.dirtyCells.size) {
      const cells = Array.from(w.dirtyCells).sort((a, b) => a - b);
      uploadRanges(cells, 32, (start, count) => q.writeBuffer(this.gridBuf, start * 4, w.grid, start, count));
      let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -1, y1 = -1, z1 = -1;
      for (const ci of cells) {
        this.density[ci] = w.cellDensity(ci);
        const x = ci % w.bx, y = Math.floor(ci / w.bx) % w.by, z = Math.floor(ci / (w.bx * w.by));
        x0 = Math.min(x0, x); y0 = Math.min(y0, y); z0 = Math.min(z0, z);
        x1 = Math.max(x1, x); y1 = Math.max(y1, y); z1 = Math.max(z1, z);
      }
      q.writeTexture(
        { texture: this.densityTex, origin: [x0, y0, z0] },
        this.density,
        { offset: x0 + w.bx * (y0 + w.by * z0), bytesPerRow: w.bx, rowsPerImage: w.by },
        [x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1],
      );
      w.dirtyCells.clear();
    }
    if (w.dirtyBricks.size) {
      const bricks = Array.from(w.dirtyBricks).sort((a, b) => a - b);
      uploadRanges(bricks, 1, (start, count) => {
        if (start + count > this.maxPoolBricks) return;
        q.writeBuffer(this.poolBuf, start * BRICK_VOXELS * 2, w.pool, start * BRICK_VOXELS, count * BRICK_VOXELS);
      });
      w.dirtyBricks.clear();
    }
  }

  /** Queue a debris particle (positions/velocities in voxel units). */
  spawn(x: number, y: number, z: number, vx: number, vy: number, vz: number, value: number, life: number): void {
    if (this.spawnCount * 8 >= this.spawnF32.length) this.flushSpawns();
    const o = this.spawnCount * 8;
    this.spawnF32[o] = x;
    this.spawnF32[o + 1] = y;
    this.spawnF32[o + 2] = z;
    this.spawnU32[o + 3] = value >>> 0;
    this.spawnF32[o + 4] = vx;
    this.spawnF32[o + 5] = vy;
    this.spawnF32[o + 6] = vz;
    this.spawnF32[o + 7] = life;
    this.spawnCount++;
  }

  private flushSpawns(): void {
    if (!this.spawnCount) return;
    const q = this.device.queue;
    let n = this.spawnCount;
    let src = 0;
    while (n > 0) {
      const room = MAX_PARTICLES - this.particleCursor;
      const k = Math.min(n, room);
      q.writeBuffer(this.particleBuf, this.particleCursor * PARTICLE_BYTES, this.spawnStaging, src * PARTICLE_BYTES, k * PARTICLE_BYTES);
      this.particleCursor = (this.particleCursor + k) % MAX_PARTICLES;
      src += k;
      n -= k;
    }
    this.spawnCount = 0;
  }

  /** (Re)create the internal render targets at the given resolution. */
  private ensureTargets(rw: number, rh: number): void {
    const w = Math.max(1, Math.round(rw));
    const h = Math.max(1, Math.round(rh));
    if (w === this.width && h === this.height && this.pixelatedBound === this.pixelated) return;
    const d = this.device;
    if (w !== this.width || h !== this.height) {
      this.sceneColor?.destroy();
      this.sceneDepth?.destroy();
      this.bloomTex?.destroy();
      for (const p of this.bloomPasses) p.buf.destroy();
      this.compositeBuf?.destroy();
      this.width = w;
      this.height = h;
      this.sceneColor = d.createTexture({ size: [w, h], format: 'rgba16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
      this.sceneDepth = d.createTexture({ size: [w, h], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT });
      const bw = Math.max(1, w >> 1), bh = Math.max(1, h >> 1);
      const levels = Math.max(1, Math.min(BLOOM_LEVELS, Math.floor(Math.log2(Math.min(bw, bh))) - 1));
      this.bloomTex = d.createTexture({ size: [bw, bh], format: 'rgba16float', mipLevelCount: levels, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
      this.bloomPasses = [];
      for (let l = 0; l < levels; l++) {
        const bright = l === 0;
        const srcView = bright ? this.sceneColor.createView() : this.bloomTex.createView({ baseMipLevel: l - 1, mipLevelCount: 1 });
        const srcW = bright ? w : Math.max(1, bw >> (l - 1));
        const srcH = bright ? h : Math.max(1, bh >> (l - 1));
        const buf = d.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        d.queue.writeBuffer(buf, 0, new Float32Array([1 / srcW, 1 / srcH, 1.0, 1, 0, 0, 0, 0]));
        const pipeline = bright ? this.brightPipeline : this.downPipeline;
        const group = d.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: srcView },
            { binding: 1, resource: this.linearSampler },
            { binding: 2, resource: { buffer: buf } },
          ],
        });
        this.bloomPasses.push({
          view: this.bloomTex.createView({ baseMipLevel: l, mipLevelCount: 1 }),
          group,
          buf,
          w: Math.max(1, bw >> l),
          h: Math.max(1, bh >> l),
          bright,
        });
      }
      this.compositeBuf = d.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }
    this.pixelatedBound = this.pixelated;
    this.compositeGroup = d.createBindGroup({
      layout: this.compositePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.sceneColor.createView() },
        { binding: 1, resource: this.pixelated ? this.nearestSampler : this.linearSampler },
        { binding: 2, resource: { buffer: this.compositeBuf } },
        { binding: 3, resource: this.bloomTex.createView() },
        { binding: 4, resource: this.linearSampler },
      ],
    });
  }

  render(f: FrameParams, renderWidth: number, renderHeight: number): void {
    const d = this.device;
    this.flushSpawns();
    this.syncWorld();
    this.ensureTargets(renderWidth, renderHeight);

    const u = this.uniformData;
    u.set(f.invViewProj, 0);
    u.set(f.viewProj, 16);
    u.set([f.camPos[0], f.camPos[1], f.camPos[2], f.time], 32);
    u.set([DIMS.x, DIMS.y, DIMS.z, VOXEL_SIZE], 36);
    u.set([f.sunDir[0], f.sunDir[1], f.sunDir[2], f.sunIntensity], 40);
    const nLights = Math.min(MAX_LIGHTS, f.lights.length);
    u.set([f.sunColor[0], f.sunColor[1], f.sunColor[2], nLights], 44);
    u.set([f.ambient[0], f.ambient[1], f.ambient[2], 1.1], 48);
    u.set([this.width, this.height, 1 / this.width, 1 / this.height], 52);
    u.set([f.reflections ? 1 : 0, f.lightShadows ? 1 : 0, f.exposure, DEBUG_VIEW], 56);
    u.set([Math.min(f.dt, 1 / 20), 9.81 / VOXEL_SIZE, MAX_PARTICLES, f.night], 60);
    d.queue.writeBuffer(this.uniformBuf, 0, u);

    for (let i = 0; i < nLights; i++) {
      const l = f.lights[i];
      this.lightData.set([l.pos[0], l.pos[1], l.pos[2], l.radius, l.color[0], l.color[1], l.color[2], l.shadow ? 1 : 0], i * 8);
    }
    d.queue.writeBuffer(this.lightBuf, 0, this.lightData);
    d.queue.writeBuffer(this.drawArgsBuf, 4, new Uint32Array([0]));
    d.queue.writeBuffer(this.compositeBuf, 0, new Float32Array([0, 0, 0, 0, f.exposure, f.bloom, 0.35, this.format.endsWith('-srgb') ? 0 : 1]));

    const enc = d.createCommandEncoder();

    // Particle simulation.
    {
      const pass = enc.beginComputePass();
      pass.setPipeline(this.simPipeline);
      pass.setBindGroup(0, this.group0);
      pass.setBindGroup(1, this.simGroup);
      pass.dispatchWorkgroups(Math.ceil(MAX_PARTICLES / 64));
      pass.end();
    }
    let readback = false;
    if (!this.readbackPending && this.onDeposits) {
      enc.copyBufferToBuffer(this.depositBuf, 0, this.readbackBuf, 0, DEPOSIT_BYTES);
      enc.clearBuffer(this.depositBuf, 0, 4);
      readback = true;
    }

    // Scene: ray-marched voxels, then debris cubes depth-tested against it.
    {
      const pass = enc.beginRenderPass({
        colorAttachments: [{ view: this.sceneColor.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
        depthStencilAttachment: { view: this.sceneDepth.createView(), depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1 },
      });
      pass.setBindGroup(0, this.group0);
      pass.setPipeline(this.raymarchPipeline);
      pass.draw(3);
      pass.setPipeline(this.drawPipeline);
      pass.setBindGroup(1, this.drawGroup);
      pass.drawIndirect(this.drawArgsBuf, 0);
      pass.end();
    }

    // Bloom chain.
    for (const p of this.bloomPasses) {
      const pass = enc.beginRenderPass({ colorAttachments: [{ view: p.view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
      pass.setPipeline(p.bright ? this.brightPipeline : this.downPipeline);
      pass.setBindGroup(0, p.group);
      pass.draw(3);
      pass.end();
    }

    {
      const pass = enc.beginRenderPass({
        colorAttachments: [{ view: this.context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
      });
      pass.setPipeline(this.compositePipeline);
      pass.setBindGroup(0, this.compositeGroup);
      pass.draw(3);
      pass.end();
    }
    d.queue.submit([enc.finish()]);

    if (readback) {
      this.readbackPending = true;
      this.readbackBuf.mapAsync(GPUMapMode.READ).then(
        () => {
          const data = new Uint32Array(this.readbackBuf.getMappedRange().slice(0));
          this.readbackBuf.unmap();
          this.readbackPending = false;
          const count = Math.min(data[0], DEPOSIT_CAP);
          if (count > 0 && this.onDeposits) this.onDeposits(data.subarray(4), count);
        },
        () => {
          this.readbackPending = false;
        },
      );
    }
  }
}

function uploadRanges(sorted: number[], gap: number, write: (start: number, count: number) => void): void {
  let start = sorted[0];
  let end = start;
  for (let i = 1; i < sorted.length; i++) {
    const v = sorted[i];
    if (v - end <= gap) end = v;
    else {
      write(start, end - start + 1);
      start = end = v;
    }
  }
  write(start, end - start + 1);
}
