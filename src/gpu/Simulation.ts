import { MAX_PARTICLES } from '../config';
import playerWGSL from '../render/shaders/player.wgsl?raw';
import sharedWGSL from '../render/shaders/shared.wgsl?raw';
import simWGSL from '../render/shaders/sim.wgsl?raw';
import { GpuWorld } from './GpuWorld';
import { ARGS, ARGS_WORDS, CTRL, CTRL_WORDS, LIMITS, layoutWGSL } from './layout';

const ENTRY_POINTS = [
  'frameBegin', 'shoot', 'impactArgs', 'impactJobs', 'allocArgs', 'allocBricks', 'carveArgs', 'carve', 'postCarve',
  'floodArgs', 'floodStep', 'searchBegin', 'searchClear', 'searchSeed', 'searchArgs', 'searchStep',
  'resolveArgs', 'resolveFrontier', 'resolveRoots', 'resolveChunks', 'resolveExpand', 'resolveExtract', 'resolveAnchors', 'resolveEnd',
  'chunkArgs', 'chunkBegin', 'chunkCollide', 'chunkStep', 'chunkLand', 'particleSim',
  'writeArgs', 'writeRequest', 'writeApply', 'maintArgs', 'maint', 'lights', 'frameEnd',
] as const;
type Entry = (typeof ENTRY_POINTS)[number];

/** Entry points that write indirect args: they get the args buffer as bind group 1. */
const ARGS_WRITERS = new Set<Entry>([
  'frameBegin', 'impactArgs', 'allocArgs', 'carveArgs', 'floodArgs', 'searchBegin', 'searchArgs',
  'resolveArgs', 'chunkArgs', 'writeArgs', 'maintArgs', 'frameEnd', 'particleSim',
]);

export interface SimStats {
  /** Dollars billed since the previous read. */
  bill: number;
  destroyed: number;
  falling: number;
  particles: number;
  liveBricks: number;
  allocFailures: number;
  searches: number;
  islands: number;
  searching: boolean;
  flying: boolean;
  eye: [number, number, number];
}

/** Bytes copied back for the HUD each frame: ctrl counters + args + the camera block. */
const READBACK_BYTES = CTRL_WORDS * 4 + ARGS_WORDS * 4 + 256;

export async function compileModule(device: GPUDevice, label: string, code: string): Promise<GPUShaderModule> {
  const m = device.createShaderModule({ label, code });
  const info = await m.getCompilationInfo();
  const errs = info.messages.filter((x) => x.type === 'error');
  if (errs.length) {
    const lines = code.split('\n');
    const text = errs.map((e) => `${label}:${e.lineNum}:${e.linePos} ${e.message}\n    ${lines[e.lineNum - 1]?.trim() ?? ''}`).join('\n');
    throw new Error(`Shader compile failed\n${text}`);
  }
  return m;
}

/**
 * The whole simulation as one compute pass of ~100 dispatches per frame.
 * Most are indirect, sized by counters earlier passes wrote, so idle stages
 * dispatch zero workgroups and nothing ever waits on the CPU.
 */
export class Simulation {
  private gw: GpuWorld;
  private pipes = {} as Record<Entry, GPUComputePipeline>;
  private player!: GPUComputePipeline;
  private simGroup!: GPUBindGroup;
  private argsGroup!: GPUBindGroup;
  private playerGroup!: GPUBindGroup;
  private readback: GPUBuffer;
  private readbackPending = false;
  stats: SimStats = { bill: 0, destroyed: 0, falling: 0, particles: 0, liveBricks: 0, allocFailures: 0, searches: 0, islands: 0, searching: false, flying: false, eye: [0, 0, 0] };
  /** Called with each stats sample (bill and destroyed are deltas). */
  onStats: ((s: SimStats) => void) | null = null;

  constructor(gw: GpuWorld) {
    this.gw = gw;
    this.readback = gw.device.createBuffer({ size: READBACK_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  }

  async init(): Promise<void> {
    const gw = this.gw;
    const d = gw.device;
    const consts = layoutWGSL(gw.heap, {
      POOL_CAP: gw.poolCapacity,
      MAX_PARTICLES,
      NUM_LIGHTS: gw.numLights,
      NUM_ANCHORS: gw.numAnchors,
    });
    const simMod = await compileModule(d, 'sim', consts + sharedWGSL + simWGSL);
    const playerMod = await compileModule(d, 'player', sharedWGSL + playerWGSL);

    const C = GPUShaderStage.COMPUTE;
    const simLayout = d.createBindGroupLayout({
      label: 'sim',
      entries: [
        { binding: 0, visibility: C, buffer: { type: 'uniform' } },
        { binding: 1, visibility: C, buffer: { type: 'uniform' } },
        ...[2, 3, 4].map((b) => ({ binding: b, visibility: C, buffer: { type: 'storage' as const } })),
        { binding: 5, visibility: C, buffer: { type: 'read-only-storage' } },
        ...[6, 7, 8].map((b) => ({ binding: b, visibility: C, buffer: { type: 'storage' as const } })),
        { binding: 10, visibility: C, storageTexture: { access: 'write-only', format: 'rgba8unorm', viewDimension: '3d' } },
      ],
    });
    const playerLayout = d.createBindGroupLayout({
      label: 'player',
      entries: [
        { binding: 0, visibility: C, buffer: { type: 'uniform' } },
        { binding: 1, visibility: C, buffer: { type: 'storage' } },
        ...[2, 3, 4].map((b) => ({ binding: b, visibility: C, buffer: { type: 'read-only-storage' as const } })),
      ],
    });
    const argsLayout = d.createBindGroupLayout({
      label: 'args',
      entries: [{ binding: 0, visibility: C, buffer: { type: 'storage' } }],
    });
    const simPL = d.createPipelineLayout({ bindGroupLayouts: [simLayout] });
    const simArgsPL = d.createPipelineLayout({ bindGroupLayouts: [simLayout, argsLayout] });
    await Promise.all(
      ENTRY_POINTS.map(async (e) => {
        const layout = ARGS_WRITERS.has(e) ? simArgsPL : simPL;
        this.pipes[e] = await d.createComputePipelineAsync({ label: e, layout, compute: { module: simMod, entryPoint: e } });
      }),
    );
    this.argsGroup = d.createBindGroup({ layout: argsLayout, entries: [{ binding: 0, resource: { buffer: gw.argsBuf } }] });
    this.player = await d.createComputePipelineAsync({
      label: 'player',
      layout: d.createPipelineLayout({ bindGroupLayouts: [playerLayout] }),
      compute: { module: playerMod, entryPoint: 'updatePlayer' },
    });

    this.simGroup = d.createBindGroup({
      layout: simLayout,
      entries: [
        { binding: 0, resource: { buffer: gw.uniformBuf } },
        { binding: 1, resource: { buffer: gw.cameraBuf } },
        { binding: 2, resource: { buffer: gw.gridBuf } },
        { binding: 3, resource: { buffer: gw.poolBuf } },
        { binding: 4, resource: { buffer: gw.superBuf } },
        { binding: 5, resource: { buffer: gw.matBuf } },
        { binding: 6, resource: { buffer: gw.ctrlBuf } },
        { binding: 7, resource: { buffer: gw.heapBuf } },
        { binding: 8, resource: { buffer: gw.particleBuf } },
        { binding: 10, resource: gw.densityTex.createView() },
      ],
    });
    this.playerGroup = d.createBindGroup({
      layout: playerLayout,
      entries: [
        { binding: 0, resource: { buffer: gw.uniformBuf } },
        { binding: 1, resource: { buffer: gw.cameraBuf } },
        { binding: 2, resource: { buffer: gw.gridBuf } },
        { binding: 3, resource: { buffer: gw.poolBuf } },
        { binding: 4, resource: { buffer: gw.matBuf } },
      ],
    });
  }

  /** Record one simulation step. The uniform block must already hold this frame's input. */
  encode(enc: GPUCommandEncoder): void {
    const ctrl = this.gw.ctrlBuf;
    const argsBuf = this.gw.argsBuf;
    const pass = enc.beginComputePass({ label: 'simulation' });
    pass.setPipeline(this.player);
    pass.setBindGroup(0, this.playerGroup);
    pass.dispatchWorkgroups(1);

    pass.setBindGroup(0, this.simGroup);
    pass.setBindGroup(1, this.argsGroup); // ignored by pipelines that don't declare group 1
    const one = (e: Entry) => {
      pass.setPipeline(this.pipes[e]);
      pass.dispatchWorkgroups(1);
    };
    const indirect = (e: Entry, args: number) => {
      pass.setPipeline(this.pipes[e]);
      pass.dispatchWorkgroupsIndirect(argsBuf, args * 4);
    };
    const alloc = () => {
      one('allocArgs');
      indirect('allocBricks', ARGS.A_ALLOC);
    };

    one('frameBegin');
    one('shoot');
    // Impacts -> carve.
    one('impactArgs');
    indirect('impactJobs', ARGS.A_IMPACT);
    alloc();
    one('carveArgs');
    indirect('carve', ARGS.A_CARVE);
    one('postCarve');
    // Glass panes crack outwards a few rings per frame.
    for (let i = 0; i < LIMITS.floodItersPerFrame; i++) {
      one('floodArgs');
      indirect('floodStep', ARGS.A_FLOOD);
    }
    // Island search, resolved whenever its frontier runs dry.
    one('searchBegin');
    indirect('searchClear', ARGS.A_CLEAR);
    indirect('searchSeed', ARGS.A_SEEDS);
    for (let i = 0; i < LIMITS.searchItersPerFrame; i++) {
      one('searchArgs');
      indirect('searchStep', ARGS.A_SEARCH);
    }
    one('resolveArgs');
    indirect('resolveFrontier', ARGS.A_R0);
    indirect('resolveRoots', ARGS.A_RESOLVE);
    indirect('resolveChunks', ARGS.A_RESOLVE);
    indirect('resolveExpand', ARGS.A_RESOLVE);
    alloc();
    indirect('resolveExtract', ARGS.A_RESOLVE);
    one('resolveAnchors');
    one('resolveEnd');
    // Falling chunks.
    one('chunkArgs');
    one('chunkBegin');
    indirect('chunkCollide', ARGS.A_ARENA);
    one('chunkStep');
    indirect('chunkLand', ARGS.A_ARENA);
    // Debris.
    pass.setPipeline(this.pipes.particleSim);
    pass.dispatchWorkgroups(Math.ceil(MAX_PARTICLES / 64));
    // Settled debris + landed chunks become voxels again.
    one('writeArgs');
    indirect('writeRequest', ARGS.A_WRITE);
    alloc();
    indirect('writeApply', ARGS.A_WRITE);
    // Housekeeping: density for AO, brick compaction, free list.
    one('maintArgs');
    indirect('maint', ARGS.A_MAINT);
    one('lights');
    one('frameEnd');
    pass.end();

    // A few hundred bytes of counters for the HUD; nothing else crosses the bus.
    if (!this.readbackPending) {
      enc.copyBufferToBuffer(ctrl, 0, this.readback, 0, CTRL_WORDS * 4);
      enc.copyBufferToBuffer(argsBuf, 0, this.readback, CTRL_WORDS * 4, ARGS_WORDS * 4);
      enc.copyBufferToBuffer(this.gw.cameraBuf, 0, this.readback, (CTRL_WORDS + ARGS_WORDS) * 4, 256);
      enc.clearBuffer(ctrl, CTRL.S_BILL * 4, 8); // bill + destroyed are deltas
      this.readbackPending = true;
      this.wantMap = true;
    }
  }

  private wantMap = false;

  /** Call after queue.submit(): maps the stats copy made by encode(). */
  afterSubmit(): void {
    if (!this.wantMap) return;
    this.wantMap = false;
    this.readback.mapAsync(GPUMapMode.READ).then(
      () => {
        const u = new Uint32Array(this.readback.getMappedRange().slice(0));
        this.readback.unmap();
        this.readbackPending = false;
        const f = new Float32Array(u.buffer);
        const cam = CTRL_WORDS + ARGS_WORDS;
        this.stats = {
          bill: u[CTRL.S_BILL] / 10,
          destroyed: u[CTRL.S_DESTROYED],
          falling: u[CTRL.S_FALLING],
          particles: u[CTRL_WORDS + ARGS.A_DRAWP + 1],
          liveBricks: this.gw.poolCapacity - u[CTRL.C_FREE_TOP],
          allocFailures: u[CTRL.S_ALLOC_FAIL],
          searches: u[CTRL.S_SEARCHES],
          islands: u[CTRL.S_ISLANDS],
          searching: u[CTRL.C_S_STATE] !== 0,
          flying: f[cam + 35] > 0.5,
          eye: [f[cam + 32], f[cam + 33], f[cam + 34]],
        };
        this.onStats?.(this.stats);
      },
      () => {
        this.readbackPending = false;
      },
    );
  }
}

