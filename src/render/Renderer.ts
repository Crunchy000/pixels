import { ARGS, layoutWGSL } from '../gpu/layout';
import { GpuWorld } from '../gpu/GpuWorld';
import { compileModule } from '../gpu/Simulation';
import { MAX_PARTICLES } from '../config';
import sharedWGSL from './shaders/shared.wgsl?raw';
import renderCommonWGSL from './shaders/render_common.wgsl?raw';
import raymarchWGSL from './shaders/raymarch.wgsl?raw';
import particleDrawWGSL from './shaders/particle_draw.wgsl?raw';
import postWGSL from './shaders/post.wgsl?raw';

const BLOOM_LEVELS = 6;

/**
 * Draws the GPU-resident world: one ray-march pass, debris + falling-chunk
 * cubes (indirect draws sized by the simulation), bloom and tonemapping.
 */
export class Renderer {
  readonly device: GPUDevice;
  readonly context: GPUCanvasContext;
  readonly format: GPUTextureFormat;
  private gw: GpuWorld;

  private linearSampler: GPUSampler;
  private nearestSampler: GPUSampler;
  private group0!: GPUBindGroup;
  private drawGroup!: GPUBindGroup;
  private raymarchPipeline!: GPURenderPipeline;
  private particlePipeline!: GPURenderPipeline;
  private chunkPipeline!: GPURenderPipeline;
  private brightPipeline!: GPURenderPipeline;
  private downPipeline!: GPURenderPipeline;
  private compositePipeline!: GPURenderPipeline;

  // Size-dependent resources.
  width = 0;
  height = 0;
  private sceneColor!: GPUTexture;
  private sceneDepth!: GPUTexture;
  private bloomTex!: GPUTexture;
  private bloomPasses: { view: GPUTextureView; group: GPUBindGroup; buf: GPUBuffer; bright: boolean }[] = [];
  private compositeGroup!: GPUBindGroup;
  private compositeBuf!: GPUBuffer;
  pixelated = false;
  private pixelatedBound = false;

  constructor(gw: GpuWorld, context: GPUCanvasContext, format: GPUTextureFormat) {
    this.gw = gw;
    this.device = gw.device;
    this.context = context;
    this.format = format;
    const d = this.device;
    this.linearSampler = d.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge' });
    this.nearestSampler = d.createSampler({ magFilter: 'nearest', minFilter: 'nearest', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
  }

  async init(): Promise<void> {
    const d = this.device;
    const gw = this.gw;
    const prelude = sharedWGSL + renderCommonWGSL;
    const consts = layoutWGSL(gw.heap, { POOL_CAP: gw.poolCapacity, MAX_PARTICLES, NUM_LIGHTS: gw.numLights, NUM_ANCHORS: gw.numAnchors });
    const rayMod = await compileModule(d, 'raymarch', prelude + raymarchWGSL);
    const drawMod = await compileModule(d, 'debris-draw', consts + prelude + particleDrawWGSL);
    const postMod = await compileModule(d, 'post', postWGSL);

    const gfx = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
    const ro = { type: 'read-only-storage' as const };
    const group0Layout = d.createBindGroupLayout({
      label: 'render',
      entries: [
        { binding: 0, visibility: gfx, buffer: { type: 'uniform' } },
        { binding: 1, visibility: gfx, buffer: ro },
        { binding: 2, visibility: gfx, buffer: ro },
        { binding: 3, visibility: gfx, buffer: ro },
        { binding: 4, visibility: gfx, buffer: ro },
        { binding: 5, visibility: gfx, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 6, visibility: gfx, sampler: { type: 'filtering' } },
        { binding: 7, visibility: gfx, buffer: ro },
        { binding: 8, visibility: gfx, buffer: { type: 'uniform' } },
      ],
    });
    const drawLayout = d.createBindGroupLayout({
      label: 'debris',
      entries: [0, 1].map((b) => ({ binding: b, visibility: GPUShaderStage.VERTEX, buffer: ro })),
    });

    this.raymarchPipeline = await d.createRenderPipelineAsync({
      layout: d.createPipelineLayout({ bindGroupLayouts: [group0Layout] }),
      vertex: { module: rayMod, entryPoint: 'vsMain' },
      fragment: { module: rayMod, entryPoint: 'fsMain', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'always' },
    });
    const cubes = (entryPoint: string) =>
      d.createRenderPipelineAsync({
        layout: d.createPipelineLayout({ bindGroupLayouts: [group0Layout, drawLayout] }),
        vertex: { module: drawMod, entryPoint },
        fragment: { module: drawMod, entryPoint: 'fsParticle', targets: [{ format: 'rgba16float' }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
      });
    this.particlePipeline = await cubes('vsParticle');
    this.chunkPipeline = await cubes('vsChunk');
    const post = (entryPoint: string, format: GPUTextureFormat) =>
      d.createRenderPipelineAsync({
        layout: 'auto',
        vertex: { module: postMod, entryPoint: 'vsFull' },
        fragment: { module: postMod, entryPoint, targets: [{ format }] },
      });
    this.brightPipeline = await post('fsBright', 'rgba16float');
    this.downPipeline = await post('fsDown', 'rgba16float');
    this.compositePipeline = await post('fsComposite', this.format);

    this.group0 = d.createBindGroup({
      layout: group0Layout,
      entries: [
        { binding: 0, resource: { buffer: gw.uniformBuf } },
        { binding: 1, resource: { buffer: gw.gridBuf } },
        { binding: 2, resource: { buffer: gw.poolBuf } },
        { binding: 3, resource: { buffer: gw.matBuf } },
        { binding: 4, resource: { buffer: gw.heapBuf, ...gw.lightsRange } },
        { binding: 5, resource: gw.densityTex.createView() },
        { binding: 6, resource: this.linearSampler },
        { binding: 7, resource: { buffer: gw.superBuf } },
        { binding: 8, resource: { buffer: gw.cameraBuf } },
      ],
    });
    this.drawGroup = d.createBindGroup({
      layout: drawLayout,
      entries: [
        { binding: 0, resource: { buffer: gw.particleBuf } },
        { binding: 1, resource: { buffer: gw.heapBuf } },
      ],
    });
  }

  /** (Re)create the internal render targets at the given resolution. */
  ensureTargets(rw: number, rh: number): void {
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
        this.bloomPasses.push({ view: this.bloomTex.createView({ baseMipLevel: l, mipLevelCount: 1 }), group, buf, bright });
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

  /** Record the frame's render passes (call ensureTargets first). */
  encode(enc: GPUCommandEncoder, exposure: number, bloom: number): void {
    const d = this.device;
    d.queue.writeBuffer(this.compositeBuf, 0, new Float32Array([0, 0, 0, 0, exposure, bloom, 0.35, this.format.endsWith('-srgb') ? 0 : 1]));
    {
      const pass = enc.beginRenderPass({
        label: 'scene',
        colorAttachments: [{ view: this.sceneColor.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
        depthStencilAttachment: { view: this.sceneDepth.createView(), depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1 },
      });
      pass.setBindGroup(0, this.group0);
      pass.setPipeline(this.raymarchPipeline);
      pass.draw(3);
      pass.setBindGroup(1, this.drawGroup);
      pass.setPipeline(this.particlePipeline);
      pass.drawIndirect(this.gw.argsBuf, ARGS.A_DRAWP * 4);
      pass.setPipeline(this.chunkPipeline);
      pass.drawIndirect(this.gw.argsBuf, ARGS.A_DRAWC * 4);
      pass.end();
    }
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
  }
}
