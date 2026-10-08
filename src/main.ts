import { BRICK_VOXELS, DIMS, INITIAL_RENDER_SCALE, MAX_PARTICLES, TARGET_FRAME_MS, VOXEL_SIZE, VOXELS_PER_METER } from './config';
import { GpuWorld, UNIFORM_FLOATS } from './gpu/GpuWorld';
import { heapLayout } from './gpu/layout';
import { Simulation } from './gpu/Simulation';
import { Renderer } from './render/Renderer';
import { buildPenthouse } from './scene/penthouse';
import { TOOLS } from './sim/tools';
import { Input } from './ui/Input';
import { initOpaqueTable, VoxelWorld } from './world/VoxelWorld';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const params = new URLSearchParams(location.search);
/** ?debug=1 normals, 2 albedo, 3 distance, 4 material id, ... (see raymarch.wgsl). */
const DEBUG_VIEW = Number(params.get('debug') ?? 0) || 0;
const EYE_HEIGHT = 1.62;
const FOV_Y = (72 * Math.PI) / 180;

async function main() {
  const canvas = $<HTMLCanvasElement>('gfx');
  const loadBar = $('load-bar');
  const loadMsg = $('load-msg');
  const isTouch = matchMedia('(pointer: coarse)').matches;
  if (isTouch) document.body.classList.add('is-touch');

  if (!navigator.gpu) throw new Error('This browser does not support WebGPU.\nTry a recent Chrome, Edge, or Safari 26+, or Firefox Nightly.');

  // ---- Build the scene on the CPU once ------------------------------------
  initOpaqueTable();
  const world = new VoxelWorld(DIMS.x, DIMS.y, DIMS.z, 1 << 15);
  const t0 = performance.now();
  const scene = await buildPenthouse(world, (msg, f) => {
    loadMsg.textContent = msg + '…';
    loadBar.style.width = `${Math.round(f * 85)}%`;
  });
  const buildMs = performance.now() - t0;
  loadMsg.textContent = 'Uploading to the GPU…';

  // ---- GPU: device sized for the world, then the world moves over ----------
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter found.');
  const poolGuess = Math.max(world.poolUsed + 98304, world.poolUsed * 2);
  const heapGuess = heapLayout(world.grid.length, poolGuess, MAX_PARTICLES).words * 4;
  const want = Math.max(poolGuess * BRICK_VOXELS * 2, heapGuess, 256 << 20);
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, want),
      maxBufferSize: Math.min(adapter.limits.maxBufferSize, want),
    },
  });
  let gpuError: string | null = null;
  device.addEventListener('uncapturederror', (e) => {
    const msg = (e as GPUUncapturedErrorEvent).error.message;
    console.error(msg);
    gpuError = msg;
  });
  device.lost.then((info) => (gpuError = `GPU device lost: ${info.message}`));
  const context = canvas.getContext('webgpu')!;
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  const startBricks = world.poolUsed;
  const gw = new GpuWorld(device, world, scene); // uploads, then frees the CPU copy
  loadMsg.textContent = 'Compiling shaders…';
  const sim = new Simulation(gw);
  const renderer = new Renderer(gw, context, format);
  await Promise.all([sim.init(), renderer.init()]);
  loadBar.style.width = '100%';
  const input = new Input(canvas);

  // Edge-triggered requests to the GPU-side player (counters, see player.wgsl).
  let flyCounter = 0;
  let teleportCounter = 0;
  let teleport: [number, number, number, number, number] = [0, 0, 0, 0, 0];
  const requestTeleport = (feet: [number, number, number], yaw: number, pitch: number) => {
    teleport = [feet[0], feet[1], feet[2], yaw, pitch];
    teleportCounter++;
  };
  // ?cam=x,y,z,yaw,pitch puts the (flying) camera somewhere specific.
  const camParam = params.get('cam');
  if (camParam) {
    const [x, y, z, yaw, pitch] = camParam.split(',').map(Number);
    requestTeleport([x, y - EYE_HEIGHT, z], yaw ?? scene.spawn.yaw, pitch ?? 0);
    flyCounter++;
  }

  // ---- UI -----------------------------------------------------------------
  $('loading').style.display = 'none';
  const help = $('help');
  const showHelp = (on: boolean) => (help.style.display = on ? 'block' : 'none');
  showHelp(true);
  $('go').addEventListener('click', () => {
    showHelp(false);
    if (!isTouch) canvas.requestPointerLock?.();
  });
  document.addEventListener('pointerlockchange', () => {
    if (document.pointerLockElement === canvas) showHelp(false);
  });

  let toolIndex = 0;
  const toolsEl = $('tools');
  const toolButtons = TOOLS.map((t, i) => {
    const b = document.createElement('button');
    b.innerHTML = `<kbd>${t.key}</kbd>${t.name}`;
    b.addEventListener('click', (e) => {
      e.stopPropagation();
      toolIndex = i;
      refreshTools();
    });
    toolsEl.appendChild(b);
    return b;
  });
  const refreshTools = () => toolButtons.forEach((b, i) => b.classList.toggle('active', i === toolIndex));
  refreshTools();

  let touchJump = false;
  const touchBtn = (name: string, down: () => void, up?: () => void) => {
    const el = document.querySelector(`[data-touch-button="${name}"]`) as HTMLElement;
    el.addEventListener('touchstart', (e) => { e.preventDefault(); down(); }, { passive: false });
    el.addEventListener('touchend', (e) => { e.preventDefault(); up?.(); }, { passive: false });
  };
  touchBtn('fire', () => (input.touchFire = true), () => (input.touchFire = false));
  touchBtn('jump', () => (touchJump = true));
  touchBtn('tool', () => { toolIndex = (toolIndex + 1) % TOOLS.length; refreshTools(); });

  // ---- Settings -------------------------------------------------------------
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  let renderScale = Math.min(1, INITIAL_RENDER_SCALE / dpr) * (isTouch ? 0.6 : 1);
  let autoRes = TARGET_FRAME_MS > 0 && !params.has('scale');
  let reflections = !isTouch;
  let lightShadows = true;
  let night = params.has('night') ? 1 : 0;
  let nightTarget = night;

  const resize = () => {
    canvas.width = Math.max(1, Math.round(canvas.clientWidth * dpr));
    canvas.height = Math.max(1, Math.round(canvas.clientHeight * dpr));
  };
  resize();
  window.addEventListener('resize', resize);

  // ---- Stats (the only thing read back from the GPU) ------------------------
  let bill = 0;
  let destroyed = 0;
  sim.onStats = (s) => {
    bill += s.bill;
    destroyed += s.destroyed;
  };

  // ---- Loop -----------------------------------------------------------------
  let last = performance.now();
  let smoothedMs = 16;
  let fireCooldown = 0;
  let hudTimer = 0.25;
  let frames = 0;
  let fpsAccum = 0;
  let fps = 0;
  let pendingFire = -1;
  let shotSeed = 1;
  const statsEl = $('stats');
  const billEl = $('bill-amount');
  const errEl = $('error');
  const money = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });
  const u = gw.uniform;
  const sun = normalize([0.55, 0.2, 0.8]);

  // ?frames=N renders N frames then stops (used for headless screenshots).
  let maxFrames = Number(params.get('frames')) || Infinity;
  let frameCount = 0;
  let fixedDt: number | null = null;
  const frame = (now: number) => {
    if (frameCount >= maxFrames) {
      requestAnimationFrame(frame);
      return;
    }
    frameCount++;
    const dt = fixedDt ?? Math.max(0, Math.min(0.05, (now - last) / 1000));
    last = now;
    smoothedMs += (dt * 1000 - smoothedMs) * 0.05;

    // Keys.
    if (input.hit('KeyH')) showHelp(help.style.display === 'none');
    if (input.hit('KeyF')) flyCounter++;
    if (input.hit('KeyN')) nightTarget = nightTarget > 0.5 ? 0 : 1;
    if (input.hit('KeyR')) reflections = !reflections;
    if (input.hit('KeyL')) lightShadows = !lightShadows;
    if (input.hit('KeyP')) renderer.pixelated = !renderer.pixelated;
    if (input.hit('BracketLeft')) { renderScale = Math.max(0.2, renderScale / 1.15); autoRes = false; }
    if (input.hit('BracketRight')) { renderScale = Math.min(2, renderScale * 1.15); autoRes = false; }
    TOOLS.forEach((t, i) => { if (input.hit(`Digit${t.key}`)) { toolIndex = i; refreshTools(); } });
    if (input.wheel) {
      toolIndex = (toolIndex + (input.wheel > 0 ? 1 : TOOLS.length - 1)) % TOOLS.length;
      refreshTools();
    }

    // Firing: the CPU only decides when; the GPU casts the ray and does the rest.
    fireCooldown -= dt;
    let shot = pendingFire >= 0 ? TOOLS[pendingFire] : null;
    pendingFire = -1;
    const trigger = input.mouseDown || input.touchFire;
    if (!shot && trigger && fireCooldown <= 0) {
      shot = TOOLS[toolIndex];
      fireCooldown = 1 / shot.rate;
      if (!shot.auto) { input.mouseDown = false; input.touchFire = false; }
    }
    night += (nightTarget - night) * Math.min(1, dt * 2);

    // Dynamic resolution.
    if (autoRes) {
      if (smoothedMs > TARGET_FRAME_MS * 1.2) renderScale = Math.max(0.25, renderScale * 0.985);
      else if (smoothedMs < TARGET_FRAME_MS * 0.75) renderScale = Math.min(1, renderScale * 1.004);
    }
    renderer.ensureTargets(canvas.width * renderScale, canvas.height * renderScale);
    const rw = renderer.width, rh = renderer.height;

    // This frame's uniform block (input + settings): the whole CPU -> GPU traffic.
    const k = (c: string) => (input.down(c) ? 1 : 0);
    const sunUp = 1 - night;
    u.set([DIMS.x, DIMS.y, DIMS.z, VOXEL_SIZE], 0);
    u.set([sun[0], sun[1], sun[2], 3.2 * sunUp], 4);
    u.set([1.0, 0.6, 0.34, 0], 8);
    u.set([0.16 * sunUp + 0.035, 0.13 * sunUp + 0.03, 0.24 * sunUp + 0.07, 1.1], 12);
    u.set([rw, rh, 1 / rw, 1 / rh], 16);
    u.set([reflections ? 1 : 0, lightShadows ? 1 : 0, 1.0 + night * 0.4, DEBUG_VIEW], 20);
    u.set([Math.min(dt, 1 / 20), 9.81 / VOXEL_SIZE, MAX_PARTICLES, night], 24);
    u.set([now / 1000, frameCount, canvas.width / canvas.height, Math.tan(FOV_Y / 2)], 28);
    u.set([
      clamp1(k('KeyW') + k('ArrowUp') - k('KeyS') - k('ArrowDown') + input.moveY),
      clamp1(k('KeyD') + k('ArrowRight') - k('KeyA') - k('ArrowLeft') + input.moveX),
      clamp1(k('Space') - k('KeyC') - k('ControlLeft')),
      input.down('Space') || touchJump ? 1 : 0,
    ], 32);
    u.set([input.lookX, input.lookY, input.down('ShiftLeft') || input.down('ShiftRight') ? 1 : 0, flyCounter], 36);
    if (shot) u.set([shot.pellets, shot.radius / VOXEL_SIZE, shot.fragileRadius / VOXEL_SIZE, shot.impulse / VOXEL_SIZE], 40);
    else u.set([0, 0, 0, 0], 40);
    u.set([shot ? shot.spread : 0, shotSeed++, teleportCounter, teleport[4]], 44);
    u.set([teleport[0], teleport[1], teleport[2], teleport[3]], 48);
    device.queue.writeBuffer(gw.uniformBuf, 0, u, 0, UNIFORM_FLOATS);
    touchJump = false;

    const enc = device.createCommandEncoder();
    sim.encode(enc);
    renderer.encode(enc, 1.0 + night * 0.4, 0.9);
    device.queue.submit([enc.finish()]);
    sim.afterSubmit();
    input.endFrame();

    // HUD.
    frames++;
    fpsAccum += dt;
    hudTimer -= dt;
    if (hudTimer <= 0) {
      hudTimer = 0.25;
      fps = frames / Math.max(1e-3, fpsAccum);
      frames = 0;
      fpsAccum = 0;
      const s = sim.stats;
      const mb = (s.liveBricks * 1024) / (1 << 20);
      statsEl.textContent =
        `${fps.toFixed(0).padStart(3)} fps  ${rw}×${rh}${autoRes ? ' auto' : ''}\n` +
        `voxel ${(VOXEL_SIZE * 100).toFixed(2)} cm  ${DIMS.x}×${DIMS.y}×${DIMS.z}\n` +
        `bricks ${s.liveBricks.toLocaleString()} (${mb.toFixed(0)} MB, GPU only)\n` +
        `smashed ${destroyed.toLocaleString()} vox  falling ${s.falling}\n` +
        `debris ${s.particles.toLocaleString()}${s.searching ? '  checking supports…' : ''}\n` +
        `${TOOLS[toolIndex].name}${s.flying ? '  [fly]' : ''}${night > 0.5 ? '  [night]' : ''}`;
      billEl.textContent = money.format(bill);
      if (gpuError) {
        errEl.style.display = 'block';
        $('error-msg').textContent = gpuError;
      }
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
  console.info(`World ${DIMS.x}x${DIMS.y}x${DIMS.z} @ ${VOXELS_PER_METER}/m built in ${buildMs.toFixed(0)} ms, ${startBricks} bricks; GPU pool ${gw.poolCapacity} bricks`);

  // Test hooks for headless runs.
  (window as unknown as { __pixels: unknown }).__pixels = {
    gw, sim, renderer, buildMs, TOOLS,
    get frameCount() { return frameCount; },
    get stats() { return { ...sim.stats, bill, destroyed }; },
    get error() { return gpuError; },
    setMaxFrames: (n: number) => { maxFrames = n; },
    setScale: (s: number) => { renderScale = s; autoRes = false; },
    setFixedDt: (s: number) => { fixedDt = s; },
    /** Fire tool i on the next frame. */
    fire: (i: number) => { pendingFire = i; },
    /** Put the eye at `eye` looking at `target` (metres). */
    look: (eye: [number, number, number], target: [number, number, number]) => {
      const dx = target[0] - eye[0], dy = target[1] - eye[1], dz = target[2] - eye[2];
      requestTeleport([eye[0], eye[1] - EYE_HEIGHT, eye[2]], Math.atan2(dx, -dz), Math.atan2(dy, Math.hypot(dx, dz)));
    },
    toggleFly: () => { flyCounter++; },
  };
}

function clamp1(v: number): number {
  return Math.max(-1, Math.min(1, v));
}

function normalize(v: [number, number, number]): [number, number, number] {
  const l = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / l, v[1] / l, v[2] / l];
}

main().catch((e) => {
  console.error(e);
  document.getElementById('loading')!.style.display = 'none';
  const el = document.getElementById('error')!;
  el.style.display = 'block';
  document.getElementById('error-msg')!.textContent = String(e?.message ?? e);
});
