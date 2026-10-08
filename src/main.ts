import { DIMS, INITIAL_RENDER_SCALE, TARGET_FRAME_MS, VOXEL_SIZE, VOXELS_PER_METER } from './config';
import { invert, multiply, perspective, viewFromYawPitch } from './render/math';
import { Renderer } from './render/Renderer';
import { buildPenthouse } from './scene/penthouse';
import { Destruction, TOOLS } from './sim/Destruction';
import { LightSystem } from './sim/LightSystem';
import { Player } from './sim/Player';
import { Input } from './ui/Input';
import { initOpaqueTable, VoxelWorld } from './world/VoxelWorld';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

async function main() {
  const canvas = $<HTMLCanvasElement>('gfx');
  const loadBar = $('load-bar');
  const loadMsg = $('load-msg');
  const isTouch = matchMedia('(pointer: coarse)').matches;
  if (isTouch) document.body.classList.add('is-touch');

  if (!navigator.gpu) throw new Error('This browser does not support WebGPU.\nTry a recent Chrome, Edge, or Safari 26+, or Firefox Nightly.');

  initOpaqueTable();
  const world = new VoxelWorld(DIMS.x, DIMS.y, DIMS.z, 1 << 15);
  const t0 = performance.now();
  const scene = await buildPenthouse(world, (msg, f) => {
    loadMsg.textContent = msg + '…';
    loadBar.style.width = `${Math.round(f * 90)}%`;
  });
  const buildMs = performance.now() - t0;
  loadMsg.textContent = 'Compiling shaders…';

  const renderer = new Renderer(canvas, world);
  await renderer.init();
  loadBar.style.width = '100%';

  const lights = new LightSystem(world, scene.lights);
  const destruction = new Destruction(world, renderer, lights);
  const player = new Player(world, scene.spawn.pos, scene.spawn.yaw, scene.spawn.pitch);
  const input = new Input(canvas);
  // ?cam=x,y,z,yaw,pitch puts the (flying) camera somewhere specific.
  const camParam = new URLSearchParams(location.search).get('cam');
  if (camParam) {
    const [x, y, z, yaw, pitch] = camParam.split(',').map(Number);
    player.fly = true;
    player.pos = [x, y - 1.62, z];
    player.yaw = yaw ?? player.yaw;
    player.pitch = pitch ?? 0;
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
  let autoRes = TARGET_FRAME_MS > 0 && !new URLSearchParams(location.search).has('scale');
  let reflections = !isTouch;
  let lightShadows = true;
  let night = new URLSearchParams(location.search).has('night') ? 1 : 0;
  let nightTarget = night;

  const resize = () => {
    canvas.width = Math.max(1, Math.round(canvas.clientWidth * dpr));
    canvas.height = Math.max(1, Math.round(canvas.clientHeight * dpr));
  };
  resize();
  window.addEventListener('resize', resize);

  // ---- Loop -----------------------------------------------------------------
  let last = performance.now();
  let smoothedMs = 16;
  let fireCooldown = 0;
  let hudTimer = 0.25;
  let frames = 0;
  let fpsAccum = 0;
  let fps = 0;
  const statsEl = $('stats');
  const billEl = $('bill-amount');
  const errEl = $('error');
  const money = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });

  // ?frames=N renders N frames then stops (used for headless screenshots).
  let maxFrames = Number(new URLSearchParams(location.search).get('frames')) || Infinity;
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
    const time = now / 1000;
    smoothedMs += ((dt * 1000) - smoothedMs) * 0.05;

    // Keys.
    if (input.hit('KeyH')) showHelp(help.style.display === 'none');
    if (input.hit('KeyF')) player.fly = !player.fly;
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

    // Look + move.
    player.yaw += input.lookX;
    player.pitch = Math.max(-1.5, Math.min(1.5, player.pitch - input.lookY));
    const k = (c: string) => (input.down(c) ? 1 : 0);
    player.update(dt, {
      forward: k('KeyW') + k('ArrowUp') - k('KeyS') - k('ArrowDown') + input.moveY,
      right: k('KeyD') + k('ArrowRight') - k('KeyA') - k('ArrowLeft') + input.moveX,
      up: k('Space') - k('KeyC') - k('ControlLeft'),
      jump: input.down('Space') || touchJump,
      sprint: input.down('ShiftLeft') || input.down('ShiftRight'),
    });
    touchJump = false;

    // Fire.
    const tool = TOOLS[toolIndex];
    fireCooldown -= dt;
    const trigger = input.mouseDown || input.touchFire;
    if (trigger && fireCooldown <= 0) {
      destruction.fire(tool, player.eye, player.look);
      fireCooldown = 1 / tool.rate;
      if (!tool.auto) { input.mouseDown = false; input.touchFire = false; }
    }

    destruction.update(dt);
    lights.update(dt);
    night += (nightTarget - night) * Math.min(1, dt * 2);

    // Camera.
    const aspect = canvas.width / canvas.height;
    const proj = perspective((72 * Math.PI) / 180, aspect, 0.03, 200);
    const view = viewFromYawPitch(player.eye, player.yaw, player.pitch);
    const viewProj = multiply(proj, view);

    // Dynamic resolution.
    if (autoRes) {
      if (smoothedMs > TARGET_FRAME_MS * 1.2) renderScale = Math.max(0.25, renderScale * 0.985);
      else if (smoothedMs < TARGET_FRAME_MS * 0.75) renderScale = Math.min(1, renderScale * 1.004);
    }
    const rw = Math.max(1, Math.round(canvas.width * renderScale));
    const rh = Math.max(1, Math.round(canvas.height * renderScale));

    const sunUp = 1 - night;
    renderer.render(
      {
        viewProj,
        invViewProj: invert(viewProj),
        camPos: player.eye,
        time,
        dt,
        sunDir: normalize([0.55, 0.2, 0.8]),
        sunIntensity: 3.2 * sunUp,
        sunColor: [1.0, 0.6, 0.34],
        ambient: [0.16 * sunUp + 0.035, 0.13 * sunUp + 0.03, 0.24 * sunUp + 0.07],
        night,
        lights: lights.frameLights(time),
        reflections,
        lightShadows,
        exposure: 1.0 + night * 0.4,
        bloom: 0.9,
      },
      rw,
      rh,
    );
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
      const mb = (world.liveBricks * 1024) / (1 << 20);
      statsEl.textContent =
        `${fps.toFixed(0).padStart(3)} fps  ${rw}×${rh}${autoRes ? ' auto' : ''}\n` +
        `voxel ${(VOXEL_SIZE * 100).toFixed(2)} cm  ${DIMS.x}×${DIMS.y}×${DIMS.z}\n` +
        `bricks ${world.liveBricks.toLocaleString()} (${mb.toFixed(0)} MB)\n` +
        `smashed ${destruction.destroyed.toLocaleString()} vox  falling ${destruction.fallingCount}\n` +
        `${tool.name}${player.fly ? '  [fly]' : ''}${night > 0.5 ? '  [night]' : ''}`;
      billEl.textContent = money.format(destruction.bill);
      if (renderer.lastError) {
        errEl.style.display = 'block';
        $('error-msg').textContent = renderer.lastError;
      }
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
  console.info(`World ${DIMS.x}x${DIMS.y}x${DIMS.z} @ ${VOXELS_PER_METER}/m built in ${buildMs.toFixed(0)} ms, ${world.liveBricks} bricks`);
  (window as unknown as { __pixels: unknown }).__pixels = {
    world, renderer, destruction, player, lights, buildMs, TOOLS,
    get frameCount() { return frameCount; },
    /** Test hooks for headless runs. */
    setMaxFrames: (n: number) => { maxFrames = n; },
    setScale: (s: number) => { renderScale = s; autoRes = false; },
    setFixedDt: (s: number) => { fixedDt = s; },
    /** Test hook: fire tool i from the current camera. */
    fire: (i: number) => destruction.fire(TOOLS[i], player.eye, player.look),
  };
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
