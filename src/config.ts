// Global tunables. Most can be overridden with URL params, e.g. ?vpm=100&scale=0.5

const params = new URLSearchParams(location.search);

function num(name: string, fallback: number, min: number, max: number): number {
  const v = Number(params.get(name));
  return Number.isFinite(v) && params.has(name) ? Math.min(max, Math.max(min, v)) : fallback;
}

/** Voxels per metre. 80 => 1.25 cm voxels, which lands at ~1 voxel per pixel
 *  for a few metres' distance at 1080p. Raise it for finer voxels (more memory). */
export const VOXELS_PER_METER = num('vpm', 80, 16, 160);
export const VOXEL_SIZE = 1 / VOXELS_PER_METER;

/** Brick edge length in voxels. The GPU structure is a grid of 8^3 bricks. */
export const BRICK = 8;
export const BRICK_VOXELS = BRICK * BRICK * BRICK;

/** Size of the playable volume in metres (x = east, y = up, z = north). */
export const WORLD_METERS = { x: 16, y: 4.4, z: 12 };

function roundToBrick(m: number): number {
  return Math.ceil((m * VOXELS_PER_METER) / BRICK) * BRICK;
}

export const DIMS = {
  x: roundToBrick(WORLD_METERS.x),
  y: roundToBrick(WORLD_METERS.y),
  z: roundToBrick(WORLD_METERS.z),
};

/** Initial internal render scale relative to the canvas' CSS size (1 = one ray per CSS pixel). */
export const INITIAL_RENDER_SCALE = num('scale', 1, 0.2, 2);
/** Dynamic resolution keeps the frame time near this target (ms). 0 disables it. */
export const TARGET_FRAME_MS = num('target', 1000 / 50, 0, 100);

export const MAX_PARTICLES = Math.round(num('particles', 1 << 18, 1 << 12, 1 << 21));
export const MAX_LIGHTS = 32;

export const GRAVITY = 9.81; // m/s^2
