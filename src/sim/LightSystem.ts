import { VOXEL_SIZE } from '../config';
import type { Light } from '../render/Renderer';
import type { SceneLight } from '../scene/penthouse';
import { mat } from '../world/materials';
import { VoxelWorld } from '../world/VoxelWorld';

interface Tracked {
  base: SceneLight;
  anchors: { x: number; y: number; z: number; y0: number }[];
  factor: number;
  target: number;
  offsetY: number;
  seed: number;
}

interface Flash {
  pos: [number, number, number];
  life: number;
  max: number;
  size: number;
}

/**
 * Scene lights tied to emissive voxels: destroy the bulbs and the light dies;
 * drop the chandelier and its light falls with it.
 */
export class LightSystem {
  private tracked: Tracked[];
  private flashes: Flash[] = [];
  private world: VoxelWorld;

  constructor(world: VoxelWorld, lights: SceneLight[]) {
    this.world = world;
    this.tracked = lights.map((l, i) => ({
      base: l,
      anchors: l.anchors.map((a) => {
        const x = Math.floor(a[0] / VOXEL_SIZE), y = Math.floor(a[1] / VOXEL_SIZE), z = Math.floor(a[2] / VOXEL_SIZE);
        return { ...snapToEmissive(world, x, y, z), y0: 0 };
      }).map((a) => ({ ...a, y0: a.y })),
      factor: 1,
      target: 1,
      offsetY: 0,
      seed: i * 7.31,
    }));
  }

  anchorsIn(keys: Set<number>): { light: number; anchor: number }[] {
    const out: { light: number; anchor: number }[] = [];
    const w = this.world;
    this.tracked.forEach((t, li) =>
      t.anchors.forEach((a, ai) => {
        if (keys.has(a.x + w.dx * (a.y + w.dy * a.z))) out.push({ light: li, anchor: ai });
      }),
    );
    return out;
  }

  moveAnchors(list: { light: number; anchor: number }[], dy: number) {
    for (const { light, anchor } of list) this.tracked[light].anchors[anchor].y += dy;
    const touched = new Set(list.map((l) => l.light));
    for (const li of touched) {
      const t = this.tracked[li];
      let s = 0;
      for (const a of t.anchors) s += a.y - a.y0;
      t.offsetY = (s / t.anchors.length) * VOXEL_SIZE;
    }
  }

  flash(pos: [number, number, number], size: number) {
    if (this.flashes.length > 6) this.flashes.shift();
    const life = 0.06 + size * 0.25;
    this.flashes.push({ pos, life, max: life, size });
  }

  update(dt: number) {
    for (const t of this.tracked) {
      if (t.anchors.length) {
        let alive = 0;
        for (const a of t.anchors) {
          const v = this.world.get(a.x, a.y, a.z);
          if (v && mat(v & 0xff).emission > 0) alive++;
        }
        t.target = alive / t.anchors.length;
      }
      t.factor += (t.target - t.factor) * Math.min(1, dt * 12);
    }
    for (const f of this.flashes) f.life -= dt;
    this.flashes = this.flashes.filter((f) => f.life > 0);
  }

  frameLights(time: number): Light[] {
    const out: Light[] = [];
    for (const t of this.tracked) {
      let k = t.factor;
      // Damaged fixtures flicker.
      if (t.target > 0 && t.target < 0.999) {
        const n = Math.sin(time * 31 + t.seed) * Math.sin(time * 17.3 + t.seed * 3);
        if (n > 0.6) k *= 0.15;
      }
      if (k < 0.02) continue;
      const b = t.base;
      out.push({
        pos: [b.pos[0], b.pos[1] + t.offsetY, b.pos[2]],
        radius: b.radius,
        color: [b.color[0] * k, b.color[1] * k, b.color[2] * k],
        shadow: b.shadow,
      });
    }
    for (const f of this.flashes) {
      const k = f.life / f.max;
      const s = 4 + f.size * 20;
      out.push({ pos: f.pos, radius: 1.5 + f.size * 8, color: [s * k, s * 0.7 * k, s * 0.4 * k], shadow: false });
    }
    // Shadowed lights are the expensive ones: keep them at the front so they survive truncation.
    out.sort((a, b) => Number(b.shadow) - Number(a.shadow));
    return out;
  }

  /** Number of lights that are on (for the HUD). */
  get activeCount(): number {
    return this.tracked.filter((t) => t.factor > 0.02).length;
  }
}

/** Find an emissive voxel at or near the anchor point (the sample point may land just off a thin tube). */
function snapToEmissive(world: VoxelWorld, x: number, y: number, z: number): { x: number; y: number; z: number } {
  for (let r = 0; r <= 2; r++)
    for (let dz = -r; dz <= r; dz++)
      for (let dy = -r; dy <= r; dy++)
        for (let dx = -r; dx <= r; dx++) {
          const v = world.get(x + dx, y + dy, z + dz);
          if (v && mat(v & 0xff).emission > 0) return { x: x + dx, y: y + dy, z: z + dz };
        }
  return { x, y, z };
}
