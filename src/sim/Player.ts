import { GRAVITY, VOXEL_SIZE } from '../config';
import { forward } from '../render/math';
import { VoxelWorld } from '../world/VoxelWorld';

export interface MoveInput {
  forward: number; // -1..1
  right: number; // -1..1
  up: number; // -1..1 (fly)
  jump: boolean;
  sprint: boolean;
}

const RADIUS = 0.22;
const HEIGHT = 1.72;
const EYE = 1.62;
const STEP = 0.36;

/** First-person walker with voxel collision, plus a no-clip fly mode. */
export class Player {
  pos: [number, number, number]; // feet, metres
  vel: [number, number, number] = [0, 0, 0];
  yaw: number;
  pitch: number;
  fly = false;
  grounded = false;
  private world: VoxelWorld;

  constructor(world: VoxelWorld, pos: [number, number, number], yaw: number, pitch: number) {
    this.world = world;
    this.pos = [...pos];
    this.yaw = yaw;
    this.pitch = pitch;
  }

  get eye(): [number, number, number] {
    return [this.pos[0], this.pos[1] + EYE, this.pos[2]];
  }

  get look(): [number, number, number] {
    return forward(this.yaw, this.pitch);
  }

  private solid(x: number, y: number, z: number): boolean {
    return this.world.get(Math.floor(x / VOXEL_SIZE), Math.floor(y / VOXEL_SIZE), Math.floor(z / VOXEL_SIZE)) !== 0;
  }

  /** Does a body cylinder with feet at (x, y, z) overlap anything above step height? */
  private blocked(x: number, y: number, z: number): boolean {
    for (let h = STEP; h <= HEIGHT; h += 0.12) {
      for (let i = 0; i < 10; i++) {
        const a = (i / 10) * Math.PI * 2;
        if (this.solid(x + Math.cos(a) * RADIUS, y + h, z + Math.sin(a) * RADIUS)) return true;
      }
      if (this.solid(x, y + h, z)) return true;
    }
    return false;
  }

  /** Highest walkable surface under the footprint between y-maxDown and y+STEP. */
  private groundHeight(x: number, y: number, z: number, maxDown: number): number | null {
    const vs = VOXEL_SIZE;
    let best: number | null = null;
    const pts: [number, number][] = [[0, 0]];
    for (let i = 0; i < 6; i++) {
      const a = (i / 6) * Math.PI * 2;
      pts.push([Math.cos(a) * RADIUS * 0.7, Math.sin(a) * RADIUS * 0.7]);
    }
    const top = Math.floor((y + STEP) / vs);
    const bottom = Math.floor((y - maxDown) / vs);
    for (const [ox, oz] of pts) {
      const ix = Math.floor((x + ox) / vs), iz = Math.floor((z + oz) / vs);
      for (let iy = top; iy >= bottom; iy--) {
        if (this.world.get(ix, iy, iz) !== 0) {
          const h = (iy + 1) * vs;
          if (best === null || h > best) best = h;
          break;
        }
      }
    }
    return best;
  }

  update(dt: number, input: MoveInput) {
    const f = forward(this.yaw, 0);
    const r: [number, number] = [-f[2], f[0]];
    if (this.fly) {
      const speed = input.sprint ? 9 : 3.5;
      const lf = forward(this.yaw, this.pitch);
      this.pos[0] += (lf[0] * input.forward + r[0] * input.right) * speed * dt;
      this.pos[1] += (lf[1] * input.forward + input.up) * speed * dt;
      this.pos[2] += (lf[2] * input.forward + r[1] * input.right) * speed * dt;
      this.vel = [0, 0, 0];
      return;
    }

    const speed = input.sprint ? 5.2 : 2.6;
    let mx = f[0] * input.forward + r[0] * input.right;
    let mz = f[2] * input.forward + r[1] * input.right;
    const ml = Math.hypot(mx, mz);
    if (ml > 1) { mx /= ml; mz /= ml; }
    const accel = this.grounded ? 14 : 3;
    this.vel[0] += (mx * speed - this.vel[0]) * Math.min(1, accel * dt);
    this.vel[2] += (mz * speed - this.vel[2]) * Math.min(1, accel * dt);
    this.vel[1] -= GRAVITY * dt;
    if (input.jump && this.grounded) {
      this.vel[1] = 4.2;
      this.grounded = false;
    }

    // Horizontal, axis by axis, with automatic step-up.
    const [x, y, z] = this.pos;
    const nx = x + this.vel[0] * dt;
    if (!this.blocked(nx, y, z)) this.pos[0] = nx;
    else this.vel[0] = 0;
    const nz = z + this.vel[2] * dt;
    if (!this.blocked(this.pos[0], y, nz)) this.pos[2] = nz;
    else this.vel[2] = 0;

    // Vertical.
    const ny = this.pos[1] + this.vel[1] * dt;
    const ground = this.groundHeight(this.pos[0], this.pos[1], this.pos[2], Math.max(0.05, -this.vel[1] * dt + 0.02));
    if (ground !== null && this.vel[1] <= 0 && ny <= ground) {
      this.pos[1] = ground;
      this.vel[1] = 0;
      this.grounded = true;
    } else if (ground !== null && ground > this.pos[1] && ground - this.pos[1] <= STEP && this.vel[1] <= 0) {
      this.pos[1] = ground;
      this.vel[1] = 0;
      this.grounded = true;
    } else {
      if (this.vel[1] > 0 && this.solid(this.pos[0], ny + HEIGHT + 0.02, this.pos[2])) this.vel[1] = 0;
      else this.pos[1] = ny;
      this.grounded = false;
    }
    // Fell out of the building: respawn inside.
    if (this.pos[1] < -30) {
      this.pos = [8, 1, 4];
      this.vel = [0, 0, 0];
    }
  }
}
