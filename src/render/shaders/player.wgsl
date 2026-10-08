// First-person controller on the GPU: walking with voxel collision (or
// no-clip flying), then the camera matrices every other pass uses.
// The CPU only sends input deltas; position and velocity live here.
// (Prepended with shared.wgsl.)

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read_write> C: Camera;
@group(0) @binding(2) var<storage, read> grid: array<u32>;
@group(0) @binding(3) var<storage, read> pool: array<u32>;
@group(0) @binding(4) var<storage, read> mats: array<vec4f>;

const RADIUS: f32 = 0.22;
const HEIGHT: f32 = 1.72;
const EYE: f32 = 1.62;
const STEP_UP: f32 = 0.36;
const GRAVITY: f32 = 9.81;
const NEAR: f32 = 0.03;
const FAR: f32 = 200.0;

fn voxelAtP(p: vec3i) -> u32 {
  if (any(p < vec3i(0)) || any(p >= idims())) { return 0u; }
  let b = vec3u(p) >> vec3u(3u);
  let bd = vec3u(bdims());
  let g = grid[b.x + bd.x * (b.y + bd.y * b.z)];
  if (g == 0u) { return 0u; }
  if ((g & UNIFORM_BIT) != 0u) { return g & 0xffffu; }
  let li = u32((p.x & 7) | ((p.y & 7) << 3u) | ((p.z & 7) << 6u));
  let idx = (g - 1u) * 512u + li;
  return (pool[idx >> 1u] >> ((idx & 1u) * 16u)) & 0xffffu;
}

fn solidM(x: f32, y: f32, z: f32) -> bool {
  return voxelAtP(vec3i(floor(vec3f(x, y, z) / U.dims.w))) != 0u;
}

/** Does a body cylinder with feet at (x, y, z) overlap anything above step height? */
fn blocked(x: f32, y: f32, z: f32) -> bool {
  for (var hgt = STEP_UP; hgt <= HEIGHT; hgt += 0.12) {
    for (var i = 0; i < 10; i++) {
      let a = f32(i) / 10.0 * 6.2831853;
      if (solidM(x + cos(a) * RADIUS, y + hgt, z + sin(a) * RADIUS)) { return true; }
    }
    if (solidM(x, y + hgt, z)) { return true; }
  }
  return false;
}

/** Highest walkable surface under the footprint between y - maxDown and y + STEP_UP (or -1e9). */
fn groundHeight(x: f32, y: f32, z: f32, maxDown: f32) -> f32 {
  let vs = U.dims.w;
  var best = -1e9;
  let top = i32(floor((y + STEP_UP) / vs));
  let bottom = i32(floor((y - maxDown) / vs));
  for (var i = 0; i < 7; i++) {
    var ox = 0.0;
    var oz = 0.0;
    if (i > 0) {
      let a = f32(i - 1) / 6.0 * 6.2831853;
      ox = cos(a) * RADIUS * 0.7;
      oz = sin(a) * RADIUS * 0.7;
    }
    let ix = i32(floor((x + ox) / vs));
    let iz = i32(floor((z + oz) / vs));
    for (var iy = top; iy >= bottom; iy--) {
      if (voxelAtP(vec3i(ix, iy, iz)) != 0u) {
        best = max(best, f32(iy + 1) * vs);
        break;
      }
    }
  }
  return best;
}

fn forwardDir(yaw: f32, pitch: f32) -> vec3f {
  let cp = cos(pitch);
  return vec3f(sin(yaw) * cp, sin(pitch), -cos(yaw) * cp);
}

@compute @workgroup_size(1)
fn updatePlayer() {
  let dt = U.sim.x;
  var pos = C.feet.xyz;
  var vel = C.vel.xyz;
  var yaw = C.feet.w;
  var pitch = C.vel.w;
  var fly = C.eye.w > 0.5;
  var grounded = C.look.w > 0.5;

  // Edge-triggered requests from the CPU (counters, so a missed frame can't double-fire).
  if (U.look.w != C.state.x) {
    fly = !fly;
    C.state.x = U.look.w;
  }
  if (U.fire2.z != C.state.y) {
    pos = U.teleport.xyz;
    yaw = U.teleport.w;
    pitch = U.fire2.w;
    vel = vec3f(0.0);
    C.state.y = U.fire2.z;
  }

  yaw += U.look.x;
  pitch = clamp(pitch - U.look.y, -1.5, 1.5);
  let f = forwardDir(yaw, 0.0);
  let r = vec2f(-f.z, f.x);
  let inF = U.moveIn.x;
  let inR = U.moveIn.y;

  if (fly) {
    let speed = select(3.5, 9.0, U.look.z > 0.5);
    let lf = forwardDir(yaw, pitch);
    pos += vec3f(lf.x * inF + r.x * inR, lf.y * inF + U.moveIn.z, lf.z * inF + r.y * inR) * speed * dt;
    vel = vec3f(0.0);
  } else {
    let speed = select(2.6, 5.2, U.look.z > 0.5);
    var mv = vec2f(f.x * inF + r.x * inR, f.z * inF + r.y * inR);
    let ml = length(mv);
    if (ml > 1.0) { mv /= ml; }
    let accel = select(3.0, 14.0, grounded);
    let k = min(1.0, accel * dt);
    vel.x += (mv.x * speed - vel.x) * k;
    vel.z += (mv.y * speed - vel.z) * k;
    vel.y -= GRAVITY * dt;
    if (U.moveIn.w > 0.5 && grounded) {
      vel.y = 4.2;
      grounded = false;
    }
    // Horizontal, axis by axis.
    let nx = pos.x + vel.x * dt;
    if (!blocked(nx, pos.y, pos.z)) { pos.x = nx; } else { vel.x = 0.0; }
    let nz = pos.z + vel.z * dt;
    if (!blocked(pos.x, pos.y, nz)) { pos.z = nz; } else { vel.z = 0.0; }
    // Vertical, with automatic step-up.
    let ny = pos.y + vel.y * dt;
    let ground = groundHeight(pos.x, pos.y, pos.z, max(0.05, -vel.y * dt + 0.02));
    let hasGround = ground > -1e8;
    if (hasGround && vel.y <= 0.0 && (ny <= ground || (ground > pos.y && ground - pos.y <= STEP_UP))) {
      pos.y = ground;
      vel.y = 0.0;
      grounded = true;
    } else {
      if (vel.y > 0.0 && solidM(pos.x, ny + HEIGHT + 0.02, pos.z)) { vel.y = 0.0; } else { pos.y = ny; }
      grounded = false;
    }
    // Fell out of the building: back inside.
    if (pos.y < -30.0) {
      pos = vec3f(8.0, 1.0, 4.0);
      vel = vec3f(0.0);
    }
  }

  let eye = pos + vec3f(0.0, EYE, 0.0);
  let fw = forwardDir(yaw, pitch);
  let right = normalize(cross(fw, vec3f(0.0, 1.0, 0.0)));
  let up = cross(right, fw);
  let view = mat4x4f(
    vec4f(right.x, up.x, -fw.x, 0.0),
    vec4f(right.y, up.y, -fw.y, 0.0),
    vec4f(right.z, up.z, -fw.z, 0.0),
    vec4f(-dot(right, eye), -dot(up, eye), dot(fw, eye), 1.0),
  );
  let fl = 1.0 / U.time.w;
  let a = fl / U.time.z;
  let A = FAR / (NEAR - FAR);
  let B = NEAR * FAR / (NEAR - FAR);
  let proj = mat4x4f(
    vec4f(a, 0.0, 0.0, 0.0),
    vec4f(0.0, fl, 0.0, 0.0),
    vec4f(0.0, 0.0, A, -1.0),
    vec4f(0.0, 0.0, B, 0.0),
  );
  let invView = mat4x4f(vec4f(right, 0.0), vec4f(up, 0.0), vec4f(-fw, 0.0), vec4f(eye, 1.0));
  let invProj = mat4x4f(
    vec4f(1.0 / a, 0.0, 0.0, 0.0),
    vec4f(0.0, 1.0 / fl, 0.0, 0.0),
    vec4f(0.0, 0.0, 0.0, 1.0 / B),
    vec4f(0.0, 0.0, -1.0, A / B),
  );
  C.viewProj = proj * view;
  C.invViewProj = invView * invProj;
  C.eye = vec4f(eye, select(0.0, 1.0, fly));
  C.look = vec4f(fw, select(0.0, 1.0, grounded));
  C.feet = vec4f(pos, yaw);
  C.vel = vec4f(vel, pitch);
}
