// Debris simulation: GPU voxel particles that collide with the brick map.
// Particles that come to rest can be "deposited" back into the world: their
// cell + value is appended to a queue the CPU reads back and writes as voxels.

fn solidAt(p: vec3i) -> bool { return voxelAt(p) != 0u; }

// ---------------------------------------------------------------------------
// Simulation

struct Deposits {
  count: atomic<u32>,
  cap: u32,
  pad0: u32,
  pad1: u32,
  items: array<vec2u>,
};

@group(1) @binding(0) var<storage, read_write> parts: array<Particle>;
@group(1) @binding(1) var<storage, read_write> alive: array<u32>;
@group(1) @binding(2) var<storage, read_write> drawArgs: array<atomic<u32>, 4>;
@group(1) @binding(3) var<storage, read_write> deposits: Deposits;

fn pushAlive(i: u32) {
  let slot = atomicAdd(&drawArgs[1], 1u);
  alive[slot] = i;
}

@compute @workgroup_size(64)
fn simulate(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u32(U.sim.z)) { return; }
  var p = parts[i];
  if (p.life <= 0.0) { return; }
  let dt = U.sim.x;
  p.life -= dt;
  if (p.life <= 0.0) {
    parts[i].life = 0.0;
    return;
  }
  if ((p.value & SETTLED) != 0u) {
    parts[i].life = p.life;
    pushAlive(i);
    return;
  }

  let dust = (p.value & DUST) != 0u;
  let g = select(U.sim.y, U.sim.y * 0.15, dust);
  p.vel.y -= g * dt;
  p.vel *= 1.0 - select(0.2, 2.5, dust) * dt;

  var cell = vec3i(floor(p.pos));
  if (solidAt(cell)) {
    // Spawned inside something (or the world changed under us): pop upwards.
    p.pos.y += 1.0;
    parts[i] = p;
    pushAlive(i);
    return;
  }

  let mv = p.vel * dt;
  let dist = length(mv);
  if (dist > 1e-6) {
    let dir = select(mv / dist, vec3f(1e-6), abs(mv / dist) < vec3f(1e-6));
    let inv = 1.0 / dir;
    let stp = vec3i(sign(dir));
    var tMax = (vec3f(cell) + step(vec3f(0.0), dir) - p.pos) * inv;
    let tDelta = abs(inv);
    var t = 0.0;
    var hitAxis = -1;
    for (var k = 0; k < 48; k++) {
      var axis = 0;
      if (tMax.x < tMax.y && tMax.x < tMax.z) { axis = 0; }
      else if (tMax.y < tMax.z) { axis = 1; }
      else { axis = 2; }
      let tn = tMax[axis];
      if (tn > dist) { break; }
      var next = cell;
      next[axis] += stp[axis];
      if (solidAt(next)) {
        t = tn;
        hitAxis = axis;
        break;
      }
      cell = next;
      tMax[axis] += tDelta[axis];
    }
    if (hitAxis < 0) {
      p.pos += mv;
    } else {
      p.pos += dir * max(t - 1e-3, 0.0);
      let vin = p.vel[hitAxis];
      p.vel[hitAxis] = -vin * 0.25;
      let fr = select(0.75, 0.55, hitAxis == 1);
      for (var a = 0; a < 3; a++) { if (a != hitAxis) { p.vel[a] *= fr; } }
    }
  }

  // Rest check: slow and supported from below.
  let below = vec3i(floor(p.pos)) - vec3i(0, 1, 0);
  let speed = length(p.vel);
  if (speed < 0.6 / U.dims.w && solidAt(below)) {
    let c = vec3i(floor(p.pos));
    p.pos = vec3f(c) + 0.5;
    p.vel = vec3f(0.0);
    p.value |= SETTLED;
    if ((p.value & DEPOSIT) != 0u && all(c >= vec3i(0)) && all(c < idims())) {
      let k = atomicAdd(&deposits.count, 1u);
      if (k < deposits.cap) {
        let d = idims();
        deposits.items[k] = vec2u(u32(c.x + d.x * (c.y + d.y * c.z)), p.value & 0xffffu);
        p.life = min(p.life, 0.5);
      } else {
        p.life = min(p.life, 2.0);
      }
    } else {
      p.life = min(p.life, select(4.0, 0.8, dust));
    }
  }
  if (p.pos.y < -200.0) { p.life = 0.0; }
  parts[i] = p;
  if (p.life > 0.0) { pushAlive(i); }
}
