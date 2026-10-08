// GPU-resident simulation. Every voxel edit happens here, in compute passes;
// the CPU never reads or writes the world after the initial upload.
//
// Frame order (see src/gpu/Simulation.ts):
//   frameBegin -> shoot -> impact jobs -> alloc -> carve -> flood x K
//   -> island search (begin, clear, seed, step x K, resolve) -> chunks
//   -> particles -> voxel writes (deposits + landings) -> maintenance -> lights
//
// (Prepended with shared.wgsl and the generated layout constants.)

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<uniform> C: Camera;
@group(0) @binding(2) var<storage, read_write> grid: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> pool: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> superGrid: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> mats: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> ctrl: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> heap: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(10) var densityOut: texture_storage_3d<rgba8unorm, write>;
/** Indirect args. Only bound for the passes that compute them (see Simulation.ts). */
@group(1) @binding(0) var<storage, read_write> args: array<atomic<u32>>;

const NONE: u32 = 0xffffffffu;
const ANY_MAT: u32 = 256u;

/** The 6 face neighbours: k = 0..5 -> +x, -x, +y, -y, +z, -z. */
fn dirOf(k: i32) -> vec3i {
  var d = vec3i(0);
  d[k >> 1u] = 1 - 2 * (k & 1);
  return d;
}

// ---------------------------------------------------------------------------
// Small helpers

fn h(i: u32) -> u32 { return atomicLoad(&heap[i]); }
fn hs(i: u32, v: u32) { atomicStore(&heap[i], v); }
fn hf(i: u32) -> f32 { return bitcast<f32>(atomicLoad(&heap[i])); }
fn hsf(i: u32, v: f32) { atomicStore(&heap[i], bitcast<u32>(v)); }
fn cget(i: u32) -> u32 { return atomicLoad(&ctrl[i]); }
fn cset(i: u32, v: u32) { atomicStore(&ctrl[i], v); }
fn setArgs(at: u32, x: u32) { atomicStore(&args[at], x); atomicStore(&args[at + 1u], 1u); atomicStore(&args[at + 2u], 1u); }
fn aset(i: u32, v: u32) { atomicStore(&args[i], v); }
fn groups(n: u32) -> u32 { return (n + 63u) / 64u; }

fn VS() -> f32 { return U.dims.w; }
fn dimsU() -> vec3u { return vec3u(U.dims.xyz); }
fn inWorld(p: vec3i) -> bool { return all(p >= vec3i(0)) && all(p < idims()); }
fn keyOf(p: vec3i) -> u32 { let d = dimsU(); return u32(p.x) + d.x * (u32(p.y) + d.y * u32(p.z)); }
fn posOf(k: u32) -> vec3i { let d = dimsU(); return vec3i(i32(k % d.x), i32((k / d.x) % d.y), i32(k / (d.x * d.y))); }
fn cellIndexB(b: vec3u) -> u32 { let bd = dimsU() >> vec3u(3u); return b.x + bd.x * (b.y + bd.y * b.z); }
fn cellOf(p: vec3i) -> u32 { return cellIndexB(vec3u(p) >> vec3u(3u)); }
fn cellCoord(ci: u32) -> vec3u { let bd = dimsU() >> vec3u(3u); return vec3u(ci % bd.x, (ci / bd.x) % bd.y, ci / (bd.x * bd.y)); }
fn superOf(ci: u32) -> u32 {
  let s = cellCoord(ci) >> vec3u(2u);
  let sd = vec3u(sdims());
  return s.x + sd.x * (s.y + sd.y * s.z);
}
fn localIndex(p: vec3i) -> u32 { return u32((p.x & 7) | ((p.y & 7) << 3u) | ((p.z & 7) << 6u)); }

fn structural(v: u32) -> bool { return matE(v & 0xffu).z > 0.5; }
/** Material ids 128..255 are rubble: settled debris that carries no load. */
fn isRubble(v: u32) -> bool { return (v & 0x80u) != 0u; }
fn rubbleOf(v: u32) -> u32 { return v | 0x80u; }
fn fragile(v: u32) -> bool { return matE(v & 0xffu).y > 0.5; }
fn priceDimes(v: u32) -> u32 { return u32(matD(v & 0xffu).w * 10.0 + 0.5); }
fn rnd3(seed: u32) -> vec3f { return vec3f(rnd(seed), rnd(seed + 1u), rnd(seed + 2u)); }
fn frameSeed() -> u32 { return u32(U.time.y) * 0x9E3779B9u; }

// ---------------------------------------------------------------------------
// World access (atomic)

fn readVoxel(p: vec3i) -> u32 {
  if (!inWorld(p)) { return 0u; }
  let g = atomicLoad(&grid[cellOf(p)]);
  if (g == 0u) { return 0u; }
  if ((g & UNIFORM_BIT) != 0u) { return g & 0xffffu; }
  let idx = (g - 1u) * 512u + localIndex(p);
  return (atomicLoad(&pool[idx >> 1u]) >> ((idx & 1u) * 16u)) & 0xffffu;
}

fn markDirty(ci: u32) {
  let bit = 1u << (ci & 31u);
  let wi = H_DIRTY_FLAGS + (ci >> 5u);
  let old = atomicOr(&heap[wi], bit);
  if ((old & bit) == 0u) {
    let k = atomicAdd(&ctrl[C_DIRTY], 1u);
    if (k < MAX_DIRTY) { hs(H_DIRTY_LIST + k, ci); } else { atomicAnd(&heap[wi], ~bit); }
  }
}

/** Ask the next alloc pass to give cell ci its own pool brick (expanding empty/uniform cells). */
fn requestBrick(ci: u32) {
  let bit = 1u << (ci & 31u);
  let wi = H_ALLOC_FLAGS + (ci >> 5u);
  let old = atomicOr(&heap[wi], bit);
  if ((old & bit) == 0u) {
    let k = atomicAdd(&ctrl[C_ALLOCS], 1u);
    if (k < MAX_ALLOC) { hs(H_ALLOC_LIST + k, ci); } else { atomicAnd(&heap[wi], ~bit); }
  }
}

/** Atomically remove the voxel at p if its material is `mat` (ANY_MAT = any). Pool bricks only. */
fn takeVoxel(p: vec3i, mat: u32) -> u32 {
  if (!inWorld(p)) { return 0u; }
  let ci = cellOf(p);
  let g = atomicLoad(&grid[ci]);
  if (g == 0u || (g & UNIFORM_BIT) != 0u) { return 0u; }
  let idx = (g - 1u) * 512u + localIndex(p);
  let wi = idx >> 1u;
  let sh = (idx & 1u) * 16u;
  for (var i = 0; i < 16; i++) {
    let old = atomicLoad(&pool[wi]);
    let v = (old >> sh) & 0xffffu;
    if (v == 0u || (mat < ANY_MAT && (v & 0xffu) != mat)) { return 0u; }
    let r = atomicCompareExchangeWeak(&pool[wi], old, old & ~(0xffffu << sh));
    if (r.exchanged) {
      markDirty(ci);
      return v;
    }
  }
  return 0u;
}

/** Write v into an empty cell. 0 = written, 1 = cell occupied, 2 = no pool brick there. */
fn putVoxel(p: vec3i, v: u32) -> u32 {
  if (!inWorld(p)) { return 2u; }
  let ci = cellOf(p);
  let g = atomicLoad(&grid[ci]);
  if (g == 0u) { return 2u; }
  if ((g & UNIFORM_BIT) != 0u) { return 1u; }
  let idx = (g - 1u) * 512u + localIndex(p);
  let wi = idx >> 1u;
  let sh = (idx & 1u) * 16u;
  for (var i = 0; i < 16; i++) {
    let old = atomicLoad(&pool[wi]);
    if (((old >> sh) & 0xffffu) != 0u) { return 1u; }
    let r = atomicCompareExchangeWeak(&pool[wi], old, old | (v << sh));
    if (r.exchanged) {
      markDirty(ci);
      return 0u;
    }
  }
  return 1u;
}

fn popFree() -> u32 {
  let old = atomicSub(&ctrl[C_FREE_TOP], 1u);
  if (old == 0u || old > POOL_CAP) {
    atomicAdd(&ctrl[C_FREE_TOP], 1u);
    atomicAdd(&ctrl[S_ALLOC_FAIL], 1u);
    return NONE;
  }
  return h(H_FREE_LIST + old - 1u);
}

fn pushFree(p: u32) {
  let k = atomicAdd(&ctrl[C_FREE_TOP], 1u);
  hs(H_FREE_LIST + k, p);
}

// ---------------------------------------------------------------------------
// Particles, seeds, writes, impacts

fn spawn(pos: vec3f, vel: vec3f, value: u32, life: f32) {
  if (atomicAdd(&ctrl[C_SPAWNED], 1u) >= SPAWN_BUDGET) { return; }
  let i = atomicAdd(&ctrl[C_PCURSOR], 1u) & (MAX_PARTICLES - 1u);
  particles[i] = Particle(pos, value, vel, life);
}

fn debrisFlags(v: u32, seed: u32) -> u32 {
  let id = v & 0xffu;
  if (matD(id).z > 0.0) { return 0u; } // liquids splash and vanish
  if (rnd(seed ^ 0x51ed27u) < select(0.75, 0.55, fragile(v))) { return DEPOSIT; }
  if (rnd(seed ^ 0x9e3779u) < 0.5) { return DUST; }
  return 0u;
}

/** Queue a voxel for the island search (it may have lost its support). */
fn pushSeed(p: vec3i) {
  let k = atomicAdd(&ctrl[C_PENDING], 1u);
  if (k < MAX_SEEDS) { hs(H_SEEDS + k, keyOf(p)); } else { atomicAdd(&ctrl[S_SEED_DROP], 1u); }
}

fn pushWrite(p: vec3i, v: u32) -> bool {
  let k = atomicAdd(&ctrl[C_WRITES], 1u);
  if (k >= MAX_WRITES) { return false; }
  hs(H_WRITE_LIST + k * 2u, keyOf(p));
  hs(H_WRITE_LIST + k * 2u + 1u, v);
  return true;
}

struct Impact {
  c: vec3f,       // centre (voxels)
  r: f32,         // carve radius (voxels)
  fr: f32,        // radius within which fragile materials break
  impulse: f32,   // debris speed (voxels/s)
  n: vec3f,       // surface normal at the hit
  dir: vec3f,     // incoming direction
};

fn pushImpact(c: vec3f, r: f32, fr: f32, impulse: f32, n: vec3f, dir: vec3f) {
  let k = atomicAdd(&ctrl[C_IMPACTS], 1u);
  if (k >= MAX_IMPACTS) { return; }
  let b = H_IMPACTS + k * 16u;
  hsf(b, c.x); hsf(b + 1u, c.y); hsf(b + 2u, c.z);
  hsf(b + 3u, r); hsf(b + 4u, fr); hsf(b + 5u, impulse);
  hsf(b + 6u, n.x); hsf(b + 7u, n.y); hsf(b + 8u, n.z);
  hsf(b + 9u, dir.x); hsf(b + 10u, dir.y); hsf(b + 11u, dir.z);
}

fn loadImpact(k: u32) -> Impact {
  let b = H_IMPACTS + k * 16u;
  var im: Impact;
  im.c = vec3f(hf(b), hf(b + 1u), hf(b + 2u));
  im.r = hf(b + 3u);
  im.fr = hf(b + 4u);
  im.impulse = hf(b + 5u);
  im.n = vec3f(hf(b + 6u), hf(b + 7u), hf(b + 8u));
  im.dir = vec3f(hf(b + 9u), hf(b + 10u), hf(b + 11u));
  return im;
}

/**
 * Does impact `im` remove voxel v at p? Jittered sphere: hard materials shrink
 * the hole, but a hit always punches ~5 voxels so rods and legs can be cut.
 */
fn removes(im: Impact, p: vec3i, v: u32) -> bool {
  let id = v & 0xffu;
  let d = vec3f(p) + 0.5 - im.c;
  let d2 = dot(d, d);
  let jit = 0.8 + 0.4 * rnd(hash3i(p));
  let E = matE(id);
  if (E.y > 0.5 && d2 < im.fr * im.fr * jit * jit) { return true; }
  if (E.x <= 0.0) { return false; }
  let rr = max(im.r * min(1.4, E.x), min(im.r, 2.5)) * jit;
  return d2 < rr * rr;
}

/** Furthest distance at which `removes` can return true for material id. */
fn reach(im: Impact, id: u32) -> f32 {
  let E = matE(id);
  var r = 0.0;
  if (E.y > 0.5) { r = im.fr * 1.2; }
  if (E.x > 0.0) { r = max(r, max(im.r * min(1.4, E.x), min(im.r, 2.5)) * 1.2); }
  return r;
}

fn pushFlood(p: vec3i, mat: u32) {
  let par = cget(C_FLOOD_PAR);
  let k = atomicAdd(&ctrl[C_FLOOD_NEXT], 1u);
  if (k < MAX_FLOOD) {
    let q = H_FLOOD_Q + ((1u - par) * MAX_FLOOD + k) * 2u;
    hs(q, keyOf(p));
    hs(q + 1u, mat);
  }
}

/** Bill + debris for one voxel broken off a glass pane. */
fn shardEffects(p: vec3i, v: u32) {
  dropRubbleAbove(p);
  atomicAdd(&ctrl[S_BILL], priceDimes(v));
  atomicAdd(&ctrl[S_DESTROYED], 1u);
  let seed = keyOf(p) ^ frameSeed();
  if (rnd(seed) < matE(v & 0xffu).w * 0.6) {
    let s = 1.0 / VS();
    let vel = vec3f(rnd(seed + 3u) - 0.5, rnd(seed + 4u) * 0.4, rnd(seed + 5u) - 0.5) * vec3f(1.6, 1.0, 1.6) * s;
    spawn(vec3f(p) + 0.5, vel, v | debrisFlags(v, seed), 4.0 + 4.0 * rnd(seed + 6u));
  }
}

/** Rubble resting on a voxel that just went away tumbles down as debris again. */
fn dropRubbleAbove(p: vec3i) {
  for (var d = 1; d <= 12; d++) {
    let q = p + vec3i(0, d, 0);
    let v = readVoxel(q);
    if (v == 0u || !isRubble(v)) { return; }
    let t = takeVoxel(q, v & 0xffu);
    if (t == 0u) { return; }
    let seed = keyOf(q) ^ frameSeed();
    spawn(vec3f(q) + 0.5, (rnd3(seed) - 0.5) * vec3f(0.4, 0.0, 0.4) / VS(), t | DEPOSIT, 6.0 + 2.0 * rnd(seed + 5u));
  }
}

// ---------------------------------------------------------------------------
// Ray cast for shots (any non-air voxel stops the ray, glass included)

struct THit {
  hit: bool,
  voxel: vec3i,
  normal: vec3f,
  value: u32,
};

fn traceSolid(ro: vec3f, rdIn: vec3f, maxT: f32) -> THit {
  var hit: THit;
  hit.hit = false;
  let rd = select(rdIn, sign(rdIn + vec3f(1e-30)) * 1e-6, abs(rdIn) < vec3f(1e-6));
  let inv = 1.0 / rd;
  let t0 = (vec3f(0.0) - ro) * inv;
  let t1 = (U.dims.xyz - ro) * inv;
  let tmn = min(t0, t1);
  let tmx = max(t0, t1);
  var t = max(max(tmn.x, tmn.y), max(tmn.z, 0.0));
  let tExit = min(min(tmx.x, tmx.y), min(tmx.z, maxT));
  if (t >= tExit) { return hit; }
  let stp = vec3i(sign(rd));
  let stpF = vec3f(stp);
  let pos = step(vec3f(0.0), rd);
  var normal = vec3f(0.0);
  if (t > 0.0) {
    if (tmn.x >= tmn.y && tmn.x >= tmn.z) { normal = vec3f(-stpF.x, 0.0, 0.0); }
    else if (tmn.y >= tmn.z) { normal = vec3f(0.0, -stpF.y, 0.0); }
    else { normal = vec3f(0.0, 0.0, -stpF.z); }
  }
  let bd = bdims();
  var bc = clamp(vec3i(floor((ro + rd * (t + 1e-3)) / 8.0)), vec3i(0), bd - 1);
  let tDeltaB = abs(inv) * 8.0;
  var tMaxB = ((vec3f(bc) + pos) * 8.0 - ro) * inv;
  let tDeltaV = abs(inv);
  for (var i = 0; i < 1024; i++) {
    let g = atomicLoad(&grid[u32(bc.x + bd.x * (bc.y + bd.y * bc.z))]);
    if (g != 0u) {
      let lo = bc * 8;
      var vc = clamp(vec3i(floor(ro + rd * (t + 1e-3))), lo, lo + 7);
      var tMaxV = (vec3f(vc) + pos - ro) * inv;
      var tv = t;
      var n = normal;
      for (var k = 0; k < 25; k++) {
        let v = readVoxel(vc);
        if (v != 0u) {
          hit.hit = true; hit.voxel = vc; hit.normal = n; hit.value = v;
          return hit;
        }
        if (tMaxV.x < tMaxV.y && tMaxV.x < tMaxV.z) {
          tv = tMaxV.x; vc.x += stp.x; tMaxV.x += tDeltaV.x; n = vec3f(-stpF.x, 0.0, 0.0);
        } else if (tMaxV.y < tMaxV.z) {
          tv = tMaxV.y; vc.y += stp.y; tMaxV.y += tDeltaV.y; n = vec3f(0.0, -stpF.y, 0.0);
        } else {
          tv = tMaxV.z; vc.z += stp.z; tMaxV.z += tDeltaV.z; n = vec3f(0.0, 0.0, -stpF.z);
        }
        if (tv > tExit) { return hit; }
        if (any(vc < lo) || any(vc > lo + 7)) { break; }
      }
    }
    if (tMaxB.x < tMaxB.y && tMaxB.x < tMaxB.z) {
      t = tMaxB.x; bc.x += stp.x; tMaxB.x += tDeltaB.x; normal = vec3f(-stpF.x, 0.0, 0.0);
    } else if (tMaxB.y < tMaxB.z) {
      t = tMaxB.y; bc.y += stp.y; tMaxB.y += tDeltaB.y; normal = vec3f(0.0, -stpF.y, 0.0);
    } else {
      t = tMaxB.z; bc.z += stp.z; tMaxB.z += tDeltaB.z; normal = vec3f(0.0, 0.0, -stpF.z);
    }
    if (t > tExit || any(bc < vec3i(0)) || any(bc >= bd)) { break; }
  }
  return hit;
}

// ---------------------------------------------------------------------------
// Frame start + shooting

@compute @workgroup_size(1)
fn frameBegin() {
  cset(C_SPAWNED, 0u);
  cset(C_DIRTY, 0u);
  cset(C_WRITES, 0u);
  cset(C_ALLOCS, 0u);
  aset(A_DRAWP, 36u); aset(A_DRAWP + 1u, 0u); aset(A_DRAWP + 2u, 0u); aset(A_DRAWP + 3u, 0u);
}

@compute @workgroup_size(16)
fn shoot(@builtin(local_invocation_index) li: u32) {
  let pellets = u32(U.fire.x);
  if (li >= pellets) { return; }
  var dir = C.look.xyz;
  if (U.fire2.x > 0.0) {
    let seed = u32(U.fire2.y) * 7919u + li * 104729u;
    dir = normalize(dir + (rnd3(seed) - 0.5) * 2.0 * U.fire2.x);
  }
  let hit = traceSolid(C.eye.xyz / VS(), dir, 80.0 / VS());
  if (!hit.hit) { return; }
  let n = hit.normal;
  let r = U.fire.y;
  let c = vec3f(hit.voxel) + 0.5 - n * min(r * 0.3, 2.0);
  pushImpact(c, r, U.fire.z, U.fire.w, n, dir);
  // Panes (windows, glass shelves, mirrors, the table top) shatter as a whole.
  let id = hit.value & 0xffu;
  if (matE(id).y > 1.5) {
    let v = takeVoxel(hit.voxel, id);
    if (v != 0u) {
      shardEffects(hit.voxel, v);
      pushFlood(hit.voxel, id);
    }
  }
  if (li == 0u) {
    let k = atomicAdd(&ctrl[C_FLASH], 1u) % MAX_FLASHES;
    let b = H_FLASHES + k * 8u;
    let wp = (c + n * 4.0) * VS();
    hsf(b, wp.x); hsf(b + 1u, wp.y); hsf(b + 2u, wp.z);
    hsf(b + 3u, r * VS());
    hsf(b + 4u, U.time.x);
  }
}

// ---------------------------------------------------------------------------
// Impacts: enumerate bricks, give uniform ones their own pool brick, carve.

@compute @workgroup_size(1)
fn impactArgs() {
  setArgs(A_IMPACT, min(cget(C_IMPACTS), MAX_IMPACTS));
  cset(C_JOBS, 0u);
}

@compute @workgroup_size(64)
fn impactJobs(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
  let im = loadImpact(wid.x);
  let R = ceil(max(im.r, im.fr) * 1.2) + 1.0;
  let bd = bdims();
  let bmin = clamp(vec3i(floor((im.c - R) / 8.0)), vec3i(0), bd - 1);
  let bmax = clamp(vec3i(floor((im.c + R) / 8.0)), vec3i(0), bd - 1);
  let ext = vec3u(bmax - bmin + 1);
  let total = ext.x * ext.y * ext.z;
  for (var j = li; j < total; j += 64u) {
    let b = vec3u(bmin) + vec3u(j % ext.x, (j / ext.x) % ext.y, j / (ext.x * ext.y));
    let ci = cellIndexB(b);
    let g = atomicLoad(&grid[ci]);
    if (g == 0u) { continue; }
    let lo = vec3f(b * 8u);
    let e = max(max(lo - im.c, vec3f(0.0)), im.c - (lo + 8.0));
    let near2 = dot(e, e);
    if (near2 > R * R) { continue; }
    if ((g & UNIFORM_BIT) != 0u) {
      let rch = reach(im, g & 0xffu);
      if (near2 > rch * rch) { continue; }
      requestBrick(ci);
    }
    let k = atomicAdd(&ctrl[C_JOBS], 1u);
    if (k < MAX_JOBS) {
      hs(H_JOBS + k * 2u, wid.x);
      hs(H_JOBS + k * 2u + 1u, ci);
    }
  }
}

@compute @workgroup_size(1)
fn allocArgs() {
  let n = min(cget(C_ALLOCS), MAX_ALLOC);
  cset(C_ALLOC_N, n);
  cset(C_ALLOCS, 0u);
  setArgs(A_ALLOC, n);
}

var<workgroup> wgP: u32;
var<workgroup> wgFill: u32;
var<workgroup> wgCi: u32;
var<workgroup> wgFlag: u32;

/** One workgroup per requested cell: pop a free pool brick, fill it, publish it in the grid. */
@compute @workgroup_size(64)
fn allocBricks(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
  if (li == 0u) {
    let ci = h(H_ALLOC_LIST + wid.x);
    atomicAnd(&heap[H_ALLOC_FLAGS + (ci >> 5u)], ~(1u << (ci & 31u)));
    let g = atomicLoad(&grid[ci]);
    var p = NONE;
    if (g == 0u || (g & UNIFORM_BIT) != 0u) { p = popFree(); }
    wgP = p;
    wgFill = select(0u, g & 0xffffu, g != 0u);
    wgCi = ci;
    wgFlag = select(0u, 1u, g == 0u);
  }
  let p = workgroupUniformLoad(&wgP);
  if (p == NONE) { return; }
  let fill = wgFill | (wgFill << 16u);
  for (var k = 0u; k < 4u; k++) { atomicStore(&pool[p * 256u + li * 4u + k], fill); }
  storageBarrier();
  workgroupBarrier();
  if (li == 0u) {
    atomicStore(&grid[wgCi], p + 1u);
    if (wgFlag == 1u) { atomicAdd(&superGrid[superOf(wgCi)], 1u); }
    markDirty(wgCi);
  }
}

@compute @workgroup_size(1)
fn carveArgs() {
  setArgs(A_CARVE, min(cget(C_JOBS), MAX_JOBS));
}

var<workgroup> wgG: u32;
var<workgroup> wgDimes: atomic<u32>;
var<workgroup> wgCount: atomic<u32>;
var<workgroup> wgAny: atomic<u32>;

fn carveEffects(im: Impact, p: vec3i, v: u32, keep: f32) {
  let id = v & 0xffu;
  dropRubbleAbove(p);
  atomicAdd(&wgDimes, priceDimes(v));
  atomicAdd(&wgCount, 1u);
  let seed = keyOf(p) ^ frameSeed();
  if (rnd(seed) < matE(id).w * keep) {
    let d = vec3f(p) + 0.5 - im.c;
    let dirn = d / max(length(d), 1e-3);
    let s = im.impulse * (0.3 + 0.9 * rnd(seed + 7u));
    let jitter = (rnd3(seed + 13u) - vec3f(0.5, 0.2, 0.5)) * im.impulse * vec3f(0.4, 0.5, 0.4);
    let vel = (dirn * 0.7 + im.n * 0.6 - im.dir * 0.2) * s + jitter;
    spawn(vec3f(p) + 0.5, vel, v | debrisFlags(v, seed), 5.0 + 4.0 * rnd(seed + 11u));
  }
  // Surviving neighbours: the rest of a glass pane shatters (flood), anything
  // else may have lost its support (island search).
  let pane = matE(id).y > 1.5;
  for (var k = 0; k < 6; k++) {
    let q = p + dirOf(k);
    let w = readVoxel(q);
    if (w == 0u) { continue; }
    if (pane && (w & 0xffu) == id) {
      let t = takeVoxel(q, id);
      if (t != 0u) {
        shardEffects(q, t);
        pushFlood(q, id);
      }
      continue;
    }
    if (!structural(w) && !isRubble(w) && !removes(im, q, w)) { pushSeed(q); }
  }
}

/** One workgroup per (impact, brick) job; each thread owns 4 words = 8 voxels. */
@compute @workgroup_size(64)
fn carve(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
  if (li == 0u) {
    let ci = h(H_JOBS + wid.x * 2u + 1u);
    wgCi = ci;
    wgG = atomicLoad(&grid[ci]);
    atomicStore(&wgDimes, 0u);
    atomicStore(&wgCount, 0u);
    atomicStore(&wgAny, 0u);
  }
  let g = workgroupUniformLoad(&wgG);
  if (g != 0u && (g & UNIFORM_BIT) == 0u) {
    let im = loadImpact(h(H_JOBS + wid.x * 2u));
    let bo = vec3i(cellCoord(wgCi)) * 8;
    let base = (g - 1u) * 256u;
    let keep = min(1.0, 30000.0 / max(1.0, 4.19 * im.r * im.r * im.r));
    for (var k = 0u; k < 4u; k++) {
      let wi = base + li * 4u + k;
      let vi0 = (li * 4u + k) * 2u;
      for (var attempt = 0; attempt < 8; attempt++) {
        let old = atomicLoad(&pool[wi]);
        var mask = 0u;
        for (var hh = 0u; hh < 2u; hh++) {
          let v = (old >> (hh * 16u)) & 0xffffu;
          if (v == 0u) { continue; }
          let l = vi0 + hh;
          let p = bo + vec3i(i32(l & 7u), i32((l >> 3u) & 7u), i32(l >> 6u));
          if (removes(im, p, v)) { mask |= 0xffffu << (hh * 16u); }
        }
        if (mask == 0u) { break; }
        let r = atomicCompareExchangeWeak(&pool[wi], old, old & ~mask);
        if (r.exchanged) {
          for (var hh = 0u; hh < 2u; hh++) {
            if (((mask >> (hh * 16u)) & 1u) == 0u) { continue; }
            let l = vi0 + hh;
            let p = bo + vec3i(i32(l & 7u), i32((l >> 3u) & 7u), i32(l >> 6u));
            carveEffects(im, p, (old >> (hh * 16u)) & 0xffffu, keep);
          }
          atomicAdd(&wgAny, 1u);
          break;
        }
      }
    }
  }
  workgroupBarrier();
  if (li == 0u) {
    if (atomicLoad(&wgAny) > 0u) { markDirty(wgCi); }
    atomicAdd(&ctrl[S_BILL], atomicLoad(&wgDimes));
    atomicAdd(&ctrl[S_DESTROYED], atomicLoad(&wgCount));
  }
}

@compute @workgroup_size(1)
fn postCarve() {
  cset(C_IMPACTS, 0u);
}

// ---------------------------------------------------------------------------
// Pane shatter: breadth-first flood through one material, a few rings per frame.

@compute @workgroup_size(1)
fn floodArgs() {
  let par = 1u - cget(C_FLOOD_PAR);
  cset(C_FLOOD_PAR, par);
  let n = min(cget(C_FLOOD_NEXT), MAX_FLOOD);
  cset(C_FLOOD_CUR, n);
  cset(C_FLOOD_NEXT, 0u);
  setArgs(A_FLOOD, groups(n));
}

@compute @workgroup_size(64)
fn floodStep(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= cget(C_FLOOD_CUR)) { return; }
  let par = cget(C_FLOOD_PAR);
  let q = H_FLOOD_Q + (par * MAX_FLOOD + i) * 2u;
  let p = posOf(h(q));
  let mat = h(q + 1u);
  for (var k = 0; k < 6; k++) {
    let nb = p + dirOf(k);
    let v = takeVoxel(nb, mat);
    if (v != 0u) {
      shardEffects(nb, v);
      pushFlood(nb, mat);
    } else {
      let w = readVoxel(nb);
      if (w != 0u && (w & 0xffu) != mat && !structural(w) && !isRubble(w)) { pushSeed(nb); }
    }
  }
}

// ---------------------------------------------------------------------------
// Island search. Seeds are voxels next to fresh damage. A multi-source BFS
// (spread over frames) records every visited voxel in a hash table whose
// slots double as union-find nodes. Components that touch something
// structural are anchored; the rest are cut loose as falling chunks.

fn hashSlot(key: u32) -> u32 { return hashu(key * 0x9E3779B1u + 0x632be5abu) & (HASH_SIZE - 1u); }

/** Returns (slot, isNew); slot = NONE if the probe sequence is full. */
fn hashInsert(key: u32) -> vec2u {
  var s = hashSlot(key);
  let tag = key + 1u;
  for (var i = 0u; i < 64u; i++) {
    let r = atomicCompareExchangeWeak(&heap[H_HASH_KEYS + s], 0u, tag);
    if (r.exchanged) {
      let n = atomicAdd(&ctrl[C_NODES], 1u);
      if (n < HASH_SIZE) { hs(H_NODE_LIST + n, s); }
      return vec2u(s, 1u);
    }
    if (r.old_value == tag) { return vec2u(s, 0u); }
    if (r.old_value != 0u) { s = (s + 1u) & (HASH_SIZE - 1u); }
  }
  return vec2u(NONE, 0u);
}

fn hashFind(key: u32) -> u32 {
  var s = hashSlot(key);
  let tag = key + 1u;
  for (var i = 0u; i < 64u; i++) {
    let k = h(H_HASH_KEYS + s);
    if (k == tag) { return s; }
    if (k == 0u) { return NONE; }
    s = (s + 1u) & (HASH_SIZE - 1u);
  }
  return NONE;
}

fn findRoot(x: u32) -> u32 {
  var a = x;
  for (var i = 0; i < 256; i++) {
    let p = h(H_PARENT + a);
    if (p == a) { return a; }
    let gp = h(H_PARENT + p);
    if (gp != p) { atomicMin(&heap[H_PARENT + a], gp); } // path halving; parents only ever decrease
    a = gp;
  }
  return a;
}

/** Lock-free union (link the larger root under the smaller one). */
fn unite(x: u32, y: u32) {
  var a = findRoot(x);
  var b = findRoot(y);
  for (var i = 0; i < 64; i++) {
    if (a == b) { return; }
    if (a < b) { let t = a; a = b; b = t; }
    let old = atomicMin(&heap[H_PARENT + a], b);
    if (old == a) {
      if ((h(H_NFLAGS + a) & 2u) != 0u) { atomicOr(&heap[H_NFLAGS + b], 2u); }
      atomicAdd(&heap[H_NCOUNT + b], h(H_NCOUNT + a)); // approximate running size
      return;
    }
    a = findRoot(old);
    b = findRoot(b);
  }
}

/** Node s touches something structural. Bit 1 = node anchored, bit 2 = (approx.) root anchored. */
fn anchorNode(s: u32) {
  atomicOr(&heap[H_NFLAGS + s], 1u);
  atomicOr(&heap[H_NFLAGS + findRoot(s)], 2u);
}

fn pushSearch(s: u32) {
  let par = cget(C_S_PAR);
  let k = atomicAdd(&ctrl[C_S_NEXT], 1u);
  if (k < HASH_SIZE) { hs(H_SEARCH_Q + (1u - par) * HASH_SIZE + k, s); }
}

@compute @workgroup_size(1)
fn searchBegin() {
  var clear = 0u;
  var seeds = 0u;
  if (cget(C_S_STATE) == 0u) {
    let pend = min(cget(C_PENDING), MAX_SEEDS);
    if (pend > 0u) {
      cset(C_S_STATE, 1u);
      cset(C_PENDING, 0u);
      cset(C_SEED_N, pend);
      cset(C_NODES, 0u);
      cset(C_S_ITERS, 0u);
      cset(C_S_NEXT, 0u);
      cset(C_S_CUR, 0u);
      clear = HASH_SIZE / 64u;
      seeds = groups(pend);
      atomicAdd(&ctrl[S_SEARCHES], 1u);
    }
  }
  setArgs(A_CLEAR, clear);
  setArgs(A_SEEDS, seeds);
}

@compute @workgroup_size(64)
fn searchClear(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  hs(H_HASH_KEYS + i, 0u);
  hs(H_PARENT + i, i);
  hs(H_NFLAGS + i, 0u);
  hs(H_NSIZE + i, 0u);
  hs(H_NCOUNT + i, 0u);
}

@compute @workgroup_size(64)
fn searchSeed(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= cget(C_SEED_N)) { return; }
  let p = posOf(h(H_SEEDS + gid.x));
  let v = readVoxel(p);
  if (v == 0u || structural(v) || isRubble(v)) { return; }
  let ins = hashInsert(keyOf(p));
  if (ins.y == 1u) {
    atomicAdd(&heap[H_NCOUNT + ins.x], 1u);
    pushSearch(ins.x);
  }
}

@compute @workgroup_size(1)
fn searchArgs() {
  if (cget(C_S_STATE) != 1u) {
    setArgs(A_SEARCH, 0u);
    return;
  }
  let par = 1u - cget(C_S_PAR);
  cset(C_S_PAR, par);
  let n = min(cget(C_S_NEXT), HASH_SIZE);
  cset(C_S_CUR, n);
  cset(C_S_NEXT, 0u);
  let iters = atomicAdd(&ctrl[C_S_ITERS], 1u);
  // Done, or over budget (whatever is still unexplored is assumed to be supported).
  if (n == 0u || cget(C_NODES) > HASH_SIZE * 7u / 10u || iters > 4000u) {
    cset(C_S_STATE, 2u);
    setArgs(A_SEARCH, 0u);
    return;
  }
  setArgs(A_SEARCH, groups(n));
}

/** Visit q from node s: returns the node for q (NONE if q is air or anchors s). */
fn visit(s: u32, q: vec3i) -> vec2u {
  if (!inWorld(q)) { return vec2u(NONE, 0u); }
  let w = readVoxel(q);
  // Rubble neither holds things up nor gets carried along (it drops separately).
  if (w == 0u || isRubble(w)) { return vec2u(NONE, 0u); }
  if (structural(w)) {
    anchorNode(s);
    return vec2u(NONE, 0u);
  }
  let ins = hashInsert(keyOf(q));
  if (ins.x == NONE) {
    atomicAdd(&ctrl[S_HASH_FULL], 1u);
    anchorNode(s); // table full: assume supported
    return ins;
  }
  unite(s, ins.x);
  if (ins.y == 1u) { atomicAdd(&heap[H_NCOUNT + findRoot(ins.x)], 1u); }
  return ins;
}

@compute @workgroup_size(64)
fn searchStep(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= cget(C_S_CUR)) { return; }
  let par = cget(C_S_PAR);
  let s = h(H_SEARCH_Q + par * HASH_SIZE + gid.x);
  // Component already known to be anchored: no need to explore it further.
  let r = findRoot(s);
  if ((h(H_NFLAGS + r) & 2u) != 0u) { return; }
  // Huge components (the sofa, the bar cabinet) are assumed to be supported.
  if (h(H_NCOUNT + r) > COMPONENT_BUDGET) {
    atomicAdd(&ctrl[S_BUDGET], 1u);
    anchorNode(s);
    return;
  }
  let p = posOf(h(H_HASH_KEYS + s) - 1u);
  if (p.y == 0) {
    anchorNode(s);
    return;
  }
  // Down first: walk straight down the column. Anything standing on the floor
  // (or on furniture that stands on it) anchors in a single step, so big
  // supported objects never get flooded voxel by voxel.
  var prev = s;
  for (var d = 1; d <= 96; d++) {
    let q = p - vec3i(0, d, 0);
    if (q.y < 0) {
      anchorNode(prev);
      break;
    }
    let ins = visit(prev, q);
    if (ins.x == NONE || ins.y == 0u) { break; }
    pushSearch(ins.x); // its sideways neighbours still need exploring (pruned if anchored)
    prev = ins.x;
  }
  if ((h(H_NFLAGS + findRoot(s)) & 2u) != 0u) { return; }
  for (var k = 0; k < 6; k++) {
    if (k == 3) { continue; } // -y handled by the column walk
    let ins = visit(s, p + dirOf(k));
    if (ins.x != NONE && ins.y == 1u) { pushSearch(ins.x); }
  }
}

@compute @workgroup_size(1)
fn resolveArgs() {
  if (cget(C_S_STATE) == 2u) {
    setArgs(A_R0, groups(cget(C_S_CUR)));
    setArgs(A_RESOLVE, groups(min(cget(C_NODES), HASH_SIZE)));
  } else {
    setArgs(A_R0, 0u);
    setArgs(A_RESOLVE, 0u);
  }
}

/** Search stopped early: unexplored frontier counts as supported. */
@compute @workgroup_size(64)
fn resolveFrontier(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= cget(C_S_CUR)) { return; }
  let s = h(H_SEARCH_Q + cget(C_S_PAR) * HASH_SIZE + gid.x);
  atomicOr(&heap[H_NFLAGS + s], 1u);
}

fn nodeAt(i: u32) -> u32 { return h(H_NODE_LIST + i); }
fn nodeCount() -> u32 { return min(cget(C_NODES), HASH_SIZE); }

/** Exact root flags (bit 4 = anchored) and component sizes. */
@compute @workgroup_size(64)
fn resolveRoots(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= nodeCount()) { return; }
  let s = nodeAt(gid.x);
  let r = findRoot(s);
  atomicAdd(&heap[H_NSIZE + r], 1u);
  if ((h(H_NFLAGS + s) & 1u) != 0u) { atomicOr(&heap[H_NFLAGS + r], 4u); }
}

/** Floating roots get a chunk slot (bits 8..16 = chunk id + 1), tiny ones crumble (bit 8). */
@compute @workgroup_size(64)
fn resolveChunks(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= nodeCount()) { return; }
  let s = nodeAt(gid.x);
  if (findRoot(s) != s) { return; }
  let f = h(H_NFLAGS + s);
  if ((f & 4u) != 0u) { return; }
  if (h(H_NSIZE + s) >= 48u) {
    let id = atomicAdd(&ctrl[C_CHUNKS], 1u);
    if (id < MAX_CHUNKS) {
      let b = H_CHUNKS + id * 16u;
      for (var k = 0u; k < 16u; k++) { hs(b + k, 0u); }
      hs(b + 6u, NONE);
      hs(b, 1u); // falling
      atomicOr(&heap[H_NFLAGS + s], (id + 1u) << 8u);
      atomicAdd(&ctrl[S_ISLANDS], 1u);
      return;
    }
  }
  atomicOr(&heap[H_NFLAGS + s], 8u);
}

/** Floating voxels inside uniform bricks need a real pool brick before they can be lifted out. */
@compute @workgroup_size(64)
fn resolveExpand(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= nodeCount()) { return; }
  let s = nodeAt(gid.x);
  if ((h(H_NFLAGS + findRoot(s)) & 4u) != 0u) { return; }
  let ci = cellOf(posOf(h(H_HASH_KEYS + s) - 1u));
  if ((atomicLoad(&grid[ci]) & UNIFORM_BIT) != 0u) { requestBrick(ci); }
}

/** Lift floating voxels out of the world into their chunk (or into debris). */
@compute @workgroup_size(64)
fn resolveExtract(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= nodeCount()) { return; }
  let s = nodeAt(gid.x);
  let f = h(H_NFLAGS + findRoot(s));
  if ((f & 4u) != 0u) { return; }
  let p = posOf(h(H_HASH_KEYS + s) - 1u);
  let v = takeVoxel(p, ANY_MAT);
  if (v == 0u) { return; }
  dropRubbleAbove(p);
  let cid = (f >> 8u) & 0x1ffu;
  if (cid > 0u && (f & 8u) == 0u) {
    let k = atomicAdd(&ctrl[C_ARENA], 1u);
    if (k < MAX_ARENA) {
      hs(H_ARENA + k * 2u, keyOf(p));
      hs(H_ARENA + k * 2u + 1u, v | ((cid - 1u) << 16u));
      let b = H_CHUNKS + (cid - 1u) * 16u;
      if (fragile(v)) { hs(b + 4u, 1u); }
      atomicAdd(&heap[b + 5u], 1u);
      return;
    }
  }
  // Crumbs: straight to debris.
  atomicAdd(&ctrl[S_BILL], priceDimes(v));
  atomicAdd(&ctrl[S_DESTROYED], 1u);
  let seed = keyOf(p) ^ frameSeed();
  let vel = (rnd3(seed) - vec3f(0.5, 0.0, 0.5)) * vec3f(0.6, 0.3, 0.6) / VS();
  spawn(vec3f(p) + 0.5, vel, v | debrisFlags(v, seed), 4.0 + 3.0 * rnd(seed + 9u));
}

/** Lights whose bulbs went with a chunk follow it. */
@compute @workgroup_size(64)
fn resolveAnchors(@builtin(local_invocation_index) li: u32) {
  if (cget(C_S_STATE) != 2u) { return; }
  for (var i = li; i < NUM_ANCHORS; i += 64u) {
    let a = H_ANCHORS + i * 8u;
    if (h(a + 3u) != 0u) { continue; }
    let slot = hashFind(keyOf(vec3i(i32(h(a)), i32(h(a + 1u)), i32(h(a + 2u)))));
    if (slot == NONE) { continue; }
    let f = h(H_NFLAGS + findRoot(slot));
    if ((f & 4u) != 0u) { continue; }
    let cid = (f >> 8u) & 0x1ffu;
    if (cid > 0u && (f & 8u) == 0u) { hs(a + 3u, cid); }
  }
}

@compute @workgroup_size(1)
fn resolveEnd() {
  if (cget(C_S_STATE) == 2u) { cset(C_S_STATE, 0u); }
}

// ---------------------------------------------------------------------------
// Falling chunks. Their voxels are lifted out of the world while they fall
// (drawn as cubes from the arena), so they can't collide with themselves.
// Chunk table (16 words): 0 state (0 free, 1 falling, 2 landed this frame),
// 1 vy (f32 voxels/s), 2 dy (voxels fallen), 3 frac (f32), 4 fragile,
// 5 voxel count, 6 blockedAt, 7 broke count, 8 steps this frame, 9 landing speed (f32 m/s).

fn chunkBase(id: u32) -> u32 { return H_CHUNKS + id * 16u; }

@compute @workgroup_size(1)
fn chunkArgs() {
  let n = min(cget(C_CHUNKS), MAX_CHUNKS);
  var activeCount = 0u;
  for (var c = 0u; c < n; c++) {
    let b = chunkBase(c);
    var st = h(b);
    if (st == 2u) { hs(b, 0u); }
    // A chunk whose voxels were destroyed before it could be lifted out (or
    // that fell out of the world) is retired so the arena can be recycled.
    if (st == 1u && (h(b + 5u) == 0u || h(b + 2u) > dimsU().y)) {
      hs(b, 0u);
      st = 0u;
    }
    if (st == 1u) { activeCount++; }
  }
  if (activeCount == 0u) {
    cset(C_CHUNKS, 0u);
    cset(C_ARENA, 0u);
  }
  cset(S_FALLING, activeCount);
  setArgs(A_ARENA, groups(min(cget(C_ARENA), MAX_ARENA)));
}

@compute @workgroup_size(256)
fn chunkBegin(@builtin(local_invocation_index) c: u32) {
  if (c >= min(cget(C_CHUNKS), MAX_CHUNKS)) { return; }
  let b = chunkBase(c);
  if (h(b) != 1u) { return; }
  let dt = U.sim.x;
  let vy = min(hf(b + 1u) + U.sim.y * dt, 25.0 / VS());
  let travel = vy * dt + hf(b + 3u);
  let steps = min(u32(floor(travel)), 48u);
  hsf(b + 1u, vy);
  hsf(b + 3u, travel - floor(travel));
  hs(b + 8u, steps);
  hs(b + 6u, NONE);
  hs(b + 7u, 0u);
}

@compute @workgroup_size(64)
fn chunkCollide(@builtin(global_invocation_id) gid: vec3u) {
  let j = gid.x;
  if (j >= min(cget(C_ARENA), MAX_ARENA)) { return; }
  let vv = h(H_ARENA + j * 2u + 1u);
  let v = vv & 0xffffu;
  if (v == 0u) { return; }
  let b = chunkBase(vv >> 16u);
  if (h(b) != 1u) { return; }
  let steps = h(b + 8u);
  if (steps == 0u) { return; }
  let p = posOf(h(H_ARENA + j * 2u)) - vec3i(0, i32(h(b + 2u)), 0);
  let speed = hf(b + 1u) * VS();
  for (var s = 1u; s <= steps; s++) {
    let q = p - vec3i(0, i32(s), 0);
    if (q.y < 0) {
      atomicMin(&heap[b + 6u], s);
      return;
    }
    let w = readVoxel(q);
    if (w != 0u) {
      // Heavy things smash fragile things they land on (chandelier -> glass table).
      if (fragile(w) && speed > 1.5 && atomicAdd(&heap[b + 7u], 1u) < 6u) {
        pushImpact(vec3f(q) + 0.5, 0.02 / VS(), 0.12 / VS(), speed * 0.4 / VS(), vec3f(0.0, 1.0, 0.0), vec3f(0.0, -1.0, 0.0));
      }
      atomicMin(&heap[b + 6u], s);
      return;
    }
  }
}

@compute @workgroup_size(256)
fn chunkStep(@builtin(local_invocation_index) c: u32) {
  if (c >= min(cget(C_CHUNKS), MAX_CHUNKS)) { return; }
  let b = chunkBase(c);
  if (h(b) != 1u) { return; }
  let steps = h(b + 8u);
  if (steps == 0u) { return; }
  let blocked = h(b + 6u);
  var drop = steps;
  if (blocked <= steps) {
    drop = blocked - 1u;
    if (h(b + 7u) > 0u) {
      hsf(b + 1u, hf(b + 1u) * 0.6); // broke through something: keep falling
    } else {
      hsf(b + 9u, hf(b + 1u) * VS());
      hs(b, 2u);
    }
  }
  hs(b + 2u, h(b + 2u) + drop);
}

@compute @workgroup_size(64)
fn chunkLand(@builtin(global_invocation_id) gid: vec3u) {
  let j = gid.x;
  if (j >= min(cget(C_ARENA), MAX_ARENA)) { return; }
  let vv = h(H_ARENA + j * 2u + 1u);
  let v = vv & 0xffffu;
  if (v == 0u) { return; }
  let b = chunkBase(vv >> 16u);
  if (h(b) != 2u) { return; }
  hs(H_ARENA + j * 2u + 1u, 0u);
  let key = h(H_ARENA + j * 2u);
  let p = posOf(key) - vec3i(0, i32(h(b + 2u)), 0);
  let speed = hf(b + 9u);
  if (h(b + 4u) != 0u && speed > 2.0 && fragile(v)) {
    atomicAdd(&ctrl[S_BILL], priceDimes(v));
    atomicAdd(&ctrl[S_DESTROYED], 1u);
    let seed = key ^ frameSeed();
    let vel = vec3f((rnd(seed) - 0.5) * 0.9, rnd(seed + 1u) * 0.5, (rnd(seed + 2u) - 0.5) * 0.9) * speed / VS();
    spawn(vec3f(p) + 0.5, vel, v | debrisFlags(v, seed), 5.0 + 3.0 * rnd(seed + 3u));
    return;
  }
  pushWrite(p, v);
  // Whatever it landed on gets re-checked for support (in case that was floating too).
  if ((hashu(key) & 15u) == 0u) {
    let below = p - vec3i(0, 1, 0);
    let w = readVoxel(below);
    if (w != 0u && !structural(w)) { pushSeed(below); }
  }
}

// ---------------------------------------------------------------------------
// Debris particles: gravity, drag, voxel-DDA collision; resting ones become voxels again.

fn solidAt(p: vec3i) -> bool { return readVoxel(p) != 0u; }

fn pushAlive(i: u32) {
  let slot = atomicAdd(&args[A_DRAWP + 1u], 1u);
  hs(H_ALIVE + slot, i);
}

@compute @workgroup_size(64)
fn particleSim(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= MAX_PARTICLES) { return; }
  var p = particles[i];
  if (p.life <= 0.0) { return; }
  let dt = U.sim.x;
  p.life -= dt;
  if (p.life <= 0.0) {
    particles[i].life = 0.0;
    return;
  }
  if ((p.value & SETTLED) != 0u) {
    particles[i].life = p.life;
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
    particles[i] = p;
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
  if (length(p.vel) < 0.6 / VS() && solidAt(below)) {
    let c = vec3i(floor(p.pos));
    p.pos = vec3f(c) + 0.5;
    p.vel = vec3f(0.0);
    p.value |= SETTLED;
    if ((p.value & DEPOSIT) != 0u && inWorld(c) && pushWrite(c, rubbleOf(p.value & 0xffffu))) {
      p.life = min(p.life, 0.5);
    } else {
      p.life = min(p.life, select(4.0, 0.8, dust));
    }
  }
  if (p.pos.y < -200.0) { p.life = 0.0; }
  particles[i] = p;
  if (p.life > 0.0) { pushAlive(i); }
}

// ---------------------------------------------------------------------------
// Voxel writes (settled debris + landed chunks): request bricks, alloc, write.

@compute @workgroup_size(1)
fn writeArgs() {
  let n = min(cget(C_WRITES), MAX_WRITES);
  cset(C_WRITE_N, n);
  cset(C_WRITES, 0u);
  setArgs(A_WRITE, groups(n));
}

@compute @workgroup_size(64)
fn writeRequest(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= cget(C_WRITE_N)) { return; }
  let ci = cellOf(posOf(h(H_WRITE_LIST + gid.x * 2u)));
  if (atomicLoad(&grid[ci]) == 0u) { requestBrick(ci); }
}

@compute @workgroup_size(64)
fn writeApply(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= cget(C_WRITE_N)) { return; }
  let p = posOf(h(H_WRITE_LIST + gid.x * 2u));
  let v = h(H_WRITE_LIST + gid.x * 2u + 1u);
  // Cell taken (another deposit got there first)? Stack on top, up to 2 voxels.
  for (var up = 0; up < 3; up++) {
    let r = putVoxel(p + vec3i(0, up, 0), v);
    if (r != 1u) { return; }
  }
}

// ---------------------------------------------------------------------------
// Maintenance: per edited brick, refresh coarse density and collapse bricks
// that became all-air or all-one-value (returning them to the free list).

var<workgroup> wgFirst: u32;
var<workgroup> wgOpaque: atomic<u32>;
var<workgroup> wgNonZero: atomic<u32>;
var<workgroup> wgDiff: atomic<u32>;

@compute @workgroup_size(1)
fn maintArgs() {
  setArgs(A_MAINT, min(cget(C_DIRTY), MAX_DIRTY));
}

@compute @workgroup_size(64)
fn maint(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
  if (li == 0u) {
    let ci = h(H_DIRTY_LIST + wid.x);
    wgCi = ci;
    let g = atomicLoad(&grid[ci]);
    wgG = g;
    wgFirst = 0u;
    if (g != 0u && (g & UNIFORM_BIT) == 0u) { wgFirst = atomicLoad(&pool[(g - 1u) * 256u]) & 0xffffu; }
    atomicStore(&wgOpaque, 0u);
    atomicStore(&wgNonZero, 0u);
    atomicStore(&wgDiff, 0u);
  }
  let g = workgroupUniformLoad(&wgG);
  let isPool = g != 0u && (g & UNIFORM_BIT) == 0u;
  if (isPool) {
    let base = (g - 1u) * 256u;
    var opq = 0u;
    var nz = 0u;
    var diff = 0u;
    for (var k = 0u; k < 4u; k++) {
      let w = atomicLoad(&pool[base + li * 4u + k]);
      for (var hh = 0u; hh < 2u; hh++) {
        let v = (w >> (hh * 16u)) & 0xffffu;
        if (v != 0u) {
          nz++;
          if (!isGlass(v & 0xffu)) { opq++; }
        }
        if (v != wgFirst) { diff = 1u; }
      }
    }
    atomicAdd(&wgOpaque, opq);
    atomicAdd(&wgNonZero, nz);
    atomicOr(&wgDiff, diff);
  }
  workgroupBarrier();
  if (li == 0u) {
    let ci = wgCi;
    var density = 0.0;
    if (isPool) {
      density = f32(atomicLoad(&wgOpaque)) / 512.0;
      if (atomicLoad(&wgNonZero) == 0u) {
        atomicStore(&grid[ci], 0u);
        pushFree(g - 1u);
        atomicSub(&superGrid[superOf(ci)], 1u);
      } else if (atomicLoad(&wgDiff) == 0u) {
        atomicStore(&grid[ci], UNIFORM_BIT | wgFirst);
        pushFree(g - 1u);
      }
    } else if (g != 0u) {
      density = select(1.0, 0.0, isGlass(g & 0xffu));
    }
    textureStore(densityOut, cellCoord(ci), vec4f(density, 0.0, 0.0, 1.0));
    atomicAnd(&heap[H_DIRTY_FLAGS + (ci >> 5u)], ~(1u << (ci & 31u)));
  }
}

// ---------------------------------------------------------------------------
// Lights: fade with their bulbs, follow falling fixtures, plus muzzle flashes.
// Light state (16 words): 0-2 pos, 3 radius, 4-6 colour, 7 shadow, 8 factor,
// 9 anchor start, 10 anchor count, 11 seed. Anchor (8 words): 0-2 voxel, 3 chunk+1, 4 original y.

var<workgroup> wgLights: atomic<u32>;

fn storeVec4(i: u32, v: vec4f) {
  let b = H_LIGHTS_OUT + i * 4u;
  hsf(b, v.x); hsf(b + 1u, v.y); hsf(b + 2u, v.z); hsf(b + 3u, v.w);
}

fn emitLight(pos: vec3f, radius: f32, color: vec3f, shadow: f32) {
  let k = atomicAdd(&wgLights, 1u);
  storeVec4(1u + k * 2u, vec4f(pos, radius));
  storeVec4(2u + k * 2u, vec4f(color, shadow));
}

@compute @workgroup_size(64)
fn lights(@builtin(local_invocation_index) li: u32) {
  if (li == 0u) { atomicStore(&wgLights, 0u); }
  workgroupBarrier();
  if (li < NUM_LIGHTS) {
    let b = H_LIGHTS + li * 16u;
    let aStart = h(b + 9u);
    let aCount = h(b + 10u);
    var goal = 1.0;
    var offset = 0.0;
    if (aCount > 0u) {
      var alive = 0u;
      var disp = 0.0;
      for (var a = aStart; a < aStart + aCount; a++) {
        let ab = H_ANCHORS + a * 8u;
        var y = h(ab + 1u);
        let chunk = h(ab + 3u);
        if (chunk > 0u) {
          let cb = chunkBase(chunk - 1u);
          let st = h(cb);
          if (st == 1u) {
            alive++;
            disp -= f32(h(cb + 2u)) + hf(cb + 3u);
            continue;
          }
          if (st == 2u) {
            y = y - h(cb + 2u);
            hs(ab + 1u, y);
          }
          hs(ab + 3u, 0u);
        }
        let v = readVoxel(vec3i(i32(h(ab)), i32(y), i32(h(ab + 2u))));
        if (v != 0u && matA(v & 0xffu).w > 0.0) {
          alive++;
          disp += f32(y) - f32(h(ab + 4u));
        }
      }
      goal = f32(alive) / f32(aCount);
      offset = disp / f32(aCount) * VS();
    }
    var k = hf(b + 8u);
    k += (goal - k) * min(1.0, U.sim.x * 12.0);
    hsf(b + 8u, k);
    // Damaged fixtures flicker.
    let seed = hf(b + 11u);
    if (goal > 0.0 && goal < 0.999) {
      let n = sin(U.time.x * 31.0 + seed) * sin(U.time.x * 17.3 + seed * 3.0);
      if (n > 0.6) { k *= 0.15; }
    }
    if (k >= 0.02) {
      let pos = vec3f(hf(b), hf(b + 1u) + offset, hf(b + 2u));
      emitLight(pos, hf(b + 3u), vec3f(hf(b + 4u), hf(b + 5u), hf(b + 6u)) * k, hf(b + 7u));
    }
  }
  if (li < MAX_FLASHES) {
    let fb = H_FLASHES + li * 8u;
    let t0 = hf(fb + 4u);
    let size = hf(fb + 3u);
    let life = 0.06 + size * 0.25;
    let age = U.time.x - t0;
    if (t0 > 0.0 && age >= 0.0 && age < life) {
      let k = 1.0 - age / life;
      let s = 4.0 + size * 20.0;
      emitLight(vec3f(hf(fb), hf(fb + 1u), hf(fb + 2u)), 1.5 + size * 8.0, vec3f(s, s * 0.7, s * 0.4) * k, 0.0);
    }
  }
  workgroupBarrier();
  if (li == 0u) { storeVec4(0u, vec4f(f32(atomicLoad(&wgLights)), 0.0, 0.0, 0.0)); }
}

@compute @workgroup_size(1)
fn frameEnd() {
  let n = min(cget(C_ARENA), MAX_ARENA);
  aset(A_DRAWC, 36u);
  aset(A_DRAWC + 1u, n);
  aset(A_DRAWC + 2u, 0u);
  aset(A_DRAWC + 3u, 0u);
}
