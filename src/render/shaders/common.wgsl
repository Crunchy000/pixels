// Shared bindings + voxel access + material colours.

struct Uniforms {
  invViewProj: mat4x4f,
  viewProj: mat4x4f,
  camPos: vec4f,     // xyz metres, w = time (s)
  dims: vec4f,       // voxel dims xyz, w = voxel size (m)
  sunDir: vec4f,     // xyz = direction to sun, w = intensity
  sunColor: vec4f,   // rgb, w = light count
  ambient: vec4f,    // rgb sky ambient, w = coarse AO strength
  resolution: vec4f, // w, h, 1/w, 1/h
  flags: vec4f,      // x reflections, y point-light shadows, z exposure, w frame
  sim: vec4f,        // x dt, y gravity (voxels/s^2), z particle capacity, w night factor
};

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> grid: array<u32>;
@group(0) @binding(2) var<storage, read> pool: array<u32>;
@group(0) @binding(3) var<storage, read> mats: array<vec4f>;
@group(0) @binding(4) var<storage, read> lights: array<vec4f>;
@group(0) @binding(5) var densityTex: texture_3d<f32>;
@group(0) @binding(6) var linSamp: sampler;

const UNIFORM_BIT: u32 = 0x80000000u;

fn idims() -> vec3i { return vec3i(U.dims.xyz); }
fn bdims() -> vec3i { return idims() >> vec3u(3u); }

fn cellEntry(b: vec3i) -> u32 {
  let bd = bdims();
  return grid[u32(b.x + bd.x * (b.y + bd.y * b.z))];
}

fn brickVoxel(g: u32, l: vec3i) -> u32 {
  if ((g & UNIFORM_BIT) != 0u) { return g & 0xffffu; }
  let li = u32(l.x | (l.y << 3u) | (l.z << 6u));
  let idx = (g - 1u) * 512u + li;
  let w = pool[idx >> 1u];
  return (w >> ((idx & 1u) * 16u)) & 0xffffu;
}

fn voxelAt(p: vec3i) -> u32 {
  if (any(p < vec3i(0)) || any(p >= idims())) { return 0u; }
  let g = cellEntry(p >> vec3u(3u));
  if (g == 0u) { return 0u; }
  return brickVoxel(g, p & vec3i(7));
}

fn matA(id: u32) -> vec4f { return mats[id * 4u]; }
fn matB(id: u32) -> vec4f { return mats[id * 4u + 1u]; }
fn matC(id: u32) -> vec4f { return mats[id * 4u + 2u]; }
fn matD(id: u32) -> vec4f { return mats[id * 4u + 3u]; }

fn isGlass(id: u32) -> bool { return matD(id).y > 0.0; }

fn opaqueAt(p: vec3i) -> f32 {
  let v = voxelAt(p);
  if (v == 0u || isGlass(v & 0xffu)) { return 0.0; }
  return 1.0;
}

// ---------------------------------------------------------------------------
// Hashing / noise

fn hashu(x: u32) -> u32 {
  var h = x;
  h ^= h >> 16u; h *= 0x7feb352du;
  h ^= h >> 15u; h *= 0x846ca68bu;
  h ^= h >> 16u;
  return h;
}
fn hash3i(p: vec3i) -> u32 {
  return hashu((u32(p.x) * 0x9E3779B1u) ^ hashu((u32(p.y) * 0x85EBCA77u) ^ hashu(u32(p.z) * 0xC2B2AE3Du)));
}
fn hashf(p: vec3i) -> f32 { return f32(hash3i(p) & 0xffffffu) / 16777215.0; }
fn hash1f(x: f32) -> f32 { return f32(hashu(u32(i32(floor(x)) + 100000)) & 0xffffu) / 65535.0; }
fn hash2f(x: f32, y: f32) -> f32 {
  return f32(hashu((u32(i32(floor(x)) + 100000) * 0x27d4eb2du) ^ hashu(u32(i32(floor(y)) + 7777))) & 0xffffu) / 65535.0;
}

fn vnoise(p: vec3f) -> f32 {
  let i = vec3i(floor(p));
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = hashf(i);
  let b = hashf(i + vec3i(1, 0, 0));
  let c = hashf(i + vec3i(0, 1, 0));
  let d = hashf(i + vec3i(1, 1, 0));
  let e = hashf(i + vec3i(0, 0, 1));
  let f1 = hashf(i + vec3i(1, 0, 1));
  let g = hashf(i + vec3i(0, 1, 1));
  let h = hashf(i + vec3i(1, 1, 1));
  return mix(mix(mix(a, b, u.x), mix(c, d, u.x), u.y), mix(mix(e, f1, u.x), mix(g, h, u.x), u.y), u.z);
}

fn fbm(p: vec3f) -> f32 {
  return vnoise(p) * 0.5 + vnoise(p * 2.03 + 17.0) * 0.3 + vnoise(p * 4.1 + 41.0) * 0.2;
}

fn hsv(h: f32, s: f32, v: f32) -> vec3f {
  let k = vec3f(1.0, 2.0 / 3.0, 1.0 / 3.0);
  let p = abs(fract(vec3f(h) + k) * 6.0 - 3.0);
  return v * mix(vec3f(1.0), clamp(p - 1.0, vec3f(0.0), vec3f(1.0)), s);
}

// ---------------------------------------------------------------------------
// Procedural material colour for voxel value v at voxel coordinate vp.

fn marble(q: vec3f, A: vec3f, B: vec3f) -> vec3f {
  let f = fbm(q * 1.3);
  let v = sin((q.x + q.y * 0.6 + q.z * 0.8) * 2.2 + f * 7.0);
  let vein = pow(1.0 - abs(v), 10.0);
  let cloud = fbm(q * 3.0 + 5.0);
  return mix(A * (0.93 + 0.07 * cloud), B, vein * 0.85);
}

fn materialColor(v: u32, vp: vec3i) -> vec3f {
  let id = v & 0xffu;
  let shade = (v >> 8u) & 0xffu;
  let A = matA(id).rgb;
  let B = matB(id).rgb;
  let C = matC(id);
  let s = max(C.x, 1e-4);
  let wp = (vec3f(vp) + 0.5) * U.dims.w;
  var c = A;
  if (shade > 0u) {
    c = mix(A, B, f32(shade - 1u) / 254.0);
  } else {
    let pat = u32(matB(id).w + 0.5);
    switch pat {
      case 1u: { c = marble(wp / s, A, B); }
      case 2u, 3u, 4u: {
        var r = length(wp.yz);
        var along = wp.x;
        if (pat == 3u) { r = length(wp.xy); along = wp.z; }
        if (pat == 4u) { r = length(wp.xz); along = wp.y; }
        let n = vnoise(vec3f(along * 3.0, r * 20.0, 0.0)) * 0.8 + vnoise(wp * 40.0) * 0.15;
        let ring = 0.5 + 0.5 * sin((r / s + n * 2.0) * 6.2831);
        c = mix(A, B, smoothstep(0.2, 0.95, ring));
      }
      case 5u: {
        let tv = max(1, i32(round(s / U.dims.w)));
        let tile = vec2i(vp.x / tv, vp.z / tv);
        let lx = vp.x - tile.x * tv;
        let lz = vp.z - tile.y * tv;
        let seed = vec3f(f32(tile.x) * 7.31, 0.0, f32(tile.y) * 3.77);
        c = marble(wp / (s * 0.45) + seed, A, B);
        if (((tile.x + tile.y) & 1) == 1) { c *= vec3f(0.94, 0.93, 0.92); }
        if (lx == 0 || lz == 0) { c = vec3f(0.12, 0.1, 0.09); }
      }
      case 6u: {
        let q = wp.xz / s;
        let cell = floor(q);
        let f = fract(q) - 0.5;
        let r = length(f);
        let ang = atan2(f.y, f.x);
        let petals = 0.22 + 0.07 * cos(ang * 6.0 + floor(hash2f(cell.x, cell.y) * 2.0) * 0.5236);
        c = A;
        let teal = vec3f(0.02, 0.18, 0.2);
        let purple = vec3f(0.12, 0.02, 0.18);
        if (abs(abs(f.x) + abs(f.y) - 0.46) < 0.035) { c = teal; }
        if (abs(r - petals) < 0.03) { c = B; }
        if (r < 0.07) { c = B * 1.1; }
        let g = fract(q * 2.0 + 0.25) - 0.5;
        if (length(g) < 0.06 && r > 0.3) { c = purple; }
        let sw = sin(q.x * 9.0 + sin(q.y * 7.0) * 1.5);
        if (abs(sw) < 0.08 && r > 0.32 && r < 0.42) { c = B * 0.8; }
      }
      case 7u: {
        let t = fract((wp.x + wp.z) / s);
        c = mix(A, B, smoothstep(0.45, 0.5, t) * (1.0 - smoothstep(0.95, 1.0, t)));
      }
      case 8u: {
        let weave = f32((vp.x + vp.y + vp.z) & 1);
        c = mix(A, B, weave * 0.35 + vnoise(wp / (s * 20.0)) * 0.4);
      }
      case 9u: { c = mix(A, B, step(0.75, hashf(vp))); }
      case 10u: { c = mix(A, B, vnoise(vec3f(wp.x / s * 0.3, wp.y / s * 6.0, wp.z / s * 0.3)) * 0.6); }
      case 11u: {
        let row = floor(wp.y / s);
        let col = floor((wp.x + wp.z) / s);
        let spin = floor(U.camPos.w * 6.0 + col * 1.7);
        let sym = hash2f(row + spin, col);
        c = hsv(fract(sym * 3.7 + U.camPos.w * 0.05), 0.85, 0.6 + 0.4 * sym);
        let edge = fract(wp.y / s);
        if (edge < 0.08) { c *= 0.15; }
      }
      case 12u: {
        let t = fract(wp.y / s + U.camPos.w * 0.03);
        let wave = 0.5 + 0.5 * sin((wp.x + wp.z) * 3.0 + U.camPos.w * 1.5 + wp.y * 5.0);
        c = mix(A, B, t * 0.6 + wave * 0.4);
        if ((vp.y & 3) == 0) { c *= 0.75; }
      }
      case 13u: { c = mix(A, B, vnoise(wp / s)); }
      default: {}
    }
  }
  let n = hashf(vp);
  c *= 1.0 + C.y * (n - 0.5) * 2.0;
  return max(c, vec3f(0.0));
}

/** Flat colour for debris (no position-dependent pattern). */
fn debrisColor(v: u32, seed: u32) -> vec3f {
  let id = v & 0xffu;
  let shade = (v >> 8u) & 0xffu;
  let A = matA(id).rgb;
  let B = matB(id).rgb;
  var c = mix(A, B, f32(hashu(seed) & 255u) / 255.0 * 0.5);
  if (shade > 0u) { c = mix(A, B, f32(shade - 1u) / 254.0); }
  return c;
}
