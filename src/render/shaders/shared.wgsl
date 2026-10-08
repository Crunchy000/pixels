// Binding-free declarations shared by every module (render + simulation).
// Each module declares its own `U` and `mats` bindings; the helpers below use them.

/** Written by the CPU once per frame (input, tool, lighting settings). */
struct Uniforms {
  dims: vec4f,       // voxel dims xyz, w = voxel size (m)
  sunDir: vec4f,     // xyz = direction to sun, w = intensity
  sunColor: vec4f,   // rgb
  ambient: vec4f,    // rgb sky ambient, w = coarse AO strength
  resolution: vec4f, // w, h, 1/w, 1/h
  flags: vec4f,      // x reflections, y lamp shadows, z exposure, w debug view
  sim: vec4f,        // x dt, y gravity (voxels/s^2), z particle capacity, w night factor
  time: vec4f,       // x seconds, y frame, z aspect, w tan(fov/2)
  moveIn: vec4f,     // x forward, y right, z up, w jump
  look: vec4f,       // x yaw delta, y pitch delta, z sprint, w fly-toggle counter
  fire: vec4f,       // x pellets (0 = no shot), y radius, z fragile radius (voxels), w impulse (voxels/s)
  fire2: vec4f,      // x spread, y random seed, z teleport counter, w teleport pitch
  teleport: vec4f,   // xyz feet position (m), w yaw
};

/** Player + camera state. Owned by the GPU (player pass); render passes read it. */
struct Camera {
  invViewProj: mat4x4f,
  viewProj: mat4x4f,
  eye: vec4f,   // xyz eye (m), w = flying
  look: vec4f,  // xyz view direction, w = grounded
  feet: vec4f,  // xyz feet (m), w = yaw
  vel: vec4f,   // xyz velocity (m/s), w = pitch
  state: vec4f, // x last fly-toggle counter seen, y last teleport counter seen
};

/** Debris particle (32 bytes). */
struct Particle {
  pos: vec3f,   // voxel units
  value: u32,   // low 16 bits voxel value, high bits flags
  vel: vec3f,   // voxels / s
  life: f32,    // seconds left
};

const SETTLED: u32 = 0x10000u;
const DEPOSIT: u32 = 0x20000u;
const DUST: u32 = 0x40000u;

const UNIFORM_BIT: u32 = 0x80000000u;

fn idims() -> vec3i { return vec3i(U.dims.xyz); }
fn bdims() -> vec3i { return idims() >> vec3u(3u); }
fn sdims() -> vec3i { return (bdims() + 3) >> vec3u(2u); }

// Material table: 5 vec4 per material id.
//   A = (colour, emission)  B = (colour 2, pattern)  C = (scale, noise, spec, shininess)
//   D = (reflect, glass, liquid, price $)  E = (softness, fragile 0/1/2, structural, debris)
const MAT_STRIDE: u32 = 5u;
fn matA(id: u32) -> vec4f { return mats[id * MAT_STRIDE]; }
fn matB(id: u32) -> vec4f { return mats[id * MAT_STRIDE + 1u]; }
fn matC(id: u32) -> vec4f { return mats[id * MAT_STRIDE + 2u]; }
fn matD(id: u32) -> vec4f { return mats[id * MAT_STRIDE + 3u]; }
fn matE(id: u32) -> vec4f { return mats[id * MAT_STRIDE + 4u]; }

fn isGlass(id: u32) -> bool { return matD(id).y > 0.0; }

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
/** Uniform float in [0, 1) from an integer seed. */
fn rnd(seed: u32) -> f32 { return f32(hashu(seed) & 0xffffffu) / 16777216.0; }

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
