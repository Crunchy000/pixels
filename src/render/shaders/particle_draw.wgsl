// Debris rendering: one instanced cube per live particle (via the compacted
// alive list + indirect draw), depth-tested against the ray-marched scene.

@group(1) @binding(0) var<storage, read> partsR: array<Particle>;
@group(1) @binding(1) var<storage, read> aliveR: array<u32>;

struct PVOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) normal: vec3f,
  @location(2) world: vec3f,
  @location(3) @interpolate(flat) emissive: f32,
  @location(4) @interpolate(flat) glass: f32,
};

fn rotAxis(v: vec3f, axis: vec3f, ang: f32) -> vec3f {
  let c = cos(ang);
  let s = sin(ang);
  return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

@vertex
fn vsParticle(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> PVOut {
  var out: PVOut;
  let idx = aliveR[ii];
  let p = partsR[idx];
  let face = vi / 6u;
  let corner = vi % 6u;
  var n = vec3f(0.0);
  var u = vec3f(0.0);
  var w = vec3f(0.0);
  switch face {
    case 0u: { n = vec3f(1.0, 0.0, 0.0); u = vec3f(0.0, 1.0, 0.0); w = vec3f(0.0, 0.0, 1.0); }
    case 1u: { n = vec3f(-1.0, 0.0, 0.0); u = vec3f(0.0, 0.0, 1.0); w = vec3f(0.0, 1.0, 0.0); }
    case 2u: { n = vec3f(0.0, 1.0, 0.0); u = vec3f(0.0, 0.0, 1.0); w = vec3f(1.0, 0.0, 0.0); }
    case 3u: { n = vec3f(0.0, -1.0, 0.0); u = vec3f(1.0, 0.0, 0.0); w = vec3f(0.0, 0.0, 1.0); }
    case 4u: { n = vec3f(0.0, 0.0, 1.0); u = vec3f(1.0, 0.0, 0.0); w = vec3f(0.0, 1.0, 0.0); }
    default: { n = vec3f(0.0, 0.0, -1.0); u = vec3f(0.0, 1.0, 0.0); w = vec3f(1.0, 0.0, 0.0); }
  }
  var ab = vec2f(-1.0, -1.0);
  switch corner {
    case 1u: { ab = vec2f(1.0, -1.0); }
    case 2u, 4u: { ab = vec2f(1.0, 1.0); }
    case 5u: { ab = vec2f(-1.0, 1.0); }
    default: {}
  }
  var local = (n + u * ab.x + w * ab.y) * 0.5;
  let settled = (p.value & SETTLED) != 0u;
  let dust = (p.value & DUST) != 0u;
  var size = 1.0;
  if (dust) { size = 0.55; }
  var nrm = n;
  if (!settled) {
    let h = hashu(idx * 2654435761u);
    let axis = normalize(vec3f(f32(h & 255u), f32((h >> 8u) & 255u), f32((h >> 16u) & 255u)) - 127.5);
    let ang = p.life * (4.0 + f32(h >> 28u));
    local = rotAxis(local, axis, ang);
    nrm = rotAxis(n, axis, ang);
  } else {
    // Shrink out at the end of a settled particle's life.
    size *= clamp(p.life * 4.0, 0.0, 1.0);
  }
  let worldV = p.pos + local * size;
  let world = worldV * U.dims.w;
  out.pos = U.viewProj * vec4f(world, 1.0);
  let v = p.value & 0xffffu;
  let id = v & 0xffu;
  out.color = debrisColor(v, idx);
  out.normal = nrm;
  out.world = worldV;
  out.emissive = matA(id).w;
  out.glass = matD(id).y;
  return out;
}

@fragment
fn fsParticle(in: PVOut) -> @location(0) vec4f {
  if (in.emissive > 0.0) {
    return vec4f(in.color * in.emissive, 1.0);
  }
  let N = normalize(in.normal);
  let P = in.world;
  let vs = U.dims.w;
  var light = mix(vec3f(0.35, 0.25, 0.22), U.ambient.rgb, N.y * 0.5 + 0.5) * 0.8;
  light += U.sunColor.rgb * U.sunDir.w * max(dot(N, U.sunDir.xyz), 0.0) * 0.6;
  let count = i32(U.sunColor.w);
  for (var i = 0; i < count; i++) {
    let lp = lights[i * 2];
    let lc = lights[i * 2 + 1];
    let L = lp.xyz / vs - P;
    let d = length(L);
    let radius = lp.w / vs;
    if (d > radius) { continue; }
    let x = d / radius;
    let dm = d * vs;
    light += lc.rgb * max(dot(N, L / d), 0.0) * (1.0 - x * x) * (1.0 - x * x) / (1.0 + dm * dm * 1.5);
  }
  var c = in.color * light;
  if (in.glass > 0.0) {
    // Glass shards: bright speculars that glitter as they tumble.
    let V = normalize(U.camPos.xyz / vs - P);
    let R = reflect(-V, N);
    c = c * 0.6 + U.sunColor.rgb * pow(max(dot(R, U.sunDir.xyz), 0.0), 40.0) * 3.0 + vec3f(0.08) * pow(1.0 - abs(dot(V, N)), 3.0);
  }
  return vec4f(c, 1.0);
}
