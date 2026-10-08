// Debris + falling chunks: one instanced cube per live particle (compacted
// alive list) or per voxel of a falling chunk (chunk arena), both drawn with
// indirect args the simulation writes. Depth-tested against the ray-marched scene.

@group(1) @binding(0) var<storage, read> partsR: array<Particle>;
@group(1) @binding(1) var<storage, read> heapR: array<u32>;

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

/** Unit cube corner (in [-0.5, 0.5]^3) and face normal for vertex vi of 36. */
fn cubeVertex(vi: u32) -> array<vec3f, 2> {
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
  return array<vec3f, 2>((n + u * ab.x + w * ab.y) * 0.5, n);
}

fn finish(worldV: vec3f, nrm: vec3f, v: u32, seed: u32) -> PVOut {
  var out: PVOut;
  out.pos = C.viewProj * vec4f(worldV * U.dims.w, 1.0);
  let id = v & 0xffu;
  out.color = debrisColor(v, seed);
  out.normal = nrm;
  out.world = worldV;
  out.emissive = matA(id).w;
  out.glass = matD(id).y;
  return out;
}

fn hidden() -> PVOut {
  var out: PVOut;
  out.pos = vec4f(0.0, 0.0, -2.0, 1.0); // outside the clip volume
  return out;
}

@vertex
fn vsParticle(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> PVOut {
  let idx = heapR[H_ALIVE + ii];
  let p = partsR[idx];
  let cv = cubeVertex(vi);
  var local = cv[0];
  var nrm = cv[1];
  let settled = (p.value & SETTLED) != 0u;
  var size = select(1.0, 0.55, (p.value & DUST) != 0u);
  if (!settled) {
    let h = hashu(idx * 2654435761u);
    let axis = normalize(vec3f(f32(h & 255u), f32((h >> 8u) & 255u), f32((h >> 16u) & 255u)) - 127.5);
    let ang = p.life * (4.0 + f32(h >> 28u));
    local = rotAxis(local, axis, ang);
    nrm = rotAxis(nrm, axis, ang);
  } else {
    // Shrink out at the end of a settled particle's life.
    size *= clamp(p.life * 4.0, 0.0, 1.0);
  }
  return finish(p.pos + local * size, nrm, p.value & 0xffffu, idx);
}

@vertex
fn vsChunk(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> PVOut {
  let key = heapR[H_ARENA + ii * 2u];
  let vv = heapR[H_ARENA + ii * 2u + 1u];
  let v = vv & 0xffffu;
  let chunk = vv >> 16u;
  let cbase = H_CHUNKS + chunk * 16u;
  if (v == 0u || heapR[cbase] != 1u) { return hidden(); }
  let dy = f32(heapR[cbase + 2u]) + bitcast<f32>(heapR[cbase + 3u]);
  let d = vec3u(U.dims.xyz);
  let p = vec3f(f32(key % d.x), f32((key / d.x) % d.y), f32(key / (d.x * d.y))) + 0.5 - vec3f(0.0, dy, 0.0);
  let cv = cubeVertex(vi);
  return finish(p + cv[0] * 1.002, cv[1], v, key);
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
  let count = lightCount();
  for (var i = 0; i < count; i++) {
    let lp = lightPos(i);
    let lc = lightCol(i);
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
    let V = normalize(C.eye.xyz / vs - P);
    let R = reflect(-V, N);
    c = c * 0.6 + U.sunColor.rgb * pow(max(dot(R, U.sunDir.xyz), 0.0), 40.0) * 3.0 + vec3f(0.08) * pow(1.0 - abs(dot(V, N)), 3.0);
  }
  return vec4f(c, 1.0);
}
