// Bloom (bright pass + mip chain) and final composite / tonemap.

struct PostParams {
  texel: vec4f,   // source texel size xy, threshold z, intensity w
  grade: vec4f,   // exposure x, bloom strength y, vignette z, srgb-encode w
};

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> P: PostParams;
@group(0) @binding(3) var bloomTex: texture_2d<f32>;
@group(0) @binding(4) var bloomSamp: sampler;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vsFull(@builtin(vertex_index) vi: u32) -> VOut {
  var o: VOut;
  let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  o.pos = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2f(xy.x, 1.0 - xy.y);
  return o;
}

fn box4(uv: vec2f, t: vec2f) -> vec3f {
  let a = textureSampleLevel(srcTex, samp, uv + vec2f(-t.x, -t.y), 0.0).rgb;
  let b = textureSampleLevel(srcTex, samp, uv + vec2f(t.x, -t.y), 0.0).rgb;
  let c = textureSampleLevel(srcTex, samp, uv + vec2f(-t.x, t.y), 0.0).rgb;
  let d = textureSampleLevel(srcTex, samp, uv + vec2f(t.x, t.y), 0.0).rgb;
  return (a + b + c + d) * 0.25;
}

@fragment
fn fsBright(in: VOut) -> @location(0) vec4f {
  let c = min(box4(in.uv, P.texel.xy), vec3f(64.0));
  let br = max(c.r, max(c.g, c.b));
  let k = max(br - P.texel.z, 0.0) / max(br, 1e-4);
  return vec4f(c * k, 1.0);
}

@fragment
fn fsDown(in: VOut) -> @location(0) vec4f {
  return vec4f(box4(in.uv, P.texel.xy), 1.0);
}

fn aces(x: vec3f) -> vec3f {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}

fn srgb(c: vec3f) -> vec3f {
  return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3f(0.0031308));
}

@fragment
fn fsComposite(in: VOut) -> @location(0) vec4f {
  var c = textureSampleLevel(srcTex, samp, in.uv, 0.0).rgb;
  let levels = textureNumLevels(bloomTex);
  var bloom = vec3f(0.0);
  let bs = vec2f(textureDimensions(bloomTex, 0));
  var wsum = 0.0;
  for (var l = 0u; l < levels; l++) {
    let t = 0.5 / (bs / f32(1u << l));
    var s = textureSampleLevel(bloomTex, bloomSamp, in.uv + vec2f(-t.x, -t.y), f32(l)).rgb;
    s += textureSampleLevel(bloomTex, bloomSamp, in.uv + vec2f(t.x, -t.y), f32(l)).rgb;
    s += textureSampleLevel(bloomTex, bloomSamp, in.uv + vec2f(-t.x, t.y), f32(l)).rgb;
    s += textureSampleLevel(bloomTex, bloomSamp, in.uv + vec2f(t.x, t.y), f32(l)).rgb;
    let w = 1.0 + f32(l) * 0.35;
    bloom += s * 0.25 * w;
    wsum += w;
  }
  bloom /= max(wsum, 1.0);
  c = c + bloom * P.grade.y;
  c *= P.grade.x;
  c = aces(c);
  let v = in.uv - 0.5;
  c *= 1.0 - P.grade.z * dot(v, v) * 1.6;
  if (P.grade.w > 0.5) { c = srgb(c); }
  // Tiny dither to hide banding.
  let n = fract(sin(dot(in.pos.xy, vec2f(12.9898, 78.233))) * 43758.5453);
  c += (n - 0.5) / 255.0;
  return vec4f(c, 1.0);
}
