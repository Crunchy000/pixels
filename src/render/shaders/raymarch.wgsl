// Primary visibility: one ray per pixel through the brick map (two-level DDA),
// then shading with sun + point-light shadow rays, coarse + fine AO, glass and
// optional glossy reflections. Writes HDR colour and a real depth value so the
// rasterised debris composites correctly.

struct Hit {
  hit: bool,
  t: f32,
  voxel: vec3i,
  normal: vec3f,
  value: u32,
  trans: vec3f, // transmittance accumulated through glass
  glow: vec3f,  // light added by glass surfaces (fresnel / glints)
};

const MAX_OUTER: i32 = 256;

// Debug counters (step heatmap).
var<private> gOuterSteps: u32 = 0u;
var<private> gInnerSteps: u32 = 0u;

fn safeDir(d: vec3f) -> vec3f {
  return select(d, sign(d + vec3f(1e-30)) * 1e-6, abs(d) < vec3f(1e-6));
}

fn glassSurface(h: ptr<function, Hit>, id: u32, n: vec3f, rd: vec3f, shadow: bool) {
  let tint = matA(id).rgb;
  let g = matD(id).y;
  if (!shadow) {
    let cosT = abs(dot(n, rd));
    let F = 0.04 + 0.96 * pow(1.0 - cosT, 5.0);
    let r = reflect(rd, n);
    let glint = pow(max(dot(r, U.sunDir.xyz), 0.0), matC(id).w) * U.sunColor.rgb * U.sunDir.w * 2.0;
    (*h).glow += (*h).trans * (F * skyColor(r) * 0.35 + glint * matC(id).z);
    (*h).trans *= (1.0 - F);
  }
  (*h).trans *= mix(vec3f(1.0), tint, 0.6) * g;
}

/**
 * March from ro (voxel units) along rd until an opaque voxel or maxT.
 * Glass voxels tint `trans` and the ray carries on.
 */
fn march(ro: vec3f, rdIn: vec3f, maxT: f32, shadow: bool) -> Hit {
  var h: Hit;
  h.hit = false;
  h.trans = vec3f(1.0);
  h.glow = vec3f(0.0);
  let rd = safeDir(rdIn);
  let inv = 1.0 / rd;
  let dims = U.dims.xyz;
  let t0 = (vec3f(0.0) - ro) * inv;
  let t1 = (dims - ro) * inv;
  let tmn = min(t0, t1);
  let tmx = max(t0, t1);
  var t = max(max(tmn.x, tmn.y), max(tmn.z, 0.0));
  let tExit = min(min(tmx.x, tmx.y), min(tmx.z, maxT));
  if (t >= tExit) { return h; }

  let stp = vec3i(sign(rd));
  let stpF = vec3f(stp);
  let pos = step(vec3f(0.0), rd);
  // Normal of the face we entered the volume through.
  var normal = vec3f(0.0);
  if (t > 0.0) {
    if (tmn.x >= tmn.y && tmn.x >= tmn.z) { normal = vec3f(-stpF.x, 0.0, 0.0); }
    else if (tmn.y >= tmn.z) { normal = vec3f(0.0, -stpF.y, 0.0); }
    else { normal = vec3f(0.0, 0.0, -stpF.z); }
  }

  // Three nested DDAs: super cells (32^3 voxels) -> bricks (8^3) -> voxels.
  let bd = bdims();
  let sd = sdims();
  let p0 = ro + rd * (t + 1e-3);
  var sc = clamp(vec3i(floor(p0 / 32.0)), vec3i(0), sd - 1);
  let tDeltaS = abs(inv) * 32.0;
  var tMaxS = ((vec3f(sc) + pos) * 32.0 - ro) * inv;
  let tDeltaB = abs(inv) * 8.0;
  let tDeltaV = abs(inv);
  var inGlass = false;
  var glassId = 0u;

  for (var i = 0; i < MAX_OUTER; i++) {
    gOuterSteps += 1u;
    if (superGrid[u32(sc.x + sd.x * (sc.y + sd.y * sc.z))] != 0u) {
      let sLo = sc * 4;
      let sHi = min(sLo + 3, bd - 1);
      var bc = clamp(vec3i(floor((ro + rd * (t + 1e-3)) / 8.0)), sLo, sHi);
      var tMaxB = ((vec3f(bc) + pos) * 8.0 - ro) * inv;
      var tb = t;
      var nb = normal;
      for (var j = 0; j < 14; j++) {
        gOuterSteps += 1u;
        let g = grid[u32(bc.x + bd.x * (bc.y + bd.y * bc.z))];
        if (g != 0u) {
          let lo = bc * 8;
          var vc = clamp(vec3i(floor(ro + rd * (tb + 1e-3))), lo, lo + 7);
          // Fast path: uniform opaque brick => hit at entry.
          if ((g & UNIFORM_BIT) != 0u && !isGlass(g & 0xffu)) {
            h.hit = true; h.t = tb; h.voxel = vc; h.normal = nb; h.value = g & 0xffffu;
            return h;
          }
          var tMaxV = (vec3f(vc) + pos - ro) * inv;
          var tv = tb;
          var n = nb;
          for (var k = 0; k < 25; k++) {
            gInnerSteps += 1u;
            let v = brickVoxel(g, vc - lo);
            if (v != 0u) {
              let id = v & 0xffu;
              if (isGlass(id)) {
                if (!inGlass || id != glassId) { glassSurface(&h, id, n, rd, shadow); }
                inGlass = true;
                glassId = id;
                if (max(h.trans.x, max(h.trans.y, h.trans.z)) < 0.02) {
                  h.hit = true; h.t = tv; h.voxel = vc; h.normal = n; h.value = v;
                  return h;
                }
              } else {
                h.hit = true; h.t = tv; h.voxel = vc; h.normal = n; h.value = v;
                return h;
              }
            } else {
              inGlass = false;
            }
            if (tMaxV.x < tMaxV.y && tMaxV.x < tMaxV.z) {
              tv = tMaxV.x; vc.x += stp.x; tMaxV.x += tDeltaV.x; n = vec3f(-stpF.x, 0.0, 0.0);
            } else if (tMaxV.y < tMaxV.z) {
              tv = tMaxV.y; vc.y += stp.y; tMaxV.y += tDeltaV.y; n = vec3f(0.0, -stpF.y, 0.0);
            } else {
              tv = tMaxV.z; vc.z += stp.z; tMaxV.z += tDeltaV.z; n = vec3f(0.0, 0.0, -stpF.z);
            }
            if (tv > tExit) { return h; }
            if (any(vc < lo) || any(vc > lo + 7)) { break; }
          }
        } else {
          inGlass = false;
        }
        if (tMaxB.x < tMaxB.y && tMaxB.x < tMaxB.z) {
          tb = tMaxB.x; bc.x += stp.x; tMaxB.x += tDeltaB.x; nb = vec3f(-stpF.x, 0.0, 0.0);
        } else if (tMaxB.y < tMaxB.z) {
          tb = tMaxB.y; bc.y += stp.y; tMaxB.y += tDeltaB.y; nb = vec3f(0.0, -stpF.y, 0.0);
        } else {
          tb = tMaxB.z; bc.z += stp.z; tMaxB.z += tDeltaB.z; nb = vec3f(0.0, 0.0, -stpF.z);
        }
        if (tb > tExit) { return h; }
        if (any(bc < sLo) || any(bc > sHi)) { break; }
      }
    } else {
      inGlass = false;
    }
    if (tMaxS.x < tMaxS.y && tMaxS.x < tMaxS.z) {
      t = tMaxS.x; sc.x += stp.x; tMaxS.x += tDeltaS.x; normal = vec3f(-stpF.x, 0.0, 0.0);
    } else if (tMaxS.y < tMaxS.z) {
      t = tMaxS.y; sc.y += stp.y; tMaxS.y += tDeltaS.y; normal = vec3f(0.0, -stpF.y, 0.0);
    } else {
      t = tMaxS.z; sc.z += stp.z; tMaxS.z += tDeltaS.z; normal = vec3f(0.0, 0.0, -stpF.z);
    }
    if (t > tExit || any(sc < vec3i(0)) || any(sc >= sd)) { break; }
  }
  return h;
}

// ---------------------------------------------------------------------------
// Sky: Las Vegas at dusk, seen from a high-rise.

fn skyline(az: f32, e: f32, layer: f32, night: f32) -> vec4f {
  // One ring of buildings at "infinity". Layer 0 = near casino towers (sparse,
  // tall), layer 2 = distant low-rise. The camera is ~150 m up, so most roofs
  // sit below the horizon.
  let cellW = select(select(0.012, 0.03, layer < 1.5), 0.075, layer < 0.5);
  let cid = floor(az / cellW);
  let present = hash2f(cid, layer * 13.0 + 1.0);
  if (layer < 0.5 && present < 0.55) { return vec4f(0.0); }
  let hseed = hash2f(cid, layer * 31.0 + 3.0);
  var lo = -0.045; var hi = -0.008;
  if (layer < 1.5) { lo = -0.07; hi = 0.025; }
  if (layer < 0.5) { lo = -0.04; hi = 0.13; }
  let top = mix(lo, hi, pow(hseed, 1.6));
  let fx = fract(az / cellW);
  let inset = select(0.1, 0.22, layer < 0.5);
  if (fx < inset || fx > 1.0 - inset) { if (hash2f(cid, 9.0 + layer) > 0.3) { return vec4f(0.0); } }
  // Building bases: farther layers sit closer to the horizon (camera ~150 m up).
  let base = select(select(-0.05, -0.095, layer < 1.5), -0.22, layer < 0.5);
  if (e > top || e < base) { return vec4f(0.0); }
  let depth = 1.0 - layer * 0.3;
  var c = vec3f(0.018, 0.016, 0.032) * (0.7 + 0.6 * hash2f(cid, 5.0)) * depth;
  // Lit windows on a coherent grid.
  let cols = select(6.0, 14.0, layer < 0.5);
  let rowsPerRad = select(260.0, 520.0, layer < 0.5);
  let wx = (fx - inset) / (1.0 - 2.0 * inset) * cols;
  let wy = (top - e) * rowsPerRad;
  let cell = vec2f(floor(wx), floor(wy));
  let lit = hash2f(cell.x + cid * 31.0, cell.y + layer * 101.0);
  let fw = fract(vec2f(wx, wy));
  let thresh = mix(0.72, 0.5, night);
  if (lit > thresh && fw.x > 0.2 && fw.x < 0.8 && fw.y > 0.35 && fw.y < 0.85 && wx >= 0.0) {
    let warm = hash2f(cid, 77.0);
    let wc = select(vec3f(1.0, 0.72, 0.42), vec3f(0.55, 0.8, 1.0), warm > 0.8);
    c += wc * (0.12 + 0.25 * (lit - thresh)) * depth * (0.7 + night);
  }
  // Casino towers: neon edge strips + glowing crowns.
  let hue = hash2f(cid, 41.0);
  if (layer < 0.5 && hue > 0.45) {
    let neon = hsv(hue * 2.3, 0.75, 1.0) * (0.8 + night * 1.2);
    if (abs(fx - inset) < 0.015 || abs(fx - (1.0 - inset)) < 0.015) { c += neon * 0.6; }
    if (top - e < 0.0035) { c += neon * 1.4; }
  }
  return vec4f(c, 1.0);
}

fn skyColor(rd: vec3f) -> vec3f {
  let e = rd.y;
  let night = U.sim.w;
  let horizon = mix(vec3f(1.25, 0.5, 0.22), vec3f(0.12, 0.06, 0.16), night);
  let mid = mix(vec3f(0.55, 0.22, 0.38), vec3f(0.04, 0.03, 0.09), night);
  let zenith = mix(vec3f(0.07, 0.07, 0.25), vec3f(0.005, 0.006, 0.02), night);
  var c = mix(horizon, mid, smoothstep(0.0, 0.18, e));
  c = mix(c, zenith, smoothstep(0.12, 0.7, e));
  let sd = max(dot(rd, U.sunDir.xyz), 0.0);
  c += U.sunColor.rgb * (pow(sd, 1500.0) * 40.0 + pow(sd, 12.0) * 0.35) * (1.0 - night);
  // Stars.
  if (e > 0.05) {
    let sp = vec3i(floor(rd * 400.0));
    if (hashf(sp) > 0.9985) { c += vec3f(0.6) * (0.3 + night); }
  }
  let az = atan2(rd.z, rd.x);
  // Luxor-style sky beam.
  let beamAz = 0.35;
  let bw = abs(az - beamAz);
  if (e > -0.02) { c += vec3f(0.6, 0.75, 1.0) * exp(-bw * 2500.0) * (0.5 + 1.5 * night) * smoothstep(0.9, 0.0, e); }
  // Pyramid silhouette under the beam.
  let pyr = -0.02 - bw * 1.6;
  if (e < pyr && e > -0.12) {
    return vec3f(0.03, 0.025, 0.04) + vec3f(0.4, 0.3, 0.15) * step(0.97, fract((pyr - e) * 140.0)) * 0.3;
  }
  // Strat-style needle tower.
  let towerAz = 1.05;
  let tw = abs(az - towerAz);
  if ((tw < 0.0025 && e < 0.16) || (tw < 0.008 && abs(e - 0.15) < 0.008)) {
    var tc = vec3f(0.03, 0.03, 0.05);
    if (abs(e - 0.15) < 0.008) { tc += vec3f(0.8, 0.3, 0.6) * step(0.5, fract(az * 900.0)); }
    return tc;
  }
  for (var l = 0; l < 3; l++) {
    let s = skyline(az + f32(l) * 1.7, e, f32(l), night);
    // Haze: farther layers pick up some sky colour.
    if (s.w > 0.0) { return s.rgb + c * (0.03 + 0.05 * f32(l)); }
  }
  if (e < 0.0) {
    // City grid ~150 m below: 100 m blocks with sodium-lit streets.
    let gp = rd.xz * (1.5 / max(-e, 1e-3));
    var city = vec3f(0.012, 0.01, 0.02);
    let f = abs(fract(gp) - 0.5);
    let street = max(f.x, f.y);
    let w = 0.012 / max(-e * 8.0, 0.05);
    let lights = mix(0.25, 0.55, night);
    if (street > 0.5 - w) { city += vec3f(1.0, 0.55, 0.2) * lights; }
    let lot = floor(gp * 4.0);
    if (hash2f(lot.x, lot.y) > 0.9) { city += hsv(hash2f(lot.x + 3.0, lot.y), 0.5, 1.0) * 0.25 * (0.5 + night); }
    return mix(city, c, exp(e * 25.0));
  }
  return c;
}

// ---------------------------------------------------------------------------
// Lighting

fn fineAO(vox: vec3i, N: vec3f, P: vec3f) -> f32 {
  let n = vec3i(round(N));
  var u = vec3i(1, 0, 0);
  var w = vec3i(0, 0, 1);
  if (abs(N.x) > 0.5) { u = vec3i(0, 1, 0); w = vec3i(0, 0, 1); }
  else if (abs(N.z) > 0.5) { u = vec3i(1, 0, 0); w = vec3i(0, 1, 0); }
  let b = vox + n;
  let su0 = opaqueAt(b - u);
  let su1 = opaqueAt(b + u);
  let sw0 = opaqueAt(b - w);
  let sw1 = opaqueAt(b + w);
  let c00 = opaqueAt(b - u - w);
  let c10 = opaqueAt(b + u - w);
  let c01 = opaqueAt(b - u + w);
  let c11 = opaqueAt(b + u + w);
  let a00 = select(3.0 - (su0 + sw0 + c00), 0.0, su0 + sw0 > 1.5) / 3.0;
  let a10 = select(3.0 - (su1 + sw0 + c10), 0.0, su1 + sw0 > 1.5) / 3.0;
  let a01 = select(3.0 - (su0 + sw1 + c01), 0.0, su0 + sw1 > 1.5) / 3.0;
  let a11 = select(3.0 - (su1 + sw1 + c11), 0.0, su1 + sw1 > 1.5) / 3.0;
  let f = clamp(P - vec3f(vox), vec3f(0.0), vec3f(1.0));
  let fu = dot(f, vec3f(u));
  let fw = dot(f, vec3f(w));
  let ao = mix(mix(a00, a10, fu), mix(a01, a11, fu), fw);
  return mix(0.35, 1.0, ao);
}

fn coarseAO(P: vec3f, N: vec3f) -> f32 {
  let inv = 1.0 / U.dims.xyz;
  var occ = 0.0;
  occ += textureSampleLevel(densityTex, linSamp, (P + N * 10.0) * inv, 0.0).r * 0.55;
  occ += textureSampleLevel(densityTex, linSamp, (P + N * 22.0) * inv, 0.0).r * 0.35;
  occ += textureSampleLevel(densityTex, linSamp, (P + N * 44.0) * inv, 0.0).r * 0.3;
  return clamp(1.0 - occ * U.ambient.w, 0.12, 1.0);
}

fn shade(h: Hit, ro: vec3f, rd: vec3f, quality: i32) -> vec3f {
  let id = h.value & 0xffu;
  let albedo = materialColor(h.value, h.voxel);
  let A = matA(id);
  let C = matC(id);
  if (A.w > 0.0) {
    return albedo * A.w;
  }
  let N = h.normal;
  let P = ro + rd * h.t;
  let vs = U.dims.w;
  var ao = coarseAO(P, N);
  if (quality > 0) { ao *= fineAO(h.voxel, N, P); }

  var diffuse = vec3f(0.0);
  var specular = vec3f(0.0);
  let V = -rd;
  let sunL = U.sunDir.xyz;
  let ndl = dot(N, sunL);
  if (ndl > 0.0 && U.sunDir.w > 0.0) {
    var sh = vec3f(1.0);
    if (quality > 0) {
      let sHit = march(P + N * 0.5, sunL, 1e9, true);
      sh = select(sHit.trans, vec3f(0.0), sHit.hit);
    } else {
      sh = vec3f(0.35);
    }
    let sc = U.sunColor.rgb * U.sunDir.w * sh;
    diffuse += sc * ndl;
    let H = normalize(sunL + V);
    specular += sc * pow(max(dot(N, H), 0.0), C.w) * C.z;
  }

  let count = i32(U.sunColor.w);
  for (var i = 0; i < count; i++) {
    let lp = lights[i * 2];
    let lc = lights[i * 2 + 1];
    let Lp = lp.xyz / vs;
    let radius = lp.w / vs;
    var L = Lp - P;
    let d = length(L);
    if (d > radius) { continue; }
    L /= d;
    let nl = dot(N, L);
    if (nl <= 0.0) { continue; }
    let x = d / radius;
    let dm = d * vs;
    let atten = (1.0 - x * x) * (1.0 - x * x) / (1.0 + dm * dm * 1.5);
    var sh = vec3f(1.0);
    if (quality > 0 && lc.w > 0.5 && U.flags.y > 0.5) {
      let sHit = march(P + N * 0.5, L, d - 2.0, true);
      sh = select(sHit.trans, vec3f(0.0), sHit.hit);
    }
    let c = lc.rgb * atten * sh;
    diffuse += c * nl;
    let H = normalize(L + V);
    specular += c * pow(max(dot(N, H), 0.0), C.w) * C.z;
  }

  let hemi = mix(vec3f(0.35, 0.25, 0.22), U.ambient.rgb, N.y * 0.5 + 0.5);
  let ambient = hemi * ao;
  return albedo * (diffuse * mix(1.0, ao, 0.5) + ambient) + specular;
}

// ---------------------------------------------------------------------------

struct VSOut {
  @builtin(position) pos: vec4f,
};

@vertex
fn vsMain(@builtin(vertex_index) vi: u32) -> VSOut {
  var out: VSOut;
  let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  out.pos = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
  return out;
}

struct FSOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
};

@fragment
fn fsMain(in: VSOut) -> FSOut {
  var out: FSOut;
  let uv = in.pos.xy * U.resolution.zw;
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let far = U.invViewProj * vec4f(ndc, 1.0, 1.0);
  let farW = far.xyz / far.w;
  let camW = U.camPos.xyz;
  let rd = normalize(farW - camW);
  let ro = camW / U.dims.w;

  let h = march(ro, rd, 1e9, false);
  let dbg = i32(U.flags.w);
  if (dbg > 0) {
    var c = vec3f(0.0);
    if (h.hit) {
      if (dbg == 1) { c = h.normal * 0.5 + 0.5; }
      if (dbg == 2) { c = materialColor(h.value, h.voxel); }
      if (dbg == 3) { c = vec3f(fract(h.t / 100.0)); }
      if (dbg == 4) { c = vec3f(f32(h.value & 0xffu) / 64.0, fract(f32(h.value & 0xffu) * 0.37), 0.5); }
      if (dbg == 5) { c = shade(h, ro, rd, 0); }
      if (dbg == 8 || dbg == 9) {
        if (dbg == 9) { let P = ro + rd * h.t; let s2 = shade(h, ro, rd, 1); c = s2 * 0.0; }
        let o = f32(gOuterSteps); let n = f32(gInnerSteps);
        c = vec3f(o / 300.0, n / 300.0, 0.0);
      }
      if (dbg == 6) { let P = ro + rd * h.t; c = vec3f(coarseAO(P, h.normal) * fineAO(h.voxel, h.normal, P)); }
      if (dbg == 7) {
        let P = ro + rd * h.t;
        let sHit = march(P + h.normal * 0.5, U.sunDir.xyz, 1e9, true);
        c = select(sHit.trans, vec3f(0.0), sHit.hit) * max(dot(h.normal, U.sunDir.xyz), 0.0);
      }
    } else { c = vec3f(0.2, 0.0, 0.3); }
    out.color = vec4f(c, 1.0);
    out.depth = 1.0;
    return out;
  }
  var color: vec3f;
  if (!h.hit) {
    color = h.trans * skyColor(rd) + h.glow;
    out.depth = 1.0;
  } else {
    var base = shade(h, ro, rd, 1);
    let id = h.value & 0xffu;
    let refl = matD(id).x;
    if (refl > 0.0 && U.flags.x > 0.5) {
      let P = ro + rd * h.t;
      let R = reflect(rd, h.normal);
      let cosT = max(dot(-rd, h.normal), 0.0);
      let F = refl + (1.0 - refl) * pow(1.0 - cosT, 5.0) * refl * 2.0;
      let rh = march(P + h.normal * 0.5, R, 1e9, false);
      var rc: vec3f;
      if (rh.hit) { rc = rh.trans * shade(rh, P + h.normal * 0.5, R, 0) + rh.glow; }
      else { rc = rh.trans * skyColor(R) + rh.glow; }
      base = mix(base, rc, clamp(F, 0.0, 0.9));
    }
    color = h.trans * base + h.glow;
    let wp = (ro + rd * h.t) * U.dims.w;
    let clip = U.viewProj * vec4f(wp, 1.0);
    out.depth = clamp(clip.z / clip.w, 0.0, 1.0);
  }
  out.color = vec4f(color, 1.0);
  return out;
}
