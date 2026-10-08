// Debug only: copy a box of voxel values into a buffer the test harness reads back.
// (Prepended with shared.wgsl.)

struct ProbeParams { origin: vec4i, size: vec4i };

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> grid: array<u32>;
@group(0) @binding(2) var<storage, read> pool: array<u32>;
@group(0) @binding(4) var<uniform> P: ProbeParams;
@group(0) @binding(5) var<storage, read_write> outVals: array<u32>;

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

@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) gid: vec3u) {
  let s = P.size.xyz;
  let n = u32(s.x * s.y * s.z);
  if (gid.x >= n) { return; }
  let i = i32(gid.x);
  let p = P.origin.xyz + vec3i(i % s.x, (i / s.x) % s.y, i / (s.x * s.y));
  outVals[gid.x] = voxelAtP(p);
}
