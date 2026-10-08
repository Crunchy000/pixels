// Shared particle layout (32 bytes).

struct Particle {
  pos: vec3f,   // voxel units
  value: u32,   // low 16 bits voxel value, high bits flags
  vel: vec3f,   // voxels / s
  life: f32,    // seconds left
};

const SETTLED: u32 = 0x10000u;
const DEPOSIT: u32 = 0x20000u;
const DUST: u32 = 0x40000u;

