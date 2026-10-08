// Material table shared by the CPU (destruction rules) and the GPU (shading).
//
// A voxel is a u16: low byte = material id (0 = air), high byte = "shade".
// shade 0 lets the material's procedural pattern pick the colour; shade 1..255
// overrides it with mix(color, color2, (shade-1)/254) for painted detail.

export enum Pattern {
  Flat = 0,
  Marble = 1,
  WoodX = 2,
  WoodZ = 3,
  WoodY = 4,
  MarbleTiles = 5,
  Carpet = 6,
  StripesY = 7,
  Fabric = 8,
  Speckle = 9,
  Brushed = 10,
  Screen = 11,
  GradientY = 12,
  Leaf = 13,
}

export type Fragility = 'none' | 'shatter' | 'pane';

export interface MaterialDef {
  color: number; // sRGB hex
  color2?: number;
  pattern?: Pattern;
  /** Pattern feature size in metres. */
  scale?: number;
  /** Per-voxel brightness jitter, 0..1. */
  noise?: number;
  emission?: number;
  spec?: number;
  shininess?: number;
  reflect?: number;
  /** 0 = opaque, otherwise how much light passes through (tinted by color). */
  glass?: number;
  /** Structural voxels anchor everything touching them (walls, floor, ...). */
  structural?: boolean;
  fragile?: Fragility;
  /** Multiplies the carve radius. 0 = indestructible. */
  softness?: number;
  liquid?: boolean;
  /** Bill per destroyed voxel, in dollars. */
  price?: number;
  /** Fraction of destroyed voxels that turn into debris particles. */
  debris?: number;
}

export interface Material extends Required<Omit<MaterialDef, 'color2'>> {
  id: number;
  name: string;
  color2: number;
}

export const materials: Material[] = [];
const byName = new Map<string, number>();

function define(name: string, def: MaterialDef): number {
  const id = materials.length + 1;
  if (id > 255) throw new Error('too many materials');
  const m: Material = {
    id,
    name,
    color: def.color,
    color2: def.color2 ?? def.color,
    pattern: def.pattern ?? Pattern.Flat,
    scale: def.scale ?? 0.1,
    noise: def.noise ?? 0.08,
    emission: def.emission ?? 0,
    spec: def.spec ?? 0.1,
    shininess: def.shininess ?? 16,
    reflect: def.reflect ?? 0,
    glass: def.glass ?? 0,
    structural: def.structural ?? false,
    fragile: def.fragile ?? 'none',
    softness: def.softness ?? 1,
    liquid: def.liquid ?? false,
    price: def.price ?? 1,
    debris: def.debris ?? 0.6,
  };
  materials.push(m);
  byName.set(name, id);
  return id;
}

export function materialId(name: string): number {
  const id = byName.get(name);
  if (id === undefined) throw new Error(`unknown material ${name}`);
  return id;
}

export function mat(id: number): Material {
  return materials[id - 1];
}

/** Pack a voxel value. */
export function vox(materialId: number, shade = 0): number {
  return (materialId & 0xff) | ((shade & 0xff) << 8);
}

// ---------------------------------------------------------------------------
// The penthouse palette.

export const M = {
  concrete: define('concrete', { color: 0x8a8580, noise: 0.12, structural: true, softness: 0.6, price: 0.2 }),
  wall: define('wall', { color: 0xd9cbb4, color2: 0xc8b89d, pattern: Pattern.StripesY, scale: 0.4, noise: 0.04, structural: true, softness: 0.7, price: 0.5 }),
  wallDark: define('wallDark', { color: 0x2b2230, color2: 0x3a2e40, pattern: Pattern.StripesY, scale: 0.25, noise: 0.05, structural: true, softness: 0.7, price: 0.5 }),
  ceiling: define('ceiling', { color: 0xe8e2d8, noise: 0.03, structural: true, softness: 0.7, price: 0.3 }),
  floor: define('floor', { color: 0xefe9e1, color2: 0x8e8679, pattern: Pattern.MarbleTiles, scale: 1.0, noise: 0.03, spec: 0.5, shininess: 60, reflect: 0.22, structural: true, softness: 0.5, price: 3 }),
  marble: define('marble', { color: 0xf2eee8, color2: 0x7d7468, pattern: Pattern.Marble, scale: 0.35, noise: 0.03, spec: 0.5, shininess: 60, reflect: 0.12, softness: 0.6, price: 4 }),
  marbleBlack: define('marbleBlack', { color: 0x1d1b1c, color2: 0xc9b37e, pattern: Pattern.Marble, scale: 0.3, noise: 0.03, spec: 0.6, shininess: 80, reflect: 0.18, softness: 0.6, price: 6 }),
  frame: define('frame', { color: 0x1e1e22, noise: 0.03, spec: 0.5, shininess: 40, structural: true, softness: 0.25, price: 2 }),
  windowGlass: define('windowGlass', { color: 0xc6dde8, glass: 0.82, spec: 1, shininess: 200, fragile: 'pane', price: 1.5, debris: 0.5 }),

  gold: define('gold', { color: 0xe7b549, color2: 0xa47722, pattern: Pattern.Brushed, scale: 0.05, noise: 0.05, spec: 0.9, shininess: 90, reflect: 0.25, softness: 0.4, price: 25 }),
  chrome: define('chrome', { color: 0xc8ccd2, color2: 0x8e949c, pattern: Pattern.Brushed, scale: 0.05, noise: 0.03, spec: 1, shininess: 120, reflect: 0.45, softness: 0.35, price: 5 }),
  steelDark: define('steelDark', { color: 0x3b3d42, noise: 0.05, spec: 0.6, shininess: 50, softness: 0.4, price: 3 }),
  lacquer: define('lacquer', { color: 0x08080a, noise: 0.01, spec: 1, shininess: 200, reflect: 0.35, softness: 0.8, price: 60 }),
  ivory: define('ivory', { color: 0xf3eedf, noise: 0.02, spec: 0.4, shininess: 40, price: 20 }),

  walnut: define('walnut', { color: 0x5a3720, color2: 0x3b2213, pattern: Pattern.WoodX, scale: 0.03, noise: 0.06, spec: 0.3, shininess: 30, price: 4 }),
  walnutZ: define('walnutZ', { color: 0x5a3720, color2: 0x3b2213, pattern: Pattern.WoodZ, scale: 0.03, noise: 0.06, spec: 0.3, shininess: 30, price: 4 }),
  walnutY: define('walnutY', { color: 0x5a3720, color2: 0x3b2213, pattern: Pattern.WoodY, scale: 0.03, noise: 0.06, spec: 0.3, shininess: 30, price: 4 }),
  oak: define('oak', { color: 0xb5834f, color2: 0x8a5d33, pattern: Pattern.WoodX, scale: 0.025, noise: 0.06, spec: 0.25, shininess: 25, price: 3 }),

  velvet: define('velvet', { color: 0x4a1a5c, color2: 0x2e0f3a, pattern: Pattern.Fabric, scale: 0.02, noise: 0.08, spec: 0.05, price: 6, softness: 1.2 }),
  velvetRed: define('velvetRed', { color: 0x8c1424, color2: 0x5c0b16, pattern: Pattern.Fabric, scale: 0.02, noise: 0.08, spec: 0.05, price: 6, softness: 1.2 }),
  leather: define('leather', { color: 0x1a1414, color2: 0x2a201c, pattern: Pattern.Speckle, scale: 0.01, noise: 0.06, spec: 0.35, shininess: 25, price: 8 }),
  foam: define('foam', { color: 0xe6d9a6, noise: 0.15, spec: 0, price: 0.5, softness: 1.4 }),
  felt: define('felt', { color: 0x0f5a32, color2: 0x0b4627, pattern: Pattern.Fabric, scale: 0.01, noise: 0.07, spec: 0, price: 3 }),
  carpet: define('carpet', { color: 0x5b0f2c, color2: 0xd4a33b, pattern: Pattern.Carpet, scale: 0.6, noise: 0.08, spec: 0, price: 2, softness: 1.2 }),

  crystal: define('crystal', { color: 0xf4f8ff, glass: 0.9, spec: 1, shininess: 300, fragile: 'shatter', price: 40, debris: 1 }),
  glassClear: define('glassClear', { color: 0xe8f2f4, glass: 0.9, spec: 1, shininess: 250, fragile: 'shatter', price: 12, debris: 1 }),
  glassTable: define('glassTable', { color: 0xa6d4cc, glass: 0.75, spec: 1, shininess: 250, fragile: 'pane', price: 10, debris: 0.8 }),
  glassGreen: define('glassGreen', { color: 0x2f7a3a, glass: 0.7, spec: 1, shininess: 200, fragile: 'shatter', price: 15, debris: 1 }),
  glassAmber: define('glassAmber', { color: 0xa8641c, glass: 0.7, spec: 1, shininess: 200, fragile: 'shatter', price: 15, debris: 1 }),
  glassBlue: define('glassBlue', { color: 0x2d5fb0, glass: 0.7, spec: 1, shininess: 200, fragile: 'shatter', price: 15, debris: 1 }),
  glassRed: define('glassRed', { color: 0xa0202a, glass: 0.7, spec: 1, shininess: 200, fragile: 'shatter', price: 15, debris: 1 }),
  glassSmoke: define('glassSmoke', { color: 0x5a5560, glass: 0.65, spec: 1, shininess: 200, fragile: 'shatter', price: 15, debris: 1 }),
  champagne: define('champagne', { color: 0xf0cf6a, glass: 0.7, spec: 0.6, shininess: 120, fragile: 'shatter', liquid: true, price: 30, debris: 0.8 }),
  wine: define('wine', { color: 0x5c0a1a, glass: 0.55, spec: 0.6, shininess: 120, fragile: 'shatter', liquid: true, price: 20, debris: 0.8 }),
  whiskey: define('whiskey', { color: 0xb5651d, glass: 0.65, spec: 0.6, shininess: 120, fragile: 'shatter', liquid: true, price: 25, debris: 0.8 }),
  absinthe: define('absinthe', { color: 0x6fd35a, glass: 0.65, spec: 0.6, shininess: 120, fragile: 'shatter', liquid: true, price: 25, debris: 0.8 }),
  ice: define('ice', { color: 0xdff4ff, glass: 0.75, spec: 1, shininess: 200, fragile: 'shatter', price: 1, debris: 1 }),
  water: define('water', { color: 0x4fb3c9, glass: 0.8, spec: 1, shininess: 200, fragile: 'shatter', liquid: true, price: 1, debris: 0.7 }),

  porcelain: define('porcelain', { color: 0xf7f6f2, color2: 0x1f3f8f, noise: 0.02, spec: 0.7, shininess: 90, reflect: 0.08, fragile: 'shatter', price: 80, debris: 1 }),
  porcelainRed: define('porcelainRed', { color: 0xa3161c, color2: 0xe2b34c, noise: 0.02, spec: 0.7, shininess: 90, reflect: 0.08, fragile: 'shatter', price: 80, debris: 1 }),
  terracotta: define('terracotta', { color: 0xb05a33, noise: 0.1, spec: 0.1, fragile: 'shatter', price: 3, debris: 1 }),
  soil: define('soil', { color: 0x3a2617, noise: 0.25, spec: 0, price: 0.1, softness: 1.5 }),
  leaf: define('leaf', { color: 0x2f7d32, color2: 0x174d1d, pattern: Pattern.Leaf, scale: 0.05, noise: 0.15, spec: 0.2, shininess: 20, price: 1, softness: 1.5 }),
  trunk: define('trunk', { color: 0x6b4a2b, color2: 0x4a321c, pattern: Pattern.WoodY, scale: 0.015, noise: 0.12, price: 1 }),

  dieRed: define('dieRed', { color: 0xc4121f, glass: 0.35, spec: 0.9, shininess: 120, price: 5 }),
  pip: define('pip', { color: 0xf8f8f0, noise: 0.02, spec: 0.5, price: 1 }),
  chip: define('chip', { color: 0xffffff, color2: 0x111111, noise: 0.04, spec: 0.3, price: 50 }),
  card: define('card', { color: 0xf8f6ef, color2: 0xb3121d, noise: 0.01, spec: 0.2, price: 1 }),

  neonPink: define('neonPink', { color: 0xff2fa8, emission: 6, noise: 0, spec: 0, fragile: 'shatter', price: 30, debris: 1 }),
  neonCyan: define('neonCyan', { color: 0x2ff3ff, emission: 6, noise: 0, spec: 0, fragile: 'shatter', price: 30, debris: 1 }),
  neonGold: define('neonGold', { color: 0xffc23d, emission: 6, noise: 0, spec: 0, fragile: 'shatter', price: 30, debris: 1 }),
  bulb: define('bulb', { color: 0xffd9a0, emission: 8, noise: 0, spec: 0, fragile: 'shatter', price: 5, debris: 1 }),
  lampshade: define('lampshade', { color: 0xf5deb0, emission: 1.3, noise: 0.05, spec: 0, price: 4 }),
  screen: define('screen', { color: 0xff3366, color2: 0x33ccff, pattern: Pattern.Screen, scale: 0.08, emission: 2.5, noise: 0, spec: 0.8, shininess: 100, fragile: 'pane', price: 200, debris: 0.8 }),
  tv: define('tv', { color: 0x1a2a6c, color2: 0xb21f8d, pattern: Pattern.GradientY, scale: 1.0, emission: 1.6, noise: 0.02, spec: 0.8, shininess: 100, fragile: 'pane', price: 300, debris: 0.8 }),
  plasticBlack: define('plasticBlack', { color: 0x141416, noise: 0.03, spec: 0.4, shininess: 40, price: 5 }),
  slotRed: define('slotRed', { color: 0xa50f1f, color2: 0xe8b23a, noise: 0.04, spec: 0.6, shininess: 60, reflect: 0.1, price: 40 }),
  paperArt: define('paperArt', { color: 0x1b3b6f, color2: 0xf2a541, noise: 0.05, spec: 0.1, price: 500 }),
  mirror: define('mirror', { color: 0x15171a, noise: 0.0, spec: 1, shininess: 300, reflect: 0.85, fragile: 'pane', price: 20, debris: 0.8 }),
  label: define('label', { color: 0xf1e7cf, color2: 0x1a1a1a, noise: 0.03, spec: 0.1, price: 1 }),
  cork: define('cork', { color: 0x9c7348, noise: 0.15, price: 0.5 }),
  velvetGold: define('velvetGold', { color: 0xc99a2e, color2: 0x8a6416, pattern: Pattern.Fabric, scale: 0.02, noise: 0.08, spec: 0.15, price: 8, softness: 1.2 }),
  chipRed: define('chipRed', { color: 0xb3141f, color2: 0xf4f0e6, noise: 0.03, spec: 0.3, price: 100 }),
  chipBlue: define('chipBlue', { color: 0x1d3fa8, color2: 0xf4f0e6, noise: 0.03, spec: 0.3, price: 100 }),
  chipGreen: define('chipGreen', { color: 0x147a3a, color2: 0xf4f0e6, noise: 0.03, spec: 0.3, price: 100 }),
  chipBlack: define('chipBlack', { color: 0x151515, color2: 0xe8c45a, noise: 0.03, spec: 0.3, price: 500 }),
  chipPurple: define('chipPurple', { color: 0x5b1f8a, color2: 0xf4f0e6, noise: 0.03, spec: 0.3, price: 1000 }),
} as const;

/** Pack the table for the GPU: 16 floats per material, index = material id. */
export function packMaterials(): Float32Array<ArrayBuffer> {
  const out = new Float32Array(256 * 16);
  for (const m of materials) {
    const o = m.id * 16;
    const a = srgbHexToLinear(m.color);
    const b = srgbHexToLinear(m.color2);
    out.set([a[0], a[1], a[2], m.emission, b[0], b[1], b[2], m.pattern], o);
    out.set([m.scale, m.noise, m.spec, m.shininess, m.reflect, m.glass, m.liquid ? 1 : 0, 0], o + 8);
  }
  return out;
}

export function srgbHexToLinear(hex: number): [number, number, number] {
  const c = [(hex >> 16) & 255, (hex >> 8) & 255, hex & 255].map((v) => {
    const s = v / 255;
    return s <= 0.04045 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
  });
  return [c[0], c[1], c[2]];
}
