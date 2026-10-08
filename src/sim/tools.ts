// Weapons. The CPU only decides *when* a tool fires; the ray cast, carving,
// shattering and everything after it run on the GPU (src/render/shaders/sim.wgsl).

export interface Tool {
  name: string;
  key: string;
  /** Carve radius in metres. */
  radius: number;
  /** Radius within which fragile materials (glass, porcelain) break. */
  fragileRadius: number;
  /** Debris launch speed, m/s. */
  impulse: number;
  /** Shots per second. */
  rate: number;
  pellets: number;
  spread: number;
  auto: boolean;
}

export const TOOLS: Tool[] = [
  { name: 'Pistol', key: '1', radius: 0.035, fragileRadius: 0.09, impulse: 2.5, rate: 7, pellets: 1, spread: 0, auto: true },
  { name: 'Shotgun', key: '2', radius: 0.045, fragileRadius: 0.1, impulse: 3.5, rate: 1.4, pellets: 10, spread: 0.055, auto: false },
  { name: 'Blaster', key: '3', radius: 0.3, fragileRadius: 0.45, impulse: 5, rate: 1.6, pellets: 1, spread: 0, auto: true },
  { name: 'Dynamite', key: '4', radius: 0.7, fragileRadius: 0.85, impulse: 8, rate: 0.6, pellets: 1, spread: 0, auto: false },
];
