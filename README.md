# Penthouse Smash

A WebGPU micro-voxel playground: a Las Vegas penthouse suite built from
**1.25 cm voxels** (≈ 400 million cells, 16 m × 4.4 m × 12 m), ray-marched on
the GPU at roughly **one voxel per pixel**, where everything can be destroyed.

Shoot the chandelier's rod and it drops onto the glass coffee table. Shatter a
glass bar shelf and the bottles on it fall and smash on the counter. Blow a
hole in the sofa and the foam shows through. Debris settles back into the world
as real voxels, lamps go dark when their bulbs break, and the damage bill keeps
running.

```
npm install
npm run dev          # http://localhost:5173
npm run build        # static build in dist/
npm run build:single # one self-contained HTML file in dist-single/
```

Needs a browser with WebGPU: Chrome / Edge 113+, Safari 26+, or Firefox 141+
(Windows) / Nightly.

## Controls

| Input | Action |
| --- | --- |
| Click | Capture the mouse |
| WASD, Shift | Move, run |
| Space | Jump (fly mode: up; C: down) |
| Left click | Fire |
| 1–4 / wheel | Pistol, Shotgun, Blaster, Dynamite |
| F | Fly / walk |
| N | Dusk / night |
| `[` `]` | Render scale down / up (turns off auto resolution) |
| P | Pixelated (nearest) upscale |
| R / L | Glossy reflections / lamp shadows on or off |
| H | Help |

Touch: left thumb moves, right thumb looks, on-screen FIRE / TOOL / JUMP.

### URL parameters

| Param | Default | Meaning |
| --- | --- | --- |
| `vpm` | 80 | Voxels per metre (80 = 1.25 cm). Higher is finer and uses more memory, see below. |
| `scale` | auto | Fixed render scale relative to device pixels (disables dynamic resolution). |
| `target` | 20 | Frame-time target in ms for dynamic resolution (0 = off). |
| `night` | off | Start at night. |
| `particles` | 262144 | Debris particle capacity. |
| `debug` | 0 | 1 normals, 2 albedo, 3 distance, 4 material id, 5 no shadows, 6 AO, 7 sun shadow, 8 step heatmap. |
| `cam` | — | `x,y,z,yaw,pitch` start camera (flying). |
| `frames` | — | Render N frames then stop (headless screenshots). |

## How it works

```
CPU (authoritative)                         GPU
───────────────────                         ───
SDF scene ──rasterise──▶ VoxelWorld ──dirty bricks──▶ grid / pool / super-grid buffers
                          │  ▲                            │
         shoot ─▶ carve ──┘  │ deposits (readback)        ├─▶ ray-march pass (fragment) ─┐
                 islands ─▶ falling chunks                 │                              ├─▶ bloom ─▶ tonemap
                 debris ─────────────────▶ particle buffer ├─▶ particle sim (compute)     │
                                                           └─▶ particle cubes (indirect) ─┘
```

### Storage: a sparse brick map

`src/world/VoxelWorld.ts`. The world is a grid of 8³-voxel **bricks**. Each
grid cell is one `u32`:

- `0`: empty brick (all air)
- `0x80000000 | value`: **uniform** brick, all 512 voxels identical (floor slab, wall core, sofa stuffing)
- `n`: pool brick `n-1`, holding 512 explicit `u16` voxels

A voxel is a `u16`: low byte is a **material id**, high byte is a "shade" for painted
detail (chip stripes, card backs, piano keys). Colour is *not* stored per voxel.
The shader evaluates each material's pattern (marble veins, wood rings, casino
carpet, wallpaper stripes, animated slot reels) from the voxel's world position,
so a 3D solid texture stays consistent when you blast into it, and big
homogeneous regions compress to uniform bricks.

Above the bricks is a **super grid** (4³ bricks = 32³ voxels) holding a count
of non-empty bricks per region, so rays skip open air 32 voxels at a time.

The CPU keeps the authoritative copy and uploads only dirty bricks/cells each
frame (`Renderer.syncWorld`), coalescing contiguous ranges.

### Rendering

`src/render/shaders/raymarch.wgsl`. One full-screen fragment pass casts one ray
per pixel through three nested DDAs (super cell → brick → voxel), with a fast
path for uniform bricks. Per hit:

- sun light with a **shadow ray** (it passes through window glass, tinted)
- up to 32 point lights, the important ones with their own shadow rays
- **glass**: rays carry on through transparent voxels, accumulating tint and fresnel
- **glossy reflections** for marble, lacquer, chrome, mirrors (one bounce)
- two AO terms: per-voxel corner AO (Minecraft-style, from the 8 neighbours) and
  coarse AO from a 3D brick-density texture
- writes real depth, so the debris (instanced cubes) depth-tests against it
- a procedural Las Vegas skyline at dusk or night outside the windows

Then a bloom mip chain and ACES tonemap. Dynamic resolution keeps the frame
time near the target, and the composite pass upsamples (linear, or nearest
with `P` for crisp voxel pixels).

### Building the scene

`src/world/shapes.ts` and `src/scene/penthouse.ts`. Everything is modelled from
signed-distance primitives (box, rounded box, cylinder, capsule, torus, lathe,
extruded polygon, with CSG and rotations), each painted with a material or a
per-voxel paint function. The rasteriser classifies whole bricks, then 4³
sub-blocks, against the SDF. Fully inside means a uniform brick with no
per-voxel work; only boundary blocks are evaluated voxel by voxel. The whole
suite builds in about 2 s.

The scene: marble-tiled floor with a coffered ceiling, floor-to-ceiling
glazing on two walls, an L-shaped velvet sectional with foam cores, a glass
coffee table (ice bucket, champagne, coupes, giant dice, chips), a crystal
chandelier, a back bar with roughly 150 bottles on glass shelves against a mirror, bar stools,
a poker table with chip stacks and cards, four slot machines, a grand piano
with its lid propped, a five-tier champagne tower, neon VEGAS / BAR / martini
signs, a TV wall, Ming vases on pedestals, a gold trophy, abstract art and palms.

### Destruction

`src/sim/Destruction.ts`:

1. **Ray cast** on the CPU copy finds the hit voxel.
2. **Carve** a jittered sphere. Material softness scales the radius (gold and
   window frames resist, foam gives), with a minimum so rods and legs can
   always be shot through. Carving walks bricks and skips empty ones.
3. **Fragile materials**: `shatter` (bottles, crystal, porcelain) break within
   a larger radius; `pane` (windows, glass shelves, the table top, mirrors)
   flood-fill and shatter the whole connected pane.
4. **Island detection**: solid neighbours of the hole seed a down-first DFS.
   A component that reaches something structural (floor, walls, ceiling,
   window frames) is supported. One that doesn't becomes a **falling chunk**.
   Tiny fragments just crumble into debris.
5. **Falling chunks** translate down through the grid (no rotation) with
   gravity. On landing they break fragile things underneath (chandelier into
   glass table), shatter if they're fragile themselves, and wake up anything
   floating they landed on.
6. **Debris** particles are simulated on the GPU (gravity, drag, voxel-DDA
   collision, bounce). When they come to rest, many are appended to a deposit
   queue that the CPU reads back and writes into the world as real voxels, so
   rubble piles up and can be shot again.

Lights (`src/sim/LightSystem.ts`) are tied to their emissive voxels: break the
bulbs and the light fades out (flickering while damaged). If a light's fixture
falls, the light falls with it.

## Voxel size vs. memory

"One voxel per pixel" depends on distance: at 1080p with a 72° field of view
a pixel spans about 1.3 mm per metre of distance. A 1.25 cm voxel matches one
pixel at about 9 m, so across a 16 m room most surfaces are 1–3 pixels per
voxel. Finer voxels push that point closer, but memory grows with surface
area, roughly with the square of `vpm`:

| `vpm` | Voxel | Grid | Live bricks | Brick memory* | Build |
| --- | --- | --- | --- | --- | --- |
| 64 | 1.56 cm | 1024×288×768 | 42 k | 41 MB | 1.6 s |
| **80** | **1.25 cm** | 1280×352×960 | 46 k | 45 MB | 2.1 s |
| 100 | 1.0 cm | 1600×448×1200 | 75 k | 73 MB | 3.3 s |
| 128 | 7.8 mm | 2048×568×1536 | 175 k | 170 MB | 5.6 s |

\*Per copy (CPU and GPU each hold one), before destruction. Measured headless,
so build times are only indicative. Explosions expand uniform bricks into pool
bricks, so memory grows as you smash.

## Status and next steps

This is a working prototype for exploring the idea. GPU timings so far come
only from a software renderer, so real-hardware performance still needs
profiling. Dynamic resolution is there to protect the frame rate.

Ideas for where to take it:

- Rigid-body chunks that tumble (render chunks as separate voxel volumes with transforms)
- Temporal accumulation / TAA for soft shadows and multi-bounce light
- Move carving and island detection to a Web Worker (dynamite costs ~170 ms on the test machine)
- Brick streaming / LOD for larger worlds and sub-centimetre voxels
- Sound, fire and liquids
