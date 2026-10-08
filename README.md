# Penthouse Smash

**Play: https://crunchy000.github.io/pixels/** (needs a WebGPU browser)

A WebGPU micro-voxel playground: a Las Vegas penthouse suite built from
**1.25 cm voxels** (≈ 400 million cells, 16 m × 4.4 m × 12 m), ray-marched on
the GPU at roughly **one voxel per pixel**, where everything can be destroyed.

Shoot the chandelier's rod and it drops onto the glass coffee table. Shatter a
glass bar shelf and the bottles on it fall and smash on the counter. Blow a
hole in the sofa and the foam shows through. Debris settles back into the world
as real voxels, lamps go dark when their bulbs break, and the damage bill keeps
running.

Everything after the initial upload (carving, glass shattering, finding
unsupported pieces, falling chunks, debris, lights, the player) runs in GPU
compute shaders, with no per-frame CPU↔GPU copies of voxel data.

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

The simulation lives entirely on the GPU. The CPU builds the scene once,
uploads it once, frees its copy, and from then on only sends input (a 208-byte
uniform block per frame) and reads back a few hundred bytes of HUD counters.
No voxel data crosses the bus while you play.

```
CPU (once)                 GPU, every frame (src/render/shaders/sim.wgsl, player.wgsl)
──────────                 ────────────────────────────────────────────────────────────
SDF scene ─rasterise─▶ upload ─▶ grid / pool / super-grid  ◀─┐
                                   │                          │ edits
input (208 B/frame) ─▶ player ─▶ camera ─▶ shoot ─▶ impacts ─▶ carve ─▶ glass flood ─┤
                                   │                                    island search ─┤
                                   │                       falling chunks ◀────────────┘
                                   │                       debris particles ─▶ deposits ─┘
                                   ├─▶ ray-march (fragment) ─┐
                                   └─▶ debris + chunk cubes ─┴─▶ bloom ─▶ tonemap
HUD ◀─ ~560 B of counters (bill, voxels smashed, bricks, chunks)
```

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

### Destruction (all compute shaders)

`src/render/shaders/sim.wgsl`, orchestrated by `src/gpu/Simulation.ts` as one
compute pass of about 100 dispatches per frame. Most are indirect, sized by
counters that earlier dispatches wrote, so idle stages cost almost nothing and
nothing waits on the CPU. Work queues, the free list, hash tables and chunk
tables all live in one GPU "heap" buffer (`src/gpu/layout.ts`).

1. **Shoot**: the CPU only says "fire tool X this frame". A compute thread per
   pellet ray-casts from the GPU-side camera and queues an **impact**.
2. **Carve**: one workgroup per (impact, brick). Compressed uniform bricks get a
   real pool brick first (request → allocate → fill). Voxels are removed with
   atomic compare-exchange, so overlapping blasts are safe. Material softness
   scales the jittered radius (gold resists, foam gives), with a minimum so rods
   and legs can always be cut. Removed voxels pay the bill and spawn debris.
3. **Glass panes** (windows, shelves, the table top, mirrors) shatter by a
   breadth-first flood through the pane's material, a dozen rings per frame,
   so the crack visibly spreads from the impact.
4. **Island search**: voxels next to fresh damage seed a multi-source BFS that
   runs over a few frames. Every visited voxel goes into a GPU hash table whose
   slots double as nodes of a lock-free union-find. Components that touch
   anything structural (floor, walls, ceiling, window frames) are anchored and
   stop expanding; the rest are lifted out of the world as **falling chunks**
   (crumbs under 48 voxels just become debris).
5. **Falling chunks** are drawn as voxel cubes while they fall, so they can't
   collide with themselves. On landing they smash fragile things underneath
   (chandelier into glass table), shatter if they are fragile themselves, and
   otherwise write their voxels back into the world.
6. **Debris** particles: gravity, drag, voxel-DDA collision and bounce. When
   they come to rest many are written straight back into the world as voxels,
   so rubble piles up and can be shot again.
7. **Player**: walking with voxel collision (or flying) runs in a one-thread
   compute pass that also builds the camera matrices.

Lights are tied to their emissive voxels: break the bulbs and the light fades
out (flickering while damaged). If a fixture falls, its light falls with it.

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

\*GPU memory before destruction (the CPU copy is freed after the upload).
Measured headless, so build times are only indicative. The GPU pool reserves
headroom for destruction: explosions expand uniform bricks into pool bricks,
and settling rubble adds more. At the default 80 voxels/m the GPU holds about
195 MB in total: a 141 MB brick pool (2-3x the scene), a 45 MB simulation heap
(queues, hash table, chunk arena) and 8 MB of debris particles.

## Status and next steps

This is a working prototype for exploring the idea. GPU timings so far come
only from a software renderer, so real-hardware performance still needs
profiling. Dynamic resolution is there to protect the frame rate.

Ideas for where to take it:

- Rigid-body chunks that tumble (render chunks as separate voxel volumes with transforms)
- Temporal accumulation / TAA for soft shadows and multi-bounce light
- Build the scene on the GPU too (SDF primitives as data), skipping the one-time upload
- Brick streaming / LOD for larger worlds and sub-centimetre voxels
- Sound, fire and liquids
