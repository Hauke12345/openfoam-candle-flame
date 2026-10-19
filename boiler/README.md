# Kettle wall: how a boiling bubble is born (GROMACS + Blender)

A molecular-dynamics picture of what happens at the bottom of a kettle before it boils. The box is
5.1 × 5.1 nm in plan: a frozen wall at the bottom, a 5 nm water layer with dissolved gas, and free space above.

**What it shows**
1. **Degassing (heating 20 → 95 °C).** Warm water holds less dissolved air. The gas (N₂, blue) leaves
   the solution. Part of it escapes through the free surface. The rest gathers on a less-wetting spot of the
   wall (amber; a stand-in for a scratch or a scale deposit), where water holds on less strongly.
2. **A gas nanobubble on the spot.** From about 2 ns (95 °C), 28–41 N₂ form a stable pocket of
   ~3.5–5 nm³ on the patch. The glass-like part of the wall stays wetted.
3. **What comes next (not reachable in this box).** Above the boiling point, water evaporates into such
   pre-existing gas pockets. That is why kettles boil from fixed spots on the wall: nucleation needs a seed.

Movie: `media/boiler_bubble.mp4` (phases 1–2, front half of the water cut away). Plot: `degas_analysis.png`.

**Superheating tried, and why the bubble does not grow here.** Two follow-up runs started from the nanobubble at 95 °C:
- 368 → 460 K over 2 ns. The pocket breaks up (largest N₂ cluster 28 → ~4), and the water-free volume
  stays ~3 nm³. The water evaporates at the free surface instead.
- A fast ramp of 368 → 600 K over 1.5 ns (`superheat_analysis.png`). The space over the patch only fluctuates
  (3–7 nm³). Near 600 K the whole slab expands to ~0.5 g/cm³ and fills the box: SPC/E approaches its
  critical point before the patch nucleates a bubble.

The reason is the Laplace pressure 2γ/r ≈ 100 MPa of a 1–2 nm pocket. A vapour bubble grows only once it is
above the critical radius, which at a few kelvin of superheat is micrometres: the size of the gas pockets
trapped in real wall scratches. MD shows how the seed forms; its growth needs a continuum (e.g. OpenFOAM VOF) model.

Dissolved air is neutral N₂/O₂, not ions. Only CO₂ partly reacts to bicarbonate, which is the limescale
chemistry; classical MD cannot do that reaction.

## Model
| | |
|---|---|
| water | SPC/E, SETTLE |
| gas | N₂ as one LJ site (σ 0.3698 nm, ε 0.79 kJ/mol), mole fraction 2.5 % (111 molecules) |
| wall | 3 frozen layers of uncharged LJ atoms. Glass-like WP: OW–WP ε 1.30. Patch WH (r = 1.2 nm): OW–WH ε 0.30, N₂–WH ε 1.00 |
| run | PME with `ewald-geometry 3dc` (slab), v-rescale with annealing, dt 2 fs, 13,401 atoms, ~35–40 ns/day on 4 cores |

## Caveats (it is a cartoon with real physics, not a quantitative kettle)
- The wall is a generic LJ solid, not atomistic silica or steel.
- The gas content is ~100× that of tap water. Otherwise a 5 nm box would hold less than one molecule.
- Time is compressed: degassing takes nanoseconds here, seconds to minutes in a kettle.
- SPC/E water boils above 100 °C (its own boiling point is higher than real water's), and only a
  nanometre-sized region is simulated. Vapour formation therefore needs an exaggerated superheat (up to 460 K).

## Files
- `build.py`, `setup.py`: wall, topology, starting configuration (`gmx solvate`), and mdp files (em, eq, heat).
- `analyze.py`: largest N₂ cluster, its position, the escaped gas, and the water-free volume per frame.
- `render_md.py`: Blender (bpy 5.x, Cycles) cut-away render, with Geometry Nodes sphere instancing.
- `make_movie.py`: captions and timeline, assembled to MP4 with ffmpeg.

Reproduce (gmx 2026 on PATH):
```
python3 setup.py run1
cd run1 && gmx grompp -f em.mdp -c conf.gro -p topol.top -n index.ndx -o em && gmx mdrun -deffnm em
gmx grompp -f eq.mdp -c em.gro -p topol.top -n index.ndx -o eq && gmx mdrun -deffnm eq
gmx grompp -f heat.mdp -c eq.gro -p topol.top -n index.ndx -o heat && gmx mdrun -deffnm heat
cd .. && python3 analyze.py run1 heat 1
python3 render_md.py run1 frames_heat heat 2 960 540 16
python3 make_movie.py degas.mp4 24 frames_heat:293:368:degas
```
