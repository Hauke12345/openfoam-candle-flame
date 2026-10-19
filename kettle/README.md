# Glass kettle: from 20 °C tap water to a full boil (CFD + Blender)

A 1.7 L, 2 kW glass electric kettle simulated from switch-on to switch-off, then rendered in Blender.
Movie: `media/kettle_boil.mp4`. History plot: `kettle_cfd_summary.png`.

## What the simulation shows
| time | water | what happens |
|---|---|---|
| 0–20 s | 20 °C | The heating plate heats from 20 to 100 °C in ~15 s. Warm water rises from the plate in plumes. |
| from ~10 s | 20–40 °C | The plate is already above 100 °C: steam bubbles form on it and collapse within millimetres in the cold water. This is the early "ticking". |
| 60–250 s | 35–90 °C | **Degassing.** Warm water holds less air, so air bubbles grow on the plate and the glass (up to ~420 on the glass). Collapsing steam bubbles leave clouds of tiny air bubbles (0.1–0.2 mm) behind. |
| 150–285 s | 60–98 °C | **Singing (subcooled boiling).** The share of plate heat carried by boiling grows from 40 to 90 %. Bubbles get bigger and rise higher before they collapse; the bubble-driven flow speeds up from ~0.1 to ~0.5 m/s. |
| 285–303 s | 100 °C | **Full boil.** Steam bubbles survive, coalesce to 4–10 mm and burst at the surface. Up to 0.5 g/s of steam. |
| 303 s | | The steam thermostat switches off. Boiling stops within ~5 s. |

- **Heating time:** 20 → 100 °C takes ~290 s. A real 2 kW kettle with 1.7 L takes 4.5–5 min.
- **Energy balance:** closes to 1.4 %.
- **Degassing:** about 45 % of the dissolved air leaves before switch-off. Most of it leaves in the last minute, carried out by steam bubbles.

## Model (`kettle_cfd.py`)
- **Water.** Axisymmetric (r, z) incompressible flow with buoyancy from the real density ρ(T).
  - Grid: MAC, 40 × 63 cells (1.75 mm).
  - Time stepping: SSP-RK3, with a projection step for pressure.
  - Advection: 2nd-order upwind for momentum, van Leer TVD for the scalars.
  - Turbulence: Smagorinsky eddy viscosity.
  - Scalars: temperature and dissolved air (Henry's law, air-saturated at 20 °C).
  - Boundaries: no-slip at the plate and the glass, a flat free-slip surface on top that loses heat by convection and evaporation and exchanges air with the room, and a small heat loss through the glass.
- **Heating plate.**
  - Construction: an aluminium/steel sandwich with heat capacity and radial conduction, heated by a ring element.
  - Heat to the water: turbulent natural convection (Nu = 0.14 Ra^1/3) plus nucleate boiling (Cooper correlation). The boiling part is weighted by the fraction of active nucleation sites, whose activation superheats are log-normally distributed.
  - Split of the boiling heat: part goes into steam, a fraction that depends on the subcooling; the rest heats the water at the plate.
- **Bubbles.** Lagrangian 3D points, each holding air and steam, two-way coupled to the water.
  - Where they form: 320 sites on the plate (boiling and air) and 420 on the glass (air only). They grow while attached and depart at a departure diameter (Fritz at saturation, smaller when subcooled), limited by the Cole frequency.
  - Air exchange: Henry's law with Sherwood 2 + 0.6 Re^½ Sc^⅓.
  - Steam exchange: limited by heat transfer (Ranz–Marshall).
  - Motion: terminal slip velocity (Schiller–Naumann / Eötvös drag), plus the water velocity, plus a path wobble.
  - Effect on the water: buoyancy force on the water, latent heat and air exchanged with it. Bubbles coalesce on contact and burst at the surface.
- **Steam thermostat:** cuts the power after 8 s of strong venting or 20 s at 99.3 °C.

## Simplifications (it is an engineering model, not a direct boiling simulation)
- **Axisymmetric flow.** The real plumes are 3D and turbulent. The bubbles are 3D, but they move in the axisymmetric flow, with random azimuths and a wobble.
- **Nucleation is modelled.** It uses site statistics and correlations; the microlayer and the bubble shape are not resolved. The model uses 320 plate sites where a real plate has thousands. So there are fewer bubbles than in reality, and each one carries more steam.
- **Micro-bubbles:** free bubbles smaller than 80 µm go back to the dissolved air. They are below a pixel in the render and rise at less than 1.5 cm/s.
- **Flat free surface in the CFD.** The render adds waves scaled by the steam release.

## Files and reproduction
```
python3 kettle_cfd.py run1 420 24                                  # ~28 min on 4 cores; snapshots at 24 Hz
python3 export_frames.py run1 frames_in 24                         # movie timeline (timeline.json)
python3 render_kettle.py frames_in frames_png 960 540 16           # Blender, ~25 s/frame
python3 make_kettle_movie.py ../media/kettle_boil.mp4 frames_png run1 24
```
- `render_kettle.py`: the Blender scene. A glass jug with spout, handle and lid, a steel heating plate, the water with a displaced surface, bubbles instanced as spheres, a blue LED ring, and steam. `--test OUT.png` renders one still.
- `make_kettle_movie.py`: adds captions, the clock and an inset cut through the CFD (temperature | dissolved air).
- `export_frames.py`: maps the movie timeline to snapshots. Real-time windows are played at true speed, and the rest is time-lapse.
