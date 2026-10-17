# quasi-3D solver — status

Plan: `docs/2p5d_plan.md`. Branch: `claude/claude-2p5d-plan-review-41ylqg` (the session's designated branch;
CLAUDE.md names `2p5d-prototype` — the harness only allows pushing to the designated branch).

## Environment (this container)
- No conda; system Python 3.13 with `pip install scipy numba matplotlib cantera==3.2.0` (numpy 2.x). Enough for
  stages 1–4 and 6. OpenFOAM (stage 5) still needs conda-forge `openfoam=2412`.
- `quasi3d/mech.py` regenerates `quasi3d/data/mech.json` from `gen.py` (needs Cantera) on first import.

## Code (`quasi3d/`)
| file | content |
|---|---|
| `core.py` | grid (stretched, wick mask, face labels), Fourier transforms, FV face values/gradients, MUSCL, per-mode sparse operators + splu cache |
| `solver.py` | low-Mach time stepper (scalars → EOS → momentum → projection) |
| `mech.py` | NASA7 / Sutherland / Eucken / 2-step kinetics / grey radiation, identical coefficients to `gen.py` (checked vs Cantera: rho exact, rates 1e-7) |
| `candle.py` | candle geometry and BCs (wick injection, pool, entrainment + draft), OpenFOAM reference interpolation, tip height |
| `stage1_crossflow.py`, `stage2_plume.py` | stage drivers (results in `out/*.json`, figures in `figs/`) |
| `plots.py` | stage figures |

### Method (decisions)
- Collocated cell-centred FV in (r, z), r_c = (i+½)Δr (no axis node), Fourier in θ (pseudo-spectral, N_θ = even ≥ 3M+1).
  Axis: zero geometric flux; parity ghosts f(−r, θ) = s f(r, θ+π) (s = −1 for u_r, u_θ).
- Face mass fluxes F = ρu on r/z faces (Rhie–Chow), ρu_θ at centres. **Projection on ρu** (not u):
  L_m δp = (∇·F* + (ρⁿ⁺¹−ρⁿ)/Δt)/Δt is a *constant-coefficient* Poisson equation per mode, so the Dodd–Ferrante
  splitting is not needed; each mode's matrix is factorised once (splu). Discrete continuity is exact (~1e-13).
- Momentum viscous term: implicit ν₀ vector Laplacian through u± = u_r ± i u_θ (decouples into scalar operators
  L_{m±1}), explicit remainder ∇·τ(μ) − ν₀∇²u, stable for ν₀ ≥ max(μ/ρ)/2 (ν₀ = 0.6 max, refactorised when needed).
  Same for scalar diffusion with D₀. Explicit stress uses ∂_r u_θ − u_θ/r (not r∂_r(u_θ/r), inaccurate at the axis).
- Advection: MUSCL/van Leer in (r, z), spectral in θ, advective form; AB2 in time. Δt from CFL 0.4–0.5 on a
  2^(1/4) ladder (factorisations reused). Sources (radiation, later chemistry) split.
- Species diffusion ρD = μ (Sc = 1) and α = κ/c_p for h_s, as OpenFOAM.com's reactingFoam family
  (`laplacian(muEff, Yi)`, `laplacian(alphaEff, he)`) — to be confirmed against the OpenFOAM source in stage 5.

## Stage 1 — cold crossflow: PASS (validator: PASS, recomputed max 0.21 % / 0.94 %)
Empty domain R = 40 mm, H = 120 mm; outer boundary u_r = V cos θ, u_θ = −V sin θ; bottom slip wall carrying the
draft; top outflow. Start from rest, V = 0.2 m/s ramped over 50 ms.
- Uniform flow is an exact discrete steady state (error 1e-14 after 30 steps).
- From rest: the impulsive start sets up a large vertical recirculation through the open top that washes out
  exponentially (rms ×0.7 per 1.2 s; <1 % after ≈11 s (M4) / ≈15.5 s (M2)). At t = 16 s: **M = 4, h = 1 mm: max
  error 0.20 %, rms 0.066 %; M = 2, h = 2 mm: max 0.93 %, rms 0.25 %** (criterion < 1 %).
  `out/stage1_M*.json`, `figs/stage1_crossflow.png`. Re-run after the projection rewrite: identical numbers.
- Operator verification (`test_operators.py`, manufactured m ≤ 2 fields, N = 16/32/64): interior L2 order ≈ 2 for
  diffusion, Laplacians (m = 0, 1, 2), div τ (z); ≈ 1.2–1.3 for div τ (r, θ) and MUSCL advection; boundary cells
  1st order (by design). Known limitation: cell divergence in the first ~3 cells next to the axis is 1st order, so
  −τ_θθ/r in the explicit viscous remainder has an O(1) error there (L2 still converges).
- Lesson: switch the draft on by superposing V e_x on the axisymmetric state, not from rest.

## Stage 2 — hot buoyant plume: m = 0 PASS; draft run to redo
Frozen chemistry (h_s + 5 species, full thermo/transport, optically thin radiation); inside the OpenFOAM flame
envelope (T > 1200 K or Y_fuel > 0.01, z < 46 mm) T and Y are held at the OpenFOAM values; velocity is free.
- M = 0 (grid 0.375 × 0.5 mm near the flame, 37 × 150 cells, t = 0.4 s from the OpenFOAM state):
  on-axis u_z max 1.98 vs 2.04 m/s (−3 %), rms dev 0.9 %; free plume T(60 mm) 1407 vs 1409 K,
  T(80 mm) 1168 vs 1169 K, u_z(80 mm) 1.93 vs 1.97 m/s. `figs/stage2_plume.png` (left).
- Draft M = 4, V = 0.2: plume leans (centroid x = 4/10/16 mm at z = 30/50/81 mm) but diverged at t = 0.50 s: the
  hard-pinned source acts as an unlimited heater for the cold cross-flow (huge dilatation, Gibbs undershoots).
  To redo with a relaxed source (finite time constant) after the solver fix below.

## Solver history (important)
1. ρu-projection with ∂ρ/∂t from EOS differencing: unstable for the flame sheet (growth ∝ 1/Δt).
2. → velocity projection, div u = S from the non-advective scalar rates, Dodd–Ferrante (current).
3. Fuel-element mass flow above the wick: 1.39e-6 kg/s vs OpenFOAM 1.79e-6 (supply 1.667e-6 + diffusion) with the
   volume-flux advective form → tips ≈ 38 mm (stage 3 and 4) vs 44.7. Mass-flux advective form: 2.3e-6 (too much);
   + volume-discrepancy correction β = 0.5: runaway. Handed to numerics-expert (in progress).
- OpenFOAM uses ρD = μ for species: fitting OpenFOAM's own steady Z equation gives ρD = 0.993 μ (vs 0.72 μ for
  κ/c_p). Stage 3 therefore uses Sc = 1 (the plan's ρD = μ/0.7 gives a 31 mm tip, i.e. 0.7 × 44 mm as expected).

## Stage 3 — flame sheet: FAIL so far (tip), pending the conservation fix
- Z with conditional-mean state relations from the npz; M = 0, Sc = 1: the flame flickers at ≈11 Hz, tip 36–43 mm,
  mean ≈ 39 mm (−12 %); same on a 0.25 mm grid and with CFL 0.2. Sc = 0.7: ≈ 31 mm.

## Stage 4 — finite-rate chemistry: implemented, m = 0 tip short (same cause as stage 3)
- `chem.py`: per-point constant-pressure reactor, adaptive ROS2 with FD Jacobian (numba, parallel); matches a
  Cantera IdealGasConstPressureReactor with the same mechanism (T 1923 vs 1921 K mid-ignition, identical final
  state). Strang splitting.
- M = 0 from the OpenFOAM state, t = 0.3 s: steady (no flicker); Tmax 1959 K (+1.8 % ✓), tip 38.5 mm (axis Zst) /
  35.8 mm (5 % fuel-rate) vs 44.7 (✗) — due to the fuel-flux loss above.

## Stage 5 — OpenFOAM half-3D reference: running
- `of3d/gen3d.py` + `of3d/Allrun` (OpenFOAM v2412 via micromamba at /opt/mm/root/envs/of; activate with
  `export CONDA_PREFIX=/opt/mm/root/envs/of PATH=$CONDA_PREFIX/bin:$PATH; source etc/env.sh`).
  Half cylinder with symmetryPlane y = 0, O-grid, 119k cells (checkMesh OK, max non-orth 21°), uniform draft
  V = 0.2 m/s on outer/open, initialised from the axisymmetric solution + draft, no igniter.
  Started with NP = 2 (dt ≈ 2.7e-5, ≈2 h for 0.3 s). Installing micromamba: only conda.anaconda.org is reachable;
  get `micromamba-2.9.0-0.tar.bz2` from conda-forge/linux-64 directly.

## Stage 6 — render: renderer being written (`render3d.py`); snapshots via `SNAP=<s>` env in the drivers.

## Next
Conservation fix → re-validate stage 3 (tip, M = 2/4/8 lean) and stage 4 (m = 0) → stage 4 draft runs → compare with
stage 5 → render/MP4/README.
