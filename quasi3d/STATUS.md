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

## Stage 1 — cold crossflow: PASS (pending validator)
Empty domain R = 40 mm, H = 120 mm; outer boundary u_r = V cos θ, u_θ = −V sin θ; bottom slip wall carrying the
draft; top outflow. Start from rest, V = 0.2 m/s ramped over 50 ms.
- Uniform flow is an exact discrete steady state (error 1e-14 after 30 steps).
- From rest: an impulsive start sets up a large vertical recirculation through the open top that washes out
  exponentially (rms ×0.7 per 1.2 s). At t = 16 s: **M = 4, h = 1 mm: max error 0.20 %, rms 0.066 %;
  M = 2, h = 2 mm: max 0.93 %, rms 0.25 %** (criterion < 1 %). `out/stage1_M*.json`, `figs/stage1_crossflow.png`.
- Lesson: switch the draft on by superposing V e_x on the axisymmetric state (used from stage 2 on), not from rest.

## Stage 2 — hot buoyant plume: in progress
Frozen chemistry (h_s + 5 species, full thermo/transport, optically thin radiation); inside the OpenFOAM flame
envelope (T > 1200 K or Y_fuel > 0.01, z < 46 mm) T and Y are held at the OpenFOAM values; velocity is free.
- M = 0 (grid 0.375 × 0.5 mm near the flame, 37 × 150 cells, t = 0.4 s from the OpenFOAM state):
  on-axis u_z max 1.98 vs 2.04 m/s (−2.9 %), rms dev 0.8 %; free plume T(60 mm) 1407 vs 1409 K,
  T(80 mm) 1168 vs 1169 K, u_z(80 mm) 1.93 vs 1.97 m/s.
- Draft run M = 4, V = 0.2 m/s: running.

## Next
Stage 2 draft → stage 3 (mixture fraction + state relations) → stage 4 (finite-rate chemistry).

## Open problems / notes
- Stage 1 convergence from rest is slow (transient through the open top), not an accuracy problem.
