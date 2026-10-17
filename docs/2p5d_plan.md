# Plan: quasi-3D (2.5D) reacting candle flame in a side draft

## Context
- Baseline: 2D axisymmetric laminar candle flame, OpenFOAM v2412 `rhoReactingBuoyantFoam`, 5° wedge, ~14k cells.
  Wick r < 0.75 mm, z < 6 mm; wax top at z = 0; candle radius 10 mm; domain r < 40 mm, z < 120 mm.
- Fuel: n-dodecane vapour from the wick at 0.1 g/min, 600 K. Air 300 K.
- Chemistry: Westbrook–Dryer 2-step (C12H26 → CO + H2O, CO → CO2) plus CO2 dissociation, from `gen.py`
  (A1 = 9.6e11 cgs, tuned in Cantera against nDodecane_Reitz, see `calib/`).
- Thermo: NASA7 from Cantera. Transport: Sutherland. Radiation: P1, grey CO2/H2O. No soot.
- Steady result at t = 0.5 s: flame tip 44 mm, peak T 1923 K, max |U| ≈ 2 m/s, ≈ 73 W.
- `data/steady_flame_t0.5.npz`: converged steady flame, native cells (`cell_*`) and a 0.1 mm (z, r)
  grid (`grid_*`) of T, Ur, Uz, p, rho and Y_* (C12H26, O2, CO, CO2, H2O, N2). SI units; see `info`.

## Goal
A validated quasi-3D solver (azimuthal Fourier, M = 4–8) for low-Mach reacting flow in Python
(NumPy/SciPy; Numba allowed for hot loops) in `quasi3d/`. It uses finite-rate chemistry identical to
the OpenFOAM case, for the candle flame in a gentle side draft (0.1–0.3 m/s). The draft must make the
flame lean and waver in simulation, not in post-processing. Correctness over features: every stage
needs a quantitative check, not just a picture.

Fields: f(r, z, θ) = Σ_{m=0..M} f_m(r, z) e^{imθ}.

## Stages (commit, `STATUS.md` and a figure after each; `validator` signs off each stage)
1. **Cold crossflow:** a uniform m = 1 draft through an empty domain stays uniform (error < 1 %).
   At the outer boundary: u_r = V cos θ, u_θ = −V sin θ.
2. **Hot buoyant plume:** variable density, with a fixed hot source in place of the flame. With m = 0
   only, compare against OpenFOAM on-axis velocity and temperature; then add the draft.
3. **Flame sheet:** transport mixture fraction Z (Le = 1, ρD = μ/Pr, Pr 0.7). Take state relations
   T(Z), ρ(Z) and Y_i(Z) from the npz as conditional means. m = 0 must reproduce a 44 mm ± 10 % tip;
   the lean must converge for M = 2/4/8.
4. **Finite-rate chemistry (main goal):**
   - Transport Y_C12H26, Y_O2, Y_CO, Y_CO2, Y_H2O and h, with the reactions from `gen.py`
     (constant/reactions), NASA7 thermo and Sutherland transport.
   - Radiation: P1 or optically thin.
   - Stiff sources: Strang splitting with vectorised implicit per-point integration.
   - Checks: m = 0 matches OpenFOAM (tip height and peak T within 5 %); the lean agrees with stage 3.
5. **Reference:**
   - A coarse OpenFOAM half-3D case, with a `symmetryPlane` on the plane containing the draft and
     refinement near the wick, started from the axisymmetric solution with the same draft.
   - Compare the lean angle and tip position with stage 4.
6. **Render:**
   - Line-of-sight emission from the reconstructed 3D field: soot-proxy blackbody (as in `render.py`)
     plus CH*/C2* near the stoichiometric surface (Zst = 0.0621).
   - An MP4 of the leaning flame (ffmpeg, H.264, yuv420p).
   - A README section covering method, validation and limits (few modes, no blow-out).

## Numerics advice
- **Grid:** cell-centred in r with r_i = (i + ½)Δr, so there is no node on the axis. Apply the axis
  regularity conditions for each mode; scalars vanish on the axis for m ≥ 1.
- **Pressure:** projection method with a Dodd & Ferrante constant-coefficient pressure Poisson
  (split with ρ0). Each mode then gets an independent sparse 2D solve of
  (1/r ∂r(r ∂r) − m²/r² + ∂zz). Factorise each mode once (`scipy.sparse.linalg.splu`) and reuse it.
- **Diffusion:** implicit per mode, because m²/r² is stiff near the axis.
- **Nonlinear terms:** pseudo-spectral, on N_θ ≥ 3M + 1 physical angles (dealiased).
- **Boundaries:** the outer boundary takes the base entrainment from the npz plus the m = 1 draft,
  with consistent outflow; the top is an outflow.
- **Resolution:** start coarse (0.25 mm in the flame region, stretched further out) and refine once
  stable. Density ratio ≈ 6. Use a CFL-adaptive time step.
