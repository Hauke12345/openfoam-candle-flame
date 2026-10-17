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

## Stage 2 — hot buoyant plume: m = 0 PASS (validator: PASS, u_z max dev 3.2 %, free-plume T within 0.4 %); draft run to redo
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
2. → velocity projection, div u = S from the non-advective scalar rates, Dodd–Ferrante.
3. Fuel-element conservation (numerics-expert): S from the chain rule assumes ρ is advected consistently with Z;
   upwind-cell face densities violate that by up to 3× in the dense vapour layer (±30 % spurious fuel source).
   Fix: face mass fluxes F = ρ_EOS(φ_face)·U, wick velocity ṁ''/ρ_w; grid refined to 0.1875 mm for r < 4.5 mm,
   z < 9 mm (`WICK_H=0.1875 WICK_RF=4.5`, 50 × 182 cells). Z budget closes to 1.5 %. Keep β (volume-discrepancy
   correction) = 0: with ρ reset to the EOS every step it recreates the mismatch.
4. Draft blow-up (numerics-expert): truncating ρ = ρ_EOS(Z) to M modes went negative when the draft bends the sheet
   (ρ(Z) very nonlinear) → ρ0 → 0. Fix: pointwise EOS density; dlnρ/dZ = 0 outside [0, 1]; smoothed rich-side
   state table (raw conditional means were non-monotone).
5. Gibbs ringing in θ (numerics-expert, then finished by the coordinator): a leaning flame/plume is an off-axis
   structure that M = 4–8 modes cannot represent; pseudo-spectral θ advection rings. One scalar (Z) with clipping
   still gave a mass mismatch ~ −2.5e-4 kg/s; six scalars (stage 4) became unrealisable (Y_O2 0.28, T 2574 K).
   Fix: THETA_FV=1 — scalars live at the N_θ collocation angles (stored as all N_θ/2+1 modes, i.e. exactly) and are
   advected in θ with conservative MUSCL/van Leer face fluxes; the θ face velocities are chosen such that the FV θ
   divergence equals the spectral divergence at every collocation angle (factor (mΔθ/2)/sin(mΔθ/2)·e^{imΔθ/2} per
   mode), so the scalars see exactly the divergence the projection enforces. Velocity and pressure stay spectral
   (M modes). Result: bounded scalars, mass mismatch ≈ −2e-5 kg/s (10× lower), stage-4 draft physical. Stage 1 is
   unchanged with and without it. Use THETA_FV=1 for all draft runs.
- OpenFOAM uses ρD = μ for species (fit of OpenFOAM's own steady Z equation: 0.993 μ vs 0.72 μ for κ/c_p), so
  stage 3 uses Sc = 1 (the plan's ρD = μ/0.7 gives a 31 mm tip).
- Chemistry: points whose initial rates change nothing over Δt are skipped, rtol 1e-3 (≈5 s/step at M = 4).

## Stage 3 — flame sheet (validator: m = 0 tip PASS, 46.8 / 45.6 mm mean; Sc = 1 justified, ρD = 0.99–1.02 μ)
- M = 0 (refined grid): flickers at ≈ 11 Hz, mean tip 45.6–46.8 mm (OpenFOAM 44.7) ✓ ±10 %.
- Draft V = 0.2 m/s, 0.2 s (started by superposing V e_x on the M = 0 state). The flame lays over ~60° — physical,
  OpenFOAM half-3D shows the same. Mean ± sd over the last 50 ms (`out/stage3_lean_summary.json`, `figs/stage3_lean.png`):

  | run | centroid x (mm) | centroid z (mm) | tip z | tip x | mass mismatch (kg/s) |
  |---|---|---|---|---|---|
  | THETA_FV M = 2 | 4.7 ± 0.7 | 13.3 ± 5.5 | 26.8 ± 8.6 | 3.7 ± 0.7 | −2.6e-5 |
  | THETA_FV M = 4 | 12.3 ± 0.2 | 9.3 ± 0.4 | 22.6 ± 1.7 | 23.8 ± 1.2 | −2.0e-5 |
  | THETA_FV M = 8 | 11.9 ± 0.4 | 10.1 ± 0.8 | 29.4 ± 3.3 | 23.8 ± 1.2 | −2.1e-5 |
  | spectral+clip M = 8 | 12.6 ± 0.5 | 12.7 ± 0.1 | 34.4 ± 0.8 | 25.9 ± 1.3 | −2.9e-4 |
  | spectral+clip M = 12 | 12.5 ± 0.3 | 11.8 ± 0.7 | 32.6 ± 2.8 | 25.5 ± 1.0 | −2.8e-4 |
  | OpenFOAM half-3D (0.2–0.3 s) | 10.5–10.9 | 8.7–9.0 | 23–24 | 20.1 | – |

  (OpenFOAM numbers with the SAME region definition as `flame_geometry`: Z ≥ Zst over all fluid cells; an earlier
  version of tools/of3d_flame.py excluded z < 6 mm and gave 12.0/12.0 mm — that comparison was inconsistent, caught
  by the validator.) Verdict (validator: PARTIAL / FAIL as worded): the centroid converges from M = 4 to 8 (x within
  0.4 mm, z within 0.9 mm), but the tip height does not (22.6 → 29.4 mm, M = 8 still oscillating); **M = 2 does not
  lean at all** (two velocity modes cannot carry the flame off-axis). Against OpenFOAM the THETA_FV flame sits
  ≈ 1.4–1.8 mm (13–17 %) further downstream; centroid height agrees within ≈ 1 mm.

## Stage 4 — finite-rate chemistry (validator: m = 0 PASS; Tmax +1.4 %, tip −3.6 / +2.9 %; axis T +45 K above the flame, axis u_z 6–9 % low at 10–30 mm)
- `chem.py` verified against Cantera (constant-pressure reactor). Strang splitting.
- M = 0, refined grid, t = 0.3 s: tip 43.1 mm (5 % fuel-rate) / 46.0 mm (axis Zst) vs 44.7 ✓; Tmax 1950 vs
  1923 K (+1.4 %) ✓; steady.
- Draft, pseudo-spectral + CLIP (M = 8): unphysical after 30 ms (species Gibbs: Y_O2 0.28, T 2574 K) — abandoned.
- Draft with THETA_FV, V = 0.2 m/s, 0.2 s (out/stage4_M4_V0.2_h0.375_fv3.json; continuation chain _fv → _fv2 →
  _fv3 with DRAFT_KICK=0). M = 4, mean over the last 50 ms: Tmax 2066 ± 2 K (OpenFOAM 3D 2061), tip (z 27.8, x 25.5)
  mm (OpenFOAM 23.2, 20.1), centroid x 13.7 ± 0.3 mm; final state (validator) centroid (14.1, 11.0) mm, tip angle 54°,
  flame volume 2.76 cm³. Still drifting at t = 0.5 (centroid x 12.4 → 14.1 over the last 70 ms) → continued to 0.6 s
  (_fv4). vs stage 3 THETA_FV M = 4 (12.3, 9.3): +15 % in x (validator: PASS for the centroid, borderline tip).
  Physics sanity (validator): T 300–2064 K, ΣY = 1, Z_C vs Z_H within 7.6e-4, mirror-symmetric about θ = 0.
  M = 8 continuation (_fv3) running as the mode-convergence check (agreed with M = 4 at 30 ms: centroid x 5.4/5.2).
- Open: Qchem diagnostic (ρΣhf ΔY over both Strang halves) reads ≈130 W vs ≈73 W from fuel × LHV; the solution
  (Tmax, shape) matches OpenFOAM, so this looks like a diagnostic issue — not resolved.

## Stage 5 — OpenFOAM half-3D reference (validator: FAIL as originally claimed — inconsistent metric, now corrected)
- Consistent comparison (Z ≥ Zst, all fluid cells), OpenFOAM t = 0.2–0.3 s: centroid x 10.5–10.9 mm, z 8.7–9.0 mm,
  tip (z 23–24, x 20.1) mm, tip angle 48–49°, flame volume 2.2–2.3 cm³. Stage 4 (M = 4, t = 0.5): centroid
  (14.1, 11.0) mm (+33 % downstream), tip (26–28, 25–28) mm, tip angle ≈ 50–54° (agrees), volume 2.76 cm³.
  So the quasi-3D flame leans the same way and with the same tip angle, but its body sits ≈ 3.5 mm further
  downstream and its tip 4–7 mm further out/up. Both plumes reach the outer boundary (R ≈ 38 mm) above z ≈ 25 mm,
  so the far-field lean is domain-dependent in both codes.
- `of3d/gen3d.py`, `of3d/Allrun`, 119k cells, V = 0.2 m/s, started from the axisymmetric solution + draft
  (OpenFOAM via micromamba at /opt/mm/root/envs/of; `export CONDA_PREFIX=/opt/mm/root/envs/of
  PATH=$CONDA_PREFIX/bin:$PATH; source etc/env.sh`). `tools/of3d_flame.py CASE times` → flame geometry.
- DONE (t = 0.3 s). Numbers in this list exclude z < 6 mm (old tool version; see the consistent numbers above):
  t = 0.05 s tip (36.1, 9.5) mm, centroid 39°; 0.1 s tip (27.3, 15.3), 55°;
  0.3 s tip (z 23.2, x 20.1) mm, centroid (x 12.0, z 12.0) mm, 63.4°, Tmax 2061 K;
  0.2 s tip (24.0, 20.1) mm, centroid (12.5, 12.4) mm, 63° (steady from ≈ 0.2 s); Tmax 2059 K; max |u| 2.18 →
  1.63 (0.12 s) → 1.95 m/s (0.24 s, at x = 37 mm, z = 83 mm: the leaning plume reaches the outer boundary).

## Stage 6 — render: done (pending validator)
- `render3d.py`: line-of-sight soot-proxy blackbody + chemiluminescence through the reconstructed 3D field, THETA_FV
  scalars interpolated linearly in θ. Movie `figs/candle_draft_quasi3d.mp4` (H.264, yuv420p, 400×640, 12 fps, 41
  frames: M = 0 state at t = 0.30, then every 5 ms of the M = 4 THETA_FV draft run to t = 0.50); exposure smoothed
  per frame (NORM=smooth; the leaning flame's chemiluminescence peak is ~30× the upright one). Frames:
  `out/movie_frames` (symlinks). Still strip `figs/candle_draft_strip.png`. README section written.

## Next
- M = 8 stage-4 draft continuation → mode convergence for stage 4; validator for stages 4–6.
- Optional: redo the stage-2 draft run with THETA_FV and a relaxed source (TAU_SRC=0.005).

Conservation fix → re-validate stage 3 (tip, M = 2/4/8 lean) and stage 4 (m = 0) → stage 4 draft runs → compare with
stage 5 → render/MP4/README.
