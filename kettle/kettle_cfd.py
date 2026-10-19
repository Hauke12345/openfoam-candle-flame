"""Glass electric kettle, 20 °C to full boil: axisymmetric liquid CFD + Lagrangian air/vapour bubbles.

Liquid (r, z), axisymmetric, incompressible with buoyancy from the real water density rho(T):
  MAC grid, SSP-RK3, 2nd-order upwind momentum advection, TVD (van Leer) scalars, Smagorinsky eddy viscosity,
  projection with a prefactored axisymmetric Poisson matrix. Scalars: temperature T and dissolved air C (mol/m3).
  No-slip bottom (heating plate) and side (glass), free-slip flat free surface on top.
Heating plate (steel/aluminium sandwich, radial 1D conduction, heat capacity) fed by a 2 kW ring element.
  Plate -> water heat flux: turbulent natural convection (Nu = 0.14 Ra^1/3) + nucleate boiling (Cooper) weighted by
  the fraction of active nucleation sites F(dT_wall) (log-normal activation superheats). A part eps(subcooling) of
  the boiling flux makes vapour at the model nucleation sites; the rest heats the liquid next to the plate.
Bubbles (3D points, each holds n_gas mol air + n_vap mol steam), two-way coupled:
  - nucleate at sites on the plate (boiling and degassing) and on the glass (degassing only), grow attached,
    depart at a departure diameter (Fritz at saturation, smaller when subcooled), frequency-limited (Cole);
  - air exchange with the dissolved air (Henry; Sherwood 2 + 0.6 Re^1/2 Sc^1/3), vapour exchange limited by
    heat transfer (Ranz-Marshall) toward the liquid temperature: vapour bubbles collapse in subcooled water
    and leave a small air bubble behind; air bubbles swell with steam in hot water;
  - rise at the terminal slip velocity (Schiller-Naumann / Eotvos drag) + liquid velocity + a path wobble;
  - push the liquid (buoyancy as a body force), give/take latent heat and dissolved air, coalesce on contact,
    burst at the free surface (steam and air leave).
Free surface: convective loss + evaporation; air exchange with the room. Glass side: small heat loss.
The kettle switches off after steam has vented strongly for a few seconds (like its steam thermostat).

usage: python3 kettle_cfd.py OUTDIR [t_end=400] [fps=24]   -> OUTDIR/snap/s%06d.npz at fps, OUTDIR/log.json
"""
import sys, os, json, time
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
from scipy.special import erf

OUT = sys.argv[1] if len(sys.argv) > 1 else 'run'
T_END = float(sys.argv[2]) if len(sys.argv) > 2 else 400.0
FPS = float(sys.argv[3]) if len(sys.argv) > 3 else 24.0
os.makedirs(f'{OUT}/snap', exist_ok=True)
rng = np.random.default_rng(7)

# ---------------- geometry, grid ----------------
R, H = 0.070, 0.110                      # inner radius, water depth at 20 C (1.69 L)
NR, NZ = 40, 63
dr, dz = R / NR, H / NZ
rc = (np.arange(NR) + 0.5) * dr          # cell centres
rf = np.arange(NR + 1) * dr              # r-faces
zc = (np.arange(NZ) + 0.5) * dz
VOL = 2 * np.pi * rc[:, None] * dr * dz * np.ones((1, NZ))
A_RING = 2 * np.pi * rc * dr             # plate / free-surface area of each radial cell
A_SIDE = 2 * np.pi * R * dz
A_PLATE = np.pi * R * R

g, RG, MW, PATM = 9.81, 8.314, 0.018015, 101325.0
T_AMB, P_HEAT = 20.0, 2000.0

# ---------------- water properties (T in C) ----------------
def rho_w(T):
    return 1000 * (1 - (T + 288.9414) / (508929.2 * (T + 68.12963)) * (T - 3.9863) ** 2)
def mu_w(T):
    return 2.414e-5 * 10 ** (247.8 / (T + 273.15 - 140))
def k_w(T):
    t = (T + 273.15) / 298.15
    return 0.6065 * (-1.48445 + 4.12292 * t - 1.63866 * t * t)
CP = 4185.0
def p_v(T):                              # Antoine, Pa
    return 133.322 * 10 ** (8.07131 - 1730.63 / (233.426 + T))
def T_sat(p):
    return 1730.63 / (8.07131 - np.log10(np.maximum(p, 1.0) / 133.322)) - 233.426
def sigma_w(T):
    return 0.0728 - 0.000155 * (T - 20)
def L_v(T):
    return 2.501e6 - 2361.0 * T
def beta_w(T):
    return -(rho_w(T + 0.5) - rho_w(T - 0.5)) / rho_w(T)
H0 = 0.82 / (PATM - p_v(20.0))           # air-saturated water at 20 C holds 0.82 mol/m3 of air (N2 + O2)
def henry(T):                            # mol/m3 per Pa of air partial pressure
    return H0 * np.exp(1500.0 * (1 / (T + 273.15) - 1 / 293.15))
def c_sat(T):                            # under 1 atm of moist air
    return henry(T) * np.maximum(PATM - p_v(T), 0.0)
def D_air(T):
    return 2.0e-9 * (T + 273.15) / 293.15 * mu_w(20.0) / mu_w(T)
TSAT_B = T_sat(PATM + 958 * g * H)       # saturation temperature at the plate (~100.3 C)

# ---------------- projection ----------------
N = NR * NZ
Lm = lil_matrix((N, N))
for i in range(NR):
    for j in range(NZ):
        k = i * NZ + j; d = 0.0
        for ii, c in ((i + 1, rf[i + 1] / (rc[i] * dr * dr)), (i - 1, rf[i] / (rc[i] * dr * dr))):
            if 0 <= ii < NR:
                Lm[k, ii * NZ + j] = c; d -= c
        for jj in (j + 1, j - 1):
            if 0 <= jj < NZ:
                Lm[k, i * NZ + jj] = 1 / dz ** 2; d -= 1 / dz ** 2
        Lm[k, k] = d
Lm[0, :] = 0; Lm[0, 0] = 1.0
LU = splu(Lm.tocsc())

def project(u, w, dt):
    div = (rf[1:, None] * u[1:] - rf[:-1, None] * u[:-1]) / (rc[:, None] * dr) + (w[:, 1:] - w[:, :-1]) / dz
    b = (div / dt).ravel(); b[0] = 0.0
    phi = LU.solve(b).reshape(NR, NZ)
    u[1:-1] -= dt * (phi[1:] - phi[:-1]) / dr
    w[:, 1:-1] -= dt * (phi[:, 1:] - phi[:, :-1]) / dz
    return u, w

# ---------------- liquid right-hand sides ----------------
def pad_u(u):   # u_r on r-faces (NR+1, NZ): odd at the axis, 0 at the glass; no-slip bottom, free-slip top
    P = np.zeros((NR + 5, NZ + 4))
    P[2:NR + 3, 2:NZ + 2] = u
    P[1] = -P[3]; P[0] = -P[4]
    P[NR + 3] = -P[NR + 1]; P[NR + 4] = -P[NR]
    P[:, 1] = -P[:, 2]; P[:, 0] = -P[:, 3]
    P[:, NZ + 2] = P[:, NZ + 1]; P[:, NZ + 3] = P[:, NZ]
    return P

def pad_w(w):   # u_z on z-faces (NR, NZ+1): even at the axis, no-slip at the glass; 0 at bottom and top
    Q = np.zeros((NR + 4, NZ + 5))
    Q[2:NR + 2, 2:NZ + 3] = w
    Q[1] = Q[2]; Q[0] = Q[3]
    Q[NR + 2] = -Q[NR + 1]; Q[NR + 3] = -Q[NR]
    Q[:, 1] = -Q[:, 3]; Q[:, 0] = -Q[:, 4]
    Q[:, NZ + 3] = -Q[:, NZ + 1]; Q[:, NZ + 4] = -Q[:, NZ]
    return Q

def d_upw(S, a, h, ax):
    """a * dq/dx with 2nd-order upwinding; S(da, db) returns the shifted interior view."""
    o = (lambda k: S(k, 0)) if ax == 0 else (lambda k: S(0, k))
    dm = (3 * o(0) - 4 * o(-1) + o(-2)) / (2 * h)
    dp = (-3 * o(0) + 4 * o(1) - o(2)) / (2 * h)
    return a * np.where(a > 0, dm, dp)

def face_tvd(q, v, ax):
    """advective face values of a cell field q along axis ax (van Leer), upwinded by the interior face velocity v."""
    q = np.moveaxis(q, ax, 0); v = np.moveaxis(v, ax, 0)
    qe = np.concatenate([q[:1], q, q[-1:]], 0)
    d = np.diff(qe, axis=0)
    dl, dr_ = d[:-1], d[1:]
    pr = dl * dr_
    s = np.where(pr > 0, 2 * pr / (dl + dr_ + 1e-30), 0.0)
    qp, qm = q + 0.5 * s, q - 0.5 * s                 # values at the + and - faces of each cell
    f = np.where(v > 0, qp[:-1], qm[1:])
    return np.moveaxis(f, 0, ax)

def nu_turb(u, w):
    uc = 0.5 * (u[1:] + u[:-1]); wc = 0.5 * (w[:, 1:] + w[:, :-1])
    srr = (u[1:] - u[:-1]) / dr; szz = (w[:, 1:] - w[:, :-1]) / dz; stt = uc / rc[:, None]
    srz = 0.5 * (np.gradient(uc, dz, axis=1) + np.gradient(wc, dr, axis=0))
    S = np.sqrt(2 * (srr ** 2 + szz ** 2 + stt ** 2 + 2 * srz ** 2))
    return (0.17 * np.sqrt(dr * dz)) ** 2 * S

def scalar_rhs(q, u, w, kap):
    """-div(u q) + div(kap grad q), zero-flux walls (boundary fluxes are added as sources)."""
    Fr = np.zeros((NR + 1, NZ)); Fz = np.zeros((NR, NZ + 1))
    kr = 0.5 * (kap[1:] + kap[:-1]); kz = 0.5 * (kap[:, 1:] + kap[:, :-1])
    Fr[1:-1] = rf[1:-1, None] * (u[1:-1] * face_tvd(q, u[1:-1], 0) - kr * (q[1:] - q[:-1]) / dr)
    Fz[:, 1:-1] = w[:, 1:-1] * face_tvd(q, w[:, 1:-1], 1) - kz * (q[:, 1:] - q[:, :-1]) / dz
    return -(Fr[1:] - Fr[:-1]) / (rc[:, None] * dr) - (Fz[:, 1:] - Fz[:, :-1]) / dz

def rhs(u, w, T, C, src):
    Tc = np.clip(T, 0.0, 130.0)
    nut = nu_turb(u, w)
    nu = mu_w(Tc) / rho_w(Tc) + nut
    kap_T = k_w(Tc) / (rho_w(Tc) * CP) + nut / 0.85
    kap_C = D_air(Tc) + nut / 0.85
    # u_r on interior faces a = 1..NR-1
    P = pad_u(u)
    Su = lambda da, db: P[3 + da:NR + 2 + da, 2 + db:NZ + 2 + db]
    ua = u[1:-1]
    wa = 0.25 * (w[:-1, :-1] + w[1:, :-1] + w[:-1, 1:] + w[1:, 1:])
    nuf = 0.5 * (nu[1:] + nu[:-1])
    r = rf[1:-1, None]
    lap_u = ((rc[1:, None]) * (Su(1, 0) - Su(0, 0)) - (rc[:-1, None]) * (Su(0, 0) - Su(-1, 0))) / (r * dr * dr) \
        - Su(0, 0) / r ** 2 + (Su(0, 1) - 2 * Su(0, 0) + Su(0, -1)) / dz ** 2
    du = -d_upw(Su, ua, dr, 0) - d_upw(Su, wa, dz, 1) + nuf * lap_u
    # u_z on interior faces b = 1..NZ-1
    Q = pad_w(w)
    Sw = lambda da, db: Q[2 + da:NR + 2 + da, 3 + db:NZ + 2 + db]
    wb = w[:, 1:-1]
    ub = 0.25 * (u[:-1, :-1] + u[1:, :-1] + u[:-1, 1:] + u[1:, 1:])
    nuz = 0.5 * (nu[:, 1:] + nu[:, :-1])
    lap_w = (rf[1:, None] * (Sw(1, 0) - Sw(0, 0)) - rf[:-1, None] * (Sw(0, 0) - Sw(-1, 0))) / (rc[:, None] * dr * dr) \
        + (Sw(0, 1) - 2 * Sw(0, 0) + Sw(0, -1)) / dz ** 2
    Tf = 0.5 * (Tc[:, 1:] + Tc[:, :-1])
    rho_ref = rho_w(np.sum(Tc * VOL) / VOL.sum())
    buoy = g * (rho_ref - rho_w(Tf)) / rho_ref + 0.5 * (src['fz'][:, 1:] + src['fz'][:, :-1])
    dw = -d_upw(Sw, ub, dr, 0) - d_upw(Sw, wb, dz, 1) + nuz * lap_w + buoy
    dU = np.zeros_like(u); dU[1:-1] = du
    dW = np.zeros_like(w); dW[:, 1:-1] = dw
    dT = scalar_rhs(Tc, u, w, kap_T) + src['qT'] / (rho_w(Tc) * CP)
    dC = scalar_rhs(C, u, w, kap_C) + src['qC']
    return dU, dW, dT, dC, nut

# ---------------- heating plate ----------------
C_PLATE = 11e3                           # J/m2K: 3 mm aluminium spreader + 1 mm steel
KD_PLATE = 0.6                           # W/K: in-plane conductance (k * thickness)
qh_shape = np.exp(-0.5 * ((rc - 0.045) / 0.008) ** 2)
qh_shape /= np.sum(qh_shape * A_RING)    # W/m2 per W of heater power
TH_MED, TH_S = 6.0, 0.5                  # site activation superheat: log-normal, median 6 K
def F_act(dT):
    x = np.maximum(dT, 1e-6)
    return np.where(dT > 0, 0.5 * (1 + erf((np.log(x) - np.log(TH_MED)) / (TH_S * np.sqrt(2)))), 0.0)
def q_boil(dT):                          # Cooper (water, p_r = 0.0046, M = 18): q = (4.26 dT)^3.03
    return np.where(dT > 0, (4.26 * np.maximum(dT, 0)) ** 3.03, 0.0)
def h_natconv(Tp, T1):
    Tf = 0.5 * (Tp + T1); dT = np.abs(Tp - T1) + 1e-6
    nu = mu_w(Tf) / rho_w(Tf); al = k_w(Tf) / (rho_w(Tf) * CP)
    return 0.14 * k_w(Tf) * (g * beta_w(Tf) * dT / (nu * al)) ** (1 / 3)

# ---------------- nucleation sites ----------------
KP, KG = 320, 420
rs = (R - 0.002) * np.sqrt(rng.random(KP)); ths = 2 * np.pi * rng.random(KP)
SP = dict(x=rs * np.cos(ths), y=rs * np.sin(ths), ring=np.minimum((rs / dr).astype(int), NR - 1),
          th_boil=TH_MED * np.exp(TH_S * rng.standard_normal(KP)),
          s_gas=1.0 * np.exp(0.5 * rng.standard_normal(KP)),
          occ=np.zeros(KP, bool), wait=np.zeros(KP))
zg = 0.002 + (H - 0.006) * rng.random(KG); thg = 2 * np.pi * rng.random(KG)
SG = dict(th=thg, z=zg, cell=np.minimum((zg / dz).astype(int), NZ - 1),
          s_gas=0.7 * np.exp(0.4 * rng.standard_normal(KG)), occ=np.zeros(KG, bool), wait=np.zeros(KG))

# ---------------- bubbles ----------------
FIELDS = ('x', 'y', 'z', 'ng', 'nv', 'r', 'kind', 'site', 'age', 'wx', 'wy', 'boil')
B = {k: np.zeros(0) for k in FIELDS}
B['kind'] = np.zeros(0, int); B['site'] = np.zeros(0, int); B['boil'] = np.zeros(0, bool)

def add_bubbles(**kw):
    n = len(kw['x'])
    for k in FIELDS:
        v = kw.get(k, np.zeros(n))
        B[k] = np.concatenate([B[k], np.asarray(v, dtype=B[k].dtype)])

def keep(mask):
    for k in FIELDS:
        B[k] = B[k][mask]

def slip_velocity(d, T):
    rl = rho_w(T); nu = mu_w(T) / rl; sg = sigma_w(T)
    Eo = g * rl * d * d / sg
    u = np.minimum(g * d * d / (18 * nu), 0.3)
    for _ in range(6):
        Re = np.maximum(u * d / nu, 1e-6)
        Cd = np.maximum(24 / Re * (1 + 0.15 * Re ** 0.687), 8 / 3 * Eo / (Eo + 4))
        u = 0.5 * u + 0.5 * np.sqrt(4 * g * d / (3 * Cd))
    return u

def bubble_volume(ng, nv, T, p):
    return (ng + nv) * RG * (T + 273.15) / p

def radius_from(ng, nv, T, p0):
    """radius including the Laplace pressure (two fixed-point passes)."""
    r = np.cbrt(3 * bubble_volume(ng, nv, T, p0) / (4 * np.pi))
    for _ in range(2):
        p = p0 + 2 * sigma_w(T) / np.maximum(r, 1e-6)
        r = np.cbrt(3 * bubble_volume(ng, nv, T, p) / (4 * np.pi))
    return r

def cell_of(x, y, z):
    rr = np.hypot(x, y)
    return np.clip((rr / dr).astype(int), 0, NR - 1), np.clip((z / dz).astype(int), 0, NZ - 1), rr

def interp_vel(u, w, rr, z):
    """bilinear (r, z) liquid velocity at the bubble positions from the MAC faces."""
    uc = 0.5 * (u[1:] + u[:-1]); wc = 0.5 * (w[:, 1:] + w[:, :-1])
    fr = np.clip(rr / dr - 0.5, 0, NR - 1.001); fz = np.clip(z / dz - 0.5, 0, NZ - 1.001)
    i0 = fr.astype(int); j0 = fz.astype(int); a = fr - i0; b = fz - j0
    bil = lambda F: (F[i0, j0] * (1 - a) * (1 - b) + F[i0 + 1, j0] * a * (1 - b) + F[i0, j0 + 1] * (1 - a) * b
                     + F[i0 + 1, j0 + 1] * a * b)
    return bil(uc), bil(wc)

# ---------------- state ----------------
u = np.zeros((NR + 1, NZ)); w = np.zeros((NR, NZ + 1))
T = np.full((NR, NZ), T_AMB) + 0.01 * rng.standard_normal((NR, NZ))
C = np.full((NR, NZ), c_sat(T_AMB))
Tp = np.full(NR, T_AMB)
t, step, heater_on, t_off = 0.0, 0, True, None
vent_v_s, vent_g_s, t_strong, t_hot = 0.0, 0.0, 0.0, 0.0     # smoothed steam / air release (kg/s, mol/s)
E_in = 0.0; E_vent = 0.0; E_loss = 0.0
m_water0 = np.sum(rho_w(T) * VOL)
log = []
next_snap, isnap, next_log = 0.0, 0, 0.0
wall0 = time.time()
z_top = H

def snapshot():
    global isnap
    kind = B['kind']
    Tm = np.sum(T * VOL) / VOL.sum()
    vb = np.sum(4 / 3 * np.pi * B['r'] ** 3)
    rho_m = np.sum(rho_w(T) * VOL) / VOL.sum()
    h_level = m_water0 / rho_m / (np.pi * R * R) + vb / (np.pi * R * R)
    np.savez_compressed(f'{OUT}/snap/s{isnap:06d}.npz',
                        xyz=np.c_[B['x'], B['y'], B['z']].astype(np.float32), r=B['r'].astype(np.float32),
                        kind=kind.astype(np.int8), boil=B['boil'], T=T.astype(np.float16), C=C.astype(np.float16),
                        u=(0.5 * (u[1:] + u[:-1])).astype(np.float16), w=(0.5 * (w[:, 1:] + w[:, :-1])).astype(np.float16),
                        Tp=Tp.astype(np.float32),
                        sc=np.array([t, Tm, h_level, vent_v_s, vent_g_s, float(heater_on), vb, np.mean(C) / c_sat(20.0)]))
    isnap += 1

while t < T_END:
    Tc = np.clip(T, 0, 130)
    nb = len(B['x'])
    # ---------------- time step ----------------
    umax = max(np.abs(u).max(), np.abs(w).max(), 1e-3)
    nut = nu_turb(u, w)
    numax = (mu_w(Tc) / rho_w(Tc) + nut).max()
    dt = min(4e-3, 0.4 * min(dr, dz) / umax, 0.2 * min(dr, dz) ** 2 / numax)
    # ---------------- plate ----------------
    T1 = Tc[:, 0]
    dTw = Tp - TSAT_B
    sub = np.maximum(TSAT_B - T1, 0.0)
    q_nc = h_natconv(Tp, T1) * (Tp - T1)
    q_nb = F_act(dTw) * q_boil(dTw)
    eps = 0.25 + 0.45 * np.exp(-sub / 20.0)
    q_ev = eps * q_nb
    q_liq = q_nc + (1 - eps) * q_nb
    qh = P_HEAT * float(heater_on) * qh_shape
    lap_p = np.zeros(NR)
    lap_p[:-1] += rf[1:-1] * (Tp[1:] - Tp[:-1]) / dr
    lap_p[1:] -= rf[1:-1] * (Tp[1:] - Tp[:-1]) / dr
    Tp = Tp + dt * (qh - q_nc - q_nb + KD_PLATE * lap_p / (rc * dr)) / C_PLATE
    E_in += dt * P_HEAT * float(heater_on)

    # ---------------- sources from bubbles ----------------
    qT = np.zeros((NR, NZ)); qC = np.zeros((NR, NZ)); fz = np.zeros((NR, NZ))
    qC0 = np.zeros((NR, NZ))
    Tf_plate = 0.5 * (Tp + T1)
    # nucleation: plate sites
    tP = SP['ring']
    boil_act = dTw[tP] > SP['th_boil']
    S_plate = C[tP, 0] / np.maximum(c_sat(np.minimum(Tf_plate[tP], 99.0)), 1e-3) - 1
    gas_act = S_plate > SP['s_gas']
    new = (~SP['occ']) & (t >= SP['wait']) & (boil_act | gas_act)
    if new.any():
        k = np.nonzero(new)[0]; r0 = 3e-5
        Tl = np.minimum(Tf_plate[tP[k]], TSAT_B); pb = PATM + 958 * g * H + 2 * sigma_w(Tl) / r0
        ntot = pb * 4 / 3 * np.pi * r0 ** 3 / (RG * (Tl + 273.15))
        # boiling sites start from a vapour nucleus; degassing sites from an air nucleus taken out of the water
        fv = np.where(boil_act[k], 0.999, np.minimum(p_v(Tl) / pb, 0.9))
        add_bubbles(x=SP['x'][k], y=SP['y'][k], z=np.full(len(k), r0), ng=ntot * (1 - fv), nv=ntot * fv,
                    r=np.full(len(k), r0), kind=np.ones(len(k), int), site=k, boil=boil_act[k])
        np.add.at(qC0, (tP[k], 0), -ntot * (1 - fv) / dt / VOL[tP[k], 0])
        SP['occ'][k] = True
    # nucleation: glass sites
    TG = Tc[NR - 1, SG['cell']]
    S_glass = C[NR - 1, SG['cell']] / np.maximum(c_sat(np.minimum(TG, 99.0)), 1e-3) - 1
    newg = (~SG['occ']) & (t >= SG['wait']) & (S_glass > SG['s_gas'])
    if newg.any():
        k = np.nonzero(newg)[0]; r0 = 3e-5
        Tl = TG[k]; pb = PATM + 958 * g * (H - SG['z'][k]) + 2 * sigma_w(Tl) / r0
        ntot = pb * 4 / 3 * np.pi * r0 ** 3 / (RG * (Tl + 273.15))
        fv = np.minimum(p_v(Tl) / pb, 0.9)
        add_bubbles(x=(R - r0) * np.cos(SG['th'][k]), y=(R - r0) * np.sin(SG['th'][k]), z=SG['z'][k],
                    ng=ntot * (1 - fv), nv=ntot * fv, r=np.full(len(k), r0), kind=np.full(len(k), 2), site=k)
        np.add.at(qC0, (np.full(len(k), NR - 1), SG['cell'][k]), -ntot * (1 - fv) / dt / VOL[NR - 1, SG['cell'][k]])
        SG['occ'][k] = True

    nb = len(B['x'])
    if nb:
        ci, cj, rr = cell_of(B['x'], B['y'], B['z'])
        Tl = Tc[ci, cj].copy(); Cl = C[ci, cj]
        kind = B['kind']; att_p = kind == 1; att_g = kind == 2; free = kind == 0
        Tl[att_p] = Tf_plate[ci[att_p]]
        pb0 = PATM + 958 * g * np.maximum(H - B['z'], 0)
        r_b = np.maximum(B['r'], 1e-6); d = 2 * r_b
        pb = pb0 + 2 * sigma_w(Tl) / r_b
        rl = rho_w(np.minimum(Tl, 100)); nul = mu_w(np.minimum(Tl, 100)) / rl
        us = slip_velocity(d, np.minimum(Tl, 100))
        ul_r, ul_z = interp_vel(u, w, rr, B['z'])
        urel = np.where(free, us, np.hypot(ul_r, ul_z) + 0.01)
        Re = urel * d / nul
        # air exchange
        Sc = nul / D_air(np.minimum(Tl, 100))
        km = D_air(np.minimum(Tl, 100)) / d * (2 + 0.6 * np.sqrt(Re) * Sc ** (1 / 3))
        A = 4 * np.pi * r_b ** 2
        frac_g = B['ng'] / np.maximum(B['ng'] + B['nv'], 1e-30)
        pg = pb * frac_g
        dng = km * A * (Cl - henry(np.minimum(Tl, 100)) * pg) * dt
        dng = np.maximum(dng, -B['ng'])
        # vapour exchange: heat transfer toward the liquid temperature (equilibrium cap)
        Prl = mu_w(np.minimum(Tl, 100)) * CP / k_w(np.minimum(Tl, 100))
        hb = k_w(np.minimum(Tl, 100)) / d * (2 + 0.6 * np.sqrt(Re) * Prl ** (1 / 3))
        pvb = pb * (1 - frac_g)
        Tb = T_sat(np.maximum(pvb, 50.0))
        Lh = L_v(np.minimum(Tl, 100))
        dnv = hb * A * (Tl - Tb) / (Lh * MW) * dt
        pvl = p_v(Tl)
        nv_eq = np.where(pvl < 0.999 * pb, B['ng'] * pvl / np.maximum(pb - pvl, 1.0), np.inf)
        nv_new = B['nv'] + dnv
        nv_new = np.where((dnv > 0) & (nv_new > nv_eq), np.maximum(nv_eq, B['nv']), nv_new)
        nv_new = np.where((dnv < 0) & (nv_new < nv_eq), np.minimum(nv_eq, B['nv']), nv_new)
        nv_new = np.maximum(nv_new, 0.0)
        # plate sites that boil: vapour fed by the evaporation part of the boiling flux, shared per ring
        bsite = att_p & (dTw[ci] > SP['th_boil'][np.where(att_p, B['site'], 0)])
        B['boil'] = np.where(att_p, bsite, B['boil'])
        cnt = np.bincount(ci[bsite], minlength=NR)
        feed = q_ev * A_RING / L_v(TSAT_B) / MW                                  # mol/s per ring
        nv_new = np.where(bsite, B['nv'] + feed[ci] / np.maximum(cnt[ci], 1) * dt, nv_new)
        q_unused = np.where(cnt == 0, q_ev, 0.0)
        # latent heat and air to/from the liquid (plate-fed vapour is already taken out of the plate flux)
        dnv_l = np.where(bsite, 0.0, nv_new - B['nv'])
        np.add.at(qT, (ci, cj), -dnv_l * MW * Lh / dt / VOL[ci, cj])
        np.add.at(qC, (ci, cj), -dng / dt / VOL[ci, cj])
        B['ng'] = B['ng'] + dng; B['nv'] = nv_new
        B['r'] = radius_from(B['ng'], B['nv'], np.minimum(Tl, TSAT_B + 2), pb0)
        B['age'] = B['age'] + dt
        # departure of attached bubbles
        d = 2 * B['r']
        rl = rho_w(np.minimum(Tl, 100))
        f_cole = np.sqrt(4 * g * (rl - 0.6) / (3 * rl * np.maximum(d, 1e-5)))
        d_boil = np.clip(2.4e-3 * np.exp(-sub[ci] / 45.0), 0.5e-3, 2.4e-3)
        dep_p = att_p & np.where(B['boil'], (d >= d_boil) & (B['age'] >= 1 / f_cole), d >= 1.2e-3)
        dep_g = att_g & (d >= 0.9e-3)
        dead = (B['r'] < 1.0e-5) & ~free
        for m, S, wt in ((dep_p | (dead & att_p), SP, None), (dep_g | (dead & att_g), SG, None)):
            s = B['site'][m]
            if len(s):
                S['occ'][s] = False
                bo = B['boil'][m] if S is SP else np.zeros(len(s), bool)
                S['wait'][s] = t + np.where(bo, 0.3 / np.maximum(f_cole[m], 5.0), rng.uniform(0.3, 2.0, len(s)))
        B['kind'] = np.where(dep_p | dep_g, 0, B['kind'])
        B['age'] = np.where(dep_p | dep_g, 0.0, B['age'])
        free = B['kind'] == 0
        # attached positions sit on their wall
        B['z'] = np.where(att_p & ~dep_p, B['r'] * 0.9, B['z'])
        gi = att_g & ~dep_g
        B['x'] = np.where(gi, (R - 0.9 * B['r']) * np.cos(SG['th'][np.where(gi, B['site'], 0)]), B['x'])
        B['y'] = np.where(gi, (R - 0.9 * B['r']) * np.sin(SG['th'][np.where(gi, B['site'], 0)]), B['y'])
        # free bubbles: rise, wobble, push the liquid
        if free.any():
            us = slip_velocity(2 * B['r'], np.minimum(Tl, 100))
            sw = np.where(B['r'] > 0.75e-3, 0.25, 0.05) * us
            a = np.exp(-dt / 0.05)
            B['wx'] = np.where(free, a * B['wx'] + np.sqrt(1 - a * a) * sw * rng.standard_normal(nb), 0.0)
            B['wy'] = np.where(free, a * B['wy'] + np.sqrt(1 - a * a) * sw * rng.standard_normal(nb), 0.0)
            cth = B['x'] / np.maximum(rr, 1e-9); sth = B['y'] / np.maximum(rr, 1e-9)
            vx = ul_r * cth + B['wx']; vy = ul_r * sth + B['wy']; vz = ul_z + us
            B['x'] = np.where(free, B['x'] + vx * dt, B['x'])
            B['y'] = np.where(free, B['y'] + vy * dt, B['y'])
            B['z'] = np.where(free, np.maximum(B['z'] + vz * dt, B['r']), B['z'])
            rr2 = np.hypot(B['x'], B['y']); lim = R - B['r']
            sc = np.where(free & (rr2 > lim), lim / np.maximum(rr2, 1e-9), 1.0)
            B['x'] *= sc; B['y'] *= sc
            Vb = 4 / 3 * np.pi * B['r'] ** 3
            ci, cj, rr = cell_of(B['x'], B['y'], B['z'])
            np.add.at(fz, (ci[free], cj[free]), g * Vb[free] / VOL[ci[free], cj[free]])
        # burst at the free surface, vanish when (almost) empty
        burst = free & (B['z'] + 0.5 * B['r'] >= z_top)
        vent_v = np.sum(B['nv'][burst]) * MW / dt; vent_g = np.sum(B['ng'][burst]) / dt
        E_vent += np.sum(B['nv'][burst]) * MW * L_v(100.0)
        # vanished bubbles, and free micro-bubbles (< 80 um, sub-pixel, rise < 1.5 cm/s) whose air we return to the
        # dissolved field (keeps the bubble count tractable; mass and energy conserved)
        gone = ((B['r'] < 1.0e-5) | ((B['kind'] == 0) & (B['r'] < 8.0e-5))) & ~burst
        if gone.any():
            ci2, cj2, _ = cell_of(B['x'][gone], B['y'][gone], B['z'][gone])
            np.add.at(qC, (ci2, cj2), B['ng'][gone] / dt / VOL[ci2, cj2])
            np.add.at(qT, (ci2, cj2), B['nv'][gone] * MW * L_v(100.0) / dt / VOL[ci2, cj2])
        keep(~(burst | gone))
        # coalescence of touching free bubbles
        if step % 4 == 0:
            fi = np.nonzero(B['kind'] == 0)[0]
            if len(fi) > 1:
                X = np.c_[B['x'][fi], B['y'][fi], B['z'][fi]]
                tr = cKDTree(X)
                pr = tr.query_pairs(2 * B['r'][fi].max() if len(fi) else 0, output_type='ndarray')
                if len(pr):
                    a_, b_ = fi[pr[:, 0]], fi[pr[:, 1]]
                    dist = np.linalg.norm(X[pr[:, 0]] - X[pr[:, 1]], axis=1)
                    touch = dist < 0.9 * (B['r'][a_] + B['r'][b_])
                    used = np.zeros(len(B['x']), bool); kill = []
                    for p, q in zip(a_[touch], b_[touch]):
                        if used[p] or used[q]:
                            continue
                        used[p] = used[q] = True
                        v1, v2 = B['r'][p] ** 3, B['r'][q] ** 3
                        for k_ in ('x', 'y', 'z'):
                            B[k_][p] = (B[k_][p] * v1 + B[k_][q] * v2) / (v1 + v2)
                        B['ng'][p] += B['ng'][q]; B['nv'][p] += B['nv'][q]
                        B['r'][p] = np.cbrt(v1 + v2)
                        kill.append(q)
                    if kill:
                        m = np.ones(len(B['x']), bool); m[kill] = False; keep(m)
    else:
        vent_v = vent_g = 0.0; q_unused = q_ev
    # ---------------- wall / surface sources ----------------
    qT[:, 0] += (q_liq + q_unused) * A_RING / VOL[:, 0]
    qT[-1, :] += -8.0 * (Tc[-1, :] - T_AMB) * A_SIDE / VOL[-1, :]
    Ts = Tc[:, -1]
    rho_vs = p_v(Ts) * MW / (RG * (Ts + 273.15)); rho_va = 0.5 * p_v(T_AMB) * MW / (RG * 293.15)
    m_ev = 0.006 * np.maximum(rho_vs - rho_va, 0.0)
    q_top = 8.0 * (Ts - T_AMB) + m_ev * L_v(Ts)
    qT[:, -1] -= q_top * A_RING / VOL[:, -1]
    qC[:, -1] -= 3e-5 * (C[:, -1] - c_sat(np.minimum(Ts, 100))) * A_RING / VOL[:, -1]
    E_loss += dt * (np.sum(q_top * A_RING) + np.sum(8.0 * (Tc[-1, :] - T_AMB) * A_SIDE))
    src = dict(qT=qT, qC=qC + qC0, fz=fz)
    # ---------------- liquid: SSP-RK3 with projection ----------------
    def stage(u_, w_, T_, C_):
        dU, dW, dT, dC, _ = rhs(u_, w_, T_, C_, src)
        return dU, dW, dT, dC
    k1 = stage(u, w, T, C)
    u1, w1 = project(u + dt * k1[0], w + dt * k1[1], dt); T1_ = T + dt * k1[2]; C1 = C + dt * k1[3]
    k2 = stage(u1, w1, T1_, C1)
    u2, w2 = project(0.75 * u + 0.25 * (u1 + dt * k2[0]), 0.75 * w + 0.25 * (w1 + dt * k2[1]), dt / 4)
    T2 = 0.75 * T + 0.25 * (T1_ + dt * k2[2]); C2 = 0.75 * C + 0.25 * (C1 + dt * k2[3])
    k3 = stage(u2, w2, T2, C2)
    u, w = project(u / 3 + 2 / 3 * (u2 + dt * k3[0]), w / 3 + 2 / 3 * (w2 + dt * k3[1]), 2 * dt / 3)
    T = T / 3 + 2 / 3 * (T2 + dt * k3[2]); C = np.maximum(C / 3 + 2 / 3 * (C2 + dt * k3[3]), 0.0)
    t += dt; step += 1
    # ---------------- steam thermostat ----------------
    a = dt / 1.0
    vent_v_s += a * (vent_v - vent_v_s); vent_g_s += a * (vent_g - vent_g_s)
    if heater_on:
        t_strong = t_strong + dt if vent_v_s > 0.35e-3 else 0.0
        t_hot = t_hot + dt if np.sum(T * VOL) / VOL.sum() > 99.3 else 0.0
        if t_strong > 8.0 or t_hot > 20.0:
            heater_on, t_off = False, t
    if t_off is not None and t > t_off + 15.0:
        T_END = t
    # ---------------- output ----------------
    if t >= next_snap:
        snapshot(); next_snap += 1 / FPS
    if t >= next_log:
        Tm = float(np.sum(T * VOL) / VOL.sum())
        H_w = float(np.sum(rho_w(T) * VOL * CP * (T - T_AMB)))
        H_p = float(np.sum(C_PLATE * A_RING * (Tp - T_AMB)))
        kinds = np.bincount(B['kind'], minlength=3) if len(B['kind']) else np.zeros(3, int)
        row = dict(t=round(t, 3), dt=dt, Tm=round(Tm, 3), Tmax=round(float(T.max()), 2), Tp_max=round(float(Tp.max()), 2),
                   Tp_mean=round(float(np.sum(Tp * A_RING) / A_PLATE), 2), q_nb_frac=round(float(np.sum(q_nb * A_RING) / max(np.sum((q_nb + q_nc) * A_RING), 1)), 3),
                   n_free=int(kinds[0]), n_plate=int(kinds[1]), n_glass=int(kinds[2]),
                   C_rel=round(float(np.sum(C * VOL) / VOL.sum() / c_sat(20.0)), 4), umax=round(float(umax), 4),
                   vent_steam_gps=round(vent_v_s * 1e3, 4), vent_air_umolps=round(vent_g_s * 1e6, 3), heater=heater_on,
                   E_in=round(E_in), E_water=round(H_w), E_plate=round(H_p), E_vent=round(E_vent), E_loss=round(E_loss),
                   wall_s=round(time.time() - wall0, 1))
        log.append(row); next_log += 1.0
        print(json.dumps(row), flush=True)
        if not np.isfinite(T).all() or not np.isfinite(u).all():
            print('NaN, stopping', flush=True); break
        if int(t) % 10 == 0:
            json.dump(log, open(f'{OUT}/log.json', 'w'))
json.dump(log, open(f'{OUT}/log.json', 'w'))
print('done', t, 'snapshots', isnap, 'wall', time.time() - wall0)
