"""Candle geometry, boundary conditions and the OpenFOAM reference (data/steady_flame_t0.5.npz).

Geometry as gen.py: wick r < 0.75 mm, z < 6 mm (solid, fuel vapour leaves its side and top at 0.1 g/min, 600 K);
wax pool 0.75 < r < 10 mm at z = 0 (no-slip wall, 340 K); open bottom 10 < r < 40 mm; outer radius 40 mm,
height 120 mm. Outer cylinder and open bottom: OpenFOAM base entrainment (m = 0) + uniform horizontal draft V
(u_r = V cos th, u_th = -V sin th); top: outflow (p = 0).
"""
import os
import numpy as np
from scipy.interpolate import griddata, LinearNDInterpolator, NearestNDInterpolator
from core import Grid, BC, stretched, OUTER, TOP, POOL, OPEN, WICK
import mech

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(os.path.dirname(HERE), 'data', 'steady_flame_t0.5.npz')
R_W, Z_W, R_C, R, Z = 0.75e-3, 6e-3, 10e-3, 0.04, 0.12
MDOT = 0.10 / 1000 / 60               # kg/s
T_WICK, T_POOL, T_AIR = 600.0, 340.0, 300.0
Y_AIR5 = np.array([0, 0.23, 0, 0, 0])
Y_FUEL5 = np.array([1.0, 0, 0, 0, 0])
SP5 = mech.SPECIES[:5]
ZST = 0.0621


def refined(L, hf, x_f, h, x_u, growth, g_loc=1.15):
    """faces on [0, L]: spacing hf up to x_f (a multiple of hf), geometric growth (<= g_loc) to h, uniform h up to
    about x_u, then growth `growth` to L (as stretched)."""
    f = list(np.arange(int(round(x_f / hf)) + 1) * hf)
    d = hf
    while d * g_loc < h:
        d *= g_loc
        f.append(f[-1] + d)
    n = max(1, int(round((x_u - f[-1]) / h)))
    f += list(f[-1] + h * np.arange(1, n + 1))
    tail = stretched(L - f[-1], h, 0.0, growth)
    return np.r_[f, f[-1] + tail[1:]]


def candle_grid(M, hr=0.375e-3, hz=0.5e-3, r_u=6e-3, z_u=60e-3, gr=1.12, gz=1.08, hf=None, rf_fine=1.5e-3):
    """hf: optional fine spacing near the wick and pool (r < rf_fine, z < 9 mm), growing smoothly to hr / hz."""
    if hf:
        rf = refined(R, hf, rf_fine, hr, r_u, gr)
        zf = refined(Z, hf, 9e-3, hz, z_u, gz)
        return Grid(rf, zf, M, wick=(R_W, Z_W), pool_r=R_C)
    return Grid(stretched(R, hr, r_u, gr), stretched(Z, hz, z_u, gz), M, wick=(R_W, Z_W), pool_r=R_C)


class Reference:
    """OpenFOAM steady flame interpolated onto (z, r) points."""

    def __init__(self):
        d = np.load(NPZ)
        self.d = d
        self.pts = np.c_[d['cell_r'], d['cell_z']].astype(float)
        self._lin = {}

    def __call__(self, name, r, z):
        if name not in self._lin:
            v = self.d['cell_' + name].astype(float)
            self._lin[name] = (LinearNDInterpolator(self.pts, v), NearestNDInterpolator(self.pts, v))
        lin, near = self._lin[name]
        out = lin(r, z)
        bad = np.isnan(out)
        out[bad] = near(r[bad], z[bad])
        return out

    def on_grid(self, g, name):
        Rg, Zg = np.meshgrid(g.rc, g.zc)
        return self(name, Rg.ravel(), Zg.ravel()).reshape(g.Nz, g.Nr)

    def axis(self, name, rmax=0.15e-3):
        d = self.d
        sel = d['cell_r'] < rmax
        o = np.argsort(d['cell_z'][sel])
        return d['cell_z'][sel][o].astype(float), d['cell_' + name][sel][o].astype(float)


class CandleBase:
    """Boundary conditions shared by stages 2-4. Subclasses define scalars, scalar_bc, props, source, fix."""

    def __init__(self, g, V=0.0, t_ramp=0.05, t_on=0.0):
        self.g = g
        self.V0, self.t_ramp, self.t_on = V, t_ramp, t_on
        self.legacy_flux = bool(int(os.environ.get('LEGACY_FLUX', 0)))   # upwind-cell face density (old scheme)
        self.ref = Reference()
        self.rho_inf = mech.density(np.array(T_AIR), mech.full_Y(Y_AIR5))
        th = g.theta[:, None, None]
        self.cos, self.sin = np.cos(th), np.sin(th)
        # wick injection: uniform mass flux over side + top (flowRateInletVelocity)
        area = 2 * np.pi * R_W * Z_W + np.pi * R_W ** 2
        self.mflux_w = MDOT / area
        self.rho_w = float(mech.density(np.array(T_WICK), mech.full_Y(Y_FUEL5)))
        # base entrainment from the reference on the outer cylinder and the open bottom
        self.ur_out = self.ref('Ur', np.full(g.Nz, R - 1.3e-3), g.zc)
        self.uz_out = self.ref('Uz', np.full(g.Nz, R - 1.3e-3), g.zc)
        self.uz_bot = self.ref('Uz', g.rc, np.full(g.Nr, 1e-4))
        self.ur_bot = self.ref('Ur', g.rc, np.full(g.Nr, 1e-4))
        self.uz_bot[g.rc < R_C] = 0
        self.ur_bot[g.rc < R_C] = 0

    def V(self, t):
        if self.V0 == 0:
            return 0.0
        return self.V0 * np.clip((t - self.t_on) / self.t_ramp, 0, 1)

    def draft(self, t):
        """horizontal draft vector (Vx, Vy) in m/s. Default: steady V along +x (ramped). GUST=<rms m/s> env: a gentle
        wandering 'restaurant table' draft — each component a sum of 4 incommensurate sinusoids (0.6-3.4 Hz),
        ramped in over 0.3 s from GUST_T0 (default 0.3 s); a uniform draft of any direction is pure m = 1."""
        rms = float(os.environ.get('GUST', 0))
        if rms <= 0:
            return self.V(t), 0.0
        if not hasattr(self, '_gust'):
            rng = np.random.default_rng(int(os.environ.get('GUST_SEED', 3)))
            f = np.array([0.6, 1.1, 1.9, 3.4])
            a = 1 / np.sqrt(f); a *= rms * np.sqrt(2) / np.sqrt((a ** 2).sum() * 2)   # rms of |V| ~ rms
            self._gust = (f, a, rng.uniform(0, 2 * np.pi, (2, 4)))
        f, a, ph = self._gust
        ramp = np.clip((t - float(os.environ.get('GUST_T0', 0.3))) / 0.3, 0, 1)
        vx = ramp * (a * np.sin(2 * np.pi * f * t + ph[0])).sum()
        vy = ramp * (a * np.sin(2 * np.pi * f * t + ph[1])).sum()
        return float(vx), float(vy)

    def vel_bc(self, t):
        g = self.g
        vx, vy = self.draft(t)
        # uniform horizontal draft in cylindrical components: u_r = Vx cos + Vy sin, u_th = -Vx sin + Vy cos
        dr_, dt_ = vx * self.cos + vy * self.sin, -vx * self.sin + vy * self.cos
        shr, shz = (g.Nt, g.Nz, g.Nr + 1), (g.Nt, g.Nz + 1, g.Nr)
        vr = [np.zeros(shr) for _ in range(3)]
        vz = [np.zeros(shz) for _ in range(3)]
        # outer cylinder
        vr[0][..., -1] = self.ur_out[None] + dr_[..., 0]
        vr[1][..., -1] = dt_[..., 0]
        vr[2][..., -1] = self.uz_out[None]
        # bottom: open part carries entrainment + draft, pool no-slip
        op = g.lz[0] == OPEN
        vz[0][:, 0, op] = self.ur_bot[op] + dr_[:, 0]
        vz[1][:, 0, op] = dt_[:, 0]
        vz[2][:, 0, op] = self.uz_bot[op]
        # wick: normal injection of fuel vapour at 600 K: volume flux mdot''/rho_w. The solver's face mass flux
        # rho_EOS(face state) U uses the Dirichlet fuel state on this face, so it carries exactly mdot''.
        legacy = getattr(self, 'legacy_flux', False) and getattr(self, 'rho_now', None) is not None
        for fam, arr in (('r', vr[0]), ('z', vz[2])):
            bf = g.bfaces[fam][WICK]
            rb = self.rho_now[:, bf['jc'], bf['ic']] if legacy else self.rho_w
            arr[:, bf['j'], bf['i']] = self.mflux_w / rb
        D = {OUTER, OPEN, POOL, WICK}
        return BC(D, -1.0, vr[0], vz[0]), BC(D, -1.0, vr[1], vz[1]), BC(D, 1.0, vr[2], vz[2])

    def bnd_mflux(self, t, rho):
        g = self.g
        ur, ut, uz = self.vel_bc(t)
        Fr = np.zeros((g.Nt, g.Nz, g.Nr + 1)); Fz = np.zeros((g.Nt, g.Nz + 1, g.Nr))
        u = ur.vr[..., -1]
        Fr[..., -1] = u * np.where(u < 0, self.rho_inf, rho[..., -1])
        w = uz.vz[:, 0]
        Fz[:, 0] = w * np.where(w > 0, self.rho_inf, rho[:, 0])
        Fz[:, 0, g.lz[0] == POOL] = 0
        bf = g.bfaces['r'][WICK]
        Fr[:, bf['j'], bf['i']] = self.mflux_w
        bf = g.bfaces['z'][WICK]
        Fz[:, bf['j'], bf['i']] = self.mflux_w
        return Fr, Fz

    def scalar_face_arrays(self, v_inflow, v_wick, v_pool=None):
        """physical face arrays: inflow value on OUTER/OPEN/TOP, wick value on WICK, pool value on POOL."""
        g = self.g
        vr = np.full((g.Nt, g.Nz, g.Nr + 1), float(v_inflow))
        vz = np.full((g.Nt, g.Nz + 1, g.Nr), float(v_inflow))
        bf = g.bfaces['r'][WICK]; vr[:, bf['j'], bf['i']] = v_wick
        bf = g.bfaces['z'][WICK]; vz[:, bf['j'], bf['i']] = v_wick
        if v_pool is not None:
            vz[:, 0, g.lz[0] == POOL] = v_pool
        return vr, vz

    def initial_velocity(self, with_draft=True):
        g = self.g
        u = np.zeros((3, g.Nt, g.Nz, g.Nr))
        u[0] = self.ref.on_grid(g, 'Ur')[None]
        u[2] = self.ref.on_grid(g, 'Uz')[None]
        if with_draft and self.V(self.t_on + 1e9) and self.t_on <= 0:
            u[0] += self.V0 * self.cos
            u[1] += -self.V0 * self.sin
        u[:, :, g.solid] = 0
        return u


def axis_profile(g, f):
    """value on the first cell row (r = r_c[0]), averaged over theta (= m = 0 at the axis)."""
    return f[..., 0].mean(0)


def tip_height(g, Zf, Zst=ZST):
    """highest z where the stoichiometric surface meets the axis region: first z above the wick where the
    theta-averaged axis value drops below Zst."""
    za = axis_profile(g, Zf)
    above = (g.zc > Z_W) & (za < Zst)
    if not above.any():
        return np.nan
    j = np.argmax(above)
    z0, z1, f0, f1 = g.zc[j - 1], g.zc[j], za[j - 1], za[j]
    return z0 + (f0 - Zst) / (f0 - f1) * (z1 - z0)
