"""Stage 3: flame sheet / mixture fraction.

Transport Z (rho DZ/Dt = div(rho D grad Z), Le = 1, rho D = mu/Pr, Pr = 0.7; Z = 1 at the wick, 0 in the air).
State relations T(Z), Y_i(Z) are conditional means of the OpenFOAM solution (Z from C and H element mass
fractions, normalised by the fuel's, Zst = 0.0621); rho = p0 W(Z)/(R T(Z)), mu from Sutherland with T(Z), Y(Z).
Checks: M = 0 tip (Zst on the axis) = 44 mm +- 10 %; the lean converges for M = 2/4/8.
usage: python stage3_flamesheet.py M V t_end [hr_mm hz_mm] [restart.npz] [Sc]
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mech
from core import BC, OUTER, OPEN, TOP, WICK
from candle import CandleBase, candle_grid, Reference, SP5, ZST, Z_W, tip_height
from stage2_plume import run

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
_WC = 12.011 * 12 / 170.34
_WH = 1.008 * 26 / 170.34


def mixture_fraction(Y):
    """Z = (Z_C + Z_H) / (Z_C + Z_H)_fuel from species mass fractions dict/array in SPECIES order."""
    ZC = Y[0] * _WC + Y[2] * 12.011 / 28.01 + Y[3] * 12.011 / 44.01
    ZH = Y[0] * _WH + Y[4] * 2 * 1.008 / 18.015
    return (ZC + ZH) / (_WC + _WH)


SM_Z0, SM_SLOPE, SM_MAX = 0.08, 0.25, float(os.environ.get('TAB_SMOOTH', 0.04))


def state_table(nbin=400):
    """conditional means <T|Z>, <Y_i|Z> (volume weighted, wedge cells) on a Z grid clustered near Zst."""
    ref = Reference()
    d = ref.d
    Y = np.array([d['cell_Y_' + s].astype(float) for s in SP5])
    Z = np.clip(mixture_fraction(Y), 0, 1)
    T = d['cell_T'].astype(float)
    r = d['cell_r'].astype(float)
    # weight ~ cell volume: r * (cell area); cell areas from local spacing estimated by neighbours is overkill,
    # use r * dA with dA from a Voronoi-free estimate: product of the distinct grid spacings at the cell
    ru, zu = np.unique(d['cell_r']), np.unique(d['cell_z'])
    dr_ = np.gradient(ru.astype(float)); dz_ = np.gradient(zu.astype(float))
    w = r * dr_[np.searchsorted(ru, d['cell_r'])] * dz_[np.searchsorted(zu, d['cell_z'])]
    edges = np.unique(np.r_[np.linspace(0, 0.2, nbin // 2), np.geomspace(0.2, 1, nbin // 2)])
    idx = np.clip(np.digitize(Z, edges) - 1, 0, len(edges) - 2)
    cnt = np.bincount(idx, w, len(edges) - 1)
    ok = cnt > 0
    zc = (np.bincount(idx, w * Z, len(edges) - 1)[ok] / cnt[ok])
    tab = {'Z': zc, 'T': np.bincount(idx, w * T, len(edges) - 1)[ok] / cnt[ok]}
    for k, s in enumerate(SP5):
        tab[s] = np.bincount(idx, w * Y[k], len(edges) - 1)[ok] / cnt[ok]
    # Rich side: few, very differently placed cells per bin (hot fuel-rich core above the wick vs cold vapour next
    # to it) make the bin means noisy (T jumps of +-150 K between neighbouring bins at Z > 0.3), so rho(Z) was
    # non-monotone and d ln rho/dZ (fine table) ranged from -314 to +159 at Z > 0.15. That sawtooth made S = div u
    # a noisy function of Z (|S| 3-5x larger in the draft runs) and broke the face-density consistency of the mass
    # fluxes (which needs 1/rho locally linear in Z): spurious fuel source 0.4-1.1e-6 kg/s at M = 4 with the draft
    # vs 1-5e-8 smoothed. Replace the bin means by a volume-weighted local-linear regression on the cell data with
    # a kernel width growing from 0 (Z < SM_Z0, flame zone untouched) to SM_MAX (env TAB_SMOOTH, 0 = raw means).
    sig = np.clip(SM_SLOPE * (zc - SM_Z0), 0, SM_MAX) if SM_MAX > 0 else np.zeros_like(zc)
    for k in np.nonzero(sig > 2e-3)[0]:
        dz = Z - zc[k]
        sel = np.abs(dz) < 3 * sig[k]
        kw = w[sel] * np.exp(-0.5 * (dz[sel] / sig[k]) ** 2)
        x = dz[sel]
        S0, S1, S2 = kw.sum(), (kw * x).sum(), (kw * x * x).sum()
        den = S0 * S2 - S1 ** 2
        ll = lambda q: ((S2 * (kw * q).sum() - S1 * (kw * x * q).sum()) / den)   # local-linear value at x = 0
        tab['T'][k] = ll(T[sel])
        for kk, s in enumerate(SP5):
            tab[s][k] = ll(Y[kk][sel])
    # end points: pure air / wick fuel vapour
    tab['Z'] = np.r_[0.0, tab['Z'], 1.0]
    tab['T'] = np.r_[300.0, tab['T'], 600.0]
    for k, s in enumerate(SP5):
        tab[s] = np.r_[[0, 0.23, 0, 0, 0][k], tab[s], [1, 0, 0, 0, 0][k]]
    # monotone Z for interp
    o = np.argsort(tab['Z'])
    for key in tab:
        tab[key] = tab[key][o]
    # tabulate rho and mu on a fine uniform-in-sqrt(Z) grid for fast lookup
    zf = np.r_[np.linspace(0, 0.2, 2001), np.linspace(0.2, 1, 801)[1:]]
    Tf = np.interp(zf, tab['Z'], tab['T'])
    Yf = mech.full_Y(np.array([np.interp(zf, tab['Z'], tab[s]) for s in SP5]))
    rho = mech.density(Tf, Yf)
    mu = mech.transport(Tf, Yf)[0]
    return dict(Z=zf, T=Tf, Y=Yf, rho=rho, mu=mu, dlnrho=np.gradient(rho, zf) / rho, raw=tab)


class FlameSheet(CandleBase):
    def __init__(self, g, V=0.0, Sc=0.7, **kw):
        super().__init__(g, V, **kw)
        self.scalars = ['Z']
        self.bounds = {'Z': (0.0, 1.0)}
        self.Sc = Sc
        self.tab = state_table()
        self.scalar_bc = {'Z': BC({WICK}, 1.0, *self.scalar_face_arrays(0.0, 1.0), solid_fill=1.0,
                                  inflow_labels={OUTER, OPEN, TOP})}

    def lookup(self, key, Z):
        return np.interp(np.clip(Z, 0, 1), self.tab['Z'], self.tab[key])

    def props(self, phi):
        Z = phi['Z']
        rho = self.lookup('rho', Z)
        mu = self.lookup('mu', Z)
        return rho, mu, mu / self.Sc

    def density(self, phi):
        return self.lookup('rho', phi['Z'])

    def dlnrho(self, phi):
        # consistent with the clipped density: rho(Z) is constant outside [0, 1], so d ln rho/dZ = 0 there. The
        # clipped lookup returned d ln rho/dZ(0) = -150 for undershoots Z < 0, i.e. a spurious volume source
        # S = 150 R_Z (R_Z: diffusion filling the undershoot) where the density does not change.
        Z = phi['Z']
        return {'Z': np.where((Z >= 0) & (Z <= 1), self.lookup('dlnrho', Z), 0.0)}

    def initial_scalars(self):
        g, ref = self.g, self.ref
        Y = np.array([ref.on_grid(g, 'Y_' + s) for s in SP5])
        Z = np.clip(mixture_fraction(Y), 0, 1)
        Z[g.solid] = 1.0
        return {'Z': np.broadcast_to(Z, (g.Nt, g.Nz, g.Nr)).copy()}


def flame_geometry(g, Z, Zst=ZST):
    """Tip = highest point of the region Z >= Zst (above the wick); centroid of that region; lean angles of tip
    and centroid relative to the wick top (r=0, z=6 mm), measured from the vertical, positive downstream (+x)."""
    th = g.theta[:, None, None]
    X = np.broadcast_to(g.rc[None, None, :] * np.cos(th), Z.shape)
    Yc = g.rc[None, None, :] * np.sin(th)
    Zc = np.broadcast_to(g.zc[None, :, None], Z.shape)
    inside = (Z >= Zst) & np.broadcast_to(g.fluid, Z.shape)
    if not inside.any():
        return {}
    vol = np.broadcast_to(g.vol, Z.shape)
    zt = Zc[inside].max()
    k = np.argmax(np.where(inside, Zc, -1).ravel())
    xt = X.ravel()[k]
    w = vol * inside
    xc, zc = (w * X).sum() / w.sum(), (w * Zc).sum() / w.sum()
    ang = lambda x, z: float(np.degrees(np.arctan2(x, z - Z_W)))
    return dict(tip_z_mm=float(zt * 1e3), tip_x_mm=float(xt * 1e3), tip_angle=ang(xt, zt),
                cen_x_mm=float(xc * 1e3), cen_z_mm=float(zc * 1e3), cen_angle=ang(xc, zc),
                vol_cm3=float(w.sum() * 2 * np.pi / g.Nt * 1e6))


if __name__ == '__main__':
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    V = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    t_end = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    hr = float(sys.argv[4]) * 1e-3 if len(sys.argv) > 4 else 0.375e-3
    hz = float(sys.argv[5]) * 1e-3 if len(sys.argv) > 5 else 0.5e-3
    restart = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] != '-' else None
    Sc = float(sys.argv[7]) if len(sys.argv) > 7 else 0.7
    tag = f'stage3_M{M}_V{V:g}_h{hr*1e3:g}_Sc{Sc:g}'

    class Case(FlameSheet):
        def __init__(self, g, V, **kw):
            super().__init__(g, V, Sc=Sc, **kw)
            self.budget = ['Z']
            self.budget_planes = (0.008, 0.012, 0.02, 0.03, 0.045)

    def extra(sol, case):
        Z = sol.g.phys(sol.phis['Z'])
        info = flame_geometry(sol.g, Z)
        info['tip_axis_mm'] = float(tip_height(sol.g, Z) * 1e3)
        info = {k: round(v, 3) for k, v in info.items()}
        info.update({k: float('%.4g' % v) for k, v in sol.budget_report().items()})
        return info
    cfl = float(os.environ.get("CFL", 0.4))
    tag += os.environ.get("TAG", "")
    sol, case, hist = run(Case, M, V, t_end, hr, hz, restart, tag, extra=extra, every=0.01, cfl=cfl)
    res = dict(M=M, V=V, Sc=Sc, t=sol.t, steps=sol.n, final=hist[-1], timing=sol.timing)
    print(json.dumps(res))
    json.dump(dict(res, hist=hist), open(os.path.join(OUT, tag + '.json'), 'w'), indent=1)
