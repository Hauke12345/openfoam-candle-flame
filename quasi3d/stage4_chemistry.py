"""Stage 4: finite-rate chemistry (main goal).

Transport hs and Y_C12H26, Y_O2, Y_CO, Y_CO2, Y_H2O (N2 = 1 - sum) with the gen.py reactions (2-step WD + CO2
dissociation), NASA7 thermo and Sutherland/Eucken transport (rho D = mu for species, kappa/cp for hs, as OpenFOAM);
optically thin grey radiation; chemistry Strang-split and integrated per point (chem.py, implicit Rosenbrock).
Checks: M = 0 matches OpenFOAM (tip height and peak T within 5 %); the lean agrees with stage 3.
usage: python stage4_chemistry.py M V t_end [hr_mm hz_mm] [restart.npz|-]
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mech, chem
from candle import SP5, ZST, Z_W, tip_height
from stage2_plume import Thermo, run
from stage3_flamesheet import mixture_fraction, flame_geometry

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')


class Chemistry(Thermo):
    strang = True

    def source(self, phi, rho, dt, t):
        T, Y = self.state(phi)
        shp = T.shape
        Y5 = Y[:5].reshape(5, -1)
        Y5n, Tn, q, nsub = chem.react(Y5, T.ravel(), dt)
        Yn = mech.full_Y(Y5n)
        hs = mech.hs_mix(Tn, Yn).reshape(shp)
        # radiation (explicit, lagged composition): optically thin grey CO2/H2O
        hs = hs + dt * mech.q_rad_thin(Tn.reshape(shp), Yn.reshape((6,) + shp)) / rho
        phi['hs'] = hs
        for k, s in enumerate(SP5):
            phi[s] = Y5n[k].reshape(shp)
        self.qdot = q.reshape(shp)
        self.nsub = int(nsub.max())
        return phi


def diagnostics(sol, case):
    g = sol.g
    f = sol.fields()
    T, Y = case.state({s: f[s] for s in case.scalars})
    Z = mixture_fraction(Y)
    rho = mech.density(T, Y)
    w, Q = mech.omega(T, Y, rho)
    rr = -w[0]                                          # fuel consumption rate kg/m3/s
    fl = np.broadcast_to(g.fluid, T.shape)
    Zc = np.broadcast_to(g.zc[None, :, None], T.shape)
    burning = Zc[(rr > 0.05 * rr[fl].max()) & fl]
    vol = np.broadcast_to(g.vol, T.shape) * 2 * np.pi / g.Nt
    info = dict(Tmax=float(T[fl].max()), tip_rr_mm=float(burning.max() * 1e3),
                tip_axis_mm=float(tip_height(g, Z) * 1e3), Qtot_W=float((Q * vol)[fl].sum()))
    info.update({k: v for k, v in flame_geometry(g, Z).items() if k in ('tip_z_mm', 'tip_x_mm', 'tip_angle', 'cen_angle', 'cen_x_mm')})
    return {k: round(v, 3) for k, v in info.items()}


if __name__ == '__main__':
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    V = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    t_end = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    hr = float(sys.argv[4]) * 1e-3 if len(sys.argv) > 4 else 0.375e-3
    hz = float(sys.argv[5]) * 1e-3 if len(sys.argv) > 5 else 0.5e-3
    restart = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] != '-' else None
    tag = f'stage4_M{M}_V{V:g}_h{hr*1e3:g}' + os.environ.get('TAG', '')
    sol, case, hist = run(Chemistry, M, V, t_end, hr, hz, restart, tag, extra=diagnostics, every=0.01,
                          cfl=float(os.environ.get('CFL', 0.4)))
    res = dict(M=M, V=V, t=sol.t, steps=sol.n, final=hist[-1], timing=sol.timing,
               ref=dict(Tmax=1923.6, tip_mm=44.7, Q_W=73))
    print(json.dumps(res))
    json.dump(dict(res, hist=hist), open(os.path.join(OUT, tag + '.json'), 'w'), indent=1)
