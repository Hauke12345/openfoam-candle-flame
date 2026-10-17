"""Stage 2: hot buoyant plume with a fixed hot source in place of the flame.

Variable-density low-Mach flow with the full thermo/transport of the OpenFOAM case (sensible enthalpy + 5 species,
no reactions). Inside the flame envelope of the OpenFOAM solution (T > 1200 K or Y_fuel > 0.01, z < 46 mm) the
temperature and composition are held at the OpenFOAM values (m = 0; m >= 1 set to zero), elsewhere they are
transported (convection, diffusion, optically thin radiation). The velocity field is free everywhere.
Check (M = 0): on-axis u_z(z) and T(z) against OpenFOAM. Then add the draft.
usage: python stage2_plume.py M V t_end [hr_mm hz_mm] [restart.npz]
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mech
from core import BC, OUTER, OPEN, TOP, POOL, WICK
from solver import Solver
from candle import CandleBase, candle_grid, Y_AIR5, Y_FUEL5, SP5, T_AIR, T_WICK, T_POOL, axis_profile

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')


class Thermo(CandleBase):
    """sensible enthalpy + 5 species with OpenFOAM thermo and transport; rho D = mu (Sc = 1) for species and
    kappa/cp for hs, as in OpenFOAM's rhoReactingBuoyantFoam."""
    def __init__(self, g, V=0.0, **kw):
        super().__init__(g, V, **kw)
        self.scalars = ['hs'] + SP5
        Yair, Yfuel = mech.full_Y(Y_AIR5[:, None])[:, 0], mech.full_Y(Y_FUEL5[:, None])[:, 0]
        self.hs_air = float(mech.hs_mix(np.array(T_AIR), Yair))
        self.hs_wick = float(mech.hs_mix(np.array(T_WICK), Yfuel))
        self.hs_pool = float(mech.hs_mix(np.array(T_POOL), Yair))
        inflow = {OUTER, OPEN, TOP}
        self.scalar_bc = {'hs': BC({WICK, POOL}, 1.0, *self.scalar_face_arrays(self.hs_air, self.hs_wick, self.hs_pool),
                                   solid_fill=self.hs_wick, inflow_labels=inflow)}
        for k, s in enumerate(SP5):
            self.scalar_bc[s] = BC({WICK}, 1.0, *self.scalar_face_arrays(Y_AIR5[k], Y_FUEL5[k]),
                                   solid_fill=Y_FUEL5[k], inflow_labels=inflow)
        self.T = None

    def state(self, phi):
        Y = mech.full_Y(np.clip(np.array([phi[s] for s in SP5]), 0, 1))
        Y[5] = np.clip(Y[5], 0, 1)
        Y /= Y.sum(0)
        T = mech.T_from_hs(phi['hs'], Y, self.T if self.T is not None and self.T.shape == phi['hs'].shape else None)
        return T, Y

    def props(self, phi):
        T, Y = self.state(phi)
        self.T, self.Y = T, Y
        rho = mech.density(T, Y)
        mu, kappa, cp, alpha = mech.transport(T, Y)
        Gam = {'hs': alpha}
        for s in SP5:
            Gam[s] = mu
        return rho, mu, Gam

    def dlnrho(self, phi):
        """rho = p W/(R T(hs, Y)): (1/rho) d rho/d hs = -1/(cp T); (1/rho) d rho/d Y_k (N2 = 1 - sum) =
        -W (1/W_k - 1/W_N2) + (hs_k - hs_N2)/(cp T), evaluated at the state of the last props() call."""
        T, Y = self.T, self.Y
        cp = mech.cp_mix(T, Y)
        Wm = mech.wmix(Y)
        hk = mech.hs_k(T)
        out = {'hs': -1.0 / (cp * T)}
        for k, s in enumerate(SP5):
            out[s] = -Wm * (1 / mech.W[k] - 1 / mech.W[5]) + (hk[k] - hk[5]) / (cp * T)
        return out

    def initial_scalars(self):
        g, ref = self.g, self.ref
        T = ref.on_grid(g, 'T')
        Y5 = np.array([ref.on_grid(g, 'Y_' + s) for s in SP5])
        Y = mech.full_Y(Y5)
        phi = {'hs': np.broadcast_to(mech.hs_mix(T, Y), (g.Nt, g.Nz, g.Nr)).copy()}
        for k, s in enumerate(SP5):
            phi[s] = np.broadcast_to(Y5[k], (g.Nt, g.Nz, g.Nr)).copy()
        return phi

    def source(self, phi, rho, dt, t):
        # radiation with the lagged temperature/composition (slow process)
        phi['hs'] = phi['hs'] + dt * mech.q_rad_thin(self.T, self.Y) / rho
        return phi


class Forced(Thermo):
    def __init__(self, g, V=0.0, **kw):
        super().__init__(g, V, **kw)
        ref = self.ref
        T = ref.on_grid(g, 'T')
        Yf = ref.on_grid(g, 'Y_C12H26')
        Zc = g.zc[:, None] + 0 * g.rc[None]
        self.mask = ((T > 1200) | (Yf > 0.01)) & (Zc < 46e-3) & g.fluid
        init = self.initial_scalars()
        self.target = {s: init[s][0] for s in self.scalars}

    def fix(self, phis, t):
        for s in self.scalars:
            phis[s][0][self.mask] = self.target[s][self.mask]
            phis[s][1:, self.mask] = 0


def compare_axis(sol, case):
    g = sol.g
    f = sol.fields()
    T, _ = case.state({s: f[s] for s in case.scalars})
    zt, Tt = case.ref.axis('T')
    _, Ut = case.ref.axis('Uz')
    Ta = axis_profile(g, T)
    Ua = axis_profile(g, f['uz'])
    sel = (g.zc > 7e-3) & (g.zc < 100e-3)
    Tr = np.interp(g.zc[sel], zt, Tt)
    Ur = np.interp(g.zc[sel], zt, Ut)
    out = dict(T_axis_max=float(Ta[sel].max()), T_axis_max_ref=float(Tr.max()),
               Uz_axis_max=float(Ua[sel].max()), Uz_axis_max_ref=float(Ur.max()),
               T_rel_rms=float(np.sqrt(np.mean((Ta[sel] - Tr) ** 2)) / Tr.max()),
               Uz_rel_rms=float(np.sqrt(np.mean((Ua[sel] - Ur) ** 2)) / Ur.max()),
               Uz_rel_maxdev=float(np.abs(Ua[sel] - Ur).max() / Ur.max()))
    for zz in (20e-3, 44e-3, 60e-3, 80e-3):
        j = np.argmin(abs(g.zc - zz))
        out[f'z{zz*1e3:.0f}'] = dict(T=float(Ta[j]), T_ref=float(np.interp(g.zc[j], zt, Tt)),
                                      Uz=float(Ua[j]), Uz_ref=float(np.interp(g.zc[j], zt, Ut)))
    return out, (g.zc, Ta, Ua, zt, Tt, Ut)


def run(case_cls, M, V, t_end, hr, hz, restart=None, tag='stage2', t_on=-1.0, cfl=0.4, every=0.02, extra=None):
    g = candle_grid(M, hr, hz)
    case = case_cls(g, V, t_on=t_on)
    print(f'{tag}: grid Nr={g.Nr} Nz={g.Nz} M={M} Nt={g.Nt} V={V}', flush=True)
    sol = Solver(case, cfl=cfl, dt_max=2e-4)
    if restart:
        sol.load(restart)
        if V > 0 and case.t_on < sol.t:
            # switch the draft on by superposing the uniform draft on the restart state (potential start)
            f = sol.fields()
            u = np.array([f['ur'] + V * case.cos, f['ut'] - V * case.sin, f['uz']])
            ps = sol.ps.copy()
            sol.set_state(u, {s: f[s] for s in case.scalars}, sol.t)
            sol.ps = ps
    else:
        sol.set_state(case.initial_velocity(), case.initial_scalars())
    t0 = time.time()
    nxt = sol.t + every
    hist = []
    while sol.t < t_end - 1e-12:
        dt = sol.step()
        if not np.isfinite(sol.us).all():
            print('NaN at t', sol.t); break
        if sol.t >= nxt - 1e-12 or sol.t >= t_end - 1e-12:
            nxt += every
            info = extra(sol, case) if extra else {}
            hist.append(dict(t=sol.t, umax=float(np.abs(sol.us).max()), **info))
            print(f't={sol.t:.4f} dt={dt:.2e} n={sol.n} div={sol.last_div:.1e} |u|max={np.abs(sol.us).sum(1).max():.3f} '
                  f'{json.dumps(info)} wall={time.time() - t0:.0f}s', flush=True)
    sol.save(os.path.join(OUT, tag + '.npz'))
    return sol, case, hist


if __name__ == '__main__':
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    V = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    t_end = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    hr = float(sys.argv[4]) * 1e-3 if len(sys.argv) > 4 else 0.375e-3
    hz = float(sys.argv[5]) * 1e-3 if len(sys.argv) > 5 else 0.5e-3
    restart = sys.argv[6] if len(sys.argv) > 6 else None
    tag = f'stage2_M{M}_V{V:g}_h{hr*1e3:g}'

    def extra(sol, case):
        res, _ = compare_axis(sol, case)
        return {k: round(v, 4) for k, v in res.items() if not isinstance(v, dict)}
    sol, case, hist = run(Forced, M, V, t_end, hr, hz, restart, tag, extra=extra)
    res, prof = compare_axis(sol, case)
    res.update(M=M, V=V, t=sol.t, steps=sol.n, timing=sol.timing)
    print(json.dumps(res, indent=1))
    json.dump(dict(res, hist=hist), open(os.path.join(OUT, tag + '.json'), 'w'), indent=1)
    np.save(os.path.join(OUT, tag + '_axis.npy'), np.array([np.interp(prof[3], prof[0], prof[1]), np.interp(prof[3], prof[0], prof[2]), prof[3], prof[4], prof[5]]))
