"""Stage 1: cold uniform cross-draft (m = 1) through the empty domain.

Outer boundary r = R: u_r = V cos(theta), u_theta = -V sin(theta), u_z = 0; bottom: slip wall carrying the same
horizontal draft; top: outflow (p = 0, zero-gradient velocity). Start from rest with a 50 ms ramp of V.
Check: the interior flow converges to the uniform draft, max |u - V e_x| / V < 1 %.
usage: python stage1_crossflow.py [M] [h_mm] [t_end] [V]
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import Grid, BC, stretched, OUTER, OPEN, TOP, P0, RU
from solver import Solver, VEL_D

M = int(sys.argv[1]) if len(sys.argv) > 1 else 4
h = float(sys.argv[2]) * 1e-3 if len(sys.argv) > 2 else 1e-3
T_END = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
V = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
R, Z = 0.04, 0.12
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
os.makedirs(OUT, exist_ok=True)


class Crossflow:
    def __init__(self):
        self.g = g = Grid(stretched(R, h, 6e-3, 1.12), stretched(Z, h, 30e-3, 1.12), M)
        self.rho_inf = P0 * 28.85 / (RU * 300.0)
        self.mu_air = 1.846e-5
        self.scalars = ['c']
        th = g.theta[:, None, None]
        self.cos, self.sin = np.cos(th), np.sin(th)
        zr = np.zeros((g.Nt, g.Nz, g.Nr + 1)); zz = np.zeros((g.Nt, g.Nz + 1, g.Nr))
        self.scalar_bc = {'c': BC(set(), 1.0, zr, zz, inflow_labels={OUTER, OPEN, TOP})}

    def V(self, t):
        return V * min(1.0, t / 0.05)

    def vel_bc(self, t):
        g, v = self.g, self.V(t)
        shr, shz = (g.Nt, g.Nz, g.Nr + 1), (g.Nt, g.Nz + 1, g.Nr)
        ur = BC({OUTER, OPEN}, -1.0, np.broadcast_to(v * self.cos, shr).copy(), np.broadcast_to(v * self.cos, shz).copy())
        ut = BC({OUTER, OPEN}, -1.0, np.broadcast_to(-v * self.sin, shr).copy(), np.broadcast_to(-v * self.sin, shz).copy())
        uz = BC({OUTER, OPEN}, 1.0, np.zeros(shr), np.zeros(shz))
        return ur, ut, uz

    def bnd_mflux(self, t, rho):
        g, v = self.g, self.V(t)
        Fr = np.zeros((g.Nt, g.Nz, g.Nr + 1)); Fz = np.zeros((g.Nt, g.Nz + 1, g.Nr))
        Fr[..., -1] = self.rho_inf * v * self.cos[..., 0]
        return Fr, Fz

    def props(self, phi):
        g = self.g
        sh = (g.Nt, g.Nz, g.Nr)
        rho = np.full(sh, self.rho_inf)
        mu = np.full(sh, self.mu_air)
        return rho, mu, mu / 0.7


def error(sol):
    g = sol.g
    f = sol.fields()
    V_ = sol.c.V(sol.t)
    th = g.theta[:, None, None]
    e = np.sqrt((f['ur'] - V_ * np.cos(th)) ** 2 + (f['ut'] + V_ * np.sin(th)) ** 2 + f['uz'] ** 2) / V_
    hi = np.abs(sol.us[:, 2:]).max() / V_ if g.M >= 2 else 0.0
    return e.max(), np.sqrt((e ** 2 * g.vol).sum() / (g.vol.sum() * g.Nt)), hi, np.unravel_index(e.argmax(), e.shape)


if __name__ == '__main__':
    case = Crossflow()
    g = case.g
    print(f'grid Nr={g.Nr} Nz={g.Nz} M={M} Nt={g.Nt}')
    sol = Solver(case, cfl=0.5, dt_max=2e-3)
    sol.set_state(np.zeros((3, g.Nt, g.Nz, g.Nr)), {'c': np.zeros((g.Nt, g.Nz, g.Nr))})
    hist = []
    t0 = time.time()
    next_out = 0.0
    while sol.t < T_END - 1e-12:
        dt = sol.step()
        if sol.t >= next_out - 1e-12:
            emax, erms, ehi, loc = error(sol)
            hist.append((sol.t, emax, erms, ehi))
            print(f't={sol.t:.3f} dt={dt:.2e} err max {100 * emax:.3f}% rms {100 * erms:.3f}% m>=2 {100 * ehi:.2e}% '
                  f'at (th,j,i)={loc} div={sol.last_div:.2e} wall {time.time() - t0:.0f}s', flush=True)
            next_out += T_END / 20
    emax, erms, ehi, loc = error(sol)
    res = dict(M=M, h_mm=h * 1e3, V=V, t=sol.t, err_max=emax, err_rms=erms, err_modes_ge2=ehi,
               div_resid=float(sol.last_div), steps=sol.n, wall_s=time.time() - t0, Nr=g.Nr, Nz=g.Nz)
    print(json.dumps(res))
    tag = f'stage1_M{M}_h{h*1e3:g}'
    json.dump(res, open(os.path.join(OUT, tag + '.json'), 'w'), indent=1)
    np.save(os.path.join(OUT, tag + '_hist.npy'), np.array(hist))
    sol.save(os.path.join(OUT, tag + '.npz'))
