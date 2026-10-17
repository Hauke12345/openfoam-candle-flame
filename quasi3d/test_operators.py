"""Verification of the spatial operators of the quasi-3D solver against manufactured Cartesian solutions.

Checks (grid without wick, Dirichlet exact values on OUTER/OPEN/TOP, axis via parity):
  1. Solver._diffuse      div(Gam grad phi)
  2. Solver._taudiv       div tau, tau = mu (grad u + grad u^T) - 2/3 mu (div u) I   (r, theta, z components)
  3. Solver._advect       u . grad phi  (MUSCL, exact face fluxes)
  4. Grid.lap_matrix      A phi_m + B b_m  vs  L_m phi_m, m = 0, 1, 2
Errors are relative to max|exact|; 'int' excludes 2 cell layers at OUTER/TOP/bottom (axis cells kept); 'axis' = 3 cells next to the axis.
NOTE: manufactured fields are limited to azimuthal content such that all products have m <= M=4 (see below).
Run with cwd quasi3d:  python3 test_operators.py
"""
import time
import numpy as np
import sympy as sy
from core import Grid, BC, div_faces, OUTER, OPEN, TOP

R = H = 0.01
M = 4
NS = (16, 32, 64)
DL = {OUTER, OPEN, TOP}
x, y, z = sy.symbols('x y z')
X, Y, Z = x / R, y / R, z / H
pi = sy.pi

# ---- manufactured fields (Cartesian polynomials / trig => regular at the axis; contain m = 0, 1, 2) ----
phi_e = (1 + X + sy.Rational(1, 2) * X * Y + Y ** 2) * (1 + sy.Rational(1, 2) * sy.cos(pi * Z)) + 0.3 * X ** 2 * sy.sin(pi * Z)
gam_e = 1 + 0.3 * X * Y + 0.2 * X + 0.2 * sy.cos(pi * Z) + 0.1 * Y ** 2   # m <= 2
S2 = X ** 2 + Y ** 2
mu_e = 1 + 0.3 * X - 0.2 * X * Y + 0.2 * sy.sin(pi * Z) + 0.1 * S2
# Cartesian velocity with azimuthal content m <= 1 (=> u_r, u_theta have m <= 2), so every product in the operators
# (m <= 4 = M) is representable after the dealiased truncation; a richer field makes the test itself non-convergent.
ux_e = (1 + X + 0.3 * Y) * (1 + 0.5 * S2) + 0.5 * sy.sin(pi * Z) * (1 + X)
uy_e = (Y + 0.5 * X + 0.4 * Z) * (1 + 0.3 * S2) + 0.2 * X * sy.cos(pi * Z)
uz_e = (1 + Y + 0.3 * X) * sy.cos(pi * Z / 2) * (1 + 0.4 * S2) + 0.2 * X
P = (x, y, z)
D = sy.diff
lap_e = D(phi_e, x, 2) + D(phi_e, y, 2) + D(phi_e, z, 2)
diff_e = sum(D(gam_e * D(phi_e, c), c) for c in P)
U = [ux_e, uy_e, uz_e]
divu = sum(D(U[i], P[i]) for i in range(3))
tau = [[mu_e * (D(U[i], P[j]) + D(U[j], P[i])) - (sy.Rational(2, 3) * mu_e * divu if i == j else 0) for j in range(3)]
       for i in range(3)]
divtau = [sum(D(tau[i][j], P[j]) for j in range(3)) for i in range(3)]
adv_e = sum(U[i] * D(phi_e, P[i]) for i in range(3))
fn = lambda e: sy.lambdify(P, e, 'numpy')
F = dict(phi=fn(phi_e), gam=fn(gam_e), mu=fn(mu_e), ux=fn(ux_e), uy=fn(uy_e), uz=fn(uz_e), lap=fn(lap_e),
         diff=fn(diff_e), dtx=fn(divtau[0]), dty=fn(divtau[1]), dtz=fn(divtau[2]), adv=fn(adv_e))


def ev(name, th, zz, rr):
    """Evaluate a field on broadcast (theta, z, r) coordinates; returns full-shape array."""
    th, zz, rr = np.broadcast_arrays(th[:, None, None], zz[None, :, None], rr[None, None, :])
    return np.asarray(F[name](rr * np.cos(th), rr * np.sin(th), zz)) * np.ones(th.shape)


def cyl(nx, ny, th):
    c, s = np.cos(th), np.sin(th)
    return nx * c + ny * s, -nx * s + ny * c


class Case:
    pass


def setup(N):
    rf, zf = np.linspace(0, R, N + 1), np.linspace(0, H, N + 1)
    g = Grid(rf, zf, M)
    c = Case(); c.g = g; c.rho_inf = 1.0; c.scalars = []; c.scalar_bc = {}
    from solver import Solver
    sol = Solver(c)
    th = g.theta[:, None, None]
    # exact face-value arrays (Nt, Nz, Nr+1) at (rf, zc) and (Nt, Nz+1, Nr) at (rc, zf)
    E = lambda n: (ev(n, g.theta, g.zc, g.rf), ev(n, g.theta, g.zf, g.rc))
    return g, sol, th, E


def errs(g, num, ex):
    """Relative errors err/max|ex|: (max_all, L2_all, max_int, L2_int, max_axis).
    L2 is volume weighted (r dr dz). 'int' excludes 2 cell layers at OUTER/TOP/bottom (axis cells kept);
    'axis' = max over the 3 cells next to the axis (same z range)."""
    e = (num - ex)
    sc = np.abs(ex).max()
    w = np.broadcast_to(g.vol, e.shape)
    msk = np.zeros(e.shape, bool); msk[:, 2:-2, :-2] = True
    ax = np.zeros(e.shape, bool); ax[:, 2:-2, :3] = True
    l2 = lambda m: np.sqrt((w[m] * e[m] ** 2).sum() / w[m].sum()) / sc
    return np.abs(e).max() / sc, l2(np.ones(e.shape, bool)), np.abs(e[msk]).max() / sc, l2(msk), np.abs(e[ax]).max() / sc


def run_all(N):
    g, sol, th, E = setup(N)
    out = {}
    gc = lambda n: ev(n, g.theta, g.zc, g.rc)
    # --- scalar phi, Gamma with exact Dirichlet face values
    pr, pz = E('phi')
    bc_s = BC(DL, 1.0, pr, pz)
    phi = gc('phi')
    # --- velocity (cylindrical)
    uxc, uyc = gc('ux'), gc('uy')
    ur, ut = cyl(uxc, uyc, th); uz = gc('uz')
    uxr, uxz = E('ux'); uyr, uyz = E('uy'); uzr, uzz = E('uz')
    thr = g.theta[:, None, None]
    urr, utr = cyl(uxr, uyr, thr)       # r faces
    urz, utz = cyl(uxz, uyz, thr)       # z faces
    bcs = [BC(DL, -1.0, urr, urz), BC(DL, -1.0, utr, utz), BC(DL, 1.0, uzr, uzz)]
    sol.us = np.array([g.spec(a) for a in (ur, ut, uz)])
    # 1 diffuse
    out['diffuse'] = [errs(g, sol._diffuse(phi, g.spec(phi), gc('gam'), bc_s), gc('diff'))]
    # 2 taudiv
    Dr, Dt, Dz = sol._taudiv(np.array([ur, ut, uz]), bcs, gc('mu'))
    ex = cyl(gc('dtx'), gc('dty'), th) + (gc('dtz'),)
    out['taudiv'] = [errs(g, a, b) for a, b in zip((Dr, Dt, Dz), ex)]
    # 3 advect
    sol.Ur, sol.Uz, sol.Ut = urr, uzz, ut
    sol.rho = np.ones_like(phi)                 # unit density: mass fluxes = volume fluxes
    sol.Fr, sol.Fz = urr, uzz
    divU = div_faces(g, sol.Ur, sol.Uz)
    out['advect'] = [errs(g, sol._advect(phi, g.spec(phi), bc_s, divU), gc('adv'))]
    # 4 lap matrix
    ps = g.spec(phi); ls = g.spec(gc('lap'))
    bsp = g.spec(g.gather_d(DL, pr, pz))
    res = []
    for m in range(3):
        A, B = g.lap_matrix(m, DL)
        v = (A @ ps[m].ravel() + B @ bsp[m]).reshape(g.Nz, g.Nr)
        # complex modes: real & imag both compared (m=0 real)
        e = v - ls[m]; sc = np.abs(ls[m]).max()
        w = g.vol
        msk = np.zeros(e.shape, bool); msk[2:-2, :-2] = True
        ax = np.zeros(e.shape, bool); ax[2:-2, :3] = True
        l2 = lambda mk: np.sqrt((w[mk] * abs(e[mk]) ** 2).sum() / w[mk].sum()) / sc
        res.append((abs(e).max() / sc, l2(np.ones(e.shape, bool)), abs(e[msk]).max() / sc, l2(msk), abs(e[ax]).max() / sc))
    out['lap_m'] = res
    return out


def main():
    t0 = time.time()
    R_ = {N: run_all(N) for N in NS}
    names = ['max_all', 'L2_all', 'max_int', 'L2_int', 'max_axis']
    labels = {'diffuse': ['diffuse'], 'taudiv': ['taudiv r', 'taudiv th', 'taudiv z'], 'advect': ['advect'],
              'lap_m': ['lap m=0', 'lap m=1', 'lap m=2']}
    print('relative errors (err/max|exact|); order = log2(e_N / e_2N)\n')
    for key, labs in labels.items():
        for k, lab in enumerate(labs):
            print(f'{lab:10s} ' + ''.join(f'{n:>14s}' for n in names))
            for N in NS:
                print(f'  N={N:<4d}  ' + ''.join(f'{v:14.3e}' for v in R_[N][key][k]))
            for a, b in zip(NS[:-1], NS[1:]):
                o = [np.log2(p / q) for p, q in zip(R_[a][key][k], R_[b][key][k])]
                print(f'  ord {a}->{b} ' + ''.join(f'{v:14.2f}' for v in o))
    print(f'\nruntime {time.time() - t0:.1f} s')


if __name__ == '__main__':
    main()
