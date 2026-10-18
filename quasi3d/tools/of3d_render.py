"""Render the OpenFOAM half-3D case with the quasi-3D renderer (render3d.py), frame by frame.

The half domain (y >= 0) is mirrored across the symmetry plane, hs and Y are interpolated (linear, Delaunay
barycentric weights computed once) onto a fine polar grid (r, z, theta at Nt angles) and wrapped into a render3d
state with THETA_FV-type scalars (values at the Nt angles).
usage: python3 of3d_render.py CASE OUTDIR [times...]     (writes OUTDIR/<time>.npz states, renderable by render3d)
"""
import sys, os, glob
import numpy as np
from scipy.spatial import Delaunay
HERE = os.path.dirname(os.path.abspath(__file__))
Q = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(Q)); sys.path.insert(0, Q)
from foamio import read_field
import mech

SP5 = ['C12H26', 'O2', 'CO', 'CO2', 'H2O']


def polar_grid(dr=0.2e-3, dz=0.25e-3, R=38e-3, Z=80e-3, Nt=64):
    rf = np.arange(0, R + 1e-12, dr); zf = np.arange(0, Z + 1e-12, dz)
    return rf, zf, Nt


def weights(C, rf, zf, Nt):
    rc, zc = 0.5 * (rf[1:] + rf[:-1]), 0.5 * (zf[1:] + zf[:-1])
    th = 2 * np.pi * np.arange(Nt) / Nt
    P = np.concatenate([C, C * np.array([1, -1, 1])])          # mirror y -> -y
    tri = Delaunay(P)
    TH, ZZ, RR = np.meshgrid(th, zc, rc, indexing='ij')
    X = np.c_[(RR * np.cos(TH)).ravel(), (RR * np.sin(TH)).ravel(), ZZ.ravel()]
    s = tri.find_simplex(X)
    ok = s >= 0
    T = tri.transform[s]
    b = np.einsum('nij,nj->ni', T[:, :3], X - T[:, 3])
    w = np.c_[b, 1 - b.sum(1)]
    idx = tri.simplices[s]
    w[~ok] = 0; idx[~ok] = 0
    return idx, w, ok, (Nt, len(zc), len(rc))


def state(case, t, idx, w, ok, shape, rf, zf, ncell):
    Tf = read_field(f'{case}/{t}/T')
    Yf = np.array([read_field(f'{case}/{t}/{s}') for s in SP5])
    vals = {'T': np.r_[Tf, Tf]}
    for k, s in enumerate(SP5):
        vals[s] = np.r_[Yf[k], Yf[k]]
    g = {k: np.where(ok, (v[idx] * w).sum(1), {'T': 300.0, 'O2': 0.23}.get(k, 0.0)).reshape(shape) for k, v in vals.items()}
    Y = mech.full_Y(np.array([g[s] for s in SP5]))
    hs = mech.hs_mix(g['T'], Y)
    Nt = shape[0]
    spec = lambda f: np.fft.rfft(f, axis=0) / Nt                # all Nt//2+1 modes (values at the Nt angles)
    phis = np.array([spec(hs)] + [spec(g[s]) for s in SP5])
    return dict(t=float(t), rf=rf, zf=zf, M=Nt // 2, us=np.zeros(1), ps=np.zeros(1), phis=phis,
                scalars=np.array(['hs'] + SP5), theta_fv=True)


if __name__ == '__main__':
    case, out = sys.argv[1], sys.argv[2]
    times = sys.argv[3:] or sorted([d for d in os.listdir(case) if d.replace('.', '').isdigit()], key=float)
    os.makedirs(out, exist_ok=True)
    C = read_field(f'{case}/0/C')
    rf, zf, Nt = polar_grid()
    idx, w, ok, shape = weights(C, rf, zf, Nt)
    print('grid', shape, 'inside', ok.mean())
    for t in times:
        s = state(case, t, idx, w, ok, shape, rf, zf, len(C))
        np.savez_compressed(os.path.join(out, f'{float(t):07.3f}.npz'), **s)
        print('wrote', t, flush=True)
