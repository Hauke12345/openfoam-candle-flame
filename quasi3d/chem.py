"""Stiff chemistry: per-point constant-pressure reactor integrated with an adaptive 2nd-order Rosenbrock method
(ROS2, gamma = 1 + 1/sqrt(2), embedded 1st-order error estimate), finite-difference Jacobian. Numba, parallel
over points. Same kinetics as mech.py / constant/reactions:

  dY_k/dt = W_k sum_j nu_kj q_j / rho,   dT/dt = (Qdot/rho - sum_k hs_k dY_k/dt) / cp,   Qdot = -sum_k hf_k w_k
(energy as sensible enthalpy, so dhs/dt = Qdot/rho; rho = p W / (R T) re-evaluated inside the reactor).
"""
import numpy as np
import numba as nb
import mech

_HI, _LO, _W, _HF = mech._HI, mech._LO, mech.W, mech.HF
_A, _TA, _ORD, _NU = mech.RA, mech.RTA, mech.ORD, mech.NU


@nb.njit(cache=True)
def _rhs(y, p, hi, lo, Wk, hf, A, Ta, ORD, NU, out):
    """y = (Y0..Y4, T); N2 = 1 - sum."""
    T = min(max(y[5], 200.0), 4000.0)
    Y = np.empty(6)
    s = 0.0
    for k in range(5):
        Y[k] = y[k]
        s += y[k]
    Y[5] = 1.0 - s
    winv = 0.0
    for k in range(6):
        winv += Y[k] / Wk[k]
    rho = p / (winv * 8314.46261815324 * T)
    c = np.empty(6)
    for k in range(6):
        c[k] = max(rho * Y[k] / Wk[k], 0.0)
    w = np.zeros(6)
    for j in range(3):
        q = A[j] * np.exp(-Ta[j] / T)
        for k in range(6):
            if ORD[j, k] != 0.0:
                q *= c[k] ** ORD[j, k]
        for k in range(6):
            w[k] += NU[j, k] * q
    cp = 0.0
    hsdY = 0.0
    Q = 0.0
    for k in range(6):
        a = hi[k] if T >= 1000.0 else lo[k]
        rw = 8314.46261815324 / Wk[k]
        cpk = (a[0] + T * (a[1] + T * (a[2] + T * (a[3] + T * a[4])))) * rw
        hk = (a[0] + T * (a[1] / 2 + T * (a[2] / 3 + T * (a[3] / 4 + T * a[4] / 5))) + a[5] / T) * rw * T - hf[k]
        wk = w[k] * Wk[k]
        cp += Y[k] * cpk
        Q -= hf[k] * wk
        dY = wk / rho
        if k < 5:
            out[k] = dY
        hsdY += hk * dY
    out[5] = (Q / rho - hsdY) / cp
    return Q


@nb.njit(cache=True)
def _solve6(M, b):
    """Gaussian elimination with partial pivoting, 6x6 (in place copies)."""
    n = 6
    a = M.copy()
    x = b.copy()
    for i in range(n):
        piv = i
        for r in range(i + 1, n):
            if abs(a[r, i]) > abs(a[piv, i]):
                piv = r
        if piv != i:
            for cc in range(n):
                tmp = a[i, cc]; a[i, cc] = a[piv, cc]; a[piv, cc] = tmp
            tmp = x[i]; x[i] = x[piv]; x[piv] = tmp
        for r in range(i + 1, n):
            f = a[r, i] / a[i, i]
            for cc in range(i, n):
                a[r, cc] -= f * a[i, cc]
            x[r] -= f * x[i]
    for i in range(n - 1, -1, -1):
        s = x[i]
        for cc in range(i + 1, n):
            s -= a[i, cc] * x[cc]
        x[i] = s / a[i, i]
    return x


@nb.njit(parallel=True, cache=True)
def _integrate(Y5, T, dt, p, active, hi, lo, Wk, hf, A, Ta, ORD, NU, rtol, atolY, atolT, qavg):
    n = T.shape[0]
    g = 1.0 + 1.0 / np.sqrt(2.0)
    nsub = np.zeros(n, np.int64)
    for i in nb.prange(n):
        if not active[i]:
            qavg[i] = 0.0
            continue
        y = np.empty(6)
        for k in range(5):
            y[k] = Y5[k, i]
        y[5] = T[i]
        f0 = np.empty(6); f1 = np.empty(6); fp = np.empty(6)
        J = np.empty((6, 6)); W = np.empty((6, 6))
        t = 0.0
        h = dt
        qint = 0.0
        it = 0
        while t < dt * (1 - 1e-12) and it < 5000:
            it += 1
            h = min(h, dt - t)
            Q0 = _rhs(y, p, hi, lo, Wk, hf, A, Ta, ORD, NU, f0)
            for cc in range(6):
                d = 1e-7 * max(abs(y[cc]), 1e-6) if cc < 5 else 1e-6 * y[5]
                yp = y.copy(); yp[cc] += d
                _rhs(yp, p, hi, lo, Wk, hf, A, Ta, ORD, NU, fp)
                for r in range(6):
                    J[r, cc] = (fp[r] - f0[r]) / d
            while True:
                for r in range(6):
                    for cc in range(6):
                        W[r, cc] = (1.0 if r == cc else 0.0) - g * h * J[r, cc]
                k1 = _solve6(W, f0)
                y1 = y + h * k1
                _rhs(y1, p, hi, lo, Wk, hf, A, Ta, ORD, NU, f1)
                k2 = _solve6(W, f1 - 2.0 * k1)
                ynew = y + h * (1.5 * k1 + 0.5 * k2)
                err = 0.0
                for cc in range(6):
                    sc = (atolY if cc < 5 else atolT) + rtol * max(abs(y[cc]), abs(ynew[cc]))
                    e = abs(0.5 * h * (k1[cc] + k2[cc])) / sc
                    err = max(err, e)
                if err <= 1.0 or h < 1e-12:
                    break
                h *= max(0.2, 0.9 / np.sqrt(err))
            qint += Q0 * h
            t += h
            for cc in range(5):
                ynew[cc] = min(max(ynew[cc], 0.0), 1.0)
            y = ynew
            h *= min(4.0, 0.9 / np.sqrt(max(err, 1e-10)))
        for k in range(5):
            Y5[k, i] = y[k]
        T[i] = y[5]
        qavg[i] = qint / dt
        nsub[i] = it
    return nsub


def react(Y5, T, dt, p=mech.P0, rtol=1e-4, atolY=1e-8, atolT=1e-3, Tmin=500.0):
    """advance (5, N) mass fractions and (N,) temperature over dt at constant pressure; returns mean heat release."""
    Y5 = np.ascontiguousarray(Y5, float).copy()
    T = np.ascontiguousarray(T, float).copy()
    active = (T > Tmin) & (Y5[1] > 1e-7) & ((Y5[0] > 1e-9) | (Y5[2] > 1e-8) | (Y5[3] > 1e-6))
    q = np.zeros_like(T)
    nsub = _integrate(Y5, T, dt, p, active, _HI, _LO, _W, _HF, _A, _TA, _ORD, _NU, rtol, atolY, atolT, q)
    return Y5, T, q, nsub


if __name__ == '__main__':
    # check against Cantera constant-pressure reactor with the same mechanism
    import sys, os, time
    import cantera as ct
    sys.path.insert(0, os.path.join(mech.ROOT, 'calib'))
    import importlib.util
    spec = importlib.util.spec_from_file_location('cmech', os.path.join(mech.ROOT, 'calib', 'mech.py'))
    cm = importlib.util.module_from_spec(spec); spec.loader.exec_module(cm)
    gas = cm.twostep(9.6e11)
    Yt = np.array([0.03, 0.2, 0.0, 0.0, 0.0])
    T0 = 1300.0
    gas.TPY = T0, mech.P0, {'c12h26': Yt[0], 'o2': Yt[1], 'n2': 1 - Yt.sum()}
    # Cantera thermo differs slightly (original NASA fits), so compare composition evolution with T from Cantera
    r = ct.IdealGasConstPressureReactor(gas, energy='on')
    net = ct.ReactorNet([r])
    dt = 2e-5
    Y5 = Yt[:, None].copy(); T = np.array([T0])
    t0 = time.time()
    for n in range(20):
        Y5, T, q, ns = react(Y5, T, dt)
        net.advance((n + 1) * dt)
        if n % 5 == 4:
            yc = [r.thermo[s.lower()].Y[0] for s in mech.SPECIES[:5]]
            print(f't={(n+1)*dt:.1e} T {T[0]:.1f} vs {r.thermo.T:.1f}  Yfuel {Y5[0,0]:.4e} vs {yc[0]:.4e}  YCO {Y5[2,0]:.4e} vs {yc[2]:.4e} nsub {ns[0]}')
    print('time', time.time() - t0)
