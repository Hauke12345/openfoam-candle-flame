"""Thermo, transport, kinetics and radiation identical to the OpenFOAM case written by gen.py.

  * NASA7 (janaf) per species, Tcommon 1000 K, sensible enthalpy hs = h(T) - h(298.15 K)
  * Sutherland mu_k = As_k sqrt(T)/(1 + Ts_k/T); mixture As, Ts mass-fraction weighted (OpenFOAM's
    sutherlandTransport mixing); kappa = mu cv (1.32 + 1.77 R/cv) (modified Eucken), alpha = kappa/cp
  * 2-step Westbrook-Dryer + CO2 dissociation (constant/reactions), SI kmol, m3, s
  * grey CO2/H2O absorption (greyMeanAbsorptionEmission coefficients of data/radiationProperties.SandiaD_LTS)
Species order: C12H26, O2, CO, CO2, H2O, N2 (N2 inert, Y_N2 = 1 - sum of the others).
"""
import os, re, json, subprocess, sys
import numpy as np
import numba as nb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SPECIES = ['C12H26', 'O2', 'CO', 'CO2', 'H2O', 'N2']
RU = 8314.46261815324          # J/(kmol K)
P0 = 101325.0
SIGMA = 5.670374419e-8
TSTD = 298.15
JSON = os.path.join(HERE, 'data', 'mech.json')


def _parse(case):
    txt = open(os.path.join(case, 'constant', 'thermo.compressibleGas')).read()
    sp = {}
    for name in SPECIES:
        blk = re.search(r'\n%s\n\{(.*?)\n\}\n' % name, txt, re.S).group(1)
        num = lambda key: float(re.search(key + r'\s+([-\d.eE+]+);', blk).group(1))
        arr = lambda key: [float(x) for x in re.search(key + r'\s*\(([^)]*)\)', blk).group(1).split()]
        sp[name] = dict(W=num('molWeight'), Tcommon=num('Tcommon'), hi=arr('highCpCoeffs'), lo=arr('lowCpCoeffs'),
                        As=num('As'), Ts=num('Ts'))
    rx = open(os.path.join(case, 'constant', 'reactions')).read()
    A = [float(x) for x in re.findall(r'\n\s+A\s+([-\d.eE+]+);', rx)]
    Ta = [float(x) for x in re.findall(r'\n\s+Ta\s+([-\d.eE+]+);', rx)]
    return dict(species=sp, A=A, Ta=Ta)


def load():
    if not os.path.exists(JSON):
        import tempfile
        tmp = tempfile.mkdtemp()
        subprocess.run([sys.executable, os.path.join(ROOT, 'gen.py'), os.path.join(tmp, 'case')], check=True,
                       capture_output=True, cwd=ROOT)
        d = _parse(os.path.join(tmp, 'case'))
        os.makedirs(os.path.dirname(JSON), exist_ok=True)
        json.dump(d, open(JSON, 'w'), indent=1)
    return json.load(open(JSON))


D = load()
W = np.array([D['species'][s]['W'] for s in SPECIES])
_HI = np.array([D['species'][s]['hi'] for s in SPECIES])     # (6, 7)
_LO = np.array([D['species'][s]['lo'] for s in SPECIES])
AS = np.array([D['species'][s]['As'] for s in SPECIES])
TS = np.array([D['species'][s]['Ts'] for s in SPECIES])
TC = 1000.0
# reactions: rate_j = A_j exp(-Ta_j/T) prod c^order ; nu = net stoichiometric coefficients (products - reactants)
RA = np.array(D['A']); RTA = np.array(D['Ta'])
ORD = np.array([[0.25, 1.5, 0, 0, 0, 0],        # C12H26^0.25 + 12.5 O2^1.5 -> 12 CO + 13 H2O
                [0, 0.25, 1.0, 0, 0.5, 0],       # CO + 0.5 O2^0.25 + H2O^0.5 -> CO2 + H2O
                [0, 0, 0, 1.0, 0, 0]])           # CO2 -> CO + 0.5 O2
NU = np.array([[-1, -12.5, 12, 0, 13, 0],
               [0, -0.5, -1, 1, 0, 0],
               [0, 0.5, 1, -1, 0, 0]], float)


def _poly(T):
    """coefficient arrays per species for T: (6, 7, ...)"""
    hi = (T >= TC)[None, None]
    return np.where(hi, _HI[(...,) + (None,) * T.ndim], _LO[(...,) + (None,) * T.ndim])


def cp_k(T):
    """J/(kg K), (6, ...)"""
    a = _poly(T)
    cpR = a[:, 0] + T * (a[:, 1] + T * (a[:, 2] + T * (a[:, 3] + T * a[:, 4])))
    return cpR * RU / W[(...,) + (None,) * T.ndim]


def ha_k(T):
    """absolute enthalpy J/kg, (6, ...)"""
    a = _poly(T)
    hRT = a[:, 0] + T * (a[:, 1] / 2 + T * (a[:, 2] / 3 + T * (a[:, 3] / 4 + T * a[:, 4] / 5))) + a[:, 5] / T
    return hRT * RU * T / W[(...,) + (None,) * T.ndim]


HF = ha_k(np.array(TSTD))       # (6,) formation (absolute at 298.15 K)


def hs_k(T):
    return ha_k(T) - HF[(...,) + (None,) * T.ndim]


def full_Y(Y5):
    """(5, ...) -> (6, ...) with N2 = 1 - sum"""
    return np.concatenate([Y5, (1 - Y5.sum(0))[None]], 0)


def wmix(Y):
    return 1.0 / np.tensordot(1.0 / W, Y, 1)


def hs_mix(T, Y):
    return (hs_k(T) * Y).sum(0)


@nb.njit(parallel=True, cache=True)
def _cp_mix_nb(T, Y, hi, lo, Wk):
    out = np.empty_like(T)
    for p in nb.prange(T.shape[0]):
        Tp = T[p]
        cp = 0.0
        for k in range(6):
            a = hi[k] if Tp >= 1000.0 else lo[k]
            cp += Y[k, p] * (a[0] + Tp * (a[1] + Tp * (a[2] + Tp * (a[3] + Tp * a[4])))) * 8314.46261815324 / Wk[k]
        out[p] = cp
    return out


def cp_mix(T, Y):
    T = np.asarray(T, float)
    return _cp_mix_nb(np.ascontiguousarray(T).ravel(), np.ascontiguousarray(Y, float).reshape(6, -1), _HI, _LO, W).reshape(T.shape)


@nb.njit(parallel=True, cache=True)
def _T_from_hs_nb(hs, Y, T, hi, lo, Wk, hf):
    n = hs.shape[0]
    for p in nb.prange(n):
        Tp = min(max(T[p], 250.0), 3500.0)
        for it in range(50):
            h = 0.0
            cp = 0.0
            for k in range(6):
                a = hi[k] if Tp >= 1000.0 else lo[k]
                cpR = a[0] + Tp * (a[1] + Tp * (a[2] + Tp * (a[3] + Tp * a[4])))
                hRT = a[0] + Tp * (a[1] / 2 + Tp * (a[2] / 3 + Tp * (a[3] / 4 + Tp * a[4] / 5))) + a[5] / Tp
                rw = 8314.46261815324 / Wk[k]
                h += Y[k, p] * (hRT * rw * Tp - hf[k])
                cp += Y[k, p] * cpR * rw
            dT = (hs[p] - h) / cp
            Tp = min(max(Tp + dT, 200.0), 4000.0)
            if abs(dT) < 1e-7:
                break
        T[p] = Tp
    return T


def T_from_hs(hs, Y, T0=None):
    """temperature from sensible enthalpy (Newton, per point, numba)"""
    shp = hs.shape
    T = np.full(hs.size, 1000.0) if T0 is None else np.array(T0, float).ravel().copy()
    return _T_from_hs_nb(np.ascontiguousarray(hs, float).ravel(), np.ascontiguousarray(Y.reshape(6, -1), float), T,
                         _HI, _LO, W, HF).reshape(shp)


def transport(T, Y):
    """mu (Pa s), kappa (W/m/K), cp (J/kg/K), alpha = kappa/cp (kg/m/s)"""
    As = np.tensordot(AS, Y, 1)
    Ts = np.tensordot(TS, Y, 1)
    mu = As * np.sqrt(T) / (1 + Ts / T)
    cp = cp_mix(T, Y)
    R = RU / wmix(Y)
    cv = cp - R
    kappa = mu * cv * (1.32 + 1.77 * R / cv)
    return mu, kappa, cp, kappa / cp


def density(T, Y, p=P0):
    return p * wmix(Y) / (RU * T)


def rates(T, Y, rho):
    """reaction rates q_j (kmol/m3/s), (3, ...); concentrations clipped at 0"""
    c = np.clip(rho * Y / W[(...,) + (None,) * T.ndim], 0, None)
    q = []
    for j in range(3):
        r = RA[j] * np.exp(-RTA[j] / T)
        for k in range(6):
            if ORD[j, k]:
                r = r * c[k] ** ORD[j, k]
        q.append(r)
    return np.array(q)


def omega(T, Y, rho):
    """species mass sources kg/m3/s (6, ...) and heat release (W/m3)"""
    q = rates(T, Y, rho)
    w = np.tensordot(NU.T, q, 1) * W[(...,) + (None,) * T.ndim]
    return w, -(HF[(...,) + (None,) * T.ndim] * w).sum(0)


# ---------------- grey radiation (OpenFOAM greyMeanAbsorptionEmission, inverse-T polynomials, 1/(m atm)) ----------------
_CO2 = [18.741, -121.31e3, 273.5e6, -194.05e9, 56.31e12, -5.8169e15]
_H2O = [-0.23093, -1.12390e3, 9.4153e6, -2.99885e9, 0.51382e12, -1.868e10]


def absorption(T, Y, p=P0):
    Ti = 1.0 / np.clip(T, 200, 2500)
    poly = lambda b: ((((b[5] * Ti + b[4]) * Ti + b[3]) * Ti + b[2]) * Ti + b[1]) * Ti + b[0]
    Wm = wmix(Y)
    X = Y * Wm / W[(...,) + (None,) * T.ndim]
    patm = p / 101325.0
    return X[3] * patm * poly(_CO2) + X[4] * patm * poly(_H2O)


def q_rad_thin(T, Y, T_inf=300.0):
    """optically thin grey emission (W/m3, negative = loss)"""
    return -4 * absorption(T, Y) * SIGMA * (T ** 4 - T_inf ** 4)


if __name__ == '__main__':
    import cantera as ct
    sys.path.insert(0, os.path.join(ROOT, 'calib'))
    from mech import twostep
    gas = twostep(9.6e11)
    T = np.array([400.0, 1200.0, 1900.0])
    Yt = np.array([0.02, 0.18, 0.01, 0.06, 0.04])
    Yf = full_Y(np.repeat(Yt[:, None], 3, 1))
    rho = density(T, Yf)
    w, Q = omega(T, Yf, rho)
    for i, Ti in enumerate(T):
        gas.TPY = Ti, P0, {'c12h26': Yt[0], 'o2': Yt[1], 'co': Yt[2], 'co2': Yt[3], 'h2o': Yt[4], 'n2': 1 - Yt.sum()}
        wc = gas.net_production_rates * gas.molecular_weights
        idx = [gas.species_index(s.lower()) for s in SPECIES]
        print(f'T={Ti}: rho {rho[i]:.5f} vs {gas.density:.5f}; cp {cp_mix(T, Yf)[i]:.2f} vs {gas.cp_mass:.2f}; '
              f'omega rel err {np.abs(w[:, i] - wc[idx]).max() / max(np.abs(wc).max(), 1e-30):.2e}; '
              f'Q {Q[i]:.4e} vs {-np.dot(gas.partial_molar_enthalpies, gas.net_production_rates):.4e}')
