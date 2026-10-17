"""Export simulated emission densities (linear Rec.709 RGB) on an (r, z) grid for Blender."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from foamio import read_field, latest_time
from scipy.interpolate import griddata

CASE, OUT = sys.argv[1], sys.argv[2]
TIME = sys.argv[3] if len(sys.argv) > 3 else latest_time(CASE)
os.makedirs(OUT, exist_ok=True)

lam = np.arange(380, 781, 5.0)
def g(x, mu, s1, s2): return np.exp(-0.5 * ((x - mu) / np.where(x < mu, s1, s2)) ** 2)
CMF = np.stack([1.056*g(lam,599.8,37.9,31.0)+0.362*g(lam,442.0,16.0,26.7)-0.065*g(lam,501.1,20.4,26.2),
                0.821*g(lam,568.8,46.9,40.5)+0.286*g(lam,530.9,16.3,31.1),
                1.217*g(lam,437.0,11.8,36.0)+0.681*g(lam,459.0,26.0,13.8)])
M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
def planck(l_nm, T):
    l = l_nm * 1e-9; return 1.191e-16 / l**5 / np.expm1(1.4388e-2 / (l * T))
Ttab = np.linspace(300, 2600, 400)
soot_tab = np.array([CMF @ (planck(lam, T) / lam) for T in Ttab]); soot_tab /= soot_tab[:, 1].max()
line = lambda mu, w: np.exp(-0.5 * ((lam - mu) / w) ** 2)
chemi = line(431.4, 3) + 0.35*line(473.7, 4) + 0.55*line(516.5, 4) + 0.25*line(563.5, 4) + 0.30*np.exp(-0.5*((lam-440)/45)**2)
chemi_xyz = CMF @ chemi; chemi_xyz /= chemi_xyz[1]

h = np.radians(2.5)
rc = read_field(f'{CASE}/0/Cx') / np.cos(h); zc = read_field(f'{CASE}/0/Cz')
T = read_field(f'{CASE}/{TIME}/T'); U = read_field(f'{CASE}/{TIME}/U')
Y = {s: read_field(f'{CASE}/{TIME}/{s}') for s in ['C12H26', 'O2', 'CO', 'CO2', 'H2O']}
W = {'C12H26': 170.34, 'O2': 32.0, 'CO': 28.01, 'CO2': 44.01, 'H2O': 18.02}
Wmix = 1 / (sum(Y[s] / W[s] for s in Y) + (1 - sum(Y.values())) / 28.01)
rho = 101325 * Wmix / (8314.47 * T)
cF = np.clip(rho * Y['C12H26'] / 170.34, 0, None); cO = np.clip(rho * Y['O2'] / 32.0, 0, None)
rr = 9.6e11 * 1000 ** (-0.75) * cF ** 0.25 * cO ** 1.5 * np.exp(-15096.6 / T)
Zc = (Y['C12H26'] * 144.13/170.34 + Y['CO'] * 12.01/28.01 + Y['CO2'] * 12.01/44.01) / (144.13/170.34)
Zst = 0.0621
soot = Y['CO'] * np.clip((T - 1300) / 400, 0, 1) * np.clip((Zc - 0.8*Zst) / (2*Zst), 0, 1) \
       * np.exp(-Y['O2'] / 0.01) * np.clip((zc - 6e-3) / 12e-3, 0, 1) ** 1.3

RM, ZM, dr = 15e-3, 70e-3, 0.05e-3
rg = np.arange(dr / 2, RM, dr); zg = np.arange(dr / 2, ZM, dr)
RG, ZG = np.meshgrid(rg, zg)
sel = (rc < RM * 1.1) & (zc < ZM * 1.1)
def onto(v):
    return np.nan_to_num(griddata(np.column_stack([rc[sel], zc[sel]]), v[sel], (RG, ZG), method='linear'))
Tg, Sg, Rg = onto(T), onto(soot), onto(rr)
wick = (RG < 0.75e-3) & (ZG < 6e-3); Sg[wick] = 0; Rg[wick] = 0
idx = np.clip(np.searchsorted(Ttab, Tg), 0, len(Ttab) - 1)
soot_rgb = np.clip(np.einsum('ij,zrj->zri', M, Sg[..., None] * soot_tab[idx]), 0, None)
chem_rgb = np.clip(np.einsum('ij,zrj->zri', M, Rg[..., None] * chemi_xyz[None, None]), 0, None)
# normalise each so that a chord through the brightest region integrates to ~1 per mm (scene tuning)
soot_rgb /= soot_rgb[..., 1].max(); chem_rgb /= chem_rgb[..., 1].max()
np.save(f'{OUT}/soot_rgb.npy', soot_rgb.astype(np.float32))
np.save(f'{OUT}/chem_rgb.npy', chem_rgb.astype(np.float32))
np.save(f'{OUT}/T.npy', Tg.astype(np.float32))
np.save(f'{OUT}/Ur.npy', onto(U[:, 0]).astype(np.float32)); np.save(f'{OUT}/Uz.npy', onto(U[:, 2]).astype(np.float32))
np.save(f'{OUT}/rr.npy', Rg.astype(np.float32))
print('grid', soot_rgb.shape, 'dr mm', dr * 1e3)
