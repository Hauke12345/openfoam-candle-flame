"""Physically based render of the simulated candle flame as a Christmas candle.

Light sources (both optically thin, integrated along the line of sight through the axisymmetric field):
  1. thermal (blackbody) emission of soot: kappa_lambda ~ 1/lambda, Planck B_lambda(T) from the simulated T.
     The CFD has no soot model, so soot volume fraction uses a proxy: hot (T>1300 K), fuel-rich gas carrying CO.
  2. chemiluminescence from the reaction zone: CH* 431 nm, C2* Swan bands 473/516 nm, CO2* blue continuum,
     intensity proportional to the simulated fuel burning rate.
Spectra -> CIE XYZ (Wyman-Sloan-Shirley fit to CIE 1931) -> linear sRGB -> tone map.
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from foamio import read_field, latest_time
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Ellipse, FancyBboxPatch

CASE, OUT = sys.argv[1], sys.argv[2]
TIME = sys.argv[3] if len(sys.argv) > 3 else latest_time(CASE)

# ---------------- colour science ----------------
lam = np.arange(380, 781, 5.0)  # nm
def g(x, mu, s1, s2):
    return np.exp(-0.5 * ((x - mu) / np.where(x < mu, s1, s2)) ** 2)
xbar = 1.056 * g(lam, 599.8, 37.9, 31.0) + 0.362 * g(lam, 442.0, 16.0, 26.7) - 0.065 * g(lam, 501.1, 20.4, 26.2)
ybar = 0.821 * g(lam, 568.8, 46.9, 40.5) + 0.286 * g(lam, 530.9, 16.3, 31.1)
zbar = 1.217 * g(lam, 437.0, 11.8, 36.0) + 0.681 * g(lam, 459.0, 26.0, 13.8)
CMF = np.stack([xbar, ybar, zbar])                       # 3 x nlam
M_XYZ2RGB = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])

def planck(lam_nm, T):
    l = lam_nm * 1e-9
    return 1.191e-16 / l ** 5 / np.expm1(1.4388e-2 / (l * T))

Ttab = np.linspace(300, 2600, 400)
soot_xyz_tab = np.array([CMF @ (planck(lam, T) / lam) for T in Ttab])     # kappa ~ 1/lambda
soot_xyz_tab /= soot_xyz_tab[:, 1].max()

def line(mu, w): return np.exp(-0.5 * ((lam - mu) / w) ** 2)
chemi = 1.0 * line(431.4, 3) + 0.35 * line(473.7, 4) + 0.55 * line(516.5, 4) + 0.25 * line(563.5, 4) \
        + 0.30 * np.exp(-0.5 * ((lam - 440) / 45) ** 2)                 # CO2* continuum
chemi_xyz = CMF @ chemi; chemi_xyz /= chemi_xyz[1]

# ---------------- simulation fields onto (r, z) ----------------
h = np.radians(2.5)
rc = read_field(f'{CASE}/0/Cx') / np.cos(h); zc = read_field(f'{CASE}/0/Cz')
T = read_field(f'{CASE}/{TIME}/T')
Y = {s: read_field(f'{CASE}/{TIME}/{s}') for s in ['C12H26', 'O2', 'CO', 'CO2', 'H2O']}
W = {'C12H26': 170.34, 'O2': 32.0, 'CO': 28.01, 'CO2': 44.01, 'H2O': 18.02}
YN2 = 1 - sum(Y.values())
Wmix = 1 / (sum(Y[s] / W[s] for s in Y) + YN2 / 28.01)
rho = 101325 * Wmix / (8314.47 * T)
cF = np.clip(rho * Y['C12H26'] / W['C12H26'], 0, None); cO = np.clip(rho * Y['O2'] / 32.0, 0, None)
rr = 9.6e11 * 1000 ** (-0.75) * cF ** 0.25 * cO ** 1.5 * np.exp(-15096.6 / T)
Zc = (Y['C12H26'] * 144.13 / 170.34 + Y['CO'] * 12.01 / 28.01 + Y['CO2'] * 12.01 / 44.01) / (144.13 / 170.34)
Zst = 0.0621
# soot proxy: grows with residence in hot rich gas; oxidised where O2 appears
soot = Y['CO'] * np.clip((T - 1300) / 400, 0, 1) * np.clip((Zc - 0.8 * Zst) / (2 * Zst), 0, 1) \
       * np.exp(-Y['O2'] / 0.01) * np.clip((zc - 6e-3) / 12e-3, 0, 1) ** 1.3   # growth with residence time

RM, ZM, dr = 15e-3, 70e-3, 0.05e-3
rg = np.arange(dr / 2, RM, dr); zg = np.arange(0, ZM, dr)
RG, ZG = np.meshgrid(rg, zg)
sel = (rc < RM * 1.1) & (zc < ZM * 1.1)
def onto(v):
    out = griddata(np.column_stack([rc[sel], zc[sel]]), v[sel], (RG, ZG), method='linear')
    return np.nan_to_num(out)
Tg, Sg, Rg = onto(T), onto(soot), onto(rr)
wick = (RG < 0.75e-3) & (ZG < 6e-3)
Sg[wick] = 0; Rg[wick] = 0; Tg[wick] = 600

# emission density in XYZ (per r,z)
idx = np.clip(np.searchsorted(Ttab, Tg), 0, len(Ttab) - 1)
soot_j = Sg[..., None] * soot_xyz_tab[idx]
chem_j = Rg[..., None] * chemi_xyz[None, None, :]

# ---------------- Abel projection (line of sight) ----------------
xg = rg.copy()
edges = np.concatenate([[0], rg + dr / 2])
Mabel = np.zeros((len(xg), len(rg)))
for i, x in enumerate(xg):
    lo = np.sqrt(np.clip(np.maximum(edges[:-1], x) ** 2 - x ** 2, 0, None))
    hi = np.sqrt(np.clip(edges[1:] ** 2 - x ** 2, 0, None))
    Mabel[i] = 2 * np.clip(hi - lo, 0, None)
def project(j):  # j: (nz, nr, 3) -> (nz, nx, 3)
    return np.einsum('xr,zrc->zxc', Mabel, j)
Is, Ic = project(soot_j), project(chem_j)

# relative brightness: soot luminosity dominates, blue reaction zone visible where soot is absent
Is /= np.percentile(Is[..., 1], 99.5)
Ic /= np.percentile(Ic[..., 1], 99.5)
XYZ = 1.0 * Is + 0.11 * Ic
half = np.clip(np.einsum('ij,zxj->zxi', M_XYZ2RGB, XYZ), 0, None)
full = np.concatenate([half[:, ::-1], half], axis=1)        # mirror x -> (nz, 2nr, 3), row 0 = z 0

# ---------------- compose scene ----------------
px = dr * 1e3
W_mm, H_bot, H_top = 64.0, -58.0, 72.0
nx, ny = int(W_mm / px), int((H_top - H_bot) / px)
yy, xx = np.mgrid[0:ny, 0:nx]
X = (xx - nx / 2 + 0.5) * px; Zmm = H_top - (yy + 0.5) * px
E = np.zeros((ny, nx, 3))                                  # flame radiance canvas (linear)
L = full[::-1]; fz, fx = L.shape[:2]
y0 = int(round(H_top / px)) - fz; x0 = nx // 2 - fx // 2
E[max(0, y0):y0 + fz, x0:x0 + fx] = L[max(0, -y0):]

def tonemap(rgb, k):
    Y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    Yt = 1 - np.exp(-k * Y)
    out = rgb * (Yt / np.maximum(Y, 1e-9))[..., None]
    # over-exposed core drifts to white, like a camera sensor
    w = np.clip((Yt - 0.80) / 0.2, 0, 1)[..., None] ** 2
    return out * (1 - w) + w * Yt[..., None]

img = np.zeros((ny, nx, 3))
tt = ((Zmm - H_bot) / (H_top - H_bot))[..., None]
img[:] = np.array([0.010, 0.007, 0.006]) * (1 - tt) + np.array([0.004, 0.006, 0.016]) * tt
glow_src = np.stack([gaussian_filter(E[..., c], 220) for c in range(3)], -1)
glow_src *= 1 / max(glow_src.max(), 1e-9)
glow = glow_src * np.array([1.0, 0.72, 0.42]) / np.array([1.0, 0.72, 0.42]).max()
img += 0.10 * glow

# candle body (Christmas red), radius 10 mm, shaded cylinder lit by the flame
Rcand = 10.0
u = np.clip(X / Rcand, -1, 1)
cyl = np.sqrt(np.clip(1 - u ** 2, 0, 1))
red = np.array([0.42, 0.018, 0.028])
lightF = 0.25 + 1.8 * np.exp(Zmm / 16.0)
col = red * (0.25 + 0.75 * cyl ** 0.8)[..., None] * lightF[..., None]
col += (np.exp(Zmm / 5.0) * np.exp(-(X / 6.5) ** 2))[..., None] * np.array([0.85, 0.22, 0.05])  # glow through wax
col += (0.10 * np.exp(-((u + 0.45) / 0.14) ** 2) * (0.4 + 0.6 * np.exp(Zmm / 25)))[..., None] * np.array([1, 0.55, 0.5])
# drips: tapered, rounded ends, attached to the rim
drip = np.zeros_like(X, dtype=bool)
for xd, ln, w0 in [(-8.2, 15, 1.6), (-3.0, 7, 1.3), (5.6, 20, 1.5), (8.9, 10, 1.1)]:
    zz = -Zmm
    w = w0 * (1 - 0.45 * np.clip(zz / ln, 0, 1))
    drip |= (np.abs(X - xd) < w) & (zz >= 0) & (zz <= ln)
    drip |= (X - xd) ** 2 + (zz - ln) ** 2 < (w0 * 0.85) ** 2
body = (np.abs(X) <= Rcand) & (Zmm <= 0) & (Zmm >= H_bot)
edge_drip = drip & ~body & (Zmm <= 0)
img[body] = col[body]
dc = red * 1.5 * lightF[..., None] * np.array([1.0, 1.1, 1.1])
img[drip & body] = 0.7 * img[drip & body] + 0.3 * dc[drip & body] + 0.04
img[edge_drip] = dc[edge_drip]
# top surface: molten pool seen slightly from above
pool = ((X / Rcand) ** 2 + (Zmm / 1.6) ** 2 < 1) & (Zmm > -1.6)
img[pool] = np.array([1.0, 0.42, 0.20]) * (0.9 + 0.6 * np.exp(-(X[pool] / 5) ** 2))[..., None]

# flame: bloom + core over everything
bloom = np.stack([gaussian_filter(E[..., c], 14) for c in range(3)], -1)
img += tonemap(E, 2.3) + 0.30 * tonemap(bloom, 2.3)

# wick: dark core seen through the (transparent) dark zone, ember tip
wk = (np.abs(X) < 0.7) & (Zmm > -0.5) & (Zmm < 6.0)
img[wk] = 0.08 * img[wk] + np.array([0.012, 0.010, 0.009])
tip = (X ** 2 + (Zmm - 6.0) ** 2) < 0.85 ** 2
img[tip] = np.array([1.0, 0.45, 0.12])
img[(X ** 2 + (Zmm - 6.6) ** 2) < 0.45 ** 2] = [1.0, 0.75, 0.4]

img = np.clip(img, 0, None)
img = np.where(img <= 0.0031308, 12.92 * img, 1.055 * np.power(img, 1 / 2.4) - 0.055)
img = np.clip(img, 0, 1)

fig = plt.figure(figsize=(nx / 200, ny / 200), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
ext = [-W_mm / 2, W_mm / 2, H_bot, H_top]
ax.imshow(img, extent=ext, origin='upper', interpolation='bilinear')
# holly at the base
def holly_leaf(cx, cy, ang, L=17, Wd=6.5):
    t = np.linspace(0, 1, 9)
    up = [(L * s, Wd / 2 * np.sin(np.pi * s) + (1.4 if i % 2 else -0.2) * np.sin(np.pi * s)) for i, s in enumerate(t)]
    dn = [(L * s, -Wd / 2 * np.sin(np.pi * s) - (1.4 if i % 2 else -0.2) * np.sin(np.pi * s)) for i, s in enumerate(t[::-1])]
    P = np.array(up + dn); c, s_ = np.cos(ang), np.sin(ang)
    P = P @ np.array([[c, s_], [-s_, c]]) + [cx, cy]
    ax.add_patch(Polygon(P, closed=True, fc='#0f3d22', ec='#1f6b3b', lw=0.8, zorder=5))
    vein = np.array([[0, 0], [L, 0]]) @ np.array([[c, s_], [-s_, c]]) + [cx, cy]
    ax.plot(vein[:, 0], vein[:, 1], color='#2e7d4a', lw=0.6, zorder=6)
holly_leaf(-4, -52, np.radians(165)); holly_leaf(3, -52, np.radians(15)); holly_leaf(-1, -50, np.radians(100), L=14)
for bx, by in [(-1.5, -51.5), (1.6, -50.8), (0.1, -53.6)]:
    ax.add_patch(Circle((bx, by), 2.1, fc='#c1121f', ec='#7a0a12', lw=0.6, zorder=7))
    ax.add_patch(Circle((bx - 0.7, by + 0.7), 0.55, fc='#ffd0c8', ec='none', alpha=0.85, zorder=8))
ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
os.makedirs(OUT, exist_ok=True)
name = f'{OUT}/xmas_candle.png' if len(sys.argv) <= 3 else f'{OUT}/frame_{float(TIME):.3f}.png'
fig.savefig(name, dpi=200)
print('saved', name, img.shape)
