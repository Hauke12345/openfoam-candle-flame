"""Export quasi-3D snapshots as emission atlases for Blender (blender_scene.py q3d mode).

For each snapshot: emission on a regular (z, r) grid at each of the solver's N_theta angles (THETA_FV scalars live
there), packed side by side into one atlas per light source (tile k = angle 2 pi k / N_theta):
  soot : soot-proxy blackbody (as export_fields.py / render.py), linear Rec.709 RGB
  chem : CH*/C2* chemiluminescence ~ flame-sheet reaction rate rho D |grad Z|^2 at Zst (stage 3, mixture fraction),
         or the fuel-oxidation rate (stage 4)
Normalised with the first frame's maxima, so brightness changes between frames are physical.
Also writes per-frame light data (soot luminosity, emission centroid).
usage: python3 export_q3d.py FRAMEDIR OUTDIR
"""
import sys, os, glob, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import mech
from core import Grid
from candle import ZST

lam = np.arange(380, 781, 5.0)
def _g(x, mu, s1, s2): return np.exp(-0.5 * ((x - mu) / np.where(x < mu, s1, s2)) ** 2)
CMF = np.stack([1.056*_g(lam,599.8,37.9,31.0)+0.362*_g(lam,442.0,16.0,26.7)-0.065*_g(lam,501.1,20.4,26.2),
                0.821*_g(lam,568.8,46.9,40.5)+0.286*_g(lam,530.9,16.3,31.1),
                1.217*_g(lam,437.0,11.8,36.0)+0.681*_g(lam,459.0,26.0,13.8)])
MRGB = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
def planck(l_nm, T):
    l = l_nm * 1e-9; return 1.191e-16 / l**5 / np.expm1(1.4388e-2 / (l * T))
TTAB = np.linspace(300, 2600, 400)
SOOT_TAB = np.array([CMF @ (planck(lam, T) / lam) for T in TTAB]); SOOT_TAB /= SOOT_TAB[:, 1].max()
_line = lambda mu, w: np.exp(-0.5 * ((lam - mu) / w) ** 2)
CHEMI = _line(431.4, 3) + 0.35*_line(473.7, 4) + 0.55*_line(516.5, 4) + 0.25*_line(563.5, 4) + 0.30*np.exp(-0.5*((lam-440)/45)**2)
CHEMI_XYZ = CMF @ CHEMI; CHEMI_XYZ /= CHEMI_XYZ[1]

RM, ZM, DR = 15e-3, 70e-3, 0.1e-3          # Blender map extent (same as export_fields.py), resolution
RG = np.arange(DR / 2, RM, DR); ZG = np.arange(DR / 2, ZM, DR)


def scalars_at_angles(d):
    """physical scalars at the solver's N_theta angles (THETA_FV: all Nt//2+1 modes stored)."""
    phis = d['phis']; n = phis.shape[1]
    Nt = 2 * (n - 1) if n > 1 else 1
    c = np.zeros((Nt // 2 + 1,) + phis.shape[2:], complex)
    out = {}
    for k, s in enumerate(d['scalars']):
        c[:] = 0; c[:n] = phis[k]
        out[str(s)] = np.fft.irfft(c, n=Nt, axis=0) * Nt if Nt > 1 else phis[k].real
    return out, Nt


def emission(path, tab):
    d = np.load(path)
    rc = 0.5 * (d['rf'][1:] + d['rf'][:-1]); zc = 0.5 * (d['zf'][1:] + d['zf'][:-1])
    f, Nt = scalars_at_angles(d)
    if 'Z' in f:
        Z = np.clip(f['Z'], 0, 1)
        T = np.interp(Z, tab['Z'], tab['T'])
        Y = np.array([np.interp(Z, tab['Z'], tab['Y'][i]) for i in range(6)])
        rho = np.interp(Z, tab['Z'], tab['rho']); mu = np.interp(Z, tab['Z'], tab['mu'])
        # flame-sheet heat release ~ rho chi at Zst, chi = 2 D |grad Z|^2 (rho D = mu, Sc = 1)
        dZr = np.gradient(Z, rc, axis=2); dZz = np.gradient(Z, zc, axis=1)
        dth = 2 * np.pi / Nt
        dZt = (np.roll(Z, -1, 0) - np.roll(Z, 1, 0)) / (2 * dth) / rc[None, None, :]
        chi = mu / rho * (dZr ** 2 + dZz ** 2 + dZt ** 2)
        rate = rho * chi * np.exp(-0.5 * ((Z - ZST) / 0.015) ** 2)
    else:
        Y5 = np.clip(np.array([f[s] for s in mech.SPECIES[:5]]), 0, 1)
        Y = mech.full_Y(Y5); T = mech.T_from_hs(f['hs'], Y)
        Z = np.clip((Y[0] * 144.13/170.34 + Y[2] * 12.01/28.01 + Y[3] * 12.01/44.01) / (144.13/170.34), 0, 1)
        rho = mech.density(T, Y); rate = np.clip(mech.rates(T, Y, rho)[0], 0, None)
    ZZ = zc[None, :, None]
    soot = Y[2] * np.clip((T - 1300) / 400, 0, 1) * np.clip((Z - 0.8 * ZST) / (2 * ZST), 0, 1) \
        * np.exp(-Y[1] / 0.01) * np.clip((ZZ - 6e-3) / 12e-3, 0, 1) ** 1.3
    # onto the regular Blender (z, r) grid, per angle (bilinear in (r, z))
    ir = np.clip(np.searchsorted(rc, RG) - 1, 0, len(rc) - 2); wr = np.clip((RG - rc[ir]) / (rc[ir + 1] - rc[ir]), 0, 1)
    iz = np.clip(np.searchsorted(zc, ZG) - 1, 0, len(zc) - 2); wz = np.clip((ZG - zc[iz]) / (zc[iz + 1] - zc[iz]), 0, 1)
    def onto(q):
        a = q[:, iz][:, :, ir] * (1 - wr) + q[:, iz][:, :, ir + 1] * wr
        b = q[:, iz + 1][:, :, ir] * (1 - wr) + q[:, iz + 1][:, :, ir + 1] * wr
        return a * (1 - wz)[None, :, None] + b * wz[None, :, None]
    Tg, Sg, Rg = onto(T), onto(soot), onto(rate)
    wick = (RG[None, :] < 0.75e-3) & (ZG[:, None] < 6e-3)
    Sg[:, wick] = 0; Rg[:, wick] = 0
    idx = np.clip(np.searchsorted(TTAB, Tg), 0, len(TTAB) - 1)
    soot_rgb = np.clip(np.einsum('ij,kzrj->kzri', MRGB, Sg[..., None] * SOOT_TAB[idx]), 0, None)
    chem_rgb = np.clip(np.einsum('ij,kzrj->kzri', MRGB, Rg[..., None] * CHEMI_XYZ), 0, None)
    return float(d['t']), Nt, soot_rgb, chem_rgb


def light_data(soot_rgb, Nt):
    """soot luminosity (relative units) and its centroid (m) — drives the scene's point light."""
    L = soot_rgb[..., 1]                                   # (Nt, nz, nr)
    th = 2 * np.pi * np.arange(Nt) / Nt
    w = L * RG[None, None, :]                              # volume weight r dr dz dth
    tot = w.sum()
    if tot <= 0:
        return 0.0, (0.0, 0.0, 0.03)
    x = (w * RG[None, None, :] * np.cos(th)[:, None, None]).sum() / tot
    y = (w * RG[None, None, :] * np.sin(th)[:, None, None]).sum() / tot
    z = (w * ZG[None, :, None]).sum() / tot
    return float(tot), (float(x), float(y), float(z))


if __name__ == '__main__':
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    from stage3_flamesheet import state_table
    tab = state_table()
    files = sorted(glob.glob(os.path.join(src, '*.npz')))
    norm = None
    meta = []
    for k, f in enumerate(files):
        t, Nt, s, c = emission(f, tab)
        if norm is None:
            norm = (s[..., 1].max(), c[..., 1].max())
        lum, cen = light_data(s, Nt)
        # atlas: tiles side by side along r -> (nz, Nt*nr, 3)
        atlas = lambda a: np.concatenate(list(a), axis=1).astype(np.float32)
        np.save(os.path.join(out, f'soot_{k:04d}.npy'), atlas(s / norm[0]))
        np.save(os.path.join(out, f'chem_{k:04d}.npy'), atlas(c / norm[1]))
        meta.append(dict(k=k, t=t, Nt=Nt, lum=lum, centroid=cen))
        print(f'{k}: t={t:.4f} lum {lum:.3e} centroid ({cen[0]*1e3:.2f}, {cen[1]*1e3:.2f}, {cen[2]*1e3:.1f}) mm', flush=True)
    lum0 = meta[0]['lum'] or 1.0
    for m in meta:
        m['lum_rel'] = m['lum'] / lum0
    json.dump(dict(Nt=meta[0]['Nt'], nr=len(RG), nz=len(ZG), RM=RM, ZM=ZM, frames=meta), open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    print('wrote', len(meta), 'frames to', out)
