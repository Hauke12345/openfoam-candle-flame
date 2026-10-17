"""Stage-6 renderer: line-of-sight image of the quasi-3D candle flame from a saved Fourier state.

Emission model copied from ../render.py (optically thin):
  * soot proxy ~ Y_CO * ramp(T) * ramp(Z) * exp(-Y_O2/0.01) * z-growth, blackbody B_lambda(T) with kappa ~ 1/lambda,
  * chemiluminescence (CH* 431, C2* 473/516/563, CO2* continuum) ~ fuel burning rate (mech.rates, stage 2/4) or
    a Gaussian in (Z - Zst) weighted by T (stage 3), through CIE 1931 (Wyman fit) -> linear sRGB -> tone map.
Fields: f(r,th,z) = sum_{m=-M..M} f_m e^{i m th}  (f_{-m} = conj f_m). Spectral states are reconstructed on a polar
grid of N_TH angles, nonlinear quantities (T, Y, soot, rate) are evaluated there once, then rays are marched with
bilinear (r,z) x periodic linear (theta) interpolation (numba, parallel). Draft blows along +x (theta = 0).
Views: 'side' (camera along +y, image x horizontal, lean visible), 'front' (camera along +x, image y horizontal).
"""
import os, sys, glob, subprocess, shutil
import numpy as np
import numba as nb
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mech

ZST = 0.0621
N_TH = 64

# ---------------- colour science (copied from render.py) ----------------
lam = np.arange(380, 781, 5.0)


def _g(x, mu, s1, s2):
    return np.exp(-0.5 * ((x - mu) / np.where(x < mu, s1, s2)) ** 2)


CMF = np.stack([1.056 * _g(lam, 599.8, 37.9, 31.0) + 0.362 * _g(lam, 442.0, 16.0, 26.7) - 0.065 * _g(lam, 501.1, 20.4, 26.2),
                0.821 * _g(lam, 568.8, 46.9, 40.5) + 0.286 * _g(lam, 530.9, 16.3, 31.1),
                1.217 * _g(lam, 437.0, 11.8, 36.0) + 0.681 * _g(lam, 459.0, 26.0, 13.8)])
M_XYZ2RGB = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])


def planck(lam_nm, T):
    l = lam_nm * 1e-9
    return 1.191e-16 / l ** 5 / np.expm1(1.4388e-2 / (l * T))


TTAB = np.linspace(300, 2600, 400)
SOOT_XYZ = np.array([CMF @ (planck(lam, T) / lam) for T in TTAB])      # kappa ~ 1/lambda
SOOT_XYZ /= SOOT_XYZ[:, 1].max()


def _line(mu, w):
    return np.exp(-0.5 * ((lam - mu) / w) ** 2)


_chemi = 1.0 * _line(431.4, 3) + 0.35 * _line(473.7, 4) + 0.55 * _line(516.5, 4) + 0.25 * _line(563.5, 4) \
    + 0.30 * np.exp(-0.5 * ((lam - 440) / 45) ** 2)
CHEMI_XYZ = CMF @ _chemi
CHEMI_XYZ /= CHEMI_XYZ[1]


def tonemap(rgb, k):
    Y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    Yt = 1 - np.exp(-k * Y)
    out = rgb * (Yt / np.maximum(Y, 1e-9))[..., None]
    w = np.clip((Yt - 0.80) / 0.2, 0, 1)[..., None] ** 2
    return out * (1 - w) + w * Yt[..., None]


# ---------------- state loading / evaluation ----------------
_TAB = None


def _table():
    global _TAB
    if _TAB is None:
        from stage3_flamesheet import state_table
        _TAB = state_table()
    return _TAB


def load(path):
    """Read a Solver.save npz into a dict (arrays + derived cell-centre coordinates)."""
    d = np.load(path)
    s = dict(t=float(d['t']), rf=d['rf'], zf=d['zf'], M=int(d['M']), us=d['us'], ps=d['ps'], phis=d['phis'],
             scalars=[str(x) for x in d['scalars']], theta_fv=bool(d['theta_fv']) if 'theta_fv' in d else False)
    s['rc'] = 0.5 * (s['rf'][1:] + s['rf'][:-1])
    s['zc'] = 0.5 * (s['zf'][1:] + s['zf'][:-1])
    return s


def eval_scalar(state, k, theta):
    """Scalar k at angles theta (array, any shape): returns (..., Nz, Nr) real, from the Fourier series
    f = f_0 + 2 Re sum_{m>=1} f_m e^{i m th}."""
    c = state['phis'][k]                                   # (M+1, Nz, Nr)
    th = np.asarray(theta, float).ravel()
    if state.get('theta_fv') and c.shape[0] > 1:
        # THETA_FV scalars: values at Nt cells (all Nt//2+1 modes stored); interpolate linearly in theta between
        # the cell values instead of the trigonometric interpolant (which would ring between the cells)
        Nt = 2 * (c.shape[0] - 1)
        cc = np.zeros((Nt // 2 + 1,) + c.shape[1:], complex); cc[:c.shape[0]] = c
        v = np.fft.irfft(cc, n=Nt, axis=0) * Nt
        x = (th % (2 * np.pi)) / (2 * np.pi / Nt)
        i0 = np.floor(x).astype(int) % Nt; w = (x - np.floor(x))[:, None, None]
        out = (1 - w) * v[i0] + w * v[(i0 + 1) % Nt]
        return out.reshape(np.shape(theta) + c.shape[1:])
    m = np.arange(c.shape[0])
    E = np.exp(1j * np.outer(th, m)); E[:, 1:] *= 2
    out = np.einsum('tm,mzr->tzr', E, c).real
    return out.reshape(np.shape(theta) + c.shape[1:])


def physical(state, nth=N_TH):
    """Nonlinear emission fields on the polar grid theta_j = 2 pi j / nth: soot proxy s, T, rate R, each (nth,Nz,Nr)."""
    th = 2 * np.pi * np.arange(nth) / nth
    sc = state['scalars']; rc, zc = state['rc'], state['zc']
    RR, ZZ = np.meshgrid(rc, zc)
    if sc[0] == 'Z':
        tab = _table()
        Z = np.clip(eval_scalar(state, 0, th), 0, 1)
        T = np.interp(Z, tab['Z'], tab['T'])
        Y = np.array([np.interp(Z, tab['Z'], tab['Y'][i]) for i in range(6)])
        rate = np.exp(-0.5 * ((Z - ZST) / 0.012) ** 2) * np.clip((T - 1000) / 800, 0, 1)
    else:
        hs = eval_scalar(state, 0, th)
        Y5 = np.clip(np.array([eval_scalar(state, i, th) for i in range(1, 6)]), 0, 1)
        Y = mech.full_Y(Y5)
        T = mech.T_from_hs(hs, Y)
        Z = np.clip(mix_fraction(Y), 0, 1)
        rho = mech.density(T, Y)
        rate = np.clip(mech.rates(T, Y, rho)[0], 0, None)
    soot = Y[2] * np.clip((T - 1300) / 400, 0, 1) * np.clip((Z - 0.8 * ZST) / (2 * ZST), 0, 1) \
        * np.exp(-Y[1] / 0.01) * np.clip((ZZ - 6e-3) / 12e-3, 0, 1) ** 1.3
    wick = (RR < 0.75e-3) & (ZZ < 6e-3)
    soot[:, wick] = 0; rate[:, wick] = 0; T[:, wick] = 600
    return soot, T, rate


def mix_fraction(Y):
    from stage3_flamesheet import mixture_fraction
    return mixture_fraction(Y)


# ---------------- ray marching ----------------
@nb.njit(parallel=True, fastmath=True, cache=True)
def _march(F, ridx, dri, rmax, rext, zidx, hpos, e_h, e_d, s0, ds, ns, nth, xyz_tab, tmin, dT):
    """F (3,nth,Nz,Nr) float32 [soot, T, rate]; ridx lookup r -> fractional r index; zidx (H) fractional z index per row;
    hpos (W) horizontal coordinate. Point = hpos*e_h + s*e_d in (x,y). Returns (H,W,4): soot XYZ + chemi."""
    H = zidx.shape[0]; W = hpos.shape[0]
    Nz = F.shape[2]; Nr = F.shape[3]
    out = np.zeros((H, W, 4), np.float32)
    for iy in nb.prange(H):
        zf = zidx[iy]
        if zf < 0 or zf > Nz - 1:
            continue
        j0 = int(zf); j1 = min(j0 + 1, Nz - 1); wz = zf - j0
        for ix in range(W):
            a0 = 0.0; a1 = 0.0; a2 = 0.0; a3 = 0.0
            hr2 = rext[iy] ** 2 - hpos[ix] ** 2
            if hr2 <= 0:
                continue
            sh = np.sqrt(hr2)                 # ray only inside |s| < sh (emitting radius at this row)
            k0 = max(0, int((-sh - s0) / ds)); k1 = min(ns, int((sh - s0) / ds) + 1)
            for k in range(k0, k1):
                s = s0 + (k + 0.5) * ds
                x = hpos[ix] * e_h[0] + s * e_d[0]; y = hpos[ix] * e_h[1] + s * e_d[1]
                r = np.sqrt(x * x + y * y)
                if r >= rmax:
                    continue
                q = r / dri; q0 = int(q); wq = q - q0
                rf = ridx[q0] * (1 - wq) + ridx[q0 + 1] * wq
                i0 = int(rf); i1 = min(i0 + 1, Nr - 1); wr = rf - i0
                tf = np.arctan2(y, x) / (2 * np.pi) * nth
                if tf < 0:
                    tf += nth
                t0 = int(tf); wt = tf - t0; t0 = t0 % nth; t1 = (t0 + 1) % nth
                w00 = (1 - wr) * (1 - wz); w01 = wr * (1 - wz); w10 = (1 - wr) * wz; w11 = wr * wz
                v = np.zeros(3)
                for c in range(3):
                    va = F[c, t0, j0, i0] * w00 + F[c, t0, j0, i1] * w01 + F[c, t0, j1, i0] * w10 + F[c, t0, j1, i1] * w11
                    vb = F[c, t1, j0, i0] * w00 + F[c, t1, j0, i1] * w01 + F[c, t1, j1, i0] * w10 + F[c, t1, j1, i1] * w11
                    v[c] = va * (1 - wt) + vb * wt
                if v[0] > 0:
                    ti = (v[1] - tmin) / dT
                    ti = min(max(ti, 0.0), xyz_tab.shape[0] - 1.001)
                    ii = int(ti); wi = ti - ii
                    a0 += v[0] * (xyz_tab[ii, 0] * (1 - wi) + xyz_tab[ii + 1, 0] * wi)
                    a1 += v[0] * (xyz_tab[ii, 1] * (1 - wi) + xyz_tab[ii + 1, 1] * wi)
                    a2 += v[0] * (xyz_tab[ii, 2] * (1 - wi) + xyz_tab[ii + 1, 2] * wi)
                a3 += max(v[2], 0.0)
            out[iy, ix, 0] = a0 * ds; out[iy, ix, 1] = a1 * ds; out[iy, ix, 2] = a2 * ds; out[iy, ix, 3] = a3 * ds
    return out


def _frac_index(centres, pts):
    return np.interp(pts, centres, np.arange(len(centres)))


def render(state, view='side', W=400, H=None, xrange=None, zrange=(-10, 70), depth=25, ds=0.15,
           norm=None, return_norm=False, nth=N_TH):
    """Render RGB uint8 (H, W, 3). Lengths in mm. xrange: horizontal image range (x for 'side', y for 'front').
    depth: integrate +-depth along the line of sight. norm: (soot, chemi) intensity scales (default: 99.5th percentile
    of this frame; pass the first frame's norm for movies so brightness is comparable)."""
    if xrange is None:
        xrange = (-15, 25) if view == 'side' else (-20, 20)
    px = (xrange[1] - xrange[0]) / W
    if H is None:
        H = int(round((zrange[1] - zrange[0]) / px)); H += H % 2
    soot, T, rate = physical(state, nth)
    F = np.ascontiguousarray(np.stack([soot, T, rate]).astype(np.float32))
    rc = state['rc']
    dri = 0.02e-3; rmax = min(state['rf'][-1], depth * 1e-3 * 1.6 + 25e-3)
    rq = np.arange(0, rmax + 3 * dri, dri)
    ridx = _frac_index(rc, rq)
    # emitting radius per z cell (threshold 1e-4 of max), for skipping empty ray segments
    ER = (soot > 1e-4 * soot.max()) | (rate > 1e-4 * max(rate.max(), 1e-300))
    rcell = np.where(ER.any(0), rc[None, :], 0).max(1) + 2e-3                  # (Nz,)
    rcell = np.maximum.reduce([rcell, np.roll(rcell, 1), np.roll(rcell, -1)])
    hpos = (xrange[0] + (np.arange(W) + 0.5) * px) * 1e-3
    zmm = zrange[1] - (np.arange(H) + 0.5) * px
    zidx = _frac_index(state['zc'], zmm * 1e-3)
    zidx[(zmm < 0) | (zmm * 1e-3 > state['zf'][-1])] = -1
    if view == 'side':
        e_h, e_d = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    elif view == 'front':
        e_h, e_d = np.array([0.0, 1.0]), np.array([1.0, 0.0])
    else:
        raise ValueError(view)
    ns = int(round(2 * depth / ds))
    rext = np.minimum(np.interp(np.clip(zidx, 0, None), np.arange(len(rcell)), rcell), rmax)
    res = _march(F, ridx, dri, rmax, rext, zidx, hpos, e_h, e_d, -depth * 1e-3, ds * 1e-3, ns, nth, SOOT_XYZ, TTAB[0],
                 TTAB[1] - TTAB[0])
    Is = res[..., :3].astype(float); Ic = res[..., 3].astype(float)
    if norm is None:
        norm = (max(np.percentile(Is[..., 1], 99.5), 1e-30), max(np.percentile(Ic, 99.5), 1e-30))
    Is = Is / norm[0]; Ic = Ic[..., None] / norm[1] * CHEMI_XYZ
    XYZ = 1.0 * Is + 0.11 * Ic
    E = np.clip(np.einsum('ij,zxj->zxi', M_XYZ2RGB, XYZ), 0, None)
    img = _compose(E, px, xrange, zrange, view)
    img = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    return (img, norm) if return_norm else img


def _compose(E, px, xrange, zrange, view):
    """Scene: background, glow, candle body below z=0, flame (bloom + core), wick. Same look as render.py."""
    ny, nx = E.shape[:2]
    X = xrange[0] + (np.arange(nx) + 0.5) * px; Zmm = zrange[1] - (np.arange(ny) + 0.5) * px
    X, Zmm = np.meshgrid(X, Zmm)
    Hb, Ht = zrange
    img = np.zeros((ny, nx, 3))
    tt = ((Zmm - Hb) / (Ht - Hb))[..., None]
    img[:] = np.array([0.010, 0.007, 0.006]) * (1 - tt) + np.array([0.004, 0.006, 0.016]) * tt
    sig = 220 * 0.05 / px
    glow = np.stack([gaussian_filter(E[..., c], sig) for c in range(3)], -1)
    glow /= max(glow.max(), 1e-9)
    img += 0.10 * glow * np.array([1.0, 0.72, 0.42])
    Rc = 10.0
    u = np.clip(X / Rc, -1, 1)
    cyl = np.sqrt(np.clip(1 - u ** 2, 0, 1))
    red = np.array([0.42, 0.018, 0.028])
    lightF = 0.25 + 1.8 * np.exp(Zmm / 16.0)
    col = red * (0.25 + 0.75 * cyl ** 0.8)[..., None] * lightF[..., None]
    col += (np.exp(Zmm / 5.0) * np.exp(-(X / 6.5) ** 2))[..., None] * np.array([0.85, 0.22, 0.05])
    col += (0.10 * np.exp(-((u + 0.45) / 0.14) ** 2) * (0.4 + 0.6 * np.exp(Zmm / 25)))[..., None] * np.array([1, 0.55, 0.5])
    body = (np.abs(X) <= Rc) & (Zmm <= 0)
    img[body] = col[body]
    pool = ((X / Rc) ** 2 + (Zmm / 1.6) ** 2 < 1) & (Zmm > -1.6) & (view == 'side')
    img[pool] = np.array([1.0, 0.42, 0.20]) * (0.9 + 0.6 * np.exp(-(X[pool] / 5) ** 2))[..., None]
    bloom = np.stack([gaussian_filter(E[..., c], 14 * 0.05 / px) for c in range(3)], -1)
    img += tonemap(E, 2.3) + 0.30 * tonemap(bloom, 2.3)
    # wick (r < 0.75 mm, z < 6 mm): dark core with ember tip
    wk = (np.abs(X) < 0.7) & (Zmm > -0.5) & (Zmm < 6.0)
    img[wk] = 0.08 * img[wk] + np.array([0.012, 0.010, 0.009])
    img[(X ** 2 + (Zmm - 6.0) ** 2) < 0.85 ** 2] = [1.0, 0.45, 0.12]
    img[(X ** 2 + (Zmm - 6.6) ** 2) < 0.45 ** 2] = [1.0, 0.75, 0.4]
    img = np.clip(img, 0, None)
    return np.where(img <= 0.0031308, 12.92 * img, 1.055 * np.power(img, 1 / 2.4) - 0.055)


# ---------------- output ----------------
def find_ffmpeg():
    for c in [shutil.which('ffmpeg'), '/opt/mm/root/envs/of/bin/ffmpeg']:
        if c and os.path.exists(c):
            return c
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'imageio-ffmpeg'])
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()


def save_png(img, path):
    import matplotlib.image as mi
    mi.imsave(path, img)


def movie(snapdir, out, view='side', fps=30, **kw):
    files = sorted(glob.glob(os.path.join(snapdir, '*.npz')))
    assert files, 'no npz in ' + snapdir
    ff = find_ffmpeg(); print('ffmpeg:', ff)
    norm = None; proc = None
    smooth = os.environ.get('NORM', 'fixed') == 'smooth'
    for i, f in enumerate(files):
        if smooth:
            # per-frame exposure, smoothed in time (log-space exponential average of the auto normalisation)
            st = load(f)
            _, n_auto = render(st, view, return_norm=True, **kw)
            norm = n_auto if norm is None else tuple(np.exp(0.6 * np.log(a) + 0.4 * np.log(b)) for a, b in zip(norm, n_auto))
            img = render(st, view, norm=norm, **kw)
        else:
            img, norm = render(load(f), view, norm=norm, return_norm=True, **kw)   # brightness fixed by the first frame
        h, w = img.shape[:2]
        if proc is None:
            assert w % 2 == 0 and h % 2 == 0
            proc = subprocess.Popen([ff, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
                                     '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', out],
                                    stdin=subprocess.PIPE)
        proc.stdin.write(img.tobytes())
        print(f'frame {i + 1}/{len(files)}', end='\r')
    proc.stdin.close(); proc.wait()
    print('\nwrote', out)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a and a[0] == '--movie':
        xr = tuple(float(v) for v in os.environ['XRANGE'].split(',')) if 'XRANGE' in os.environ else None
        movie(a[1], a[2], a[3] if len(a) > 3 else 'side', fps=int(os.environ.get('FPS', 30)), xrange=xr)
    else:
        img = render(load(a[0]), a[2] if len(a) > 2 else 'side')
        save_png(img, a[1]); print('saved', a[1], img.shape)
