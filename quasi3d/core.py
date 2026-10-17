"""Quasi-3D (azimuthal Fourier) low-Mach solver core.

Fields f(r, z, theta) = sum_{m=-M..M} f_m(r, z) e^{i m theta}, real => f_{-m} = conj(f_m); only m >= 0 stored.

Discretisation
  * (r, z): cell-centred finite volumes, r_c = (i + 1/2) dr near the axis (no node on the axis), stretched outside.
    Cell volume per radian = r_c dr dz, so  L f = [r_f g]_{i-1/2}^{i+1/2} / (r_c dr) + [g]_{j-1/2}^{j+1/2} / dz - m^2/r^2 f.
  * theta: pseudo-spectral, products on N_theta >= 3M+1 angles, truncated to M after every transform (dealiased).
  * axis: zero geometric flux (r_f = 0); ghost values across the axis from parity f(-r, th) = s f(r, th + pi),
    s = +1 for scalars and u_z, s = -1 for u_r and u_theta.
  * mass fluxes F = rho u live on r- and z-faces (Rhie-Chow), the theta mass flux rho u_theta at cell centres.
Time step (low-Mach, momentum-based projection)
  1. scalars: rho Dphi/Dt = div(Gamma grad phi) + S, advective form with the face fluxes F^n, AB2 advection,
     implicit constant-coefficient diffusion D0 L_m + explicit (variable - D0) remainder, stiff sources split.
  2. rho^{n+1} from the equation of state.
  3. momentum: rho Du/Dt = -grad p + div tau + (rho - rho_inf) g; implicit nu0 vector Laplacian via
     u_pm = u_r +- i u_theta, which decouple into scalar operators L_{m+-1}; explicit remainder.
  4. projection on rho u:  L_m dp = (div F* + (rho^{n+1} - rho^n)/dt) / dt, a constant-coefficient Poisson
     equation per mode (factorised once with splu). Exact discrete continuity for F^{n+1}.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# face labels
INT, AXIS, OUTER, TOP, POOL, OPEN, WICK, OFF = range(8)
P0, RU = 101325.0, 8314.46
G_Z = -9.81


def stretched(L, h, x_u, growth=1.1):
    """Faces on [0, L]: uniform spacing h up to x_u, then geometric growth (rescaled to end exactly at L)."""
    n = int(round(x_u / h))
    f = list(np.arange(n + 1) * h)
    d = h
    while f[-1] < L - 1e-12:
        d *= growth
        f.append(f[-1] + d)
    f = np.array(f)
    if len(f) > n + 2 and (f[-1] - L) > 0.5 * (f[-1] - f[-2]):
        f = f[:-1]
    f[n:] = f[n] + (f[n:] - f[n]) * (L - f[n]) / (f[-1] - f[n])
    return f


class Grid:
    """Mesh, Fourier transforms and face bookkeeping. wick=(r_w, z_w) makes r<r_w, z<z_w solid;
    pool_r: bottom faces with r < pool_r are a wall (POOL), the rest OPEN."""

    def __init__(self, rf, zf, M, wick=None, pool_r=0.0, Nt=None):
        self.rf, self.zf = np.asarray(rf, float), np.asarray(zf, float)
        self.rc = 0.5 * (self.rf[1:] + self.rf[:-1])
        self.zc = 0.5 * (self.zf[1:] + self.zf[:-1])
        self.dr, self.dz = np.diff(self.rf), np.diff(self.zf)
        self.Nr, self.Nz = len(self.rc), len(self.zc)
        Nr, Nz = self.Nr, self.Nz
        self.M = M
        self.Nt = Nt or (1 if M == 0 else int(2 * np.ceil((3 * M + 1) / 2)))
        self.m = np.arange(M + 1)
        self.mm = self.m[:, None, None]
        self.theta = 2 * np.pi * np.arange(self.Nt) / self.Nt
        # face interpolation weights (weight of the low-index cell) and centre distances
        self.wr = (self.rc[1:] - self.rf[1:-1]) / (self.rc[1:] - self.rc[:-1])
        self.wz = (self.zc[1:] - self.zf[1:-1]) / (self.zc[1:] - self.zc[:-1])
        self.drc = np.diff(self.rc)
        self.dzc = np.diff(self.zc)
        # solid region and labels
        self.solid = np.zeros((Nz, Nr), bool)
        lr = np.full((Nz, Nr + 1), INT)
        lr[:, 0] = AXIS
        lr[:, -1] = OUTER
        lz = np.full((Nz + 1, Nr), INT)
        lz[0, :] = OPEN
        lz[0, self.rc < pool_r] = POOL
        lz[-1, :] = TOP
        self.iw = self.jw = 0
        if wick is not None:
            iw = int(np.argmin(abs(self.rf - wick[0])))
            jw = int(np.argmin(abs(self.zf - wick[1])))
            assert abs(self.rf[iw] - wick[0]) < 1e-9 and abs(self.zf[jw] - wick[1]) < 1e-9, 'grid must align with wick'
            self.iw, self.jw = iw, jw
            self.solid[:jw, :iw] = True
            lr[:jw, :iw] = OFF
            lr[:jw, iw] = WICK
            lz[:jw, :iw] = OFF
            lz[jw, :iw] = WICK
        self.fluid = ~self.solid
        self.lr, self.lz = lr, lz
        # boundary face lists: (j, i) of the face, fluid cell (jc, ic), sign (+1 if fluid on the low side), distance
        self.bfaces = {}
        for fam, lab in (('r', lr), ('z', lz)):
            out = {}
            for L in (OUTER, TOP, POOL, OPEN, WICK):
                jj, ii = np.nonzero(lab == L)
                if len(jj) == 0:
                    continue
                if fam == 'r':
                    low = (ii >= 1) & (ii - 1 < Nr)
                    low = low & np.array([ii_ - 1 >= 0 and not self.solid[j_, ii_ - 1] for j_, ii_ in zip(jj, ii)])
                    ic = np.where(low, ii - 1, ii)
                    jc = jj
                    d = np.where(low, self.rf[ii] - self.rc[np.clip(ii - 1, 0, Nr - 1)],
                                 self.rc[np.clip(ii, 0, Nr - 1)] - self.rf[ii])
                else:
                    low = np.array([j_ - 1 >= 0 and not self.solid[j_ - 1, i_] for j_, i_ in zip(jj, ii)])
                    jc = np.where(low, jj - 1, jj)
                    ic = ii
                    d = np.where(low, self.zf[jj] - self.zc[np.clip(jj - 1, 0, Nz - 1)],
                                 self.zc[np.clip(jj, 0, Nz - 1)] - self.zf[jj])
                out[L] = dict(j=jj, i=ii, jc=jc, ic=ic, s=np.where(low, 1.0, -1.0), d=d)
            self.bfaces[fam] = out
        self.vol = self.rc[None, :] * self.dr[None, :] * self.dz[:, None]   # per radian
        self.ncell = Nr * Nz
        self._A = {}

    # ---------- Fourier ----------
    def phys(self, fs):
        """physical values at the Nt angles of a spectrum with M+1 modes, or with all Nt//2+1 modes (full-resolution
        scalars of the THETA_FV option, see spec_full)."""
        if self.M == 0:
            return fs.real.copy()
        c = np.zeros((self.Nt // 2 + 1,) + fs.shape[1:], complex)
        c[:fs.shape[0]] = fs
        return np.fft.irfft(c, n=self.Nt, axis=0) * self.Nt

    def spec(self, fp):
        if self.M == 0:
            return fp.astype(complex)
        s = np.fft.rfft(fp, axis=0)[:self.M + 1] / self.Nt
        s[0] = s[0].real
        return s

    def spec_full(self, fp):
        """all Nt//2+1 modes: an exact (invertible) representation of the values at the Nt collocation angles."""
        if self.M == 0:
            return fp.astype(complex)
        s = np.fft.rfft(fp, axis=0) / self.Nt
        s[0] = s[0].real
        if self.Nt % 2 == 0:
            s[-1] = s[-1].real
        return s

    def trunc(self, fp):
        return self.phys(self.spec(fp))

    def dth(self, fp=None, fs=None):
        """d/dtheta of a physical field (or of its spectrum fs), returned physical."""
        if fs is None:
            fs = self.spec(fp)
        m = np.arange(fs.shape[0]).reshape((-1,) + (1,) * (fs.ndim - 1))
        if fs.shape[0] == self.Nt // 2 + 1 and self.Nt % 2 == 0 and self.M > 0:
            m = np.where(m == self.Nt // 2, 0, m)      # derivative of the Nyquist mode is undefined: drop it
        return self.phys(1j * m * fs)

    def axis_ghost(self, f0, s):
        """value at r = -r_c[0], theta: s * f(r_c[0], theta + pi). f0: (Nt, Nz)."""
        return s * np.roll(f0, self.Nt // 2, axis=0)

    # ---------- per-mode sparse operators ----------
    def lap_matrix(self, k, dlabels):
        """FV operator A for L_k with zero-flux faces except Dirichlet labels (homogeneous part); returns
        (A csr, Bmat) where L f = A f + Bmat @ b, b = values on the Dirichlet faces in self.dfaces(dlabels) order."""
        key = (k, tuple(sorted(dlabels)))
        if key in self._A:
            return self._A[key]
        Nr, Nz = self.Nr, self.Nz
        idx = np.arange(self.ncell).reshape(Nz, Nr)
        rows, cols, vals = [], [], []
        # interior r faces i=1..Nr-1 between cells i-1 and i
        fl = self.fluid
        both = fl[:, :-1] & fl[:, 1:]
        jj, ii = np.nonzero(both)              # ii: low cell index, face = ii+1
        a = self.rf[ii + 1] / self.drc[ii]
        for (cL, cR, rcL, rcR, drL, drR) in [(idx[jj, ii], idx[jj, ii + 1], self.rc[ii], self.rc[ii + 1], self.dr[ii], self.dr[ii + 1])]:
            aL = a / (rcL * drL); aR = a / (rcR * drR)
            rows += [cL, cL, cR, cR]; cols += [cR, cL, cL, cR]; vals += [aL, -aL, aR, -aR]
        both = fl[:-1, :] & fl[1:, :]
        jj, ii = np.nonzero(both)
        a = 1.0 / self.dzc[jj]
        cL, cR = idx[jj, ii], idx[jj + 1, ii]
        aL = a / self.dz[jj]; aR = a / self.dz[jj + 1]
        rows += [cL, cL, cR, cR]; cols += [cR, cL, cL, cR]; vals += [aL, -aL, aR, -aR]
        # -k^2/r^2
        jj, ii = np.nonzero(fl)
        rows.append(idx[jj, ii]); cols.append(idx[jj, ii]); vals.append(-(k ** 2) / self.rc[ii] ** 2)
        # Dirichlet faces
        brow, bcol, bval = [], [], []
        off = 0
        for fam, L, bf in self.dfaces(dlabels):
            if fam == 'r':
                a = self.rf[bf['i']] / (bf['d'] * self.rc[bf['ic']] * self.dr[bf['ic']])
            else:
                a = 1.0 / (bf['d'] * self.dz[bf['jc']])
            c = idx[bf['jc'], bf['ic']]
            rows.append(c); cols.append(c); vals.append(-a)
            brow.append(c); bcol.append(off + np.arange(len(c))); bval.append(a)
            off += len(c)
        A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(self.ncell,) * 2)
        if off:
            B = sp.csr_matrix((np.concatenate(bval), (np.concatenate(brow), np.concatenate(bcol))), shape=(self.ncell, off))
        else:
            B = sp.csr_matrix((self.ncell, 0))
        self._A[key] = (A, B)
        return A, B

    def dfaces(self, dlabels):
        for fam in ('r', 'z'):
            for L in sorted(dlabels):
                if L in self.bfaces[fam]:
                    yield fam, L, self.bfaces[fam][L]

    def gather_d(self, dlabels, vr, vz):
        """Values on Dirichlet faces (in dfaces order) from full face arrays vr (..., Nz, Nr+1), vz (..., Nz+1, Nr)."""
        parts = []
        for fam, L, bf in self.dfaces(dlabels):
            v = vr if fam == 'r' else vz
            parts.append(v[..., bf['j'], bf['i']])
        if not parts:
            return np.zeros(vr.shape[:-2] + (0,), vr.dtype)
        return np.concatenate(parts, axis=-1)


class BC:
    """Boundary data of one cell field: Dirichlet labels, parity across the axis, face values (physical,
    used on Dirichlet faces and as inflow values for advection), value used inside solid cells."""

    def __init__(self, dlabels, parity=1.0, vr=None, vz=None, solid_fill=0.0, inflow_labels=()):
        self.d = set(dlabels)
        self.s = parity
        self.vr, self.vz = vr, vz
        self.solid_fill = solid_fill
        self.inflow = set(inflow_labels)     # Neumann faces where advection takes the face value on inflow


def faces(g, f, bc, extrap_n=False):
    """Face values and outward-normal (+r / +z) gradients of a physical cell field f (Nt, Nz, Nr).
    Neumann faces: value = adjacent cell, gradient 0 (or, if extrap_n, the gradient of the opposite face)."""
    Nt = f.shape[0]
    fr = np.zeros((Nt, g.Nz, g.Nr + 1)); gr = np.zeros_like(fr)
    fz = np.zeros((Nt, g.Nz + 1, g.Nr)); gz = np.zeros_like(fz)
    fr[..., 1:-1] = g.wr * f[..., :-1] + (1 - g.wr) * f[..., 1:]
    gr[..., 1:-1] = (f[..., 1:] - f[..., :-1]) / g.drc
    gh = g.axis_ghost(f[..., 0], bc.s)
    fr[..., 0] = 0.5 * (f[..., 0] + gh)
    gr[..., 0] = (f[..., 0] - gh) / (2 * g.rc[0])
    fz[:, 1:-1] = g.wz[:, None] * f[:, :-1] + (1 - g.wz[:, None]) * f[:, 1:]
    gz[:, 1:-1] = (f[:, 1:] - f[:, :-1]) / g.dzc[:, None]
    for fam, (fv, gv, vv) in (('r', (fr, gr, bc.vr)), ('z', (fz, gz, bc.vz))):
        for L, bf in g.bfaces[fam].items():
            j, i, jc, ic = bf['j'], bf['i'], bf['jc'], bf['ic']
            fc = f[:, jc, ic]
            if L in bc.d:
                b = vv[:, j, i]
                fv[:, j, i] = b
                gv[:, j, i] = bf['s'] * (b - fc) / bf['d']
            else:
                fv[:, j, i] = fc
                if extrap_n:
                    if fam == 'r':
                        io = np.where(bf['s'] > 0, i - 1, i + 1)
                        gv[:, j, i] = gv[:, j, io]
                    else:
                        jo = np.where(bf['s'] > 0, j - 1, j + 1)
                        gv[:, j, i] = gv[:, jo, i]
                else:
                    gv[:, j, i] = 0.0
    if g.jw:   # faces inside the solid
        fr[:, g.lr == OFF] = 0; gr[:, g.lr == OFF] = 0
        fz[:, g.lz == OFF] = 0; gz[:, g.lz == OFF] = 0
    return fr, gr, fz, gz


def interp_faces(g, q, s=1.0):
    """Plain interpolation of a cell field to faces, adjacent-cell value on boundary faces, parity on the axis."""
    Nt = q.shape[0]
    fr = np.empty((Nt, g.Nz, g.Nr + 1)); fz = np.empty((Nt, g.Nz + 1, g.Nr))
    fr[..., 1:-1] = g.wr * q[..., :-1] + (1 - g.wr) * q[..., 1:]
    fr[..., 0] = 0.5 * (q[..., 0] + g.axis_ghost(q[..., 0], s))
    fr[..., -1] = q[..., -1]
    fz[:, 1:-1] = g.wz[:, None] * q[:, :-1] + (1 - g.wz[:, None]) * q[:, 1:]
    fz[:, 0] = q[:, 0]; fz[:, -1] = q[:, -1]
    if g.jw:
        bf = g.bfaces['r'][WICK]; fr[:, bf['j'], bf['i']] = q[:, bf['jc'], bf['ic']]
        bf = g.bfaces['z'][WICK]; fz[:, bf['j'], bf['i']] = q[:, bf['jc'], bf['ic']]
    return fr, fz


def div_faces(g, Fr, Fz):
    """(1/r) d(r F_r)/dr + dF_z/dz per cell (works for physical or spectral leading axis)."""
    return ((g.rf[1:] * Fr[..., 1:] - g.rf[:-1] * Fr[..., :-1]) / (g.rc * g.dr)
            + (Fz[..., 1:, :] - Fz[..., :-1, :]) / g.dz[:, None])


def vanleer(r):
    return (r + np.abs(r)) / (1 + np.abs(r))


def muscl_faces(g, f, Fr, Fz, bc):
    """Upwind-biased (MUSCL, van Leer) face values of f for the mass fluxes Fr, Fz. Boundary faces: Dirichlet
    value, inflow value on inflow-labelled Neumann faces, otherwise the upwind cell."""
    Nt, Nz, Nr = f.shape
    f = f.copy()
    if g.jw:
        f[:, g.solid] = bc.solid_fill
    # r direction: extended array with 2 ghosts low (axis parity), 2 high (copy)
    gh0 = g.axis_ghost(f[..., 0], bc.s); gh1 = g.axis_ghost(f[..., 1], bc.s) if Nr > 1 else gh0
    fe = np.concatenate([gh1[..., None], gh0[..., None], f, f[..., -1:], f[..., -1:]], axis=-1)   # index k = i + 2
    fr = np.empty((Nt, Nz, Nr + 1))
    # face i (between cells i-1 and i): cells k=i+1 (L), i+2 (R)
    L_, R_ = fe[..., 1:Nr + 2], fe[..., 2:Nr + 3]
    LL, RR = fe[..., 0:Nr + 1], fe[..., 3:Nr + 4]
    pos = Fr >= 0
    U = np.where(pos, L_, R_); D = np.where(pos, R_, L_); UU = np.where(pos, LL, RR)
    den = D - U
    rat = np.where(np.abs(den) > 1e-30, (U - UU) / np.where(np.abs(den) > 1e-30, den, 1.0), 0.0)
    fr[:] = U + 0.5 * vanleer(rat) * den
    fe = np.concatenate([f[:, :1], f[:, :1], f, f[:, -1:], f[:, -1:]], axis=1)
    L_, R_ = fe[:, 1:Nz + 2], fe[:, 2:Nz + 3]
    LL, RR = fe[:, 0:Nz + 1], fe[:, 3:Nz + 4]
    pos = Fz >= 0
    U = np.where(pos, L_, R_); D = np.where(pos, R_, L_); UU = np.where(pos, LL, RR)
    den = D - U
    rat = np.where(np.abs(den) > 1e-30, (U - UU) / np.where(np.abs(den) > 1e-30, den, 1.0), 0.0)
    fz = U + 0.5 * vanleer(rat) * den
    for fam, (fv, F, vv) in (('r', (fr, Fr, bc.vr)), ('z', (fz, Fz, bc.vz))):
        for L, bf in g.bfaces[fam].items():
            j, i = bf['j'], bf['i']
            fc = f[:, bf['jc'], bf['ic']]
            if L in bc.d:
                fv[:, j, i] = vv[:, j, i]
            elif L in bc.inflow:
                inflow = bf['s'][None] * F[:, j, i] < 0
                fv[:, j, i] = np.where(inflow, vv[:, j, i], fc)
            else:
                fv[:, j, i] = fc
    return fr, fz


class Factor:
    """Cache of splu factorisations of (I - c A_k) (c=None: A_k itself, solid rows set to identity)."""

    def __init__(self, g):
        self.g = g
        self.cache = {}

    def get(self, k, dlabels, c):
        key = (k, tuple(sorted(dlabels)), c)
        if key not in self.cache:
            A, _ = self.g.lap_matrix(k, dlabels)
            sol = self.g.solid.ravel().astype(float)
            if c is None:
                Mx = A + sp.diags(sol)
            else:
                Mx = sp.eye(self.g.ncell) - c * A
            self.cache[key] = spla.splu(sp.csc_matrix(Mx))
        return self.cache[key]

    def drop(self, pred):
        for k in [k for k in self.cache if pred(k)]:
            del self.cache[k]


def csolve(lu, rhs):
    """Solve a real factorised system for complex right-hand side(s) rhs (..., ncell)."""
    shp = rhs.shape
    R = rhs.reshape(-1, shp[-1])
    X = lu.solve(np.ascontiguousarray(np.concatenate([R.real, R.imag], 0).T))
    n = R.shape[0]
    return (X[:, :n] + 1j * X[:, n:]).T.reshape(shp)


# ---------------------------------------------------------------- finite-volume theta operators (THETA_FV option)
def theta_face_velocity(g, Ut):
    """u_theta at the theta faces theta_k + dth/2 of a (M-mode) cell field Ut, chosen such that the finite-volume
    theta divergence (Utf_{k+1/2} - Utf_{k-1/2}) / dth equals the spectral derivative d Ut/d theta at every
    collocation angle: Utf_m = Ut_m (m dth/2)/sin(m dth/2) e^{i m dth/2} (m <= M < Nt/2, factor <= 1.21). Then the
    FV mass fluxes of the scalars see exactly the divergence that the spectral projection enforces."""
    if g.M == 0:
        return np.zeros_like(Ut)
    dth = 2 * np.pi / g.Nt
    a = 0.5 * g.m * dth
    fac = np.where(g.m > 0, a / np.where(g.m > 0, np.sin(a), 1.0), 1.0) * np.exp(1j * a)
    return g.phys(g.spec(Ut) * fac.reshape((-1,) + (1,) * (Ut.ndim - 1)))


def muscl_theta(f, U):
    """MUSCL (van Leer) face values at theta_k + dth/2 (index k) of a periodic cell field f (axis 0), upwind by
    the sign of the face velocity U (same shape)."""
    fp = np.roll(f, -1, 0)                 # f_{k+1}
    pos = U >= 0
    Uc = np.where(pos, f, fp)              # upwind cell
    D = np.where(pos, fp, f)               # downwind cell
    UU = np.where(pos, np.roll(f, 1, 0), np.roll(f, -2, 0))
    den = D - Uc
    ok = np.abs(den) > 1e-30
    rat = np.where(ok, (Uc - UU) / np.where(ok, den, 1.0), 0.0)
    return Uc + 0.5 * vanleer(rat) * den


def theta_k_eff(g, n):
    """eigenvalues k_eff(m)^2 of minus the second-order periodic difference operator in theta (spacing 2 pi/Nt) for
    modes m = 0..n-1: k_eff = (Nt/pi) sin(m pi/Nt) (-> m for m << Nt)."""
    m = np.arange(n)
    return g.Nt / np.pi * np.sin(m * np.pi / g.Nt) if g.M > 0 else m.astype(float)
