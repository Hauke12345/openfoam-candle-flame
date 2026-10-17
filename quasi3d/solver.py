"""Quasi-3D low-Mach time stepper (see core.py for the discretisation)."""
import time
import numpy as np
from core import (faces, interp_faces, div_faces, muscl_faces, Factor, csolve, BC,
                  OUTER, TOP, POOL, OPEN, WICK, OFF, G_Z)

VEL_D = {OUTER, OPEN, POOL, WICK}
P_BC_LABELS = {TOP}


class Solver:
    """case must provide:
        g (Grid), rho_inf, scalars: list of names, scalar_bc: dict name -> BC,
        vel_bc(t) -> (BC u_r, BC u_theta, BC u_z) with physical face values,
        bnd_mflux(t, rho) -> (Fr, Fz) physical face arrays, used on OUTER/OPEN/POOL/WICK faces,
        props(phi) -> (rho, mu, Gam) physical (phi: dict of physical scalar fields; Gam an array or a dict per scalar),
        optional source(phi, rho, dt) -> phi (split stiff sources), optional fix(phis) (spectral constraints)."""

    def __init__(self, case, cfl=0.4, dt_max=2e-4, r_theta_min=None, ab2=True):
        self.c = case
        g = self.g = case.g
        self.f = Factor(g)
        self.cfl, self.dt_max, self.ab2 = cfl, dt_max, ab2
        self.r_theta_min = r_theta_min if r_theta_min is not None else 4 * g.rc[0]
        self.t, self.n = 0.0, 0
        sh = (g.M + 1, g.Nz, g.Nr)
        self.us = np.zeros((3,) + sh, complex)
        self.ps = np.zeros(sh, complex)
        self.phis = {s: np.zeros(sh, complex) for s in case.scalars}
        self.prev_adv_u = None
        self.prev_adv_s = None
        self.dt_prev = None
        self.klev = None
        self.nu0 = self.D0 = None
        self.rinv = 1.0 / g.rc
        self.timing = dict(scal=0.0, mom=0.0, proj=0.0, src=0.0)
        self.bc_p = BC(P_BC_LABELS, 1.0, np.zeros((g.Nt, g.Nz, g.Nr + 1)), np.zeros((g.Nt, g.Nz + 1, g.Nr)))

    # ------------------------------------------------------------------ setup
    def set_state(self, u_phys, phi_phys, t=0.0):
        """u_phys: (3, Nt, Nz, Nr) physical velocity; phi_phys: dict of physical scalars."""
        g = self.g
        self.t = t
        for s in self.c.scalars:
            self.phis[s] = g.spec(phi_phys[s])
        for k in range(3):
            self.us[k] = g.spec(u_phys[k])
        self.us[:, :, g.solid] = 0
        self.rho, self.mu, self.Gam = self._props()
        self.rhos = g.spec(self.rho)
        # initial mass fluxes: interpolate rho u and project with d(rho)/dt = 0
        u = np.array([g.phys(self.us[k]) for k in range(3)])
        self.Fr, self.Fz, self.Gt = self._flux_star(self.rho * u, np.zeros_like(self.rho), self.t, 1.0, init=True)
        self._project(1.0, self.rhos, self.rhos, self.rho, u_star=u, keep_p=False)

    def _props(self):
        g = self.g
        phi = {s: g.phys(self.phis[s]) for s in self.c.scalars}
        rho, mu, Gam = self.c.props(phi)
        return g.trunc(rho), mu, Gam

    # ------------------------------------------------------------------ helpers
    def _mode_k(self, shift):
        return np.abs(self.g.m + shift)

    def _lap_apply(self, xs, k_of_m, dlabels, bvals):
        """A_k x + B b per mode; xs (M+1, Nz, Nr) spectral, bvals (M+1, nD) spectral."""
        g = self.g
        out = np.empty_like(xs)
        for m in range(g.M + 1):
            A, B = g.lap_matrix(int(k_of_m[m]), dlabels)
            v = A @ xs[m].ravel()
            if B.shape[1]:
                v = v + B @ bvals[m]
            out[m] = v.reshape(g.Nz, g.Nr)
        return out

    def _helm_solve(self, rhs, k_of_m, dlabels, c, bvals):
        """(I - c A_k) x = rhs + c B b per mode."""
        g = self.g
        out = np.empty_like(rhs)
        for m in range(g.M + 1):
            k = int(k_of_m[m])
            A, B = g.lap_matrix(k, dlabels)
            r = rhs[m].ravel().copy()
            if B.shape[1]:
                r = r + c * (B @ bvals[m])
            r[g.solid.ravel()] = 0
            out[m] = csolve(self.f.get(k, dlabels, c), r).reshape(g.Nz, g.Nr)
        return out

    def _choose_dt(self):
        g = self.g
        u = [g.phys(self.us[k]) for k in range(3)]
        reff = np.maximum(g.rc, self.r_theta_min)
        rate = (np.abs(u[0]) / g.dr + np.abs(u[2]) / g.dz[:, None] + np.abs(u[1]) * max(g.M, 1) / reff)
        dt_cfl = self.cfl / max(rate.max(), 1e-12)
        k_need = max(0, int(np.ceil(4 * np.log2(self.dt_max / dt_cfl))))
        if self.klev is None or k_need > self.klev:
            self.klev, self._kage = k_need, 0
        else:
            self._kage += 1
            if k_need < self.klev and self._kage > 50 and self.dt_max * 2 ** (-(self.klev - 1) / 4) < 0.9 * dt_cfl:
                self.klev, self._kage = self.klev - 1, 0
        return self.dt_max * 2 ** (-self.klev / 4)

    def _check_coeffs(self):
        nu = (self.mu / self.rho)[:, self.g.fluid].max()
        gams = self.Gam.values() if isinstance(self.Gam, dict) else [self.Gam]
        D = max((G / self.rho)[:, self.g.fluid].max() for G in gams)
        if self.nu0 is None or nu > 1.9 * self.nu0 or nu < 0.5 * self.nu0:
            self.nu0 = 0.6 * nu
        if self.D0 is None or D > 1.9 * self.D0 or D < 0.5 * self.D0:
            self.D0 = 0.6 * D
        if len(self.f.cache) > 6 * (self.g.M + 3) * 4:
            self.f.cache.clear()

    def _ab2(self, a, prev, dt):
        if prev is None or not self.ab2:
            return a
        w = dt / self.dt_prev
        return (1 + 0.5 * w) * a - 0.5 * w * prev

    # ------------------------------------------------------------------ physics operators
    def _divF(self, Fr, Fz, Gt):
        g = self.g
        return div_faces(g, Fr, Fz) + self.rinv * g.dth(Gt)

    def _advect(self, f, fs, bc, divF):
        """rho u . grad f  =  div_rz(F f_face) - f div_rz F + (rho u_theta / r) df/dtheta."""
        g = self.g
        fr, fz = muscl_faces(g, f, self.Fr, self.Fz, bc)
        return div_faces(g, self.Fr * fr, self.Fz * fz) - f * divF + self.Gt * self.rinv * g.dth(fs=fs)

    def _diffuse(self, f, fs, Gam, bc):
        """div(Gam grad f) with face gradients, theta part pseudo-spectral."""
        g = self.g
        _, gr, _, gz = faces(g, f, bc)
        Gr, Gz = interp_faces(g, Gam)
        return div_faces(g, Gr * gr, Gz * gz) + self.rinv ** 2 * g.dth(Gam * g.dth(fs=fs))

    def _taudiv(self, u, bcs, mu):
        """div tau, tau = mu (grad u + grad u^T) - 2/3 mu div(u) I, in cylindrical components."""
        g = self.g
        r, rf = g.rc, g.rf
        ur, ut, uz = u
        fr_r, gr_r, fz_r, gz_r = faces(g, ur, bcs[0])
        fr_t, gr_t, fz_t, gz_t = faces(g, ut, bcs[1])
        fr_z, gr_z, fz_z, gz_z = faces(g, uz, bcs[2])
        rf_safe = np.where(rf > 0, rf, 1.0)
        th_r, th_t, th_z = (g.dth(fs=self.us[k]) for k in range(3))
        div = div_faces(g, fr_r, fz_z) + th_t / r
        dz_ur = (fz_r[:, 1:] - fz_r[:, :-1]) / g.dz[:, None]
        dz_ut = (fz_t[:, 1:] - fz_t[:, :-1]) / g.dz[:, None]
        dr_uz = (fr_z[..., 1:] - fr_z[..., :-1]) / g.dr
        dr_ut = (fr_t[..., 1:] - fr_t[..., :-1]) / g.dr
        mur, muz = interp_faces(g, mu)
        divr, divz = interp_faces(g, div)
        thr_r, _ = interp_faces(g, th_r)
        dzur_r, _ = interp_faces(g, dz_ur)
        _, druz_z = interp_faces(g, dr_uz)
        _, thz_z = interp_faces(g, th_z)
        t_rr = mur * (2 * gr_r - 2 / 3 * divr)
        t_rt = mur * (gr_t + (thr_r - fr_t) / rf_safe)       # r d(u_t/r)/dr = du_t/dr - u_t/r
        t_rt[..., 0] = 0
        t_rz = mur * (gr_z + dzur_r)
        t_zr = muz * (gz_r + druz_z)
        t_zt = muz * (gz_t + thz_z / r)
        t_zz = muz * (2 * gz_z - 2 / 3 * divz)
        t_tt = mu * (2 * (th_t / r + ur / r) - 2 / 3 * div)
        t_tr = mu * (dr_ut + (th_r - ut) / r)
        t_tz = mu * (dz_ut + th_z / r)
        dzf = lambda q: (q[:, 1:] - q[:, :-1]) / g.dz[:, None]
        Dr = (rf[1:] * t_rr[..., 1:] - rf[:-1] * t_rr[..., :-1]) / (r * g.dr) + g.dth(t_tr) / r + dzf(t_zr) - t_tt / r
        Dt = (rf[1:] ** 2 * t_rt[..., 1:] - rf[:-1] ** 2 * t_rt[..., :-1]) / (r ** 2 * g.dr) + g.dth(t_tt) / r + dzf(t_zt)
        Dz = (rf[1:] * t_rz[..., 1:] - rf[:-1] * t_rz[..., :-1]) / (r * g.dr) + g.dth(t_tz) / r + dzf(t_zz)
        return Dr, Dt, Dz

    def _gradp_cell(self, p, ps):
        g = self.g
        _, gr, _, gz = faces(g, p, self.bc_p, extrap_n=True)
        return 0.5 * (gr[..., :-1] + gr[..., 1:]), self.rinv * g.dth(fs=ps), 0.5 * (gz[:, :-1] + gz[:, 1:]), gr, gz

    def _flux_star(self, rhou, rho_unused, t, dt, init=False, gradp=None):
        """Rhie-Chow face mass fluxes from cell rho u* (+ dt (interp<grad p> - grad_f p)); boundary fluxes prescribed."""
        g = self.g
        Fr, _ = interp_faces(g, rhou[0], -1.0)
        _, Fz = interp_faces(g, rhou[2])
        if gradp is not None:
            gcr, _, gcz, gr, gz = gradp
            ir, _ = interp_faces(g, gcr, -1.0)
            _, iz = interp_faces(g, gcz)
            Fr += dt * (ir - gr)
            Fz += dt * (iz - gz)
        Fr[..., 0] = 0
        bFr, bFz = self.c.bnd_mflux(t, self.rho)
        for fam, F, bF in (('r', Fr, bFr), ('z', Fz, bFz)):
            for L, bf in g.bfaces[fam].items():
                if L in VEL_D:
                    F[:, bf['j'], bf['i']] = bF[:, bf['j'], bf['i']]
        Fr[:, g.lr == OFF] = 0
        Fz[:, g.lz == OFF] = 0
        return g.trunc(Fr), g.trunc(Fz), g.trunc(rhou[1])

    def _project(self, dt, rhos_new, rhos_old, rho_new, u_star=None, keep_p=True):
        """Solve L_m dp = (div F* + (rho^{n+1}-rho^n)/dt)/dt; correct F, rho u_theta, cell rho u."""
        g = self.g
        Frs, Fzs, Gts = g.spec(self.Fr), g.spec(self.Fz), g.spec(self.Gt)
        divs = div_faces(g, Frs, Fzs) + 1j * g.mm * self.rinv * Gts
        rhs = (divs + (rhos_new - rhos_old) / dt) / dt
        rhs[:, g.solid] = 0
        dps = np.empty_like(rhs)
        for m in range(g.M + 1):
            dps[m] = csolve(self.f.get(m, P_BC_LABELS, None), rhs[m].ravel()).reshape(g.Nz, g.Nr)
        dp = g.phys(dps)
        gcr, gct, gcz, gr, gz = self._gradp_cell(dp, dps)
        _, grn, _, gzn = faces(g, dp, self.bc_p)       # zero on Neumann faces
        self.Fr = g.phys(Frs) - dt * grn
        self.Fz = g.phys(Fzs) - dt * gzn
        self.Fr[..., 0] = 0
        self.Fr[:, g.lr == OFF] = 0
        self.Fz[:, g.lz == OFF] = 0
        self.Gt = g.phys(Gts) - dt * gct
        rhou = rho_new * u_star - dt * np.array([gcr, gct, gcz])
        u = rhou / rho_new
        for k in range(3):
            self.us[k] = g.spec(u[k])
        self.us[:, :, g.solid] = 0
        self.ps = (self.ps + dps) if keep_p else self.ps
        self.last_div = np.abs(g.phys(div_faces(g, g.spec(self.Fr), g.spec(self.Fz)) + 1j * g.mm * self.rinv * g.spec(self.Gt)
                               + (rhos_new - rhos_old) / dt)[:, g.fluid]).max()

    # ------------------------------------------------------------------ time step
    def step(self):
        g, c = self.g, self.c
        self._check_coeffs()
        dt = self._choose_dt()
        t1 = self.t + dt
        divF = div_faces(g, self.Fr, self.Fz)    # (r, z) part only: theta advection is in advective form
        # 1. scalars
        t0 = time.time()
        adv_s = {}
        new = {}
        for s in c.scalars:
            bc = c.scalar_bc[s]
            fs = self.phis[s]
            f = g.phys(fs)
            a = g.spec(self._advect(f, fs, bc, divF) / self.rho)
            adv_s[s] = a
            A = self._ab2(a, None if self.prev_adv_s is None else self.prev_adv_s[s], dt)
            Gs = self.Gam[s] if isinstance(self.Gam, dict) else self.Gam
            dif = g.spec(self._diffuse(f, fs, Gs, bc) / self.rho)
            dl = bc.d
            bv = g.spec(g.gather_d(dl, bc.vr, bc.vz)) if dl else np.zeros((g.M + 1, 0), complex)
            cc = dt * self.D0
            rhs = fs + dt * (dif - A) - cc * self._lap_apply(fs, g.m, dl, bv)
            new[s] = self._helm_solve(rhs, g.m, dl, cc, bv)
            new[s][:, g.solid] = g.spec(np.full((g.Nt,) + g.solid.shape, bc.solid_fill))[:, g.solid]
        self.prev_adv_s = adv_s
        self.phis = new
        self.timing['scal'] += time.time() - t0
        t0 = time.time()
        if hasattr(c, 'source'):
            phi = {s: g.phys(self.phis[s]) for s in c.scalars}
            phi = c.source(phi, self.rho, dt, t1)
            self.phis = {s: g.spec(phi[s]) for s in c.scalars}
        if hasattr(c, 'fix'):
            c.fix(self.phis, t1)
        self.timing['src'] += time.time() - t0
        # 2. equation of state
        rho_old, rhos_old = self.rho, self.rhos
        self.rho, self.mu, self.Gam = self._props()
        self.rhos = g.spec(self.rho)
        # 3. momentum predictor
        t0 = time.time()
        u = np.array([g.phys(self.us[k]) for k in range(3)])
        bcs_n = c.vel_bc(self.t)
        bcs_1 = c.vel_bc(t1)
        adv = np.empty_like(u)
        for k in range(3):
            adv[k] = self._advect(u[k], self.us[k], bcs_n[k], divF)
        adv[0] -= self.Gt * u[1] * self.rinv
        adv[1] += self.Gt * u[0] * self.rinv
        a_u = np.array([g.spec(adv[k] / self.rho) for k in range(3)])
        A = self._ab2(a_u, self.prev_adv_u, dt)
        self.prev_adv_u = a_u
        tau = self._taudiv(u, bcs_n, self.mu)
        gcr, gct, gcz, gr, gz = self._gradp_cell(g.phys(self.ps), self.ps)
        buoy = (self.rho - c.rho_inf) * G_Z
        E = np.array([g.spec((tau[0] - gcr) / self.rho), g.spec((tau[1] - gct) / self.rho),
                      g.spec((tau[2] - gcz + buoy) / self.rho)])
        R = self.us + dt * (E - A)
        cc = dt * self.nu0
        bvn = [g.spec(g.gather_d(VEL_D, b.vr, b.vz)) for b in bcs_n]
        bv1 = [g.spec(g.gather_d(VEL_D, b.vr, b.vz)) for b in bcs_1]
        ustar = np.empty_like(self.us)
        xs = []
        for sgn, shift in ((1, 1), (-1, -1)):
            k_of_m = self._mode_k(shift)
            un = self.us[0] + sgn * 1j * self.us[1]
            Rpm = R[0] + sgn * 1j * R[1]
            bn = bvn[0] + sgn * 1j * bvn[1]
            b1 = bv1[0] + sgn * 1j * bv1[1]
            rhs = Rpm - cc * self._lap_apply(un, k_of_m, VEL_D, bn)
            xs.append(self._helm_solve(rhs, k_of_m, VEL_D, cc, b1))
        ustar[0] = 0.5 * (xs[0] + xs[1])
        ustar[1] = (xs[0] - xs[1]) / 2j
        rhs = R[2] - cc * self._lap_apply(self.us[2], g.m, VEL_D, bvn[2])
        ustar[2] = self._helm_solve(rhs, g.m, VEL_D, cc, bv1[2])
        ustar[0, 0] = ustar[0, 0].real
        ustar[1, 0] = ustar[1, 0].real
        self.timing['mom'] += time.time() - t0
        # 4. projection
        t0 = time.time()
        up = np.array([g.phys(ustar[k]) for k in range(3)])
        self.t = t1
        self.Fr, self.Fz, self.Gt = self._flux_star(self.rho * up, None, t1, dt,
                                                    gradp=(gcr, gct, gcz, gr, gz))
        self._project(dt, self.rhos, rhos_old, self.rho, u_star=up)
        self.timing['proj'] += time.time() - t0
        self.dt_prev = dt
        self.n += 1
        return dt

    # ------------------------------------------------------------------ output
    def fields(self):
        g = self.g
        out = {'ur': g.phys(self.us[0]), 'ut': g.phys(self.us[1]), 'uz': g.phys(self.us[2]),
               'p': g.phys(self.ps), 'rho': self.rho}
        for s in self.c.scalars:
            out[s] = g.phys(self.phis[s])
        return out

    def save(self, path, **extra):
        g = self.g
        np.savez_compressed(path, t=self.t, rf=g.rf, zf=g.zf, M=g.M, us=self.us, ps=self.ps,
                            phis=np.array([self.phis[s] for s in self.c.scalars]),
                            scalars=np.array(self.c.scalars), **extra)

    def load(self, path):
        d = np.load(path)
        g = self.g
        M = min(int(d['M']), g.M)
        self.us[:] = 0; self.us[:, :M + 1] = d['us'][:, :M + 1]
        self.ps[:] = 0; self.ps[:M + 1] = d['ps'][:M + 1]
        names = list(d['scalars'])
        u = np.array([g.phys(self.us[k]) for k in range(3)])
        phi = {}
        for s in self.c.scalars:
            fs = np.zeros_like(self.ps)
            if s in names:
                fs[:M + 1] = d['phis'][names.index(s)][:M + 1]
            phi[s] = g.phys(fs)
        ps = self.ps.copy()
        self.set_state(u, phi, float(d['t']))
        self.ps = ps
