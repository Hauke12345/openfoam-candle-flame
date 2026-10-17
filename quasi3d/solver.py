"""Quasi-3D low-Mach time stepper (see core.py for the discretisation).

Variable-density projection with a divergence constraint (velocity projection, Dodd & Ferrante splitting):
  1. scalars  phi^{n+1}:  Dphi/Dt = (div(Gam grad phi) + S_phi)/rho, mass-consistent advective form
     rho_c u.grad phi = div(F phi_f) - phi_c div F with MUSCL face states phi_f and face mass fluxes
     F = rho_EOS(phi_f) U^n (see _mass_fluxes: the EOS density at the face state makes div F agree with the
     density change implied by div U = S to O(v'' dphi^2), v = 1/rho; the upwind cell density does not, and acted
     as an O(30 %) spurious fuel source next to the wick)
  2. rho^{n+1} = EOS(phi^{n+1});  S = div u = -(1/rho) Drho/Dt = -sum_s (1/rho) d rho/d phi_s * R_s, where R_s is the
     non-advective rate of phi_s (diffusion + sources). Advection does not enter S, which removes the feedback
     "d rho/dt from the EOS -> divergence -> advection -> rho" that makes EOS-differenced d rho/dt unstable.
  3. momentum: u* = u^n + dt [-(u.grad)u + (div tau + (rho - rho_inf) g)/rho - G(p^n, p_hat)], implicit nu0 part.
     G(p, p_hat) = grad p / rho0 + (1/rho - 1/rho0) grad p_hat,  p_hat = 2 p^n - p^{n-1}  (Dodd & Ferrante 2014)
  4. projection: L_m dp = rho0 (div U* - S)/dt (constant coefficient per mode),  U^{n+1} = U* - dt grad dp / rho0.
"""
import time
import numpy as np
from core import (faces, interp_faces, div_faces, muscl_faces, Factor, csolve, BC,
                  OUTER, TOP, POOL, OPEN, WICK, OFF, G_Z)

VEL_D = {OUTER, OPEN, POOL, WICK}
P_BC_LABELS = {TOP}


class Solver:
    """case must provide:
        g (Grid), rho_inf, scalars: list of names, scalar_bc: dict name -> BC,
        vel_bc(t) -> (BC u_r, BC u_theta, BC u_z) with physical face values (also the boundary volume fluxes),
        props(phi) -> (rho, mu, Gam) physical (phi: dict of physical scalar fields; Gam an array or a dict per scalar),
        dlnrho(phi) -> dict s -> (1/rho) d rho / d phi_s (physical), only for scalars that affect rho,
        optional source(phi, rho, dt, t) -> phi (split stiff sources), optional fix(phis, t) (spectral constraints)."""

    def __init__(self, case, cfl=0.4, dt_max=2e-4, r_theta_min=None, ab2=True, beta=0.0):
        # beta: volume-discrepancy relaxation S += beta (rho* - rho_EOS)/(rho dt). Keep 0: rho is reset to the EOS
        # every step, so the extra volume source has no matching scalar change and recreates the mismatch
        # (stage 3, beta = 0.3: spurious fuel source 8.6e-7 kg/s vs 1.0e-7 with beta = 0).
        self.c = case
        g = self.g = case.g
        self.f = Factor(g)
        self.cfl, self.dt_max, self.ab2, self.beta = cfl, dt_max, ab2, beta
        self.mismatch = 0.0
        self.r_theta_min = r_theta_min if r_theta_min is not None else 4 * g.rc[0]
        self.t, self.n = 0.0, 0
        sh = (g.M + 1, g.Nz, g.Nr)
        self.us = np.zeros((3,) + sh, complex)
        self.ps = np.zeros(sh, complex)
        self.ps_old = np.zeros(sh, complex)
        self.phis = {s: np.zeros(sh, complex) for s in case.scalars}
        self.prev_adv_u = None
        self.prev_adv_s = None
        self.dt_prev = None
        self.klev = None
        self.nu0 = self.D0 = None
        self.rinv = 1.0 / g.rc
        self.timing = dict(scal=0.0, mom=0.0, proj=0.0, src=0.0)
        self.bc_p = BC(P_BC_LABELS, 1.0, np.zeros((g.Nt, g.Nz, g.Nr + 1)), np.zeros((g.Nt, g.Nz + 1, g.Nr)))
        self.S = np.zeros((g.Nt, g.Nz, g.Nr))

    # ------------------------------------------------------------------ setup
    def set_state(self, u_phys, phi_phys, t=0.0):
        """u_phys: (3, Nt, Nz, Nr) physical velocity; phi_phys: dict of physical scalars. Projects u (S = 0)."""
        g = self.g
        c = self.c
        self.t = t
        for s in self.c.scalars:
            self.phis[s] = g.spec(phi_phys[s])
        for k in range(3):
            self.us[k] = g.spec(u_phys[k])
        self.us[:, :, g.solid] = 0
        self.rho, self.mu, self.Gam = self._props()
        self.rho0 = self.rho[:, g.fluid].min()
        c.rho_now = self.rho
        u = np.array([g.phys(self.us[k]) for k in range(3)])
        self.Ur, self.Uz, self.Ut = self._flux_star(u, self.t, 1.0, None)
        self._project(1.0, u, np.zeros_like(self.S), keep_p=False)

    def _props(self):
        g = self.g
        phi = {s: g.phys(self.phis[s]) for s in self.c.scalars}
        rho, mu, Gam = self.c.props(phi)
        # rho is the pointwise EOS density at the collocation angles, NOT truncated to M modes. rho(Z) is very
        # nonlinear (rho(0) = 1.17, rho(0.005) = 0.72, rho(0.05) = 0.21), so a flame sheet that is sharp in theta
        # (draft runs) gives a rho(theta) whose M-mode truncation overshoots below zero: with M = 2 min(rho)
        # fell from 0.18 to -0.001 within 9.6 ms of switching on the draft (lee side, r = 7 mm, z = 5 mm), and
        # 1/rho (nu0 = 0.6 max mu/rho, buoyancy, pressure terms) blew up. The pointwise value is positive and is
        # the same density the face mass fluxes rho_EOS(phi_f) U use; products with it are truncated afterwards.
        return rho, mu, Gam

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
    def _face_states(self):
        """MUSCL face values (r and z faces) of every scalar for the volume fluxes U^n: used both for the scalar
        fluxes and for the face density of the mass fluxes."""
        g, c = self.g, self.c
        self._sf = {}
        for s in c.scalars:
            self._sf[s] = muscl_faces(g, g.phys(self.phis[s]), self.Ur, self.Uz, c.scalar_bc[s])

    def _mass_fluxes(self):
        """face mass fluxes F = rho_f U and their (r, z) divergence.

        rho_f = rho_EOS(phi_f), the equation of state at the MUSCL face state used by the scalar fluxes. This is
        the choice for which the divergence constraint div U = S (S from the non-advective rates) and discrete
        continuity agree: with F = rho_f U the conservative scalar update of cell c changes the EOS density by
        -dt div F only if rho_f (v_c + v'_c (phi_f - phi_c)) = 1 on every face (v = 1/rho), i.e. rho_f is the EOS
        density at the face state for a specific volume linear in phi (exact for mixing, close for the flame-sheet
        and ideal-gas states). The upwind cell density violates it by a factor rho_up/rho_c (up to ~3 next to the
        wick), which acted as a spurious source of fuel-element mass of 30-70 % of the wick supply. On the wick face
        the face state is the Dirichlet fuel state, so F = rho_w U_w = mdot''. Cases without a density(phi)
        method (constant density) use the upwind cell density."""
        g, rho, c = self.g, self.rho, self.c
        dens = getattr(c, 'density', None)
        if dens is not None and c.scalars and not getattr(c, 'legacy_flux', False):
            rr = dens({s: self._sf[s][0] for s in c.scalars})
            rz = dens({s: self._sf[s][1] for s in c.scalars})
            Fr = self.Ur * rr
            Fz = self.Uz * rz
        else:
            Fr = np.empty_like(self.Ur); Fz = np.empty_like(self.Uz)
            Fr[..., 1:-1] = self.Ur[..., 1:-1] * np.where(self.Ur[..., 1:-1] >= 0, rho[..., :-1], rho[..., 1:])
            Fr[..., -1] = self.Ur[..., -1] * rho[..., -1]
            Fz[:, 1:-1] = self.Uz[:, 1:-1] * np.where(self.Uz[:, 1:-1] >= 0, rho[:, :-1], rho[:, 1:])
            Fz[:, 0] = self.Uz[:, 0] * rho[:, 0]
            Fz[:, -1] = self.Uz[:, -1] * rho[:, -1]
            for fam, F, U in (('r', Fr, self.Ur), ('z', Fz, self.Uz)):
                for L, bf in g.bfaces[fam].items():
                    F[:, bf['j'], bf['i']] = U[:, bf['j'], bf['i']] * rho[:, bf['jc'], bf['ic']]
        Fr[..., 0] = 0
        Fr[:, g.lr == OFF] = 0
        Fz[:, g.lz == OFF] = 0
        self.Fr, self.Fz = Fr, Fz
        return div_faces(g, Fr, Fz)

    def _advect(self, f, fs, bc, divF, ffaces=None):
        """u . grad f = [div_rz(F f_face) - f div_rz F] / rho + (u_theta / r) df/dtheta, F = rho U on faces
        (mass-consistent advective form: conservative for rho f where div F matches the density change)."""
        g = self.g
        fr, fz = muscl_faces(g, f, self.Ur, self.Uz, bc) if ffaces is None else ffaces
        return (div_faces(g, self.Fr * fr, self.Fz * fz) - f * divF) / self.rho + self.Ut * self.rinv * g.dth(fs=fs)

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

    def _grad(self, p, ps, extrap_n=True):
        """cell gradient (r, theta, z) and face gradients of a pressure-like field."""
        g = self.g
        _, gr, _, gz = faces(g, p, self.bc_p, extrap_n=extrap_n)
        return [0.5 * (gr[..., :-1] + gr[..., 1:]), self.rinv * g.dth(fs=ps), 0.5 * (gz[:, :-1] + gz[:, 1:])], gr, gz

    def _G(self):
        """Dodd-Ferrante pressure acceleration G = grad p/rho0 + (1/rho - 1/rho0) grad p_hat: cells and faces."""
        g = self.g
        phs = 2 * self.ps - self.ps_old
        gc, gr, gz = self._grad(g.phys(self.ps), self.ps)
        hc, hr, hz = self._grad(g.phys(phs), phs)
        ir0 = 1.0 / self.rho0
        dr_ = 1.0 / self.rho - ir0
        rr, rz = interp_faces(g, self.rho)
        Gc = [ir0 * gc[k] + dr_ * hc[k] for k in range(3)]
        Gr = ir0 * gr + (1.0 / rr - ir0) * hr
        Gz = ir0 * gz + (1.0 / rz - ir0) * hz
        return Gc, Gr, Gz

    def _flux_star(self, u, t, dt, G):
        """face volume fluxes from cell u* (+ dt (interp<G> - G_face), Rhie-Chow); boundary fluxes prescribed."""
        g = self.g
        Ur, _ = interp_faces(g, u[0], -1.0)
        _, Uz = interp_faces(g, u[2])
        if G is not None:
            Gc, Gr, Gz = G
            ir, _ = interp_faces(g, Gc[0], -1.0)
            _, iz = interp_faces(g, Gc[2])
            Ur += dt * (ir - Gr)
            Uz += dt * (iz - Gz)
        Ur[..., 0] = 0
        ur, _, uz = self.c.vel_bc(t)
        for fam, F, bv in (('r', Ur, ur.vr), ('z', Uz, uz.vz)):
            for L, bf in g.bfaces[fam].items():
                if L in VEL_D:
                    F[:, bf['j'], bf['i']] = bv[:, bf['j'], bf['i']]
        Ur[:, g.lr == OFF] = 0
        Uz[:, g.lz == OFF] = 0
        return g.trunc(Ur), g.trunc(Uz), g.trunc(u[1])

    def _project(self, dt, u_star, S, keep_p=True):
        """L_m dp = rho0 (div U* - S)/dt; U^{n+1} = U* - dt grad dp/rho0 (faces), same with cell gradients."""
        g = self.g
        Urs, Uzs, Uts = g.spec(self.Ur), g.spec(self.Uz), g.spec(self.Ut)
        Ss = g.spec(S)
        divs = div_faces(g, Urs, Uzs) + 1j * g.mm * self.rinv * Uts
        rhs = self.rho0 * (divs - Ss) / dt
        rhs[:, g.solid] = 0
        dps = np.empty_like(rhs)
        for m in range(g.M + 1):
            dps[m] = csolve(self.f.get(m, P_BC_LABELS, None), rhs[m].ravel()).reshape(g.Nz, g.Nr)
        dp = g.phys(dps)
        gc, _, _ = self._grad(dp, dps)
        _, grn, _, gzn = faces(g, dp, self.bc_p)       # zero on Neumann faces
        c = dt / self.rho0
        self.Ur = g.phys(Urs) - c * grn
        self.Uz = g.phys(Uzs) - c * gzn
        self.Ur[..., 0] = 0
        self.Ur[:, g.lr == OFF] = 0
        self.Uz[:, g.lz == OFF] = 0
        self.Ut = g.phys(Uts) - c * gc[1]
        u = np.array([u_star[0] - c * gc[0], self.Ut, u_star[2] - c * gc[2]])
        for k in range(3):
            self.us[k] = g.spec(u[k])
        self.us[:, :, g.solid] = 0
        if keep_p:
            self.ps_old = self.ps
            self.ps = self.ps + dps
        self.last_div = np.abs(g.phys(div_faces(g, g.spec(self.Ur), g.spec(self.Uz)) + 1j * g.mm * self.rinv
                                      * g.spec(self.Ut) - Ss)[:, g.fluid]).max()

    # ------------------------------------------------------------------ time step
    def step(self):
        g, c = self.g, self.c
        self._check_coeffs()
        dt = self._choose_dt()
        t1 = self.t + dt
        # 1. scalars (Strang: half source step before and after transport if case.strang)
        t0 = time.time()
        old = self.phis
        strang = getattr(c, 'strang', False)
        if strang:
            phi = {s: g.phys(self.phis[s]) for s in c.scalars}
            phi = c.source(phi, self.rho, 0.5 * dt, self.t)
            self.phis = {s: g.spec(phi[s]) for s in c.scalars}
        self._last_phi_n = self.phis
        self._face_states()           # face states of the transported (post half-source) scalars
        divF = self._mass_fluxes()    # (r, z) part only: theta advection is in advective form
        adv_s, used, new = {}, {}, {}
        for s in c.scalars:
            bc = c.scalar_bc[s]
            fs = self.phis[s]
            f = g.phys(fs)
            a = g.spec(self._advect(f, fs, bc, divF, self._sf[s]))
            adv_s[s] = a
            A = self._ab2(a, None if self.prev_adv_s is None else self.prev_adv_s[s], dt)
            used[s] = A
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
            phi = c.source(phi, self.rho, 0.5 * dt if strang else dt, t1)
            self.phis = {s: g.spec(phi[s]) for s in c.scalars}
        if hasattr(c, 'fix'):
            c.fix(self.phis, t1)
        self.timing['src'] += time.time() - t0
        # 2. equation of state and divergence constraint from the non-advective scalar rates
        rho_n = self.rho
        self.rho, self.mu, self.Gam = self._props()
        self.rho0 = self.rho[:, g.fluid].min()
        c.rho_now = self.rho
        phi = {s: g.phys(self.phis[s]) for s in c.scalars}
        dl = c.dlnrho(phi) if hasattr(c, 'dlnrho') else {}
        S = np.zeros_like(self.rho)
        for s, d in dl.items():
            S -= d * g.phys((self.phis[s] - old[s]) / dt + used[s])
        # volume-discrepancy correction: drive the continuity density rho* = rho^n - dt div F towards the EOS density
        if self.beta:
            rho_star = rho_n - dt * (divF + self.rinv * g.dth(rho_n * self.Ut))
            S += self.beta * (rho_star - self.rho) / (self.rho * dt)
            self.mismatch = np.abs((rho_star - self.rho) / self.rho)[:, g.fluid].max()
        S[:, g.solid] = 0
        self.S = g.trunc(S)
        if getattr(c, 'budget', None):
            self._budget(rho_n, divF, dt)
        # 3. momentum predictor
        t0 = time.time()
        u = np.array([g.phys(self.us[k]) for k in range(3)])
        bcs_n = c.vel_bc(self.t)
        bcs_1 = c.vel_bc(t1)
        adv = np.empty_like(u)
        for k in range(3):
            adv[k] = self._advect(u[k], self.us[k], bcs_n[k], divF)
        adv[0] -= self.Ut * u[1] * self.rinv
        adv[1] += self.Ut * u[0] * self.rinv
        a_u = np.array([g.spec(adv[k]) for k in range(3)])
        A = self._ab2(a_u, self.prev_adv_u, dt)
        self.prev_adv_u = a_u
        tau = self._taudiv(u, bcs_n, self.mu)
        G = self._G()
        Gc = G[0]
        buoy = (self.rho - c.rho_inf) * G_Z
        E = np.array([g.spec(tau[0] / self.rho - Gc[0]), g.spec(tau[1] / self.rho - Gc[1]),
                      g.spec((tau[2] + buoy) / self.rho - Gc[2])])
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
        self.Ur, self.Uz, self.Ut = self._flux_star(up, t1, dt, G)
        self._project(dt, up, self.S)
        self.timing['proj'] += time.time() - t0
        self.dt_prev = dt
        self.n += 1
        return dt

    # ------------------------------------------------------------------ diagnostics
    def _budget(self, rho_n, divF, dt):
        """Mass budget (kg/s, whole domain, theta-integrated) of each scalar in case.budget, accumulated as
        sums over steps (divide by the summed dt): spurious source phi^{n+1} (rho_EOS^{n+1} - rho*)/dt with
        rho* = rho^n - dt div F (the continuity density the transport assumed), convective and diffusive inflow
        through the wick faces, outflow through all other boundaries, total mass of rho phi."""
        g, c = self.g, self.c
        w = 2 * np.pi / g.Nt
        rho_star = rho_n - dt * (divF + self.rinv * g.dth(rho_n * self.Ut))
        b = self.__dict__.setdefault('bud', {})
        b['dt'] = b.get('dt', 0.0) + dt
        b['mass_err'] = b.get('mass_err', 0.0) + dt * w * (g.vol * (self.rho - rho_star))[:, g.fluid].sum() / dt
        for s in c.budget:
            f = g.phys(self.phis[s])
            bc = c.scalar_bc[s]
            Gs = self.Gam[s] if isinstance(self.Gam, dict) else self.Gam
            spur = w * (g.vol * f * (self.rho - rho_star))[:, g.fluid].sum() / dt
            fold = g.phys(self._last_phi_n[s])
            fr, fz = self._sf[s]
            _, gr, _, gz = faces(g, fold, bc)
            Gr, Gz = interp_faces(g, Gs)
            win = wdif = out = 0.0
            for fam, F, fv, gv, Gv in (('r', self.Fr, fr, gr, Gr), ('z', self.Fz, fz, gz, Gz)):
                for L, bf in g.bfaces[fam].items():
                    j, i = bf['j'], bf['i']
                    A = g.rf[i] * g.dz[j] if fam == 'r' else g.rc[i] * g.dr[i]
                    # flux in +r/+z direction; inflow into the fluid = -s * flux
                    conv = -bf['s'] * F[:, j, i] * fv[:, j, i] * A
                    dif = bf['s'] * Gv[:, j, i] * gv[:, j, i] * A
                    if L == WICK:
                        win += w * conv.sum(); wdif += w * dif.sum()
                    else:
                        out -= w * (conv + dif).sum()
            for zp in getattr(c, 'budget_planes', ()):
                j = int(np.argmin(abs(g.zf - zp)))
                A = g.rc * g.dr
                pl = w * ((self.Fz[:, j] * fz[:, j] - Gz[:, j] * gz[:, j]) * A).sum()
                b[s + '_z%.0f' % (zp * 1e3)] = b.get(s + '_z%.0f' % (zp * 1e3), 0.0) + dt * pl
            tot = w * (g.vol * self.rho * f)[:, g.fluid].sum()
            for k, v in (('spur', spur), ('wick_conv', win), ('wick_diff', wdif), ('out', out)):
                b[s + '_' + k] = b.get(s + '_' + k, 0.0) + dt * v
            if s + '_mass0' not in b:
                b[s + '_mass0'] = w * (g.vol * rho_n * fold)[:, g.fluid].sum()
            b[s + '_mass1'] = tot

    def budget_report(self, reset=True):
        b = getattr(self, 'bud', None)
        if not b:
            return {}
        T = b['dt']
        out = {'mass_err': b['mass_err'] / T}
        for s in self.c.budget:
            for k in ('spur', 'wick_conv', 'wick_diff', 'out'):
                out[s + '_' + k] = b[s + '_' + k] / T
            out[s + '_dMdt'] = (b[s + '_mass1'] - b[s + '_mass0']) / T
            for zp in getattr(self.c, 'budget_planes', ()):
                k = s + '_z%.0f' % (zp * 1e3)
                out[k] = b[k] / T
        if reset:
            self.bud = {}
        return out

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
        np.savez_compressed(path, t=self.t, rf=g.rf, zf=g.zf, M=g.M, us=self.us, ps=self.ps, ps_old=self.ps_old,
                            phis=np.array([self.phis[s] for s in self.c.scalars]),
                            scalars=np.array(self.c.scalars), **extra)

    def load(self, path):
        d = np.load(path)
        g = self.g
        M = min(int(d['M']), g.M)
        self.us[:] = 0; self.us[:, :M + 1] = d['us'][:, :M + 1]
        ps = np.zeros_like(self.ps); ps[:M + 1] = d['ps'][:M + 1]
        pso = ps.copy()
        if 'ps_old' in d:
            pso[:] = 0; pso[:M + 1] = d['ps_old'][:M + 1]
        names = list(d['scalars'])
        u = np.array([g.phys(self.us[k]) for k in range(3)])
        phi = {}
        for s in self.c.scalars:
            fs = np.zeros_like(self.ps)
            if s in names:
                fs[:M + 1] = d['phis'][names.index(s)][:M + 1]
            phi[s] = g.phys(fs)
        self.set_state(u, phi, float(d['t']))
        self.ps, self.ps_old = ps, pso
