"""Stage figures. usage: python plots.py stage1|stage2|..."""
import sys, os, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
FIG = os.path.join(HERE, 'figs')
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, HERE)


def stage1():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for f in sorted(glob.glob(os.path.join(OUT, 'stage1_M*_hist.npy'))):
        h = np.load(f)
        lab = os.path.basename(f).replace('_hist.npy', '').replace('stage1_', '')
        axs[0].semilogy(h[:, 0], 100 * h[:, 1], label=lab + ' max')
        axs[0].semilogy(h[:, 0], 100 * h[:, 2], '--', label=lab + ' rms')
    axs[0].axhline(1, color='k', lw=0.8)
    axs[0].set_xlabel('t (s)'); axs[0].set_ylabel('|u - V e_x| / V  (%)'); axs[0].legend(fontsize=7)
    axs[0].set_title('Stage 1: start from rest, uniform draft V = 0.2 m/s')
    # final field: u_x in a horizontal plane at z = 20 mm, M = 4 run
    from core import Grid
    f = sorted(glob.glob(os.path.join(OUT, 'stage1_M4*.npz')))[-1]
    d = np.load(f)
    g = Grid(d['rf'], d['zf'], int(d['M']))
    th = g.theta[:, None, None]
    ur, ut = g.phys(d['us'][0]), g.phys(d['us'][1])
    ux = ur * np.cos(th) - ut * np.sin(th)
    j = np.argmin(abs(g.zc - 0.02))
    TH, RR = np.meshgrid(np.r_[g.theta, 2 * np.pi], g.rc, indexing='ij')
    U = np.r_[ux[:, j], ux[:1, j]]
    ax = axs[1]
    pc = ax.pcolormesh(RR * np.cos(TH) * 1e3, RR * np.sin(TH) * 1e3, U / 0.2, shading='gouraud', vmin=0.99, vmax=1.01, cmap='RdBu_r')
    ax.set_aspect('equal'); plt.colorbar(pc, ax=ax, label='u_x / V')
    ax.set_title(f'u_x / V at z = {g.zc[j]*1e3:.0f} mm, t = {float(d["t"]):.0f} s (M = {int(d["M"])})')
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'stage1_crossflow.png'), dpi=120)
    print('wrote figs/stage1_crossflow.png')



def slice_xz(g, f):
    """physical field (Nt, Nz, Nr) -> x-z plane through the draft (theta = 0 for x>0, pi for x<0)."""
    i0, ipi = 0, g.Nt // 2
    x = np.r_[-g.rc[::-1], g.rc]
    return x, np.concatenate([f[ipi][:, ::-1], f[i0]], axis=1)


def load_case(path, case_cls, **kw):
    """grid + case + physical fields of a saved run"""
    from core import Grid
    from candle import candle_grid
    d = np.load(path)
    from solver import Solver
    g = Grid(d['rf'], d['zf'], int(d['M']), wick=(0.75e-3, 6e-3), pool_r=10e-3)
    case = case_cls(g, **kw)
    sol = Solver(case)
    sol.load(path)
    return g, case, sol


def plume_lean(g, T, zs=(0.03, 0.05, 0.08, 0.1)):
    """x-centroid of (T - 300 K) in horizontal planes"""
    th = g.theta[:, None]
    X = g.rc[None, :] * np.cos(th)
    out = {}
    for z in zs:
        j = np.argmin(abs(g.zc - z))
        w = np.clip(T[:, j] - 300, 0, None) * g.rc[None, :] * g.dr[None, :]
        out[f'{g.zc[j]*1e3:.0f}'] = float((w * X).sum() / w.sum() * 1e3)
    return out


def stage2():
    from stage2_plume import Forced
    fig, axs = plt.subplots(1, 3, figsize=(13, 5.2), gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
    a = np.load(os.path.join(OUT, 'stage2_M0_V0_h0.375_axis.npy'))   # T, Uz (ours at ref z), z_ref, T_ref, Uz_ref
    ax = axs[0]
    ax.plot(a[2] * 1e3, a[3], 'k-', lw=2, label='OpenFOAM T')
    ax.plot(a[2] * 1e3, a[0], 'r--', label='quasi-3D T (M=0)')
    ax.axvspan(0, 46, color='0.9', zorder=0)
    ax.text(2, 400, 'forced source region\n(T, Y held, z < 46 mm)', fontsize=8)
    ax.set_xlabel('z (mm)'); ax.set_ylabel('T on axis (K)'); ax.set_xlim(0, 110)
    ax2 = ax.twinx()
    ax2.plot(a[2] * 1e3, a[4], 'b-', lw=2, label='OpenFOAM u_z')
    ax2.plot(a[2] * 1e3, a[1], 'c--', label='quasi-3D u_z')
    ax2.set_ylabel('u_z on axis (m/s)')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc='lower right')
    ax.set_title('Stage 2, M = 0: axis vs OpenFOAM')
    f = os.path.join(OUT, 'stage2_M4_V0.2_h0.375.npz')
    if os.path.exists(f):
        g, case, sol = load_case(f, Forced, V=0.2)
        F = sol.fields()
        T, _ = case.state({s: F[s] for s in case.scalars})
        x, Ts = slice_xz(g, T)
        _, Us = slice_xz(g, np.sqrt(F['ur'] ** 2 + F['ut'] ** 2 + F['uz'] ** 2))
        for ax, q, lab, cm in ((axs[1], Ts, 'T (K)', 'inferno'), (axs[2], Us, '|u| (m/s)', 'viridis')):
            pc = ax.pcolormesh(x * 1e3, g.zc * 1e3, q, cmap=cm, shading='auto')
            plt.colorbar(pc, ax=ax, label=lab)
            ax.set_xlim(-25, 25); ax.set_ylim(0, 120); ax.set_aspect('equal')
            ax.set_xlabel('x (mm), draft →'); ax.set_ylabel('z (mm)')
        lean = plume_lean(g, T)
        axs[1].set_title(f'M = 4, V = 0.2 m/s, t = {float(sol.t):.2f} s')
        axs[2].set_title('plume x-centroid (mm): ' + ', '.join(f'{k}:{v:.1f}' for k, v in lean.items()), fontsize=8)
        json.dump(dict(t=float(sol.t), plume_centroid_x_mm=lean), open(os.path.join(OUT, 'stage2_lean.json'), 'w'), indent=1)
        print('lean', lean)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'stage2_plume.png'), dpi=120)
    print('wrote figs/stage2_plume.png')



def stage3():
    """lean vs number of modes (time-averaged over the last 50 ms) against the OpenFOAM half-3D reference"""
    rows = json.load(open(os.path.join(OUT, 'stage3_lean_summary.json')))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, k, lab, ref in ((axs[0], 'cen_x_mm', 'flame centroid x (downstream, mm)', (12.0, 12.5)),
                            (axs[1], 'cen_z_mm', 'flame centroid z (mm)', (12.0, 12.4))):
        for scheme, mk in (('FV', 'o-'), ('clip', 's--')):
            r = [x for x in rows if x['case'].startswith(scheme)]
            Ms = [int(x['case'].split('=')[1]) for x in r]
            ax.errorbar(Ms, [x[k][0] for x in r], yerr=[x[k][1] for x in r], fmt=mk, capsize=3,
                        label='θ finite volume (THETA_FV)' if scheme == 'FV' else 'pseudo-spectral + clip')
        ax.axhspan(*ref, color='0.8', label='OpenFOAM half-3D (0.2–0.3 s)')
        ax.set_xlabel('M (azimuthal modes)'); ax.set_ylabel(lab); ax.set_xticks([2, 4, 8, 12])
    axs[0].legend(fontsize=8)
    fig.suptitle('Stage 3: flame-sheet lean in a 0.2 m/s draft (mean ± sd over the last 50 ms of 0.2 s)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'stage3_lean.png'), dpi=120)
    print('wrote figs/stage3_lean.png')



def stage5(ours=None, of_time='0.3'):
    """temperature in the plane of the draft (y = 0): OpenFOAM half-3D vs the quasi-3D stage-4 run"""
    sys.path.insert(0, os.path.dirname(HERE))
    from foamio import read_field
    from matplotlib.tri import Triangulation
    case = os.path.join(HERE, 'of3d', 'case')
    C = read_field(f'{case}/0/C'); T = read_field(f'{case}/{of_time}/T')
    rr = np.hypot(C[:, 0], C[:, 1])
    sel = (np.degrees(np.arctan2(C[:, 1], np.abs(C[:, 0]))) < 3.5) | (rr < 0.5e-3)   # cells next to the symmetry plane
    fig, axs = plt.subplots(1, 2, figsize=(9, 6.2), sharey=True)
    kw = dict(cmap='inferno', vmin=300, vmax=2100)
    tri = Triangulation(C[sel, 0] * 1e3, C[sel, 2] * 1e3)
    axs[0].tricontourf(tri, T[sel], levels=40, **kw)
    axs[0].set_title(f'OpenFOAM half-3D, t = {of_time} s', fontsize=10)
    if ours is None:
        cands = sorted(glob.glob(os.path.join(OUT, 'stage4_M4_V0.2_h0.375_fv*.npz')))
        ours = cands[-1]
    from stage4_chemistry import Chemistry
    g, cs, sol = load_case(ours, Chemistry, V=0.2)
    F = sol.fields()
    Tq, _ = cs.state({s: F[s] for s in cs.scalars})
    x, Ts = slice_xz(g, Tq)
    pc = axs[1].pcolormesh(x * 1e3, g.zc * 1e3, Ts, shading='auto', **kw)
    axs[1].set_title(f'quasi-3D stage 4, M = {g.M}, t = {float(sol.t):.3f} s', fontsize=10)
    for ax in axs:
        ax.set_xlim(-15, 38); ax.set_ylim(0, 80); ax.set_aspect('equal'); ax.set_xlabel('x (mm), draft 0.2 m/s →')
        ax.add_patch(plt.Rectangle((-0.75, 0), 1.5, 6, color='0.4'))
    axs[0].set_ylabel('z (mm)')
    plt.colorbar(pc, ax=axs, shrink=0.7, label='T (K)')
    fig.savefig(os.path.join(FIG, 'stage5_compare.png'), dpi=120, bbox_inches='tight')
    print('wrote figs/stage5_compare.png', ours)


if __name__ == '__main__':
    globals()[sys.argv[1]]()
