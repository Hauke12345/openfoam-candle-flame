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


if __name__ == '__main__':
    globals()[sys.argv[1]]()
