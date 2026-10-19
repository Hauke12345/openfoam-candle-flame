"""Degassing / gas-bubble analysis of a run: per frame
  T (from the annealing schedule), largest N2 cluster (neighbour cutoff 0.55 nm, periodic in x, y), its centre and
  whether it sits on the less-wetting patch, N2 in the free space above the water (degassed), and the bubble volume
  (grid cells of 0.2 nm inside the liquid slab with no water oxygen within 0.32 nm).
usage: python3 analyze.py RUNDIR [deffnm=heat] [stride=1] [T0=293] [T1=368]  -> RUNDIR/analysis_<deffnm>.json/.png
  (temperature ramps linearly T0 -> T1 over the first 2 ns, as in the annealing schedule)
"""
import sys, os, json
import numpy as np
import MDAnalysis as mda
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix

RUN = sys.argv[1]
NAME = sys.argv[2] if len(sys.argv) > 2 else 'heat'
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 1
T0 = float(sys.argv[4]) if len(sys.argv) > 4 else 293.0
T1 = float(sys.argv[5]) if len(sys.argv) > 5 else 368.0
meta = json.load(open(f'{RUN}/meta.json'))
u = mda.Universe(f'{RUN}/conf.gro', f'{RUN}/{NAME}.xtc')     # .gro as topology (MDAnalysis cannot read GROMACS 2026 .tpr)
n2 = u.select_atoms('name N2')
ow = u.select_atoms('name OW')
L = meta['LXY']; zt = meta['z_wall_top']; cx = cy = L / 2


def clusters(p, box):
    pp = p.copy(); pp[:, :2] %= box[:2]; pp[:, 2] = pp[:, 2] % box[2]
    pp = np.minimum(pp, np.array([box[0], box[1], box[2] * 10]) * (1 - 1e-7))
    t = cKDTree(pp, boxsize=[box[0], box[1], box[2] * 10])      # periodic in x, y only
    pairs = t.query_pairs(0.55, output_type='ndarray')
    n = len(p)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    nc, lab = connected_components(g, directed=False)
    return lab


rows = []
for ts in u.trajectory[::STRIDE]:
    box = ts.dimensions[:3] / 10.0
    p = n2.positions / 10.0
    L = float(box[0])
    o = ow.positions / 10.0
    zs = np.sort(o[:, 2]); z_surf = np.percentile(o[:, 2], 99.5)     # top of the liquid
    lab = clusters(p, box)
    sizes = np.bincount(lab)
    big = np.argmax(sizes)
    mem = p[lab == big]
    # cluster centre with periodic xy (circular mean)
    ang = mem[:, :2] / L * 2 * np.pi
    c_xy = (np.arctan2(np.sin(ang).mean(0), np.cos(ang).mean(0)) % (2 * np.pi)) / (2 * np.pi) * L
    c_z = mem[:, 2].mean()
    on_patch = bool(np.hypot(c_xy[0] - cx, c_xy[1] - cy) < meta['R_PATCH'] + 0.5 and c_z < zt + 1.5)
    degassed = int((p[:, 2] > z_surf + 0.3).sum())
    # bubble volume: empty grid cells between the wall and 0.5 nm below the free surface
    gx = np.arange(0.1, L, 0.2); gz = np.arange(zt + 0.2, z_surf - 0.5, 0.2)
    G = np.stack(np.meshgrid(gx, gx, gz, indexing='ij'), -1).reshape(-1, 3)
    ow_w = np.c_[o[:, 0] % L, o[:, 1] % L, o[:, 2] % box[2]]
    tw = cKDTree(np.minimum(ow_w, [L * (1 - 1e-7), L * (1 - 1e-7), 1e3]), boxsize=[L, L, 1e3])
    empty = np.array([len(x) == 0 for x in tw.query_ball_point(G, 0.32)])
    vol = float(empty.sum() * 0.2 ** 3)
    t = ts.time
    T = T0 + (T1 - T0) * min(t / 2000.0, 1.0)
    rows.append(dict(t_ps=float(t), T=T, n2_largest=int(sizes.max()), n2_clustered=int(sizes[sizes >= 4].sum()),
                     centre=[float(c_xy[0]), float(c_xy[1]), float(c_z)], on_patch=on_patch, degassed=degassed,
                     bubble_nm3=vol, z_surface=float(z_surf)))
json.dump(dict(meta=meta, frames=rows), open(f'{RUN}/analysis_{NAME}.json', 'w'), indent=1)
r = rows[-1]
print(f"{len(rows)} frames, last t={r['t_ps']:.0f} ps T={r['T']:.0f} K: largest N2 cluster {r['n2_largest']}, "
      f"centre {np.round(r['centre'], 2)}, on patch {r['on_patch']}, degassed {r['degassed']}, empty volume {r['bubble_nm3']:.2f} nm3")

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
t = np.array([x['t_ps'] for x in rows]) / 1000
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
ax[0].plot(t, [x['n2_largest'] for x in rows], label='largest N2 cluster')
ax[0].plot(t, [x['degassed'] for x in rows], label='N2 escaped above the water')
ax[0].set_xlabel('t (ns)'); ax[0].set_ylabel('molecules'); ax[0].legend(fontsize=8)
ax[1].plot(t, [x['bubble_nm3'] for x in rows]); ax[1].set_xlabel('t (ns)'); ax[1].set_ylabel('water-free volume (nm³)')
ax[2].plot(t, [x['T'] - 273.15 for x in rows]); ax[2].set_xlabel('t (ns)'); ax[2].set_ylabel('T (°C)')
fig.tight_layout(); fig.savefig(f'{RUN}/analysis_{NAME}.png', dpi=110)
