"""Build the GROMACS system: water with dissolved N2 on a solid wall with a less-wetting patch.

Model (nm, kJ/mol, combination rule 2 = Lorentz-Berthelot unless overridden in [ nonbond_params ]):
  water  SPC/E (OW sigma 0.316557, eps 0.650194; q_O -0.8476, q_H +0.4238; SETTLE)
  N2     one Lennard-Jones site (sigma 0.3698, eps 0.7903 = 95.05 K), mass 28.013
  wall   3 frozen layers of a square lattice (a = 0.30 nm) of uncharged LJ atoms:
         WP  glass-like (wetting): strong water attraction
         WH  less-wetting patch (disk of radius R_PATCH at the box centre): weak water attraction,
             slightly stronger gas attraction — a stand-in for a scratch / scale spot on the kettle wall
  layout: wall at the bottom, water slab above it, free space on top (free surface keeps p ~ 0).
usage: python3 build.py OUTDIR [Lxy_nm=5.0] [water_nm=5.0] [x_N2=0.025]
"""
import sys, os, math
import numpy as np

OUT = sys.argv[1]
LXY = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
HW = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
XN2 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.025
A = 0.30                       # wall lattice spacing
NLAY = 3
R_PATCH = 1.2
Z_WALL0 = 0.30
z_wall_top = Z_WALL0 + (NLAY - 1) * A
Z_WAT0 = z_wall_top + 0.30     # start of the water slab
LZ = Z_WAT0 + HW + 6.0         # free space above the water
os.makedirs(OUT, exist_ok=True)

# pair interactions with the wall (sigma, epsilon); tune wettability here
SIG_W = 0.32
PAIRS = {('OW', 'WP'): (0.318, 1.30),     # glass-like: water wets
         ('OW', 'WH'): (0.318, 0.30),     # patch: weak water attraction (less wetting)
         ('N2', 'WP'): (0.345, 0.60),
         ('N2', 'WH'): (0.345, 1.00),
         ('WP', 'WP'): (SIG_W, 0.0), ('WH', 'WH'): (SIG_W, 0.0), ('WP', 'WH'): (SIG_W, 0.0)}

n = int(round(LXY / A))
LXY = n * A
wall = []
for k in range(NLAY):
    for i in range(n):
        for j in range(n):
            x, y, z = (i + 0.5) * A, (j + 0.5) * A, Z_WALL0 + k * A
            t = 'WH' if math.hypot(x - LXY / 2, y - LXY / 2) < R_PATCH else 'WP'
            wall.append((t, x, y, z))
nwh = sum(1 for w in wall if w[0] == 'WH')

def gro_line(resnr, resname, atomname, atnr, x, y, z):
    return f'{resnr % 100000:5d}{resname:<5s}{atomname:>5s}{atnr % 100000:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n'

# wall-only gro (water and gas are added with gmx insert-molecules / solvate)
with open(f'{OUT}/wall.gro', 'w') as f:
    f.write('wall\n'); f.write(f'{len(wall)}\n')
    for k, (t, x, y, z) in enumerate(wall):
        f.write(gro_line(k + 1, 'WAL', t, k + 1, x, y, z))
    f.write(f'{LXY:10.5f}{LXY:10.5f}{LZ:10.5f}\n')
with open(f'{OUT}/n2.gro', 'w') as f:
    f.write('N2\n1\n'); f.write(gro_line(1, 'N2', 'N2', 1, 0, 0, 0)); f.write('   1.0   1.0   1.0\n')

# number of molecules: water density 33.4 /nm3 in the slab volume, N2 mole fraction XN2
nwat_est = int(33.4 * LXY * LXY * HW)
nn2 = int(round(XN2 / (1 - XN2) * nwat_est))

itp_atoms = """
[ atomtypes ]
; name  at.num  mass     charge  ptype  sigma      epsilon
  OW    8       15.9994  0.0     A      0.316557   0.650194
  HW    1       1.008    0.0     A      0.0        0.0
  N2    7       28.0134  0.0     A      0.3698     0.7903
  WP    14      28.0855  0.0     A      {sw}       0.0
  WH    14      28.0855  0.0     A      {sw}       0.0

[ nonbond_params ]
; i   j   func  sigma  epsilon
""".format(sw=SIG_W) + ''.join(f'  {a:<3s} {b:<3s} 1  {s:.4f}  {e:.4f}\n' for (a, b), (s, e) in PAIRS.items()) + """
[ moleculetype ]
SOL 2
[ atoms ]
1 OW 1 SOL OW 1 -0.8476 15.9994
2 HW 1 SOL HW1 1 0.4238 1.008
3 HW 1 SOL HW2 1 0.4238 1.008
[ settles ]
1 1 0.1 0.16330
[ exclusions ]
1 2 3
2 1 3
3 1 2

[ moleculetype ]
N2 1
[ atoms ]
1 N2 1 N2 N2 1 0.0 28.0134

[ moleculetype ]
WAL 1
[ atoms ]
1 WP 1 WAL WP 1 0.0 28.0855

[ moleculetype ]
WALH 1
[ atoms ]
1 WH 1 WAL WH 1 0.0 28.0855
"""
open(f'{OUT}/model.itp', 'w').write('[ defaults ]\n; nbfunc comb-rule gen-pairs\n  1      2         no\n' + itp_atoms)

# wall atoms must be listed per moleculetype in gro order: write gro sorted by type blocks
wall.sort(key=lambda w: 0 if w[0] == 'WP' else 1)
with open(f'{OUT}/wall.gro', 'w') as f:
    f.write('wall\n'); f.write(f'{len(wall)}\n')
    for k, (t, x, y, z) in enumerate(wall):
        f.write(gro_line(k + 1, 'WAL', t, k + 1, x, y, z))
    f.write(f'{LXY:10.5f}{LXY:10.5f}{LZ:10.5f}\n')

meta = dict(LXY=LXY, LZ=LZ, Z_WAT0=Z_WAT0, HW=HW, z_wall_top=z_wall_top, R_PATCH=R_PATCH, n_wp=len(wall) - nwh,
            n_wh=nwh, n_n2=nn2, n_wat_est=nwat_est, pairs={f'{a}-{b}': v for (a, b), v in PAIRS.items()})
import json
json.dump(meta, open(f'{OUT}/meta.json', 'w'), indent=1)
print(json.dumps(meta))
