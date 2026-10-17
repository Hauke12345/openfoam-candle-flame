"""flame geometry (Z >= Zst region: tip, centroid, lean angles) of the OpenFOAM half-3D case.
usage: python3 of3d_flame.py CASE time [time ...]"""
import sys, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, os.path.dirname(HERE))
from foamio import read_field
from stage3_flamesheet import mixture_fraction

case = sys.argv[1]
C = read_field(f'{case}/0/C'); Vc = read_field(f'{case}/0/V')
x, y, z = C[:, 0], C[:, 1], C[:, 2]
for t in sys.argv[2:]:
    Y = np.array([read_field(f'{case}/{t}/{s}') for s in ['C12H26', 'O2', 'CO', 'CO2', 'H2O']])
    T = read_field(f'{case}/{t}/T')
    Z = mixture_fraction(Y)
    ins = (Z >= 0.0621)      # all fluid cells, same definition as stage3_flamesheet.flame_geometry
    w = Vc * ins
    k = np.argmax(np.where(ins, z, -1))
    xc, zc = (w * x).sum() / w.sum(), (w * z).sum() / w.sum()
    ang = lambda a, b: np.degrees(np.arctan2(a, b - 6e-3))
    print(f't={t}: [all fluid cells] tip z {z[k]*1e3:.1f} mm x {x[k]*1e3:.1f} mm ({ang(x[k], z[k]):.1f} deg); centroid x {xc*1e3:.2f} z {zc*1e3:.2f} mm '
          f'({ang(xc, zc):.1f} deg); vol {2*w.sum()*1e6:.2f} cm3; Tmax {T.max():.0f} K')
