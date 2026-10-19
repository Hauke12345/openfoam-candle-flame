"""Molecular close-up of a real bubble wall: flat water surface + gas at 1 atm with the real composition.

A 1 mm bubble is flat at the nm scale, so the model is a water slab (8 x 8 x 4 nm, SPC/E) with a 40 nm gas column
(periodic in all directions, two interfaces). The gas is at 1 atm total pressure and temperature T:
  air bubble   (20 C): water vapour at p_v(T), the rest dry air: 78.1 % N2, 20.9 % O2, 0.93 % Ar
                       (CO2 0.04 % would be 0.03 molecules here: none)
  steam bubble (98 C): the same rule gives ~94 % H2O vapour and a few N2 / O2.
Gas models: TraPPE-like 2-site N2 (sigma 0.331 nm, eps 36 K, bond 0.110 nm) and O2 (0.302 nm, 49 K, 0.121 nm),
1-site Ar (0.3405 nm, 119.8 K); Lorentz-Berthelot mixing with SPC/E water.
Steps (run by this script): solvate the slab, equilibrate it alone (NVT, 50 ps), add the gas molecules at random
positions (no overlaps), fresh Maxwell velocities, 5 ps settling, then a production run written every 0.05 ps.
usage: python3 build_interface.py RUNDIR T_C [prod_ps=40] [threads=4]   (gmx on PATH)
"""
import sys, os, json, subprocess
import numpy as np

RUN = os.path.abspath(sys.argv[1]); TC = float(sys.argv[2])
PROD = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0
NT = sys.argv[4] if len(sys.argv) > 4 else '4'
os.makedirs(RUN, exist_ok=True)
TK = TC + 273.15
LX, SLAB, GAS = 8.0, 4.0, 40.0
LZ = SLAB + GAS
Z0 = GAS / 2                      # slab from Z0 to Z0 + SLAB; gas on both sides (periodic)
rng = np.random.default_rng(int(TC * 10) + 1)

def gmx(*a, inp=None):
    r = subprocess.run(['gmx', '-quiet'] + [str(x) for x in a], cwd=RUN, input=inp, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-3000:]); raise SystemExit('gmx failed: ' + ' '.join(map(str, a)))
    return r

# ---- composition: 1 atm, vapour at the (real) saturation pressure ----
pv = 133.322 * 10 ** (8.07131 - 1730.63 / (233.426 + TC))
n_tot = 101325.0 / (1.380649e-23 * TK) * (LX * LX * GAS * 1e-27)
x_h2o = min(pv / 101325.0, 0.99)
dry = 1 - x_h2o
N = {'H2O': x_h2o * n_tot, 'N2': dry * 0.781 * n_tot, 'O2': dry * 0.2095 * n_tot, 'AR': dry * 0.0093 * n_tot}
cnt = {k: int(np.floor(v + rng.random())) for k, v in N.items()}     # stochastic rounding
print('gas molecules', {k: round(v, 2) for k, v in N.items()}, '->', cnt)

# ---- topology ----
K = 0.0083144626
itp = f"""[ defaults ]
  1  2  no
[ atomtypes ]
  OW   8  15.9994  0.0  A  0.316557  0.650194
  HW   1   1.008   0.0  A  0.0       0.0
  NN   7  14.0067  0.0  A  0.331     {36.0 * K:.5f}
  OO   8  15.9994  0.0  A  0.302     {49.0 * K:.5f}
  AR  18  39.948   0.0  A  0.3405    {119.8 * K:.5f}

[ moleculetype ]
SOL 2
[ atoms ]
1 OW 1 SOL OW  1 -0.8476 15.9994
2 HW 1 SOL HW1 1  0.4238  1.008
3 HW 1 SOL HW2 1  0.4238  1.008
[ settles ]
1 1 0.1 0.16330
[ exclusions ]
1 2 3
2 1 3
3 1 2

[ moleculetype ]
N2 1
[ atoms ]
1 NN 1 N2 N1 1 0.0 14.0067
2 NN 1 N2 N2 1 0.0 14.0067
[ constraints ]
1 2 1 0.110

[ moleculetype ]
O2 1
[ atoms ]
1 OO 1 O2 O1 1 0.0 15.9994
2 OO 1 O2 O2 1 0.0 15.9994
[ constraints ]
1 2 1 0.121

[ moleculetype ]
AR 1
[ atoms ]
1 AR 1 AR AR 1 0.0 39.948
"""
open(f'{RUN}/model.itp', 'w').write(itp)

common = f"""cutoff-scheme = Verlet
pbc = xyz
coulombtype = PME
rcoulomb = 1.2
vdwtype = Cut-off
rvdw = 1.2
DispCorr = no
tcoupl = v-rescale
tc-grps = System
ref-t = {TK:.2f}
comm-mode = Linear
"""

def read_gro(p):
    L = open(p).read().splitlines(); n = int(L[1])
    return [(l[5:10].strip(), l[10:15].strip(), float(l[20:28]), float(l[28:36]), float(l[36:44])) for l in L[2:2 + n]]

def write_gro(p, mols, title):
    lines = []; r = 0
    for resname, atoms in mols:
        r += 1
        for name, x, y, z in atoms:
            lines.append(f'{r % 100000:5d}{resname:<5s}{name:>5s}{(len(lines) + 1) % 100000:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n')
    with open(p, 'w') as f:
        f.write(title + '\n' + f'{len(lines)}\n'); f.writelines(lines); f.write(f'{LX:10.5f}{LX:10.5f}{LZ:10.5f}\n')

def top(p, counts):
    open(p, 'w').write('#include "model.itp"\n[ system ]\nbubble wall\n[ molecules ]\n' +
                       ''.join(f'{k} {v}\n' for k, v in counts if v))

# ---- 1. water slab, equilibrated alone ----
if not os.path.exists(f'{RUN}/slab_eq.gro'):
    gmx('solvate', '-cs', 'spc216.gro', '-box', LX, LX, SLAB, '-o', 'wat.gro')
    wat = read_gro(f'{RUN}/wat.gro')
    mols = [('SOL', [(a[1], a[2], a[3], a[4] + Z0) for a in wat[i:i + 3]]) for i in range(0, len(wat), 3)]
    write_gro(f'{RUN}/slab.gro', mols, 'water slab')
    top(f'{RUN}/slab.top', [('SOL', len(mols))])
    open(f'{RUN}/em.mdp', 'w').write('integrator = steep\nemtol = 500\nnsteps = 3000\n' + ''.join(l + '\n' for l in common.splitlines() if not l.startswith(('tcoupl', 'tc-grps', 'ref-t'))))
    open(f'{RUN}/slab_eq.mdp', 'w').write(f'integrator = md\ndt = 0.002\nnsteps = 25000\ngen-vel = yes\ngen-temp = {TK:.2f}\n'
                                          'tau-t = 0.1\nnstxout-compressed = 0\nnstenergy = 1000\n' + common)
    gmx('grompp', '-f', 'em.mdp', '-c', 'slab.gro', '-p', 'slab.top', '-o', 'slab_em', '-maxwarn', '1')
    gmx('mdrun', '-deffnm', 'slab_em', '-nt', NT)
    gmx('grompp', '-f', 'slab_eq.mdp', '-c', 'slab_em.gro', '-p', 'slab.top', '-o', 'slab_eq', '-maxwarn', '1')
    gmx('mdrun', '-deffnm', 'slab_eq', '-nt', NT)

# ---- 2. add the gas ----
slab = read_gro(f'{RUN}/slab_eq.gro')
sol = [('SOL', [(a[1], a[2] % LX, a[3] % LX, a[4]) for a in slab[i:i + 3]]) for i in range(0, len(slab), 3)]
oz = np.array([m[1][0][3] for m in sol])
z_lo, z_hi = np.percentile(oz, 1) - 0.6, np.percentile(oz, 99) + 0.6        # keep the gas off the liquid surface
placed = []
def spot():
    while True:
        p = np.array([rng.random() * LX, rng.random() * LX, rng.random() * LZ])
        if z_lo < p[2] < z_hi:
            continue
        if all(np.linalg.norm((p - q + [LX / 2, LX / 2, LZ / 2]) % [LX, LX, LZ] - [LX / 2, LX / 2, LZ / 2]) > 1.0 for q in placed):
            placed.append(p); return p
def rot():
    v = rng.standard_normal(3); return v / np.linalg.norm(v)
w0 = sol[0][1]; wO = np.array(w0[0][1:]); wH = [np.array(a[1:]) - wO for a in w0[1:]]
gas = {'N2': [], 'O2': [], 'AR': [], 'SOL': []}
for _ in range(cnt['N2']):
    p = spot(); e = rot() * 0.055; gas['N2'].append(('N2', [('N1', *(p - e)), ('N2', *(p + e))]))
for _ in range(cnt['O2']):
    p = spot(); e = rot() * 0.0605; gas['O2'].append(('O2', [('O1', *(p - e)), ('O2', *(p + e))]))
for _ in range(cnt['AR']):
    p = spot(); gas['AR'].append(('AR', [('AR', *p)]))
for _ in range(cnt['H2O']):
    p = spot()
    q = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    gas['SOL'].append(('SOL', [('OW', *p)] + [(n, *(p + q @ h)) for n, h in zip(('HW1', 'HW2'), wH)]))
order = [('N2', gas['N2']), ('O2', gas['O2']), ('AR', gas['AR']), ('SOL', sol + gas['SOL'])]
write_gro(f'{RUN}/conf.gro', [m for _, ms in order for m in ms], f'bubble wall {TC:g} C')
top(f'{RUN}/topol.top', [(k, len(ms)) for k, ms in order])
json.dump(dict(T_C=TC, LX=LX, LZ=LZ, slab=[float(z_lo + 0.6), float(z_hi - 0.6)], counts=cnt, expected=N,
               n_water_liquid=len(sol)), open(f'{RUN}/meta.json', 'w'), indent=1)

# ---- 3. settle 5 ps, then production every 0.05 ps ----
open(f'{RUN}/settle.mdp', 'w').write(f'integrator = md\ndt = 0.002\nnsteps = 2500\ngen-vel = yes\ngen-temp = {TK:.2f}\n'
                                     'tau-t = 0.5\nnstenergy = 500\n' + common)
open(f'{RUN}/prod.mdp', 'w').write(f'integrator = md\ndt = 0.002\nnsteps = {int(PROD / 0.002)}\ncontinuation = yes\n'
                                   'tau-t = 2.0\nnstxout-compressed = 25\ncompressed-x-precision = 10000\nnstenergy = 500\n' + common)
gmx('grompp', '-f', 'settle.mdp', '-c', 'conf.gro', '-p', 'topol.top', '-o', 'settle', '-maxwarn', '1')
gmx('mdrun', '-deffnm', 'settle', '-nt', NT)
gmx('grompp', '-f', 'prod.mdp', '-c', 'settle.gro', '-t', 'settle.cpt', '-p', 'topol.top', '-o', 'prod', '-maxwarn', '1')
gmx('mdrun', '-deffnm', 'prod', '-nt', NT)
print('done', RUN)
