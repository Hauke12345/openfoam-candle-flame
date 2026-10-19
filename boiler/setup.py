"""Assemble conf.gro + topol.top + mdp files for the kettle-wall degassing run.
usage: python3 setup.py RUNDIR [Lxy] [water_nm] [x_N2]   (needs gmx on PATH)
"""
import sys, os, json, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.abspath(sys.argv[1])
args = sys.argv[2:]
subprocess.run([sys.executable, os.path.join(HERE, 'build.py'), RUN] + args, check=True)
meta = json.load(open(f'{RUN}/meta.json'))
L, HW, Z0 = meta['LXY'], meta['HW'], meta['Z_WAT0']
gmx = lambda *a: subprocess.run(['gmx', '-quiet'] + list(a), cwd=RUN, check=True, capture_output=True, text=True)
gmx('solvate', '-cs', 'spc216.gro', '-box', f'{L}', f'{L}', f'{HW}', '-o', 'wat.gro')


def read_gro(p):
    L_ = open(p).read().splitlines()
    n = int(L_[1])
    at = [(l[5:10].strip(), l[10:15].strip(), float(l[20:28]), float(l[28:36]), float(l[36:44])) for l in L_[2:2 + n]]
    return at


wat = read_gro(f'{RUN}/wat.gro')
mols = [wat[i:i + 3] for i in range(0, len(wat), 3)]
rng = np.random.default_rng(1)
nn2 = meta['n_n2']
pick = set(rng.choice(len(mols), nn2, replace=False).tolist())
n2 = [(m[0][2], m[0][3], m[0][4] + Z0) for k, m in enumerate(mols) if k in pick]
sol = [m for k, m in enumerate(mols) if k not in pick]
wall = read_gro(f'{RUN}/wall.gro')
nwp = sum(1 for a in wall if a[1] == 'WP'); nwh = len(wall) - nwp

lines = []
def add(resnr, resname, name, x, y, z):
    lines.append(f'{resnr % 100000:5d}{resname:<5s}{name:>5s}{(len(lines) + 1) % 100000:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n')
r = 0
for a in wall:
    r += 1; add(r, 'WAL', a[1], *a[2:])
for x, y, z in n2:
    r += 1; add(r, 'N2', 'N2', x, y, z)
for m in sol:
    r += 1
    for a in m:
        add(r, 'SOL', a[1], a[2], a[3], a[4] + Z0)
with open(f'{RUN}/conf.gro', 'w') as f:
    f.write('kettle wall: water + dissolved N2\n'); f.write(f'{len(lines)}\n'); f.writelines(lines)
    f.write(f'{L:10.5f}{L:10.5f}{meta["LZ"]:10.5f}\n')

open(f'{RUN}/topol.top', 'w').write(f"""#include "{HERE}/{os.path.relpath(RUN + '/model.itp', HERE)}"
[ system ]
kettle wall: water + dissolved N2
[ molecules ]
WAL  {nwp}
WALH {nwh}
N2   {len(n2)}
SOL  {len(sol)}
""".replace(f'{HERE}/{os.path.relpath(RUN + "/model.itp", HERE)}', 'model.itp'))

# index: wall group (frozen) and fluid group
nwall = nwp + nwh
with open(f'{RUN}/index.ndx', 'w') as f:
    def grp(name, a, b):
        f.write(f'[ {name} ]\n'); ids = list(range(a, b + 1))
        for i in range(0, len(ids), 15): f.write(' '.join(map(str, ids[i:i + 15])) + '\n')
    nat = len(lines)
    grp('System', 1, nat); grp('Wall', 1, nwall); grp('Fluid', nwall + 1, nat)
    grp('N2', nwall + 1, nwall + len(n2)); grp('Water', nwall + len(n2) + 1, nat)

common = """
cutoff-scheme   = Verlet
pbc             = xyz
coulombtype     = PME
rcoulomb        = 1.0
vdwtype         = Cut-off
rvdw            = 1.0
DispCorr        = no
ewald-geometry  = 3dc
freezegrps      = Wall
freezedim       = Y Y Y
"""
open(f'{RUN}/em.mdp', 'w').write('integrator = steep\nemtol = 500\nnsteps = 5000\n' + common)
open(f'{RUN}/eq.mdp', 'w').write(f"""integrator = md
dt         = 0.002
nsteps     = 50000          ; 100 ps at 20 C
tcoupl     = v-rescale
tc-grps    = Fluid Wall
tau-t      = 0.5 0.5
ref-t      = 293 293
gen-vel    = yes
gen-temp   = 293
nstxout-compressed = 5000
nstenergy  = 500
comm-mode  = none
""" + common)
open(f'{RUN}/heat.mdp', 'w').write(f"""integrator = md
dt         = 0.002
nsteps     = 15000000       ; 30 ns: heat 20 -> 95 C over 2 ns, then hold (the kettle)
tcoupl     = v-rescale
tc-grps    = Fluid Wall          ; the wall is frozen (no degrees of freedom); listed for completeness
tau-t      = 0.5 0.5
ref-t      = 293 293
annealing  = single single
annealing-npoints = 2 2
annealing-time  = 0 2000 0 2000
annealing-temp  = 293 368 293 368
nstxout-compressed = 5000   ; every 10 ps
compressed-x-grps  = System
nstenergy  = 5000
nstcalcenergy = 100
comm-mode  = none
""" + common)
print('atoms', len(lines), 'water', len(sol), 'N2', len(n2), 'wall', nwp, '+', nwh)
