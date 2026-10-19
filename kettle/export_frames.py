"""Turn CFD snapshots into per-video-frame render inputs (f%05d.npz for render_kettle.py) along a timeline.

The heating takes minutes, bubbles move in milliseconds: the movie plays chosen windows in real time and
time-lapses the rest. TIMELINE rows: (t_start, t_end, sim seconds per video second).
usage: python3 export_frames.py RUNDIR OUTDIR [fps=24] [timeline=kettle/timeline.json]   -> OUTDIR/f%05d.npz + OUTDIR/frames.json
"""
import sys, os, json, glob
import numpy as np

RUN, OUT = sys.argv[1], sys.argv[2]
FPS = float(sys.argv[3]) if len(sys.argv) > 3 else 24.0
os.makedirs(OUT, exist_ok=True)
files = sorted(glob.glob(f'{RUN}/snap/s*.npz'))
t_snap = np.array([float(np.load(f)['sc'][0]) for f in files])
TL = json.load(open(sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.path.dirname(__file__), 'timeline.json')))  # [[t0, t1, speed, label]]

info = []
k = 0
for t0, t1, speed, label in TL:
    nfr = int(round((t1 - t0) / speed * FPS))
    for n in range(nfr):
        ts = t0 + n * speed / FPS
        i = int(np.argmin(np.abs(t_snap - ts)))
        d = np.load(files[i])
        sc = d['sc']
        t, Tm, h, vent_v, vent_g, on = (float(x) for x in sc[:6])
        boil = min(1.0, vent_v / 0.6e-3)
        A = 0.0035 * boil ** 0.7 + 0.0002
        steam = min(1.0, boil + max(0.0, (Tm - 80) / 20) * 0.25)
        np.savez(f'{OUT}/f{k:05d}.npz', xyz=d['xyz'], r=d['r'], h=h, A=A, phase=t, steam=steam, led=on)
        info.append(dict(frame=k, snap=i, t=t, Tm=Tm, label=label, speed=speed))
        k += 1
json.dump(info, open(f'{OUT}/frames.json', 'w'))
print('frames', k, f'{k / FPS:.1f} s video')
