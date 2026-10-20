"""Portrait (phone) cut: kettle close-up with true-size bubbles, a CFD cross-section cut, a zoom onto one air bubble
on the glass, the molecular close-up of its wall, and back out to the boil.

Writes OUT/f%05d.npz (render inputs for render_kettle.py, with a per-frame camera) and OUT/plan.json, the list of
video frames for make_portrait_movie.py: kind K (kettle render), T (CFD cut), M (molecular frame), X (cross-fade
kettle <-> molecules, with a digital zoom into the bubble's lower wall).
usage: python3 export_portrait.py RUNDIR OUT [fps=24]
"""
import sys, os, json, glob
import numpy as np

RUN, OUT = sys.argv[1], sys.argv[2]
FPS = float(sys.argv[3]) if len(sys.argv) > 3 else 24.0
os.makedirs(OUT, exist_ok=True)
files = sorted(glob.glob(f'{RUN}/snap/s*.npz'))
snap_of = lambda t: min(len(files) - 1, max(0, int(round(t * FPS))))      # snapshots are written at 24 Hz

R_W = 0.0695                                   # water radius in the render (0.5 mm air gap to the glass)
BUB = np.array([0.0, -(R_W - 0.9 * 0.206e-3), 0.063])     # the air bubble on the front glass (attached 120-174 s)
WIDE = dict(cam=(-0.10, -1.00, 0.26), tgt=(0.0, 0.0, 0.085), fstop=5.6)
CLOSE = dict(cam=(0.0, -0.40, 0.075), tgt=(0.0, -0.02, 0.058), fstop=5.6)
MACRO = dict(cam=tuple(BUB + [0.0, -0.005, 0.0006]), tgt=tuple(BUB), fstop=128.0)

def ease(s):
    s = min(max(s, 0.0), 1.0); return s * s * (3 - 2 * s)

Y_OUT = -0.0735                                # camera never closer to the axis than this (glass outer face -0.073)

def cam_between(a, b, s):
    """target moves linearly; the camera stays in front of the glass and its stand-off distance changes
    geometrically, so the zoom feels uniform from 0.33 m down to 0.8 mm."""
    s = ease(s)
    ta, tb = np.array(a['tgt']), np.array(b['tgt']); ca, cb = np.array(a['cam']), np.array(b['cam'])
    t = ta + (tb - ta) * s
    da, db = Y_OUT - ca[1], Y_OUT - cb[1]
    d = np.exp(np.log(da) * (1 - s) + np.log(db) * s)
    w = (da - d) / (da - db) if da != db else 1.0
    cam = np.array([ca[0] + (cb[0] - ca[0]) * w, Y_OUT - d, ca[2] + (cb[2] - ca[2]) * w])
    dist = np.linalg.norm(cam - t)
    # the glass (3 mm, n = 1.5) and the air gap make objects behind it look ~1 mm closer: refocus
    # zooming: focus on the bubble's plane on the glass (which looks ~1 mm closer through the glass); else on the target
    focus = np.linalg.norm(cam - BUB) - 1.0e-3 if a is not b else dist
    return dict(cam=tuple(cam), tgt=tuple(t), fstop=float(np.exp(np.log(a['fstop']) * (1 - s) + np.log(b['fstop']) * s)), focus=focus)

# (kind, t0, t1, speed, label, camera(a, b) or None)
SEG = [('K', 0, 16, 8, 'cold', (WIDE, CLOSE)),
       ('K', 16, 136, 20, 'degas', (CLOSE, CLOSE)),
       ('K', 136, 139, 1, 'degas', (CLOSE, CLOSE)),
       ('T', 139, 159, 5, 'cut', None),
       ('K', 159, 163, 1, 'zoom', (CLOSE, MACRO)),
       ('X', 163, 164.5, 1, 'zoomin', (MACRO, MACRO)),       # macro, then digital zoom + cross-fade into the molecules
       ('M', 0, 10, 1, 'md', None),                          # 10 s of the 20 C bubble-wall MD (1 s = 1.2 ps)
       ('X', 164.5, 165.5, 1, 'zoomout', (MACRO, MACRO)),
       ('K', 165.5, 169.5, 1, 'zoomback', (MACRO, CLOSE)),
       ('K', 169.5, 268, 20, 'sing', (CLOSE, CLOSE)),
       ('K', 268, 271, 1, 'sing', (CLOSE, CLOSE)),
       ('K', 271, 291, 10, 'boil', (CLOSE, CLOSE)),
       ('K', 291, 300, 1, 'boil', (CLOSE, CLOSE)),
       ('K', 302, 317, 3, 'off', (CLOSE, CLOSE))]

plan, k = [], 0
for kind, t0, t1, speed, label, cams in SEG:
    n = int(round((t1 - t0) / speed * FPS))
    for i in range(n):
        s = i / max(n - 1, 1)
        if kind == 'M':
            plan.append(dict(kind='M', md=i, label=label)); continue
        t = t0 + i * speed / FPS
        si = snap_of(t)
        if kind == 'T':
            plan.append(dict(kind='T', snap=si, t=t, speed=speed, label=label)); continue
        d = np.load(files[si]); sc = d['sc']
        Tm, h, vent_v, on = float(sc[1]), float(sc[2]), float(sc[3]), float(sc[5])
        boil = min(1.0, vent_v / 0.6e-3)
        cam = cam_between(cams[0], cams[1], s)
        xyz = d['xyz'].astype(float); r = d['r'].astype(float)
        rr = np.hypot(xyz[:, 0], xyz[:, 1]); lim = R_W - 0.9 * r                 # keep wall bubbles inside the water
        f = np.where(rr > lim, lim / np.maximum(rr, 1e-9), 1.0); xyz[:, 0] *= f; xyz[:, 1] *= f
        np.savez(f'{OUT}/f{k:05d}.npz', xyz=xyz, r=r, rmin=1e-5, h=h, A=0.0035 * boil ** 0.7 + 0.0002, phase=t,
                 steam=min(1.0, boil + max(0.0, (Tm - 80) / 20) * 0.25), led=on,
                 cam=np.array(cam['cam']), tgt=np.array(cam['tgt']), lens=70.0, fstop=cam['fstop'], focus=cam['focus'])
        e = dict(kind=kind, render=k, snap=si, t=t, Tm=Tm, speed=speed, label=label)
        if kind == 'X':
            e['phase'] = label; e['s'] = s
        plan.append(e); k += 1
json.dump(dict(fps=FPS, bubble=BUB.tolist(), plan=plan), open(f'{OUT}/plan.json', 'w'))
print('video frames', len(plan), f'({len(plan) / FPS:.1f} s), kettle renders', k)
