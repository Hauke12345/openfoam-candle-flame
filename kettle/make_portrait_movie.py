"""Assemble the portrait (phone) cut from plan.json (export_portrait.py):
K  kettle render (upscaled to W x H)
T  CFD cross-section through the kettle axis: temperature + tracer dots that follow the computed flow (+ bubbles
   within 2 mm of the cut plane, true size)
X  macro of the chosen air bubble with a digital zoom into its lower wall, cross-faded with the molecular close-up
M  molecular close-up (transparent render) over a blurred, magnified macro of the same bubble
usage: python3 make_portrait_movie.py OUT.mp4 PLANDIR KETTLE_PNG_DIR RUNDIR MD_PNG_DIR [W=720] [H=1280]
"""
import sys, os, json, glob, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT, PD, KD, RUN, MDD = sys.argv[1:6]
W = int(sys.argv[6]) if len(sys.argv) > 6 else 720
H = int(sys.argv[7]) if len(sys.argv) > 7 else 1280
P = json.load(open(f'{PD}/plan.json')); FPS = P['fps']; plan = P['plan']
snaps = sorted(glob.glob(f'{RUN}/snap/s*.npz'))
mdf = sorted(glob.glob(f'{MDD}/f*.png'))
fdir = os.path.join(matplotlib.get_data_path(), 'fonts/ttf')
F1 = ImageFont.truetype(f'{fdir}/DejaVuSans-Bold.ttf', 38)
F2 = ImageFont.truetype(f'{fdir}/DejaVuSans.ttf', 25)
F3 = ImageFont.truetype(f'{fdir}/DejaVuSans.ttf', 22)
CAPS = {
    'cold': ('Switched on: 20 °C', ['The plate under the water', 'heats up first']),
    'degas': ('Air comes out of the water', ['Warm water holds less air.', 'Bubbles at true size: most < 0.5 mm']),
    'cut': ('Cut through the kettle (CFD)', ['Hot water rises from the plate,', 'cool water sinks: turbulent plumes.', 'Colour = temperature, dots follow the flow']),
    'zoom': ('One air bubble on the glass', ['0.4 mm across, true size']),
    'zoomin': ('Into the bottom of that bubble', ['about 100 000 × closer']),
    'md': ('The bubble wall, molecule by molecule', ['Water below, air above (1 atm):', 'N₂ blue, O₂ green, Ar violet,', 'water vapour glowing. 1 s = 1.2 ps']),
    'zoomout': ('Back out', ['']),
    'zoomback': ('Back out', ['']),
    'sing': ('The kettle "sings"', ['Steam bubbles form on the plate', 'and collapse in the cooler water']),
    'boil': ('Full boil: 100 °C', ['Steam bubbles survive, merge', 'and burst at the surface']),
    'off': ('Switched off', ['The steam thermostat cuts the power']),
}

def fit(im):
    return im.convert('RGB').resize((W, H), Image.LANCZOS)

def overlay(im, label, bottom):
    d = ImageDraw.Draw(im, 'RGBA')
    c1, c2 = CAPS[label]
    hh = 70 + 34 * len(c2)
    d.rectangle((0, 0, W, hh), fill=(6, 7, 10, 160))
    d.text((24, 16), c1, font=F1, fill=(245, 241, 233))
    for i, l in enumerate(c2):
        d.text((24, 68 + 34 * i), l, font=F2, fill=(198, 204, 212))
    if bottom:
        d.rectangle((0, H - 52, W, H), fill=(6, 7, 10, 150))
        d.text((24, H - 40), bottom, font=F3, fill=(216, 220, 228))
    return im

def clock(e):
    t = e['t']; sp = e['speed']
    mode = 'real time' if sp <= 1.01 else f'time-lapse ×{sp:g}'
    return f'{int(t // 60)}:{t % 60:04.1f}  ({mode})   water {e.get("Tm", 0):5.1f} °C'

# ---------------- CFD cut with tracers ----------------
R, HH = 0.070, 0.110
rng = np.random.default_rng(3)
NTR = 1400
tr = np.c_[rng.uniform(-R, R, NTR), rng.uniform(0, HH, NTR)]
trail = [tr.copy()]
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
fig.patch.set_facecolor((0.03, 0.035, 0.05))
axc = fig.add_axes([0.04, 0.30, 0.92, 0.42]); cax = fig.add_axes([0.2, 0.25, 0.6, 0.012])

def vel(d, x, z):
    u = d['u'].astype(float); w = d['w'].astype(float); NR, NZ = u.shape
    fr = np.clip(np.abs(x) / (R / NR) - 0.5, 0, NR - 1.001); fz = np.clip(z / (HH / NZ) - 0.5, 0, NZ - 1.001)
    i0 = fr.astype(int); j0 = fz.astype(int); a = fr - i0; b = fz - j0
    bil = lambda F: F[i0, j0] * (1 - a) * (1 - b) + F[i0 + 1, j0] * a * (1 - b) + F[i0, j0 + 1] * (1 - a) * b + F[i0 + 1, j0 + 1] * a * b
    return bil(u) * np.sign(x), bil(w)

def cut_frame(e, prev_snap):
    global tr, trail
    for si in range(prev_snap + 1, e['snap'] + 1) if prev_snap is not None else [e['snap']]:
        d = np.load(snaps[si]); ux, uz = vel(d, tr[:, 0], tr[:, 1])
        tr = tr + np.c_[ux, uz] / FPS
        out = (np.abs(tr[:, 0]) > R) | (tr[:, 1] < 0) | (tr[:, 1] > HH)
        tr[:, 0] = np.clip(tr[:, 0], -R, R); tr[:, 1] = np.clip(tr[:, 1], 0, HH)
        n = rng.random(NTR) < 0.004                                   # slowly refresh dots (keeps them spread)
        re = out | n
        tr[re] = np.c_[rng.uniform(-R, R, re.sum()), rng.uniform(0, HH, re.sum())]
    trail = (trail + [tr.copy()])[-6:]
    d = np.load(snaps[e['snap']]); T = d['T'].astype(float)
    axc.clear(); cax.clear()
    full = np.r_[T[::-1], T]
    im = axc.imshow(full.T, extent=(-R * 100, R * 100, 0, HH * 100), origin='lower', cmap='inferno', vmin=40, vmax=100,
                    interpolation='bicubic', aspect='equal')
    pts = np.array(trail)                                            # (n_trail, NTR, 2)
    for k in range(len(trail) - 1):
        seg_ok = np.hypot(*(pts[k + 1] - pts[k]).T) < 0.005
        axc.plot(np.c_[pts[k, seg_ok, 0], pts[k + 1, seg_ok, 0]].T * 100, np.c_[pts[k, seg_ok, 1], pts[k + 1, seg_ok, 1]].T * 100,
                 color='w', lw=0.6, alpha=0.15 + 0.12 * k, solid_capstyle='round')
    axc.scatter(tr[:, 0] * 100, tr[:, 1] * 100, s=1.5, c='w', alpha=0.85, lw=0)
    xyz = d['xyz'].astype(float); rb = d['r'].astype(float)
    m = np.abs(xyz[:, 1]) < 0.002
    for (x, _, z), r in zip(xyz[m], rb[m]):
        axc.add_patch(plt.Circle((x * 100, z * 100), max(r * 100, 0.02), fill=False, ec=(0.75, 0.9, 1.0), lw=0.7))
    axc.set_xlim(-R * 100, R * 100); axc.set_ylim(-0.5, HH * 100 + 0.3)
    axc.axhspan(-0.5, 0, color=(0.55, 0.57, 0.6))
    axc.axis('off')
    cb = fig.colorbar(im, cax=cax, orientation='horizontal'); cb.ax.tick_params(labelsize=14, colors='w')
    cb.set_label('water temperature (°C)', color='w', fontsize=15)
    axc.text(0, -1.4, 'heating plate', color='w', ha='center', fontsize=13)
    fig.canvas.draw()
    return Image.frombuffer('RGBA', fig.canvas.get_width_height(), bytes(fig.canvas.buffer_rgba())).convert('RGB')

# ---------------- macro <-> molecules ----------------
ZC = (0.5, 0.5 + 0.105)          # bubble's lower wall in the macro frame (fractions of width / height)
def zoom_into(im, z):
    w, h = im.size; cw, ch = w / z, h / z
    cx = min(max(ZC[0] * w, cw / 2), w - cw / 2); cy = min(max(ZC[1] * h, ch / 2), h - ch / 2)
    box = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
    return im.resize((W, H), Image.LANCZOS, box=box)

macro_bg = None
def md_over(i, bg):
    m = Image.open(mdf[min(i, len(mdf) - 1)]).convert('RGBA').resize((W, H), Image.LANCZOS)
    out = bg.copy().convert('RGBA'); out.alpha_composite(m)
    return out.convert('RGB')

proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                         '-r', str(int(FPS)), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19', OUT],
                        stdin=subprocess.PIPE)
prev_cut = None; last_macro = None; n_md = sum(1 for e in plan if e['kind'] == 'M'); n = 0
for e in plan:
    if e['kind'] in 'KX':
        p = f'{KD}/f{e["render"]:05d}.png'
        if not os.path.exists(p):
            continue
        im = Image.open(p).convert('RGB')
    if e['kind'] == 'K':
        frame = overlay(fit(im), e['label'], clock(e))
    elif e['kind'] == 'T':
        frame = overlay(cut_frame(e, prev_cut), 'cut', clock(e) + '  (cross-section)'); prev_cut = e['snap']
    elif e['kind'] == 'X':
        s = e['s']
        if e['label'] == 'zoomin':
            z = np.exp(np.log(1.0) * (1 - s) + np.log(6.0) * s); a = max(0.0, (s - 0.35) / 0.65)
            md_i = 0
        else:
            z = np.exp(np.log(6.0) * (1 - s) + np.log(1.0) * s); a = max(0.0, 1 - s / 0.65)
            md_i = n_md - 1
        base = zoom_into(im, z)
        bg = zoom_into(im, 6.0).filter(ImageFilter.GaussianBlur(14))
        last_macro = bg
        frame = Image.blend(base, md_over(md_i, bg), a) if a > 0 else base
        frame = overlay(frame, 'zoomin' if e['label'] == 'zoomin' else 'zoomout', None)
    else:                                                          # M
        frame = overlay(md_over(e['md'], last_macro), 'md', f't = {e["md"] * 0.05:5.2f} ps  (slowed ~10¹²×)')
    proc.stdin.write(np.asarray(frame).tobytes()); n += 1
proc.stdin.close(); proc.wait()
print('wrote', OUT, n, 'frames', f'{n / FPS:.1f} s')
