"""Compose the rendered kettle frames with captions, a clock and an inset of the CFD fields; encode to MP4.
Inset: a cut through the kettle (axis in the middle): left half water temperature, right half dissolved air
(relative to air-saturated water at 20 °C).
usage: python3 make_kettle_movie.py OUT.mp4 FRAMEDIR(renders + frames.json) RUNDIR [fps=24]
"""
import sys, os, json, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob

out, FD, RUN = sys.argv[1], sys.argv[2], sys.argv[3]
FPS = int(sys.argv[4]) if len(sys.argv) > 4 else 24
fdir = os.path.join(matplotlib.get_data_path(), 'fonts/ttf')
F1 = ImageFont.truetype(f'{fdir}/DejaVuSans-Bold.ttf', 26)
F2 = ImageFont.truetype(f'{fdir}/DejaVuSans.ttf', 18)
F3 = ImageFont.truetype(f'{fdir}/DejaVuSans.ttf', 14)
CAPS = {
    'cold': ('Switched on: 20 °C tap water', 'The plate heats up; warm water rises from it and cold water sinks (convection)'),
    'degas': ('Dissolved air comes out', 'Warm water holds less air: tiny air bubbles grow on the plate and the glass'),
    'sing': ('The kettle "sings": early boiling', 'Steam bubbles form on the hot plate and collapse in the cooler water above'),
    'boil': ('Full boil', 'The water is at 100 °C: steam bubbles survive, grow and burst at the surface'),
    'off': ('Switched off', 'The steam thermostat cuts the power; boiling stops within seconds'),
}
info = json.load(open(f'{FD}/frames.json'))
snaps = sorted(glob.glob(f'{RUN}/snap/s*.npz'))

fig = plt.figure(figsize=(2.9, 2.5), dpi=100)
ax = fig.add_axes([0.02, 0.17, 0.96, 0.74])
cax1 = fig.add_axes([0.04, 0.08, 0.42, 0.04]); cax2 = fig.add_axes([0.54, 0.08, 0.42, 0.04])
fig.patch.set_alpha(0)

def inset(d):
    T = d['T'].astype(float); C = d['C'].astype(float) / 0.82
    ax.clear()
    NR, NZ = T.shape; R, H = 0.070, 0.110
    im1 = ax.imshow(T[::-1].T, extent=(-R * 100, 0, 0, H * 100), origin='lower', cmap='inferno', vmin=20, vmax=101)
    im2 = ax.imshow(C.T, extent=(0, R * 100, 0, H * 100), origin='lower', cmap='viridis', vmin=0.5, vmax=1.0)
    ax.axvline(0, color='w', lw=0.6)
    ax.set_xlim(-R * 100, R * 100); ax.set_ylim(0, H * 100); ax.axis('off')
    ax.set_title('CFD cut: temperature | dissolved air', fontsize=8, color='w')
    for c, im, lab in ((cax1, im1, '°C'), (cax2, im2, 'air (rel.)')):
        c.clear(); cb = fig.colorbar(im, cax=c, orientation='horizontal')
        cb.ax.tick_params(labelsize=6, colors='w', length=2); cb.set_label(lab, fontsize=6, color='w', labelpad=1)
    fig.canvas.draw()
    return Image.frombuffer('RGBA', fig.canvas.get_width_height(), bytes(fig.canvas.buffer_rgba()))

first = Image.open(f'{FD}/f{info[0]["frame"]:05d}.png'); W, H = first.size
proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                         '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', out],
                        stdin=subprocess.PIPE)
n = 0
for f in info:
    p = f'{FD}/f{f["frame"]:05d}.png'
    if not os.path.exists(p):
        continue
    im = Image.open(p).convert('RGB')
    d = np.load(snaps[f['snap']])
    dr = ImageDraw.Draw(im, 'RGBA')
    c1, c2 = CAPS[f['label']]
    dr.rectangle((0, 0, W, 74), fill=(6, 7, 10, 165))
    dr.text((18, 8), c1, font=F1, fill=(244, 240, 232)); dr.text((18, 42), c2, font=F2, fill=(196, 202, 210))
    ins = inset(d)
    bx, by = W - ins.size[0] - 14, 86
    dr.rounded_rectangle((bx - 6, by - 4, bx + ins.size[0] + 4, by + ins.size[1] + 2), 8, fill=(6, 7, 10, 150))
    im.paste(ins, (bx, by), ins)
    t = f['t']; sp = f['speed']
    mode = 'real time' if sp <= 1.01 else f'time-lapse ×{sp:g}'
    air = float(np.mean(d['C'].astype(float))) / 0.82 * 100
    dr.rectangle((0, H - 38, W, H), fill=(6, 7, 10, 150))
    dr.text((18, H - 30), f'{int(t // 60)}:{t % 60:04.1f}   ({mode})     water {f["Tm"]:5.1f} °C     dissolved air {air:3.0f} %'
            f'     bubbles < 0.5 mm drawn enlarged', font=F2, fill=(214, 218, 226))
    proc.stdin.write(np.asarray(im).tobytes()); n += 1
proc.stdin.close(); proc.wait()
print('wrote', out, n, 'frames', f'{n / FPS:.1f} s')
