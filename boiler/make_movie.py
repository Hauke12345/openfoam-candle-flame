"""Assemble rendered MD frames into an MP4 with time, temperature and a phase caption.
usage: python3 make_movie.py OUT.mp4 FPS DIR1:T0:T1:SECTION [DIR2:T0:T1:SECTION ...]
  DIRk     frames from render_md.py (f*.png + frames.json with t_ps)
  T0, T1   annealing: temperature ramps linearly from T0 (K) at t = 0 to T1 at t = 2 ns, then holds
  SECTION  'degas' or 'boil' (selects the captions)
"""
import sys, os, json, glob, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

fdir = os.path.join(matplotlib.get_data_path(), 'fonts/ttf')
F1 = ImageFont.truetype(f'{fdir}/DejaVuSans-Bold.ttf', 24)
F2 = ImageFont.truetype(f'{fdir}/DejaVuSans.ttf', 17)
CAPS = {
    'degas': [(0.0, 'Heating the kettle: 20 → 95 °C', 'Dissolved air (blue, N₂) comes out of solution as the water warms'),
              (0.5, 'Gas gathers on the less-wetting spot', 'Amber patch = a scratch / scale spot; the glass-like wall stays wetted'),
              (1.2, 'A gas nanobubble sits on the patch', 'Water pulls back from the patch; some gas also escapes at the surface')],
    'boil': [(0.0, 'Superheating: water evaporates into the gas pocket', 'The trapped gas is the seed of the boiling bubble'),
             (2.0, 'The seed grows into a vapour bubble', 'Without a seed the same water would stay liquid much longer')],
}
out, fps = sys.argv[1], int(sys.argv[2])
items = []
t_off = 0.0
for spec in sys.argv[3:]:
    d, T0, T1, sec = spec.split(':'); T0, T1 = float(T0), float(T1)
    fr = json.load(open(os.path.join(d, 'frames.json')))
    for f in fr:
        p = os.path.join(d, f'f{f["frame"]:05d}.png')
        if os.path.exists(p):
            t = f['t_ps'] / 1000
            T = T0 + (T1 - T0) * min(t / 2.0, 1.0)
            items.append((p, t + t_off, T, sec, t))
    t_off = items[-1][1] if items else 0
im0 = Image.open(items[0][0]); W, H = im0.size
proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                         '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', out],
                        stdin=subprocess.PIPE)
for p, t, T, sec, tl in items:
    im = Image.open(p).convert('RGB')
    d = ImageDraw.Draw(im, 'RGBA')
    cap = [c for c in CAPS[sec] if tl >= c[0]][-1]
    d.rectangle((0, 0, W, 66), fill=(5, 6, 10, 170))
    d.text((16, 8), cap[1], font=F1, fill=(240, 238, 230))
    d.text((16, 38), cap[2], font=F2, fill=(190, 196, 205))
    d.rectangle((0, H - 34, W, H), fill=(5, 6, 10, 150))
    d.text((16, H - 28), f't = {t:5.2f} ns     T = {T - 273.15:5.1f} °C     box 5.1 nm wide · SPC/E water · N₂ · model wall',
           font=F2, fill=(210, 215, 222))
    proc.stdin.write(np.asarray(im).tobytes())
proc.stdin.close(); proc.wait()
print('wrote', out, len(items), 'frames', f'{len(items) / fps:.1f} s')
