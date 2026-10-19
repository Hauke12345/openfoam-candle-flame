"""Bubble-wall close-up movie: captions, a colour legend and a picosecond clock over the rendered frames.
usage: python3 make_interface_movie.py OUT.mp4 FPS DIR:CASE [DIR:CASE ...]    CASE = air20 | steam98
"""
import sys, os, json, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

fdir = os.path.join(matplotlib.get_data_path(), 'fonts/ttf')
F1 = ImageFont.truetype(f'{fdir}/DejaVuSans-Bold.ttf', 25)
F2 = ImageFont.truetype(f'{fdir}/DejaVuSans.ttf', 17)
F3 = ImageFont.truetype(f'{fdir}/DejaVuSans.ttf', 15)
CAPS = {
    'air20': ('Inside an air bubble at 20 °C: the bubble wall up close',
              '1 atm of air is mostly empty space: N₂ and O₂ fly at ~500 m/s and bounce off the water surface'),
    'steam98': ('Inside a steam bubble at 98 °C (boiling)',
                'The gas is now ~94 % water vapour: molecules leave and re-enter the liquid all the time'),
}
LEG = [((235, 70, 55), 'water (liquid)'), ((255, 140, 110), 'water vapour'), ((60, 120, 255), 'N₂'),
       ((60, 210, 100), 'O₂'), ((170, 90, 255), 'Ar')]
out, fps = sys.argv[1], int(sys.argv[2])
items = []
for spec in sys.argv[3:]:
    d, case = spec.split(':')
    fr = json.load(open(os.path.join(d, 'frames.json')))['frames']
    t0 = fr[0]['t_ps']
    items += [(os.path.join(d, f'f{f["frame"]:05d}.png'), f['t_ps'] - t0, case) for f in fr
              if os.path.exists(os.path.join(d, f'f{f["frame"]:05d}.png'))]
W, H = Image.open(items[0][0]).size
proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                         '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', out],
                        stdin=subprocess.PIPE)
for p, t, case in items:
    im = Image.open(p).convert('RGB'); d = ImageDraw.Draw(im, 'RGBA')
    c1, c2 = CAPS[case]
    d.rectangle((0, 0, W, 70), fill=(5, 6, 10, 170))
    d.text((16, 8), c1, font=F1, fill=(242, 240, 232)); d.text((16, 40), c2, font=F2, fill=(192, 198, 208))
    x = 16
    d.rectangle((0, H - 36, W, H), fill=(5, 6, 10, 160))
    for col, lab in LEG:
        d.ellipse((x, H - 26, x + 14, H - 12), fill=col); d.text((x + 19, H - 29), lab, font=F3, fill=(214, 218, 226))
        x += 26 + int(d.textlength(lab, font=F3)) + 12
    d.text((x + 10, H - 29), f't = {t:5.2f} ps  (1 s of video = 1.2 ps; slowed ~10¹²×)', font=F3, fill=(214, 218, 226))
    d.text((W - 300, 78), 'nm-scale view of a ~1 mm bubble wall', font=F3, fill=(170, 176, 186))
    proc.stdin.write(np.asarray(im).tobytes())
proc.stdin.close(); proc.wait()
print('wrote', out, len(items), 'frames', f'{len(items) / fps:.1f} s')
