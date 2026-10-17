"""Assemble the candle GIF: visible candle -> light sources -> physics panels -> back to the candle."""
import os, sys, numpy as np, matplotlib
from PIL import Image, ImageDraw, ImageFont

D = os.path.dirname(os.path.abspath(__file__))
BEAUTY, SOOT, CHEM = f'{D}/blend/candle_blender.png', f'{D}/blend/pass_soot.png', f'{D}/blend/pass_chem.png'
CROP = tuple(int(v) for v in sys.argv[1].split(','))        # x0,y0,x1,y1 in the full beauty frame
OUT = sys.argv[2]
VIDEO = OUT.endswith('.mp4')
GH = 1080 if VIDEO else 648; GW = GH * 2 // 3; K = GH / 648
fdir = os.path.join(matplotlib.get_data_path(), 'fonts/ttf')
F_T = ImageFont.truetype(f'{fdir}/DejaVuSans-Bold.ttf', round(19 * K))
F_S = ImageFont.truetype(f'{fdir}/DejaVuSans.ttf', round(12 * K))

full = Image.open(BEAUTY).convert('RGB')
FW, FH = full.size
crop_b = full.crop(CROP).resize((GW, GH), Image.LANCZOS)
crop_s = Image.open(SOOT).convert('RGB').resize((GW, GH), Image.LANCZOS)
crop_c = Image.open(CHEM).convert('RGB').resize((GW, GH), Image.LANCZOS)
panel = lambda n: Image.open(f'{D}/gif/{n}.png').convert('RGB').resize((GW, GH), Image.LANCZOS)

def caption(im, title, sub=None):
    im = im.copy()
    band = Image.new('RGBA', (GW, round((74 if sub else 50) * K)), (8, 8, 10, 170))
    im.paste(Image.alpha_composite(im.crop((0, 0, *band.size)).convert('RGBA'), band).convert('RGB'), (0, 0))
    d = ImageDraw.Draw(im)
    d.text((round(18 * K), round(14 * K)), title, font=F_T, fill=(240, 238, 230))
    if sub: d.text((round(18 * K), round(44 * K)), sub, font=F_S, fill=(196, 192, 182))
    return im

frames, durs = [], []
def hold(im, ms): frames.append(im); durs.append(ms)
def fade(a, b, n=7, ms=70):
    if VIDEO: n, ms = 2 * n + 1, ms * (n + 1) / (2 * n + 2)
    for k in range(1, n + 1):
        frames.append(Image.blend(a, b, k / (n + 1))); durs.append(ms)
def zoom(reverse=False, n=14, ms=70):
    if VIDEO: n, ms = 2 * n, ms / 2
    seq = []
    for k in range(n + 1):
        t = k / n; t = t * t * (3 - 2 * t)
        x0 = CROP[0] * t; y0 = CROP[1] * t
        x1 = FW + (CROP[2] - FW) * t; y1 = FH + (CROP[3] - FH) * t
        seq.append(full.crop((int(x0), int(y0), int(x1), int(y1))).resize((GW, GH), Image.LANCZOS))
    for im in (seq[::-1] if reverse else seq):
        frames.append(im); durs.append(ms)

full_small = full.resize((GW, GH), Image.LANCZOS)
A = caption(crop_b, 'What you see', 'The simulated flame, rendered in Blender')
S = caption(crop_s, 'Light source 1: glowing soot', 'Blackbody (Planck) light at the simulated 1500–1900 K')
C = caption(crop_c, 'Light source 2: chemiluminescence', 'Blue CH* (431 nm) + C₂* (516 nm) from the reaction zone')
B = caption(crop_b, 'Together', 'Yellow soot glow above, blue reaction zone at the base')
P1, P2, P3 = panel('p_reaction'), panel('p_temperature'), panel('p_flow')

import glob
ANIM = [Image.open(p).convert('RGB').resize((GW, GH), Image.LANCZOS) for p in sorted(glob.glob(f'{D}/blend/anim/f*.png'))]
def flicker(loops, ms=60):
    for _ in range(loops):
        for im in ANIM: frames.append(im); durs.append(ms)
flicker(2)
fade(ANIM[0], full_small, 3, 60)
zoom()
fade(crop_b, A, 4); hold(A, 1800)
fade(A, S); hold(S, 2400)
fade(S, C); hold(C, 2400)
fade(C, B); hold(B, 2000)
fade(B, P1); hold(P1, 2800)
fade(P1, P2); hold(P2, 2800)
fade(P2, P3); hold(P3, 2800)
fade(P3, crop_b, 8)
zoom(reverse=True)
fade(full_small, ANIM[0], 3, 60)
flicker(1)

if VIDEO:
    import subprocess
    FPS, PW = 30, GH * 4 // 5                     # 4:5 portrait canvas for the LinkedIn feed
    t_end = np.cumsum(durs) / 1000.0; total = t_end[-1]
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{PW}x{GH}', '-r', str(FPS), '-i', '-',
           '-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-movflags', '+faststart', OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    bgcache = {}
    for n in range(int(round(total * FPS))):
        k = min(int(np.searchsorted(t_end, (n + 0.5) / FPS)), len(frames) - 1)
        f = frames[k]
        if k not in bgcache:
            from PIL import ImageFilter, ImageEnhance
            bg = f.resize((PW, PW * GH // GW), Image.BILINEAR).crop((0, (PW * GH // GW - GH) // 2, PW, (PW * GH // GW - GH) // 2 + GH))
            bgcache[k] = ImageEnhance.Brightness(bg.filter(ImageFilter.GaussianBlur(40))).enhance(0.45)
        c = bgcache[k].copy(); c.paste(f, ((PW - GW) // 2, 0))
        p.stdin.write(c.tobytes())
    p.stdin.close(); p.wait()
    print(OUT, f'{total:.1f} s', f'{os.path.getsize(OUT) / 1e6:.1f} MB', f'{PW}x{GH}')
    sys.exit()

# one shared palette (sampled from all frames) -> stable colours, better compression
idx = list(range(0, len(frames), max(1, len(frames) // 24)))
mosaic = Image.new('RGB', (GW * 6, GH * ((len(idx) + 5) // 6)))
for n, i in enumerate(idx): mosaic.paste(frames[i], ((n % 6) * GW, (n // 6) * GH))
gpal = mosaic.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
pal = [f.quantize(palette=gpal, dither=Image.Dither.FLOYDSTEINBERG) for f in frames]
pal[0].save(OUT, save_all=True, append_images=pal[1:], duration=durs, loop=0, optimize=True, disposal=1)
print(OUT, len(frames), 'frames', f'{sum(durs) / 1000:.1f} s', f'{os.path.getsize(OUT) / 1e6:.1f} MB')
