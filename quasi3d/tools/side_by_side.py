"""Side-by-side movie: OpenFOAM half-3D (left) vs quasi-3D (right), same renderer, same exposure per frame.
Frames are matched by time since the draft started (OpenFOAM t, quasi-3D t - 0.30).
usage: python3 side_by_side.py OF_FRAMEDIR QUASI_FRAMEDIR out.mp4 [t_end=0.2] [dt=0.01] [fps=6]
"""
import sys, os, glob, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import render3d as r

of_dir, q_dir, out = sys.argv[1:4]
t_end = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
dt = float(sys.argv[5]) if len(sys.argv) > 5 else 0.01
fps = int(sys.argv[6]) if len(sys.argv) > 6 else 6
T_ON = 0.30
XR = (-15, 35)

of_files = sorted(glob.glob(os.path.join(of_dir, '*.npz')))
of_t = np.array([float(np.load(f)['t']) for f in of_files])
q_files = sorted(glob.glob(os.path.join(q_dir, '*.npz')))
q_t = np.array([float(np.load(f)['t']) for f in q_files]) - T_ON

try:
    font = ImageFont.truetype('DejaVuSans.ttf', 15)
except OSError:
    font = ImageFont.load_default()
ff = r.find_ffmpeg()
proc = None
norm = None
for k, t in enumerate(np.arange(0, t_end + 1e-9, dt)):
    fo = of_files[int(np.argmin(abs(of_t - t)))]
    fq = q_files[int(np.argmin(abs(q_t - t)))]
    so, sq = r.load(fo), r.load(fq)
    _, no = r.render(so, 'side', xrange=XR, return_norm=True)
    _, nq = r.render(sq, 'side', xrange=XR, return_norm=True)
    n_now = tuple(np.exp(0.5 * (np.log(a) + np.log(b))) for a, b in zip(no, nq))
    norm = n_now if norm is None else tuple(np.exp(0.6 * np.log(a) + 0.4 * np.log(b)) for a, b in zip(norm, n_now))
    io = r.render(so, 'side', xrange=XR, norm=norm)
    iq = r.render(sq, 'side', xrange=XR, norm=norm)
    h, w = io.shape[:2]
    canvas = Image.new('RGB', (2 * w + 8, h + 30), (8, 10, 18))
    canvas.paste(Image.fromarray(io), (0, 30)); canvas.paste(Image.fromarray(iq), (w + 8, 30))
    d = ImageDraw.Draw(canvas)
    d.text((10, 6), 'OpenFOAM half-3D (119k cells)', fill=(230, 230, 230), font=font)
    d.text((w + 18, 6), 'quasi-3D, M = 4', fill=(230, 230, 230), font=font)
    d.text((10, h + 8), f'draft 0.2 m/s →   t = {1e3 * t:5.0f} ms after the draft starts', fill=(200, 200, 200), font=font)
    img = np.asarray(canvas)
    H, W = img.shape[:2]
    img = img[:H - H % 2, :W - W % 2]
    if proc is None:
        proc = subprocess.Popen([ff, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s',
                                 f'{img.shape[1]}x{img.shape[0]}', '-r', str(fps), '-i', '-', '-c:v', 'libx264',
                                 '-pix_fmt', 'yuv420p', '-crf', '18', out], stdin=subprocess.PIPE)
    proc.stdin.write(np.ascontiguousarray(img).tobytes())
    if k in (0, 5, 10, 20):
        Image.fromarray(img).save(out.replace('.mp4', f'_t{int(round(1e3 * t)):03d}.png'))
    print(f'frame {k}: t={t:.3f}  OF {os.path.basename(fo)}  quasi {os.path.basename(fq)}', flush=True)
proc.stdin.close(); proc.wait()
print('wrote', out)
