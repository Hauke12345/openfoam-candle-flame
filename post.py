"""Post-process the candle case: temperature/reaction-rate images, animation, and sanity numbers."""
import os, sys, glob, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from foamio import read_field
from scipy.interpolate import griddata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as anim

CASE = sys.argv[1]; OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
h = np.radians(2.5)
r = read_field(f'{CASE}/0/Cx') / np.cos(h); z = read_field(f'{CASE}/0/Cz')
times = sorted([d for d in os.listdir(CASE) if d.replace('.', '').replace('e-', '').isdigit() and float(d) > 0], key=float)
last = times[-1]

# single-hue warm sequential ramp on a dark surface (dark -> light = cold -> hot)
BG, INK, MUTED = '#141413', '#f0eee6', '#a6a39a'
cmap = LinearSegmentedColormap.from_list('heat', ['#1d1a17', '#5a2a12', '#a8461b', '#e0752e', '#f6b26b', '#fde8c8'])
cmap_rr = LinearSegmentedColormap.from_list('rr', ['#1d1a17', '#1b3550', '#2f6aa3', '#6fa8dc', '#d6e8f7'])

RMAX, ZMAX = 15e-3, 70e-3
gr = np.linspace(-RMAX, RMAX, 301); gz = np.linspace(0, ZMAX, 701)
GR, GZ = np.meshgrid(gr, gz)
sel = (r < RMAX * 1.1) & (z < ZMAX * 1.1)
pts = np.column_stack([np.concatenate([r[sel], -r[sel]]), np.concatenate([z[sel], z[sel]])])
def grid(v):
    return griddata(pts, np.concatenate([v[sel], v[sel]]), (GR, GZ), method='linear')
wick = (np.abs(GR) < 0.75e-3) & (GZ < 6e-3)

def fields(t):
    T = read_field(f'{CASE}/{t}/T'); U = read_field(f'{CASE}/{t}/U')
    Y = {s: read_field(f'{CASE}/{t}/{s}') for s in ['C12H26', 'O2', 'CO', 'CO2', 'H2O']}
    return T, U, Y

def fuel_rate(T, Y):
    # Westbrook-Dryer fuel oxidation rate (as in constant/reactions), kmol/m3/s -> kg/m3/s
    W = {'C12H26': 170.34, 'O2': 32.0, 'CO': 28.01, 'CO2': 44.01, 'H2O': 18.02, 'N2': 28.01}
    YN2 = 1 - sum(Y.values())
    Wmix = 1 / (sum(Y[s] / W[s] for s in Y) + YN2 / W['N2'])
    rho = 101325 * Wmix / (8314.47 * T)
    cF = np.clip(rho * Y['C12H26'] / W['C12H26'], 0, None); cO = np.clip(rho * Y['O2'] / W['O2'], 0, None)
    A = 9.6e11 * 1000 ** (1 - 1.75)
    return A * cF ** 0.25 * cO ** 1.5 * np.exp(-30000 / 1.98720425864083 / T) * W['C12H26'], rho

def style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel('r (mm)', color=MUTED, fontsize=9)
    ax.set_aspect('equal')

def draw_candle(ax):
    ax.add_patch(plt.Rectangle((-10, -8), 20, 8, color='#e8e2d4', zorder=5))
    ax.add_patch(plt.Rectangle((-0.75, 0), 1.5, 6, color='#2b2a27', zorder=6))

# ---------- final-state figure ----------
T, U, Y = fields(last)
rr, rho = fuel_rate(T, Y)
TG = grid(T); RRG = grid(rr); TG[wick] = np.nan; RRG[wick] = np.nan
UrG = grid(U[:, 0]); UzG = grid(U[:, 2])
UrG = np.where(GR < 0, -UrG, UrG)

fig, axs = plt.subplots(1, 2, figsize=(9, 9.6), facecolor=BG)
ext = [-RMAX * 1e3, RMAX * 1e3, 0, ZMAX * 1e3]
im = axs[0].imshow(TG, origin='lower', extent=ext, cmap=cmap, vmin=300, vmax=2300)
axs[0].streamplot(GR[0] * 1e3, GZ[:, 0] * 1e3, np.nan_to_num(UrG), np.nan_to_num(UzG), color='#ffffff', linewidth=0.5,
                  density=0.9, arrowsize=0.6)
axs[0].set_title('Temperature and flow', color=INK, fontsize=11, loc='left')
cb = fig.colorbar(im, ax=axs[0], fraction=0.05, pad=0.03); cb.set_label('K', color=MUTED); cb.ax.tick_params(colors=MUTED, labelsize=8)
im2 = axs[1].imshow(RRG, origin='lower', extent=ext, cmap=cmap_rr, vmin=0, vmax=np.nanpercentile(RRG, 99.8))
axs[1].contour(GR * 1e3, GZ * 1e3, TG, levels=[1500], colors=['#f6b26b'], linewidths=0.8)
axs[1].set_title('Reaction zone (fuel burning rate)', color=INK, fontsize=11, loc='left')
cb2 = fig.colorbar(im2, ax=axs[1], fraction=0.05, pad=0.03); cb2.set_label('kg/m³/s', color=MUTED); cb2.ax.tick_params(colors=MUTED, labelsize=8)
for ax in axs:
    style(ax); draw_candle(ax); ax.set_ylim(-8, ZMAX * 1e3); ax.set_xlim(-RMAX * 1e3, RMAX * 1e3)
axs[0].set_ylabel('height above wax (mm)', color=MUTED, fontsize=9)
axs[1].text(-14, 66, 'orange line: 1500 K', color='#f6b26b', fontsize=8)
fig.suptitle(f'2D axisymmetric candle flame, OpenFOAM, t = {float(last):.2f} s', color=INK, fontsize=12, x=0.06, ha='left')
fig.tight_layout()
fig.savefig(f'{OUT}/candle_final.png', dpi=150, facecolor=BG)

# ---------- numbers ----------
dV = r ** 2 * np.tan(h) * 0  # placeholder (volume computed below)
axis = r < 0.15e-3
zax = z[axis]; Tax = T[axis]; o = np.argsort(zax); zax, Tax = zax[o], Tax[o]
hot = zax[Tax > 1500]
# visible flame height proxy: highest point where fuel is still being consumed significantly
rrmax = rr.max(); burning = z[rr > 0.05 * rrmax]
stats = dict(time=float(last), Tmax=float(T.max()), z_Tmax_mm=float(z[np.argmax(T)] * 1e3), r_Tmax_mm=float(r[np.argmax(T)] * 1e3),
             axis_T_gt1500_mm=(float(hot.min() * 1e3), float(hot.max() * 1e3)) if len(hot) else None,
             reaction_zone_top_mm=float(burning.max() * 1e3), Umax=float(np.linalg.norm(U, axis=1).max()),
             unburnt_fuel_at_top=float(Y['C12H26'][z > 0.115].max()))
print(stats)

# ---------- animation ----------
frames = times[::1]
fig, ax = plt.subplots(figsize=(4.2, 8.2), facecolor=BG)
style(ax); draw_candle(ax)
im = ax.imshow(np.full_like(GR, np.nan), origin='lower', extent=ext, cmap=cmap, vmin=300, vmax=2300)
ax.set_ylim(-8, ZMAX * 1e3); ax.set_xlim(-RMAX * 1e3, RMAX * 1e3)
ax.set_ylabel('height (mm)', color=MUTED, fontsize=9)
ttl = ax.set_title('', color=INK, fontsize=10, loc='left')
def upd(i):
    t = frames[i]
    Tg = grid(read_field(f'{CASE}/{t}/T')); Tg[wick] = np.nan
    im.set_data(Tg); ttl.set_text(f'Temperature, t = {float(t):.2f} s')
    return [im, ttl]
fig.tight_layout()
a = anim.FuncAnimation(fig, upd, frames=len(frames), blit=False)
a.save(f'{OUT}/candle_ignition.gif', writer=anim.PillowWriter(fps=10), dpi=90)
print('frames', len(frames))
