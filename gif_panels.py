"""Physics breakdown panels (cross-sections through the simulated flame), 2:3 portrait, dark theme."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from foamio import read_field, latest_time
from scipy.interpolate import griddata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CASE, OUT = sys.argv[1], sys.argv[2]
T0 = sys.argv[3] if len(sys.argv) > 3 else latest_time(CASE)
os.makedirs(OUT, exist_ok=True)
BG, INK, MUTED = '#0b0b0d', '#f0eee6', '#a6a39a'
W_PX, H_PX, DPI = 864, 1296, 200
RM, Z0, Z1 = 16.0, -2.0, 46.0            # mm, 2:3 window

h = np.radians(2.5)
r = read_field(f'{CASE}/0/Cx') / np.cos(h) * 1e3; z = read_field(f'{CASE}/0/Cz') * 1e3
T = read_field(f'{CASE}/{T0}/T'); U = read_field(f'{CASE}/{T0}/U')
YF = read_field(f'{CASE}/{T0}/C12H26'); YO = read_field(f'{CASE}/{T0}/O2')
rr = np.load(os.path.join(os.path.dirname(__file__), 'fields/rr.npy'))   # on (z, r) grid 0.05 mm
gr = np.linspace(-RM, RM, 321); gz = np.linspace(Z0, Z1, 481)
GR, GZ = np.meshgrid(gr, gz)
sel = (r < RM * 1.1) & (z < Z1 * 1.1)
pts = np.column_stack([np.concatenate([r[sel], -r[sel]]), np.concatenate([z[sel], z[sel]])])
def grid(v, flip_sign=False):
    vv = np.concatenate([v[sel], -v[sel] if flip_sign else v[sel]])
    return griddata(pts, vv, (GR, GZ), method='linear')
solid = ((np.abs(GR) < 0.75) & (GZ < 6)) | (GZ < 0)
def masked(a):
    a = a.copy(); a[solid] = np.nan; return a

heat = LinearSegmentedColormap.from_list('heat', ['#0b0b0d', '#5a2a12', '#a8461b', '#e0752e', '#f6b26b', '#fde8c8'])
blue = LinearSegmentedColormap.from_list('blue', ['#0b0b0d', '#12304f', '#2f6aa3', '#6fa8dc', '#e3f0fb'])
violet = LinearSegmentedColormap.from_list('vio', ['#0b0b0d', '#2b2350', '#5b4fa8', '#a79be0'])

def base(title, sub):
    fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.84]); ax.set_facecolor(BG)
    ax.set_xlim(-RM, RM); ax.set_ylim(Z0, Z1); ax.set_aspect('equal'); ax.axis('off')
    ax.add_patch(plt.Rectangle((-10, Z0 - 1), 20, -Z0 + 1, color='#6e1520', zorder=5))
    ax.add_patch(plt.Rectangle((-0.75, 0), 1.5, 6, color='#121110', zorder=6))
    fig.text(0.05, 0.955, title, color=INK, fontsize=11, weight='bold', va='top')
    fig.text(0.05, 0.915, sub, color=MUTED, fontsize=7.5, va='top')
    return fig, ax
def scalebar(ax):
    ax.plot([RM - 11, RM - 1], [Z1 - 2.5] * 2, color=MUTED, lw=1.2, zorder=9)
    ax.text(RM - 6, Z1 - 1.8, '10 mm', color=MUTED, fontsize=6.5, ha='center', zorder=9)
def cbar(fig, im, label, ticks=None):
    cax = fig.add_axes([0.05, 0.875, 0.38, 0.010]); cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=ticks)
    cb.outline.set_visible(False); cb.ax.tick_params(colors=MUTED, labelsize=6, length=2)
    fig.text(0.45, 0.88, label, color=MUTED, fontsize=6.5, va='center')

# 1 temperature
TG = masked(grid(T))
fig, ax = base('Temperature', 'Simulated gas temperature in a slice through the flame')
im = ax.imshow(TG, origin='lower', extent=[-RM, RM, Z0, Z1], cmap=heat, vmin=300, vmax=2000, interpolation='bilinear')
cs = ax.contour(GR, GZ, TG, levels=[1000, 1500], colors=['#a6a39a', '#fde8c8'], linewidths=0.6)
ax.clabel(cs, fmt='%d K', fontsize=5.5, colors=MUTED)
iT = np.nanargmax(TG); ax.plot(GR.flat[iT], GZ.flat[iT], 'o', ms=3, color=INK)
ax.annotate(f'peak {np.nanmax(TG):.0f} K', (GR.flat[iT], GZ.flat[iT]), (GR.flat[iT] + 3, GZ.flat[iT] + 4), color=INK, fontsize=7,
            arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6))
ax.annotate('cool core:\nwax vapour', (0, 9.5), (-12.5, 17), color=INK, fontsize=6.5, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6), zorder=8)
scalebar(ax); cbar(fig, im, 'K', [300, 1000, 1500, 2000])
fig.savefig(f'{OUT}/p_temperature.png', dpi=DPI, facecolor=BG); plt.close(fig)

# 2 flow
UrG = grid(U[:, 0], flip_sign=True); UzG = grid(U[:, 2]); sp = np.sqrt(UrG ** 2 + UzG ** 2)
fig, ax = base('Buoyant flow', 'Hot gas rises; fresh air is pulled in from the sides')
im = ax.imshow(masked(sp), origin='lower', extent=[-RM, RM, Z0, Z1], cmap=violet, vmin=0, vmax=np.nanmax(sp), interpolation='bilinear')
ax.streamplot(gr, gz, np.nan_to_num(masked(UrG)), np.nan_to_num(masked(UzG)), color='#e6e1ff', linewidth=0.5, density=1.1, arrowsize=0.5)
ax.text(-RM + 1, 4, 'air in', color=INK, fontsize=7); ax.text(RM - 1, 4, 'air in', color=INK, fontsize=7, ha='right')
ax.text(0, Z1 - 7, f'up to {np.nanmax(sp):.1f} m/s', color=INK, fontsize=7, ha='center')
scalebar(ax); cbar(fig, im, 'speed (m/s)')
fig.savefig(f'{OUT}/p_flow.png', dpi=DPI, facecolor=BG); plt.close(fig)

# 3 reaction zone: fuel vapour meets oxygen
FG = masked(grid(YF)); OG = masked(grid(YO))
rrg_r = np.arange(0.025, 15, 0.05); rrg_z = np.arange(0.025, 70, 0.05)
from scipy.interpolate import RegularGridInterpolator
fr = RegularGridInterpolator((rrg_z, rrg_r), rr, bounds_error=False, fill_value=0)
RRG = masked(fr(np.column_stack([np.clip(GZ.ravel(), 0, 70), np.abs(GR.ravel())])).reshape(GR.shape))
fig, ax = base('Reaction zone', 'Wax vapour (inside) meets oxygen (outside) at a thin flame sheet')
ax.imshow(FG, origin='lower', extent=[-RM, RM, Z0, Z1], cmap=LinearSegmentedColormap.from_list('f', ['#0b0b0d', '#3d3424', '#8a7448']),
          vmin=0, vmax=0.6, interpolation='bilinear')
im = ax.imshow(np.ma.masked_less(RRG, 0.02 * np.nanmax(RRG)), origin='lower', extent=[-RM, RM, Z0, Z1], cmap=blue,
               vmin=0, vmax=np.nanpercentile(RRG, 99.9), interpolation='bilinear')
ax.contour(GR, GZ, OG, levels=[0.12], colors=['#7fd1a6'], linewidths=0.6, linestyles='--')
ax.text(0, 14, 'fuel\nvapour', color='#d9c79a', fontsize=7, ha='center')
ax.text(8.5, 36, 'O₂', color='#7fd1a6', fontsize=8)
ax.annotate('flame sheet', (5.0, 12), (9.5, 20), color=INK, fontsize=7, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6))
scalebar(ax); cbar(fig, im, 'fuel burning rate', [])
fig.savefig(f'{OUT}/p_reaction.png', dpi=DPI, facecolor=BG); plt.close(fig)
print('panels done')
