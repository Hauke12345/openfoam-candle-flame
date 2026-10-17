import sys; sys.path.insert(0,'..'); sys.path.insert(0,'.')
import numpy as np, mech
from plots import load_case
from stage4_chemistry import Chemistry
from stage3_flamesheet import mixture_fraction
g,case,sol=load_case('out/stage4_M0_V0_h0.375.npz',Chemistry)
F=sol.fields(); T,Y=case.state({s:F[s] for s in case.scalars}); Z=mixture_fraction(Y)[0]
rho=mech.density(T,Y)[0]; mu=mech.transport(T,Y)[0][0]
uz=F['uz'][0]
# OF on 0.1mm grid
d=np.load('../data/steady_flame_t0.5.npz'); r=d['grid_r'].astype(float); z=d['grid_z'].astype(float)
Y5=np.array([d['grid_Y_'+s].astype(float) for s in mech.SPECIES[:5]]); Zo=mixture_fraction(Y5)
To=d['grid_T'].astype(float); Yo=mech.full_Y(np.nan_to_num(Y5)); muo=mech.transport(np.nan_to_num(To,nan=300),Yo)[0]
rhoo=d['grid_rho'].astype(float); uzo=d['grid_Uz'].astype(float)
print('supply 0.1 g/min = %.3e kg/s'%(0.1/60e3))
for zz in [0.004,0.008,0.012,0.02,0.03]:
    j=np.argmin(abs(g.zc-zz)); dzZ=(Z[j+1]-Z[j-1])/(g.zc[j+1]-g.zc[j-1])
    fl=g.fluid[j]
    fx=((rho[j]*uz[j]*Z[j]-mu[j]*dzZ)*2*np.pi*g.rc*g.dr)[fl].sum()
    jo=np.argmin(abs(z-zz)); dZo=(Zo[jo+1]-Zo[jo-1])/(2e-4); ok=np.isfinite(Zo[jo])&~d['grid_solid_wick'][jo]
    fo=np.nansum(((rhoo[jo]*uzo[jo]*Zo[jo]-muo[jo]*dZo)*2*np.pi*r*1e-4)[ok])
    print('z=%.0fmm  ours %.3e  OF(grid r<25mm) %.3e'%(zz*1e3,fx,fo))
