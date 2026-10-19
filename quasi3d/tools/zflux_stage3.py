import sys; sys.path.insert(0,'..'); sys.path.insert(0,'.')
import numpy as np
from plots import load_case
from stage3_flamesheet import FlameSheet
f=sys.argv[1]
g,case,sol=load_case(f,FlameSheet,Sc=1.0)
F=sol.fields(); Z=F['Z'][0]; rho=case.lookup('rho',Z); mu=case.lookup('mu',Z); uz=F['uz'][0]
for zz in [0.004,0.008,0.012,0.02,0.03]:
    j=np.argmin(abs(g.zc-zz)); dzZ=(Z[j+1]-Z[j-1])/(g.zc[j+1]-g.zc[j-1]); fl=g.fluid[j]
    a=(rho[j]*uz[j]*Z[j]*2*np.pi*g.rc*g.dr)[fl].sum(); b=(-mu[j]*dzZ*2*np.pi*g.rc*g.dr)[fl].sum()
    print('z=%.0fmm conv %.3e diff %.3e total %.3e'%(zz*1e3,a,b,a+b))
