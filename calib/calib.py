"""Calibrate the 2-step n-dodecane chemistry against the detailed mechanism with 1D freely propagating flames.

    python calib/calib.py det          # detailed nDodecane_Reitz mechanism (100 species): flame speed, T, equilibrium T
    python calib/calib.py 2s 9.6e11    # Westbrook-Dryer 2-step with pre-exponential A1 (cgs, as in the WD paper)

Conditions: n-dodecane/air, 470 K preheat (wax-vapour-like), 1 atm, phi = 0.8 / 1.0 / 1.3.
"""
import os, sys, time
import cantera as ct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mech import detailed, twostep

PHIS, T0 = (0.8, 1.0, 1.3), 470.0

def flame(gas, phi):
    gas.TP = T0, ct.one_atm
    gas.set_equivalence_ratio(phi, 'c12h26', 'o2:1, n2:3.76')
    f = ct.FreeFlame(gas, width=0.05)
    f.max_time_step_count = 5000
    f.set_refine_criteria(ratio=3, slope=0.3, curve=0.5)
    f.solve(loglevel=0, auto=True)
    f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
    f.solve(loglevel=0, refine_grid=True)
    return f.velocity[0], f.T.max()

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'det'
    gas = detailed() if mode == 'det' else twostep(float(sys.argv[2]))
    label = 'detailed' if mode == 'det' else f'2-step A1={float(sys.argv[2]):.3g}'
    for phi in PHIS:
        t = time.time(); s, T = flame(gas, phi)
        gas.TP = T0, ct.one_atm; gas.set_equivalence_ratio(phi, 'c12h26', 'o2:1, n2:3.76'); gas.equilibrate('HP')
        print(f'{label} phi={phi}: SL={s:.3f} m/s  Tmax={T:.0f} K  (equilibrium {gas.T:.0f} K, {time.time() - t:.0f} s)', flush=True)
