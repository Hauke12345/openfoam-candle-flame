import cantera as ct, re
gri = {s.name.lower(): s for s in ct.Species.list_from_file('gri30.yaml')}
FUEL_TR = dict(geometry='nonlinear', well_depth=789.98, diameter=7.047, rotational_relaxation=1.0)  # LLNL n-dodecane
def est(s):
    c = s.composition.get('C',0); h=s.composition.get('H',0); o=s.composition.get('O',0)
    if s.name.startswith('A'):   # aromatics: scale from benzene
        return dict(geometry='nonlinear', well_depth=412.3+40*(c-6), diameter=5.349+0.25*(c-6), rotational_relaxation=1.0)
    n = c + 0.6*o
    return dict(geometry='nonlinear', well_depth=min(250+45*n, 790), diameter=4.2+0.24*n, rotational_relaxation=1.0)
def mk(d):
    t = ct.GasTransportData()
    t.set_customary_units(d['geometry'], d['diameter'], d['well_depth'], rotational_relaxation=d['rotational_relaxation'])
    return t
def with_transport(species):
    out=[]
    for s in species:
        n=s.name.lower()
        if n=='c12h26': s.transport = mk(FUEL_TR)
        elif n in gri: s.transport = gri[n].transport
        else: s.transport = mk(est(s))
        out.append(s)
    return out
def detailed():
    sp = with_transport(ct.Species.list_from_file('nDodecane_Reitz.yaml'))
    base = ct.Solution('nDodecane_Reitz.yaml')
    rx = ct.Reaction.list_from_file('nDodecane_Reitz.yaml', base)
    return ct.Solution(thermo='ideal-gas', kinetics='gas', transport_model='mixture-averaged', species=sp, reactions=rx)
NAMES=['c12h26','o2','n2','co','co2','h2o']
def twostep(A1, Ta1=None):
    all_sp = {s.name: s for s in with_transport(ct.Species.list_from_file('nDodecane_Reitz.yaml'))}
    sp=[all_sp[n] for n in NAMES]
    tmp = ct.Solution(thermo='ideal-gas', kinetics='gas', species=sp, reactions=[])
    Ea1 = 30000 if Ta1 is None else Ta1*1.98720425864083
    R=[ct.Reaction.from_yaml(y, tmp) for y in [
f"""equation: c12h26 + 12.5 o2 => 12 co + 13 h2o
rate-constant: {{A: {A1*1000**(1-1.75)}, b: 0, Ea: {Ea1} cal/mol}}
orders: {{c12h26: 0.25, o2: 1.5}}""",
f"""equation: co + 0.5 o2 + h2o => co2 + h2o
rate-constant: {{A: {3.98e14*1000**(1-1.75)}, b: 0, Ea: 40000 cal/mol}}
orders: {{co: 1.0, o2: 0.25, h2o: 0.5}}""",
"""equation: co2 => co + 0.5 o2
rate-constant: {A: 5.0e8, b: 0, Ea: 40000 cal/mol}"""]]
    return ct.Solution(thermo='ideal-gas', kinetics='gas', transport_model='mixture-averaged', species=sp, reactions=R)
