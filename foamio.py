import re, numpy as np, os

def read_field(path):
    """Read an ASCII OpenFOAM volField internalField (uniform or nonuniform, scalar or vector)."""
    txt = open(path).read()
    m = re.search(r'internalField\s+uniform\s+([^;]+);', txt)
    if m:
        v = m.group(1).strip()
        return ('uniform', np.array([float(x) for x in v.strip('()').split()]) if v.startswith('(') else float(v))
    m = re.search(r'internalField\s+nonuniform\s+List<(\w+)>\s*\n?(\d+)\s*\n?\(', txt)
    kind, n = m.group(1), int(m.group(2))
    body = txt[m.end():]
    if kind == 'scalar':
        arr = np.array(body.split(None, n)[:n], dtype=float)
    else:
        nums = re.findall(r'[-+0-9.eE]+', body.split('\n)\n', 1)[0] if False else body[:body.index(';')])
        arr = np.array(nums[:3 * n], dtype=float).reshape(n, 3)
    return arr

def load(case, time, names, ncell):
    out = {}
    for nm in names:
        f = read_field(os.path.join(case, time, nm))
        if isinstance(f, tuple):
            val = f[1]
            out[nm] = np.full(ncell, val) if np.isscalar(val) else np.tile(val, (ncell, 1))
        else:
            out[nm] = f
    return out

def latest_time(case):
    """Name of the latest written time directory (> 0) of a reconstructed case."""
    ts = []
    for d in os.listdir(case):
        try:
            if float(d) > 0 and os.path.isfile(os.path.join(case, d, 'T')): ts.append(d)
        except ValueError:
            pass
    return max(ts, key=float)
