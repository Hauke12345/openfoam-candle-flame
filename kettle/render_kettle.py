"""Photoreal glass electric kettle (Blender bpy 5.x, Cycles CPU) for the boiling-water bubble simulation.

Scene units: metres, origin at the centre of the inner bottom (top of the heating plate), z up.
The scene is built once; per frame only the bubble point cloud (Geometry Nodes sphere instancing), the water
surface mesh and a few scalar parameters (LED, steam, mist) are updated.

Surface displacement (travelling-wave sum):  z(x,y) = h + A * sum_k a_k sin(k_k . x - w_k t + phi_k) / (2 sigma),
with w_k = c sqrt(g |k|) and sigma = sqrt(sum a_k^2 / 2) (so std(z - h) = A/2).
Water: Principled BSDF transmission, IOR 1.33 + Volume Absorption. Bubbles: IOR 1/1.33 = 0.75 (air in water).

usage: python3 render_kettle.py FRAMEDIR OUTDIR [RX=1280] [RY=720] [SPP=32] [first=0] [last=-1] [stride=1]
       python3 render_kettle.py --test OUTPNG [RX=1280] [RY=720] [SPP=32]
FRAMEDIR holds f%05d.npz with xyz (N,3) m, r (N,) m, h, A, phase, steam (0..1), led (0..1).
"""
import sys, os, glob, math, time
import numpy as np
import bpy, bmesh
from mathutils import Vector

# ----------------------------------------------------------------------------------------------------- constants
R_IN = 0.070            # glass inner radius
R_W = 0.0695            # water radius (small air gap to the glass)
T_GLASS = 0.003
H_JUG = 0.200
Z_PLATE = -0.004
R_BODY = 0.0755
Z_BODY0 = -0.034
Z_BASE0 = -0.054
R_BASE = 0.098
MIN_R_RENDER = 1.2e-4   # bubble radii are clamped to this for rendering
GRID_N = 94             # water surface grid: 94 x 94 points mapped to the disk (~1.5 mm spacing)

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene


# ----------------------------------------------------------------------------------------------------- helpers
def smooth_by_angle(bm, deg=35):
    """face smooth + sharp edges where the dihedral angle exceeds deg."""
    for f in bm.faces:
        f.smooth = True
    for e in bm.edges:
        if len(e.link_faces) == 2 and e.calc_face_angle(0.0) > math.radians(deg):
            e.smooth = False


def to_object(name, bm, mat=None, deg=35):
    smooth_by_angle(bm, deg)
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob)
    if mat is not None:
        ob.data.materials.append(mat)
    return ob


def fillet(poly, rc, n=5):
    """round every corner of a closed polygon [(r,z),..] with a quadratic Bezier of 'radius' rc."""
    P = np.asarray(poly, float); out = []
    for i in range(len(P)):
        p, a, b = P[i], P[i - 1], P[(i + 1) % len(P)]
        da, db = a - p, b - p
        la, lb = np.linalg.norm(da), np.linalg.norm(db)
        t = min(rc, la / 2.01, lb / 2.01)
        p1, p2 = p + da / la * t, p + db / lb * t
        for s in np.linspace(0, 1, n):
            out.append((1 - s) ** 2 * p1 + 2 * s * (1 - s) * p + s * s * p2)
    return np.array(out)


def revolve(name, prof, mat, nseg=160, deg=35, rc=0.0, z0=0.0):
    """solid of revolution of the closed (r,z) polygon around the z axis."""
    prof = fillet(prof, rc) if rc > 0 else np.asarray(prof, float)
    bm = bmesh.new(); m = len(prof)
    ang = np.linspace(0, 2 * math.pi, nseg, endpoint=False)
    V = [[bm.verts.new((r * math.cos(a), r * math.sin(a), z)) for (r, z) in prof] for a in ang]
    for i in range(nseg):
        for k in range(m):
            try:
                bm.faces.new((V[i][k], V[(i + 1) % nseg][k], V[(i + 1) % nseg][(k + 1) % m], V[i][(k + 1) % m]))
            except ValueError:
                pass
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return to_object(name, bm, mat, deg)


def tube(name, path, mat, rw=0.0045, rt=0.0035, nc=16):
    """swept elliptical tube along a polyline in the x-z plane (rw across y, rt in the plane)."""
    P = np.asarray(path, float)
    for _ in range(4):   # Chaikin smoothing
        Q = [P[0]]
        for a, b in zip(P[:-1], P[1:]):
            Q += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        Q.append(P[-1]); P = np.array(Q)
    P = np.stack([P[:, 0], np.zeros(len(P)), P[:, 1]], 1)
    T = np.gradient(P, axis=0); T /= np.linalg.norm(T, axis=1)[:, None]
    Y = np.array([0, 1.0, 0]); N = np.cross(Y, T); N /= np.linalg.norm(N, axis=1)[:, None]
    bm = bmesh.new(); th = np.linspace(0, 2 * math.pi, nc, endpoint=False); rings = []
    for p, n in zip(P, N):
        rings.append([bm.verts.new(p + rw * math.cos(t) * Y + rt * math.sin(t) * n) for t in th])
    for i in range(len(rings) - 1):
        for j in range(nc):
            bm.faces.new((rings[i][j], rings[i][(j + 1) % nc], rings[i + 1][(j + 1) % nc], rings[i + 1][j]))
    bm.faces.new(rings[0][::-1]); bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return to_object(name, bm, mat, 50)


def hull(name, pts, mat):
    bm = bmesh.new()
    for p in pts:
        bm.verts.new(p)
    bmesh.ops.convex_hull(bm, input=bm.verts)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return to_object(name, bm, mat, 40)


# ----------------------------------------------------------------------------------------------------- materials
def nodes_of(m):
    m.use_nodes = True
    return m.node_tree.nodes, m.node_tree.links


def shadow_mix(m, shader_out, nodes, links, out):
    """camera rays see the dielectric; shadow rays see a transparent surface (light reaches the interior)."""
    lp = nodes.new('ShaderNodeLightPath'); tr = nodes.new('ShaderNodeBsdfTransparent')
    mx = nodes.new('ShaderNodeMixShader')
    links.new(lp.outputs['Is Shadow Ray'], mx.inputs[0])
    links.new(shader_out, mx.inputs[1]); links.new(tr.outputs[0], mx.inputs[2])
    links.new(mx.outputs[0], out.inputs['Surface'])


def principled(name, base, rough=0.4, metal=0.0, **kw):
    m = bpy.data.materials.new(name); n, l = nodes_of(m)
    p = n['Principled BSDF']
    p.inputs['Base Color'].default_value = (*base, 1); p.inputs['Roughness'].default_value = rough
    p.inputs['Metallic'].default_value = metal
    for k, v in kw.items():
        p.inputs[k.replace('_', ' ')].default_value = v
    return m


def dielectric(name, ior, tint, rough=0.0):
    m = bpy.data.materials.new(name); n, l = nodes_of(m)
    p = n['Principled BSDF']; out = n['Material Output']
    p.inputs['Base Color'].default_value = (*tint, 1); p.inputs['Roughness'].default_value = rough
    p.inputs['Transmission Weight'].default_value = 1.0; p.inputs['IOR'].default_value = ior
    shadow_mix(m, p.outputs[0], n, l, out)
    return m, n, l, out


mat_glass, *_ = dielectric('glass', 1.47, (0.93, 0.98, 0.96))
mat_water, wn, wl, wout = dielectric('water', 1.33, (1, 1, 1))
ab = wn.new('ShaderNodeVolumeAbsorption'); ab.inputs['Color'].default_value = (0.72, 0.90, 1.0, 1)
ab.inputs['Density'].default_value = 3.0; wl.new(ab.outputs[0], wout.inputs['Volume'])
mat_bubble, *_ = dielectric('bubble', 0.75, (1, 1, 1), 0.0)

# brushed stainless plate: concentric ring bump + roughness variation
mat_plate = principled('plate', (0.75, 0.76, 0.78), 0.28, 1.0)
n, l = nodes_of(mat_plate)
tc = n.new('ShaderNodeTexCoord'); wv = n.new('ShaderNodeTexWave'); wv.wave_type = 'RINGS'
wv.inputs['Scale'].default_value = 450; wv.inputs['Distortion'].default_value = 3
bp = n.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = 0.08; bp.inputs['Distance'].default_value = 0.0005
l.new(tc.outputs['Object'], wv.inputs['Vector']); l.new(wv.outputs['Fac'], bp.inputs['Height'])
l.new(bp.outputs['Normal'], n['Principled BSDF'].inputs['Normal'])
ramp = n.new('ShaderNodeMapRange'); ramp.inputs['To Min'].default_value = 0.22; ramp.inputs['To Max'].default_value = 0.4
l.new(wv.outputs['Fac'], ramp.inputs['Value']); l.new(ramp.outputs[0], n['Principled BSDF'].inputs['Roughness'])

mat_body = principled('body', (0.03, 0.03, 0.034), 0.38, 0.0, Specular_IOR_Level=0.6)
mat_base = principled('base', (0.008, 0.008, 0.009), 0.3, 0.0)
mat_knob = principled('knob', (0.02, 0.02, 0.022), 0.25, 0.0)
mat_steel = principled('steel', (0.7, 0.71, 0.73), 0.22, 1.0)
mat_led = principled('led', (0.02, 0.05, 0.2), 0.3, 0.0)
nl, ll = nodes_of(mat_led); _ledp = nl['Principled BSDF']
_ledp.inputs['Emission Color'].default_value = (0.05, 0.3, 1.0, 1); _ledp.inputs['Emission Strength'].default_value = 0.0

# light oak counter
mat_wood = principled('wood', (0.6, 0.4, 0.22), 0.6, 0.0)
n, l = nodes_of(mat_wood)
tc = n.new('ShaderNodeTexCoord'); mp = n.new('ShaderNodeMapping')
mp.inputs['Scale'].default_value = (1.0, 9.0, 1.0)
wv = n.new('ShaderNodeTexWave'); wv.wave_type = 'BANDS'; wv.bands_direction = 'X'
wv.inputs['Scale'].default_value = 9; wv.inputs['Distortion'].default_value = 5; wv.inputs['Detail'].default_value = 3
nz = n.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 18
cr = n.new('ShaderNodeValToRGB'); cr.color_ramp.elements[0].color = (0.36, 0.22, 0.10, 1)
cr.color_ramp.elements[1].color = (0.58, 0.40, 0.22, 1)
l.new(tc.outputs['Object'], mp.inputs['Vector']); l.new(mp.outputs[0], wv.inputs['Vector'])
l.new(mp.outputs[0], nz.inputs['Vector'])
mx = n.new('ShaderNodeMath'); mx.operation = 'MULTIPLY_ADD'; mx.inputs[1].default_value = 0.35
l.new(wv.outputs['Fac'], mx.inputs[0]); l.new(nz.outputs['Fac'], mx.inputs[2])
l.new(mx.outputs[0], cr.inputs['Fac']); l.new(cr.outputs[0], n['Principled BSDF'].inputs['Base Color'])

# subway-tile back wall
mat_tile = principled('tile', (0.05, 0.09, 0.10), 0.12, 0.0)
n, l = nodes_of(mat_tile)
tc = n.new('ShaderNodeTexCoord'); br = n.new('ShaderNodeTexBrick')
br.inputs['Scale'].default_value = 3.3; br.inputs['Mortar Size'].default_value = 0.02
br.inputs['Color1'].default_value = (0.02, 0.045, 0.052, 1); br.inputs['Color2'].default_value = (0.028, 0.055, 0.062, 1)
br.inputs['Mortar'].default_value = (0.005, 0.005, 0.006, 1); br.inputs['Bias'].default_value = 0.0
l.new(tc.outputs['Object'], br.inputs['Vector']); l.new(br.outputs['Color'], n['Principled BSDF'].inputs['Base Color'])
l.new(br.outputs['Fac'], n['Principled BSDF'].inputs['Roughness'])
l.remove(l[-1]); mat_tile.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.3
mat_glow = bpy.data.materials.new('glow'); n, l = nodes_of(mat_glow)
em = n.new('ShaderNodeEmission'); em.inputs['Color'].default_value = (1.0, 0.7, 0.35, 1); em.inputs['Strength'].default_value = 40
l.new(em.outputs[0], n['Material Output'].inputs['Surface'])

# ----------------------------------------------------------------------------------------------------- kettle geometry
ZR = H_JUG                                  # rim top
# glass: outer wall 0.073, rounded rim, inner wall 0.070, open bottom at the plate level
glass_prof = [(R_IN, Z_PLATE), (R_IN + T_GLASS, Z_PLATE), (R_IN + T_GLASS, ZR - 0.0015), (R_IN + T_GLASS * 0.5, ZR + 0.0003),
              (R_IN, ZR - 0.0015)]
ob_glass = revolve('glass', glass_prof, mat_glass, rc=0.0013, deg=60, nseg=192)
# spout: solid glass wedge on the +x side
th = np.radians(np.linspace(-20, 20, 7)); pts = []
for t in th:
    for z in (ZR - 0.022, ZR + 0.0003):
        pts.append((0.0735 * math.cos(t), 0.0735 * math.sin(t), z))
pts += [(0.094, -0.005, ZR - 0.003), (0.094, 0.005, ZR - 0.003), (0.094, -0.005, ZR - 0.0015), (0.094, 0.005, ZR - 0.0015),
        (0.085, -0.012, ZR - 0.004), (0.085, 0.012, ZR - 0.004)]
ob_spout = hull('spout', pts, mat_glass)
# handle (dark plastic) on -x with two brackets
hp = [(-0.071, 0.178), (-0.098, 0.178), (-0.118, 0.150), (-0.121, 0.115), (-0.118, 0.075), (-0.098, 0.040), (-0.071, 0.040)]
ob_handle = tube('handle', hp, mat_body, rw=0.0085, rt=0.0055)
# lid: glass disc + steel rim ring + dark knob
lid_prof = [(0.0, ZR + 0.0003), (0.0735, ZR + 0.0003), (0.0745, ZR + 0.0015), (0.0745, ZR + 0.0040), (0.0, ZR + 0.0040)]
ob_lid = revolve('lid', lid_prof, mat_glass, rc=0.0008)
ob_ring = revolve('lid_ring', [(0.0105, ZR + 0.004), (0.0185, ZR + 0.004), (0.0185, ZR + 0.0062), (0.0105, ZR + 0.0062)], mat_steel, rc=0.0004)
knob_prof = [(0.0, ZR + 0.006), (0.0100, ZR + 0.006), (0.0075, ZR + 0.014), (0.0125, ZR + 0.0185), (0.0125, ZR + 0.0215), (0.0, ZR + 0.0215)]
ob_knob = revolve('knob', knob_prof, mat_knob, rc=0.0020)
# heating plate (brushed steel disc, 4 mm), body ring, power base, LED
ob_plate = revolve('plate', [(0.0, Z_PLATE), (0.0697, Z_PLATE), (0.0697, 0.0), (0.0, 0.0)], mat_plate, rc=0.0004)
ob_body = revolve('body', [(0.0, Z_BODY0), (R_BODY - 0.002, Z_BODY0), (R_BODY, Z_BODY0 + 0.004), (R_BODY, Z_PLATE - 0.0005),
                           (0.0, Z_PLATE - 0.0005)], mat_body, rc=0.0012)
ob_base = revolve('base', [(0.0, Z_BASE0), (R_BASE, Z_BASE0), (R_BASE, Z_BODY0 - 0.004), (R_BASE - 0.008, Z_BODY0),
                           (0.0, Z_BODY0)], mat_base, rc=0.003)
ob_led1 = revolve('led_top', [(0.0733, Z_PLATE - 0.0007), (0.0752, Z_PLATE - 0.0007), (0.0752, Z_PLATE - 0.0004), (0.0733, Z_PLATE - 0.0004)],
                  mat_led, deg=80)
ob_led2 = revolve('led_band', [(R_BODY - 0.0002, Z_PLATE - 0.0085), (R_BODY + 0.0003, Z_PLATE - 0.0085), (R_BODY + 0.0003, Z_PLATE - 0.0055),
                               (R_BODY - 0.0002, Z_PLATE - 0.0055)], mat_led, deg=80)
ob_led3 = revolve('led_under', [(0.0705, Z_PLATE - 0.0004), (0.0725, Z_PLATE - 0.0004), (0.0725, Z_PLATE - 0.0002), (0.0705, Z_PLATE - 0.0002)],
                  mat_led, deg=80)
for o in (ob_led1, ob_led2, ob_led3):
    o.visible_shadow = False

# counter and wall
bm = bmesh.new(); bmesh.ops.create_grid(bm, x_segments=2, y_segments=2, size=1.6)
oc = to_object('counter', bm, mat_wood); oc.location = (0, 0, Z_BASE0)
bm = bmesh.new(); bmesh.ops.create_grid(bm, x_segments=2, y_segments=2, size=1.8)
ow = to_object('wall', bm, mat_tile); ow.location = (0, 0.75, 0.9); ow.rotation_euler = (math.pi / 2, 0, 0)
rng = np.random.default_rng(3)
for i in range(40):                       # warm out-of-focus lights on the wall for bokeh
    bm = bmesh.new(); bmesh.ops.create_circle(bm, cap_ends=True, radius=0.012, segments=24)
    o = to_object(f'dot{i}', bm, mat_glow); o.location = (rng.uniform(-0.8, 0.8), 0.74, rng.uniform(0.05, 0.5))
    o.rotation_euler = (math.pi / 2, 0, 0); o.visible_shadow = False

# ----------------------------------------------------------------------------------------------------- water surface
def build_water():
    n = GRID_N
    u = np.linspace(-1, 1, n)
    U, Vv = np.meshgrid(u, u, indexing='ij')
    X = U * np.sqrt(np.clip(1 - Vv ** 2 / 2, 0, 1)); Y = Vv * np.sqrt(np.clip(1 - U ** 2 / 2, 0, 1))   # square -> disk
    gx, gy = (X * R_W).ravel(), (Y * R_W).ravel()
    idx = lambda i, j: i * n + j
    ring = [idx(i, 0) for i in range(n - 1)] + [idx(n - 1, j) for j in range(n - 1)] + \
           [idx(i, n - 1) for i in range(n - 1, 0, -1)] + [idx(0, j) for j in range(n - 1, 0, -1)]
    nb = len(ring)
    bx, by = gx[ring], gy[ring]
    verts = np.zeros((n * n + nb, 3)); verts[:n * n, 0], verts[:n * n, 1] = gx, gy
    verts[n * n:, 0], verts[n * n:, 1] = bx, by; verts[n * n:, 2] = 0.00005
    faces = [(idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)) for i in range(n - 1) for j in range(n - 1)]
    for k in range(nb):
        k2 = (k + 1) % nb
        faces.append((ring[k], n * n + k, n * n + k2, ring[k2]))
    faces.append(tuple(n * n + k for k in range(nb - 1, -1, -1)))
    me = bpy.data.meshes.new('water'); me.from_pydata(verts.tolist(), [], faces); me.update()
    bm = bmesh.new(); bm.from_mesh(me)
    smooth_by_angle(bm, 45); bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new('water', me); sc.collection.objects.link(ob); me.materials.append(mat_water)
    return ob, gx, gy


ob_water, GX, GY = build_water()
_wr = np.random.default_rng(11)
KW = 9
_th = _wr.uniform(0, 2 * math.pi, KW); _lam = _wr.uniform(0.006, 0.030, KW)
_k = 2 * math.pi / _lam; _kx, _ky = _k * np.cos(_th), _k * np.sin(_th)
_om = 0.3 * np.sqrt(9.81 * _k); _ph = _wr.uniform(0, 2 * math.pi, KW)
_a = np.sqrt(_lam / 0.03); _norm = 1.0 / (2.0 * math.sqrt(np.sum(_a ** 2) / 2))


def surface_height(h, A, phase, x, y):
    """z = h + A * sum a_k sin(kx x + ky y - w t + phi) * norm."""
    s = np.zeros_like(x)
    for kx, ky, om, ph, a in zip(_kx, _ky, _om, _ph, _a):
        s += a * np.sin(kx * x + ky * y - om * phase + ph)
    return h + A * s * _norm


def set_surface(h, A, phase):
    co = np.zeros((len(ob_water.data.vertices), 3), np.float32)
    ob_water.data.vertices.foreach_get('co', co.ravel())
    nt = GX.size
    co[:nt, 0], co[:nt, 1], co[:nt, 2] = GX, GY, surface_height(h, A, phase, GX, GY)
    ob_water.data.vertices.foreach_set('co', co.ravel()); ob_water.data.update()


# ----------------------------------------------------------------------------------------------------- bubbles (GN)
def build_bubbles():
    me = bpy.data.meshes.new('bubbles'); ob = bpy.data.objects.new('bubbles', me); sc.collection.objects.link(ob)
    mod = ob.modifiers.new('inst', 'NODES')
    ng = bpy.data.node_groups.new('bubbles_gn', 'GeometryNodeTree')
    ng.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    N, Lk = ng.nodes, ng.links
    gi = N.new('NodeGroupInput'); go = N.new('NodeGroupOutput')
    m2p = N.new('GeometryNodeMeshToPoints')
    sph = N.new('GeometryNodeMeshUVSphere'); sph.inputs['Segments'].default_value = 24; sph.inputs['Rings'].default_value = 12
    sph.inputs['Radius'].default_value = 1.0
    sm = N.new('GeometryNodeSetShadeSmooth'); Lk.new(sph.outputs['Mesh'], sm.inputs['Geometry'])
    inst = N.new('GeometryNodeInstanceOnPoints')
    na = N.new('GeometryNodeInputNamedAttribute'); na.data_type = 'FLOAT'; na.inputs['Name'].default_value = 'rad'
    cx = N.new('ShaderNodeCombineXYZ')
    for c in 'XYZ':
        Lk.new(na.outputs['Attribute'], cx.inputs[c])
    setm = N.new('GeometryNodeSetMaterial'); setm.inputs['Material'].default_value = mat_bubble
    Lk.new(gi.outputs[0], m2p.inputs['Mesh']); Lk.new(m2p.outputs['Points'], inst.inputs['Points'])
    Lk.new(sm.outputs['Geometry'], inst.inputs['Instance']); Lk.new(cx.outputs['Vector'], inst.inputs['Scale'])
    Lk.new(inst.outputs['Instances'], setm.inputs['Geometry']); Lk.new(setm.outputs['Geometry'], go.inputs[0])
    mod.node_group = ng
    me.attributes.new('rad', 'FLOAT', 'POINT')
    return ob


ob_bub = build_bubbles()


def set_bubbles(xyz, r):
    me = ob_bub.data; me.clear_geometry()
    n = len(r)
    me.vertices.add(n)
    if n:
        me.vertices.foreach_set('co', np.asarray(xyz, np.float32).ravel())
    if 'rad' not in me.attributes:
        me.attributes.new('rad', 'FLOAT', 'POINT')
    if n:
        me.attributes['rad'].data.foreach_set('value', np.maximum(np.asarray(r, np.float32), MIN_R_RENDER))
    me.update()


# ----------------------------------------------------------------------------------------------------- steam / mist
def volume_material(name, plume):
    """Principled Volume; density = dens_value * mask * noise(4D: object coords + phase). Returns (mat, dens_node, phase_node)."""
    m = bpy.data.materials.new(name); n, l = nodes_of(m)
    for x in list(n):
        n.remove(x)
    out = n.new('ShaderNodeOutputMaterial'); pv = n.new('ShaderNodeVolumePrincipled')
    pv.inputs['Color'].default_value = (0.55, 0.57, 0.6, 1); pv.inputs['Anisotropy'].default_value = 0.3
    l.new(pv.outputs[0], out.inputs['Volume'])
    tr = n.new('ShaderNodeBsdfTransparent'); l.new(tr.outputs[0], out.inputs['Surface'])   # boundary invisible
    dens = n.new('ShaderNodeValue'); dens.label = 'dens'
    ph = n.new('ShaderNodeValue'); ph.label = 'phase'
    tc = n.new('ShaderNodeTexCoord')
    drift = n.new('ShaderNodeMath'); drift.operation = 'MULTIPLY'; drift.inputs[1].default_value = -0.05
    l.new(ph.outputs[0], drift.inputs[0])
    cz = n.new('ShaderNodeCombineXYZ'); l.new(drift.outputs[0], cz.inputs['Z'])
    mp = n.new('ShaderNodeMapping'); l.new(tc.outputs['Object'], mp.inputs['Vector']); l.new(cz.outputs[0], mp.inputs['Location'])
    nz = n.new('ShaderNodeTexNoise'); nz.noise_dimensions = '4D'
    nz.inputs['Scale'].default_value = 28 if plume else 40; nz.inputs['Detail'].default_value = 3
    w = n.new('ShaderNodeMath'); w.operation = 'MULTIPLY'; w.inputs[1].default_value = 0.7
    l.new(ph.outputs[0], w.inputs[0]); l.new(w.outputs[0], nz.inputs['W']); l.new(mp.outputs[0], nz.inputs['Vector'])
    mr = n.new('ShaderNodeMapRange'); mr.inputs['From Min'].default_value = 0.35; mr.inputs['From Max'].default_value = 0.7
    l.new(nz.outputs['Fac'], mr.inputs['Value'])
    dn = n.new('ShaderNodeMath'); dn.operation = 'MULTIPLY'
    l.new(mr.outputs[0], dn.inputs[0]); l.new(dens.outputs[0], dn.inputs[1])
    last = dn
    if plume:       # soft blob: 1 - |g - c| from generated coordinates
        vs = n.new('ShaderNodeVectorMath'); vs.operation = 'SUBTRACT'; vs.inputs[1].default_value = (0.3, 0.5, 0.1)
        l.new(tc.outputs['Object'], vs.inputs[0])
        sc2 = n.new('ShaderNodeVectorMath'); sc2.operation = 'MULTIPLY'; sc2.inputs[1].default_value = (4.0, 4.0, 1.4)
        l.new(vs.outputs[0], sc2.inputs[0])
        ln = n.new('ShaderNodeVectorMath'); ln.operation = 'LENGTH'; l.new(sc2.outputs[0], ln.inputs[0])
        mk = n.new('ShaderNodeMapRange'); mk.inputs['From Min'].default_value = 0.0; mk.inputs['From Max'].default_value = 1.0
        mk.inputs['To Min'].default_value = 1.0; mk.inputs['To Max'].default_value = 0.0; mk.interpolation_type = 'SMOOTHSTEP'
        l.new(ln.outputs['Value'], mk.inputs['Value'])
        mm = n.new('ShaderNodeMath'); mm.operation = 'MULTIPLY'
        l.new(dn.outputs[0], mm.inputs[0]); l.new(mk.outputs[0], mm.inputs[1]); last = mm
        sz = n.new('ShaderNodeSeparateXYZ'); l.new(tc.outputs['Object'], sz.inputs[0])
        fz = n.new('ShaderNodeMapRange'); fz.inputs['From Max'].default_value = 0.5; l.new(sz.outputs['Z'], fz.inputs['Value'])
        m2 = n.new('ShaderNodeMath'); m2.operation = 'MULTIPLY'; l.new(mm.outputs[0], m2.inputs[0]); l.new(fz.outputs[0], m2.inputs[1]); last = m2
        sz = n.new('ShaderNodeSeparateXYZ'); l.new(tc.outputs['Object'], sz.inputs[0])
        fz = n.new('ShaderNodeMapRange'); fz.inputs['From Max'].default_value = 0.5; l.new(sz.outputs['Z'], fz.inputs['Value'])
        m2 = n.new('ShaderNodeMath'); m2.operation = 'MULTIPLY'; l.new(mm.outputs[0], m2.inputs[0]); l.new(fz.outputs[0], m2.inputs[1]); last = m2
    l.new(last.outputs[0], pv.inputs['Density'])
    return m, dens, ph


mat_plume, plume_dens, plume_ph = volume_material('plume', True)
mat_mist, mist_dens, mist_ph = volume_material('mist', False)
bm = bmesh.new(); bmesh.ops.create_cube(bm, size=1.0); bmesh.ops.translate(bm, verts=bm.verts, vec=(0.5, 0.5, 0.5))
ob_plume = to_object('plume', bm, mat_plume); ob_plume.location = (0.04, -0.07, 0.185); ob_plume.scale = (0.17, 0.14, 0.16)
ob_plume.visible_shadow = False
bm = bmesh.new(); bmesh.ops.create_cone(bm, cap_ends=True, segments=64, radius1=0.0693, radius2=0.0693, depth=1.0)
bmesh.ops.translate(bm, verts=bm.verts, vec=(0, 0, 0.5))
ob_mist = to_object('mist', bm, mat_mist); ob_mist.visible_shadow = False

# ----------------------------------------------------------------------------------------------------- lights, world, camera
world = bpy.data.worlds.new('w'); sc.world = world; world.use_nodes = True
wn_, wl_ = world.node_tree.nodes, world.node_tree.links
bg = wn_['Background']; bg.inputs['Strength'].default_value = 0.04
tc = wn_.new('ShaderNodeTexCoord'); sp = wn_.new('ShaderNodeSeparateXYZ'); wl_.new(tc.outputs['Window'], sp.inputs[0])
tc.outputs  # window coords: gradient in screen-y for camera rays; reflections use generated direction below
vec = wn_.new('ShaderNodeTexCoord'); sx = wn_.new('ShaderNodeSeparateXYZ'); wl_.new(vec.outputs['Generated'], sx.inputs[0])
gr = wn_.new('ShaderNodeValToRGB')
gr.color_ramp.elements[0].position = 0.35; gr.color_ramp.elements[0].color = (0.04, 0.03, 0.025, 1)
gr.color_ramp.elements[1].position = 0.8; gr.color_ramp.elements[1].color = (0.5, 0.42, 0.36, 1)
wl_.new(sx.outputs['Z'], gr.inputs['Fac']); wl_.new(gr.outputs[0], bg.inputs['Color'])


def area(name, loc, target, size, energy, color, size_y=None, shape='RECTANGLE', spread=180):
    ld = bpy.data.lights.new(name, 'AREA'); ld.spread = math.radians(spread); ld.energy = energy; ld.color = color; ld.shape = shape; ld.size = size
    if size_y:
        ld.size_y = size_y
    o = bpy.data.objects.new(name, ld); sc.collection.objects.link(o); o.location = loc
    c = o.constraints.new('TRACK_TO'); c.track_axis = 'TRACK_NEGATIVE_Z'; c.up_axis = 'UP_Y'
    tgt = bpy.data.objects.new(name + '_t', None); tgt.location = target; sc.collection.objects.link(tgt); c.target = tgt
    o.visible_camera = False
    return o


area('key', (-0.75, -0.9, 0.55), (0, 0, 0.09), 0.7, 12, (1.0, 0.80, 0.60), 0.5)
area('rimL', (-0.25, 0.42, 0.18), (0, 0, 0.1), 0.07, 160, (0.85, 0.92, 1.0), 0.32, spread=35)
area('rimR', (0.27, 0.40, 0.18), (0, 0, 0.1), 0.07, 160, (1.0, 0.9, 0.78), 0.32, spread=35)
area('fill', (0.9, -0.7, 0.25), (0, 0, 0.09), 0.5, 3, (0.8, 0.88, 1.0), 0.5)
area('top', (0.0, -0.1, 0.8), (0, 0, 0.0), 0.35, 0.5, (1.0, 0.95, 0.9), 0.35, spread=60)
# rim lights only illuminate the kettle (light linking), so they do not wash out the counter
rim_col = bpy.data.collections.new('rim_receivers'); sc.collection.children.link(rim_col)
for o in (ob_glass, ob_spout, ob_handle, ob_lid, ob_ring, ob_knob, ob_water, ob_bub, ob_plate, ob_body, ob_plume, ob_mist):
    rim_col.objects.link(o)
for o in sc.objects:
    if o.name in ('rimL', 'rimR'):
        o.light_linking.receiver_collection = rim_col
blue = bpy.data.lights.new('ledlight', 'POINT'); blue.energy = 0.0; blue.color = (0.1, 0.35, 1.0); blue.shadow_soft_size = 0.02
ob_ledlight = bpy.data.objects.new('ledlight', blue); sc.collection.objects.link(ob_ledlight); ob_ledlight.location = (0, -0.083, -0.012)

cam_d = bpy.data.cameras.new('cam'); cam_d.lens = 70; cam_d.sensor_width = 36
ob_cam = bpy.data.objects.new('cam', cam_d); sc.collection.objects.link(ob_cam); sc.camera = ob_cam
ob_cam.location = (-0.20, -1.28, 0.34)
tgt = bpy.data.objects.new('camtgt', None); tgt.location = (0.0, 0, 0.082); sc.collection.objects.link(tgt)
c = ob_cam.constraints.new('TRACK_TO'); c.target = tgt; c.track_axis = 'TRACK_NEGATIVE_Z'; c.up_axis = 'UP_Y'
cam_d.dof.use_dof = True; cam_d.dof.focus_object = tgt; cam_d.dof.aperture_fstop = 4.0

# ----------------------------------------------------------------------------------------------------- render settings
cy = sc.cycles
sc.render.engine = 'CYCLES'; cy.device = 'CPU'
sc.render.threads_mode = 'FIXED'; sc.render.threads = 4
sc.render.image_settings.file_format = 'PNG'; sc.render.image_settings.color_depth = '8'
sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'None'; sc.view_settings.exposure = 0.0
cy.use_denoising = True; cy.denoiser = 'OPENIMAGEDENOISE'
cy.use_adaptive_sampling = True; cy.adaptive_threshold = 0.05; cy.adaptive_min_samples = 8
cy.max_bounces = 16; cy.transmission_bounces = 12; cy.diffuse_bounces = 3; cy.glossy_bounces = 4
cy.volume_bounces = 2; cy.transparent_max_bounces = 16
cy.caustics_reflective = False; cy.caustics_refractive = False
cy.sample_clamp_indirect = 6.0; cy.sample_clamp_direct = 0.0; cy.blur_glossy = 0.3
cy.volume_step_rate = 1.0; cy.volume_max_steps = 32
sc.render.use_persistent_data = True


def set_params(h, A, phase, steam, led):
    set_surface(h, A, phase)
    _ledp.inputs['Emission Strength'].default_value = 25.0 * led
    blue.energy = 0.6 * led
    on = steam > 0.02
    ob_plume.hide_render = not on; ob_mist.hide_render = not on
    for d, p in ((plume_dens, plume_ph), (mist_dens, mist_ph)):
        p.outputs[0].default_value = phase
    plume_dens.outputs[0].default_value = 2.5 * steam
    mist_dens.outputs[0].default_value = 1.0 * steam
    zt = h + A + 0.002
    ob_mist.location = (0, 0, zt); ob_mist.scale = (1, 1, max(ZR - 0.001 - zt, 0.001))


def render_to(path):
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


def configure_res(RX, RY, SPP):
    sc.render.resolution_x, sc.render.resolution_y = RX, RY; sc.render.resolution_percentage = 100
    cy.samples = SPP


# ----------------------------------------------------------------------------------------------------- main
def synthetic():
    r = np.random.default_rng(5)
    xyz, rad = [], []
    for _ in range(120):                                # tiny bubbles on the heating plate
        a, d = r.uniform(0, 2 * math.pi), 0.064 * math.sqrt(r.uniform())
        rr = r.uniform(1e-4, 3e-4); xyz.append((d * math.cos(a), d * math.sin(a), rr)); rad.append(rr)
    for _ in range(180):                                # on the lower glass wall (front half, towards the camera)
        a = r.uniform(-math.pi * 0.95, -math.pi * 0.05); rr = r.uniform(1e-4, 3e-4); z = r.uniform(0.002, 0.08)
        d = R_W - rr; xyz.append((d * math.cos(a), d * math.sin(a), z)); rad.append(rr)
    for _ in range(40):                                 # rising mid-size bubbles
        a, d = r.uniform(0, 2 * math.pi), 0.062 * math.sqrt(r.uniform()); rr = r.uniform(1e-3, 3e-3)
        xyz.append((d * math.cos(a), d * math.sin(a), r.uniform(0.01, 0.10))); rad.append(rr)
    return dict(xyz=np.array(xyz), r=np.array(rad), h=0.110, A=0.002, phase=1.3, steam=0.6, led=1.0)


def apply_frame(d):
    set_bubbles(np.asarray(d['xyz'], float).reshape(-1, 3), np.asarray(d['r'], float).ravel())
    set_params(float(d['h']), float(d['A']), float(d['phase']), float(d['steam']), float(d['led']))


if __name__ == '__main__':
    a = sys.argv[1:]
    if a and a[0] == '--test':
        out = a[1]; RX, RY, SPP = (int(a[i]) if len(a) > i else d for i, d in ((2, 1280), (3, 720), (4, 32)))
        configure_res(RX, RY, SPP); apply_frame(synthetic())
        t0 = time.time(); render_to(os.path.abspath(out)); print(f'test render {time.time() - t0:.1f} s')
    else:
        FR, OUT = a[0], a[1]
        RX, RY, SPP = (int(a[i]) if len(a) > i else d for i, d in ((2, 1280), (3, 720), (4, 32)))
        first = int(a[5]) if len(a) > 5 else 0
        last = int(a[6]) if len(a) > 6 else -1
        stride = int(a[7]) if len(a) > 7 else 1
        os.makedirs(OUT, exist_ok=True); configure_res(RX, RY, SPP)
        files = sorted(glob.glob(os.path.join(FR, 'f*.npz')))
        sel = files[first:None if last < 0 else last][::stride]
        for f in sel:
            k = int(os.path.basename(f)[1:-4]); z = np.load(f)
            apply_frame({key: z[key] for key in ('xyz', 'r', 'h', 'A', 'phase', 'steam', 'led')})
            t0 = time.time(); render_to(os.path.abspath(f'{OUT}/f{k:05d}.png'))
            print(f'frame {k} N={len(z["r"])} {time.time() - t0:.1f} s', flush=True)
