"""Blender (bpy 5.x, Cycles) scene: Christmas candle lit by the simulated flame.

Scene units: 1 BU = 1 cm. Wick base at the origin, wax top at z = 0.
The flame is a volume whose emission is looked up from the simulated (r, z) emission maps:
  soot blackbody light (Planck spectrum at the simulated T)  +  CH*/C2* chemiluminescence (simulated reaction rate).
Usage (either the bpy Python module or the Blender app):
  python3 blender_scene.py MODE RES_X RES_Y SAMPLES OUT [extra]
  blender -b -P blender_scene.py -- MODE RES_X RES_Y SAMPLES OUT [extra]
  MODE: beauty | soot | chem   -> OUT is a .png; optional extra "x0,y0,x1,y1" renders only that pixel region
        anim                   -> OUT is a folder; extra = number of frames (seamless flicker loop)
        q3d                    -> OUT is a folder; extra = atlas dir from quasi3d/export_q3d.py [first [step [last]]]:
                                  the simulated quasi-3D flame (real lean/flicker), one frame per snapshot
  Set CANDLE_GPU=1 to render on an NVIDIA GPU (OptiX, falling back to CUDA).
"""
import sys, math, numpy as np, bpy, os

if '--' in sys.argv:                       # running inside the Blender app: keep only our arguments
    sys.argv = [sys.argv[0]] + sys.argv[sys.argv.index('--') + 1:]
MODE, RX, RY, SPP, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
FD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fields')
K_SOOT, K_CHEM = 3.2, 3.2 * 0.148           # exposure; blue/soot ratio from line-of-sight integrals (matches render.py)
if MODE == 'soot': K_CHEM = 0.0
if MODE == 'chem': K_SOOT, K_CHEM = 0.0, K_CHEM

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = 'CYCLES'
sc.cycles.device = 'CPU'
if os.environ.get('CANDLE_GPU'):
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        for backend in ('OPTIX', 'CUDA'):
            prefs.compute_device_type = backend; prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == backend]
            if gpus:
                for d in prefs.devices: d.use = d.type == backend
                sc.cycles.device = 'GPU'; print('rendering on', backend, [d.name for d in gpus]); break
    except Exception as e:
        print('GPU setup failed, using CPU:', e)
sc.cycles.samples = SPP
sc.cycles.use_denoising = True
try: sc.cycles.denoiser = 'OPENIMAGEDENOISE'
except Exception: pass
for attr, val in [('volume_step_rate', 0.1), ('volume_preview_step_rate', 0.1), ('volume_max_steps', 4096)]:
    if hasattr(sc.cycles, attr): setattr(sc.cycles, attr, val)
sc.cycles.max_bounces = 6
if hasattr(sc.cycles, 'volume_biased'): sc.cycles.volume_biased = True
sc.unit_settings.system = 'METRIC'; sc.unit_settings.scale_length = 0.01
sc.render.resolution_x, sc.render.resolution_y = RX, RY
sc.render.film_transparent = False
sc.view_settings.view_transform = 'AgX' if 'AgX' in [v.identifier for v in sc.view_settings.bl_rna.properties['view_transform'].enum_items] else 'Filmic'
sc.view_settings.look = 'None'
sc.view_settings.exposure = 0.0
sc.render.threads_mode = 'AUTO'

world = bpy.data.worlds.new('w'); sc.world = world; world.use_nodes = True
world.node_tree.nodes['Background'].inputs[0].default_value = (0.0015, 0.002, 0.005, 1)
world.node_tree.nodes['Background'].inputs[1].default_value = 1.0

def mat(name, base, rough=0.4, sss=0.0, sss_rad=(1, .2, .1), sss_scale=0.1, coat=0.0, emit=None, estr=0.0, spec=0.5):
    m = bpy.data.materials.new(name); m.use_nodes = True
    p = m.node_tree.nodes['Principled BSDF']
    p.inputs['Base Color'].default_value = (*base, 1)
    p.inputs['Roughness'].default_value = rough
    p.inputs['Subsurface Weight'].default_value = sss
    p.inputs['Subsurface Radius'].default_value = sss_rad
    p.inputs['Subsurface Scale'].default_value = sss_scale
    p.inputs['Coat Weight'].default_value = coat
    if emit:
        p.inputs['Emission Color'].default_value = (*emit, 1)
        p.inputs['Emission Strength'].default_value = estr
    return m

wax = mat('wax', (0.50, 0.018, 0.03), rough=0.38, sss=0.6, sss_rad=(1.0, 0.25, 0.15), sss_scale=0.35, coat=0.15)
molten = mat('molten', (0.55, 0.03, 0.035), rough=0.04, sss=0.6, sss_rad=(1.0, 0.3, 0.2), sss_scale=0.4, coat=0.6)
wickm = mat('wick', (0.003, 0.0025, 0.002), rough=1.0)
wickm.node_tree.nodes['Principled BSDF'].inputs['Specular IOR Level'].default_value = 0.0
ember = mat('ember', (0.1, 0.02, 0.0), emit=(1.0, 0.28, 0.05), estr=12.0)
leafm = mat('leaf', (0.015, 0.10, 0.035), rough=0.22, coat=0.8)
berrym = mat('berry', (0.55, 0.01, 0.02), rough=0.15, coat=1.0, sss=0.3, sss_rad=(1, .2, .2), sss_scale=0.05)
tablem = mat('table', (0.045, 0.022, 0.012), rough=0.55)

def add(obj, m):
    obj.data.materials.append(m); return obj
def smooth(obj):
    for poly in obj.data.polygons: poly.use_smooth = True

# candle body
bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=1.0, depth=6.0, location=(0, 0, -3.0))
body = add(bpy.context.object, wax); smooth(body)
bv = body.modifiers.new('bev', 'BEVEL'); bv.width = 0.06; bv.segments = 4; bv.limit_method = 'ANGLE'
# melted rim and pool
bpy.ops.mesh.primitive_torus_add(major_radius=0.94, minor_radius=0.075, major_segments=128, minor_segments=24, location=(0, 0, 0.0))
smooth(add(bpy.context.object, wax))
bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=0.9, depth=0.02, location=(0, 0, 0.012))
add(bpy.context.object, molten)
# drips
for ang, L, w in [(-115, 1.4, 0.13), (-80, 0.7, 0.11), (-48, 2.0, 0.14), (-20, 0.9, 0.1), (25, 1.6, 0.12)]:
    a = math.radians(ang); R = 1.0 + 0.04
    x, y = R * math.cos(a), R * math.sin(a)
    bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=w * 0.75, radius2=w, depth=L, location=(x, y, -L / 2 + 0.02))
    smooth(add(bpy.context.object, wax))
    bpy.ops.mesh.primitive_uv_sphere_add(radius=w * 0.95, location=(x, y, -L + 0.02), segments=24, ring_count=12)
    o = bpy.context.object; o.scale = (1, 1, 1.25); smooth(add(o, wax))
# wick (slightly bent) + ember tip
bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.07, depth=0.62, location=(0.02, 0, 0.3), rotation=(0, math.radians(6), 0))
add(bpy.context.object, wickm)
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.07, location=(0.05, 0, 0.6), segments=16, ring_count=8)
add(bpy.context.object, ember)

# flame volume: emission looked up from the simulated (r, z) maps
def load_img(name, arr):
    nz, nr, _ = arr.shape
    img = bpy.data.images.new(name, width=nr, height=nz, alpha=True, float_buffer=True)
    rgba = np.concatenate([arr, np.ones((nz, nr, 1), np.float32)], axis=2)
    try: img.colorspace_settings.name = 'Linear Rec.709'
    except Exception: pass
    img.pixels.foreach_set(rgba.ravel())
    img.update()
    return img
Q3D = MODE == 'q3d'
if Q3D:
    import json
    QD = sys.argv[6]
    QM = json.load(open(os.path.join(QD, 'meta.json')))
    soot_img = load_img('soot', np.load(os.path.join(QD, 'soot_0000.npy')))
    chem_img = load_img('chem', np.load(os.path.join(QD, 'chem_0000.npy')))
else:
    soot_img = load_img('soot', np.load(f'{FD}/soot_rgb.npy'))
    chem_img = load_img('chem', np.load(f'{FD}/chem_rgb.npy'))
RMAX, ZMAX = 1.5, 7.0
bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=RMAX, depth=ZMAX, location=(0, 0, ZMAX / 2))
dom = bpy.context.object; dom.name = 'flame'
fm = bpy.data.materials.new('flame'); fm.use_nodes = True
nt = fm.node_tree; N = nt.nodes; Lk = nt.links
for n in list(N): N.remove(n)
tc = N.new('ShaderNodeTexCoord'); sep = N.new('ShaderNodeSeparateXYZ'); Lk.new(tc.outputs['Object'], sep.inputs[0])
def val(name, v=0.0):
    n = N.new('ShaderNodeValue'); n.name = name; n.outputs[0].default_value = v; return n
def op(o, a, b):
    n = N.new('ShaderNodeMath'); n.operation = o
    for i, x in enumerate((a, b)):
        if isinstance(x, (int, float)): n.inputs[i].default_value = x
        else: Lk.new(x, n.inputs[i])
    return n.outputs[0]
# flicker deformation (all zero = the simulated steady flame): base anchored at the wax, sway grows with height^2
SX, SY, ST = val('SX'), val('SY'), val('ST')
hgt = op('ADD', sep.outputs[2], ZMAX / 2)                     # cm above the wax surface
h2 = op('MULTIPLY', hgt, hgt)
xs = op('SUBTRACT', sep.outputs[0], op('MULTIPLY', h2, SX.outputs[0]))
ys = op('SUBTRACT', sep.outputs[1], op('MULTIPLY', h2, SY.outputs[0]))
hs = op('DIVIDE', hgt, op('ADD', ST.outputs[0], 1.0))        # ST > 0 stretches the flame upwards
cxy = N.new('ShaderNodeCombineXYZ'); Lk.new(xs, cxy.inputs[0]); Lk.new(ys, cxy.inputs[1])
ln = N.new('ShaderNodeVectorMath'); ln.operation = 'LENGTH'; Lk.new(cxy.outputs[0], ln.inputs[0])
u = N.new('ShaderNodeMath'); u.operation = 'DIVIDE'; Lk.new(ln.outputs['Value'], u.inputs[0]); u.inputs[1].default_value = RMAX
v = N.new('ShaderNodeMath'); v.operation = 'DIVIDE'; Lk.new(hs, v.inputs[0]); v.inputs[1].default_value = ZMAX
uv = N.new('ShaderNodeCombineXYZ'); Lk.new(u.outputs[0], uv.inputs[0]); Lk.new(v.outputs[0], uv.inputs[1])
def tex(img):
    t = N.new('ShaderNodeTexImage'); t.image = img; t.interpolation = 'Linear'; t.extension = 'CLIP'
    Lk.new(uv.outputs[0], t.inputs['Vector']); return t
def vscale(vec_out, s):
    n = N.new('ShaderNodeVectorMath'); n.operation = 'SCALE'; Lk.new(vec_out, n.inputs[0])
    if isinstance(s, (int, float)): n.inputs['Scale'].default_value = s
    else: Lk.new(s, n.inputs['Scale'])
    return n.outputs[0]
def vadd(a, b):
    n = N.new('ShaderNodeVectorMath'); n.operation = 'ADD'; Lk.new(a, n.inputs[0]); Lk.new(b, n.inputs[1]); return n.outputs[0]
if not Q3D:
    ts, tch = tex(soot_img), tex(chem_img)
    soot_col, chem_col = ts.outputs['Color'], tch.outputs['Color']
else:
    # atlas lookup: tile k holds the (z, r) emission at theta_k = 2 pi k / Nt; linear interpolation in theta
    Nt, nr = QM['Nt'], QM['nr']
    theta = op('ADD', op('ARCTAN2', sep.outputs[1], sep.outputs[0]), 2 * math.pi)
    kf = op('DIVIDE', op('MODULO', theta, 2 * math.pi), 2 * math.pi / Nt)
    k0 = op('FLOOR', kf, 0.0); wth = op('SUBTRACT', kf, k0)
    k1 = op('MODULO', op('ADD', k0, 1.0), float(Nt))
    rn = op('MINIMUM', op('MAXIMUM', u.outputs[0], 0.5 / nr), 1 - 0.5 / nr)
    inside = op('LESS_THAN', u.outputs[0], 1.0)
    def tex_at(img, k):
        uu = op('DIVIDE', op('ADD', k, rn), float(Nt))
        c = N.new('ShaderNodeCombineXYZ'); Lk.new(uu, c.inputs[0]); Lk.new(v.outputs[0], c.inputs[1])
        t = N.new('ShaderNodeTexImage'); t.image = img; t.interpolation = 'Linear'; t.extension = 'CLIP'
        Lk.new(c.outputs[0], t.inputs['Vector']); return t.outputs['Color']
    w0 = op('MULTIPLY', op('SUBTRACT', 1.0, wth), inside); w1 = op('MULTIPLY', wth, inside)
    soot_col = vadd(vscale(tex_at(soot_img, k0), w0), vscale(tex_at(soot_img, k1), w1))
    chem_col = vadd(vscale(tex_at(chem_img, k0), w0), vscale(tex_at(chem_img, k1), w1))
ms = N.new('ShaderNodeVectorMath'); ms.operation = 'SCALE'; Lk.new(soot_col, ms.inputs[0]); ms.inputs['Scale'].default_value = K_SOOT
mc = N.new('ShaderNodeVectorMath'); mc.operation = 'SCALE'; Lk.new(chem_col, mc.inputs[0]); mc.inputs['Scale'].default_value = K_CHEM
addn = N.new('ShaderNodeVectorMath'); addn.operation = 'ADD'; Lk.new(ms.outputs[0], addn.inputs[0]); Lk.new(mc.outputs[0], addn.inputs[1])
em = N.new('ShaderNodeEmission'); Lk.new(addn.outputs[0], em.inputs['Color']); em.inputs['Strength'].default_value = 1.0
out = N.new('ShaderNodeOutputMaterial'); Lk.new(em.outputs[0], out.inputs['Volume'])
dom.data.materials.append(fm)
for attr, val in [('volume_step_rate', 0.1), ('homogeneous_volume', False)]:
    if hasattr(fm.cycles, attr): setattr(fm.cycles, attr, val)
dom.visible_shadow = False

# the flame's light on the scene (point light at the luminous zone, colour of ~1900 K soot)
lw = {'beauty': 1.0, 'anim': 1.0, 'q3d': 1.0, 'soot': 1.0, 'chem': 0.04}[MODE]
bpy.ops.object.light_add(type='POINT', location=(0, 0, 3.0))
LT = bpy.context.object; lt = LT.data; lt.energy = 1500.0 * lw; lt.shadow_soft_size = 0.4; lt.color = (1.0, 0.58, 0.26)
dom_light_hidden = True
bpy.context.object.visible_camera = False

bpy.ops.object.light_add(type='AREA', location=(-14, -30, 6)); fl = bpy.context.object; fl.data.energy = 350.0; fl.data.size = 12; fl.data.color = (0.55, 0.62, 1.0)
fl.rotation_euler = (math.radians(75), 0, math.radians(-25))
# table + backdrop
bpy.ops.mesh.primitive_plane_add(size=400, location=(0, 110, 0), rotation=(math.radians(90), 0, 0)); add(bpy.context.object, mat('back', (0.004, 0.005, 0.01), rough=0.9))
bpy.ops.mesh.primitive_plane_add(size=220, location=(0, 0, -6.0)); add(bpy.context.object, tablem)

# holly at the base, in front of the candle
def leaf(cx, cy, cz, yaw, L=2.1, W=0.85, tilt=0.3):
    t = np.linspace(0, 1, 11)
    up = [(L * s, (W / 2 + (0.16 if i % 2 else -0.03)) * math.sin(math.pi * s)) for i, s in enumerate(t)]
    dn = [(L * s, -(W / 2 + (0.16 if i % 2 else -0.03)) * math.sin(math.pi * s)) for i, s in reversed(list(enumerate(t)))]
    pts = up + dn[1:-1]
    me = bpy.data.meshes.new('leaf'); verts = [(x, y, 0.12 * math.sin(math.pi * x / L) - 0.05 * abs(y)) for x, y in pts]
    me.from_pydata(verts + [(L / 2, 0, 0.14)], [], [(i, (i + 1) % len(pts), len(pts)) for i in range(len(pts))])
    ob = bpy.data.objects.new('leaf', me); bpy.context.collection.objects.link(ob)
    ob.location = (cx, cy, cz); ob.rotation_euler = (tilt, 0, yaw)
    so = ob.modifiers.new('s', 'SOLIDIFY'); so.thickness = 0.03
    ob.data.materials.append(leafm); smooth(ob)
leaf(-0.35, -2.0, -5.93, math.radians(195), L=3.4, W=1.35); leaf(0.35, -2.05, -5.93, math.radians(-15), L=3.4, W=1.35)
leaf(0.0, -1.9, -5.88, math.radians(250), L=2.9, W=1.2, tilt=0.4)
for bx, by, bz in [(-0.32, -2.4, -5.68), (0.3, -2.45, -5.7), (0.0, -2.85, -5.7)]:
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.3, location=(bx, by, bz), segments=32, ring_count=16)
    smooth(add(bpy.context.object, berrym))

# out-of-focus Christmas lights in the background
rng = np.random.default_rng(7)
cols = [(1.0, 0.62, 0.22), (1.0, 0.45, 0.12), (1.0, 0.80, 0.45), (0.9, 0.12, 0.10), (0.15, 0.55, 0.25)]
for i in range(34):
    x = rng.uniform(-24, 24); z = rng.uniform(-4.5, 7); y = rng.uniform(40, 80)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.18, location=(x, y, z), segments=12, ring_count=6)
    c = cols[i % len(cols)]
    o = bpy.context.object; o.data.materials.append(mat(f'b{i}', (0, 0, 0), emit=c, estr=rng.uniform(4, 10)))
    o.visible_shadow = False; o.visible_diffuse = False; o.visible_glossy = True

# camera
bpy.ops.object.camera_add(location=(0, -35.0, 7.5))
cam = bpy.context.object; sc.camera = cam
cam.data.lens = 85; cam.data.sensor_fit = 'VERTICAL'; cam.data.sensor_height = 36
tgt = np.array([0, 0, -0.2]); d = tgt - np.array(cam.location)
cam.rotation_euler = (math.atan2(math.hypot(d[0], d[1]), -d[2]), 0, math.atan2(d[1], d[0]) - math.pi / 2)
cam.data.dof.use_dof = True; cam.data.dof.focus_distance = float(np.linalg.norm(d)); cam.data.dof.aperture_fstop = 5.6 / 100   # scene is in cm
cam.data.clip_start = 0.5; cam.data.clip_end = 500

if MODE not in ('anim', 'q3d') and len(sys.argv) > 6:          # border render: x0,y0,x1,y1 in pixels from the top-left
    x0, y0, x1, y1 = [float(t) for t in sys.argv[6].split(',')]
    sc.render.use_border = True; sc.render.use_crop_to_border = True
    sc.render.border_min_x, sc.render.border_max_x = x0 / RX, x1 / RX
    sc.render.border_min_y, sc.render.border_max_y = 1 - y1 / RY, 1 - y0 / RY
sc.render.filepath = OUT
sc.render.image_settings.file_format = 'PNG'
if Q3D:                         # simulated quasi-3D flame: one frame per snapshot, textures and light from the data
    os.makedirs(OUT, exist_ok=True)
    first = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    step = int(sys.argv[8]) if len(sys.argv) > 8 else 1
    last = int(sys.argv[9]) if len(sys.argv) > 9 else len(QM['frames'])
    E0 = lt.energy; Z0 = 3.0
    c0 = QM['frames'][0]['centroid']
    for fr in QM['frames'][first:last:step]:
        k = fr['k']
        for img, name in ((soot_img, 'soot'), (chem_img, 'chem')):
            a = np.load(os.path.join(QD, f'{name}_{k:04d}.npy'))
            rgba = np.concatenate([a, np.ones(a.shape[:2] + (1,), np.float32)], axis=2)
            img.pixels.foreach_set(rgba.ravel()); img.update()
        cx, cy, cz = fr['centroid']                        # m -> cm, light follows the soot glow
        LT.location = (cx * 100, cy * 100, Z0 + (cz - c0[2]) * 100)
        lt.energy = E0 * fr['lum_rel']
        sc.render.filepath = f'{OUT}/f{k:04d}.png'
        bpy.ops.render.render(write_still=True)
        print('frame', k, 't', fr['t'], flush=True)
elif MODE != 'anim':
    bpy.ops.render.render(write_still=True)
    print('rendered', OUT)
else:                          # seamless loop; frame 0 is the undisturbed simulated flame
    NF = int(sys.argv[6]); os.makedirs(OUT, exist_ok=True)
    E0 = lt.energy; first = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    for k in range(first, NF):
        t = 2 * math.pi * k / NF
        sx = 0.012 * math.sin(t) + 0.004 * math.sin(3 * t)
        sy = 0.006 * math.sin(2 * t)
        st = 0.05 * math.sin(2 * t) + 0.025 * math.sin(5 * t)
        N['SX'].outputs[0].default_value = sx; N['SY'].outputs[0].default_value = sy; N['ST'].outputs[0].default_value = st
        LT.location = (sx * 9.0, sy * 9.0, 3.0 * (1 + st))
        lt.energy = E0 * (1 + 1.5 * st)
        sc.render.filepath = f'{OUT}/f{k:03d}.png'
        bpy.ops.render.render(write_still=True)
        print('frame', k, flush=True)
