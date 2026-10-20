"""Blender close-up of the bubble wall (water surface + gas at 1 atm) from build_interface.py runs.

The view looks along the water surface (slightly down) from inside the gas. The box is tiled 3 x 3 laterally
(periodic images) so the surface looks extended. Only the top ~1.5 nm of the liquid is drawn (the rest is hidden).
Colours: water O red / H white (vapour molecules in the gas glow slightly), N2 blue, O2 green, Ar violet.
usage: [TRANSPARENT=1] python3 render_interface.py RUNDIR OUTDIR [RX=960] [RY=540] [SPP=16] [first=0] [last=-1] [stride=1]
"""
import sys, os, json, math
import numpy as np
import bpy
import MDAnalysis as mda

RUN, OUT = sys.argv[1], sys.argv[2]
RX, RY, SPP = (int(sys.argv[i]) if len(sys.argv) > i else d for i, d in ((3, 960), (4, 540), (5, 16)))
FIRST = int(sys.argv[6]) if len(sys.argv) > 6 else 0
LAST = int(sys.argv[7]) if len(sys.argv) > 7 else -1
STRIDE = int(sys.argv[8]) if len(sys.argv) > 8 else 1
os.makedirs(OUT, exist_ok=True)
meta = json.load(open(f'{RUN}/meta.json'))
LX, LZ = meta['LX'], meta['LZ']
u = mda.Universe(f'{RUN}/conf.gro', f'{RUN}/prod.xtc')
OW = u.select_atoms('name OW'); H1 = u.select_atoms('name HW1'); H2 = u.select_atoms('name HW2')
NN = u.select_atoms('resname N2'); OO = u.select_atoms('resname O2'); AR = u.select_atoms('resname AR')
TILE = (-1, 0, 1)

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = SPP; sc.cycles.use_denoising = True
sc.render.resolution_x, sc.render.resolution_y = RX, RY
sc.view_settings.view_transform = 'AgX'
world = bpy.data.worlds.new('w'); sc.world = world; world.use_nodes = True
world.node_tree.nodes['Background'].inputs[0].default_value = (0.010, 0.012, 0.018, 1)
if os.environ.get('TRANSPARENT'):                 # for compositing over another picture (portrait cut)
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = 'PNG'; sc.render.image_settings.color_mode = 'RGBA'

def material(name, rgb, rough=0.35, emit=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    p = m.node_tree.nodes['Principled BSDF']
    p.inputs['Base Color'].default_value = (*rgb, 1); p.inputs['Roughness'].default_value = rough
    if emit:
        p.inputs['Emission Color'].default_value = (*rgb, 1); p.inputs['Emission Strength'].default_value = emit
    return m

def instancer(name, radius, mat):
    me = bpy.data.meshes.new(name); ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob)
    mod = ob.modifiers.new('inst', 'NODES')
    ng = bpy.data.node_groups.new(name + '_gn', 'GeometryNodeTree')
    ng.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    N_, Lk = ng.nodes, ng.links
    gi = N_.new('NodeGroupInput'); go = N_.new('NodeGroupOutput')
    m2p = N_.new('GeometryNodeMeshToPoints')
    sph = N_.new('GeometryNodeMeshUVSphere'); sph.inputs['Segments'].default_value = 20; sph.inputs['Rings'].default_value = 10
    sph.inputs['Radius'].default_value = radius
    sm = N_.new('GeometryNodeSetShadeSmooth'); Lk.new(sph.outputs['Mesh'], sm.inputs['Geometry'])
    inst = N_.new('GeometryNodeInstanceOnPoints')
    setm = N_.new('GeometryNodeSetMaterial'); setm.inputs['Material'].default_value = mat
    Lk.new(gi.outputs[0], m2p.inputs['Mesh']); Lk.new(m2p.outputs['Points'], inst.inputs['Points'])
    Lk.new(sm.outputs['Geometry'], inst.inputs['Instance']); Lk.new(inst.outputs['Instances'], setm.inputs['Geometry'])
    Lk.new(setm.outputs['Geometry'], go.inputs[0]); mod.node_group = ng
    return ob

def set_points(ob, P):
    me = ob.data; me.clear_geometry()
    if len(P):
        me.vertices.add(len(P)); me.vertices.foreach_set('co', np.asarray(P, np.float32).ravel())
    me.update()

obO = instancer('O', 0.15, material('O', (0.78, 0.07, 0.05), 0.3))
obH = instancer('H', 0.10, material('H', (0.93, 0.93, 0.93), 0.3))
obVO = instancer('vO', 0.15, material('vO', (1.0, 0.25, 0.15), 0.3, emit=1.2))
obVH = instancer('vH', 0.10, material('vH', (1.0, 1.0, 1.0), 0.3, emit=0.8))
obN = instancer('N2', 0.16, material('N2', (0.10, 0.35, 1.0), 0.25, emit=0.5))
obOO = instancer('O2', 0.15, material('O2', (0.15, 0.85, 0.30), 0.25, emit=0.5))
obAR = instancer('Ar', 0.19, material('Ar', (0.65, 0.30, 1.0), 0.25, emit=0.6))

# surface height and camera
u.trajectory[0]
oz = OW.positions[:, 2] / 10
zmid = 0.5 * sum(meta['slab'])
oz_u = oz[np.abs(oz - zmid) < 4]
z_s = float(np.percentile(oz_u, 98))           # upper liquid surface
c = LX / 2
bpy.ops.object.camera_add(location=(c, c - 15.0, z_s + 3.2))
cam = bpy.context.object; sc.camera = cam; cam.data.lens = 24; cam.data.clip_end = 300
tgt = np.array([c, c + 2.0, z_s + 1.4])
dvec = tgt - np.array(cam.location)
cam.rotation_euler = (math.atan2(math.hypot(dvec[0], dvec[1]), -dvec[2]), 0, math.atan2(dvec[1], dvec[0]) - math.pi / 2)
cam.data.dof.use_dof = True; cam.data.dof.focus_distance = float(np.linalg.norm(dvec)); cam.data.dof.aperture_fstop = 1.6
for loc, e, s in (((c - 14, c - 10, z_s + 18), 9000, 12), ((c + 16, c - 4, z_s + 6), 2500, 8), ((c, c + 20, z_s + 14), 4000, 12)):
    bpy.ops.object.light_add(type='AREA', location=loc); lt = bpy.context.object; lt.data.energy = e; lt.data.size = s
    dd = np.array([c, c, z_s]) - np.array(loc)
    lt.rotation_euler = (math.atan2(math.hypot(dd[0], dd[1]), -dd[2]), 0, math.atan2(dd[1], dd[0]) - math.pi / 2)

def tiles(P):
    return np.concatenate([P + [i * LX, j * LX, 0] for i in TILE for j in TILE]) if len(P) else P

z_b = float(np.percentile(oz_u, 2))            # lower liquid surface
def unwrap_up(z):            # gas below the slab is the periodic continuation of the gas above it
    return np.where(z < z_b - 1.0, z + LZ, z)

frames = list(range(len(u.trajectory)))[FIRST:None if LAST < 0 else LAST:STRIDE]
info = []
for k in frames:
    ts = u.trajectory[k]
    o = OW.positions / 10; h1 = H1.positions / 10; h2 = H2.positions / 10
    zo = unwrap_up(o[:, 2])
    sh = np.c_[np.floor(o[:, 0] / LX) * LX, np.floor(o[:, 1] / LX) * LX, o[:, 2] - zo]
    o2 = o - sh
    hh1 = h1 - sh; hh2 = h2 - sh
    vap = zo > z_s + 0.8                         # water molecules in the gas (vapour)
    liq = (zo > z_s - 1.6) & ~vap
    set_points(obO, tiles(o2[liq])); set_points(obH, tiles(np.r_[hh1[liq], hh2[liq]]))
    set_points(obVO, tiles(o2[vap])); set_points(obVH, tiles(np.r_[hh1[vap], hh2[vap]]))
    for grp, ob in ((NN, obN), (OO, obOO), (AR, obAR)):
        if len(grp) == 0:
            continue
        p = grp.positions / 10
        nat = 1 if grp is AR else 2
        p = p.reshape(-1, nat, 3)
        first = p[:, 0, :]
        shg = np.c_[np.floor(first[:, 0] / LX) * LX, np.floor(first[:, 1] / LX) * LX, first[:, 2] - unwrap_up(first[:, 2])]
        p = (p - shg[:, None, :]).reshape(-1, 3)
        set_points(ob, tiles(p))
    sc.render.filepath = f'{OUT}/f{k:05d}.png'
    bpy.ops.render.render(write_still=True)
    info.append(dict(frame=k, t_ps=float(ts.time)))
    print('frame', k, 't', ts.time, 'vapour', int(vap.sum()), flush=True)
json.dump(dict(frames=info, z_surface=z_s, meta=meta), open(f'{OUT}/frames.json', 'w'))
