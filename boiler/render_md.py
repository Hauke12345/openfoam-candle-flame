"""Render the kettle-wall MD run in Blender (bpy 5.x, Cycles): cut-away view of the water on the wall.

The front half of the liquid and gas (y < Ly/2 - CUT) is removed so the view looks into the water and through the gas
bubble on the less-wetting patch. Water: red O / white H; dissolved N2: glowing blue; wall: glass-like blue atoms,
the less-wetting patch amber. One Geometry-Nodes sphere instancer per species (fast for ~10^4 atoms).
usage: python3 render_md.py RUNDIR OUTDIR [deffnm=heat] [stride=5] [RX=960] [RY=540] [SPP=24] [first=0] [last=-1]
"""
import sys, os, json, math
import numpy as np
import bpy
import MDAnalysis as mda

RUN, OUT = sys.argv[1], sys.argv[2]
NAME = sys.argv[3] if len(sys.argv) > 3 else 'heat'
STRIDE = int(sys.argv[4]) if len(sys.argv) > 4 else 5
RX, RY, SPP = (int(sys.argv[i]) if len(sys.argv) > i else d for i, d in ((5, 960), (6, 540), (7, 24)))
FIRST = int(sys.argv[8]) if len(sys.argv) > 8 else 0
LAST = int(sys.argv[9]) if len(sys.argv) > 9 else -1
os.makedirs(OUT, exist_ok=True)
meta = json.load(open(f'{RUN}/meta.json'))
L = meta['LXY']; CUT = 0.0

u = mda.Universe(f'{RUN}/conf.gro', f'{RUN}/{NAME}.xtc')     # .gro as topology (MDAnalysis cannot read GROMACS 2026 .tpr)
OW, HW = u.select_atoms('name OW'), u.select_atoms('name HW1 HW2')
N2 = u.select_atoms('name N2')
WP, WH = u.select_atoms('name WP'), u.select_atoms('name WH')

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = SPP; sc.cycles.use_denoising = True
sc.render.resolution_x, sc.render.resolution_y = RX, RY
sc.render.image_settings.file_format = 'PNG'
sc.view_settings.view_transform = 'AgX'
world = bpy.data.worlds.new('w'); sc.world = world; world.use_nodes = True
world.node_tree.nodes['Background'].inputs[0].default_value = (0.012, 0.014, 0.02, 1)


def material(name, rgb, rough=0.35, emit=0.0, trans=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    p = m.node_tree.nodes['Principled BSDF']
    p.inputs['Base Color'].default_value = (*rgb, 1); p.inputs['Roughness'].default_value = rough
    if emit:
        p.inputs['Emission Color'].default_value = (*rgb, 1); p.inputs['Emission Strength'].default_value = emit
    if trans:
        p.inputs['Transmission Weight'].default_value = trans
    return m


def instancer(name, radius, mat):
    """mesh object whose vertices get a sphere instance each (Geometry Nodes)."""
    me = bpy.data.meshes.new(name); ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob)
    mod = ob.modifiers.new('inst', 'NODES')
    ng = bpy.data.node_groups.new(name + '_gn', 'GeometryNodeTree')
    ng.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    N, Lk = ng.nodes, ng.links
    gi = N.new('NodeGroupInput'); go = N.new('NodeGroupOutput')
    m2p = N.new('GeometryNodeMeshToPoints')
    sph = N.new('GeometryNodeMeshUVSphere'); sph.inputs['Segments'].default_value = 16; sph.inputs['Rings'].default_value = 8
    sph.inputs['Radius'].default_value = radius
    sm = N.new('GeometryNodeSetShadeSmooth'); Lk.new(sph.outputs['Mesh'], sm.inputs['Geometry'])
    inst = N.new('GeometryNodeInstanceOnPoints')
    setm = N.new('GeometryNodeSetMaterial'); setm.inputs['Material'].default_value = mat
    Lk.new(gi.outputs[0], m2p.inputs['Mesh']); Lk.new(m2p.outputs['Points'], inst.inputs['Points'])
    Lk.new(sm.outputs['Geometry'], inst.inputs['Instance']); Lk.new(inst.outputs['Instances'], setm.inputs['Geometry'])
    Lk.new(setm.outputs['Geometry'], go.inputs[0])
    mod.node_group = ng
    return ob


def set_points(ob, P):
    me = ob.data; me.clear_geometry()
    if len(P):
        me.vertices.add(len(P)); me.vertices.foreach_set('co', np.asarray(P, np.float32).ravel())
    me.update()


m_o = material('O', (0.75, 0.06, 0.05), 0.3)
m_h = material('H', (0.92, 0.92, 0.92), 0.3)
m_n = material('N2', (0.05, 0.35, 1.0), 0.2, emit=0.6)
m_wp = material('glass', (0.55, 0.75, 0.85), 0.15)
m_wh = material('patch', (0.95, 0.55, 0.15), 0.35)
obO, obH, obN = instancer('O', 0.15, m_o), instancer('H', 0.10, m_h), instancer('N2', 0.17, m_n)
obWP, obWH = instancer('WP', 0.16, m_wp), instancer('WH', 0.16, m_wh)

# camera: in front of the cut, slightly above, looking at the patch; scene units = nm
cx = cy = L / 2; zt = meta['z_wall_top']
bpy.ops.object.camera_add(location=(cx, -10.5, zt + 4.2))
cam = bpy.context.object; sc.camera = cam; cam.data.lens = 36
d = np.array([cx, cy, zt + 2.6]) - np.array(cam.location)
cam.rotation_euler = (math.atan2(math.hypot(d[0], d[1]), -d[2]), 0, math.atan2(d[1], d[0]) - math.pi / 2)
cam.data.dof.use_dof = True; cam.data.dof.focus_distance = float(np.linalg.norm(d)); cam.data.dof.aperture_fstop = 2.8
for loc, e, s in (((cx - 6, -6, 8), 2500, 6), ((cx + 7, -2, 4), 900, 5), ((cx, cy + 6, 9), 1200, 6)):
    bpy.ops.object.light_add(type='AREA', location=loc); lt = bpy.context.object
    lt.data.energy = e; lt.data.size = s
    dd = np.array([cx, cy, zt + 1.5]) - np.array(loc)
    lt.rotation_euler = (math.atan2(math.hypot(dd[0], dd[1]), -dd[2]), 0, math.atan2(dd[1], dd[0]) - math.pi / 2)

frames = list(range(len(u.trajectory)))[::STRIDE]
frames = frames[FIRST:None if LAST < 0 else LAST]
info = []
for k in frames:
    ts = u.trajectory[k]
    o = OW.positions / 10.0
    o[:, :2] %= L
    h = HW.positions.reshape(-1, 2, 3) / 10.0
    # put the H atoms next to their (wrapped) oxygen with the minimum image in x, y
    rel = h - (OW.positions / 10.0)[:, None, :]
    rel[..., :2] -= L * np.round(rel[..., :2] / L)
    h = o[:, None, :] + rel
    keep = o[:, 1] >= cy - CUT                  # cut-away: back half of the water only
    n = N2.positions / 10.0; n[:, :2] %= L
    set_points(obO, o[keep]); set_points(obH, h[keep].reshape(-1, 3)); set_points(obN, n[n[:, 1] >= cy - CUT - 0.2])
    set_points(obWP, WP.positions / 10.0); set_points(obWH, WH.positions / 10.0)
    sc.render.filepath = f'{OUT}/f{k:05d}.png'
    bpy.ops.render.render(write_still=True)
    info.append(dict(frame=k, t_ps=float(ts.time)))
    print('frame', k, 't', ts.time, flush=True)
json.dump(info, open(f'{OUT}/frames.json', 'w'))
