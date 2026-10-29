"""Original Kibo-inspired cabin, guided by the user-supplied interior reference.

The Earth viewport and rack-mounted work surface retain their locations.
Props are visible to every camera and have no collision geometry.
"""
import math
import xml.etree.ElementTree as ET
from .config import ROOT

ENVIRONMENTS = ('iss', 'lab')
ISS_ENVIRONMENT_VERSION = 6
REFERENCE_URL = 'https://assets.moguravr.com/uploads/2024/12/cbc87ace5d2e20ad6b3e90468aa46c73-kix.ej7onl6a7uq7-jpg.webp'
IVORY = '0.91 0.91 0.87 1'
PANEL = '0.95 0.95 0.92 1'
CLOTH = '0.83 0.83 0.78 1'
TRIM = '0.52 0.56 0.57 1'
BLUE = '0.045 0.20 0.36 1'
DARK = '0.035 0.045 0.052 1'
SEAM = '0.60 0.64 0.63 1'
YELLOW = '0.87 0.76 0.24 1'
BRONZE = '0.44 0.37 0.24 1'
ASSETS = ROOT / 'assets/kibo'


def add(parent, tag, value=None, **attrs):
    item = ET.SubElement(parent, tag, attrs)
    if value is not None:
        item.text = str(value)
    return item


def vector(values):
    return ' '.join(f'{v:.8g}' for v in values)


class Props:
    def __init__(self, world, name):
        self.model = add(world, 'model', name=name)
        add(self.model, 'static', 'true')
        self.link = add(self.model, 'link', name='structure')
        self.count = 0

    def visual(self, name, xyz, rgba, rpy=(0,0,0), unlit=False, glow=False):
        self.count += 1
        v = add(self.link, 'visual', name=f'{name}_{self.count}')
        add(v, 'pose', vector([*xyz, *rpy]))
        add(v, 'cast_shadows', 'false')
        m = add(v, 'material')
        add(m, 'ambient', rgba); add(m, 'diffuse', rgba)
        if rgba in (IVORY, PANEL):
            # Mild baked bounce-light appearance; it emits no light onto the
            # robot or target, so their calibrated illumination is unchanged.
            add(m, 'emissive', '0.12 0.12 0.11 1')
        if rgba in (TRIM, BLUE):
            add(m, 'specular', '0.28 0.28 0.28 1');add(m, 'shininess', 55)
        if unlit: add(m, 'lighting', 'false')
        if glow: add(m, 'emissive', rgba)
        return v, add(v, 'geometry')

    def box(self, name, xyz, size, color=IVORY, rpy=(0,0,0), **kwargs):
        v, g = self.visual(name, xyz, color, rpy, **kwargs)
        add(add(g, 'box'), 'size', vector(size))
        return v

    def sphere(self, name, xyz, radius, color, **kwargs):
        v, g = self.visual(name, xyz, color, **kwargs)
        add(add(g, 'sphere'), 'radius', radius)
        return v

    def rod(self, name, a, b, radius, color=TRIM):
        d = [y-x for x,y in zip(a,b)]
        length = math.sqrt(sum(x*x for x in d))
        if length < 1e-8: return None
        rpy = (0, math.acos(max(-1.,min(1.,d[2]/length))), math.atan2(d[1],d[0]))
        v, g = self.visual(name, [(x+y)/2 for x,y in zip(a,b)], color, rpy)
        cylinder = add(g, 'cylinder')
        add(cylinder, 'radius', radius);add(cylinder, 'length', length)
        return v

    def decal(self, name, xyz, size, texture, rpy):
        path = ASSETS / texture
        if not path.is_file():
            raise FileNotFoundError(f'Kibo texture missing: {path}; run scripts/build_kibo_textures.py')
        v, g = self.visual(name, xyz, '1 1 1 1', rpy)
        add(v.find('material'),'emissive','0.08 0.08 0.07 1')
        mesh = add(g,'mesh');add(mesh,'uri',(ASSETS/'decal.obj').as_uri())
        add(mesh,'scale',vector([*size,1]))
        metal = add(add(v.find('material'),'pbr'),'metal')
        add(metal,'metalness',0);add(metal,'roughness',.86)
        add(metal,'albedo_map',path.as_uri())
        return v

    def cable(self, name, points, radius=.006, color=DARK):
        # Catmull-Rom bends made of short smooth-shaded cylinders.
        padded = [points[0],*points,points[-1]]
        previous = points[0]
        for i in range(1,len(padded)-2):
            a,b,c,d = padded[i-1:i+3]
            for step in range(1,5):
                t=step/4
                point=[.5*((2*b[j])+(-a[j]+c[j])*t+(2*a[j]-5*b[j]+4*c[j]-d[j])*t*t+(-a[j]+3*b[j]-3*c[j]+d[j])*t*t*t) for j in range(3)]
                self.rod(name,previous,point,radius,color)
                previous=point


def _shell(p):
    p.box('deck',(-.9,0,-.75),(5.4,2.3,.07),BLUE)
    p.box('ceiling',(-.9,0,1.45),(5.4,1.76,.05),PANEL)
    p.box('aft_bulkhead',(-3.60,0,.35),(.06,2.3,2.2),IVORY)
    p.box('near_wall',(-.9,-1.15,.35),(5.4,.05,2.2),IVORY)
    # Leave the original Earth viewport opening entirely unobstructed.
    p.box('far_wall_left',(-1.45,1.15,.35),(4.30,.05,2.2),IVORY)
    p.box('far_wall_right',(1.55,1.15,.35),(.50,.05,2.2),IVORY)
    p.box('window_below',(1.,1.15,-.30),(.6,.05,.9),IVORY)
    p.box('window_above',(1.,1.15,1.15),(.6,.05,.6),IVORY)
    for x in (-3.15,-2.35,-1.55,-.75,.05,.85,1.5):
        width=.56 if x==1.5 else .77
        p.box('deck_access_panel',(x,0,-.704),(width,1.68,.019),PANEL)
        p.box('deck_center_seam',(x,0,-.692),(width,.006,.003),SEAM)
        for y in (-.81,.81):
            p.box('deck_edge_track',(x,y,-.682),(width,.035,.018),TRIM)
        for y in (-.68,.68):
            for dx in (-width*.38,width*.38):
                p.rod('deck_fastener',(x+dx,y,-.691),(x+dx,y,-.686),.005,TRIM)
    for sign in (-1,1):
        p.box('navy_kick_strip',(-.9,sign*.93,-.64),(5.3,.12,.15),BLUE)
        p.box('angled_ceiling_panel',(-.9,sign*1.005,1.28),(5.4,.39,.045),PANEL,rpy=(sign*math.pi/4,0,0))
        p.box('ceiling_utility_trunk',(-.9,sign*.80,1.38),(5.30,.16,.14),TRIM)
        for x in (-2.9,-1.65,-.40,.85):
            p.box('luminaire_body',(x,sign*.91,1.24),(1.08,.115,.065),TRIM,rpy=(sign*math.pi/4,0,0))
            p.box('luminaire_diffuser',(x,sign*.875,1.205),(.98,.075,.022),'0.88 0.92 0.94 1',rpy=(sign*math.pi/4,0,0),glow=True)
            for dx in (-.51,.51):
                p.box('lamp_end_cap',(x+dx,sign*.875,1.205),(.035,.09,.04),IVORY,rpy=(sign*math.pi/4,0,0))
        for x in (-3.20,-2.35,-1.50,-.65,.20,1.05,1.65):
            for offset in range(7):
                p.box('duct_perforation',(x+offset*.042,sign*.709,1.37),(.020,.004,.053),DARK)
        p.rod('ceiling_blue_handrail',(-3.3,sign*.70,1.34),(1.6,sign*.70,1.34),.014,BLUE)
    for x in (-2.7,-1.3,.1,1.2):
        p.box('ceiling_service_panel',(x,0,1.412),(.78,1.16,.016),IVORY)
        for y in (-.49,.49):
            for dx in (-.32,.32):p.box('ceiling_panel_latch',(x+dx,y,1.396),(.035,.045,.018),TRIM)
        p.rod('overhead_crossbar',(x,-.30,1.36),(x,.30,1.36),.012,BLUE)


def _rack(p,x,sign,index,width=.68):
    y=sign*.965
    p.box('rack_structural_frame',(x,sign*1.055,.27),(width,.15,1.83),TRIM)
    p.box('rack_panel',(x,y,.27),(width-.035,.035,1.77),PANEL)
    for dx in (-width/2+.012,width/2-.012):
        p.box('rack_blue_edge',(x+dx,y-sign*.032,.27),(.019,.024,1.80),BLUE)
    # Asymmetric hardware bays alternate with broad padded access panels.
    hardware = (sign==-1 and index%3==1) or (sign==1 and index%3==0)
    texture='instrument.png' if hardware else ('quilt.png' if index%2==0 else 'storage.png')
    rpy=(math.pi/2,0,0 if sign==1 else math.pi)
    p.decal('instrument_labels' if hardware else 'padded_access_panel',
            (x,y-sign*.020,.27),(width-.068,1.72),texture,rpy)
    if hardware:
        for row,z in enumerate((.82,.48,.12,-.25)):
            p.box('connector_tray',(x,y-sign*.044,z),(width-.11,.028,.11),TRIM)
            count=5 if width>.5 else 3
            for j in range(count):
                cx=x+(j-(count-1)/2)*.095
                p.rod('connector_outer',(cx,y-sign*.058,z),(cx,y-sign*.092,z),.025,TRIM)
                p.rod('connector_insert',(cx,y-sign*.093,z),(cx,y-sign*.097,z),.017,DARK)
                if row==0:
                    p.sphere('power_indicator',(cx,y-sign*.101,z+.038),.006,'0.18 0.48 0.24 1')
        p.box('black_interface_module',(x-.12,y-sign*.075,.65),(.15,.07,.14),DARK)
        p.box('white_interface_module',(x+.14,y-sign*.06,.65),(.17,.045,.14),IVORY)
        for row in range(5):
            p.box('module_vent',(x-.12,y-sign*.113,.60+row*.022),(.105,.003,.005),TRIM)
        for j,color in enumerate((IVORY,TRIM,BLUE)):
            cx=x-.14+j*.14
            p.cable('connected_patch_cable',[(cx,y-sign*.102,.47),(cx+.06,y-sign*.18,.40),
                    (cx+.04,y-sign*.20,.22),(cx-.07,y-sign*.105,.13)],.006,color)
    else:
        for z in (-.36,.20,.79):
            for dx in (-width*.37,width*.37):
                p.box('panel_toggle_latch',(x+dx,y-sign*.037,z),(.035,.026,.085),TRIM)
                p.rod('latch_pin',(x+dx-.013,y-sign*.053,z),(x+dx+.013,y-sign*.053,z),.005,DARK)
        if index%2==0:
            for z in (-.06,.61):
                p.rod('panel_restraint_rail',(x-width*.43,y-sign*.074,z),(x+width*.43,y-sign*.074,z),.008,TRIM)
            p.rod('yellow_pull_handle',(x+.16,y-sign*.08,.26),(x+.16,y-sign*.08,.39),.010,YELLOW)
    for z in (-.59,1.11):
        for dx in (-width*.39,width*.39):
            p.rod('rack_fastener',(x+dx,y-sign*.020,z),(x+dx,y-sign*.028,z),.005,TRIM)
    if index in (1,3,5):
        railx=x-width*.38
        p.rod('vertical_blue_handrail',(railx,y-sign*.11,-.35),(railx,y-sign*.11,.95),.014,BLUE)
        for z in (-.35,.95):p.rod('handrail_standoff',(railx,y-sign*.03,z),(railx,y-sign*.11,z),.012,TRIM)


def _racks_and_hoses(p):
    positions=(-2.95,-2.19,-1.43,-.67,.09)
    for sign in (-1,1):
        for i,x in enumerate(positions):_rack(p,x,sign,i)
        if sign==-1:
            _rack(p,.85,sign,5)
            _rack(p,1.5,sign,6,.46)
        else:_rack(p,1.55,sign,5,.40)
        for offset,color in ((0.,IVORY),(.034,TRIM)):
            p.cable('long_service_pipe',[(-3.40,sign*.87,1.08+offset),(-2.7,sign*.86,1.08+offset),
                    (-1.4,sign*.86,1.10+offset),(-.6,sign*.86,1.07+offset),(.45,sign*.87,1.09+offset)],.009,color)
        # A few conspicuous hose loops and exposed service bundles, as in the photo.
        for x in (-2.15,-.64):
            p.cable('overhead_hose_loop',[(x-.15,sign*.86,1.12),(x-.25,sign*.78,.93),
                    (x-.17,sign*.75,.78),(x+.04,sign*.77,.81),(x+.13,sign*.85,1.10)],.012,DARK)
        p.cable('green_service_line',[(-2.80,sign*.87,1.17),(-2.70,sign*.83,.90),
                (-2.56,sign*.84,.55),(-2.58,sign*.85,-.48)],.006,'0.26 0.43 0.34 1')
    # Fabric-covered stowage at the lower edges, not on the task worktop.
    for i,x in enumerate((-2.75,-1.84,-1.05)):
        p.box('soft_stowage_bag',(x,.76,-.47),(.46,.30,.35),CLOTH)
        p.decal('bag_fabric',(x,.603,-.47),(.43,.31),'quilt.png',(math.pi/2,0,0))
        for dx in (-.15,.15):p.box('bag_restraint',(x+dx,.59,-.47),(.026,.016,.35),IVORY)
    p.rod('padded_supply_bottle',(-2.40,-.80,-.52),(-2.40,-.80,-.01),.075,CLOTH)
    p.rod('bottle_neck',(-2.40,-.80,-.01),(-2.40,-.80,.045),.025,TRIM)


def _aft_equipment(p):
    # Two dark circular ports and a large round equipment/airlock face.
    for y in (-.61,.61):
        p.rod('port_bronze_ring',(-3.56,y,.70),(-3.49,y,.70),.245,BRONZE)
        p.rod('port_dark_glass',(-3.485,y,.70),(-3.478,y,.70),.211,DARK)
        for i in range(20):
            a=i*2*math.pi/20
            cy,cz=y+.229*math.cos(a),.70+.229*math.sin(a)
            p.rod('port_ring_bolt',(-3.477,cy,cz),(-3.469,cy,cz),.007,IVORY)
    p.box('airlock_support',(-3.33,0,-.54),(.51,1.16,.22),TRIM)
    p.box('airlock_housing',(-3.37,0,.0),(.38,1.18,.96),IVORY)
    p.rod('airlock_outer_rim',(-3.25,0,.03),(-3.08,0,.03),.568,TRIM)
    p.rod('airlock_white_face',(-3.075,0,.03),(-3.045,0,.03),.543,PANEL)
    p.rod('airlock_inset_lip',(-3.041,0,.03),(-3.030,0,.03),.507,IVORY)
    p.rod('upper_sight_ring',(-3.020,0,.27),(-2.995,0,.27),.123,TRIM)
    p.rod('upper_sight_glass',(-2.990,0,.27),(-2.983,0,.27),.094,DARK)
    for i in range(8):
        a=i*2*math.pi/8
        p.box('airlock_latch',(-3.010,.50*math.sin(a),.03+.50*math.cos(a)),(.025,.046,.025),YELLOW,rpy=(a,0,0))
    for i in range(18):
        a=i*2*math.pi/18
        p.rod('airlock_face_bolt',(-3.028,.467*math.cos(a),.03+.467*math.sin(a)),
              (-3.016,.467*math.cos(a),.03+.467*math.sin(a)),.005,TRIM)
    p.rod('airlock_handle',(-2.98,-.31,.22),(-2.98,.28,-.09),.015,TRIM)
    p.rod('airlock_handle_hub',(-3.02,0,.05),(-2.955,0,.05),.047,SEAM)
    for y,z in ((-.14,-.24),(.14,-.24),(0,-.35)):
        p.rod('airlock_small_socket',(-3.023,y,z),(-3.012,y,z),.019,DARK)
    for y in (-.59,.59):
        p.rod('airlock_side_grab',(-3.04,y,-.22),(-3.04,y,.27),.014,IVORY)
    p.box('overhead_fabric_banner',(-3.12,0,1.11),(.026,1.20,.36),CLOTH)
    p.decal('japan_kibo_banner',(-3.101,0,1.11),(1.17,.33),'banner.png',(math.pi/2,0,math.pi/2))
    p.rod('banner_support',(-3.12,-.64,1.32),(-3.12,.64,1.32),.01,TRIM)


def _viewport(p):
    # Viewport: frame, transparent pane, dark space beyond and a blue limb.
    for x in (.70, 1.30):
        p.box("window_frame_vertical", (x, 1.105, .5), (.045, .10, .75), TRIM)
    for z in (.15, .85):
        p.box("window_frame_horizontal", (1.0, 1.105, z), (.64, .10, .045), TRIM)
    pane = p.box("viewport_glass", (1.0, 1.15, .50), (.56, .012, .65), "0.25 0.49 0.64 1")
    add(pane, "transparency", .86)
    p.box("space_backdrop", (1.0, 2.65, .45), (2.2, .04, 2.1), "0.001 0.003 0.009 1", unlit=True)
    p.sphere("earth_limb", (1.0, 2.5, -.6), 1.04, "0.02 0.24 0.52 1")
    for x, z in ((.76, .9), (1.16, 1.02), (.85, .72), (1.30, .8)):
        p.sphere("distant_star", (x, 2.4, z), .006, "0.8 0.9 1 1", glow=True)


def _workstation(world):
    # Worktop cantilevers from the far-side equipment rack. Nothing under the
    # desk reaches the cabin floor; the task's original work surface stays put.
    f = Props(world, "iss_experiment_fixture")
    f.box("rack_worktop_extension", (.23, .5925, -.025), (.65, .455, .025), "0.12 0.16 0.23 1")
    for y in (.45, .55, .65, .75):
        f.box("extension_grid_y", (.23, y, -.011), (.64, .001, .001), "0.28 0.38 0.50 1")
    for x in (-.07, .03, .13, .23, .33, .43, .53):
        f.box("extension_grid_x", (x, .5925, -.011), (.001, .445, .001), "0.28 0.38 0.50 1")
    f.box("rack_mount_interface", (.23, .875, -.265), (.76, .13, .48), IVORY)
    for x in (-.06, .52):
        f.box("rack_mount_rail", (x, .835, -.26), (.055, .045, .48), TRIM)
        f.box("cantilever_beam", (x, .24, -.065), (.045, 1.19, .055), TRIM)
        f.rod("cantilever_brace", (x, .815, -.435), (x, -.23, -.095), .022, TRIM)
        for z in (-.08, -.43):
            f.rod("rack_mount_bolt", (x, .803, z), (x, .825, z), .010, BLUE)
    f.box("worktop_front_fascia", (.23, -.365, -.06), (.65, .022, .06), TRIM)
    for x in (-.095, .555):
        f.box("worktop_side_rim", (x, .225, -.035), (.014, 1.19, .035), TRIM)
    f.box("worktop_back_rim", (.23, .815, -.035), (.65, .014, .035), TRIM)
    f.rod("desk_grab_handle", (.575, -.28, -.09), (.575, .07, -.09), .012, BLUE)
    for y in (-.28, .07):
        f.rod("desk_handle_mount", (.55, y, -.065), (.575, y, -.09), .009, TRIM)
    return f


def add_iss_cabin(world):
    p=Props(world,'iss_cabin')
    _shell(p)
    _racks_and_hoses(p)
    _aft_equipment(p)
    _viewport(p)
    f=_workstation(world)
    return {'profile':'iss','version':ISS_ENVIRONMENT_VERSION,
            'workstation':'side-rack cantilever; no floor legs or standing sign panels',
            'reference':REFERENCE_URL,'reference_interpretation':'Kibo-like white equipment and padded panels, blue rails, angled lights, hoses and round aft equipment; original simplified geometry',
            'decorative_visual_count':p.count+f.count,'cabin_props_have_collision':False,
            'earth_viewport_preserved':True,'policy_cameras_changed':False,'hidden_camera_layers':False}
