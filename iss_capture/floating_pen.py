"""A tumbling pen, shared by the GUI and sensor cameras."""
import xml.etree.ElementTree as ET
from .config import ROOT

POSE = (.50, .50, .12, .10, 1.40, .55)
ANGULAR_VELOCITY = (.24, .30, .36)
MASS = .012
LENGTH = .146
RADIUS = .0055


def add(parent, tag, value=None, **attrs):
    node = ET.SubElement(parent, tag, attrs)
    if value is not None:
        node.text = str(value)
    return node


def vector(values):
    return ' '.join(f'{v:.10g}' for v in values)


def pen_metadata():
    return {'name': 'floating_pen', 'pose': list(POSE), 'mass_kg': MASS,
            'length_m': LENGTH, 'angular_velocity_command_rad_s': list(ANGULAR_VELOCITY),
            'motion': 'rotation at a fixed centre using Gazebo VelocityControl',
            'collision_enabled': False, 'visible_to_policy_cameras': True,
            'capture_target': False}


def add_floating_pen(world):
    model = add(world, 'model', name='floating_pen')
    add(model, 'static', 'false')
    add(model, 'pose', vector(POSE))
    link = add(model, 'link', name='pen_body')
    add(link, 'gravity', 'false')
    inertial = add(link, 'inertial')
    add(inertial, 'mass', MASS)
    inertia = add(inertial, 'inertia')
    transverse = MASS * (3 * RADIUS**2 + LENGTH**2) / 12
    for name, value in {'ixx': transverse, 'iyy': transverse,
                        'izz': MASS * RADIUS**2 / 2, 'ixy': 0, 'ixz': 0, 'iyz': 0}.items():
        add(inertia, name, value)

    orange, rubber, silver = '0.94 0.28 0.035 1', '0.045 0.05 0.055 1', '0.68 0.72 0.75 1'

    def visual(name, position, color):
        item = add(link, 'visual', name=name)
        add(item, 'pose', vector([*position, 0, 0, 0]))
        add(item, 'cast_shadows', 'true')
        material = add(item, 'material')
        add(material, 'ambient', color)
        add(material, 'diffuse', color)
        add(material, 'specular', '.55 .55 .55 1' if color == silver else '.12 .12 .12 1')
        return add(item, 'geometry')

    def cylinder(name, z, radius, length, color):
        geometry = add(visual(name, (0, 0, z), color), 'cylinder')
        add(geometry, 'radius', radius)
        add(geometry, 'length', length)

    cylinder('orange_barrel', .015, .0048, .09, orange)
    cylinder('rubber_grip', -.045, .0053, .03, rubber)
    for index, z in enumerate((-.056, -.050, -.044, -.038)):
        cylinder(f'grip_ring_{index}', z, .0056, .0012, '0.16 0.17 0.18 1')
    cylinder('grip_collar', -.03, .0054, .002, silver)
    cylinder('rear_collar', .059, .0051, .004, silver)
    cylinder('push_button', .065, .0033, .010, rubber)
    nib = add(visual('metal_nib', (0, 0, -.06), silver), 'mesh')
    path = ROOT / 'assets/floating_pen/nib.obj'
    if not path.is_file():
        raise FileNotFoundError(f'ペンの先端メッシュがありません: {path}')
    add(nib, 'uri', path.as_uri())
    tip = add(visual('ballpoint', (0, 0, -.075), rubber), 'sphere')
    add(tip, 'radius', .0008)
    for name, xyz, size in (
            ('clip_anchor', (.0052, 0, .052), (.003, .0025, .006)),
            ('metal_clip', (.0062, 0, .025), (.0015, .0025, .052))):
        add(add(visual(name, xyz, silver), 'box'), 'size', vector(size))

    # Decorative prop: it must not change water/robot contacts or count as a
    # capture. The standard velocity controller animates it while time runs.
    motion = add(model, 'plugin', filename='gz-sim-velocity-control-system',
                 name='gz::sim::systems::VelocityControl')
    add(motion, 'initial_linear', '0 0 0')
    add(motion, 'initial_angular', vector(ANGULAR_VELOCITY))
    return model
