"""Collidable, freely rotating pen and two contact-sensing finger pads."""
import json
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .config import ROOT, JOINTS
from .world import child, box, numbers, plugin
from .moving_world import build_moving
from .pen_kinematics import PenArm, HOME, SCENARIOS, FIXED_PAD, FIXED_SIZE, MOVING_PAD, MOVING_SIZE,LOWER,UPPER
from .floating_pen import MASS, LENGTH, RADIUS
from .trajectory_control import configure_robot_control

SPIN_SPEED = float(np.linalg.norm([.24,.30,.36]))


def friction(collision):
    surface = child(collision, 'surface')
    ode = child(child(surface, 'friction'), 'ode')
    child(ode, 'mu', 1.5); child(ode, 'mu2', 1.5)
    bounce = child(surface, 'bounce'); child(bounce, 'restitution_coefficient', 0)


def build_pen(output, scenario='center', goal_q=None, spin_speed=SPIN_SPEED, phase=0., view='close', record_view=False):
    output = Path(output).resolve()
    path, base = build_moving(output, scenario='left', appearance='training', environment='iss', view='wide')
    root = ET.parse(path).getroot(); world = root.find('world')
    arm = PenArm(); goal = np.array(SCENARIOS[scenario] if goal_q is None else goal_q, dtype=float)
    goal[4] += phase
    target = arm.forward(goal)
    hand = arm.transform(goal)[:3,:3]
    pen_rotation = np.column_stack([hand[:,0], -hand[:,2], hand[:,1]])
    quaternion = Rotation.from_matrix(pen_rotation).as_quat()
    omega = -hand[:,2] * spin_speed  # Perpendicular to the pen's long axis.
    world.remove(world.find("model[@name='droplet']"))
    robot = world.find("model[@name='so101']")
    robot.remove(robot.find("plugin[@name='gz::sim::systems::TouchPlugin']"))
    fixed = robot.find("link[@name='gripper_link']")
    moving = robot.find("link[@name='moving_jaw_so101_v1_link']")
    for link in (fixed, moving):
        for item in list(link):
            if item.tag in ('collision','sensor') or item.get('name') == 'sponge_visual': link.remove(item)
    box(fixed, 'fixed_pad', FIXED_SIZE, [*FIXED_PAD,0,0,0], '.09 .10 .11 1', True)
    # A shallow, visible groove prevents a round barrel from rolling out of
    # the hinged jaws. These are ordinary colliders, never an attachment.
    for side,z in [('front',-.103),('back',-.081)]:
        box(fixed, f'fixed_lip_{side}', [.009,.026,.008], [-.0064,-.0002,z,0,0,0], '.12 .13 .14 1', True)
    box(moving, 'moving_pad', MOVING_SIZE, [*MOVING_PAD,0,0,0], '.09 .10 .11 1', True)
    box(fixed, 'palm', [.044,.038,.030], [.006,0,-.024,0,0,0], '.08 .09 .10 1', True)
    for side, link in [('fixed',fixed),('moving',moving)]:
        for collision in link.findall('collision'): friction(collision)
        sensor = child(link, 'sensor', name=f'{side}_pad_touch', type='contact')
        child(sensor, 'always_on', 'true'); child(sensor, 'update_rate', 200)
        contact = child(sensor, 'contact')
        child(contact, 'collision', f'{side}_pad_collision')
        if side=='fixed':
            child(contact,'collision','fixed_lip_front_collision')
            child(contact,'collision','fixed_lip_back_collision')
        child(contact, 'topic', f'/pen/{side}_contacts')
    control = configure_robot_control(world, output, HOME)
    pen = world.find("model[@name='floating_pen']")
    # Park the object while the arm homes. The one initial reset occurs before
    # the task starts; no pose reset or attachment is allowed during grasping.
    pen.find('pose').text = numbers([target[0],target[1],.8,0,0,0])
    for item in list(pen.findall('plugin')): pen.remove(item)
    plugin(pen, 'gz-sim-pose-publisher-system', 'gz::sim::systems::PosePublisher',
           publish_model_pose='true', publish_link_pose='false',
           use_pose_vector_msg='true', update_frequency=100,
           topic='/evaluation/object_pose')
    body = pen.find('link')
    collider = child(body, 'collision', name='pen_collision')
    cylinder = child(child(collider, 'geometry'), 'cylinder')
    child(cylinder, 'radius', .0054); child(cylinder, 'length', .143)
    friction(collider)
    if view == 'close':
        eye = target + [.20,-.30,.20]; delta = target+[0,0,.025]-eye
        pose = [*eye,0,np.arctan2(-delta[2], np.linalg.norm(delta[:2])),np.arctan2(delta[1],delta[0])]
        world.find("gui/plugin[@filename='MinimalScene']/camera_pose").text = numbers(pose)
        world.find("gui/plugin[@filename='MinimalScene']/horizontal_fov").text = '60'
    if record_view:
        camera=child(world,'model',name='pen_display_camera');child(camera,'static','true')
        child(camera,'pose',world.findtext("gui/plugin[@filename='MinimalScene']/camera_pose"))
        sensor=child(child(camera,'link',name='body'),'sensor',name='display',type='camera')
        child(sensor,'always_on','true');child(sensor,'update_rate',10);child(sensor,'topic','/pen/display/image')
        spec=child(sensor,'camera');child(spec,'horizontal_fov',np.deg2rad(float(world.findtext("gui/plugin[@filename='MinimalScene']/horizontal_fov",'60'))))
        image=child(spec,'image');child(image,'width',640);child(image,'height',360);child(image,'format','R8G8B8')
        clip=child(spec,'clip');child(clip,'near',.02);child(clip,'far',5)
    ET.indent(root, space='  ')
    path = output/'rotating_pen.sdf'
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
    result = subprocess.run(['gz','sdf','-k',str(path)], capture_output=True,text=True,timeout=60)
    (output/'pen_sdf_validation.log').write_text(result.stdout+result.stderr)
    if result.returncode: raise RuntimeError(result.stdout+result.stderr)
    manifest = {'task':'rotating_pen_grasp','version':1,'scenario':scenario,
        'environment':'iss','environment_version':base['environment_version'],
        'initial_pen_xyz':target.tolist(),'initial_pen_quaternion_xyzw':quaternion.tolist(),
        'initial_angular_velocity_world_rad_s':omega.tolist(),'spin_speed_rad_s':spin_speed,
        'home_q':HOME.tolist(),'generating_q':goal.tolist(),'urdf_sha256':arm.sha256,
        'command_limits':{'lower':LOWER.tolist(),'upper':UPPER.tolist()},
        'policy_camera_poses':base['policy_camera_poses'], 'pen_collision_enabled':True,
        'policy_camera_far_clip_m':base['policy_camera_far_clip_m'],
        'viewer_pose':[float(x) for x in world.findtext("gui/plugin[@filename='MinimalScene']/camera_pose").split()],
        'display_camera_recording':record_view,'display_camera_is_policy_input':False,
        'initial_motion':'single angular impulse; no persistent velocity control',
        'grasp_mechanism':'grooved frictional finger pads; no attachment or fixed joint',
        'joint_control':control,
        'arm_servo':'standard JointTrajectoryController in Gazebo; no application servo thread',
        'success':'bilateral finger contacts, closed jaw, >=30 mm upward carry and >=0.6 s stable hold'}
    (output/'scene.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return path, manifest
