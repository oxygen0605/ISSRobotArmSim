"""Task-specific scene variants, preserving the original evaluation runtime."""
import json
import os
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
from ros_gz_interfaces.msg import Entity, EntityWrench
from sensor_msgs.msg import Image

from .unified_config import ROOT, case_from_seed
from .unified_world import build as build_shared
from .unified_session import Session as SharedSession, Client
from .capping_motion import start_motion
from .config import DROPLET_RADIUS
from .ros_transport import stamp_seconds, rgb_image
from .world import child, numbers, plugin
from .bimanual_world import add_camera
from run_demo import stop_process


def load_scene(task, seed, config=None):
    source = Path(config or ROOT / 'config/task_scenes.json')
    spec = json.loads(source.read_text())['tasks'][task]
    present = set(spec['objects'])
    if not present or not present <= {'droplet', 'floating_pen', 'floating_cap'}:
        raise ValueError('Invalid scene objects')
    if spec.get('sponge_side') not in (None, 'left', 'right'):
        raise ValueError('Invalid sponge side')
    case = case_from_seed(seed)
    for key in ('pen_position', 'cap_position', 'water_position', 'water_velocity',
                'linear_velocity', 'angular_velocity', 'pen_tilt', 'cap_tilt'):
        if key in spec:
            value = np.asarray(spec[key], float)
            if value.shape != ((2,) if key.endswith('_tilt') else (3,)) or not np.isfinite(value).all():
                raise ValueError('Invalid scene vector: ' + key)
            case[key] = value.tolist()
    return spec, case


def prepare_sponge_grasp(world, robot, manifest, spec):
    """Place a free rigid sponge between the pads, then close with contact physics."""
    from .pen_kinematics import PenArm, MOVING_PAD, MOVING_SIZE, FIXED_PAD
    from .kinematics import rotation
    from .bimanual_kinematics import BimanualArm
    side = spec['sponge_side']
    link = robot.find(f"link[@name='{side}_gripper_link']")
    size = np.asarray(spec.get('sponge_size', [.018, .030, .024]), float)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError('Invalid sponge size')
    # The V-pad inner surfaces are x = -0.012 + abs(z - FIXED_PAD.z).
    # Start with 0.15 mm clearance from the fixed pad; closing creates contacts.
    centre = np.array([-.012 + size[2]/2 + size[0]/2 + .00015, FIXED_PAD[1], FIXED_PAD[2]])
    arm = PenArm()
    def clearance(angle):
        orientation = arm.jaw_origin[:3, :3] @ rotation([0, 0, 1], angle)
        inner = orientation @ (MOVING_PAD - [MOVING_SIZE[0]/2, 0, 0]) + arm.jaw_origin[:3, 3]
        normal = orientation[:, 0]
        return float(normal @ (inner-centre) - np.abs(normal) @ (size/2))
    jaw = brentq(lambda angle: clearance(angle)-.0004, -.17, .55)
    home = np.asarray(manifest['home_q'], float)
    index = 5 if side == 'left' else 11
    home[index] = jaw
    manifest['home_q'] = home.tolist()
    urdf_path = Path(manifest['joint_control']['robot_description'])
    urdf = ET.parse(urdf_path)
    initial = urdf.find(f".//ros2_control/joint[@name='{side}_gripper']/state_interface[@name='position']/param[@name='initial_value']")
    if initial is None:
        raise RuntimeError('Gripper initial-value parameter missing')
    initial.text = str(jaw)
    urdf.write(urdf_path, encoding='utf-8', xml_declaration=True)
    offset = 0 if side == 'left' else 6
    transform = BimanualArm(side).transform(home[offset:offset+6])
    position = transform[:3, :3] @ centre + transform[:3, 3]
    pose = [*position, *Rotation.from_matrix(transform[:3, :3]).as_euler('xyz')]
    model = child(world, 'model', name='held_sponge')
    child(model, 'static', 'false')
    child(model, 'pose', numbers(pose))
    body = child(model, 'link', name='body')
    child(body, 'gravity', 'false')
    mass = .002
    inertial = child(body, 'inertial')
    child(inertial, 'mass', mass)
    inertia = child(inertial, 'inertia')
    for axis, pair in [('ixx', (1, 2)), ('iyy', (0, 2)), ('izz', (0, 1))]:
        child(inertia, axis, mass * sum(size[i]**2 for i in pair) / 12)
    for axis in ('ixy', 'ixz', 'iyz'):
        child(inertia, axis, 0)
    for item in list(link):
        if not item.get('name', '').startswith('sponge'):
            continue
        link.remove(item)
        if item.tag in ('collision', 'visual'):
            item.find('pose').text = '0 0 0 0 0 0'
            item.find('geometry/box/size').text = numbers(size)
        if item.tag == 'collision':
            from .pen_world import friction
            friction(item)
            item.find('surface/friction/ode/mu').text = '1.2'
            item.find('surface/friction/ode/mu2').text = '1.2'
        body.append(item)
    server = world.find("plugin[@name='iss::WaterSystem']")
    server.find('robot_model').text = 'held_sponge'
    server.find('sponge_link').text = 'body'
    server.find('sponge_center').text = '0 0 0'
    manifest['sponge'] = dict(model='held_sponge', link='body', size=size.tolist(),
        mass_kg=mass, dynamic=True, fixed_joint=False, initial_pose=pose,
        local_initial_centre=centre.tolist(), initial_jaw=jaw,
        holding_arm=side, grasp='bilateral physical pad contact; no rigid attachment')
    plugin(model, 'gz-sim-pose-publisher-system', 'gz::sim::systems::PosePublisher',
           publish_model_pose='true', publish_link_pose='false', use_pose_vector_msg='true',
           update_frequency=100, topic='/evaluation/object_pose')


def build_scene(output, case, spec, record=True):
    path, manifest = build_shared(output, case, record)
    root = ET.parse(path).getroot()
    world = root.find('world')
    for name in ('droplet', 'floating_pen', 'floating_cap'):
        model = world.find(f"model[@name='{name}']")
        if name not in spec['objects'] and model is not None:
            world.remove(model)
    robot = world.find("model[@name='bimanual_so101']")
    source = robot.find("link[@name='left_gripper_link']")
    target = spec.get('sponge_side')
    destination = robot.find(f"link[@name='{target}_gripper_link']") if target else None
    for item in list(source):
        if item.get('name', '').startswith('sponge'):
            source.remove(item)
            if destination is not None:
                destination.append(item)
    server = world.find("plugin[@name='iss::WaterSystem']")
    if 'droplet' in spec['objects'] and target:
        server.find('sponge_link').text = target + '_gripper_link'
        manifest['sponge']['link'] = target + '_gripper_link'
    else:
        world.remove(server)
        gui = world.find('gui')
        for item in list(gui):
            if item.get('name') == 'WaterGui':
                gui.remove(item)
        manifest['sponge'] = None
        manifest['water_optics'] = None
        manifest['water_depth_model'] = None
    if spec.get('sponge_grasp'):
        if target != 'right' or 'droplet' not in spec['objects']:
            raise ValueError('Sponge-grasp start currently requires the right-arm water task')
        prepare_sponge_grasp(world, robot, manifest, spec)
    if record:
        observer = dict(eye=[.82, 0., .41], target=[.21, 0., .33],
                        horizontal_fov=1.10, resolution=[960, 640],
                        topic='/camera/opposite/image', policy_input=False,
                        purpose='recording only: opposite table edge, facing both arms')
        add_camera(world, 'opposite', observer['eye'], observer['target'],
                   size=tuple(observer['resolution']), fov=observer['horizontal_fov'])
        sensor = world.find("model[@name='capping_camera_opposite']/link/sensor")
        sensor.find('update_rate').text = '10'
        # Preserve transparent RGB water; the idealized depth surface is hidden.
        child(sensor.find('camera'), 'visibility_mask', 0xFFFFFFFF ^ (1 << 24))
        manifest['recording_only_cameras'] = {'opposite': observer}
    ET.indent(root, space='  ')
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
    check = subprocess.run(['gz', 'sdf', '-k', str(path)], capture_output=True, text=True, timeout=60)
    (Path(output) / 'task_scene_validation.log').write_text(check.stdout + check.stderr)
    if check.returncode:
        raise RuntimeError(check.stdout + check.stderr)
    manifest.update(task='task_specific_layout', scene_version=7 if record else 6, task_scene=spec,
                    present_objects=spec['objects'], all_objects_present_for_every_task=False)
    (Path(output) / 'scene.json').write_text(json.dumps(manifest, indent=2))
    return path, manifest


class SceneClient(Client):
    def __init__(self, *args, record=False, **kwargs):
        self.record_opposite = record
        self.opposite_frames = []
        self.opposite_last = -1.
        self.sponge_contacts = {'fixed': False, 'moving': False}
        self.sponge_contact_stamps = {'fixed': -1., 'moving': -1.}
        super().__init__(*args, **kwargs)
        if record:
            self.subscribe(Image, '/camera/opposite/image', self._opposite)

    def start_bridge(self):
        # Called by the base constructors before the simulator starts. This
        # observer never enters self.frames or synchronized_observation().
        if self.record_opposite:
            entries = json.loads(self.config.read_text())
            if not any(row.get('topic_name') == '/camera/opposite/image' for row in entries):
                entries.append(dict(topic_name='/camera/opposite/image',
                    ros_type_name='sensor_msgs/msg/Image', gz_type_name='gz.msgs.Image',
                    direction='GZ_TO_ROS', qos_profile='SENSOR_DATA'))
                self.config.write_text(json.dumps(entries, indent=2))
        super().start_bridge()

    def _opposite(self, message):
        stamp = stamp_seconds(message.header.stamp)
        if stamp - self.opposite_last < .09:
            return
        rgb = rgb_image(message)
        if rgb is not None:
            with self.lock:
                self.opposite_frames.append((stamp, rgb.copy()))
                self.opposite_last = stamp

    def _poses(self, message):
        super()._poses(message)
        # Privileged state is for teacher labels and independent evaluation only.
        for transform in message.transforms:
            if transform.child_frame_id != 'held_sponge':
                continue
            p, q = transform.transform.translation, transform.transform.rotation
            quaternion = np.array([q.x, q.y, q.z, q.w])
            if np.linalg.norm(quaternion) < 1e-9:
                continue
            rotation = Rotation.from_quat(quaternion).as_matrix()
            with self.lock:
                self.objects['held_sponge'] = dict(position=np.array([p.x, p.y, p.z]),
                    quaternion=quaternion, rotation=rotation, axis=rotation[:, 2],
                    stamp=stamp_seconds(transform.header.stamp))

    def _contact(self, side, pad, message):
        super()._contact(side, pad, message)
        if side != 'right':
            return
        hit = any('sponge_collision' in c.collision1.name or 'sponge_collision' in c.collision2.name
                  for c in message.contacts)
        with self.lock:
            self.sponge_contacts[pad] = hit
            self.sponge_contact_stamps[pad] = stamp_seconds(message.header.stamp)

    def sponge_grasped(self):
        with self.lock:
            return all(self.sponge_contacts[p] and -.02 <= self.sim_time-self.sponge_contact_stamps[p] < .15
                       for p in ('fixed', 'moving'))

    def _display(self, message):
        stamp = stamp_seconds(message.header.stamp)
        if stamp - self.display_last < .09:
            return
        rgb = rgb_image(message)
        if rgb is not None:
            with self.lock:
                self.display_frames.append((stamp, rgb.copy()))
                self.display_last = stamp


class SceneSession(SharedSession):
    def __init__(self, output, case, spec, gui=False, record=False):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.world, self.manifest = build_scene(self.output / 'scene', case, spec, record=record)
        self.case, self.spec, self.gui = case, spec, gui
        self.record = record
        self.client = self.process = self.log = None
        self.partition = os.environ.get('GZ_PARTITION')

    def __enter__(self):
        os.environ['GZ_PARTITION'] = 'task_scene_' + uuid.uuid4().hex
        (self.output / 'partition.txt').write_text(os.environ['GZ_PARTITION'])
        try:
            self.client = SceneClient(self.output, self.manifest['joint_control'], True, record=self.record)
            self.log = (self.output / 'gazebo.log').open('w')
            self.process = subprocess.Popen(['gz', 'sim', '-r', '-v', '3'] +
                ([] if self.gui else ['-s']) + [str(self.world)], stdout=self.log,
                stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 10
            while self.client.clock_time is None and time.monotonic() < deadline:
                time.sleep(.05)
            if self.client.clock_time is None:
                stop_process(self.client.bridge)
                self.client.bridge_log.close()
                self.client.bridge = self.client.bridge_log = None
                self.client.start_bridge()
            self.client.activate_controllers()
            deadline = time.monotonic() + 40
            while True:
                q, stamp, _ = self.client.snapshot()
                if q is not None and stamp > .5 and set(self.spec['objects']) <= self.client.object_states().keys():
                    break
                if self.process.poll() is not None:
                    raise RuntimeError('Gazebo exited')
                if time.monotonic() > deadline:
                    raise TimeoutError('Task scene not ready')
                time.sleep(.05)
            self.client.pause()
            self.client.step(100)
            self.client.camera_observation()
            if self.spec.get('sponge_grasp'):
                self.settle_sponge_grasp()
            if 'floating_pen' in self.spec['objects']:
                start_motion(self.client, self.case, self.output)
            if 'droplet' in self.spec['objects']:
                mass = 1000 * 4 * np.pi * DROPLET_RADIUS**3 / 3
                request = EntityWrench()
                request.entity.name, request.entity.type = 'droplet', Entity.MODEL
                request.wrench.force.x, request.wrench.force.y, request.wrench.force.z = map(
                    float, mass * np.asarray(self.case['water_velocity']) / .001)
                self.client.impulse_publisher.publish(request)
                time.sleep(.05)
                self.client.step(1)
                self.client.step(100)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def settle_sponge_grasp(self):
        client = self.client
        q, _, _ = client.snapshot()
        command = q.copy()
        command[11] = -.17
        client.command(command)
        time.sleep(.03)
        samples, stable_start = [], None
        for _ in range(30):
            client.step(100)
            time.sleep(.015)
            q, stamp, _ = client.snapshot()
            grasped = client.sponge_grasped()
            samples.append(dict(time=stamp, jaw=float(q[11]), bilateral=grasped,
                                contacts=dict(client.sponge_contacts)))
            if grasped and stable_start is None:
                stable_start = stamp
            if not grasped:
                stable_start = None
            if stable_start is not None and stamp-stable_start >= .3:
                break
        ready = stable_start is not None and stamp-stable_start >= .3
        audit = dict(ready=ready, method='close right jaw before policy; verify both physical pad contacts for 0.3 seconds',
                     fixed_attachment=False, samples=samples)
        (self.output/'initial_sponge_grasp.json').write_text(json.dumps(audit, indent=2))
        if not ready:
            raise RuntimeError('Right gripper did not establish a stable physical sponge grasp')


class SceneMonitor:
    def __init__(self, task):
        from .capping_monitor import CappingMonitor
        self.task = task
        self.capping = CappingMonitor(require_carry=False)
        self.start = None
        self.success = False
        self.last = {}
        self.samples = []

    def update(self, client):
        q, stamp, _ = client.snapshot()
        if self.task == 'task1':
            grasped = client.sponge_grasped() if hasattr(client, 'sponge_grasped') else True
            self.success = bool(client.water_contact and grasped)
            self.last = dict(time=stamp, right_sponge_contact=bool(client.water_contact),
                             right_holds_sponge=grasped, success=self.success)
        elif self.task == 'task2':
            contact = bool(client.bilateral('left') and -.15 < q[5] < .15)
            if contact and self.start is None:
                self.start = stamp
            if not contact:
                self.start = None
            stable = 0. if self.start is None else stamp - self.start
            self.success = stable >= .6
            self.last = dict(time=stamp, bilateral_contact=contact,
                             stable_seconds=stable, lift_required=False, success=self.success)
        else:
            self.last = self.capping.update(client)
            self.success = self.capping.success
        self.samples.append(dict(self.last))
        return self.last
