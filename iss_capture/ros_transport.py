"""ROS 2 Jazzy observations, commands and episode setup via ros_gz_bridge.

The learned policies still receive only RGB, joint state and text. World poses
and contacts are kept separate for initialization and success measurement.
"""
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, DurabilityPolicy
from ament_index_python.packages import get_package_prefix
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration, Time
from control_msgs.msg import JointTrajectoryControllerState
from controller_manager_msgs.srv import LoadController, ConfigureController, SwitchController, ListControllers
from tf2_msgs.msg import TFMessage
from rosgraph_msgs.msg import Clock
from ros_gz_interfaces.msg import Contacts, Entity, EntityWrench
from ros_gz_interfaces.srv import ControlWorld, SetEntityPose
from scipy.spatial.transform import Rotation

from .config import JOINTS, STATE_TOPIC, TOUCH_TOPIC, WORLD
from .moving_world import STEP, MASS
from .pen_kinematics import bounded_pen_action
from .trajectory_control import (NAMESPACE, CONTROLLERS, TRAJECTORY_TOPICS,
                                 DESCRIPTION_TOPIC, timed_waypoints)


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def rgb_image(message):
    """Respect row padding and explicitly preserve the training RGB convention."""
    if message.encoding not in ('rgb8', 'bgr8') or not message.width or not message.height:
        return None
    if message.step < message.width * 3:
        return None
    try:
        rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
        rgb = rows[:, :message.width * 3].reshape(message.height, message.width, 3)
        return (rgb if message.encoding == 'rgb8' else rgb[:, :, ::-1]).copy()
    except ValueError:
        return None


def bridge_config(pen=False, extra_image_topics=()):
    def bridge(topic, ros, gz, direction='GZ_TO_ROS', qos='SENSOR_DATA'):
        entry = dict(topic_name=topic, ros_type_name=ros, gz_type_name=gz, direction=direction)
        if qos != 'DEFAULT':
            entry['qos_profile'] = qos
        return entry
    # The world-qualified clock exists even when Gazebo suppresses its /clock
    # alias because another publisher is detected during GUI startup.
    clock = bridge('/clock', 'rosgraph_msgs/msg/Clock', 'gz.msgs.Clock', qos='CLOCK')
    clock.pop('topic_name')
    clock.update(ros_topic_name='/clock', gz_topic_name=f'/world/{WORLD}/clock')
    entries = [clock,
               bridge('/evaluation/object_pose', 'tf2_msgs/msg/TFMessage', 'gz.msgs.Pose_V'),
               bridge('/gui/camera/pose', 'geometry_msgs/msg/Pose', 'gz.msgs.Pose'),
               bridge(f'/world/{WORLD}/wrench', 'ros_gz_interfaces/msg/EntityWrench',
                      'gz.msgs.EntityWrench', 'ROS_TO_GZ', 'DEFAULT')]
    for name in ('front', 'side'):
        entries.append(bridge(f'/camera/{name}/image', 'sensor_msgs/msg/Image', 'gz.msgs.Image'))
    if pen:
        for side in ('fixed', 'moving'):
            entries.append(bridge(f'/pen/{side}_contacts', 'ros_gz_interfaces/msg/Contacts', 'gz.msgs.Contacts'))
        entries.append(bridge('/pen/display/image', 'sensor_msgs/msg/Image', 'gz.msgs.Image'))
    else:
        entries.extend([bridge(TOUCH_TOPIC, 'std_msgs/msg/Bool', 'gz.msgs.Boolean', qos='DEFAULT'),
                        bridge('/capture/contacts', 'ros_gz_interfaces/msg/Contacts', 'gz.msgs.Contacts'),
                        bridge('/capture/water_adhesion', 'geometry_msgs/msg/PoseStamped', 'gz.msgs.Pose', qos='DEFAULT')])
    for topic in extra_image_topics:
        if not topic.startswith('/preview/'):
            raise ValueError('Preview cameras must use the /preview/ namespace')
        entries.append(bridge(topic, 'sensor_msgs/msg/Image', 'gz.msgs.Image'))
    return entries


class ROSMovingClient:
    def __init__(self, output, *, control_spec, pen=False, start_bridge=True, extra_image_topics=()):
        # Do not initialize Gazebo's Python Transport: all application IO is ROS.
        self.output = Path(output)
        self.control_spec, self.pen_mode = control_spec, pen
        self.lock = threading.Lock()
        self.positions = None
        self.max_joint_velocity = np.zeros(6)
        self.max_gripper_effort_command = 0.
        self.sim_time = self.updated_at = 0.
        self.clock_time = None
        self.touched = threading.Event()
        self.touch_count = 0
        self.touch_wall_time = self.touch_sim_time = None
        self.contacts = []
        self.water_adhesion = None
        self.frames = {name: deque(maxlen=8) for name in ('front', 'side')}
        self.truth = self.truth_stamp = self.viewer_pose = None
        self.counts = {}
        self.EntityWrench, self.Entity = EntityWrench, Entity
        self.context = Context()
        rclpy.init(args=[], context=self.context)
        self.node = Node('iss_pen_controller' if pen else 'iss_water_controller', context=self.context,
                         parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.subscriptions = []
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.trajectory_publishers = [self.node.create_publisher(JointTrajectory, topic, 10)
                                     for topic in TRAJECTORY_TOPICS]
        self.trajectory_count = 0
        self.trajectory_states = {}
        self.controller_status = []
        self.last_trajectory = None
        self.description_publisher = self.node.create_publisher(
            String, DESCRIPTION_TOPIC, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.description_publisher.publish(String(data=Path(control_spec['robot_description']).read_text()))
        self.impulse_publisher = self.node.create_publisher(EntityWrench, f'/world/{WORLD}/wrench', 10)
        self.control_client = self.node.create_client(ControlWorld, f'/world/{WORLD}/control')
        self.pose_client = self.node.create_client(SetEntityPose, f'/world/{WORLD}/set_pose')
        self.subscribe(JointState, STATE_TOPIC, self._state)
        self.subscribe(Clock, '/clock', self._clock)
        self.subscribe(TFMessage, '/evaluation/object_pose', self._poses)
        for name in CONTROLLERS[1:]:
            self.subscribe(JointTrajectoryControllerState, f'{NAMESPACE}/{name}/controller_state',
                           lambda msg, n=name: self._trajectory_state(n, msg))
        for name in self.frames:
            self.subscribe(Image, f'/camera/{name}/image', lambda msg, n=name: self._image(n, msg))
        if not pen:
            self.subscribe(Bool, TOUCH_TOPIC, self._touch, qos=10)
            self.subscribe(Contacts, '/capture/contacts', self._contacts)
            self.subscribe(PoseStamped, '/capture/water_adhesion', self._water_adhesion, qos=10)
        self.config = self.output / 'ros_bridge.json'
        self.config.write_text(json.dumps(bridge_config(pen, extra_image_topics), indent=2) + '\n')
        self.bridge = self.bridge_log = None
        if start_bridge:
            self.start_bridge()
        self.thread.start()

    def start_bridge(self):
        if self.bridge is not None:
            return
        self.bridge_log = (self.output / 'ros_bridge.log').open('w')
        executable = Path(get_package_prefix('ros_gz_bridge')) / 'lib/ros_gz_bridge/parameter_bridge'
        self.bridge = subprocess.Popen([
            str(executable),
            f'/world/{WORLD}/control@ros_gz_interfaces/srv/ControlWorld',
            f'/world/{WORLD}/set_pose@ros_gz_interfaces/srv/SetEntityPose',
            '--ros-args', '-r', '__node:=iss_gazebo_bridge', '-p', f'config_file:={self.config}',
            '-p', 'use_sim_time:=true'], stdout=self.bridge_log, stderr=subprocess.STDOUT,
            start_new_session=True)

    def subscribe(self, kind, topic, callback, qos=qos_profile_sensor_data):
        def counted(message):
            self.counts[topic] = self.counts.get(topic, 0) + 1
            callback(message)
        self.subscriptions.append(self.node.create_subscription(kind, topic, counted, qos))

    def _clock(self, message):
        self.clock_time = stamp_seconds(message.clock)

    def _state(self, message):
        positions = dict(zip(message.name, message.position))
        if not all(name in positions for name in JOINTS):
            return
        q = np.array([positions[name] for name in JOINTS])
        if not np.isfinite(q).all():
            return
        with self.lock:
            self.positions = q
            self.sim_time = stamp_seconds(message.header.stamp)
            self.updated_at = time.monotonic()
            if hasattr(self, 'max_joint_velocity') and len(message.velocity) == len(message.name):
                velocities = dict(zip(message.name, message.velocity))
                self.max_joint_velocity = np.maximum(self.max_joint_velocity, np.abs([velocities[n] for n in JOINTS]))

    def _image(self, name, message):
        rgb = rgb_image(message)
        if rgb is not None:
            with self.lock:
                self.frames[name].append((stamp_seconds(message.header.stamp), rgb))

    def _poses(self, message):
        for transform in message.transforms:
            if transform.child_frame_id == 'droplet':
                p = transform.transform.translation
                with self.lock:
                    self.truth = np.array([p.x, p.y, p.z])
                    self.truth_stamp = stamp_seconds(transform.header.stamp)

    def _contacts(self, message):
        if message.contacts:
            with self.lock:
                if len(self.contacts) < 20:
                    self.contacts.append(str(message))

    def _water_adhesion(self, message):
        point = message.pose.position
        value = dict(start_sim_time=stamp_seconds(message.header.stamp),
                     initial_centre_in_sponge_m=[point.x, point.y, point.z])
        if not np.isfinite([value['start_sim_time'], *value['initial_centre_in_sponge_m']]).all():
            return
        with self.lock:
            self.water_adhesion = value

    def adhesion_snapshot(self):
        with self.lock:
            return None if self.water_adhesion is None else dict(self.water_adhesion)

    def monitor_viewer(self):
        self.subscribe(Pose, '/gui/camera/pose', self._viewer)

    def _viewer(self, pose):
        with self.lock:
            self.viewer_pose = dict(position_m=[pose.position.x, pose.position.y, pose.position.z],
                                    quaternion_xyzw=[pose.orientation.x, pose.orientation.y,
                                                     pose.orientation.z, pose.orientation.w])

    def viewer_snapshot(self):
        with self.lock:
            return None if self.viewer_pose is None else {k: v.copy() for k, v in self.viewer_pose.items()}

    def _touch(self, message):
        if message.data:
            with self.lock:
                self.touch_count += 1
                if self.touch_wall_time is None:
                    self.touch_wall_time = time.monotonic()
                    self.touch_sim_time = self.sim_time
            self.touched.set()

    def snapshot(self):
        with self.lock:
            return (None if self.positions is None else self.positions.copy(),
                    self.sim_time, self.updated_at)

    def camera_observation(self):
        """Policy receives RGB and joint positions; never object truth or contacts."""
        with self.lock:
            if self.positions is None or any(not frames for frames in self.frames.values()):
                return None
            front_t, front = self.frames['front'][-1]
            side_t, side = self.frames['side'][-1]
            previous = min(self.frames['front'], key=lambda frame: abs(frame[0] - (front_t - .20)))
            return dict(front=front.copy(), side=side.copy(), previous_front=previous[1].copy(),
                        q=self.positions.copy(), sim_time=self.sim_time,
                        image_time=min(front_t, side_t), previous_image_time=previous[0])

    def ground_truth(self):
        with self.lock:
            return None if self.truth is None else self.truth.copy(), self.truth_stamp

    def _trajectory_state(self, name, message):
        with self.lock:
            if name == 'gripper_controller' and message.output.effort:
                self.max_gripper_effort_command = max(self.max_gripper_effort_command,
                                                      max(abs(v) for v in message.output.effort))
            self.trajectory_states[name] = dict(
                stamp=stamp_seconds(message.header.stamp), joint_names=list(message.joint_names),
                reference=list(message.reference.positions), feedback=list(message.feedback.positions),
                error=list(message.error.positions), output_effort=list(message.output.effort))

    def activate_controllers(self):
        """Lifecycle services run once at setup; no spawner process remains."""
        for name in CONTROLLERS:
            self._controller_request(LoadController, 'load_controller', LoadController.Request(name=name))
            self._controller_request(ConfigureController, 'configure_controller', ConfigureController.Request(name=name))
        request = SwitchController.Request(activate_controllers=list(CONTROLLERS),
                                           strictness=SwitchController.Request.STRICT,
                                           activate_asap=True, timeout=Duration(sec=15))
        self._controller_request(SwitchController, 'switch_controller', request)
        result = self._controller_request(ListControllers, 'list_controllers', ListControllers.Request())
        self.controller_status = [dict(name=c.name, state=c.state, type=c.type,
                                       claimed_interfaces=list(c.claimed_interfaces)) for c in result.controller]
        if any(not any(c['name'] == name and c['state'] == 'active' for c in self.controller_status)
               for name in CONTROLLERS):
            raise RuntimeError('Standard trajectory controllers are not active')
        deadline = time.monotonic() + 15
        while any(p.get_subscription_count() != 1 for p in self.trajectory_publishers):
            if time.monotonic() > deadline:
                raise TimeoutError('Expected one standard controller on each trajectory topic')
            time.sleep(.02)

    def _controller_request(self, kind, name, request):
        client = self.node.create_client(kind, f'{NAMESPACE}/controller_manager/{name}')
        try:
            return self._request(client, request)
        finally:
            self.node.destroy_client(client)

    def command(self, q):
        return self.command_plan(np.asarray(q, dtype=float)[None, :], self.snapshot()[1], .08)

    def command_plan(self, positions, start_time, dt):
        positions = np.asarray(positions, dtype=float)
        if self.pen_mode:
            positions = np.stack([bounded_pen_action(q) for q in positions])
        q, now, _ = self.snapshot()
        if q is None:
            raise RuntimeError('Joint state is required before sending a trajectory')
        points, times, skipped = timed_waypoints(positions, start_time, dt, now, q,
                                                 self.control_spec['velocity_limits'])
        # Both controllers use the same simulation start stamp and waypoint times.
        ns = round(now * 1e9)
        for publisher, indices in zip(self.trajectory_publishers, (slice(0, 5), slice(5, 6))):
            message = JointTrajectory(joint_names=list(JOINTS[indices]))
            message.header.stamp = Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
            for point, seconds in zip(points, times):
                duration = round(seconds * 1e9)
                message.points.append(JointTrajectoryPoint(positions=point[indices].tolist(),
                    time_from_start=Duration(sec=duration // 1_000_000_000, nanosec=duration % 1_000_000_000)))
            publisher.publish(message)
        self.trajectory_count += 1
        self.last_trajectory = dict(stamp=now, skipped_elapsed_steps=skipped,
                                    positions=points.tolist(), times=times.tolist())
        return points[0].copy()

    def _request(self, client, request):
        if not client.wait_for_service(timeout_sec=15):
            raise TimeoutError(f'ROS service unavailable: {client.srv_name}')
        future = client.call_async(request)
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(15):
            future.cancel()
            raise TimeoutError(f'ROS service timeout: {client.srv_name}')
        response = future.result()
        if response is None or not getattr(response, 'success', getattr(response, 'ok', True)):
            raise RuntimeError(f'ROS service rejected request: {client.srv_name}')
        return response

    def set_running(self, running):
        request = ControlWorld.Request()
        request.world_control.pause = not running
        self._request(self.control_client, request)
        if not running:
            # Let already queued state callbacks drain before counting steps.
            previous = self.snapshot()[1]
            for _ in range(20):
                time.sleep(.02)
                current = self.snapshot()[1]
                if current == previous:
                    break
                previous = current

    def initial_impulse(self, speed):
        deadline = time.monotonic() + 10
        while self.impulse_publisher.get_subscription_count() == 0:
            if time.monotonic() > deadline:
                raise TimeoutError('ROS wrench bridge not connected')
            time.sleep(.02)
        request = EntityWrench()
        request.entity.name = 'droplet'
        request.entity.type = Entity.MODEL
        request.wrench.force.z = MASS * speed / STEP
        self.impulse_publisher.publish(request)
        time.sleep(.05)
        self.step(1)

    def pause(self):
        self.set_running(False)
        return True

    def step(self, count, timeout=15):
        start = self.clock_time
        if start is None:
            raise RuntimeError('Simulation clock is not available')
        request = ControlWorld.Request()
        request.world_control.pause = True
        request.world_control.multi_step = int(count)
        self._request(self.control_client, request)
        deadline = time.monotonic() + timeout
        while self.clock_time < start + count * STEP - STEP * .5:
            if time.monotonic() > deadline:
                raise TimeoutError('ROS joint stream did not complete simulation step')
            time.sleep(.005)

    def reset_pen(self, target):
        request = SetEntityPose.Request()
        request.entity.name = 'floating_pen'
        request.entity.type = Entity.MODEL
        request.pose.position.x, request.pose.position.y, request.pose.position.z = map(float, target['position'])
        q = request.pose.orientation
        q.x, q.y, q.z, q.w = map(float, target['quaternion'])
        self._request(self.pose_client, request)

    def close(self):
        # Evidence contains observations of ROS delivery, not merely launch flags.
        (self.output / 'ros_transport.json').write_text(json.dumps({
            'transport': 'ROS 2 Jazzy / ros_gz_bridge',
            'ros_domain_id': os.environ.get('ROS_DOMAIN_ID', '0'),
            'use_sim_time': True, 'clock_time': self.clock_time,
            'joint_state_time': self.sim_time, 'received_messages': self.counts,
            'bridge_exit_before_cleanup': self.bridge.poll() if self.bridge else 'not_started',
            'command_topics': list(TRAJECTORY_TOPICS),
            'joint_state_source': 'ros2_control joint_state_broadcaster',
            'policy_input': ['front_rgb', 'side_rgb', 'previous_front_rgb', 'six_joint_positions', 'text'],
        }, indent=2) + '\n')
        (self.output / 'trajectory_control.json').write_text(json.dumps({
            'backend': 'ros2_control / gz_ros2_control',
            'controllers': self.controller_status, 'application_servo_thread': False,
            'sent_trajectories_per_controller': self.trajectory_count,
            'last_trajectory': self.last_trajectory, 'controller_states': self.trajectory_states,
            'config': self.control_spec,
            'max_observed_joint_velocity_rad_s': self.max_joint_velocity.tolist(),
            'max_gripper_effort_command_nm': self.max_gripper_effort_command,
        }, indent=2) + '\n')
        self.executor.shutdown(timeout_sec=2)
        self.thread.join(timeout=2)
        self.node.destroy_node()
        self.context.try_shutdown()
        from run_demo import stop_process
        stop_process(self.bridge)
        if self.bridge_log:
            self.bridge_log.close()


class ROSPenClient(ROSMovingClient):
    """Pen-specific observations only; trajectory execution is shared with water."""
    def __init__(self, output, *, control_spec, start_bridge=True):
        self.pen = None
        self.pad_contact = dict(fixed=False, moving=False)
        self.pad_stamp = dict(fixed=-1., moving=-1.)
        self.contact_log = []
        self.display_frames = []
        self.record_display = False
        super().__init__(output, control_spec=control_spec, pen=True, start_bridge=start_bridge)
        for side in ('fixed', 'moving'):
            self.subscribe(Contacts, f'/pen/{side}_contacts', lambda msg, s=side: self._pad_contacts(s, msg))

    def pen_state(self):
        with self.lock:
            return None if self.pen is None else {k: v.copy() if isinstance(v, np.ndarray) else v
                                                  for k, v in self.pen.items()}

    def bilateral_contact(self):
        with self.lock:
            return all(self.pad_contact[s] and self.sim_time-self.pad_stamp[s] < .12 for s in self.pad_contact)

    def _poses(self, message):
        for transform in message.transforms:
            if transform.child_frame_id == 'floating_pen':
                p, q = transform.transform.translation, transform.transform.rotation
                xyz = np.array([p.x, p.y, p.z])
                quat = np.array([q.x, q.y, q.z, q.w])
                if np.linalg.norm(quat) < 1e-9:
                    continue
                stamp = stamp_seconds(transform.header.stamp)
                with self.lock:
                    self.truth, self.truth_stamp = xyz, stamp
                    self.pen = dict(position=xyz, quaternion=quat,
                                    axis=Rotation.from_quat(quat).as_matrix()[:, 2], stamp=stamp)

    def _pad_contacts(self, side, message):
        hit = any('pen_collision' in c.collision1.name or 'pen_collision' in c.collision2.name
                  for c in message.contacts)
        stamp = stamp_seconds(message.header.stamp)
        with self.lock:
            self.pad_contact[side], self.pad_stamp[side] = hit, stamp
            if hit and len(self.contact_log) < 80:
                self.contact_log.append(dict(side=side, sim_time=stamp, contact=str(message)))

    def start_display_recording(self):
        self.record_display = True
        self.subscribe(Image, '/pen/display/image', self._display_image)

    def _display_image(self, message):
        if not self.record_display:
            return
        rgb = rgb_image(message)
        if rgb is not None:
            with self.lock:
                if len(self.display_frames) < 400:
                    self.display_frames.append((stamp_seconds(message.header.stamp), rgb))
