"""Standard ros2_control scene setup and VLA-to-trajectory conversion.

No servo loop lives here: JointTrajectoryController interpolates and executes
the trajectory in Gazebo; controller_manager enforces hardware command limits.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from .config import ROOT, JOINTS

NAMESPACE = '/so101'
CONTROLLERS = ('joint_state_broadcaster', 'arm_controller', 'gripper_controller')
TRAJECTORY_TOPICS = tuple(f'{NAMESPACE}/{name}/joint_trajectory' for name in CONTROLLERS[1:])
DESCRIPTION_TOPIC = f'{NAMESPACE}/robot_description'
VELOCITY_LIMITS = np.array([.65, .65, .65, 1.3, 1.2, 1.0])
GRIPPER_EFFORT_LIMIT = .01


def configure_robot_control(world, output, initial_positions, velocity_limits=VELOCITY_LIMITS):
    """Replace native joint controllers with the upstream Gazebo/ROS plugin."""
    output = Path(output).resolve()
    robot = world.find("model[@name='so101']")
    for item in list(robot.findall('plugin')):
        if item.get('name') in ('gz::sim::systems::JointPositionController',
                               'gz::sim::systems::JointController',
                               'gz::sim::systems::JointStatePublisher',
                               'gz_ros2_control::GazeboSimROS2ControlPlugin'):
            robot.remove(item)
    tree = ET.parse(output / 'so101_resolved.urdf')
    urdf = tree.getroot()
    urdf.append(ET.Comment('Project change: ros2_control interfaces, initial state and command limits.'))
    control = ET.SubElement(urdf, 'ros2_control', name='GazeboSimSystem', type='system')
    ET.SubElement(ET.SubElement(control, 'hardware'), 'plugin').text = 'gz_ros2_control/GazeboSimSystem'
    config = yaml.safe_load((ROOT / 'config/ros2_controllers.yaml').read_text())
    for name, initial, speed in zip(JOINTS, initial_positions, velocity_limits):
        original = urdf.find(f"joint[@name='{name}']/limit")
        original.set('velocity', str(float(speed)))
        if name == 'gripper':
            original.set('effort', str(GRIPPER_EFFORT_LIMIT))
        joint = ET.SubElement(control, 'joint', name=name)
        kind = 'effort' if name == 'gripper' else 'position'
        command = ET.SubElement(joint, 'command_interface', name=kind)
        lo = -GRIPPER_EFFORT_LIMIT if kind == 'effort' else float(original.get('lower'))
        hi = GRIPPER_EFFORT_LIMIT if kind == 'effort' else float(original.get('upper'))
        for key, value in [('min', lo), ('max', hi)]:
            ET.SubElement(command, 'param', name=key).text = str(value)
        position = ET.SubElement(joint, 'state_interface', name='position')
        ET.SubElement(position, 'param', name='initial_value').text = str(float(initial))
        ET.SubElement(joint, 'state_interface', name='velocity')
        ET.SubElement(joint, 'state_interface', name='effort')
    description = output / 'so101_control.urdf'
    tree.write(description, encoding='utf-8', xml_declaration=True)
    parameters = output / 'ros2_controllers.yaml'
    parameters.write_text(yaml.safe_dump(config, sort_keys=False))
    plugin = ET.SubElement(robot, 'plugin', filename='gz_ros2_control-system',
                           name='gz_ros2_control::GazeboSimROS2ControlPlugin')
    ET.SubElement(plugin, 'parameters').text = str(parameters)
    # 0.008 * 1000 Hz = 8 / s. The gripper PID runs at the physics rate.
    ET.SubElement(plugin, 'position_proportional_gain').text = '.008'
    ros = ET.SubElement(plugin, 'ros')
    ET.SubElement(ros, 'namespace').text = NAMESPACE
    ET.SubElement(ros, 'remapping').text = f'robot_description:={DESCRIPTION_TOPIC}'
    ET.SubElement(ros, 'remapping').text = f'/joint_states:={NAMESPACE}/joint_states'
    return dict(backend='ros2_control / gz_ros2_control', robot_description=str(description),
                parameters=str(parameters), controllers=list(CONTROLLERS),
                trajectory_topics=list(TRAJECTORY_TOPICS), update_rate_hz=1000,
                velocity_limits=np.asarray(velocity_limits).tolist(),
                gripper_effort_limit_nm=GRIPPER_EFFORT_LIMIT)


def timed_waypoints(positions, start_time, dt, now, current, velocity_limits=VELOCITY_LIMITS):
    """Discard elapsed chunk steps and stretch segments to respect speed limits.

    This is a message adapter, not a feedback controller. Feedback, interpolation
    and command limiting are performed by the standard controllers.
    """
    positions = np.asarray(positions, dtype=float)
    current = np.asarray(current, dtype=float)
    speeds = np.asarray(velocity_limits, dtype=float)
    if (positions.ndim != 2 or positions.shape[1] != 6 or not 1 <= len(positions) <= 32
            or current.shape != (6,) or speeds.shape != (6,)
            or not np.isfinite(positions).all() or not np.isfinite(current).all()
            or not np.isfinite(speeds).all() or np.any(speeds <= 0)
            or not np.isfinite([start_time, dt, now]).all() or dt <= 0):
        raise ValueError('Invalid trajectory timing or joint values')
    if start_time > now + .05:
        raise ValueError('Observation time is in the future')
    first = min(len(positions) - 1, max(0, int((now - start_time + 1e-9) / dt)))
    remaining = positions[first:].copy()
    elapsed = .02  # Both controllers receive the same short delivery allowance.
    times = []
    previous = current
    for point in remaining:
        elapsed += max(dt, float(np.max(np.abs(point - previous) / speeds)))
        times.append(elapsed)
        previous = point
    return remaining, np.asarray(times), first
