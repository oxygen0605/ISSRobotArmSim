"""Dynamic water proxy, RGB cameras, and one-shot physical impulse input."""
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from .config import ROOT, SCENARIOS, DROPLET_RADIUS
from .kinematics import Arm
from .world import child, plugin, numbers, make_world, convert_urdf, initial_clearance
from .appearance import APPEARANCES, reflection_map
from .cabin import ENVIRONMENTS, ISS_ENVIRONMENT_VERSION
from .prop_meshes import compile_cabin_meshes
from .water_plugin import install_water_plugins, DEFAULT_WOBBLE_AMPLITUDE, validate_wobble_amplitude
from .viewpoint import viewer_pose
from .floating_pen import pen_metadata
from .trajectory_control import configure_robot_control

MASS = 1000 * (4 / 3) * np.pi * DROPLET_RADIUS ** 3
STEP = 0.001
CAMERAS = {"front": (0.36, -0.64, 0.34, 0, 0.12, np.pi / 2),
           "side": (0.91, 0.015, 0.36, 0, 0.20, np.pi)}


def build_moving(output, scenario="center", target=None, speed=0.01, image_size=256, actuator_speed=.35, appearance="refractive", environment="iss", view="close", wobble_amplitude=DEFAULT_WOBBLE_AMPLITUDE):
    wobble_amplitude = validate_wobble_amplitude(wobble_amplitude)
    if not np.isfinite(speed) or speed < 0 or speed > .1:
        raise ValueError("上向き初速度は0〜0.1 m/sの範囲で指定してください。")
    if appearance not in APPEARANCES:
        raise ValueError("未知の水滴外観です。")
    if environment not in ENVIRONMENTS:
        raise ValueError("未知の周囲環境です。")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    arm = Arm()
    xyz = np.asarray(target if target is not None else arm.forward(SCENARIOS[scenario]), dtype=float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError("水滴位置には有限なXYZを指定してください。")
    if initial_clearance(arm, xyz) < .01:
        raise ValueError("初期姿勢から水滴まで10 mm以上の隙間が必要です。")
    root = make_world(convert_urdf(output), xyz, appearance=appearance, environment=environment, view=view)
    world = root.find("world")
    drop = world.find("model[@name='droplet']")
    drop.find("static").text = "false"
    # SceneBroadcaster's pose/info omits TF frame identifiers. Publish only the
    # target model pose with names and simulation stamps for ROS evaluation.
    plugin(drop, 'gz-sim-pose-publisher-system', 'gz::sim::systems::PosePublisher',
           publish_model_pose='true', publish_link_pose='false',
           use_pose_vector_msg='true', update_frequency=100,
           topic='/evaluation/object_pose')
    body = drop.find("link")
    inertia = child(body, "inertial")
    child(inertia, "mass", MASS)
    tensor = child(inertia, "inertia")
    for tag in ("ixx", "iyy", "izz"):
        child(tensor, tag, .4 * MASS * DROPLET_RADIUS ** 2)
    for tag in ("ixy", "ixz", "iyz"):
        child(tensor, tag, 0)
    plugin(world, "gz-sim-apply-link-wrench-system", "gz::sim::systems::ApplyLinkWrench")
    plugin(world, "gz-sim-sensors-system", "gz::sim::systems::Sensors", render_engine="ogre2")
    for name, pose in CAMERAS.items():
        camera = child(world, "model", name=f"camera_{name}")
        child(camera, "static", "true")
        child(camera, "pose", numbers(pose))
        link = child(camera, "link", name="body")
        sensor = child(link, "sensor", name=name, type="camera")
        child(sensor, "always_on", "true")
        child(sensor, "update_rate", 20)
        child(sensor, "topic", f"/camera/{name}/image")
        spec = child(sensor, "camera")
        child(spec, "horizontal_fov", .9)
        image = child(spec, "image")
        child(image, "width", image_size)
        child(image, "height", image_size)
        child(image, "format", "R8G8B8")
        clip = child(spec, "clip")
        child(clip, "near", .02)
        child(clip, "far", 8)
    optics = install_water_plugins(root, wobble_amplitude) if appearance == 'refractive' else None
    mesh_stats = compile_cabin_meshes(world, output / "cabin_meshes")
    control = configure_robot_control(world, output, np.zeros(6),
                                      velocity_limits=np.array([actuator_speed] * 5 + [1.]))
    ET.indent(root, space="  ")
    path = output / "moving_droplet.sdf"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    check = subprocess.run(["gz", "sdf", "-k", str(path)], capture_output=True, text=True, timeout=60)
    (output / "sdf_validation.log").write_text(check.stdout + check.stderr)
    if check.returncode:
        raise RuntimeError("SDF検証失敗: " + check.stdout + check.stderr)
    manifest = {"task": "moving_droplet", "target_initial_xyz_m": xyz.tolist(),
                "initial_velocity_m_s": [0, 0, speed], "mass_kg": MASS,
                "max_step_size": STEP, "image_size": image_size, "actuator_speed_rad_s": actuator_speed,
                "urdf_sha256": arm.sha256, "physics": "free rigid sphere; one-shot impulse; zero gravity",
                "appearance": appearance, "appearance_version": 1,
                "water_optics": optics,
                "viewer": view, "viewer_pose": list(viewer_pose(xyz, view, environment)),
                "policy_camera_poses": {name: list(pose) for name, pose in CAMERAS.items()},
                "policy_camera_far_clip_m": 8,
                "environment": environment, "environment_version": ISS_ENVIRONMENT_VERSION if environment == "iss" else 1,
                "workstation": "side-rack cantilever" if environment == "iss" else "original lab platform",
                "floating_pen": pen_metadata() if environment == "iss" else None,
                "decorative_meshes": mesh_stats,
                "surroundings": "ISS/Kibo inspired decorative cabin, not an exact replica" if environment == "iss" else "original lab",
                "decorative_props_have_collision": False,
                "joint_control": control,
                "reflection_map_available": bool(reflection_map()) if appearance != "training" else False,
                "success": "sponge/droplet contact, not fluid absorption"}
    (output / "scene.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return path, manifest
