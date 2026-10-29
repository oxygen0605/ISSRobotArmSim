"""Build the real Gazebo world through Gazebo's official URDF converter."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from .config import (URDF, ROOT, JOINTS, SPONGE_CENTER, SPONGE_SIZE,
                     DROPLET_RADIUS, WORLD, STATE_TOPIC, SCENARIOS, UPSTREAM_COMMIT)
from .kinematics import Arm
from .appearance import apply_droplet_appearance
from .cabin import ENVIRONMENTS, ISS_ENVIRONMENT_VERSION, add_iss_cabin
from .prop_meshes import compile_cabin_meshes
from .water_plugin import install_water_plugins
from .viewpoint import viewer_pose
from .floating_pen import add_floating_pen, pen_metadata


def child(parent, tag, text=None, **attrs):
    node = ET.SubElement(parent, tag, attrs)
    if text is not None:
        node.text = str(text)
    return node


def numbers(values):
    return " ".join(f"{v:.10g}" for v in values)


def plugin(parent, filename, name, **params):
    node = child(parent, "plugin", filename=filename, name=name)
    for tag, value in params.items():
        child(node, tag, value)
    return node


def material(visual, rgba):
    mat = child(visual, "material")
    child(mat, "ambient", rgba)
    child(mat, "diffuse", rgba)


def box(parent, name, size, pose, rgba, collision=False):
    visual = child(parent, "visual", name=name + "_visual")
    child(visual, "pose", numbers(pose))
    child(child(child(visual, "geometry"), "box"), "size", numbers(size))
    material(visual, rgba)
    if collision:
        coll = child(parent, "collision", name=name + "_collision")
        child(coll, "pose", numbers(pose))
        child(child(child(coll, "geometry"), "box"), "size", numbers(size))


def initial_clearance(arm, target):
    """Exact signed sphere/box separation at the initial (zero) arm pose."""
    transform = arm.transform(np.zeros(3))
    local = transform[:3, :3].T @ (np.asarray(target) - transform[:3, 3])
    outside = np.maximum(np.abs(local - SPONGE_CENTER) - np.array(SPONGE_SIZE) / 2, 0)
    return float(np.linalg.norm(outside) - DROPLET_RADIUS)


def convert_urdf(output):
    if shutil.which("gz") is None:
        raise RuntimeError("Gazeboがありません。bash scripts/install_gazebo.sh を実行してください。")
    urdf = ET.parse(URDF)
    for mesh in urdf.findall(".//mesh"):
        name = mesh.get("filename", "")
        path = (URDF.parent / name).resolve()
        if not name.startswith("assets/") or not path.is_file():
            raise ValueError(f"Mesh is missing or unsupported: {name}")
        mesh.set("filename", path.as_uri())
    urdf.getroot().insert(0, ET.Comment(
        f"SO-101 source: TheRobotStudio/SO-ARM100 @ {UPSTREAM_COMMIT}; Apache-2.0. "
        "Modified by this project: relative mesh paths resolved to local file URIs. "
        "Retain THIRD_PARTY_NOTICES.md and licenses/SO-ARM100-LICENSE.txt when distributing."))
    patched = output / "so101_resolved.urdf"
    urdf.write(patched, encoding="utf-8", xml_declaration=True)
    proc = subprocess.run(["gz", "sdf", "-p", str(patched)], capture_output=True,
                          text=True, timeout=60)
    (output / "urdf_conversion.log").write_text(proc.stderr)
    if proc.returncode:
        raise RuntimeError(f"URDF変換に失敗しました: {proc.stderr}")
    try:
        model = ET.fromstring(proc.stdout).find("model")
    except ET.ParseError as exc:
        raise RuntimeError("gz sdf の出力がSDFではありません。変換ログを確認してください。") from exc
    if model is None:
        raise RuntimeError("URDF変換結果にmodelがありません。")
    # URDF/SDF converter dependencies resolve repeated material references differently.
    # Preserve the rendered colors seen by both frozen adapters during training.
    palette = json.loads((ROOT / 'assets/so101_training_materials.json').read_text())
    if hashlib.sha256(URDF.read_bytes()).hexdigest() != palette['urdf_sha256']:
        raise ValueError('Robot appearance profile does not match the URDF')
    actual = {link.get('name') + '/' + visual.get('name'): visual
              for link in model.findall('link') for visual in link.findall('visual')}
    if set(actual) != set(palette['materials']):
        raise ValueError('Converted robot visual names differ from the training model')
    for key, colors in palette['materials'].items():
        mat = actual[key].find('material')
        if mat is None:
            mat = child(actual[key], 'material')
        for tag, color in colors.items():
            field = mat.find(tag)
            if field is None:
                field = child(mat, tag)
            field.text = color
    return model


def make_world(converted_model, target, appearance="training", environment="lab", view="close"):
    if environment not in ENVIRONMENTS:
        raise ValueError("未知の周囲環境です。")
    camera_pose = viewer_pose(target, view, environment)
    gui_hfov = 80 if view == 'wide' and environment == 'iss' else 60
    model = copy.deepcopy(converted_model)
    model.set("name", "so101")
    links = {x.get("name"): x for x in model.findall("link")}
    joints = {x.get("name") for x in model.findall("joint")}
    if not {"base_link", "gripper_link"} <= links.keys() or not set(JOINTS) <= joints:
        raise ValueError("公式モデルに必要なリンクまたは関節がありません。")
    gripper = links["gripper_link"]
    box(gripper, "sponge", SPONGE_SIZE, (*SPONGE_CENTER, 0, 0, 0), "0.45 0.95 0.58 1", True)
    sensor = child(gripper, "sensor", name="sponge_touch", type="contact")
    child(sensor, "always_on", "true")
    child(sensor, "update_rate", "1000")
    contact = child(sensor, "contact")
    child(contact, "collision", "sponge_collision")
    # Gazebo Sim 8's Contact system reads the topic from <contact>, not <sensor>.
    child(contact, "topic", "/capture/contacts")
    fixed = child(model, "joint", name="base_world_fixed", type="fixed")
    child(fixed, "parent", "world")
    child(fixed, "child", "base_link")
    for name in JOINTS:
        # Ideal velocity-limited actuators are a deliberate MVP simplification.
        plugin(model, "gz-sim-joint-position-controller-system",
               "gz::sim::systems::JointPositionController", joint_name=name,
               topic=f"/so101/{name}/target", initial_position=0,
               use_velocity_commands="true", cmd_max=0.35, cmd_min=-0.35)
    plugin(model, "gz-sim-joint-state-publisher-system",
           "gz::sim::systems::JointStatePublisher", topic=STATE_TOPIC)
    plugin(model, "gz-sim-touchplugin-system", "gz::sim::systems::TouchPlugin",
           target="droplet", namespace="capture", time=0.001, enabled="true")

    root = ET.Element("sdf", version="1.9")
    root.append(ET.Comment(
        f"Contains SO-101 derived from TheRobotStudio/SO-ARM100 @ {UPSTREAM_COMMIT}; Apache-2.0. "
        "Project changes: URDF-to-SDF conversion, material adjustments, task geometry, "
        "sensors and controllers. Retain THIRD_PARTY_NOTICES.md and "
        "licenses/SO-ARM100-LICENSE.txt when distributing."))
    world = child(root, "world", name=WORLD)
    physics = child(world, "physics", name="fixed_step", type="ignored")
    child(physics, "max_step_size", "0.001")
    child(physics, "real_time_factor", "1")
    child(world, "gravity", "0 0 0")
    scene = child(world, "scene")
    child(scene, "ambient", "0.65 0.65 0.65 1")
    child(scene, "background", "0.025 0.045 0.085 1")
    child(scene, "shadows", "true")
    for filename, cls in [("physics", "Physics"), ("user-commands", "UserCommands"),
                          ("scene-broadcaster", "SceneBroadcaster"), ("contact", "Contact")]:
        plugin(world, f"gz-sim-{filename}-system", f"gz::sim::systems::{cls}")
    light = child(world, "light", name="sun", type="directional")
    child(light, "pose", "0 0 3 0 0 0")
    child(light, "diffuse", "0.9 0.95 1 1")
    child(light, "specular", "0.2 0.2 0.2 1")
    child(light, "direction", "-0.4 0.3 -1")
    world.append(model)
    droplet = child(world, "model", name="droplet")
    child(droplet, "static", "true")
    child(droplet, "pose", numbers([*target, 0, 0, 0]))
    body = child(droplet, "link", name="body")
    for kind, name in [("collision", "water_collision"), ("visual", "water_visual")]:
        node = child(body, kind, name=name)
        child(child(child(node, "geometry"), "sphere"), "radius", DROPLET_RADIUS)
        if kind == "visual":
            material(node, "0.05 0.65 1 1")
    apply_droplet_appearance(world, appearance)
    # The display platform has no collision; the base is fixed to world.
    platform = child(world, "model", name="station_panel")
    child(platform, "static", "true")
    panel = child(platform, "link", name="panel")
    box(panel, "deck", (0.65, 0.75, 0.025), (0.23, 0, -0.025, 0, 0, 0), "0.12 0.16 0.23 1")
    for i in range(8):
        box(panel, f"grid_y{i}", (0.64, 0.001, 0.001),
            (0.23, -0.35 + i * 0.1, -0.011, 0, 0, 0), "0.28 0.38 0.50 1")
    for i in range(7):
        box(panel, f"grid_x{i}", (0.001, 0.74, 0.001),
            (-0.07 + i * 0.1, 0, -0.011, 0, 0, 0), "0.28 0.38 0.50 1")
    if environment == "iss":
        add_iss_cabin(world)
        add_floating_pen(world)
    gui = child(world, "gui", fullscreen="false")
    view = plugin(gui, "MinimalScene", "3D View")
    child(view, "engine", "ogre2")
    child(view, "scene", "scene")
    child(view, "ambient_light", "0.65 0.65 0.65")
    child(view, "background_color", "0.025 0.045 0.085")
    child(view, "camera_pose", numbers(camera_pose))
    child(view, "horizontal_fov", gui_hfov)  # MinimalScene expects degrees.
    props = child(view, "gz-gui")
    child(props, "title", "ISS / SO-ARM101 / Water droplet capture")
    child(props, "property", "false", type="bool", key="showTitleBar")
    child(props, "property", "docked", type="string", key="state")
    for filename in ("GzSceneManager", "InteractiveViewControl", "CameraTracking"):
        plugin(gui, filename, filename)
    return root


def build(output=ROOT / "build", scenario="center", target=None, appearance="training", environment="lab", view="close"):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    arm = Arm()
    scenario_name = scenario if target is None else "custom"
    target = np.asarray(target if target is not None else arm.forward(SCENARIOS[scenario]), dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("target は有限な3要素の座標で指定してください。")
    clearance = initial_clearance(arm, target)
    if clearance < 0.01:
        raise ValueError("初期姿勢と水滴が近すぎます。10 mm以上の初期隙間が必要です。")
    root = make_world(convert_urdf(output), target, appearance=appearance, environment=environment, view=view)
    if appearance == 'refractive':
        install_water_plugins(root)
    compile_cabin_meshes(root.find("world"), output / "cabin_meshes")
    ET.indent(root, space="  ")
    path = output / "microgravity.sdf"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    check = subprocess.run(["gz", "sdf", "-k", str(path)], capture_output=True, text=True, timeout=60)
    (output / "sdf_validation.log").write_text(check.stdout + check.stderr)
    if check.returncode:
        raise RuntimeError(f"SDF検証に失敗しました: {check.stdout}\n{check.stderr}")
    manifest = {"world": WORLD, "scenario": scenario_name,
                "target_xyz_m": target.tolist(), "initial_clearance_m": clearance,
                "droplet_radius_m": DROPLET_RADIUS, "gravity": [0, 0, 0],
                "upstream_commit": UPSTREAM_COMMIT, "urdf_sha256": arm.sha256,
                "success_source": "/capture/touched", "sdf_schema_checked": True, "appearance": appearance, "environment": environment,
                "environment_version": ISS_ENVIRONMENT_VERSION if environment == "iss" else 1,
                "floating_pen": pen_metadata() if environment == "iss" else None,
                "viewer": view, "viewer_pose": list(viewer_pose(target, view, environment))}
    (output / "scene.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return path, manifest
