"""Shared geometry and the deliberately restricted training workspace."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "a758567c3978dfeefe282ede0500085a48fe8f78"
UPSTREAM_URL = "https://github.com/TheRobotStudio/SO-ARM100"
URDF = ROOT / "third_party/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
POLICY = ROOT / "artifacts/ik_policy.npz"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ACTIVE = JOINTS[:3]
# Restrict the inverse to one elbow branch. The wrist and gripper stay at zero.
LOWER = (-0.65, -0.30, -0.35)
UPPER = (0.65, 0.30, 0.35)
SPONGE_CENTER = (-0.0079, -0.000218, -0.0981)
SPONGE_SIZE = (0.03, 0.05, 0.05)
DROPLET_RADIUS = 0.012
WORLD = "microgravity"
TOUCH_TOPIC = "/capture/touched"
STATE_TOPIC = "/so101/joint_states"
# Positions are calculated with FK from these task-generation poses. The
# controller receives XYZ only; it never receives these generating angles.
SCENARIOS = {
    "left": (0.40, -0.12, 0.18),
    "center": (0.30, 0.0, 0.0),
    "right": (-0.40, 0.14, -0.18),
}
