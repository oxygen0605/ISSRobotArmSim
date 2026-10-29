"""NumPy inference from safe NPZ weights (no pickle or remote inference)."""
import json
from pathlib import Path
import numpy as np

from .config import POLICY, SPONGE_CENTER, ACTIVE


class Policy:
    def __init__(self, path: Path = POLICY):
        if not Path(path).is_file():
            raise FileNotFoundError("学習済みモデルがありません。python train_policy.py を実行してください。")
        with np.load(path, allow_pickle=False) as data:
            self.metadata = json.loads(str(data["metadata"]))
            self.mean, self.scale = data["mean"], data["scale"]
            self.center, self.half_range = data["center"], data["half_range"]
            self.weights = [(data[f"w{i}"], data[f"b{i}"]) for i in range(3)]

    def predict(self, xyz):
        xyz = np.asarray(xyz, dtype=float)
        if xyz.shape[-1:] != (3,) or not np.isfinite(xyz).all():
            raise ValueError("水滴位置は有限なXYZ座標で指定してください。")
        value = (xyz - self.mean) / self.scale
        for i, (w, b) in enumerate(self.weights):
            value = value @ w + b
            if i < len(self.weights) - 1:
                value = np.tanh(value)
        return self.center + self.half_range * value

    def plan(self, arm, xyz, refine=True):
        if arm.sha256 != self.metadata["urdf_sha256"]:
            raise ValueError("学習時とURDFが異なります。モデルを再学習してください。")
        if (self.metadata.get("sponge_center_m") != list(SPONGE_CENTER)
                or self.metadata.get("active_joints") != list(ACTIVE)
                or self.metadata["joint_lower_rad"] != arm.lower.tolist()
                or self.metadata["joint_upper_rad"] != arm.upper.tolist()):
            raise ValueError("学習時と手先・関節設定が異なります。モデルを再学習してください。")
        xyz = np.asarray(xyz, dtype=float)
        if xyz.shape != (3,):
            raise ValueError("計画する水滴位置は1点のXYZ座標で指定してください。")
        raw = self.predict(xyz)
        # Reject extrapolation instead of silently commanding a different task.
        if np.any(raw < arm.lower - 0.04) or np.any(raw > arm.upper + 0.04):
            raise ValueError("水滴が学習した作業範囲の外です。")
        q = np.clip(raw, arm.lower, arm.upper)
        neural_error = float(np.linalg.norm(arm.forward(q) - xyz))
        if neural_error > 0.015:
            raise ValueError("AI推定誤差が15 mmを超えるため実行を中止します。")
        if refine:
            q = arm.refine(xyz, q)
        error = float(np.linalg.norm(arm.forward(q) - xyz))
        if error > 0.003:
            raise ValueError(f"目標位置に到達できません（計算誤差 {error * 1000:.2f} mm）。")
        return {"target_xyz_m": xyz.tolist(), "neural_q_rad": raw.tolist(),
                "command_q_rad": q.tolist(), "neural_error_mm": neural_error * 1000,
                "planned_error_mm": error * 1000, "refinement": bool(refine)}
