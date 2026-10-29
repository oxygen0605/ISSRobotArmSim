"""URDF forward kinematics, with a batched implementation for dataset creation."""
from __future__ import annotations

import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

from .config import ACTIVE, LOWER, UPPER, SPONGE_CENTER, URDF


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    a = np.asarray(angle)[..., None, None]
    return np.cos(a) * np.eye(3) + (1 - np.cos(a)) * np.outer(axis, axis) + np.sin(a) * cross


def origin(element):
    pose = np.eye(4)
    if element is not None:
        pose[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        r, p, y = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        pose[:3, :3] = rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)
    return pose


class Arm:
    def __init__(self, path: Path = URDF):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError("公式URDFがありません。python fetch_model.py を実行してください。")
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.root = ET.parse(self.path).getroot()
        self.joints = {j.get("name"): j for j in self.root.findall("joint")}
        by_child = {j.find("child").get("link"): j for j in self.joints.values()}
        self.chain = []
        link = "gripper_link"
        while link != "base_link":
            joint = by_child[link]
            self.chain.insert(0, joint)
            link = joint.find("parent").get("link")
        self.transforms = [origin(j.find("origin")) for j in self.chain]
        self.lower, self.upper = np.array(LOWER), np.array(UPPER)
        for i, name in enumerate(ACTIVE):
            limit = self.joints[name].find("limit")
            if self.lower[i] < float(limit.get("lower")) or self.upper[i] > float(limit.get("upper")):
                raise ValueError(f"学習範囲がURDF関節制限を超えています: {name}")

    def transform(self, q):
        """Return base-to-gripper transforms; q has shape (..., 3)."""
        q = np.asarray(q, dtype=float)
        if q.shape[-1:] != (3,) or not np.isfinite(q).all():
            raise ValueError("関節角度は有限な3要素で指定してください。")
        t = np.broadcast_to(np.eye(4), q.shape[:-1] + (4, 4)).copy()
        for j, fixed in zip(self.chain, self.transforms):
            t = t @ fixed
            name = j.get("name")
            angle = q[..., ACTIVE.index(name)] if name in ACTIVE else np.zeros(q.shape[:-1])
            motion = np.broadcast_to(np.eye(4), t.shape).copy()
            if j.get("type") != "fixed":
                if j.get("type") not in ("revolute", "continuous"):
                    raise ValueError(f"Unsupported joint type: {j.get('type')}")
                axis = np.fromstring(j.find("axis").get("xyz"), sep=" ")
                motion[..., :3, :3] = rotation(axis, angle)
            t = t @ motion
        return t

    def forward(self, q):
        t = self.transform(q)
        return (t[..., :3, :3] @ np.array(SPONGE_CENTER)) + t[..., :3, 3]

    def refine(self, target, q, steps=12, tolerance=0.0002):
        """Bounded damped least-squares correction of the neural prediction."""
        target = np.asarray(target, dtype=float)
        q = np.clip(np.asarray(q, dtype=float), self.lower, self.upper)
        for _ in range(steps):
            error = target - self.forward(q)
            if np.linalg.norm(error) < tolerance:
                break
            h = 1e-5
            jac = np.column_stack([(self.forward(q + np.eye(3)[i] * h)
                                    - self.forward(q - np.eye(3)[i] * h)) / (2 * h)
                                   for i in range(3)])
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 1e-6 * np.eye(3), error)
            q = np.clip(q + np.clip(delta, -0.08, 0.08), self.lower, self.upper)
        return q
