"""Native Gazebo Transport client, independent of ROS 2."""
import threading
import time
import numpy as np
from .config import JOINTS, STATE_TOPIC, TOUCH_TOPIC, WORLD


class GazeboClient:
    def __init__(self):
        try:
            from gz.transport13 import Node
            from gz.msgs10.boolean_pb2 import Boolean
            from gz.msgs10.double_pb2 import Double
            from gz.msgs10.model_pb2 import Model
            from gz.msgs10.contacts_pb2 import Contacts
            from gz.msgs10.world_control_pb2 import WorldControl
        except ImportError as exc:
            raise RuntimeError("Gazebo Python連携がありません。python3-gz-transport13 と "
                               "python3-gz-msgs10 を導入し、--system-site-packages付きの "
                               "Python 3.10仮想環境を使ってください。") from exc
        self.Double, self.Boolean, self.WorldControl = Double, Boolean, WorldControl
        self.node = Node()
        self.lock = threading.Lock()
        self.positions = None
        self.sim_time = 0.0
        self.updated_at = 0.0
        self.touched = threading.Event()
        self.touch_count = 0
        self.touch_wall_time = None
        self.touch_sim_time = None
        self.contacts = []
        # Subscribe before launching the simulator: TouchPlugin is a one-shot event.
        for kind, topic, callback in [(Boolean, TOUCH_TOPIC, self._touch),
                                     (Model, STATE_TOPIC, self._state),
                                     (Contacts, "/capture/contacts", self._contacts)]:
            if not self.node.subscribe(kind, topic, callback):
                raise RuntimeError(f"トピック購読に失敗しました: {topic}")
        self.publishers = {name: self.node.advertise(f"/so101/{name}/target", Double)
                           for name in JOINTS}

    def _touch(self, message):
        if message.data:
            with self.lock:
                self.touch_count += 1
                if self.touch_wall_time is None:
                    self.touch_wall_time = time.monotonic()
                    self.touch_sim_time = self.sim_time
            self.touched.set()

    def _state(self, message):
        positions = {j.name: j.axis1.position for j in message.joint}
        if not all(name in positions for name in JOINTS):
            return
        q = np.array([positions[name] for name in JOINTS], dtype=float)
        if not np.isfinite(q).all():
            return
        stamp = message.header.stamp
        with self.lock:
            self.positions = q
            self.sim_time = stamp.sec + stamp.nsec * 1e-9
            self.updated_at = time.monotonic()

    def _contacts(self, message):
        if message.contact:
            with self.lock:
                if len(self.contacts) < 20:
                    self.contacts.append(str(message))

    def snapshot(self):
        with self.lock:
            return (None if self.positions is None else self.positions.copy(),
                    self.sim_time, self.updated_at)

    def command(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("6関節分の有限な指令値が必要です。")
        for name, value in zip(JOINTS, q):
            if not self.publishers[name].publish(self.Double(data=float(value))):
                raise RuntimeError(f"関節指令の送信に失敗しました: {name}")

    def pause(self):
        result, reply = self.node.request(f"/world/{WORLD}/control", self.WorldControl(pause=True),
                                         self.WorldControl, self.Boolean, 3000)
        return bool(result and reply.data)

    def close(self):
        for topic in (TOUCH_TOPIC, STATE_TOPIC, "/capture/contacts"):
            self.node.unsubscribe(topic)
