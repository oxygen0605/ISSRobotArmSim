"""Privileged demonstrations for right-sponge / left-pen / dual-cap tasks."""
import numpy as np
from .unified_teacher import Teacher
from .scene_pose_teacher import ScenePoseTeacher
from .capping_pose import PoseCodec, OPEN, CLOSED


class SceneTeacher:
    def __init__(self, task):
        self.task = task
        self.shared = Teacher(task)
        self.shared.teacher = ScenePoseTeacher()
        self.teacher = self.shared.teacher
        self.version = 3
        self.codec = PoseCodec()
        self.home = None
        self.last_water = None
        self.hold = None
        self.phase = 'approach' if task == 'task1' else 'grasp'

    def action(self, client):
        q, stamp, _ = client.snapshot()
        tcp, axes = self.codec.positions(q), self.codec.axes(q)
        if self.home is None:
            self.home = tcp.copy(), axes.copy()
        if self.task == 'task1':
            objects = client.object_states()
            water, sponge = objects['droplet'], objects['held_sponge']
            velocity = np.zeros(3)
            if self.last_water is not None:
                dt = water['stamp']-self.last_water['stamp']
                if dt > .001:
                    velocity = (water['position']-self.last_water['position'])/dt
            self.last_water = water
            # Touch the exposed upper face, not the centre enclosed by the jaws.
            # Keep the sphere above both pad tops while approaching the sponge.
            contact_offset = sponge['rotation'][:, 1] * (.030/2 + .012 - .0015)
            goal = tcp[1] + water['position'] - sponge['position'] - contact_offset + np.clip(velocity, -.02, .02)*.15
            if client.water_contact:
                if self.hold is None:
                    self.hold = tcp[1].copy()
                goal = self.hold
                self.phase = 'hold'
            return self.codec.pack(self.home[0][0], goal, self.home[1], [OPEN, CLOSED])
        if self.task == 'task3':
            goal = self.shared.action(client)
            self.phase = self.shared.phase
            return goal
        # Reuse exactly the physical left-pen grasp teacher; the right arm is idle.
        # A virtual, unused cap exists only inside this demonstration-generator view.
        teacher = self.teacher
        teacher.grasp['right'] = 'hold'
        teacher.hold['right'] = self.home[0][1].copy()
        if teacher.grasp['left'] == 'hold':
            self.phase = 'carry'  # Existing stage ID 3 now means pen hold, no lift.
            return self.codec.pack(teacher.hold['left'], self.home[0][1],
                                   [axes[0], self.home[1][1]], [CLOSED, OPEN])
        class View:
            def __getattr__(self, name): return getattr(client, name)
            def object_states(self):
                objects = client.object_states()
                objects['floating_cap'] = dict(position=tcp[1], axis=axes[1], rotation=np.eye(3), stamp=stamp)
                return objects
        goal = teacher.action(View())
        goal[3:6] = self.home[0][1]-goal[:3]
        goal[8:10] = self.home[1][1,:2]
        goal[11] = OPEN
        self.phase = 'grasp'
        return goal
