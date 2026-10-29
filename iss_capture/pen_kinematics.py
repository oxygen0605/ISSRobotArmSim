"""Five-axis arm and real hinged-jaw kinematics for the separate pen task."""
import numpy as np
from scipy.optimize import least_squares
from .config import JOINTS
from .kinematics import Arm, origin, rotation

HOME = np.array([0., -.55, .95, -.40, 0., .55])
LOWER = np.array([-.85, -1.2, -1.4, -1.4, -2.5, -.174533])
UPPER = np.array([.85, 1.2, 1.5, 1.4, 2.5, .55])
STATE_SCALE = np.array([1., 1.3, 1.5, 1.5, 2.5, 1.], dtype=np.float32)
ACTION_SCALE = np.array([.10, .10, .10, .10, .12, .10], dtype=np.float32)
FIXED_PAD = np.array([-.0089, -.0002, -.092])
FIXED_SIZE = (.004, .026, .030)
MOVING_PAD = np.array([-.010, -.071, .0188])
MOVING_SIZE = (.004, .030, .026)
SCENARIOS = {'left': [.22, -.15, .35, -.20, -.10, .55],
             'center': [0., -.15, .35, -.20, 0., .55],
             'right': [-.22, -.15, .35, -.20, .12, .55]}
TASK = 'Grasp the rotating orange pen with the gripper, then lift it and hold it securely.'
HOLD_TASK = 'Keep the robot still. Do not grasp the pen.'


def bounded_pen_action(raw):
    q=np.asarray(raw,dtype=float)
    if q.shape!=(6,) or not np.isfinite(q).all():raise ValueError('Invalid pen joint command')
    q=np.clip(q,LOWER,UPPER)
    q[2]=np.clip(q[2],max(LOWER[2],-UPPER[3]-q[1]),min(UPPER[2],-LOWER[3]-q[1]))
    q[3]=-q[1]-q[2]
    return q


class PenArm(Arm):
    def __init__(self):
        super().__init__()
        self.jaw_origin = origin(self.joints['gripper'].find('origin'))

    def transform(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape[-1:] != (6,) or not np.isfinite(q).all():
            raise ValueError('Pen task needs six finite joint angles.')
        t = np.broadcast_to(np.eye(4), q.shape[:-1] + (4, 4)).copy()
        for j, fixed in zip(self.chain, self.transforms):
            motion = np.broadcast_to(np.eye(4), t.shape).copy()
            if j.get('type') != 'fixed':
                axis = np.fromstring(j.find('axis').get('xyz'), sep=' ')
                motion[..., :3, :3] = rotation(axis, q[..., JOINTS.index(j.get('name'))])
            t = t @ fixed @ motion
        return t

    def centre_local(self, jaw):
        point = MOVING_PAD + [-MOVING_SIZE[0]/2, 0, 0]
        moving = (self.jaw_origin[:3,:3] @ rotation([0,0,1], jaw) @ point[...,None])[...,0] + self.jaw_origin[:3,3]
        return np.stack([(moving[...,0] + FIXED_PAD[0] + FIXED_SIZE[0]/2)/2,
                         np.zeros_like(jaw) + FIXED_PAD[1], np.zeros_like(jaw) + FIXED_PAD[2]], axis=-1)

    def forward(self, q):
        q = np.asarray(q); t = self.transform(q)
        return (t[..., :3,:3] @ self.centre_local(q[...,5])[...,None])[...,0] + t[..., :3,3]

    def axis(self, q):
        return self.transform(q)[..., :3,1]

    def solve(self, target, pen_axis, q, jaw=None):
        q = np.asarray(q, dtype=float)
        jaw = q[5] if jaw is None else float(jaw)
        axis = np.asarray(pen_axis, dtype=float); axis /= np.linalg.norm(axis)
        if np.dot(self.axis(q), axis) < 0: axis = -axis
        # Keep the approach direction level. A cylindrical pen leaves rotation
        # about its long axis unconstrained; without this posture constraint a
        # redundant pitch solution can jump through the floating object.
        indices = [0,1,2,4]
        def unpack(joints):
            return np.array([joints[0],joints[1],joints[2],-joints[1]-joints[2],joints[3],jaw])
        def residual(joints):
            full = unpack(joints)
            return np.r_[self.forward(full) - target,
                         .02 * (self.axis(full) - axis), .0005 * (joints-q[indices])]
        fit = least_squares(residual, np.clip(q[indices], LOWER[indices]+1e-7, UPPER[indices]-1e-7),
                            bounds=(LOWER[indices], UPPER[indices]), max_nfev=35,
                            ftol=1e-5, xtol=1e-5, gtol=1e-6)
        return unpack(fit.x)
