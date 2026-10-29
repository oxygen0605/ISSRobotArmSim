"""Human viewer framing only; policy camera poses are defined separately."""
import math
import numpy as np

VIEWS = ('close', 'wide')


def viewer_pose(target, view='close', environment='iss'):
    if view not in VIEWS:
        raise ValueError('表示視点はcloseまたはwideを指定してください。')
    if view == 'wide':
        # Look through the open end along the cabin, retaining the Earth window
        # and cantilevered worktop in the same view. Policy cameras are separate.
        return ((1.65, -.85, .55, 0, .06, 2.50)
                if environment == 'iss' else (.9, -.75, .6, 0, .4, 2.3))
    target = np.asarray(target, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError('視点の注目位置が不正です。')
    # View from the droplet side of the arm so the initial sponge does not hide
    # the water. Allow some space above it for the upward motion.
    side = 1 if target[1] > 0 else -1
    eye = target + [.070, side * .180, .110]
    focus = target + [0, -side * .025, .018]
    direction = focus - eye
    yaw = math.atan2(direction[1], direction[0])
    pitch = math.atan2(-direction[2], np.linalg.norm(direction[:2]))
    return (*eye.tolist(), 0, pitch, yaw)
