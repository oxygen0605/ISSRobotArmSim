"""One 12-axis policy: left SO-101 holds the pen, right holds its cap."""
import numpy as np
from .config import ROOT, JOINTS
JOINT_NAMES = tuple(side+'_'+n for side in ('left','right') for n in JOINTS)
ARM_INDICES = [0,1,2,3,4,6,7,8,9,10]
GRIPPER_INDICES = [5,11]
BASE_POSES = {'left': [0.,.18,.025,0.,0.,0.], 'right': [0.,-.18,.025,0.,0.,0.]}
LOWER = np.tile([-.85,-1.4,-1.4,-1.6,-2.5,-.174533],2)
UPPER = np.tile([.85,1.4,1.5,1.6,2.5,.55],2)
STATE_SCALE = np.tile([1.,1.5,1.5,1.7,2.5,1.],2).astype(np.float32)
ACTION_SCALE = np.tile([.25,.25,.25,.25,.30,.50],2).astype(np.float32)
SPEEDS = np.tile([.65,.65,.65,1.3,1.2,1.],2)
NAMESPACE = '/bimanual'
CONTROLLERS = ('joint_state_broadcaster','arms_controller','grippers_controller')
TOPICS = tuple(NAMESPACE+'/'+c+'/joint_trajectory' for c in CONTROLLERS[1:])
TASK = 'Grasp the floating orange pen with the left arm and the blue cap with the right arm. Align the cap opening with the pen tip, then slide the cap onto the tip and hold both steadily.'
HOLD_TASK = 'Keep both robot arms still. Do not grasp or cap the pen.'
CHECKPOINT = ROOT/'artifacts/smolvla_bimanual_capping'
CAP_LENGTH = .055
CAP_INNER_RADIUS = .0075
CAP_OUTER_RADIUS = .009
PEN_TIP = .075
INSERT_DEPTH = .032
CHUNK_SIZE = 8
DT = .10

def bounded_action(q):
    q=np.asarray(q,dtype=float)
    if q.shape!=(12,) or not np.isfinite(q).all():raise ValueError('12 finite joint values required')
    return np.clip(q,LOWER,UPPER)
