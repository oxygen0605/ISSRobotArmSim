"""Shared, observable object-relative features for the two-arm policy."""
import numpy as np
from .depth_geometry import RGBDFeatures
from .pen_kinematics import FIXED_PAD,FIXED_SIZE
from .unified_config import SPONGE
from .capping_pose import PoseCodec
PHASES=('water_approach','water_hold','pen_grasp','pen_carry','dual_grasp','dual_orient','dual_centre','dual_align','dual_insert','dual_hold')
def phase_name(task,phase):
    if task=='task1':return 'water_hold' if phase=='hold' else 'water_approach'
    if task=='task2':return 'pen_carry' if phase=='carry' else 'pen_grasp'
    return 'dual_'+phase
def unit_axis(xy):
    xy=np.asarray(xy,float);return np.r_[xy,np.sqrt(max(1e-8,1-float(xy@xy)))]
def gap_offset(arm,jaw,radius):
    flat=arm.arm.centre_local(np.array(jaw))
    moving=2*flat[0]-(FIXED_PAD[0]+FIXED_SIZE[0]/2)
    return np.array([(moving-radius-arm.tcp_point[0])/2,0,0])
def enrich(state34,q,velocity,dt,age,codec=None):
    codec=codec or PoseCodec();q=np.asarray(q,float);state=np.asarray(state34,float)
    rotations=np.stack([a.transform(q[i*6:i*6+6])[:3,:3] for i,a in enumerate(codec.arms)])
    tcp=np.stack([state[:3],state[:3]+state[3:6]])
    pen_axis=unit_axis(state[29:31]);cap_axis=unit_axis(state[32:34])
    pen=state[23:26]+.057*pen_axis;cap=state[23:26]+state[26:29]-.0275*cap_axis
    local=np.r_[rotations[0].T@(state[20:23]-tcp[0])-(SPONGE-codec.arms[0].tcp_point),
        rotations[0].T@(pen-tcp[0])-gap_offset(codec.arms[0],q[5],.0054),
        rotations[1].T@(cap-tcp[1])-gap_offset(codec.arms[1],q[11],.009),
        rotations[0].T@pen_axis,rotations[1].T@cap_axis]
    full=np.r_[state,dt,age,np.asarray(velocity)[[5,11]],local].astype(np.float32)
    return full,rotations.astype(np.float32)
class WorkspaceFeatures:
    def __init__(self):self.depth=RGBDFeatures();self.codec=self.depth.codec;self.last_stamp=None
    def __call__(self,obs):
        state=self.depth(obs);stamp=obs['image_time'];dt=.1 if self.last_stamp is None else max(stamp-self.last_stamp,0)
        age=max(0.,obs.get('controller_time',stamp)-stamp)
        value,rotation=enrich(state,obs['q'],obs.get('q_velocity',np.zeros(12)),dt,age,self.codec)
        self.last_stamp=stamp
        return value,rotation
