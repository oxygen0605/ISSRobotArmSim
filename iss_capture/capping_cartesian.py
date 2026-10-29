"""Bimanual Cartesian policy codec: robot proprioception and generic IK only."""
import numpy as np
from .capping_posture import UprightArm
from .pen_kinematics import FIXED_PAD
from .bimanual_config import LOWER,UPPER,CAP_OUTER_RADIUS

TCP_POINTS={
    'left':np.array([-.012+np.sqrt(2)*.0054,FIXED_PAD[1],FIXED_PAD[2]]),
    'right':np.array([-.012+np.sqrt(2)*CAP_OUTER_RADIUS,FIXED_PAD[1],FIXED_PAD[2]])}
OPEN=.55
CLOSED=-.17

class FixedTCPArm(UprightArm):
    def __init__(self,side):
        super().__init__(side);self.tcp_point=TCP_POINTS[side]
    def forward(self,q):
        t=self.transform(q);return t[:3,:3]@self.tcp_point+t[:3,3]

class CartesianCodec:
    def __init__(self):
        self.arms=[FixedTCPArm(s) for s in ('left','right')]
        self.last_residuals=None
    def positions(self,q):
        q=np.asarray(q,dtype=float)
        return np.stack([a.forward(q[i*6:(i+1)*6]) for i,a in enumerate(self.arms)])
    def features(self,observation):
        q=np.asarray(observation['q'],dtype=float)
        previous=np.asarray(observation['previous_command'],dtype=float)
        current=self.positions(q);goal=self.positions(previous)
        return np.r_[current[0],current[1]-current[0],goal[0],goal[1]-goal[0],q[[5,11]],previous[[5,11]],
            np.linalg.norm(goal-current,axis=1),np.linalg.norm((current[1]-current[0])[:2])].astype(np.float32)
    def pack(self,left,right,grippers):
        return np.r_[left,np.asarray(right)-left,grippers].astype(np.float32)
    def decode(self,goal,q):
        goal=np.asarray(goal,dtype=float);q=np.asarray(q,dtype=float)
        if goal.shape!=(8,) or q.shape!=(12,) or not np.isfinite(np.r_[goal,q]).all():raise ValueError('Invalid Cartesian goal')
        left=np.clip(goal[:3],[.23,-.10,.14],[.43,.10,.34])
        relative=np.clip(goal[3:6],[-.12,-.12,.025],[.22,.12,.24])
        right=left+relative
        right=np.clip(right,[.26,-.10,.20],[.48,.10,.40])
        commands=[];residuals=[]
        for i,(arm,xyz) in enumerate(zip(self.arms,(left,right))):
            command=arm.solve(xyz,[0,0,1],q[i*6:(i+1)*6],jaw=float(np.clip(goal[6+i],LOWER[5],UPPER[5])))
            commands.append(command);residuals.append(float(np.linalg.norm(arm.forward(command)-xyz)))
        self.last_residuals=residuals
        return np.concatenate(commands)
