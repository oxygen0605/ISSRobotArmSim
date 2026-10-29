"""Generic paired position/orientation decoder. No assembly path constraints."""
import numpy as np
from .bimanual_kinematics import BimanualArm
from .capping_cartesian import TCP_POINTS,OPEN,CLOSED
from .bimanual_config import LOWER,UPPER

class PoseArm(BimanualArm):
    def __init__(self,side):
        super().__init__(side);self.tcp_point=TCP_POINTS[side]
    def forward(self,q):
        t=self.transform(q);return t[:3,:3]@self.tcp_point+t[:3,3]

def up_axis(xy):
    xy=np.asarray(xy,dtype=float);length=np.linalg.norm(xy)
    if length>.85:xy=xy*.85/length
    return np.r_[xy,np.sqrt(max(1e-8,1-np.dot(xy,xy)))]

class PoseCodec:
    def __init__(self):self.arms=[PoseArm(s) for s in ('left','right')];self.last_residuals=None
    def positions(self,q):
        return np.stack([a.forward(np.asarray(q)[6*i:6*i+6]) for i,a in enumerate(self.arms)])
    def axes(self,q):
        axes=np.stack([a.axis(np.asarray(q)[6*i:6*i+6]) for i,a in enumerate(self.arms)])
        return axes*np.where(axes[:,2:3]<0,-1.,1.)
    def pack(self,left,right,axes,grips):
        return np.r_[left,np.asarray(right)-left,np.asarray(axes)[:,:2].ravel(),grips].astype(np.float32)
    def decode(self,goal,q):
        goal=np.asarray(goal,float);q=np.asarray(q,float)
        if goal.shape!=(12,) or q.shape!=(12,) or not np.isfinite(np.r_[goal,q]).all():raise ValueError('Finite paired poses required')
        left=np.clip(goal[:3],[.18,-.14,.12],[.40,.20,.40])
        right=np.clip(goal[:3]+goal[3:6],[.18,-.20,.14],[.40,.14,.44])
        axes=[up_axis(goal[6:8]),up_axis(goal[8:10])]
        commands=[];errors=[]
        for i,(a,xyz,axis) in enumerate(zip(self.arms,(left,right),axes)):
            command=a.solve(xyz,axis,q[i*6:i*6+6],jaw=np.clip(goal[10+i],LOWER[5],UPPER[5]))
            commands.append(command);errors.append(np.linalg.norm(a.forward(command)-xyz))
        self.last_residuals=errors
        return np.concatenate(commands)
