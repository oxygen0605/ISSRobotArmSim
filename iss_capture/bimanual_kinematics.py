"""Generic robot forward/inverse kinematics shared by teacher and learned pose decoding."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .pen_kinematics import PenArm
from .bimanual_config import BASE_POSES, LOWER, UPPER

class BimanualArm:
    def __init__(self, side):
        self.arm=PenArm();self.side=side
        pose=BASE_POSES[side]
        self.base=np.eye(4);self.base[:3,:3]=Rotation.from_euler('xyz',pose[3:]).as_matrix()
        self.base[:3,3]=pose[:3]
    def transform(self,q):return self.base@self.arm.transform(q)
    def forward(self,q):
        t=self.transform(q);return t[:3,:3]@self.arm.centre_local(np.asarray(q)[5])+t[:3,3]
    def axis(self,q):return self.transform(q)[:3,1]
    def solve(self,target,axis,q,jaw=None):
        q=np.asarray(q,dtype=float);jaw=q[5] if jaw is None else jaw
        target=np.asarray(target);axis=np.asarray(axis,dtype=float);axis/=np.linalg.norm(axis)
        if np.dot(self.axis(q),axis)<0:axis=-axis
        def unpack(x):return np.r_[x,jaw]
        def residual(x):
            full=unpack(x)
            return np.r_[self.forward(full)-target,.10*(self.axis(full)-axis),.00003*(x-q[:5])]
        fit=least_squares(residual,np.clip(q[:5],LOWER[:5]+1e-6,UPPER[:5]-1e-6),
                          bounds=(LOWER[:5],UPPER[:5]),max_nfev=100,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        return unpack(fit.x)
    def initial(self,target):
        q=np.array([0.,-.5,.7,-.2,1.52,.55])
        return self.solve(target,[0,0,1],q,.55)
