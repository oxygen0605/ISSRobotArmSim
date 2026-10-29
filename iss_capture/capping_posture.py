"""Optional fixed upright tool manifold, using robot joints only."""
import numpy as np
from scipy.optimize import least_squares
from .bimanual_config import LOWER,UPPER
from .bimanual_kinematics import BimanualArm
VERTICAL_ROLL=1.61947846

def upright_joints(q):
    q=np.asarray(q,dtype=float).copy()
    if q.shape not in ((6,),(12,)) or not np.isfinite(q).all():raise ValueError('Finite SO-101 joint target required')
    for begin in range(0,len(q),6):
        s=q[begin:begin+6]
        s[:]=np.clip(s,LOWER[:6],UPPER[:6])
        s[2]=np.clip(s[2],max(LOWER[2],-UPPER[3]-s[1]),min(UPPER[2],-LOWER[3]-s[1]))
        s[3]=-s[1]-s[2];s[4]=VERTICAL_ROLL
    return q

class UprightArm(BimanualArm):
    def solve(self,target,axis,q,jaw=None):
        q=np.asarray(q,dtype=float);jaw=q[5] if jaw is None else jaw
        def unpack(x):return upright_joints(np.r_[x,0,VERTICAL_ROLL,jaw])
        def residual(x):return np.r_[self.forward(unpack(x))-target,.00003*(x-q[:3])]
        fit=least_squares(residual,np.clip(q[:3],LOWER[:3]+1e-6,UPPER[:3]-1e-6),
            bounds=(LOWER[:3],UPPER[:3]),max_nfev=100,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        return unpack(fit.x)
