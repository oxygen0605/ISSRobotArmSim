"""Differentiable robot geometry used ONLY as an additional training loss."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .pen_kinematics import PenArm,FIXED_PAD
from .bimanual_config import BASE_POSES
from .config import JOINTS

class KinematicLoss:
    def __init__(self,device='cuda',position_tolerance=.004,axis_tolerance=.04,pair_tolerance=.003):
        self.position_tolerance=position_tolerance;self.axis_tolerance=axis_tolerance;self.pair_tolerance=pair_tolerance
        arm=PenArm();self.steps=[]
        for joint,transform in zip(arm.chain,arm.transforms):
            axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ') if joint.get('type')!='fixed' else None
            if axis is not None:axis=axis/np.linalg.norm(axis)
            self.steps.append((torch.tensor(transform,dtype=torch.float32,device=device),
                None if axis is None else torch.tensor(axis,dtype=torch.float32,device=device),
                JOINTS.index(joint.get('name')) if axis is not None else None))
        self.bases=[]
        for side in ('left','right'):
            pose=BASE_POSES[side];t=np.eye(4);t[:3,:3]=Rotation.from_euler('xyz',pose[3:]).as_matrix();t[:3,3]=pose[:3]
            self.bases.append(torch.tensor(t,dtype=torch.float32,device=device))
        # A fixed palm point avoids using commanded gripper closure as a
        # fictitious change of contact width in the geometric objective.
        self.point=torch.tensor(FIXED_PAD,dtype=torch.float32,device=device)
    def fk(self,q):
        q=q.float();positions=[];axes=[]
        for side,base in enumerate(self.bases):
            t=base.expand(len(q),4,4)
            for fixed,axis,index in self.steps:
                t=t@fixed
                if axis is None:continue
                angle=q[:,side*6+index]
                x,y,z=axis.unbind()
                zero=torch.zeros_like(x)
                cross=torch.stack([zero,-z,y,z,zero,-x,-y,x,zero]).reshape(3,3)
                c=angle.cos()[:,None,None];s=angle.sin()[:,None,None]
                r=c*torch.eye(3,device=q.device)+(1-c)*(axis[:,None]*axis[None,:])+s*cross
                motion=torch.eye(4,device=q.device).expand(len(q),4,4).clone()
                motion[:,:3,:3]=r
                t=t@motion
            positions.append((t[:,:3,:3]@self.point)+t[:,:3,3]);axes.append(t[:,:3,1])
        return torch.stack(positions,dim=1),torch.stack(axes,dim=1)
    def __call__(self,predicted,target):
        position,axis=self.fk(predicted)
        target_position,target_axis=self.fk(target)
        position_loss=((position-target_position)/self.position_tolerance).square().mean()
        axis_loss=((axis-target_axis)/self.axis_tolerance).square().mean()
        pair=((position[:,1]-position[:,0]-(target_position[:,1]-target_position[:,0]))/self.pair_tolerance).square().mean()
        return position_loss+axis_loss+pair
