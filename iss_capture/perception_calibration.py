"""Learned RGB/encoder sensor calibration, common to all three tasks."""
import numpy as np
def basis(state):
    s=np.asarray(state,dtype=np.float64)
    one=s.ndim==1
    if one:s=s[None,:]
    axes=s[:,6:10].reshape(-1,2,2)
    z=np.sqrt(np.maximum(1e-6,1-np.sum(axes*axes,axis=2)))
    axis3=np.concatenate([axes,z[:,:,None]],axis=2)
    gates=np.clip((.4-s[:,10:12])/.3,0,1)*(s[:,18:20]<.19)*(s[:,10:12]>-.15)
    pen=s[:,23:26];cap=pen+s[:,26:29]
    tcp=np.stack([s[:,:3],s[:,:3]+s[:,3:6]],axis=1)
    points=np.stack([pen,cap],axis=1)
    held=np.concatenate([tcp-points,axis3,s[:,10:12,None]],axis=2)*gates[:,:,None]
    values=np.concatenate([s[:,23:32],s[:,:12],z,gates,held.reshape(len(s),-1)],axis=1)
    return values[0] if one else values
class PerceptionCalibration:
    def __init__(self,path):
        data=np.load(path,allow_pickle=False)
        self.mean=data['mean'];self.scale=data['scale'];self.weight=data['weight'];self.target_scale=data['target_scale']
    def __call__(self,state):
        s=np.asarray(state,dtype=np.float32).copy();x=(basis(s)-self.mean)/self.scale
        correction=np.r_[x,1.]@self.weight*self.target_scale
        correction[:6]=np.clip(correction[:6],-.05,.05);correction[6:]=np.clip(correction[6:],-.4,.4)
        s[23:26]+=correction[:3];s[26:29]+=correction[3:6];s[29:31]+=correction[6:]
        return s
