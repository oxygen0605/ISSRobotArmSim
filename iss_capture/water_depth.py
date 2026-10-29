"""Sphere fitting on the explicitly idealized water depth-camera return."""
import numpy as np
from scipy.optimize import least_squares
from .unified_features import CAL
from .config import DROPLET_RADIUS
class WaterDepth:
    def __init__(self):
        self.position=np.array([.275,.160,.245],float);self.velocity=np.zeros(3);self.stamp=None
        self.initialized=False;self.last_info={}
    def __call__(self,obs):
        stamp=float(obs.get('image_time',obs.get('sim_time',0.)))
        if self.stamp is not None and stamp==self.stamp:return self.position.copy()
        dt=.1 if self.stamp is None else max(.001,stamp-self.stamp)
        prior=self.position+self.velocity*min(dt,.2);parts=[]
        for i,name in enumerate(('front','side')):
            rgb=np.asarray(obs[name],float);depth=np.asarray(obs[name+'_depth'],float)
            r,g,b=rgb.transpose(2,0,1)
            mask=(b>.95*r)&(g>.9*r)&(b>.82*g)&(r<200)&(b>40)&np.isfinite(depth)&(depth>.05)&(depth<2)
            y,x=np.nonzero(mask);d=depth[y,x];eye,rotation,focal=CAL[i]
            scale=depth.shape[1]/256.;cx=(depth.shape[1]-1)/2;cy=(depth.shape[0]-1)/2;f=focal*scale
            points=np.column_stack([d,-(x-cx)/f*d,-(y-cy)/f*d])@rotation.T+eye
            points=points[np.linalg.norm(points-prior,axis=1)<.035];parts.append(points)
        points=np.concatenate(parts);accepted=False;error=None;inlier_count=0
        if len(points)>=24:
            if len(points)>1200:points=points[np.linspace(0,len(points)-1,1200).astype(int)]
            residual=lambda c,p:np.linalg.norm(p-c,axis=1)-DROPLET_RADIUS
            # A robust sphere consensus rejects nearby planar gripper surfaces.
            candidates=[prior.copy()];rng=np.random.default_rng(91)
            for _ in range(96):
                sample=points[rng.choice(len(points),4,replace=False)]
                try:
                    centre=np.linalg.solve(2*(sample[1:]-sample[0]),np.sum(sample[1:]**2,axis=1)-np.sum(sample[0]**2))
                except np.linalg.LinAlgError:continue
                if np.linalg.norm(centre-prior)<.03 and abs(np.linalg.norm(sample[0]-centre)-DROPLET_RADIUS)<.002:candidates.append(centre)
            initial=max(candidates,key=lambda c:int(np.sum(np.abs(residual(c,points))<.0008))-.1*np.linalg.norm(c-prior)/.001)
            fit=least_squares(residual,initial,args=(points,),bounds=(prior-.035,prior+.035),loss='cauchy',f_scale=.0003,max_nfev=50,x_scale=.01)
            keep=np.abs(residual(fit.x,points))<.0015;inlier_count=int(keep.sum())
            if inlier_count>=24:
                fit=least_squares(residual,fit.x,args=(points[keep],),bounds=(prior-.035,prior+.035),loss='cauchy',f_scale=.0002,max_nfev=30,x_scale=.01)
                error=float(np.median(np.abs(residual(fit.x,points[keep]))))
                accepted=error<.0007 and np.linalg.norm(fit.x-prior)<.035 and keep.mean()>.15
            if accepted:
                if self.stamp is not None:self.velocity=.65*self.velocity+.35*np.clip((fit.x-self.position)/dt,-.05,.05)
                self.position=fit.x;self.initialized=True
        if not accepted:
            if not self.initialized:raise RuntimeError('No valid initial water depth sphere detected')
            self.position=prior
        self.stamp=stamp;self.last_info=dict(points=len(points),inliers=inlier_count,accepted=bool(accepted),radial_error_m=error,source='idealized depth surface fitted from RGB-D pixels')
        return self.position.copy()
