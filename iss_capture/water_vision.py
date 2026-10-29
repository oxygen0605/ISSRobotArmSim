"""Track the transparent water silhouette in two RGB views; no object truth."""
import cv2
import numpy as np
from .unified_features import CAL,project,stereo
from .config import DROPLET_RADIUS
class WaterVision:
    def __init__(self):
        self.position=np.array([.275,.160,.245],float);self.velocity=np.zeros(3);self.stamp=None
        self.last_info={}
    def __call__(self,observation):
        stamp=float(observation.get('image_time',observation.get('sim_time',0.)))
        if self.stamp is not None and stamp==self.stamp:return self.position.copy()
        dt=.1 if self.stamp is None else max(.001,stamp-self.stamp)
        prior=self.position+self.velocity*min(dt,.2);pixels=[];confidence=[];radii=[]
        for camera,name in enumerate(('front','side')):
            rgb=np.asarray(observation[name],np.uint8);height,width=rgb.shape[:2];scale=width/256.
            pixel=(project(prior,camera)+.5)*scale-.5
            eye,rotation,focal=CAL[camera];depth=(rotation.T@(prior-eye))[0]
            radius=scale*focal*DROPLET_RADIUS/max(depth,.05)
            extent=max(24.,radius*2.5)
            start=np.maximum(np.floor(pixel-extent),0).astype(int);stop=np.minimum(np.ceil(pixel+extent),[width,height]).astype(int)
            candidates=None
            if min(stop-start)>12:
                crop=rgb[start[1]:stop[1],start[0]:stop[0]]
                gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
                candidates=cv2.HoughCircles(cv2.GaussianBlur(gray,(3,3),.7),cv2.HOUGH_GRADIENT,1,8,
                    param1=45,param2=8,minRadius=max(4,int(radius*.65)),maxRadius=max(5,int(radius*1.35)+1))
            if candidates is None:
                pixels.append((pixel+.5)/scale-.5);confidence.append(0.);radii.append(None);continue
            circles=candidates[0].copy();circles[:,:2]+=start
            score=np.linalg.norm(circles[:,:2]-pixel,axis=1)+2*np.abs(circles[:,2]-radius)
            chosen=circles[np.argmin(score)]
            if np.linalg.norm(chosen[:2]-pixel)>max(12.,radius*1.4):
                pixels.append((pixel+.5)/scale-.5);confidence.append(0.);radii.append(None);continue
            pixels.append((chosen[:2]+.5)/scale-.5);confidence.append(1.);radii.append(float(chosen[2]))
        estimate=stereo(pixels,confidence,prior)
        valid=any(confidence) and np.isfinite(estimate).all() and np.linalg.norm(estimate-prior)<.035
        if valid:
            if self.stamp is not None:self.velocity=.65*self.velocity+.35*np.clip((estimate-self.position)/dt,-.04,.04)
            self.position=estimate
        else:self.position=prior
        self.stamp=stamp
        self.last_info=dict(views=int(sum(confidence)),radii_px=radii,accepted=bool(valid),source='two RGB sphere silhouettes')
        return self.position.copy()
