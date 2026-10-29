"""Common RGB/proprioceptive features; no simulator state or task identifier."""
import numpy as np
from scipy.ndimage import label,find_objects
from scipy.spatial.transform import Rotation
from .unified_config import CAMERAS
from .capping_pose import PoseCodec
CAL=[]
for name in ('front','side'):
    spec=CAMERAS[name];eye=np.array(spec['eye']);d=np.array(spec['target'])-eye
    rot=Rotation.from_euler('xyz',[0,np.arctan2(-d[2],np.linalg.norm(d[:2])),np.arctan2(d[1],d[0])]).as_matrix()
    CAL.append((eye,rot,128/np.tan(spec['fov']/2)))
def project(point,camera):
    eye,rot,f=CAL[camera];p=rot.T@(np.asarray(point)-eye)
    return np.array([127.5-f*p[1]/p[0],127.5-f*p[2]/p[0]])
def ray(pixel,camera):
    eye,rot,f=CAL[camera];u,v=pixel;d=rot@np.array([1.,-(u-127.5)/f,-(v-127.5)/f])
    return d/np.linalg.norm(d)
def stereo(pixels,confidence,prior):
    ds=[ray(p,i) for i,p in enumerate(pixels)]
    if min(confidence)>0:
        dist=np.linalg.lstsq(np.column_stack([ds[0],-ds[1]]),CAL[1][0]-CAL[0][0],rcond=None)[0]
        if np.all(dist>.05):return (CAL[0][0]+dist[0]*ds[0]+CAL[1][0]+dist[1]*ds[1])/2
    i=int(np.argmax(confidence))
    if confidence[i]>0:return CAL[i][0]+max(.1,np.dot(np.asarray(prior)-CAL[i][0],ds[i]))*ds[i]
    return np.asarray(prior)
def landmark(image,kind,camera,prior):
    resolution=np.asarray(image).shape[0]/256.;area_scale=resolution**2
    r,g,b=np.asarray(image,dtype=np.float32).transpose(2,0,1)
    if kind=='pen':mask=(r>1.35*g)&(r>1.7*b)&(r>65)
    elif kind=='cap':mask=(b>1.5*r)&(b>1.45*g)&(b>70)
    else:mask=(b>1.22*r)&(g>1.04*r)&(b<1.6*g)&(r<145)&(b>65)
    yy,xx=np.indices(mask.shape);center=(project(prior,camera)+.5)*resolution-.5;radius=(55 if kind!='water' else 24)*resolution
    mask&=((xx-center[0])**2+(yy-center[1])**2<radius**2)
    labels,_=label(mask);choices=[]
    for idx,slices in enumerate(find_objects(labels),1):
        local=labels[slices]==idx;area=int(local.sum())
        if area<(8 if kind=='pen' else 6)*area_scale or area>600*area_scale:continue
        y,x=np.nonzero(local);xy=np.column_stack([x+slices[1].start,y+slices[0].start]).astype(float)
        c=xy.mean(0);values,vectors=np.linalg.eigh(np.cov(xy.T));axis=vectors[:,-1]
        if axis[1]>0:axis=-axis
        if kind=='water':
            if area>180*area_scale or values[-1]/max(values[0],.2*area_scale)>4:continue
            point=c
        else:
            if kind=='pen' and (values[-1]<4*area_scale or values[-1]/max(values[0],.2*area_scale)<2):continue
            proj=(xy-c)@axis;edge=proj.min() if kind=='pen' else proj.max()
            chosen=proj<edge+1.4*resolution if kind=='pen' else proj>edge-1.4*resolution;point=xy[chosen].mean(0)
        score=np.linalg.norm(point-center)/resolution-.02*min(area/area_scale,100)
        choices.append((score,point,axis,min(area/(40*area_scale),1)))
    if not choices:return (center+.5)/resolution-.5,np.array([0.,-1.]),0.
    _,point,axis,confidence=min(choices,key=lambda v:v[0]);return (point+.5)/resolution-.5,axis,confidence
class Features:
    def __init__(self):self.codec=PoseCodec()
    def __call__(self,obs):
        q=np.asarray(obs['q']);prev=np.asarray(obs.get('previous_command',q))
        tcp=self.codec.positions(q);axes=self.codec.axes(q);last=self.codec.positions(prev)
        priors=[np.array([.275,.160,.28]),tcp[0]-.057*axes[0],tcp[1]+.0275*axes[1]]
        if q[5]>.3:priors[1]=np.array([.295,0,.16])
        if q[11]>.3:priors[2]=np.array([.295,0,.3525])
        points=[];detections=[];conf=[]
        for kind,prior in zip(('water','pen','cap'),priors):
            det=[landmark(obs[name],kind,i,prior) for i,name in enumerate(('front','side'))]
            points.append(stereo([v[0] for v in det],[v[2] for v in det],prior));detections.append(det);conf.extend(v[2] for v in det)
        normals=[np.cross(ray(d[0],i),ray(d[0]+20*d[1],i)) for i,d in enumerate(detections[1])]
        axis=np.cross(*normals)
        if min(v[2] for v in detections[1])<=0 or np.linalg.norm(axis)<1e-8:axis=axes[0]
        else:
            axis/=np.linalg.norm(axis)
            if axis[2]<0:axis=-axis
        state=np.r_[tcp[0],tcp[1]-tcp[0],axes[:,:2].ravel(),q[[5,11]],
            last[0],last[1]-last[0],prev[[5,11]],points[0],points[1],points[2]-points[1],axis[:2],np.mean(conf)]
        if state.shape!=(32,) or not np.isfinite(state).all():raise ValueError('Invalid visual state')
        return state.astype(np.float32)
def delta_target(goal,q,codec):
    tcp=codec.positions(q);axes=codec.axes(q);baseline=codec.pack(tcp[0],tcp[1],axes,[0,0])
    return (np.asarray(goal)-baseline).astype(np.float32)
def delta_decode(delta,q,codec):
    tcp=codec.positions(q);axes=codec.axes(q);baseline=codec.pack(tcp[0],tcp[1],axes,[0,0])
    bounded=np.asarray(delta).copy();bounded[:6]=np.clip(bounded[:6],-.06,.06);bounded[6:10]=np.clip(bounded[6:10],-.35,.35)
    return codec.decode(baseline+bounded,q)

def network_rgb(image):
    image=np.asarray(image,dtype=np.uint8)
    if image.shape==(256,256,3):return image
    from PIL import Image
    return np.asarray(Image.fromarray(image).resize((256,256),Image.Resampling.BILINEAR))
