"""Object geometry from RGB/depth images only, never simulator poses."""
import numpy as np
from scipy.optimize import least_squares
from .unified_features import CAL
def cloud(rgb,depth,camera,kind,prior):
    rgb=np.asarray(rgb);depth=np.asarray(depth,dtype=float)
    if rgb.shape[:2]!=depth.shape:raise ValueError('RGB and depth dimensions differ')
    r,g,b=rgb.astype(float).transpose(2,0,1)
    mask=(r>1.35*g)&(r>1.7*b)&(r>65) if kind=='pen' else (b>1.5*r)&(b>1.45*g)&(b>70)
    mask&=np.isfinite(depth)&(depth>.05)&(depth<2)
    v,u=np.nonzero(mask);d=depth[v,u]
    eye,rotation,old_focal=CAL[camera];scale=depth.shape[1]/256.;f=old_focal*scale
    cx=(depth.shape[1]-1)/2;cy=(depth.shape[0]-1)/2
    local=np.column_stack([d,-(u-cx)/f*d,-(v-cy)/f*d])
    points=local@rotation.T+eye
    keep=(points[:,0]>.12)&(points[:,0]<.5)&(np.abs(points[:,1])<.26)&(points[:,2]>.07)&(points[:,2]<.5)
    keep&=np.linalg.norm(points-np.asarray(prior),axis=1)<(.10 if kind=='pen' else .065)
    return points[keep]

def fit_cylinder(points,kind,axis_prior=None,centre_prior=None,held=False):
    points=np.asarray(points,float)
    if len(points)<24:return None
    if len(points)>1200:points=points[np.linspace(0,len(points)-1,1200).astype(int)]
    centre=np.median(points,axis=0);cov=np.cov(points.T);values,vectors=np.linalg.eigh(cov)
    axis=np.asarray(axis_prior).copy() if held and axis_prior is not None else vectors[:,-1]
    if kind=='cap' and values[-1]/max(values[-2],1e-12)<1.8 and axis_prior is not None:axis=np.asarray(axis_prior).copy()
    if axis[2]<0:axis=-axis
    axis/=np.linalg.norm(axis)
    e1=np.cross(axis,[0,1,0])
    if np.linalg.norm(e1)<.1:e1=np.cross(axis,[1,0,0])
    e1/=np.linalg.norm(e1);e2=np.cross(axis,e1)
    radius=.0048 if kind=='pen' else .00902
    prior=axis if axis_prior is None else np.asarray(axis_prior)
    def unpack(x):
        c=centre+x[0]*e1+x[1]*e2;n=axis+x[2]*e1+x[3]*e2;n/=np.linalg.norm(n)
        return c,n
    def radial(x,pts):
        c,n=unpack(x);d=pts-c
        return np.linalg.norm(d-(d@n)[:,None]*n,axis=1)-radius
    def residual(x,pts):
        c,n=unpack(x)
        extra=[]
        if held and centre_prior is not None:
            difference=np.asarray(centre_prior)-c;perp=difference-n*np.dot(difference,n)
            extra=np.sqrt(len(pts))*.2*perp
        return np.r_[radial(x,pts),np.sqrt(len(pts))*.00015*(n-prior),extra]
    xy=np.column_stack([(points-centre)@e1,(points-centre)@e2])
    circle=np.linalg.lstsq(np.c_[2*xy,np.ones(len(xy))],np.sum(xy*xy,axis=1),rcond=None)[0]
    initial=np.r_[np.clip(circle[:2],-.029,.029),0.,0.]
    if held and centre_prior is not None:
        initial[:2]=np.clip([np.dot(np.asarray(centre_prior)-centre,e1),np.dot(np.asarray(centre_prior)-centre,e2)],-.029,.029)
    fit=least_squares(residual,initial,args=(points,),bounds=([-.03,-.03,-.5,-.5],[.03,.03,.5,.5]),
        loss='soft_l1',f_scale=.0003,max_nfev=45,x_scale=[.01,.01,.1,.1])
    errors=radial(fit.x,points);wall=points[np.abs(errors)<.001]
    if len(wall)>=20:
        fit=least_squares(residual,fit.x,args=(wall,),bounds=([-.03,-.03,-.5,-.5],[.03,.03,.5,.5]),
            loss='soft_l1',f_scale=.00015,max_nfev=30,x_scale=[.01,.01,.1,.1])
    c,n=unpack(fit.x)
    if n[2]<0:n=-n
    projected=(points-c)@n
    endpoint=np.quantile(projected,.005 if kind=='pen' else .995)
    marker=c+n*endpoint
    centre_object=marker+n*.057 if kind=='pen' else marker-n*.0275
    return dict(marker=marker,centre=centre_object,axis=n,points=len(points),
        radial_error_m=float(np.median(np.abs(radial(fit.x,wall if len(wall) else points)))))

class DepthGeometry:
    def __init__(self):
        self.centres={'pen':np.array([.29,.035,.215]),'cap':np.array([.29,-.060,.325])}
        self.axes={'pen':np.array([0.,0.,1.]),'cap':np.array([0.,0.,1.])}
        self.last={};self.tool_transforms={};self.last_jaws={};self.last_stamp=None
        from .capping_pose import PoseCodec
        self.codec=PoseCodec()
    def __call__(self,observation):
        result={};stamp=float(observation.get('image_time',observation.get('sim_time',0.)))
        if self.last_stamp is not None and abs(stamp-self.last_stamp)<1e-9 and len(self.last)==2:return dict(self.last)
        for index,kind in enumerate(('pen','cap')):
            transform=self.codec.arms[index].transform(np.asarray(observation['q'])[index*6:index*6+6])
            jaw=float(observation['q'][index*6+5]);previous=float(observation.get('previous_command',observation['q'])[index*6+5])
            propagated=None
            stopped=kind in self.last_jaws and self.last_stamp is not None and stamp>self.last_stamp and abs(jaw-self.last_jaws[kind])/(stamp-self.last_stamp)<.05
            held=bool(stopped and -.15<jaw<.2 and previous<.19)
            if kind in self.last and kind in self.tool_transforms and -.15<jaw<.2 and previous<.19:
                delta=transform@np.linalg.inv(self.tool_transforms[kind]);prior=self.last[kind]
                propagated={**prior,'centre':delta[:3,:3]@prior['centre']+delta[:3,3],
                    'marker':delta[:3,:3]@prior['marker']+delta[:3,3],'axis':delta[:3,:3]@prior['axis'],'points':0,'stale':True}
                self.centres[kind]=propagated['centre'];self.axes[kind]=propagated['axis']
            pts=np.concatenate([cloud(observation[name],observation[name+'_depth'],i,kind,self.centres[kind]) for i,name in enumerate(('front','side'))])
            fitted=fit_cylinder(pts,kind,self.axes[kind],self.centres[kind],held)
            if fitted is None or fitted['radial_error_m']>.0007:
                if kind not in self.last:raise RuntimeError('RGB-D object not detected: '+kind)
                fitted=propagated or {**self.last[kind],'points':0,'stale':True}
            else:
                self.centres[kind]=fitted['centre'];self.axes[kind]=fitted['axis'];self.last[kind]=fitted
            self.last[kind]=fitted;self.tool_transforms[kind]=transform.copy();self.last_jaws[kind]=jaw
            result[kind]=fitted
        self.last_stamp=stamp
        return result

class RGBDFeatures:
    def __init__(self):
        from .unified_features import Features
        self.rgb=Features();self.geometry=DepthGeometry();self.codec=self.rgb.codec;self.last_geometry=None
    def __call__(self,observation):
        if 'front_depth' not in observation or 'side_depth' not in observation:raise ValueError('RGB-D observations are required')
        state=self.rgb(observation);geometry=self.geometry(observation);self.last_geometry=geometry
        pen,cap=geometry['pen'],geometry['cap']
        state[23:26]=pen['marker'];state[26:29]=cap['marker']-pen['marker'];state[29:31]=pen['axis'][:2]
        state[31]=np.mean([min(g['points']/100.,1.) for g in (pen,cap)])
        return np.r_[state,cap['axis'][:2]].astype(np.float32)
