"""Independent success measurement. Never supplies control inputs to the VLA."""
import numpy as np
from .bimanual_config import CAP_LENGTH,PEN_TIP

def fit_metrics(pen,cap):
    direction=-pen['axis']
    tip=pen['position']+PEN_TIP*direction
    opening=cap['position']-CAP_LENGTH/2*cap['axis']
    delta=tip-opening;depth=float(np.dot(delta,cap['axis']))
    radial=float(np.linalg.norm(delta-depth*cap['axis']))
    angle=float(np.arccos(np.clip(np.dot(direction,cap['axis']),-1,1)))
    return dict(insertion_depth_m=depth,lateral_error_m=radial,axis_error_degrees=float(np.rad2deg(angle)))

class CappingMonitor:
    def __init__(self,require_carry=True):
        self.require_carry=require_carry
        self.sides=('left','right')
        self.grasp_origin={};self.grasp_seen={s:False for s in self.sides};self.carried={s:False for s in self.sides}
        self.start=None;self.last=None;self.success=False;self.samples=[];self.max_depth=-1.
    def update(self,client):
        objects=client.object_states();q,t,_=client.snapshot()
        if len(objects)!=2:return {}
        contacts={s:client.bilateral(s) for s in self.sides}
        for side,name in [('left','floating_pen'),('right','floating_cap')]:
            if contacts[side]:
                self.grasp_seen[side]=True
                pos=objects[name]['position']
                if side not in self.grasp_origin:self.grasp_origin[side]=pos.copy()
                if np.linalg.norm(pos-self.grasp_origin[side])>=.006:self.carried[side]=True
        m=fit_metrics(objects['floating_pen'],objects['floating_cap'])
        self.max_depth=max(self.max_depth,m['insertion_depth_m'])
        speed=abs(m['insertion_depth_m']-self.last['insertion_depth_m'])/max(t-self.last['time'],1e-6) if self.last else 1.
        m['axial_relative_speed_m_s']=speed
        fresh=all(-.02<=t-v.get('stamp',t)<=.15 for v in objects.values())
        m['object_pose_fresh']=fresh
        valid=fresh and speed<.0015 and all(contacts.values()) and bool(np.all(q[[5,11]]<.15)) and (not self.require_carry or all(self.carried.values())) and .029<=m['insertion_depth_m']<=.038 and m['lateral_error_m']<.002 and m['axis_error_degrees']<8.
        if valid:
            if self.start is None:self.start=t
            if t-self.start>=.6:self.success=True
        else:self.start=None
        m.update(time=t,require_carry=self.require_carry,contacts=contacts,grasp_seen=dict(self.grasp_seen),carried=dict(self.carried),
            stable_seconds=0. if self.start is None else t-self.start,success=self.success)
        self.last=m;self.samples.append(m)
        return m
