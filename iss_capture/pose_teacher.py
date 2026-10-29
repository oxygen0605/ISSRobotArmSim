"""Ground-truth teacher for moving-object demonstrations only."""
import numpy as np
from scipy.spatial.transform import Rotation
from .capping_pose import PoseCodec,OPEN,CLOSED
from .bimanual_config import PEN_TIP,CAP_LENGTH,INSERT_DEPTH
from .pen_kinematics import FIXED_PAD,FIXED_SIZE

def tool_axis_for_object(tool_axis,object_axis,target):
    a=np.asarray(object_axis,float);b=np.asarray(target,float)
    cross=np.cross(a,b);norm=np.linalg.norm(cross)
    if norm<1e-9:return np.asarray(tool_axis)
    rotation=Rotation.from_rotvec(cross/norm*np.arctan2(norm,np.dot(a,b)))
    return rotation.apply(tool_axis)

class PoseTeacher:
    def __init__(self):
        self.codec=PoseCodec();self.phase='grasp';self.grasp={s:'pregrasp' for s in ('left','right')}
        self.hold={};self.closed_since={};self.last={};self.capture={};self.ready=None
    def action(self,client):
        q,t,_=client.snapshot();objects=client.object_states();tcp=self.codec.positions(q);axes=self.codec.axes(q)
        if self.phase=='grasp':
            goals=[];directions=[];grips=[]
            for i,(side,name) in enumerate([('left','floating_pen'),('right','floating_cap')]):
                obj=objects[name];pos=obj['position'];axis=(-obj['axis'] if i==0 else obj['axis']).copy()
                velocity=np.zeros(3);omega=np.zeros(3)
                if name in self.last:
                    old=self.last[name];dt=obj['stamp']-old['stamp']
                    if dt>.001:
                        velocity=(pos-old['position'])/dt
                        omega=Rotation.from_matrix(obj['rotation']@old['rotation'].T).as_rotvec()/dt
                self.last[name]=obj
                stage=self.grasp[side]
                if stage=='hold':
                    goals.append(self.hold[side]);directions.append(axes[i]);grips.append(CLOSED);continue
                contact=client.bilateral(side)
                lead=.18 if not contact else 0.
                anticipated=Rotation.from_rotvec(omega*lead).apply(axis)
                arm=self.codec.arms[i];rotation=arm.transform(q[6*i:6*i+6])[:3,:3]
                jaw=q[6*i+5] if stage=='close' else OPEN
                flat=arm.arm.centre_local(np.array(jaw))
                moving_x=2*flat[0]-(FIXED_PAD[0]+FIXED_SIZE[0]/2)
                radius=.0054 if side=='left' else .009
                gap=arm.tcp_point.copy();gap[0]=(arm.tcp_point[0]+moving_x-radius)/2
                offset=gap-arm.tcp_point
                goal=pos+np.clip(velocity,-.02,.02)*lead-rotation@offset
                if stage=='pregrasp':
                    goal+=.035*rotation[:,2]
                    if np.linalg.norm(tcp[i]-goal)<.012:self.grasp[side]='approach'
                elif stage=='approach':
                    with client.lock:
                        first_touch=any(client.contacts[side][pad] and t-client.contact_stamps[side][pad]<.15 for pad in ('fixed','moving'))
                    close_distance=.012 if side=='right' else .0075
                    if (np.linalg.norm(tcp[i]-goal)<close_distance or (first_touch and np.linalg.norm(tcp[i]-goal)<.020)) and np.dot(axes[i],axis)>.94:
                        self.grasp[side]='close';self.capture[side]=goal.copy()
                if self.grasp[side]=='close':
                    # Hold the capture frame after the first bilateral touch.
                    if contact:
                        self.closed_since.setdefault(side,t)
                        goal=self.capture[side]
                        if t-self.closed_since[side]>.35:
                            self.grasp[side]='hold';self.hold[side]=goal.copy()
                    else:
                        self.closed_since.pop(side,None)
                        self.capture[side]=goal.copy()
                goals.append(goal);directions.append(anticipated);grips.append(CLOSED if self.grasp[side] in ('close','hold') else OPEN)
            if all(v=='hold' for v in self.grasp.values()):self.phase='orient'
            return self.codec.pack(*goals,directions,grips)
        pen,cap=objects['floating_pen'],objects['floating_cap']
        # The teacher supplies desired poses; these task rules never run at policy inference.
        axis=-pen['axis'];left=self.hold['left'];right=tcp[1].copy()
        if self.phase=='orient':
            directions=np.array([tool_axis_for_object(axes[0],axis,[0,0,1]),tool_axis_for_object(axes[1],cap['axis'],[0,0,1])])
            if np.dot(axis,[0,0,1])>.999 and np.dot(cap['axis'],[0,0,1])>.999:
                self.phase='align'
            return self.codec.pack(left,right,directions,[CLOSED,CLOSED])
        phase_before=self.phase
        tip=pen['position']+PEN_TIP*axis
        gap=.010 if self.phase=='align' else -INSERT_DEPTH
        desired=tip+(CAP_LENGTH/2+gap)*axis
        correction=desired-cap['position']
        # Dense insertion corrections teach fine control rather than one fixed end pose.
        norm=np.linalg.norm(correction)
        right=tcp[1]+correction*min(1.,.008/max(norm,1e-9))
        lateral=np.linalg.norm(correction-np.dot(correction,axis)*axis)
        if self.phase=='align' and norm<.006 and lateral<.0015 and np.dot(axis,cap['axis'])>.995:
            self.ready=t if self.ready is None else self.ready
            if t-self.ready>.2:self.phase='insert'
        elif self.phase=='align':self.ready=None
        if phase_before=='insert' and norm<.0008:self.phase='hold'
        return self.codec.pack(left,right,[tool_axis_for_object(axes[0],axis,[0,0,1]),tool_axis_for_object(axes[1],cap['axis'],axis)],[CLOSED,CLOSED])


class RecoveryTeacher(PoseTeacher):
    """Teacher for perturbed post-grasp states, without a hidden left anchor."""
    def action(self,client):
        q,_,_=client.snapshot()
        if 'left' in self.hold:self.hold['left']=self.codec.positions(q)[0].copy()
        return super().action(client)
