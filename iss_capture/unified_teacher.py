"""Privileged demonstration generator. Never imported by learned inference."""
import numpy as np
from .pose_teacher import PoseTeacher,tool_axis_for_object
from .capping_pose import PoseCodec,OPEN,CLOSED
from .unified_config import SPONGE,ASSEMBLY_POSITION

class Teacher:
    def __init__(self,task):
        self.task=task;self.teacher=PoseTeacher();self.codec=PoseCodec()
        self.home=None;self.phase='approach';self.last=None;self.water_hold=None;self.pen_goal=None;self.centering=False;self.centred=False;self.centre_ready=None
    def action(self,client):
        q,t,_=client.snapshot();tcp=self.codec.positions(q);axes=self.codec.axes(q)
        if self.home is None:self.home=(tcp.copy(),axes.copy())
        if self.task=='task1':
            obj=client.object_states()['droplet'];velocity=np.zeros(3)
            if self.last is not None:
                velocity=(obj['position']-self.last['position'])/max(obj['stamp']-self.last['stamp'],.001)
            self.last=obj
            rotation=self.codec.arms[0].transform(q[:6])[:3,:3]
            goal=obj['position']+np.clip(velocity,-.02,.02)*.25-rotation@(SPONGE-self.codec.arms[0].tcp_point)
            if client.water_contact:
                if self.water_hold is None:self.water_hold=tcp[0].copy()
                goal=self.water_hold;self.phase='hold'
            return self.codec.pack(goal,self.home[0][1],[self.home[1][0],self.home[1][1]],[OPEN,OPEN])
        if self.task=='task3' and self.centering:
            objects=client.object_states()
            directions=[tool_axis_for_object(axes[0],-objects['floating_pen']['axis'],[0,0,1]),
                tool_axis_for_object(axes[1],objects['floating_cap']['axis'],[0,0,1])]
            movement=ASSEMBLY_POSITION-tcp[0]
            goal=self.codec.pack(ASSEMBLY_POSITION,tcp[1]+movement,directions,[CLOSED,CLOSED]);self.phase='centre'
            if np.linalg.norm(movement)<.003:
                self.centre_ready=t if self.centre_ready is None else self.centre_ready
                if t-self.centre_ready>.2:
                    self.centering=False;self.centred=True;self.teacher.phase='align';self.teacher.hold['left']=ASSEMBLY_POSITION.copy()
            else:self.centre_ready=None
            return goal
        before=self.teacher.phase
        goal=self.teacher.action(client);self.phase=self.teacher.phase
        if self.task=='task3' and not self.centred and before=='orient' and self.teacher.phase=='align':
            self.centering=True;self.teacher.phase='centre'
            return self.action(client)
        if self.task=='task2':
            goal[3:6]=self.home[0][1]-goal[:3];goal[8:10]=self.home[1][1,:2];goal[11]=OPEN
            if self.teacher.grasp['left']=='hold':
                if self.pen_goal is None:self.pen_goal=tcp[0]+[-.015,0,.050]
                goal[:3]=self.pen_goal;goal[3:6]=self.home[0][1]-goal[:3];goal[10]=CLOSED;self.phase='carry'
        return goal
