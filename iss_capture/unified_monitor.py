"""Independent task metrics; never passed to policy inference."""
import numpy as np

class Monitor:
    def __init__(self,task):
        from .capping_monitor import CappingMonitor
        self.task=task;self.capping=CappingMonitor(require_carry=False);self.origin=None;self.start=None
        self.success=False;self.last={};self.samples=[]
    def update(self,client):
        q,t,_=client.snapshot()
        if self.task=='task3':
            # CappingMonitor predates the extra water distractor.
            class View:
                def snapshot(self):return client.snapshot()
                def bilateral(self,side):return client.bilateral(side)
                def object_states(self):return {k:v for k,v in client.object_states().items() if k!='droplet'}
            self.last=self.capping.update(View());self.success=self.capping.success
        elif self.task=='task1':
            stable=0 if client.water_contact_time is None else t-client.water_contact_time
            self.success=client.water_contact and stable>=.6
            self.last=dict(time=t,actual_sponge_contact=client.water_contact,stable_seconds=stable,success=self.success)
        else:
            obj=client.object_states()['floating_pen'];hit=bool(client.bilateral('left') and q[5]<.15)
            if hit and self.origin is None:self.origin=obj['position'].copy()
            lift=0. if self.origin is None else float(obj['position'][2]-self.origin[2])
            if hit and lift>.025:
                if self.start is None:self.start=t
            else:self.start=None
            stable=0. if self.start is None else t-self.start
            self.success=stable>=.6
            self.last=dict(time=t,bilateral_contact=hit,lift_m=lift,stable_seconds=stable,success=self.success)
        self.samples.append(self.last);return self.last
