"""ROS 2 episode transport; object truth is isolated from policy observations."""
import os,json,time,uuid,subprocess
from collections import deque
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from ros_gz_interfaces.msg import Contacts,EntityWrench,Entity
from .bimanual_transport import BimanualClient
from .unified_world import build
from .config import DROPLET_RADIUS
from .capping_motion import start_motion
from run_demo import stop_process

class Client(BimanualClient):
    def __init__(self,*args,**kwargs):
        self.use_depth=kwargs.pop('use_depth',True)
        self.capture_recoveries=[]
        self.water_contact=False;self.water_contact_time=None;self.depth_frames={n:deque(maxlen=16) for n in ('front','side')}
        super().__init__(*args,**kwargs)
        self.subscribe(Contacts,'/capture/contacts',self._water)
        self.capture_publisher=self.node.create_publisher(Bool,'/camera/capture',10)
        # Extend bridge before launching the simulator.
        stop_process(self.bridge);self.bridge_log.close();self.bridge=self.bridge_log=None
        entries=json.loads(self.config.read_text())
        entries.append(dict(topic_name='/camera/capture',ros_type_name='std_msgs/msg/Bool',gz_type_name='gz.msgs.Boolean',direction='ROS_TO_GZ'))
        entries.append(dict(topic_name='/capture/contacts',ros_type_name='ros_gz_interfaces/msg/Contacts',
            gz_type_name='gz.msgs.Contacts',direction='GZ_TO_ROS',qos_profile='SENSOR_DATA'))
        for name in (('front','side') if self.use_depth else ()):
            topic='/camera/'+name+'/depth'
            entries.append(dict(topic_name=topic,ros_type_name='sensor_msgs/msg/Image',gz_type_name='gz.msgs.Image',direction='GZ_TO_ROS',qos_profile='SENSOR_DATA'))
            self.subscribe(Image,topic,lambda msg,n=name:self._depth(n,msg))
        self.config.write_text(json.dumps(entries,indent=2));self.start_bridge()
    def camera_observation(self):
        # Trigger all four sensor channels at the same paused simulation time.
        target=self.clock_time
        if target is None:return None
        observation=super().camera_observation()
        if observation is not None and observation['image_time']>=target-.0020001:return observation
        deadline=time.monotonic()+(15 if observation is None else 5);last_trigger=-1.;recoveries=0
        while True:
            now=time.monotonic()
            if now-last_trigger>.5:
                self.capture_publisher.publish(Bool(data=True));last_trigger=now
            observation=super().camera_observation()
            if observation is not None and observation['image_time']>=target-.0020001:
                observation['fresh_at_sensor_tick']=True;return observation
            if now>deadline:
                with self.lock:
                    details=dict(target_time=target,clock_time=self.clock_time,
                        rgb={name:(frames[-1][0] if frames else None) for name,frames in self.frames.items()},
                        depth={name:(frames[-1][0] if frames else None) for name,frames in self.depth_frames.items()})
                if recoveries>=2:raise TimeoutError('Triggered RGB-D capture did not complete after resynchronization: '+json.dumps(details))
                # Wake a delayed rendering update without accepting stale sensor data.
                # The prior joint command continues for exactly two physics ticks.
                self.capture_recoveries.append(details);self.step(2);recoveries+=1
                target=self.clock_time;deadline=time.monotonic()+5;last_trigger=-1.
            time.sleep(.003)
    def close(self):
        (self.output/'capture_recoveries.json').write_text(json.dumps(self.capture_recoveries,indent=2))
        super().close()
    def _depth(self,name,message):
        from .ros_transport import stamp_seconds
        if message.encoding not in ('32FC1','32FC'):return
        if message.step<message.width*4 or message.step%4:return
        dtype='>f4' if message.is_bigendian else '<f4'
        values=np.frombuffer(message.data,dtype=dtype).reshape(message.height,message.step//4)[:,:message.width].astype(np.float32)
        with self.lock:self.depth_frames[name].append((stamp_seconds(message.header.stamp),values))
    def synchronized_observation(self):
        if not self.use_depth:
            return super().synchronized_observation()
        with self.lock:
            if not self.joint_history or any(not f for f in self.frames.values()) or any(not f for f in self.depth_frames.values()):return None
            choices=[]
            for ft,front in list(self.frames['front'])[-8:]:
                side=min(self.frames['side'],key=lambda f:abs(f[0]-ft))
                fd=min(self.depth_frames['front'],key=lambda f:abs(f[0]-ft))
                sd=min(self.depth_frames['side'],key=lambda f:abs(f[0]-ft))
                if max(abs(side[0]-ft),abs(fd[0]-ft),abs(sd[0]-ft))<=.0020001:choices.append((ft,front,side[1],fd[1],sd[1]))
            if not choices:return None
            stamp,front,side,front_depth,side_depth=max(choices,key=lambda v:v[0])
            history=list(self.joint_history);times=np.array([r[0] for r in history])
            hi=min(len(history)-1,int(np.searchsorted(times,stamp)));lo=max(0,hi-1)
            weight=np.clip((stamp-times[lo])/max(times[hi]-times[lo],1e-9),0,1)
            q=history[lo][1]*(1-weight)+history[hi][1]*weight
            previous=min(self.frames['front'],key=lambda f:abs(f[0]-(stamp-.2)))
            commands=[target for t,target in self.command_history if t<=stamp+1e-6]
            return dict(front=front.copy(),side=side.copy(),previous_front=previous[1].copy(),
                front_depth=front_depth.copy(),side_depth=side_depth.copy(),q=q.copy(),
                previous_command=commands[-1].copy() if commands else q.copy(),q_velocity=self.joint_velocities.copy(),
                sim_time=stamp,image_time=stamp,previous_image_time=previous[0],controller_time=self.sim_time,
                encoder_image_alignment=True,depth_image_alignment=True)
    def _poses(self,message):
        from .ros_transport import stamp_seconds
        for trans in message.transforms:
            if trans.child_frame_id not in ('floating_pen','floating_cap','droplet'):continue
            p,q=trans.transform.translation,trans.transform.rotation
            quat=np.array([q.x,q.y,q.z,q.w])
            if np.linalg.norm(quat)<1e-9:continue
            matrix=Rotation.from_quat(quat).as_matrix()
            with self.lock:self.objects[trans.child_frame_id]=dict(position=np.array([p.x,p.y,p.z]),
                quaternion=quat,rotation=matrix,axis=matrix[:,2],stamp=stamp_seconds(trans.header.stamp))
    def _water(self,message):
        from .ros_transport import stamp_seconds
        hit=any(('water_collision' in c.collision1.name and 'sponge_collision' in c.collision2.name) or
                ('water_collision' in c.collision2.name and 'sponge_collision' in c.collision1.name) for c in message.contacts)
        if hit:
            self.water_contact=True
            if self.water_contact_time is None:self.water_contact_time=stamp_seconds(message.header.stamp)

class Session:
    def __init__(self,output,case,gui=False,record=False):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.world,self.manifest=build(self.output/'scene',case,record)
        self.case=case;self.gui=gui;self.client=self.process=self.log=None
        self.partition=os.environ.get('GZ_PARTITION')
    def __enter__(self):
        os.environ['GZ_PARTITION']='one_policy_'+uuid.uuid4().hex
        (self.output/'partition.txt').write_text(os.environ['GZ_PARTITION'])
        try:
            self.client=Client(self.output,self.manifest['joint_control'],True)
            self.log=(self.output/'gazebo.log').open('w')
            self.process=subprocess.Popen(['gz','sim','-r','-v','3']+([] if self.gui else ['-s'])+[str(self.world)],
                stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+10
            while self.client.clock_time is None and time.monotonic()<deadline:time.sleep(.05)
            if self.client.clock_time is None:
                stop_process(self.client.bridge);self.client.bridge_log.close();self.client.bridge=self.client.bridge_log=None
                self.client.start_bridge()
            self.client.activate_controllers();deadline=time.monotonic()+40
            while True:
                q,t,_=self.client.snapshot()
                if q is not None and t>.5 and len(self.client.object_states())==3:break
                if self.process.poll() is not None:raise RuntimeError('Gazebo exited')
                if time.monotonic()>deadline:raise TimeoutError('Shared scene not ready')
                time.sleep(.05)
            self.client.pause();self.client.step(100);self.client.camera_observation()
            start_motion(self.client,self.case,self.output)
            mass=1000*4*np.pi*DROPLET_RADIUS**3/3
            req=EntityWrench();req.entity.name='droplet';req.entity.type=Entity.MODEL
            req.wrench.force.x,req.wrench.force.y,req.wrench.force.z=map(float,mass*np.asarray(self.case['water_velocity'])/.001)
            self.client.impulse_publisher.publish(req);time.sleep(.05);self.client.step(1);self.client.step(100)
            return self
        except BaseException:self.__exit__(None,None,None);raise
    def __exit__(self,*args):
        stop_process(self.process)
        if self.client:self.client.close()
        if self.log:self.log.close()
        if self.partition is None:os.environ.pop('GZ_PARTITION',None)
        else:os.environ['GZ_PARTITION']=self.partition
