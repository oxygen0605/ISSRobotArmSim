"""ROS-only transport for a single twelve-axis coordinated policy."""
from collections import deque
import json,os,threading,time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile,DurabilityPolicy
from std_msgs.msg import String
from sensor_msgs.msg import JointState,Image
from rosgraph_msgs.msg import Clock
from tf2_msgs.msg import TFMessage
from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
from builtin_interfaces.msg import Time,Duration
from ros_gz_interfaces.msg import Contacts,Entity,EntityWrench
from ros_gz_interfaces.srv import ControlWorld,SetEntityPose
from controller_manager_msgs.srv import LoadController,ConfigureController,SwitchController,ListControllers
from scipy.spatial.transform import Rotation
from .ros_transport import ROSMovingClient,bridge_config,stamp_seconds,rgb_image
from .bimanual_config import *
from .trajectory_timing import group_times,arm_schedule_indices

class BimanualClient(ROSMovingClient):
    def __init__(self,output,control_spec,synchronize_observations=False):
        self.output=Path(output);self.control_spec=control_spec
        self.synchronize_observations=synchronize_observations
        self.joint_history=deque(maxlen=600);self.command_history=deque(maxlen=100)
        self.lock=threading.Lock();self.positions=None;self.sim_time=self.updated_at=0.;self.clock_time=None
        self.frames={n:deque(maxlen=12) for n in ('front','side')}
        self.objects={};self.contacts={s:{p:False for p in ('fixed','moving')} for s in ('left','right')}
        self.contact_stamps={s:{p:-1. for p in ('fixed','moving')} for s in ('left','right')}
        self.contact_log=[];self.counts={};self.subscriptions=[];self.display_frames=[];self.display_last=-1.
        self.touched=threading.Event();self.trajectory_count=0;self.last_trajectory=None;self.previous_command=None
        self.joint_velocities=np.zeros(12);self.max_joint_velocity=np.zeros(12);self.controller_status=[]
        self.context=Context();rclpy.init(args=[],context=self.context)
        self.node=Node('biarm_task_controller',context=self.context,parameter_overrides=[Parameter('use_sim_time',value=True)])
        self.executor=SingleThreadedExecutor(context=self.context);self.executor.add_node(self.node)
        self.thread=threading.Thread(target=self.executor.spin,daemon=True)
        self.trajectory_publishers=[self.node.create_publisher(JointTrajectory,t,10) for t in TOPICS]
        self.description_publisher=self.node.create_publisher(String,NAMESPACE+'/robot_description',
            QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.description_publisher.publish(String(data=Path(control_spec['robot_description']).read_text()))
        self.control_client=self.node.create_client(ControlWorld,'/world/microgravity/control')
        self.pose_client=self.node.create_client(SetEntityPose,'/world/microgravity/set_pose')
        self.impulse_publisher=self.node.create_publisher(EntityWrench,'/world/microgravity/wrench',10)
        self.subscribe(JointState,NAMESPACE+'/joint_states',self._state)
        self.subscribe(Clock,'/clock',self._clock)
        self.subscribe(TFMessage,'/evaluation/object_pose',self._poses)
        for name in self.frames:self.subscribe(Image,'/camera/'+name+'/image',lambda msg,n=name:self._image(n,msg))
        entries=[e for e in bridge_config() if not str(e.get('topic_name','')).startswith('/capture/')]
        for side in ('left','right'):
            for pad in ('fixed','moving'):
                topic='/bimanual/'+side+'/'+pad+'_contacts'
                entries.append(dict(topic_name=topic,ros_type_name='ros_gz_interfaces/msg/Contacts',gz_type_name='gz.msgs.Contacts',direction='GZ_TO_ROS',qos_profile='SENSOR_DATA'))
                self.subscribe(Contacts,topic,lambda msg,s=side,p=pad:self._contact(s,p,msg))
        entries.append(dict(topic_name='/bimanual/display/image',ros_type_name='sensor_msgs/msg/Image',gz_type_name='gz.msgs.Image',direction='GZ_TO_ROS',qos_profile='SENSOR_DATA'))
        self.subscribe(Image,'/bimanual/display/image',self._display)
        self.config=self.output/'ros_bridge.json';self.config.write_text(json.dumps(entries,indent=2))
        self.bridge=self.bridge_log=None;self.thread.start()
        self.start_bridge()

    def camera_observation(self):
        if self.synchronize_observations:return self.synchronized_observation()
        result=super().camera_observation()
        if result is not None:
            with self.lock:result['previous_command']=result['q'].copy() if self.previous_command is None else self.previous_command.copy()
        return result

    def synchronized_observation(self):
        # Encoders are sampled at 200 Hz; both images refer to one render timestamp.
        with self.lock:
            if not self.joint_history or any(not f for f in self.frames.values()):return None
            pairs=[(min(ft,st),front,side) for ft,front in list(self.frames['front'])[-6:]
                   for st,side in list(self.frames['side'])[-6:] if abs(ft-st)<.002]
            if not pairs:return None
            stamp,front,side=max(pairs,key=lambda pair:pair[0])
            history=list(self.joint_history);times=np.array([v[0] for v in history])
            hi=min(len(history)-1,int(np.searchsorted(times,stamp)))
            lo=max(0,hi-1)
            weight=np.clip((stamp-times[lo])/max(times[hi]-times[lo],1e-9),0,1)
            q=history[lo][1]*(1-weight)+history[hi][1]*weight
            previous=min(self.frames['front'],key=lambda f:abs(f[0]-(stamp-.2)))
            commands=[target for t,target in self.command_history if t<=stamp+1e-6]
            return dict(front=front.copy(),side=side.copy(),previous_front=previous[1].copy(),q=q.copy(),
                previous_command=commands[-1].copy() if commands else q.copy(),
                sim_time=stamp,image_time=stamp,previous_image_time=previous[0],
                controller_time=self.sim_time,encoder_image_alignment=True)

    def _state(self,message):
        pos=dict(zip(message.name,message.position))
        if not all(n in pos for n in JOINT_NAMES):return
        q=np.array([pos[n] for n in JOINT_NAMES])
        if not np.isfinite(q).all():return
        with self.lock:
            self.positions=q;self.sim_time=stamp_seconds(message.header.stamp);self.updated_at=time.monotonic()
            self.joint_history.append((self.sim_time,q.copy()))
            if len(message.velocity)==len(message.name):
                vel=dict(zip(message.name,message.velocity))
                self.joint_velocities=np.array([vel[n] for n in JOINT_NAMES]);self.max_joint_velocity=np.maximum(self.max_joint_velocity,np.abs(self.joint_velocities))

    def _poses(self,message):
        for trans in message.transforms:
            if trans.child_frame_id not in ('floating_pen','floating_cap'):continue
            p,q=trans.transform.translation,trans.transform.rotation
            quat=np.array([q.x,q.y,q.z,q.w])
            if np.linalg.norm(quat)<1e-9:continue
            matrix=Rotation.from_quat(quat).as_matrix()
            with self.lock:self.objects[trans.child_frame_id]=dict(position=np.array([p.x,p.y,p.z]),
                quaternion=quat,rotation=matrix,axis=matrix[:,2],stamp=stamp_seconds(trans.header.stamp))

    def object_states(self):
        with self.lock:return {name:{k:v.copy() if isinstance(v,np.ndarray) else v for k,v in value.items()} for name,value in self.objects.items()}

    def _contact(self,side,pad,message):
        name='floating_pen' if side=='left' else 'floating_cap'
        hit=any(name in c.collision1.name or name in c.collision2.name for c in message.contacts)
        stamp=stamp_seconds(message.header.stamp)
        with self.lock:
            self.contacts[side][pad]=hit;self.contact_stamps[side][pad]=stamp
            if hit and len(self.contact_log)<1000:self.contact_log.append(dict(side=side,pad=pad,time=stamp))
    def bilateral(self,side):
        with self.lock:return all(self.contacts[side][p] and self.sim_time-self.contact_stamps[side][p]<.15 for p in ('fixed','moving'))
    def _display(self,msg):
        t=stamp_seconds(msg.header.stamp)
        if t-self.display_last<.19 or len(self.display_frames)>=350:return
        rgb=rgb_image(msg)
        if rgb is not None:
            self.display_frames.append((t,rgb));self.display_last=t

    def activate_controllers(self):
        for name in CONTROLLERS:
            self._controller_request(LoadController,'load_controller',LoadController.Request(name=name))
            self._controller_request(ConfigureController,'configure_controller',ConfigureController.Request(name=name))
        req=SwitchController.Request(activate_controllers=list(CONTROLLERS),strictness=SwitchController.Request.STRICT,
                                    activate_asap=True,timeout=Duration(sec=15))
        self._controller_request(SwitchController,'switch_controller',req)
        res=self._controller_request(ListControllers,'list_controllers',ListControllers.Request())
        self.controller_status=[dict(name=c.name,state=c.state,type=c.type) for c in res.controller]
        if any(not any(c['name']==n and c['state']=='active' for c in self.controller_status) for n in CONTROLLERS):
            raise RuntimeError('Bimanual controllers not active')
        deadline=time.monotonic()+15
        while any(p.get_subscription_count()!=1 for p in self.trajectory_publishers):
            if time.monotonic()>deadline:raise TimeoutError('Trajectory subscriber missing or duplicated')
            time.sleep(.03)
    def _controller_request(self,kind,name,request):
        client=self.node.create_client(kind,NAMESPACE+'/controller_manager/'+name)
        try:return self._request(client,request)
        finally:self.node.destroy_client(client)

    def command(self,q,dt=DT):return self.command_plan(np.asarray(q)[None,:],self.snapshot()[1],dt)
    def command_plan(self,positions,start_time,dt):
        positions=np.asarray(positions,dtype=float)
        if positions.ndim!=2 or positions.shape[1]!=12 or not 1<=len(positions)<=32 or not np.isfinite(positions).all():
            raise ValueError('Expected a finite [steps,12] chunk')
        q,now,_=self.snapshot()
        if q is None or dt<=0 or not np.isfinite([start_time,dt]).all():raise ValueError('Invalid trajectory timing')
        first=min(len(positions)-1,max(0,int((now-start_time+1e-9)/dt)))
        points=np.stack([bounded_action(p) for p in positions[first:]])
        # A gripper holding an object cannot reach its empty-jaw angle. Its
        # persistent position error must not slow down either arm's trajectory.
        controller_times=[]
        with self.lock:velocity=self.joint_velocities.copy()
        arm_timing_ids=arm_schedule_indices(points,q,velocity,ARM_INDICES,GRIPPER_INDICES)
        ns=round(now*1e9)
        for pub,ids in zip(self.trajectory_publishers,(ARM_INDICES,GRIPPER_INDICES)):
            timing_ids=arm_timing_ids if ids is ARM_INDICES else ids
            times=group_times(points,q,SPEEDS,timing_ids,dt);controller_times.append(times)
            msg=JointTrajectory(joint_names=[JOINT_NAMES[i] for i in ids])
            msg.header.stamp=Time(sec=ns//10**9,nanosec=ns%10**9)
            for p,t in zip(points,times):
                n=round(t*1e9);msg.points.append(JointTrajectoryPoint(positions=p[ids].tolist(),time_from_start=Duration(sec=n//10**9,nanosec=n%10**9)))
            pub.publish(msg)
        with self.lock:
            self.previous_command=points[0].copy()
            self.command_history.append((now,points[0].copy()))
        self.trajectory_count+=1;self.last_trajectory=dict(points=points.tolist(),times=controller_times[0],gripper_times=controller_times[1],stamp=now,timing='active jaw synchronization; blocked-grip residual excluded from arm timing')
        return points[0]
    def reset_object(self,name,position,quaternion):
        req=SetEntityPose.Request();req.entity.name=name;req.entity.type=Entity.MODEL
        req.pose.position.x,req.pose.position.y,req.pose.position.z=map(float,position)
        req.pose.orientation.x,req.pose.orientation.y,req.pose.orientation.z,req.pose.orientation.w=map(float,quaternion)
        self._request(self.pose_client,req)
    def close(self):
        (self.output/'ros_transport.json').write_text(json.dumps(dict(transport='ROS 2 Jazzy',received=self.counts,
            trajectory_count=self.trajectory_count,controllers=self.controller_status,command_topics=list(TOPICS),
            max_joint_velocity=self.max_joint_velocity.tolist(),policy_axes=12),indent=2))
        (self.output/'contacts.json').write_text(json.dumps(self.contact_log,indent=2))
        self.executor.shutdown(timeout_sec=2);self.thread.join(timeout=2);self.node.destroy_node();self.context.try_shutdown()
        from run_demo import stop_process
        stop_process(self.bridge)
        if self.bridge_log:self.bridge_log.close()
