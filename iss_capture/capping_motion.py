"""One initial impulse; no object velocity/pose commands after task begins."""
import json,time
import numpy as np
from scipy.spatial.transform import Rotation
from ros_gz_interfaces.msg import EntityWrench,Entity
from .floating_pen import MASS,LENGTH,RADIUS

def start_motion(client,case,output):
    v=np.asarray(case.get('linear_velocity',[0,0,0]),float)
    w=np.asarray(case.get('angular_velocity',[0,0,0]),float)
    if not np.any(v) and not np.any(w):return
    deadline=time.monotonic()+10
    while not client.impulse_publisher.get_subscription_count():
        if time.monotonic()>deadline:raise TimeoutError('Initial impulse bridge not ready')
        time.sleep(.02)
    initial=client.object_states()['floating_pen'];rotation=initial['rotation']
    inertia=rotation@np.diag([MASS*(3*RADIUS**2+LENGTH**2)/12]*2+[MASS*RADIUS**2/2])@rotation.T
    req=EntityWrench();req.entity.name='floating_pen';req.entity.type=Entity.MODEL
    req.wrench.force.x,req.wrench.force.y,req.wrench.force.z=map(float,MASS*v/.001)
    req.wrench.torque.x,req.wrench.torque.y,req.wrench.torque.z=map(float,inertia@w/.001)
    client.impulse_publisher.publish(req);time.sleep(.05);client.step(1)
    for _ in range(2):
        client.step(100)
        if hasattr(client,'capture_publisher'):client.camera_observation()
    time.sleep(.04)
    final=client.object_states()['floating_pen'];dt=final['stamp']-initial['stamp']
    measured_v=(final['position']-initial['position'])/max(dt,1e-9)
    measured_w=Rotation.from_matrix(final['rotation']@rotation.T).as_rotvec()/max(dt,1e-9)
    audit=dict(method='single 1 ms wrench impulse before policy start',requested_linear_m_s=v.tolist(),
        requested_angular_rad_s=w.tolist(),measured_linear_m_s=measured_v.tolist(),measured_angular_rad_s=measured_w.tolist(),
        observation_interval_s=dt,subsequent_object_forces=False)
    (output/'initial_motion.json').write_text(json.dumps(audit,indent=2))
    if dt<.1 or np.linalg.norm(measured_w-w)>.04+np.linalg.norm(w)*.15 or np.linalg.norm(measured_v-v)>.001:
        raise RuntimeError('Initial motion did not match request: '+json.dumps(audit))
