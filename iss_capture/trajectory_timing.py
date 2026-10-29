"""Independent timing for native arm and gripper trajectory controllers."""
import numpy as np
def group_times(points,current,speeds,indices,dt):
    points=np.asarray(points,float);current=np.asarray(current,float);speeds=np.asarray(speeds,float);ids=np.asarray(indices,int)
    if dt<=0 or np.any(speeds[ids]<=0):raise ValueError('Positive control interval and joint speeds required')
    elapsed=.02;previous=current;times=[]
    for point in points:
        elapsed+=max(float(dt),float(np.max(np.abs(point[ids]-previous[ids])/speeds[ids])))
        times.append(elapsed);previous=point
    return times

def arm_schedule_indices(points,current,velocity,arms,grippers):
    """Synchronize active jaw motion, but never wait for blocked grasping fingers."""
    points=np.asarray(points,float);q=np.asarray(current,float);velocity=np.asarray(velocity,float)
    active=[]
    for i in grippers:
        holding=(-.15<q[i]<.2 and abs(velocity[i])<.05 and bool(np.all(points[:,i]<.19)))
        if not holding:active.append(i)
    return list(arms)+active
