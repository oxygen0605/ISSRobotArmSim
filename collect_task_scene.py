#!/usr/bin/env python3
"""Record measured RGB-D demonstrations in the requested task-specific scenes."""
import argparse, json, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image
from iss_capture.unified_config import ROOT
from iss_capture.task_scenes import SceneSession, SceneMonitor
from iss_capture.scene_training_cases import scenario, write_splits
from iss_capture.scene_teacher import SceneTeacher
from iss_capture.task_scene_observation import OptionalGeometry, OptionalWater, SceneRGB
from iss_capture.depth_geometry import RGBDFeatures
from iss_capture.workspace_features import enrich

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.tolist() if isinstance(x,np.ndarray) else x.item()))

def run(output, task, seed, split, max_steps=240, jitter=True):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    spec,case,nominal=scenario(task,seed,jitter)
    teacher=SceneTeacher(task);monitor=SceneMonitor(task)
    water=OptionalWater(spec,nominal);features=RGBDFeatures()
    features.rgb=SceneRGB(water);features.geometry=OptionalGeometry(spec,nominal)
    rows=[];started=time.monotonic();last=None;completed=None;error=None;termination=None
    try:
        with SceneSession(output,case,spec) as session:
            client=session.client;begin=client.snapshot()[1]
            for index in range(max_steps):
                obs=client.camera_observation();state=features(obs)
                stamp=obs['image_time'];dt=.1 if last is None else stamp-last;last=stamp
                full,rotation=enrich(state,obs['q'],obs['q_velocity'],dt,max(0.,obs['controller_time']-stamp),features.codec)
                goal=teacher.action(client);command=teacher.codec.decode(goal,obs['q'])
                frame=f'{index:05d}'
                for name in ('front','side','previous_front'):Image.fromarray(obs[name]).save(output/(frame+'_'+name+'.png'))
                np.savez_compressed(output/(frame+'_depth.npz'),front=obs['front_depth'],side=obs['side_depth'])
                rows.append(dict(frame=frame,q=obs['q'],previous_command=obs['previous_command'],action=command,goal=goal,
                    teacher_phase=teacher.phase,grasp_phases=dict(teacher.teacher.grasp),q_velocity=obs['q_velocity'],
                    controller_time=obs['controller_time'],instruction=spec['instruction'],sim_time=obs['sim_time'],image_time=stamp,
                    depth_state=state,head_state=full,rotation=rotation,depth_geometry=features.last_geometry,
                    water_position=water.position,evaluation_objects=client.object_states(),
                    contact_labels={side:client.bilateral(side) for side in ('left','right')},sponge_grasped=client.sponge_grasped()))
                client.command(command);time.sleep(.015);client.step(100);monitor.update(client)
                if index%20==0:
                    dump(output/'live_samples.json',rows[-3:])
                    print(json.dumps(dict(task=task,step=index,phase=teacher.phase,**monitor.last)),flush=True)
                if monitor.success and completed is None:completed=client.snapshot()[1]
                if completed is not None and client.snapshot()[1]-completed>=.6:
                    termination='success';break
                if index>=100 and teacher.phase=='grasp' and completed is None:
                    termination='teacher_grasp_timeout';break
            elapsed=client.snapshot()[1]-begin
    except Exception as exc:
        error=dict(message=str(exc),traceback=traceback.format_exc());elapsed=None
    result=dict(episode=seed,case=case,nominal_case=nominal,scene=spec,task=task,split=split,success=completed is not None and monitor.success and error is None,
                teacher_only=True,teacher_version=teacher.version,data_kind='actual_rgbd_teacher_rollout_task_scene',frames=len(rows),final=monitor.last,
                error=error,termination=termination or ('error' if error else 'step_limit'),wall_seconds=time.monotonic()-started,simulated_seconds=elapsed)
    dump(output/'samples.json',rows);dump(output/'result.json',result)
    print('RESULT '+json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['task1','task2','task3'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--split',choices=['train','validation','development'],default='development')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-steps',type=int,default=240)
    p.add_argument('--exact-layout',action='store_true');a=p.parse_args()
    write_splits();run(a.output,a.task,a.seed,a.split,a.max_steps,not a.exact_layout)
