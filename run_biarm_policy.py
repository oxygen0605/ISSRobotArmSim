#!/usr/bin/env python3
"""Run one loaded SmolVLA checkpoint for every task through ROS 2 controllers."""
import argparse,json,hashlib,time,sys
from pathlib import Path
import numpy as np
from PIL import Image
from iss_capture.unified_config import ROOT,TASKS,CHECKPOINT,case_from_seed
from iss_capture.policy_loader import load_policy as OnePolicy
from iss_capture.unified_session import Session
from iss_capture.unified_monitor import Monitor

def run(policy,output,task,case,gui=False,record=False,max_steps=220,linger=0,instruction=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);monitor=Monitor(task);trace=[];object_trace=[]
    instruction=instruction or TASKS[task];started=time.monotonic();policy.reset()
    with Session(out,case,gui,record) as session:
        client=session.client;begin=client.snapshot()[1]
        for index in range(max_steps):
            observation=client.camera_observation()
            if index==0:
                for name in ('front','side','previous_front'):Image.fromarray(observation[name]).save(out/f'initial_{name}.png')
            command,debug=policy.predict(observation,instruction)
            client.command(command);time.sleep(.015);client.step(100);metric=monitor.update(client)
            trace.append(dict(step=index,sim_time=observation['sim_time'],command=command.tolist(),q=observation['q'].tolist(),**debug))
            object_trace.append({name:{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in obj.items()} for name,obj in client.object_states().items()})
            if index%20==0:(out/'live_trace.json').write_text(json.dumps(trace))
            if index%20==0:print(json.dumps(dict(task=task,step=index,predicted_phase=(policy.config.get('phase_names',['unknown'])[int(np.argmax(debug['phase_probabilities']))] if 'phase_probabilities' in debug else 'unlabelled'),**metric)),flush=True)
            if monitor.success:break
        result=dict(task=task,success=monitor.success,instruction=instruction,case=case,steps=len(trace),
            checkpoint=str(policy.directory.resolve()),adapter_sha256=hashlib.sha256((policy.directory/'adapter_model.safetensors').read_bytes()).hexdigest(),
            checkpoint_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in policy.directory.iterdir() if p.is_file()},policy_instance=id(policy),teacher_used=False,task_specific_model_routing=False,final=monitor.last,
            wall_seconds=time.monotonic()-started,simulated_seconds=client.snapshot()[1]-begin)
        for name in ('front','side'):Image.fromarray(client.camera_observation()[name]).save(out/f'final_{name}.png')
        if record and client.display_frames:
            images=[Image.fromarray(im).resize((720,480)) for _,im in client.display_frames]
            images[0].save(out/'policy.gif',save_all=True,append_images=images[1:],duration=200,loop=0)
            images[0].save(out/'initial_overview.png');images[-1].save(out/'final_overview.png')
        (out/'result.json').write_text(json.dumps(result,indent=2));(out/'trace.json').write_text(json.dumps(trace,indent=2))
        (out/'metrics.json').write_text(json.dumps(monitor.samples,indent=2))
        (out/'evaluation_object_trace.json').write_text(json.dumps(object_trace,indent=2))
        print('RESULT '+json.dumps(result),flush=True)
        if gui and linger>0:time.sleep(min(linger,60))
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=[*TASKS,'all'],default='task1');p.add_argument('--seed',type=int,default=3300000)
    p.add_argument('--checkpoint',type=Path,default=CHECKPOINT);p.add_argument('--output',type=Path);p.add_argument('--gui',action='store_true')
    p.add_argument('--record',action='store_true');p.add_argument('--max-steps',type=int,default=220);p.add_argument('--linger',type=float,default=15)
    p.add_argument('--instruction')
    from rclpy.utilities import remove_ros_args
    a=p.parse_args(remove_ros_args(sys.argv)[1:])
    if a.task=='all' and a.instruction:p.error('--instruction requires one task for the evaluation monitor')
    policy=OnePolicy(a.checkpoint);out=a.output or ROOT/'runs'/time.strftime('one_policy_%Y%m%d_%H%M%S')
    results=[]
    for task in TASKS if a.task=='all' else [a.task]:
        if policy.config.get('policy_variant')=='shared_three_task_smolvla_task_scenes_v1':
            from iss_capture.task_scenes import load_scene
            from run_task_scene_preview import run as run_scene
            spec,case=load_scene(task,a.seed)
            if a.instruction:spec={**spec,'instruction':a.instruction}
            results.append(run_scene(policy,out/task,task,spec,case,max_steps=a.max_steps,
                                     gui=a.gui,record=a.record,linger=a.linger))
        else:
            results.append(run(policy,out/task,task,case_from_seed(a.seed),a.gui,a.record,a.max_steps,a.linger,a.instruction))
    (out/'summary.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
