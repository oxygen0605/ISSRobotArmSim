#!/usr/bin/env python3
"""Collect corrective RGB-only demonstrations with expert/learner mixed behavior."""
from pathlib import Path
import argparse,json,time,traceback
import numpy as np
from PIL import Image
from iss_capture.config import ROOT
from iss_capture.scene_training_cases import scenario
from iss_capture.scene_teacher import SceneTeacher
from iss_capture.task_scenes import SceneSession,SceneMonitor
from iss_capture.policy_loader import load_policy
from iss_capture.rgb_joint_policy import select_inputs

def serializable(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--count",type=int,default=4);p.add_argument("--seed",type=int,default=5100000)
    p.add_argument("--tasks",nargs="+",default=["task2","task3"],choices=["task1","task2","task3"])
    p.add_argument("--teacher-probability",type=float,default=.85);p.add_argument("--max-steps",type=int,default=240)
    p.add_argument("--post-grasp-teacher-probability",type=float)
    p.add_argument("--force-expert-grippers",action="store_true")
    p.add_argument("--teacher-tail",type=int,default=0)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    policy=load_policy(a.checkpoint);assert getattr(policy,"rgb_only",False)
    summaries=[]
    for offset in range(a.count):
        for task in a.tasks:
            seed=a.seed+offset;out=a.output/f"train_{task}_{seed}"
            if out.exists():raise FileExistsError(out)
            out.mkdir();spec,case,nominal=scenario(task,seed,True);teacher=SceneTeacher(task);monitor=SceneMonitor(task)
            policy.reset();rng=np.random.default_rng(seed+int(task[-1]));rows=[];error=None;complete=None
            print("START "+json.dumps({"task":task,"seed":seed}),flush=True)
            try:
                with SceneSession(out,case,spec,use_depth=False) as session:
                    client=session.client;begin=client.snapshot()[1]
                    for index in range(a.max_steps):
                        obs=client.camera_observation();expert_goal=teacher.action(client)
                        expert=teacher.codec.decode(expert_goal,obs["q"])
                        probability=a.teacher_probability
                        if a.post_grasp_teacher_probability is not None:
                            probability=1. if teacher.phase=="grasp" else a.post_grasp_teacher_probability
                        if a.teacher_tail and index>=a.max_steps-a.teacher_tail:probability=1.
                        use_teacher=bool(rng.random()<probability)
                        if not use_teacher or policy.config.get("joint_history") or hasattr(policy,"temporal"):
                            learned,_=policy.predict(select_inputs(obs,policy.config.get('positions_only',False)),spec["instruction"])
                        else:learned=expert
                        command=(expert if use_teacher else learned).copy()
                        if a.force_expert_grippers:command[[5,11]]=expert[[5,11]]
                        frame=f"{index:05d}"
                        for name in ("front","side","previous_front"):
                            Image.fromarray(obs[name]).save(out/(frame+"_"+name+".png"))
                        rows.append({"frame":frame,"q":obs["q"],"q_velocity":obs["q_velocity"],
                            "previous_command":obs["previous_command"],"action":expert,
                            "instruction":spec["instruction"],"image_time":obs["image_time"],
                            "controller_time":obs["controller_time"],"sim_time":obs["sim_time"],
                            "behavior_action":command,"behavior_was_teacher":use_teacher,
                            "teacher_phase":teacher.phase})
                        client.command(command);time.sleep(.015);client.step(100);monitor.update(client)
                        if monitor.success and complete is None:complete=client.snapshot()[1]
                        if complete is not None and client.snapshot()[1]-complete>=.5:break
                        if index%40==0:print(json.dumps({"task":task,"seed":seed,"step":index,"phase":teacher.phase,"success":monitor.success}),flush=True)
                    elapsed=client.snapshot()[1]-begin
            except Exception as exception:
                error={"message":str(exception),"traceback":traceback.format_exc()};elapsed=None
            result={"episode":seed,"task":task,"split":"train","success":bool(monitor.success and error is None),
                "error":error,"frames":len(rows),"scene":spec,"case":case,"nominal_case":nominal,
                "teacher_labels":True,"behavior":"mixed expert and RGB-only learner",
                "teacher_probability":a.teacher_probability,"post_grasp_teacher_probability":a.post_grasp_teacher_probability,
                "expert_grippers_during_collection":a.force_expert_grippers,"teacher_tail_steps":a.teacher_tail,"depth_inputs":False,"simulated_seconds":elapsed,
                "final":monitor.last}
            (out/"samples.json").write_text(json.dumps(rows,default=serializable))
            (out/"result.json").write_text(json.dumps(result,default=serializable,indent=2))
            summaries.append(result);(a.output/"collection_summary.json").write_text(json.dumps(summaries,default=serializable,indent=2))
            print("RESULT "+json.dumps({k:result[k] for k in ["task","episode","success","frames","error"]}),flush=True)
if __name__=="__main__":main()
