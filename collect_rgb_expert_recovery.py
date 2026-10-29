#!/usr/bin/env python3
"""Query an expert on RGB learner states, then complete the rollout with the expert."""
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
    p.add_argument("--count",type=int,default=4);p.add_argument("--seed",type=int,default=5103000)
    p.add_argument("--learner-steps",type=int,default=100);p.add_argument("--max-steps",type=int,default=300)
    p.add_argument("--handoff-when-held",action="store_true")
    p.add_argument("--motion-seed",type=int)
    p.add_argument("--position-jitter",type=float)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);policy=load_policy(a.checkpoint)
    assert getattr(policy,"rgb_only",False)
    summaries=[]
    for offset in range(a.count):
        seed=a.seed+offset;out=a.output/f"train_task3_{seed}";out.mkdir(exist_ok=False)
        spec,case,nominal=scenario("task3",seed,a.position_jitter is None)
        if a.motion_seed is not None:
            _,motion,_=scenario("task3",a.motion_seed,False)
            for key in ("linear_velocity","angular_velocity","pen_tilt","cap_tilt"):
                case[key]=motion[key];nominal[key]=motion[key]
        if a.position_jitter is not None:
            rng=np.random.default_rng(seed)
            for key in ("pen_position","cap_position"):
                case[key]=(np.asarray(case[key])+rng.uniform(-a.position_jitter,a.position_jitter,3)).tolist()
        oracle=SceneTeacher("task3");monitor=SceneMonitor("task3")
        policy.reset();rows=[];error=None;done=None;takeover=None;held_since=None;termination=None
        print("START "+str(seed),flush=True)
        try:
            with SceneSession(out,case,spec,use_depth=False) as session:
                client=session.client;begin=client.snapshot()[1]
                for index in range(a.max_steps):
                    obs=client.camera_observation()
                    # The training expert accepts real grasps reached by the learner,
                    # even if they differ from the expert's preferred approach pose.
                    teacher=oracle.teacher;tcp=oracle.codec.positions(obs["q"])
                    if teacher.phase=="grasp":
                        for side,i in (("left",0),("right",1)):
                            if client.bilateral(side) and teacher.grasp[side] not in ("close","hold"):
                                teacher.grasp[side]="close";teacher.capture[side]=tcp[i].copy()
                                teacher.closed_since[side]=obs["sim_time"]
                    goal=oracle.action(client);expert=oracle.codec.decode(goal,obs["q"])
                    both_held=client.bilateral("left") and client.bilateral("right")
                    held_since=(obs["sim_time"] if held_since is None else held_since) if both_held else None
                    if a.handoff_when_held:
                        if takeover is None and held_since is not None and obs["sim_time"]-held_since>=.4:takeover=index
                        if takeover is None and index>=a.learner_steps:
                            termination="learner_grasp_timeout";break
                    elif takeover is None and index>=a.learner_steps:takeover=index
                    if takeover is None:
                        command,_=policy.predict(select_inputs(obs,policy.config.get('positions_only',False)),spec["instruction"])
                        behavior="rgb_learner"
                    else:
                        command=expert;behavior="expert_recovery"
                    frame=f"{index:05d}"
                    for name in ("front","side","previous_front"):
                        Image.fromarray(obs[name]).save(out/(frame+"_"+name+".png"))
                    rows.append({"frame":frame,"q":obs["q"],"q_velocity":obs["q_velocity"],
                        "previous_command":obs["previous_command"],"action":expert,
                        "instruction":spec["instruction"],"image_time":obs["image_time"],
                        "controller_time":obs["controller_time"],"sim_time":obs["sim_time"],
                        "teacher_phase":oracle.phase,"behavior_action":command,"behavior":behavior})
                    client.command(command);time.sleep(.015);client.step(100);monitor.update(client)
                    if monitor.success and done is None:done=client.snapshot()[1]
                    if done is not None and client.snapshot()[1]-done>=.5:break
                    if index%25==0:print(json.dumps({"seed":seed,"step":index,"teacher_phase":oracle.phase,"behavior":behavior,**monitor.last}),flush=True)
                elapsed=client.snapshot()[1]-begin
        except Exception as exception:
            error={"message":str(exception),"traceback":traceback.format_exc()};elapsed=None
        result={"episode":seed,"task":"task3","split":"train","success":bool(monitor.success and error is None),
            "error":error,"frames":len(rows),"scene":spec,"case":case,"nominal_case":nominal,
            "labels":"expert queried at every observed learner/expert state",
            "behavior":"RGB learner prefix, then privileged expert recovery for TRAINING ONLY",
            "teacher_takeover_step":takeover,"learner_checkpoint":str(a.checkpoint),
            "motion_seed":a.motion_seed,"position_jitter_m":a.position_jitter,"termination":termination,
            "depth_inputs":False,"simulated_seconds":elapsed,"final":monitor.last}
        (out/"samples.json").write_text(json.dumps(rows,default=serializable))
        (out/"result.json").write_text(json.dumps(result,default=serializable,indent=2))
        summaries.append(result);(a.output/"collection_summary.json").write_text(json.dumps(summaries,default=serializable,indent=2))
        print("RESULT "+json.dumps({k:result[k] for k in ["episode","success","frames","error"]}),flush=True)
if __name__=="__main__":main()
