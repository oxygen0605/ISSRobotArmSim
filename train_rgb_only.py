#!/usr/bin/env python3
"""Fine-tune one SmolVLA on RGB, raw joint encoders and language for all tasks."""
import argparse,hashlib,json,time,shutil,subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
from peft import PeftModel
from iss_capture.config import ROOT,URDF,JOINTS
from iss_capture.public_dataset import ensure_dataset,dataset_provenance
from iss_capture.vla_policy import load_base,new_adapter,base_model,make_batch,training_loss,BASE_REVISION,BASE_ID,PROCESSOR_REVISION
from iss_capture.rgb_joint_policy import VARIANT,load_demonstrations,observations,action_target,decode_action

def endpoint_actions(velocity, bias, scale):
    # LeRobot uses u_t = noise - action. With zero noise, t=1 and one
    # Euler step dt=-1, the normalized joint target is -predicted velocity.
    return -velocity[..., :12].float()*scale+bias


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=ROOT/"datasets/task_scenes_v1")
    p.add_argument("--output",type=Path,required=True);p.add_argument("--resume",type=Path)
    p.add_argument("--extra-data",type=Path,action="append",default=[])
    p.add_argument("--steps",type=int,default=1500);p.add_argument("--batch-size",type=int,default=3)
    p.add_argument("--accumulate",type=int,default=1);p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--eval-every",type=int,default=250);p.add_argument("--image-size",type=int,default=256,choices=[256,512])
    p.add_argument("--validation-frames-per-task",type=int,default=128,help="0 evaluates every validation frame")
    p.add_argument("--patience",type=int,default=0,help="Stop after this many validation checks without improvement; 0 disables")
    p.add_argument("--kinematic-weight",type=float,default=0.)
    p.add_argument("--positions-only",action="store_true")
    p.add_argument("--robust-observations",action="store_true")
    p.add_argument("--chunk-size",type=int,default=1)
    p.add_argument("--execution-steps",type=int,default=1)
    p.add_argument("--joint-history",action="store_true")
    p.add_argument("--endpoint",action="store_true")
    p.add_argument("--action-mode",choices=["delta","absolute","arm_delta_grip_absolute","cartesian"],default="delta")
    a=p.parse_args()
    if a.kinematic_weight<0:p.error("kinematic-weight must be nonnegative")
    if a.kinematic_weight and (not a.endpoint or a.action_mode!="absolute"):p.error("Kinematic loss requires endpoint training with absolute joints")
    if a.positions_only and a.joint_history:p.error("Angle-only input uses recurrent history instead of explicit gripper history")
    if a.chunk_size>1 and a.action_mode!="absolute":p.error("Action chunks require absolute joint targets")
    if not 1<=a.execution_steps<=a.chunk_size:p.error("execution-steps must be within the chunk")
    if a.output.exists():p.error("Choose a new output directory")
    a.data=ensure_dataset(a.data);a.output.mkdir(parents=True)
    torch.manual_seed(20261029);rng=np.random.default_rng(20261029)
    space="cartesian" if a.action_mode=="cartesian" else "joint"
    rows=load_demonstrations(a.data,space,a.joint_history,a.positions_only)
    for extra in a.extra_data:rows.extend(load_demonstrations(extra,space,a.joint_history,a.positions_only))
    for name in dict.fromkeys(row["episode"] for row in rows):
        episode=[row for row in rows if row["episode"]==name]
        for i,row in enumerate(episode):
            row["future"]=np.stack([action_target(episode[min(i+j,len(episode)-1)],a.action_mode) for j in range(a.chunk_size)])
    train=[r for r in rows if r["split"]=="train"];validation=[r for r in rows if r["split"]=="validation"]
    tasks=("task1","task2","task3");groups={t:[r for r in train if r["task"]==t] for t in tasks}
    assert train and validation and all(groups.values())
    phase_groups={task:{phase:[r for r in group if r["phase"]==phase] for phase in sorted({r["phase"] for r in group})} for task,group in groups.items()}
    states=np.stack([r["state"] for r in train]);targets=np.stack([action_target(r,a.action_mode) for r in train])
    floor=np.full(12,.1) if a.positions_only else np.r_[np.full(12,.1),np.full(12,.05)]
    if a.joint_history:floor=np.r_[floor,np.full(8,.1)]
    sb=states.mean(0);ss=np.maximum(states.std(0),floor)
    ab=targets.mean(0);asc=np.maximum(targets.std(0),.03)
    if a.resume:
        old=json.loads((a.resume/"capture_config.json").read_text())
        assert old["policy_variant"]==VARIANT
        dim=min(len(old["state_bias"]),len(sb));sb[:dim]=np.array(old["state_bias"])[:dim];ss[:dim]=np.array(old["state_scale"])[:dim]
        if old.get("action_mode","delta")==a.action_mode:
            ab=np.array(old["action_bias"]);asc=np.array(old["action_scale"])
    sb,ss,ab,asc=[np.asarray(v,np.float32) for v in (sb,ss,ab,asc)]
    base=load_base(joint_count=12,state_count=len(sb),action_count=12,policy_image_size=a.image_size,chunk_size=a.chunk_size)
    policy=PeftModel.from_pretrained(base,a.resume,is_trainable=True) if a.resume else new_adapter(base,vision_lora_layers=2)
    base=base_model(policy);base.config.num_steps=base.model.config.num_steps=1 if a.endpoint else 4
    if a.resume:
        base.config.freeze_vision_encoder=False;base.config.train_expert_only=False
        base.model.vlm_with_expert.freeze_vision_encoder=False;base.model.vlm_with_expert.train_expert_only=False
    if a.resume and len(old["state_bias"])<len(sb):
        with torch.no_grad():base.model.state_proj.modules_to_save.default.weight[:,len(old["state_bias"]):].zero_()
    if a.resume and len(old["state_bias"])>len(sb):
        projection=base.model.state_proj.modules_to_save.default
        missing=-np.asarray(old["state_bias"][len(sb):])/np.asarray(old["state_scale"][len(sb):])
        with torch.no_grad():projection.bias.add_(projection.weight[:,len(sb):len(old["state_bias"])]@torch.tensor(missing,device="cuda",dtype=projection.weight.dtype))
    captured={};geometry=None
    if a.kinematic_weight:
        from iss_capture.capping_kinematic_loss import KinematicLoss
        geometry=KinematicLoss()
        base.model.action_out_proj.register_forward_hook(lambda module,inputs,output:captured.update(velocity=output))
    action_bias=torch.tensor(ab,device="cuda");action_scale=torch.tensor(asc,device="cuda")
    tokenizer=base.model.vlm_with_expert.processor.tokenizer
    opt=torch.optim.AdamW([v for v in policy.parameters() if v.requires_grad],lr=a.lr,weight_decay=1e-4)
    root=ET.parse(URDF).getroot();limits={}
    for joint in root.findall("joint"):
        limit=joint.find("limit")
        if limit is not None and joint.get("name") in JOINTS:
            limits[joint.get("name")]=(float(limit.get("lower")),float(limit.get("upper")))
    lower=[limits[n][0] for n in JOINTS]*2;upper=[limits[n][1] for n in JOINTS]*2
    metadata=dict(policy_variant=VARIANT,positions_only=a.positions_only,robust_observations=a.robust_observations,input_modalities=["rgb","joint_position","language"] if a.positions_only else ["rgb","joint_position","joint_velocity","language"],
        uses_depth=False,uses_geometry_features=False,teacher_at_inference=False,task_specific_model_routing=False,
        base_model=BASE_ID,base_revision=BASE_REVISION,processor_revision=PROCESSOR_REVISION,
        training_code_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        training_source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ("train_rgb_only.py","iss_capture/rgb_joint_policy.py","iss_capture/vla_policy.py")},
        resume_checkpoint=str(a.resume) if a.resume else None,resume_adapter_sha256=old["adapter_sha256"] if a.resume else None,
        chunk_size=a.chunk_size,execution_steps=a.execution_steps,policy_image_size=a.image_size,denoising_steps=1 if a.endpoint else 4,state_dim=len(sb),action_dim=12,joint_history=a.joint_history,
        state_bias=sb.tolist(),state_scale=ss.tolist(),action_bias=ab.tolist(),action_scale=asc.tolist(),
        joint_lower=lower,joint_upper=upper,joint_order=list(JOINTS)*2,training_dataset=dataset_provenance(a.data),
        action_mode=a.action_mode,action_representation=a.action_mode,
        gripper_values=np.unique(np.round(np.stack([r["action"] for r in train])[:,[5,11]],6)).tolist(),
        training_method="deterministic endpoint" if a.endpoint else "SmolVLA flow matching",
        kinematic_loss_weight=a.kinematic_weight,kinematics_used_at_inference=a.action_mode=="cartesian",
        train_episodes=sorted({r["episode"] for r in train}),validation_episodes=sorted({r["episode"] for r in validation}),
        phase_balanced_sampling=True,train_frames=len(train),validation_frames=len(validation),
        train_episode_counts={t:len({r["episode"] for r in groups[t]}) for t in tasks},
        validation_episode_counts={t:len({r["episode"] for r in validation if r["task"]==t}) for t in tasks},
        optimization=dict(steps=a.steps,batch_size=a.batch_size,accumulate=a.accumulate,lr=a.lr,seed=20261029,patience=a.patience),
        validation_frames_per_task=a.validation_frames_per_task,extra_data=[str(p) for p in a.extra_data],test_used_for_training_or_selection=False)
    selected={t:[r for r in validation if r["task"]==t] for t in tasks}
    selected={t:([v[i] for i in np.linspace(0,len(v)-1,min(a.validation_frames_per_task,len(v))).astype(int)] if a.validation_frames_per_task else v) for t,v in selected.items()}
    best=float("inf");history=[];started=time.monotonic();rounds_without_improvement=0
    axis=torch.tensor([3,3,3,3,3,3,1,1,1,1,2,2] if a.action_mode=="cartesian" else [1,1,1,1,1,2,1,1,1,1,1,2],device="cuda",dtype=torch.float32)
    def validate(step):
        nonlocal best,rounds_without_improvement
        policy.eval();metrics={}
        with torch.inference_mode():
            for task,examples in selected.items():
                predicted=[]
                for start in range(0,len(examples),a.batch_size):
                    batch_rows=examples[start:start+a.batch_size]
                    batch=make_batch(observations(batch_rows,a.image_size),[r["instruction"] for r in batch_rows],tokenizer,"cuda",ss,sb)
                    with torch.autocast("cuda",dtype=torch.bfloat16):
                        y=base.predict_action_chunk(batch,noise=torch.zeros((len(batch_rows),a.chunk_size,32),device="cuda"))[:,0]
                    predicted.extend(y.float().cpu().numpy()*asc+ab)
                if a.action_mode=="cartesian":
                    command=np.stack(predicted);truth=np.stack([row["pose"] for row in examples]);error=np.abs(command-truth)
                    metrics[task]={"position_mae_m":float(error[:,:6].mean()),"axis_mae":float(error[:,6:10].mean()),
                        "gripper_mae_rad":float(error[:,10:12].mean()),
                        "gripper_class_error":float(((command[:,10:12]<.19)!=(truth[:,10:12]<.19)).mean())}
                else:
                    command=np.stack([decode_action(value,row["q"],a.action_mode) for value,row in zip(predicted,examples)])
                    truth=np.stack([row["action"] for row in examples]);error=np.abs(command-truth)
                    metrics[task]={"joint_mae_rad":float(error[:,[0,1,2,3,4,6,7,8,9,10]].mean()),
                        "gripper_mae_rad":float(error[:,[5,11]].mean()),
                        "gripper_class_error":float(((command[:,[5,11]]<.19)!=(truth[:,[5,11]]<.19)).mean())}
                if geometry is not None:
                    metrics[task]["kinematic_loss"]=float(geometry(torch.tensor(command,device="cuda",dtype=torch.float32),torch.tensor(truth,device="cuda",dtype=torch.float32)))
        score=np.mean([v["position_mae_m"]+.02*v["axis_mae"]+.015*v["gripper_class_error"] if a.action_mode=="cartesian"
                       else v["joint_mae_rad"]+.25*v["gripper_mae_rad"]+.05*v["gripper_class_error"]+a.kinematic_weight*v.get("kinematic_loss",0.) for v in metrics.values()])
        record=dict(step=step,score=float(score),tasks=metrics,seconds=time.monotonic()-started)
        history.append(record);print("VALIDATION "+json.dumps(record),flush=True)
        dest=a.output/f"step_{step}";dest.mkdir();policy.save_pretrained(dest)
        cfg={**metadata,"selected_step":step,"validation":record,"adapter_sha256":hashlib.sha256((dest/"adapter_model.safetensors").read_bytes()).hexdigest()}
        (dest/"capture_config.json").write_text(json.dumps(cfg,indent=2))
        if score<best:
            best=score;rounds_without_improvement=0
            if (a.output/"best").exists():shutil.rmtree(a.output/"best")
            shutil.copytree(dest,a.output/"best")
        else:rounds_without_improvement+=1
        (a.output/"validation_history.json").write_text(json.dumps(history,indent=2))
        policy.train()
    print("DATA "+json.dumps({"train":len(train),"validation":len(validation),"train_episodes":len(metadata["train_episodes"]),"validation_episodes":len(metadata["validation_episodes"]),"input":"3 RGB + raw q/dq + language only"}),flush=True)
    policy.train()
    for step in range(1,a.steps+1):
        opt.zero_grad(set_to_none=True);loss_value=0.
        for micro in range(a.accumulate):
            chosen=[]
            for j in range(a.batch_size):
                task=tasks[(step*a.batch_size+j)%3];pool=groups[task]
                if rng.random()<.65:
                    phase=rng.choice(list(phase_groups[task]));pool=phase_groups[task][phase]
                chosen.append(pool[rng.integers(len(pool))])
            obs=observations(chosen,a.image_size)
            if a.robust_observations:
                for sample in obs:
                    sample["q"]=sample["q"].copy();sample["q"][:12]+=rng.normal(0,.0005,12)
                    if not a.positions_only:
                        sample["q"][12:24]+=rng.normal(0,.02,12)
                        if rng.random()<.25:sample["q"][12:24]=0.
                    brightness=rng.uniform(.98,1.02)
                    for name in ("front","side","previous_front"):
                        sample[name]=np.clip(sample[name].astype(np.float32)*brightness,0,255).astype(np.uint8)
            batch=make_batch(obs,[r["instruction"] for r in chosen],tokenizer,"cuda",ss,sb)
            batch["action"]=torch.tensor((np.stack([r["future"] for r in chosen])-ab)/asc,device="cuda",dtype=torch.float32)
            with torch.autocast("cuda",dtype=torch.bfloat16):loss=training_loss(policy,batch,axis,deterministic_endpoint=a.endpoint)/a.accumulate
            if geometry is not None:
                predicted=endpoint_actions(captured["velocity"],action_bias,action_scale)
                target=batch["action"]*action_scale+action_bias
                loss+=a.kinematic_weight*geometry(predicted.reshape(-1,12),target.reshape(-1,12))/a.accumulate
            if not torch.isfinite(loss):raise RuntimeError("Non-finite training loss")
            loss.backward();loss_value+=float(loss.detach())
        torch.nn.utils.clip_grad_norm_(policy.parameters(),1.);opt.step()
        if step%20==0:
            print(json.dumps({"step":step,"loss":loss_value,"seconds":time.monotonic()-started,"gpu_allocated_MB":torch.cuda.max_memory_allocated()/1e6}),flush=True)
        if step%a.eval_every==0 or step==a.steps:
            validate(step)
            if a.patience and rounds_without_improvement>=a.patience:
                print("EARLY_STOP "+json.dumps(dict(step=step,patience=a.patience)),flush=True)
                break
    print("DONE "+str(a.output/"best"),flush=True)
if __name__=="__main__":main()
