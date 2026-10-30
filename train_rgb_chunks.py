#!/usr/bin/env python3
"""Train RGB SmolVLA action chunks with warmup/cosine LR and resumable milestones."""
import argparse,hashlib,json,time,shutil,subprocess,os
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
from peft import PeftModel
from iss_capture.config import ROOT,URDF,JOINTS
from iss_capture.public_dataset import ensure_dataset,dataset_provenance
from iss_capture.vla_policy import load_base,new_adapter,base_model,make_batch,training_loss,BASE_REVISION,BASE_ID,PROCESSOR_REVISION
from iss_capture.rgb_joint_policy import VARIANT,load_demonstrations,observations,action_target,decode_action
from iss_capture.rgb_training_schedule import learning_rate,batch_layout,task_index,attach_future_targets,masked_mean

def chunk_training_loss(policy,batch,axis,endpoint):
    base=base_model(policy)
    images,masks=base.prepare_images(batch);actions=base.prepare_action(batch)
    losses=base.model(images,masks,batch['observation.language.tokens'],batch['observation.language.attention_mask'],base.prepare_state(batch),actions,
        noise=torch.zeros_like(actions) if endpoint else None,
        time=torch.ones(actions.shape[0],device=actions.device) if endpoint else None)[...,:12]
    weights=axis/axis.mean()
    valid=batch['action_valid'].unsqueeze(-1)
    return (losses*weights*valid).sum()/(valid.sum()*losses.shape[-1])

def save_json(path, value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(path)

def endpoint_actions(velocity, bias, scale):
    # LeRobot uses u_t = noise - action. With zero noise, t=1 and one
    # Euler step dt=-1, the normalized joint target is -predicted velocity.
    return -velocity[..., :12].float()*scale+bias


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=ROOT/"datasets/task_scenes_v1")
    p.add_argument("--output",type=Path,required=True);p.add_argument("--resume",type=Path)
    p.add_argument("--extra-data",type=Path,action="append",default=[])
    p.add_argument("--steps",type=int,default=30000);p.add_argument("--batch-size",type=int,default=64,help="Effective batch per optimizer update")
    p.add_argument("--micro-batch-size",type=int,default=16)
    p.add_argument("--lr",type=float,default=3e-4);p.add_argument("--final-lr",type=float,default=3e-5)
    p.add_argument("--warmup-steps",type=int,default=1000)
    p.add_argument("--save-steps",type=int,nargs='+',default=[10000,20000,30000])
    p.add_argument("--continue-training",type=Path,help="Resume our own checkpoint including Adam/RNG and optimizer step")
    p.add_argument("--profile-steps",type=int,default=0,help="GPU preflight only; discard weights after this many updates")
    p.add_argument("--cache-images",action='store_true',help="Cache resized RGB in host RAM")
    p.add_argument("--eval-every",type=int,default=1000);p.add_argument("--image-size",type=int,default=256,choices=[256,512])
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
    try:
        micro_batch,accumulation=batch_layout(a.batch_size,a.micro_batch_size)
        learning_rate(1,a.steps,a.warmup_steps,a.lr,a.final_lr)
    except ValueError as error:p.error(str(error))
    if a.steps < 1 or a.eval_every < 1 or a.profile_steps < 0:p.error('Invalid step count')
    if any(s<1 or s>a.steps for s in a.save_steps):p.error('save-steps must be within the optimizer update budget')
    if a.patience:p.error('This fixed-budget trainer disables early stopping')
    continuing=a.continue_training
    original_args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    if continuing:
        a.resume=continuing
        if not (continuing/'training_state.pt').is_file():p.error('Full training state is missing')
    if a.kinematic_weight<0:p.error("kinematic-weight must be nonnegative")
    if a.kinematic_weight and (not a.endpoint or a.action_mode!="absolute"):p.error("Kinematic loss requires endpoint training with absolute joints")
    if a.positions_only and a.joint_history:p.error("Angle-only input uses recurrent history instead of explicit gripper history")
    if a.chunk_size>1 and a.action_mode!="absolute":p.error("Action chunks require absolute joint targets")
    if not 1<=a.execution_steps<=a.chunk_size:p.error("execution-steps must be within the chunk")
    if a.output.exists() and not continuing:p.error("Choose a new output directory")
    a.data=ensure_dataset(a.data);a.output.mkdir(parents=True,exist_ok=bool(continuing))
    torch.manual_seed(20261029);rng=np.random.default_rng(20261029)
    space="cartesian" if a.action_mode=="cartesian" else "joint"
    rows=load_demonstrations(a.data,space,a.joint_history,a.positions_only)
    for extra in a.extra_data:rows.extend(load_demonstrations(extra,space,a.joint_history,a.positions_only))
    attach_future_targets(rows,a.chunk_size,lambda row:action_target(row,a.action_mode))
    train=[r for r in rows if r["split"]=="train"];validation=[r for r in rows if r["split"]=="validation"]
    if a.cache_images:
        from iss_capture import rgb_joint_policy as rgb_module
        image_paths=sorted({p for row in rows for p in row['paths'].values()})
        rgb_module.read_rgb=lru_cache(maxsize=len(image_paths))(rgb_module.read_rgb.__wrapped__)
        print('CACHE_START '+json.dumps(dict(images=len(image_paths),bytes=len(image_paths)*a.image_size*a.image_size*3)),flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for index,_ in enumerate(pool.map(lambda path:rgb_module.read_rgb(path,a.image_size),image_paths),1):
                if index%5000==0:print('CACHE_PROGRESS '+str(index),flush=True)
        print('CACHE_READY',flush=True)
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
        training_source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ("train_rgb_chunks.py","iss_capture/rgb_training_schedule.py","iss_capture/rgb_joint_policy.py","iss_capture/vla_policy.py")},
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
        optimization=dict(steps=a.steps,batch_size=a.batch_size,effective_batch_size=a.batch_size,micro_batch_size=micro_batch,accumulate=accumulation,lr=a.lr,final_lr=a.final_lr,warmup_steps=a.warmup_steps,lr_schedule='linear_warmup_cosine_decay',optimizer='AdamW',weight_decay=1e-4,gradient_clip_norm=1.,seed=20261029,patience=0),
        checkpoint_steps=a.save_steps,chunk_end_padding='repeat_last_for_storage; padded_targets_masked_from_loss_and_validation',validation_score_horizon=a.execution_steps,
        dataset_manifest_sha256=hashlib.sha256((a.data/'selection_manifest.json').read_bytes()).hexdigest(),
        validation_frames_per_task=a.validation_frames_per_task,extra_data=[str(p) for p in a.extra_data],test_used_for_training_or_selection=False)
    selected={t:[r for r in validation if r["task"]==t] for t in tasks}
    selected={t:([v[i] for i in np.linspace(0,len(v)-1,min(a.validation_frames_per_task,len(v))).astype(int)] if a.validation_frames_per_task else v) for t,v in selected.items()}
    best=float("inf");history=[];started=time.monotonic();first_step=1;previous_seconds=0.
    if continuing:
        state=torch.load(continuing/'training_state.pt',map_location='cpu',weights_only=False)
        # This file is produced by this trainer locally, never an untrusted Hub file.
        previous=state['metadata']
        for key in ['optimization','chunk_size','execution_steps','dataset_manifest_sha256','training_method','kinematic_loss_weight','robust_observations','training_source_sha256']:
            if metadata[key]!=previous[key]:raise ValueError('Resume configuration changed: '+key)
        opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng']);rng.bit_generator.state=state['numpy_rng']
        first_step=state['step']+1;best=state['best_score'];history=state['history'];previous_seconds=state['elapsed_seconds']
        metadata=previous
    save_json(a.output/'training_config.json',metadata)
    save_json(a.output/'invocation.json',original_args)
    axis=torch.tensor([3,3,3,3,3,3,1,1,1,1,2,2] if a.action_mode=="cartesian" else [1,1,1,1,1,2,1,1,1,1,1,2],device="cuda",dtype=torch.float32)
    def checkpoint(destination,step,record,full_state=False,replace=False):
        temporary=a.output/('.'+destination.name+'.writing')
        if temporary.exists():raise RuntimeError('Incomplete checkpoint exists: '+str(temporary))
        temporary.mkdir();policy.save_pretrained(temporary)
        cfg={**metadata,'selected_step':step,'validation':record,'adapter_sha256':hashlib.sha256((temporary/'adapter_model.safetensors').read_bytes()).hexdigest()}
        save_json(temporary/'capture_config.json',cfg)
        if full_state:
            torch.save(dict(step=step,optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=rng.bit_generator.state,metadata=metadata,best_score=best,history=history,elapsed_seconds=previous_seconds+time.monotonic()-started),temporary/'training_state.pt')
        if destination.exists():
            if not replace:raise FileExistsError(destination)
            backup=a.output/('.'+destination.name+'.previous')
            if backup.exists():shutil.rmtree(backup)
            destination.rename(backup);temporary.rename(destination);shutil.rmtree(backup)
        else:temporary.rename(destination)
    def validate(step):
        nonlocal best
        policy.eval();metrics={}
        with torch.inference_mode():
            for task,examples in selected.items():
                predicted=[]
                for start in range(0,len(examples),micro_batch):
                    batch_rows=examples[start:start+micro_batch]
                    batch=make_batch(observations(batch_rows,a.image_size),[r["instruction"] for r in batch_rows],tokenizer,"cuda",ss,sb)
                    with torch.autocast("cuda",dtype=torch.bfloat16):
                        y=base.predict_action_chunk(batch,noise=torch.zeros((len(batch_rows),a.chunk_size,32),device="cuda"))
                    predicted.extend(y.float().cpu().numpy()*asc+ab)
                sequence=np.stack(predicted);predicted=sequence[:,0]
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
                if a.action_mode=='absolute':
                    future=np.stack([row['future'] for row in examples]);error=np.abs(sequence-future)
                    valid=np.stack([row['future_valid'] for row in examples])
                    arms=[0,1,2,3,4,6,7,8,9,10];grippers=[5,11];h=a.execution_steps
                    metrics[task].update(chunk_joint_mae_rad=masked_mean(error[:,:,arms],valid),executed_chunk_joint_mae_rad=masked_mean(error[:,:h,arms],valid[:,:h]),executed_chunk_gripper_mae_rad=masked_mean(error[:,:h,grippers],valid[:,:h]),executed_chunk_gripper_class_error=masked_mean((sequence[:,:h,grippers]<.19)!=(future[:,:h,grippers]<.19),valid[:,:h]))
                    if geometry is not None:
                        metrics[task]['executed_chunk_kinematic_loss']=float(geometry(torch.tensor(sequence[:,:h][valid[:,:h]],device='cuda',dtype=torch.float32),torch.tensor(future[:,:h][valid[:,:h]],device='cuda',dtype=torch.float32)))
        score=np.mean([v["position_mae_m"]+.02*v["axis_mae"]+.015*v["gripper_class_error"] if a.action_mode=="cartesian"
                       else v.get('executed_chunk_joint_mae_rad',v['joint_mae_rad'])+.25*v.get('executed_chunk_gripper_mae_rad',v['gripper_mae_rad'])+.05*v.get('executed_chunk_gripper_class_error',v['gripper_class_error'])+a.kinematic_weight*v.get('executed_chunk_kinematic_loss',v.get('kinematic_loss',0.)) for v in metrics.values()])
        record=dict(step=step,score=float(score),tasks=metrics,seconds=previous_seconds+time.monotonic()-started,learning_rate=opt.param_groups[0]['lr'])
        history.append(record);print("VALIDATION "+json.dumps(record),flush=True)
        if score<best:
            best=score;checkpoint(a.output/'best',step,record,replace=True)
        if step in a.save_steps or step==a.steps:checkpoint(a.output/f'step_{step}',step,record,full_state=True)
        checkpoint(a.output/'latest',step,record,full_state=True,replace=True)
        save_json(a.output/'validation_history.json',history)
        policy.train()
    print("DATA "+json.dumps({"train":len(train),"validation":len(validation),"train_episodes":len(metadata["train_episodes"]),"validation_episodes":len(metadata["validation_episodes"]),"input":"3 RGB + raw q/dq + language only"}),flush=True)
    policy.train()
    for step in range(first_step,a.steps+1):
        lr=learning_rate(step,a.steps,a.warmup_steps,a.lr,a.final_lr)
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True);loss_value=0.
        chosen_all=[]
        for j in range(a.batch_size):
            task=tasks[task_index(step,a.batch_size,j)];pool=groups[task]
            if rng.random()<.65:
                phase=rng.choice(list(phase_groups[task]));pool=phase_groups[task][phase]
            chosen_all.append(pool[rng.integers(len(pool))])
        valid_total=sum(int(row['future_valid'].sum()) for row in chosen_all)
        for micro in range(accumulation):
            chosen=chosen_all[micro*micro_batch:(micro+1)*micro_batch]
            micro_weight=sum(int(row['future_valid'].sum()) for row in chosen)/valid_total
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
            batch['action_valid']=torch.tensor(np.stack([r['future_valid'] for r in chosen]),device='cuda',dtype=torch.bool)
            # Weight each microbatch by its valid target count, so accumulation
            # equals the same objective on the full effective batch.
            with torch.autocast("cuda",dtype=torch.bfloat16):loss=chunk_training_loss(policy,batch,axis,a.endpoint)*micro_weight
            if geometry is not None:
                predicted=endpoint_actions(captured["velocity"],action_bias,action_scale)
                target=batch["action"]*action_scale+action_bias
                valid=batch['action_valid']
                loss+=a.kinematic_weight*geometry(predicted[valid],target[valid])*micro_weight
            if not torch.isfinite(loss):raise RuntimeError("Non-finite training loss")
            loss.backward();loss_value+=float(loss.detach())
        norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step%20==0 or step==first_step or a.profile_steps:
            progress=dict(step=step,total_steps=a.steps,loss=loss_value,learning_rate=lr,gradient_norm=float(norm),seconds=previous_seconds+time.monotonic()-started,gpu_allocated_MB=torch.cuda.max_memory_allocated()/1e6,gpu_reserved_MB=torch.cuda.max_memory_reserved()/1e6,effective_batch_size=a.batch_size,micro_batch_size=micro_batch,accumulation=accumulation,complete=False)
            print(json.dumps(progress),flush=True);save_json(a.output/'progress.json',progress)
        if a.profile_steps and step-first_step+1>=a.profile_steps:
            print('PROFILE_COMPLETE',flush=True);return
        if step%a.eval_every==0 or step in a.save_steps or step==a.steps:
            validate(step)
    save_json(a.output/'progress.json',{**progress,'complete':True,'final_checkpoint':str(a.output/f'step_{a.steps}')})
    print("DONE "+str(a.output/f'step_{a.steps}'),flush=True)
if __name__=="__main__":main()
