#!/usr/bin/env python3
"""Train a shared learned temporal action head on RGB-only SmolVLA features."""
from pathlib import Path
import argparse,json,hashlib,time,shutil,gc
import numpy as np
import torch
from peft import PeftModel
from safetensors.torch import save_file
from iss_capture.config import ROOT
from iss_capture.vla_policy import load_base,base_model,make_batch
from iss_capture.rgb_joint_policy import load_demonstrations,observations
from iss_capture.scene_language import language_embedding
from iss_capture.rgb_temporal_policy import RGBTemporalHead,VARIANT,PHASES,phase_label

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--data",type=Path,default=ROOT/"datasets/task_scenes_v1")
    p.add_argument("--extra-data",type=Path,action="append",default=[])
    p.add_argument("--steps",type=int,default=6000);p.add_argument("--lr",type=float,default=0.0003)
    p.add_argument("--preserve-tasks",nargs="*",default=[])
    p.add_argument("--phase-end-goals",action="store_true")
    p.add_argument("--assembly-anchors",action="store_true")
    p.add_argument("--feature-cache",type=Path)
    p.add_argument("--resume-head",type=Path)
    p.add_argument("--kinematic-weight",type=float,default=0.)
    p.add_argument("--eval-every",type=int,default=500);p.add_argument("--batch-episodes",type=int,default=6)
    a=p.parse_args()
    if a.output.exists():p.error("Choose new output directory")
    a.output.mkdir(parents=True);cfg=json.loads((a.checkpoint/"capture_config.json").read_text())
    assert cfg["action_mode"]=="absolute" and cfg.get("chunk_size",1)==1
    torch.manual_seed(20261030);torch.set_num_threads(4);rng=np.random.default_rng(20261030)
    rows=load_demonstrations(a.data,joint_history=cfg.get("joint_history",False),positions_only=cfg.get("positions_only",False))
    for d in a.extra_data:rows+=load_demonstrations(d,joint_history=cfg.get("joint_history",False),positions_only=cfg.get("positions_only",False))
    sb=np.array(cfg["state_bias"],np.float32);ss=np.array(cfg["state_scale"],np.float32)
    ab=np.array(cfg["action_bias"],np.float32);asc=np.array(cfg["action_scale"],np.float32)
    started=time.monotonic()
    if a.feature_cache:
        cached_cfg=json.loads((a.feature_cache.parent/'best/capture_config.json').read_text())
        for key in ('adapter_sha256','state_dim','state_bias','state_scale','policy_image_size'):
            if cached_cfg[key]!=cfg[key]:raise ValueError("Feature cache encoder mismatch: "+key)
        expected={r['episode'] for r in rows}
        actual=set(cached_cfg['temporal_train_episodes']+cached_cfg['temporal_validation_episodes'])
        if expected!=actual:raise ValueError("Feature cache episode mismatch")
        saved=np.load(a.feature_cache,allow_pickle=False)
        hidden=saved['hidden'];language=saved['language'];core=saved['core']
        if len(hidden)!=len(rows) or core.shape!=(len(rows),12):raise ValueError("Feature cache row mismatch")
        shutil.copy2(a.feature_cache,a.output/'features.npz')
        print("REUSED ENCODER CACHE "+str(len(rows)),flush=True)
    else:
        policy=PeftModel.from_pretrained(load_base(joint_count=12,state_count=cfg["state_dim"],action_count=12,policy_image_size=cfg["policy_image_size"]),a.checkpoint).eval()
        base=base_model(policy);base.config.num_steps=base.model.config.num_steps=cfg["denoising_steps"]
        tokenizer=base.model.vlm_with_expert.processor.tokenizer;captured={}
        hook=base.model.action_out_proj.register_forward_pre_hook(lambda m,inputs:captured.update(hidden=inputs[0][:,0].detach().float()))
        hidden=[];language=[];core=[];started=time.monotonic()
        with torch.inference_mode():
            for i in range(0,len(rows),12):
                chunk=rows[i:i+12]
                batch=make_batch(observations(chunk,cfg["policy_image_size"]),[r["instruction"] for r in chunk],tokenizer,"cuda",ss,sb)
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    value=base.predict_action_chunk(batch,noise=torch.zeros((len(chunk),1,32),device="cuda"))[:,0]
                hidden.append(captured["hidden"].cpu().numpy());core.append(value.float().cpu().numpy())
                language.append(language_embedding(base,batch).cpu().numpy())
                if i%240==0:print("CACHE "+json.dumps({"frames":min(i+12,len(rows)),"total":len(rows),"seconds":time.monotonic()-started}),flush=True)
        hook.remove();hidden=np.concatenate(hidden);language=np.concatenate(language);core=np.concatenate(core)
        np.savez_compressed(a.output/"features.npz",hidden=hidden,language=language,core=core)
        del policy,base,captured,batch;gc.collect();torch.cuda.empty_cache()
    episodes=[]
    for name in dict.fromkeys(r["episode"] for r in rows):
        indices=[i for i,r in enumerate(rows) if r["episode"]==name]
        episodes.append({"name":name,"task":rows[indices[0]]["task"],"split":rows[indices[0]]["split"],"indices":indices,"phase_end_goals_valid":rows[indices[0]].get("phase_end_goals_valid",True)})
    training_actions=np.stack([r["action"] for r in rows])
    anchor_samples={phase:[] for phase in range(5,10)}
    if a.phase_end_goals or a.assembly_anchors:
        for episode in episodes:
            if episode["task"]!="task3" or not episode["phase_end_goals_valid"]:continue
            for phase in range(5,10):
                ids=[i for i in episode["indices"] if phase_label(rows[i]["task"],rows[i]["phase"])==phase]
                if not ids:continue
                endpoint=rows[ids[-1]]["action"]
                if episode["split"]=="train":anchor_samples[phase].append((endpoint-ab)/asc)
                if a.phase_end_goals:training_actions[ids]=endpoint
    arrays={
        "hidden":hidden,"language":language,"core":core,
        "state":(np.stack([r["state"] for r in rows])-sb)/ss,
        "target":(training_actions-ab)/asc,
        "phase":np.array([phase_label(r["task"],r["phase"]) for r in rows],np.int64)}
    arrays={k:torch.tensor(v,device="cuda",dtype=torch.long if k=="phase" else torch.float32) for k,v in arrays.items()}
    train={task:[e for e in episodes if e["split"]=="train" and e["task"]==task] for task in ("task1","task2","task3")}
    val=[e for e in episodes if e["split"]=="validation"]
    head=RGBTemporalHead(hidden.shape[1],len(sb),language.shape[1],assembly_anchors=a.assembly_anchors).to("cuda")
    if a.resume_head:
        from safetensors.torch import load_file
        missing,unexpected=head.load_state_dict(load_file(str(a.resume_head/"rgb_temporal_head.safetensors"),device="cuda"),strict=False)
        if unexpected or any(key!="anchors" for key in missing):raise ValueError("Incompatible temporal checkpoint")
    anchor_target=None
    if a.assembly_anchors:
        if any(not values for values in anchor_samples.values()):raise ValueError("Missing demonstrated assembly phase")
        anchor_target=torch.tensor(np.stack([np.mean(anchor_samples[phase],axis=0) for phase in range(5,10)]),device="cuda",dtype=torch.float32)
        with torch.no_grad():head.anchors.copy_(anchor_target)
    opt=torch.optim.AdamW(head.parameters(),lr=a.lr,weight_decay=1e-4)
    from iss_capture.capping_kinematic_loss import KinematicLoss
    geometry=KinematicLoss() if a.kinematic_weight else None
    best=float("inf");history=[]
    weights=torch.tensor([1,1,1,1,1,2,1,1,1,1,1,2],device="cuda")
    bias=torch.tensor(ab,device="cuda");scale=torch.tensor(asc,device="cuda")
    def validate(step):
        nonlocal best
        head.eval();metrics={};phase_correct=0;phase_count=0
        with torch.inference_mode():
            for task in train:
                errors=[];grips=[];mis=[]
                for episode in [e for e in val if e["task"]==task]:
                    idx=episode["indices"];x={k:v[idx][None] for k,v in arrays.items()}
                    prediction,_,logits=head(x["hidden"],x["state"],x["language"],x["core"])
                    actual=prediction[0]*scale+bias;truth=x["target"][0]*scale+bias
                    error=(actual-truth).abs();errors.append(error[:,[0,1,2,3,4,6,7,8,9,10]].mean().item())
                    grips.append(error[:,[5,11]].mean().item());mis.append(((actual[:,[5,11]]<.19)!=(truth[:,[5,11]]<.19)).float().mean().item())
                    phase_correct+=(logits[0].argmax(-1)==x["phase"][0]).sum().item();phase_count+=len(idx)
                metrics[task]={"joint_mae_rad":float(np.mean(errors)),"gripper_mae_rad":float(np.mean(grips)),"gripper_class_error":float(np.mean(mis))}
        score=np.mean([m["joint_mae_rad"]+.25*m["gripper_mae_rad"]+.05*m["gripper_class_error"] for m in metrics.values()])
        record={"step":step,"score":float(score),"tasks":metrics,"phase_accuracy":phase_correct/phase_count,"seconds":time.monotonic()-started}
        history.append(record);print("VALIDATION "+json.dumps(record),flush=True)
        dest=a.output/f"step_{step}";dest.mkdir()
        for name in ("adapter_config.json","adapter_model.safetensors"):shutil.copy2(a.checkpoint/name,dest/name)
        save_file(head.state_dict(),str(dest/"rgb_temporal_head.safetensors"))
        metadata={**cfg,"policy_variant":VARIANT,"hidden_dim":hidden.shape[1],"language_dim":language.shape[1],
            "temporal_width":192,"phase_end_goals":a.phase_end_goals,"assembly_anchors":a.assembly_anchors,
            "anchor_supervision":"training-episode phase-end joint targets only" if a.assembly_anchors else None,
            "phase_names":list(PHASES),"selected_temporal_step":step,
            "temporal_head_sha256":hashlib.sha256((dest/"rgb_temporal_head.safetensors").read_bytes()).hexdigest(),
            "validation":record,"temporal_train_episodes":[e["name"] for v in train.values() for e in v],
            "temporal_validation_episodes":[e["name"] for e in val],
            "temporal_training_method":"learned GRU residual joint-action head with auxiliary phase loss; no teacher forcing",
            "preserved_task_distillation":a.preserve_tasks,"kinematic_loss_weight":a.kinematic_weight,
            "input_modalities":cfg["input_modalities"],
            "depth_used":False,"geometry_features_used":False,"test_used_for_training_or_selection":False}
        (dest/"capture_config.json").write_text(json.dumps(metadata,indent=2))
        if score<best:
            best=score
            if (a.output/"best").exists():shutil.rmtree(a.output/"best")
            shutil.copytree(dest,a.output/"best")
        (a.output/"validation_history.json").write_text(json.dumps(history,indent=2));head.train()
    for step in range(1,a.steps+1):
        chosen=[train[task][rng.integers(len(train[task]))] for task in train for _ in range(max(1,a.batch_episodes//3))]
        lists=[]
        for e in chosen:
            idx=np.array(e["indices"])
            if rng.random()<.25:
                stride=float(rng.choice([.8,1.2]));idx=idx[np.floor(np.arange(0,len(idx),stride)).astype(int)]
            lists.append(idx)
        length=max(map(len,lists))
        index=torch.tensor(np.stack([np.pad(v,(0,length-len(v)),constant_values=v[-1]) for v in lists]),device="cuda")
        mask=torch.tensor(np.stack([np.arange(length)<len(v) for v in lists]),device="cuda",dtype=torch.float32)
        x={k:v[index] for k,v in arrays.items()}
        state=x["state"]+torch.randn_like(x["state"])*.005
        prediction,_,phase=head(torch.nn.functional.dropout(x["hidden"],p=.02,training=True),state,x["language"],x["core"])
        preservation=torch.tensor([episode["task"] in a.preserve_tasks for episode in chosen],device="cuda")[:,None,None]
        target=torch.where(preservation,x["core"].detach(),x["target"])
        loss=(((prediction-target).square()*weights).mean(-1)*mask).sum()/mask.sum()
        if a.preserve_tasks:
            loss+=3.*(((prediction-x["core"]).square().mean(-1))*mask*preservation[:,:,0]).sum()/mask.sum()
        loss+=.1*(torch.nn.functional.cross_entropy(phase.transpose(1,2),x["phase"],reduction="none")*mask).sum()/mask.sum()
        actual=prediction*scale+bias;truth=target*scale+bias
        if geometry is not None:
            valid=(mask>0)&(~preservation[:,:,0])
            if valid.any():loss+=a.kinematic_weight*geometry(actual[valid],truth[valid])
        loss+=.03*(torch.nn.functional.binary_cross_entropy_with_logits((.19-actual[:,:,[5,11]])*12,(truth[:,:,[5,11]]<.19).float(),reduction="none").mean(-1)*mask).sum()/mask.sum()
        if anchor_target is not None:loss+=.5*(head.anchors-anchor_target).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1);opt.step()
        if step%100==0:print(json.dumps({"step":step,"loss":float(loss.detach()),"seconds":time.monotonic()-started}),flush=True)
        if step%a.eval_every==0 or step==a.steps:validate(step)
    print("DONE "+str(a.output/"best"),flush=True)
if __name__=="__main__":main()
