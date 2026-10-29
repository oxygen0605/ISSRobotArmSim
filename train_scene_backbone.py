#!/usr/bin/env python3
"""One adapter trained on all three language tasks using SmolVLA flow matching."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from peft import PeftModel
from iss_capture.config import ROOT
from iss_capture.public_dataset import ensure_dataset,dataset_provenance
from iss_capture.vla_policy import load_base,new_adapter,base_model,make_batch,training_loss,BASE_REVISION,BASE_ID,PROCESSOR_REVISION
from iss_capture.unified_dataset import observations
from iss_capture.scene_dataset import load_workspace
TASKS={k:v['instruction'] for k,v in json.loads((ROOT/'config/task_scenes.json').read_text())['tasks'].items()}
def load(directory,representation):
    if representation!='world':raise ValueError('Scene demonstrations use independent world TCP goals')
    return load_workspace(directory)[0]
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'datasets/task_scenes_v1')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/task_scene_backbone_v1')
    p.add_argument('--steps',type=int,default=4000);p.add_argument('--batch-size',type=int,default=6)
    p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--eval-every',type=int,default=250)
    p.add_argument('--episode-manifest',type=Path)
    p.add_argument('--resume-from',type=Path);p.add_argument('--endpoint',action='store_true')
    p.add_argument('--representation',choices=('delta','goal','world'),default='world')
    a=p.parse_args();a.data=ensure_dataset(a.data);a.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(20261029);rng=np.random.default_rng(20261029);items=load(a.data,a.representation)
    if a.episode_manifest:
        requested={Path(name).name for name in json.loads(a.episode_manifest.read_text())}
        items=[r for r in items if Path(r['source']).name in requested]
        if {Path(r['source']).name for r in items}!=requested:raise ValueError('Episode manifest contains missing or unsuccessful episodes')
    train=np.array([i for i,r in enumerate(items) if r['split']=='train']);val=np.array([i for i,r in enumerate(items) if r['split']=='validation'])
    groups={t:np.array([i for i in train if items[i]['task']==t]) for t in TASKS}
    if not len(val) or any(not len(g) for g in groups.values()):raise ValueError('Every task needs training and held-out episodes')
    splits={s:{t:dict(episodes=len({items[i]['episode'] for i in ids if items[i]['task']==t}),frames=sum(items[i]['task']==t for i in ids)) for t in TASKS} for s,ids in [('train',train),('validation',val)]}
    print('DATA '+json.dumps(splits),flush=True)
    states=np.stack([items[i]['state'] for i in train])
    sb=states.mean(0).astype(np.float32);ss=np.maximum(states.std(0),np.r_[np.full(6,.01),np.full(6,.1),np.full(6,.01),np.full(2,.1),np.full(9,.01),np.full(3,.1)]).astype(np.float32)
    ab=np.r_[np.zeros(10),.19,.19].astype(np.float32);asc=np.r_[np.full(6,.03),np.full(4,.25),.36,.36].astype(np.float32)
    if a.representation in ('goal','world'):
        ab=np.array([.28,0,.24,0,0,.1,0,0,0,0,.19,.19],np.float32)
        asc=np.r_[np.full(6,.06),np.full(4,.25),.36,.36].astype(np.float32)
        if a.representation=='world':ab[3:6]=[.28,0,.32]
    if a.resume_from:
        old=json.loads((a.resume_from/'capture_config.json').read_text())
        # Refit normalization on training observations including absent objects.
        policy=PeftModel.from_pretrained(load_base(joint_count=12,state_count=32,action_count=12),a.resume_from,is_trainable=True)
    else:policy=new_adapter(load_base(joint_count=12,state_count=32,action_count=12),vision_lora_layers=2)
    base=base_model(policy)
    if a.resume_from and a.representation=='world' and old.get('representation','delta')!='world':
        if old.get('representation')!='goal' or old.get('denoising_steps')!=1:raise ValueError('World conversion requires endpoint pose-goal checkpoint')
        projection=base.model.action_out_proj.modules_to_save.default
        previous_bias=np.array(old['action_bias'])
        offset=(previous_bias[:3]+previous_bias[3:6]-ab[3:6])/asc[3:6]
        with torch.no_grad():
            projection.weight[3:6].add_(projection.weight[:3].clone())
            projection.bias[3:6].add_(projection.bias[:3].clone()-torch.tensor(offset,device='cuda',dtype=projection.bias.dtype))
        print('Exact endpoint output conversion: right-relative pose to independent right world pose',flush=True)
    base.config.freeze_vision_encoder=False;base.config.train_expert_only=False
    base.model.vlm_with_expert.freeze_vision_encoder=False;base.model.vlm_with_expert.train_expert_only=False
    base.config.num_steps=base.model.config.num_steps=1 if a.endpoint else 5
    tokenizer=base.model.vlm_with_expert.processor.tokenizer
    trainable=[v for v in policy.parameters() if v.requires_grad]
    opt=torch.optim.AdamW(trainable,lr=a.lr,weight_decay=1e-4);best=float('inf');history=[];started=time.monotonic()
    phasegroups={t:{phase:np.array([i for i in groups[t] if items[i]['phase']==phase]) for phase in {items[i]['phase'] for i in groups[t]}} for t in TASKS}
    selected={t:rng.choice([i for i in val if items[i]['task']==t],min(64,sum(items[i]['task']==t for i in val)),replace=False) for t in TASKS}
    weight=torch.tensor([2,2,2,3,3,3,1,1,1,1,2,2],dtype=torch.float32,device='cuda')
    metadata=dict(training_dataset=dataset_provenance(a.data),policy_variant='shared_three_task_smolvla_scene_backbone_v1',base_model=BASE_ID,base_revision=BASE_REVISION,processor_revision=PROCESSOR_REVISION,
        tasks=TASKS,joint_count=12,state_dim=32,action_dim=12,chunk_size=1,denoising_steps=1 if a.endpoint else 5,
        state_bias=sb.tolist(),state_scale=ss.tolist(),action_bias=ab.tolist(),action_scale=asc.tolist(),
        representation=a.representation,
        action_representation='left TCP XYZ, right-minus-left TCP XYZ, two axis XY, grippers; '+a.representation,
        one_shared_adapter=True,task_specific_action_heads=False,teacher_at_inference=False,geometry_assembly_guard=False,
        trainable_parameters=sum(v.numel() for v in trainable),vision_lora_layers=2,data_summary=splits,
        training_method='deterministic endpoint supervised SmolVLA' if a.endpoint else 'official SmolVLA conditional flow matching, shared LoRA and projections',
        train_episodes=sorted({r['source'] for r in items if r['split']=='train'}),validation_episodes=sorted({r['source'] for r in items if r['split']=='validation'}),data_directory=str(a.data),parent_checkpoint=str(a.resume_from) if a.resume_from else None,state_normalization='refitted on new training split only',test_used_for_training_or_selection=False,
        train_episode_seeds=sorted({items[i]['episode'] for i in train}),validation_episode_seeds=sorted({items[i]['episode'] for i in val}))
    def validate(step):
        nonlocal best
        policy.eval();metrics={}
        with torch.inference_mode():
            for task,ids in selected.items():
                pred=[]
                for start in range(0,len(ids),4):
                    rows=[items[i] for i in ids[start:start+4]]
                    batch=make_batch(observations(rows),[r['instruction'] for r in rows],tokenizer,'cuda',ss,sb)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        y=base.predict_action_chunk(batch,noise=torch.zeros((len(rows),1,32),device='cuda'))[:,0]
                    pred.append(y.float().cpu().numpy()*asc+ab)
                actual=np.concatenate(pred);expected=np.stack([items[i]['goal'] for i in ids])
                errors=np.abs(actual-expected);grip=((actual[:,10:]<.19)!=(expected[:,10:]<.19)).mean()
                metrics[task]=dict(position_mae_m=float(errors[:,:6].mean()),axis_mae=float(errors[:,6:10].mean()),gripper_error=float(grip),
                    score=float(errors[:,:6].mean()+.02*errors[:,6:10].mean()+.015*grip))
        score=float(np.mean([v['score'] for v in metrics.values()]));row=dict(step=step,score=score,tasks=metrics,wall_seconds=time.monotonic()-started)
        history.append(row);print('VALIDATION '+json.dumps(row),flush=True)
        if score<best:
            best=score;dest=a.output/'best';dest.mkdir(exist_ok=True);policy.save_pretrained(dest)
            cfg={**metadata,'selected_step':step,'validation':row,'adapter_sha256':hashlib.sha256((dest/'adapter_model.safetensors').read_bytes()).hexdigest()}
            (dest/'capture_config.json').write_text(json.dumps(cfg,indent=2))
        (a.output/'validation_history.json').write_text(json.dumps(history,indent=2));policy.train()
    # Never select the untrained model.
    for step in range(1,a.steps+1):
        ids=[]
        for j in range(a.batch_size):
            task=list(TASKS)[(step*a.batch_size+j)%3]
            phase=rng.choice(list(phasegroups[task]));pool=phasegroups[task][phase] if rng.random()<.5 else groups[task]
            if rng.random()<.2:pool=np.array([i for i in groups[task] if items[i]['initial']])
            ids.append(rng.choice(pool))
        rows=[items[i] for i in ids]
        batch=make_batch(observations(rows),[r['instruction'] for r in rows],tokenizer,'cuda',ss,sb)
        batch['action']=torch.tensor((np.stack([r['goal'] for r in rows])-ab)/asc,device='cuda')[:,None,:]
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=training_loss(policy,batch,weight,deterministic_endpoint=a.endpoint)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(trainable,1.);opt.step()
        if step%25==0:print(json.dumps(dict(step=step,loss=float(loss),seconds=time.monotonic()-started)),flush=True)
        if step%a.eval_every==0 or step==a.steps:validate(step)
    print('DONE '+str(a.output/'best'),flush=True)
if __name__=='__main__':main()
