#!/usr/bin/env python3
"""Fit one language-conditioned SmolVLA goal policy from measured RGB-D episodes."""
import argparse,json,hashlib,shutil,time
from pathlib import Path
import numpy as np
import torch
from peft import PeftModel
from safetensors.torch import save_file,load_file
from iss_capture.config import ROOT
from iss_capture.public_dataset import ensure_dataset,dataset_provenance
from iss_capture.unified_config import TASKS
from iss_capture.workspace_features import PHASES
from iss_capture.scene_dataset import load_workspace,goal_targets,supervised_targets,attach_water_positions
from iss_capture.scene_goal_head import SceneGoalHead
from iss_capture.scene_language import language_embedding,PARAPHRASES
from iss_capture.unified_dataset import observations
from iss_capture.vla_policy import load_base,base_model,make_batch
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,default=ROOT/'artifacts/task_scene_backbone_v1/best')
    p.add_argument('--dataset',type=Path,default=ROOT/'datasets/task_scenes_v1')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/task_scene_head_v1')
    p.add_argument('--steps',type=int,default=10000);p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--water-depth-sensor',action='store_true');p.add_argument('--water-goal-embedding',action='store_true');p.add_argument('--water-only',action='store_true');p.add_argument('--remember-task',action='store_true');p.add_argument('--grasp-refinement',action='store_true');p.add_argument('--grasp-only',action='store_true');p.add_argument('--carry-goal-embedding',action='store_true');p.add_argument('--remove-grasp-dwell',action='store_true');p.add_argument('--alignment-refinement',action='store_true');p.add_argument('--alignment-only',action='store_true');p.add_argument('--remove-alignment-dwell',action='store_true');p.add_argument('--physical-phase-features',action='store_true');p.add_argument('--phase-only',action='store_true');p.add_argument('--moving-grasp-frame',action='store_true');p.add_argument('--augment',action='store_true');p.add_argument('--episode-manifest',type=Path);p.add_argument('--resume-head',type=Path);p.add_argument('--limit-train',type=int,default=0)
    p.set_defaults(water_goal_embedding=True,water_depth_sensor=True,remember_task=True,moving_grasp_frame=True,physical_phase_features=True,alignment_refinement=True,grasp_refinement=True,augment=True,remove_alignment_dwell=True,remove_grasp_dwell=True)
    p.add_argument('--grip-closure-refinement',action='store_true');p.add_argument('--closure-only',action='store_true')
    p.add_argument('--assembly-step-refinement',action='store_true');p.add_argument('--assembly-only',action='store_true')
    a=p.parse_args();a.dataset=ensure_dataset(a.dataset);a.output.mkdir(parents=True,exist_ok=True);torch.manual_seed(20261031);rng=np.random.default_rng(20261031)
    cfg=json.loads((a.checkpoint/'capture_config.json').read_text());items,episodes=load_workspace(a.dataset)
    if a.episode_manifest:
        names={Path(name).name for name in json.loads(a.episode_manifest.read_text())}
        old=items;items=[];selected=[]
        for e in episodes:
            if Path(e['source']).name not in names:continue
            rows=[old[i] for i in e['indices']];ids=list(range(len(items),len(items)+len(rows)));items.extend(rows);selected.append({**e,'indices':ids})
        episodes=selected
        if {Path(e['source']).name for e in episodes}!=names:raise ValueError('Manifest contains missing or unsuccessful episodes')
    if a.limit_train:
        keep=[];counts={t:0 for t in TASKS}
        for e in episodes:
            if e['split']=='train':
                counts[e['task']]+=1
                if counts[e['task']]>a.limit_train:continue
            keep.append(e)
        old=items;items=[];episodes=[]
        for e in keep:
            rows=[old[i] for i in e['indices']];ids=list(range(len(items),len(items)+len(rows)));items.extend(rows);episodes.append({**e,'indices':ids})
    groups={t:np.array([i for i,e in enumerate(episodes) if e['split']=='train' and e['task']==t]) for t in TASKS}
    if any(len(v)==0 for v in groups.values()):raise ValueError('Need successful measured demonstrations of all three tasks')
    sb=np.array(cfg['state_bias'],np.float32);ss=np.array(cfg['state_scale'],np.float32)
    if a.water_goal_embedding:attach_water_positions(items,episodes,use_depth=a.water_depth_sensor)
    start=time.monotonic();prototypes=goal_targets(items,episodes);relabelled={'train':0,'validation':0}
    if a.remove_alignment_dwell:
        # Training-only teacher label refinement: do not teach two different actions
        # for the same stationary, already aligned observation.
        for row in items:
            s=row['head_state']
            if row['phase']=='dual_align' and np.linalg.norm(s[26:29]-prototypes['relative'][0])<.0015 and max(np.linalg.norm(s[29:31]),np.linalg.norm(s[32:34]))<.04:
                row['phase']='dual_insert';row['phase_id']=PHASES.index('dual_insert');relabelled[row['split']]+=1
    grasp_relabelled={'train':0,'validation':0}
    if a.remove_grasp_dwell:
        for episode in episodes:
            rows=[items[i] for i in episode['indices']]
            carry=[r for r in rows if r['phase']=='pen_carry']
            for row in rows:
                s=row['head_state'];ready=(row['pre']==2)&(s[10:12]<.2)&(np.abs(s[36:38])<.1)
                if row['phase']=='pen_grasp' and ready[0] and carry:
                    row['phase']='pen_carry';row['phase_id']=3;row['goal'][:3]=carry[0]['goal'][:3]
                    row['pre_mask'][:]=0;grasp_relabelled[row['split']]+=1
                elif row['phase']=='dual_grasp' and ready.all():
                    row['phase']='dual_orient';row['phase_id']=5;row['pre_mask'][:]=0
                    grasp_relabelled[row['split']]+=1
    target,local,axis,lmask,amask=supervised_targets(items,prototypes,a.moving_grasp_frame,a.water_goal_embedding,a.assembly_step_refinement)
    train=np.array([i for i,r in enumerate(items) if r['split']=='train']);raw=np.stack([r['head_state'] for r in items])
    hb=raw[train].mean(0);hs=np.maximum(raw[train].std(0),np.r_[np.full(34,.02),.05,.05,.1,.1,np.full(9,.015),np.full(6,.1)]).astype(np.float32)
    hs[[6,7,8,9,10,11,18,19,29,30,31,32,33]]=np.maximum(hs[[6,7,8,9,10,11,18,19,29,30,31,32,33]],.15)
    hab=target[train].mean(0);hasc=np.maximum(target[train].std(0),.025).astype(np.float32)
    hab[6:10]=0;hasc[6:10]=.35;hab[10:]=.19;hasc[10:]=.36
    if a.resume_head:
        resume_cfg=json.loads((a.resume_head/'capture_config.json').read_text())
        hb=np.array(resume_cfg['head_state_bias'],np.float32);hs=np.array(resume_cfg['head_state_scale'],np.float32)
        hab=np.array(resume_cfg['head_action_bias'],np.float32);hasc=np.array(resume_cfg['head_action_scale'],np.float32)
    rb=raw[train,26:29].mean(0);rs=np.maximum(raw[train,26:29].std(0),.04)
    if a.resume_head:
        resume_weights=load_file(str(a.resume_head/'shared_head.safetensors'),device='cpu')
        rb=resume_weights['relative_bias'].numpy();rs=resume_weights['relative_scale'].numpy()
    cache=a.output/'backbone_cache.npz'
    signature=hashlib.sha256(json.dumps([(r['source'],r['time']) for r in items]).encode()+sb.tobytes()+ss.tobytes()+(a.checkpoint/'adapter_model.safetensors').read_bytes()).hexdigest()
    if cache.exists() and str(np.load(cache)['signature'])==signature:
        cached=np.load(cache);hidden=cached['hidden'];language=cached['language'];language_bank=cached['language_bank'];language_labels=cached['language_labels'];print('CACHE '+str(hidden.shape),flush=True)
    else:
        policy=PeftModel.from_pretrained(load_base(joint_count=12,state_count=32,action_count=12),a.checkpoint).eval();base=base_model(policy)
        base.config.num_steps=base.model.config.num_steps=1;tokenizer=base.model.vlm_with_expert.processor.tokenizer;capture={}
        hook=base.model.action_out_proj.register_forward_pre_hook(lambda m,i:capture.update(hidden=i[0][:,0].detach()))
        hidden=[];language=[]
        with torch.inference_mode():
            for offset in range(0,len(items),16):
                rows=items[offset:offset+16];batch=make_batch(observations(rows),[r['instruction'] for r in rows],tokenizer,'cuda',ss,sb)
                with torch.autocast('cuda',dtype=torch.bfloat16):base.predict_action_chunk(batch,noise=torch.zeros((len(rows),1,32),device='cuda'))
                hidden.append(capture['hidden'].float().cpu().numpy())
                language.append(language_embedding(base,batch).cpu().numpy())
                if offset%320==0:print('CACHE '+json.dumps(dict(cached=min(len(items),offset+16),total=len(items))),flush=True)
        instructions=[];language_labels=[]
        for task_id,task in enumerate(TASKS):
            texts=sorted({r['instruction'] for r in items if r['task']==task and r['split']=='train'})+PARAPHRASES[task]
            instructions.extend(texts);language_labels.extend([task_id]*len(texts))
        tokens=tokenizer(instructions,padding='max_length',max_length=48,truncation=True,return_tensors='pt')
        language_bank=language_embedding(base,{'observation.language.tokens':tokens['input_ids'].cuda(),'observation.language.attention_mask':tokens['attention_mask'].bool().cuda()}).cpu().numpy()
        language_labels=np.array(language_labels,np.int64)
        hook.remove();hidden=np.concatenate(hidden);language=np.concatenate(language);np.savez_compressed(cache,signature=signature,hidden=hidden,language=language,language_bank=language_bank,language_labels=language_labels)
        del policy,base;torch.cuda.empty_cache()
    states=(raw-hb)/hs;targets=(target-hab)/hasc;maxlen=max(len(e['indices']) for e in episodes);n=len(episodes);arrays={}
    fields=[('hidden',hidden),('language',language),('state',states),('rotation',np.stack([r['rotation'] for r in items])),('target',targets),('local',local),('axis',axis),('lmask',lmask),('amask',amask),('pre',np.stack([r['pre'] for r in items])),('pmask',np.stack([r['pre_mask'] for r in items])),('water',np.stack([r.get('water_position',r['head_state'][20:23]) for r in items]))]
    for name,values in fields:
        array=np.zeros((n,maxlen,*values.shape[1:]),values.dtype)
        for j,e in enumerate(episodes):array[j,:len(e['indices'])]=values[e['indices']]
        arrays[name]=torch.tensor(array,device='cuda')
    weights=np.zeros((n,maxlen),np.float32);phases=np.zeros((n,maxlen),np.int64)
    for j,e in enumerate(episodes):
        rows=[items[i] for i in e['indices']];counts={name:sum(r['phase']==name for r in rows) for name in {r['phase'] for r in rows}}
        w=np.array([len(rows)/len(counts)/counts[r['phase']] for r in rows],np.float32);w[:3]*=2
        weights[j,:len(rows)]=w;phases[j,:len(rows)]=[r['phase_id'] for r in rows]
    weights=torch.tensor(weights,device='cuda');phases=torch.tensor(phases,device='cuda')
    head=SceneGoalHead(hidden.shape[1],moving_grasp_frame=a.moving_grasp_frame,physical_phase_features=a.physical_phase_features,alignment_refinement=a.alignment_refinement,grasp_refinement=a.grasp_refinement,carry_goal_embedding=a.carry_goal_embedding,water_goal_embedding=a.water_goal_embedding,remember_task=a.remember_task,language_dim=language.shape[1],grip_closure_refinement=a.grip_closure_refinement,assembly_step_refinement=a.assembly_step_refinement).cuda()
    language_bank=torch.tensor(language_bank,device='cuda');language_labels=torch.tensor(language_labels,device='cuda')
    for key,value in [('state_bias',hb),('state_scale',hs),('action_bias',hab),('action_scale',hasc),('relative_bias',rb),('relative_scale',rs)]:
        getattr(head,key).copy_(torch.tensor(value,device='cuda'))
    proto=[torch.tensor((prototypes['centre']-hab[:3])/hasc[:3],device='cuda'),
           torch.tensor((prototypes['relative']-rb)/rs,device='cuda'),torch.tensor(prototypes['axes']/hasc[6:10],device='cuda')]
    if a.carry_goal_embedding:proto.append(torch.tensor((prototypes['carry']-hab[:3])/hasc[:3],device='cuda'))
    if a.water_goal_embedding:
        if not a.carry_goal_embedding:proto.append(None)
        proto.append(torch.tensor(prototypes['water']/.05,device='cuda'))
    with torch.no_grad():
        if a.water_goal_embedding:head.water_goal.copy_(proto[-1])
        if a.carry_goal_embedding:head.carry_goal.copy_(proto[3])
        head.centre_goal.copy_(proto[0]);head.relative_goals.copy_(proto[1]);head.object_axis_goal.copy_(proto[2])
        for name in ('left_hold','right_idle','right_orient','centre_relative','water_axis_hold','grasp_hold_offsets'):getattr(head,name).zero_()
    if a.resume_head:
        previous=load_file(str(a.resume_head/'shared_head.safetensors'),device='cuda')
        for k in ('state_bias','state_scale','action_bias','action_scale','relative_bias','relative_scale'):previous[k]=head.state_dict()[k]
        missing,unexpected=head.load_state_dict(previous,strict=False)
        if unexpected or any(not k.startswith(('physical_phase.','advance.','grasp_advance.','carry_goal','carry_axis_hold','water_goal','phase_task_ids','grip_closure.','assembly_step_logit')) for k in missing):raise ValueError((missing,unexpected))
    if a.phase_only:
        if not a.resume_head or not a.physical_phase_features:raise ValueError('phase-only needs a parent checkpoint and physical features')
        for name,param in head.named_parameters():param.requires_grad_(name.startswith('physical_phase.'))
    if a.alignment_only:
        if not a.resume_head or not a.alignment_refinement:raise ValueError('alignment-only needs a parent and alignment-refinement')
        for name,param in head.named_parameters():param.requires_grad_(name.startswith('advance.'))
    if a.grasp_only:
        if not a.resume_head or not a.grasp_refinement:raise ValueError('grasp-only needs a parent and grasp-refinement')
        for name,param in head.named_parameters():param.requires_grad_(name.startswith(('grasp_advance.','carry_goal','carry_axis_hold','water_goal','phase_task_ids')))
    if a.water_only:
        if not a.resume_head or not a.water_goal_embedding:raise ValueError('water-only needs a parent and water goal embedding')
        for name,param in head.named_parameters():param.requires_grad_(name in ('water_goal','water_axis_hold'))
    if a.closure_only:
        if not a.resume_head or not a.grip_closure_refinement:raise ValueError('closure-only needs a parent and grip-closure-refinement')
        for name,param in head.named_parameters():param.requires_grad_(name.startswith('grip_closure.'))
    if a.assembly_only:
        if not a.resume_head or not a.assembly_step_refinement:raise ValueError('assembly-only needs a parent and assembly-step-refinement')
        for name,param in head.named_parameters():param.requires_grad_(name.startswith(('physical_phase.','advance.','assembly_step_logit')))
    optimizer=torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],lr=a.lr,weight_decay=1e-5)
    history=[];best=float('inf');val=[i for i,e in enumerate(episodes) if e['split']=='validation']
    summary={s:{t:dict(episodes=sum(e['split']==s and e['task']==t for e in episodes),frames=sum(r['split']==s and r['task']==t for r in items)) for t in TASKS} for s in ('train','validation')}
    print('DATA '+json.dumps(summary),flush=True);print('GOALS '+json.dumps({k:v.tolist() for k,v in prototypes.items()}),flush=True)
    def predict(ids):return head(arrays['hidden'][ids],arrays['state'][ids],arrays['rotation'][ids],water_position=arrays['water'][ids],language=arrays['language'][ids])
    def validate(step):
        nonlocal best
        head.eval();metrics={}
        with torch.inference_mode():
            for task in TASKS:
                ids=[i for i in val if episodes[i]['task']==task]
                if not ids:raise ValueError('Missing validation task '+task)
                out,_,phase,pre,_=predict(ids);pred=out.cpu().numpy()*hasc+hab;expected=arrays['target'][ids].cpu().numpy()*hasc+hab
                phase=phase.argmax(-1).cpu().numpy();pre=pre.argmax(-1).cpu().numpy();errors=[];initial=[];grip=[];pe=[];ge=[];buckets={}
                for j,eid in enumerate(ids):
                    e=episodes[eid];length=len(e['indices']);err=np.abs(pred[j,:length]-expected[j,:length]);errors.append(err);initial.append(err[:3,:6])
                    grip.extend(((pred[j,:length,10:]<.19)!=(expected[j,:length,10:]<.19)).ravel())
                    pe.extend(phase[j,:length]!=phases[eid,:length].cpu().numpy())
                    mask=arrays['pmask'][eid,:length].cpu().numpy().astype(bool)
                    ge.extend((pre[j,:length]!=arrays['pre'][eid,:length].cpu().numpy())[mask])
                    for k,i in enumerate(e['indices']):buckets.setdefault(items[i]['phase'],[]).append(err[k,:6].mean())
                err=np.concatenate(errors);byphase={k:float(np.mean(v)) for k,v in buckets.items()};ini=float(np.concatenate(initial).mean());ge=float(np.mean(ge)) if len(ge) else 0.
                score=float(np.mean(list(byphase.values()))+.5*ini+.02*err[:,6:10].mean()+.02*np.mean(grip)+.008*np.mean(pe)+.005*ge)
                metrics[task]=dict(position_mae_m=float(err[:,:6].mean()),phase_position_mae_m=byphase,initial_mae_m=ini,grip_error=float(np.mean(grip)),phase_error=float(np.mean(pe)),grasp_stage_error=ge,language_task_accuracy=float((head.language_task(arrays['language'][ids,0]).argmax(-1)==list(TASKS).index(task)).float().mean()),score=score)
        score=float(np.mean([v['score'] for v in metrics.values()]));row=dict(step=step,score=score,tasks=metrics,seconds=time.monotonic()-start)
        history.append(row);print('VALIDATION '+json.dumps(row),flush=True)
        if score<best:
            best=score;dest=a.output/'best';dest.mkdir(exist_ok=True)
            for name in ('adapter_model.safetensors','adapter_config.json'):shutil.copy2(a.checkpoint/name,dest/name)
            save_file({k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()},str(dest/'shared_head.safetensors'))
            metadata={**cfg,'training_dataset':dataset_provenance(a.dataset),'policy_variant':'shared_three_task_smolvla_task_scenes_v1','hidden_dim':hidden.shape[1],'language_dim':language.shape[1],'language_task_classifier':True,'right_water_goal':True,'pen_hold_without_lift':True,'grip_closure_refinement':a.grip_closure_refinement,'closure_only_training':a.closure_only,'assembly_step_refinement':a.assembly_step_refinement,'assembly_only_training':a.assembly_only,'learned_assembly_step_m':float(.020*head.assembly_step_logit.sigmoid()) if a.assembly_step_refinement else None,'phase_names':PHASES,'head_state_dim':53,
                'head_state_bias':hb.tolist(),'head_state_scale':hs.tolist(),'head_action_bias':hab.tolist(),'head_action_scale':hasc.tolist(),
                'head_sha256':hashlib.sha256((dest/'shared_head.safetensors').read_bytes()).hexdigest(),
                'state_source':'synchronized RGB-D camera images, joint encoders, previous commands; no simulator object poses or contacts',
                'temporal_augmentation':a.augment,'moving_grasp_frame':a.moving_grasp_frame,'physical_phase_features':a.physical_phase_features,'phase_only_training':a.phase_only,'water_goal_embedding':a.water_goal_embedding,'water_only_training':a.water_only,'remember_task':a.remember_task,'water_depth_sensor':a.water_depth_sensor,'water_sensor':('idealized RGB-D sphere surface' if a.water_depth_sensor else 'RGB sphere silhouettes in two calibrated views') if a.water_goal_embedding else 'RGB color stereo','grasp_refinement':a.grasp_refinement,'carry_goal_embedding':a.carry_goal_embedding,'grasp_only_training':a.grasp_only,'relabelled_ready_grasp_frames':grasp_relabelled,'alignment_refinement':a.alignment_refinement,'alignment_only_training':a.alignment_only,'alignment_dwell_removed':a.remove_alignment_dwell,'relabelled_ready_alignment_frames':relabelled,'parent_head':str(a.resume_head) if a.resume_head else None,'training_method':'one frozen fine-tuned SmolVLA backbone plus one shared language-conditioned GRU, learned goal embeddings and local action offsets',
                'action_representation':'two world TCP positions, two tool axes and two grippers; learned grasp/assembly stage probabilities',
                'learned_goal_targets':{k:v.tolist() for k,v in prototypes.items()},'selected_readout_step':step,'validation':row,'data_summary':summary,
                'train_episodes':[e['source'] for e in episodes if e['split']=='train'],'validation_episodes':[e['source'] for e in episodes if e['split']=='validation'],
                'backbone_checkpoint':str(a.checkpoint),'backbone_training_method':cfg.get('training_method'),'test_data_used':False,'teacher_at_inference':False,'task_based_model_routing':False}
            for key in ('perception_calibration','shared_head'):metadata.pop(key,None)
            (dest/'capture_config.json').write_text(json.dumps(metadata,indent=2))
        (a.output/'validation_history.json').write_text(json.dumps(history,indent=2));head.train()
    validate(0)
    for step in range(1,a.steps+1):
        ids=np.concatenate([rng.choice(groups[t],3,replace=True) for t in TASKS])
        selected={k:v[ids] for k,v in arrays.items()};w=weights[ids];phase_label=phases[ids]
        if a.augment:
            orders=[]
            for i in ids:
                length=len(episodes[i]['indices'])
                speed=float(rng.choice([.6,.8,1.,1.25,1.5,2.]))
                index=np.floor(np.arange(0,length,speed)).astype(int)
                if rng.random()<.25:index=index[int(rng.integers(0,max(1,len(index)//3))):]
                orders.append(index)
            length=max(map(len,orders));index=torch.tensor(np.stack([np.pad(v,(0,length-len(v)),constant_values=v[-1]) for v in orders]),device='cuda')
            valid=torch.tensor(np.stack([np.arange(length)<len(v) for v in orders]),device='cuda')
            selected={k:torch.gather(v,1,index.reshape(*index.shape,*([1]*(v.ndim-2))).expand(-1,-1,*v.shape[2:])) for k,v in selected.items()}
            w=torch.gather(w,1,index)*valid;phase_label=torch.gather(phase_label,1,index)
            selected['state']=selected['state']+torch.randn_like(selected['state'])*.025
            selected['hidden']=torch.nn.functional.dropout(selected['hidden'],p=.04,training=True)
        prediction,_,logits,pre,out=head(selected['hidden'],selected['state'],selected['rotation'],water_position=selected['water'],language=selected['language']);truth=selected['target']
        optimizer.zero_grad(set_to_none=True)
        loss=(((prediction-truth).square()*torch.tensor([3,3,3,4,4,4,2,2,2,2,2,2],device='cuda')).mean(-1)*w).sum()/w.sum()
        lm=selected['lmask'];am=selected['amask'];pm=selected['pmask']
        loss+=.3*(((out[...,:6]-selected['local']).square()*lm).sum(-1)*w).sum()/(lm.sum(-1)*w).sum().clamp_min(1)
        loss+=.1*(((out[...,6:10]-selected['axis']).square()*am).sum(-1)*w).sum()/(am.sum(-1)*w).sum().clamp_min(1)
        loss+=.2*(torch.nn.functional.cross_entropy(logits.transpose(1,2),phase_label,reduction='none')*w).sum()/w.sum()
        pl=torch.nn.functional.cross_entropy(pre.reshape(-1,3),selected['pre'].reshape(-1),reduction='none').reshape_as(pm)
        loss+=.15*(pl*pm*w[:,:,None]).sum()/(pm*w[:,:,None]).sum().clamp_min(1)
        loss+=.02*(torch.nn.functional.binary_cross_entropy_with_logits(-prediction[...,10:]*4,(truth[...,10:]<0).float(),reduction='none').mean(-1)*w).sum()/w.sum()
        loss+=.5*head.prototype_loss(*proto)
        loss+=torch.nn.functional.cross_entropy(head.language_task(language_bank),language_labels)
        if a.grip_closure_refinement:
            closure_mask=torch.stack([(phase_label==2)|(phase_label==4),phase_label==4],-1).to(w.dtype)
            closure_loss=torch.nn.functional.binary_cross_entropy_with_logits(head.last_closure_logits,(truth[...,10:]<0).float(),reduction='none')
            loss+=(closure_loss*closure_mask*w[...,None]).sum()/(closure_mask*w[...,None]).sum().clamp_min(1)
        loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
        if step%250==0:print(json.dumps(dict(step=step,loss=float(loss.detach()))),flush=True)
        if step%500==0 or step==a.steps:validate(step)
    print('DONE',flush=True)
if __name__=='__main__':main()
