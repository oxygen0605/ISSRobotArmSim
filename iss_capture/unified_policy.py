"""A single text-conditioned SmolVLA adapter emits both arms' pose actions."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from peft import PeftModel
from .vla_policy import load_base,base_model,make_batch,BASE_REVISION
from .unified_features import Features,delta_decode
from .unified_config import CHECKPOINT

class OnePolicy:
    def __init__(self,checkpoint=CHECKPOINT,device='cuda'):
        self.directory=Path(checkpoint);self.config=json.loads((self.directory/'capture_config.json').read_text())
        if self.config['policy_variant']!='shared_three_task_smolvla_v1' or self.config['base_revision']!=BASE_REVISION:
            raise ValueError('Not a compatible shared policy checkpoint')
        self.device=device;self.features=Features();self.retained=np.zeros(2,dtype=bool);self.calibration=None
        for spec in [dict(file='adapter_model.safetensors',sha256=self.config.get('adapter_sha256')),self.config.get('shared_head',{}),self.config.get('perception_calibration',{})]:
            if spec.get('sha256') and hashlib.sha256((self.directory/spec['file']).read_bytes()).hexdigest()!=spec['sha256']:raise ValueError('Checkpoint weight hash mismatch: '+spec['file'])
        if self.config.get('perception_calibration'):
            from .perception_calibration import PerceptionCalibration
            self.calibration=PerceptionCalibration(self.directory/self.config['perception_calibration']['file'])
        self.policy=PeftModel.from_pretrained(load_base(device,joint_count=12,state_count=32,action_count=12,chunk_size=1),self.directory).eval()
        self.base=base_model(self.policy);self.tokenizer=self.base.model.vlm_with_expert.processor.tokenizer
        self.base.config.num_steps=self.base.model.config.num_steps=self.config.get('denoising_steps',5)
        self.sb=np.array(self.config['state_bias'],np.float32);self.ss=np.array(self.config['state_scale'],np.float32)
        self.ab=np.array(self.config['action_bias'],np.float32);self.asc=np.array(self.config['action_scale'],np.float32)
        self.hab=np.array(self.config.get('head_action_bias',self.ab),np.float32);self.hasc=np.array(self.config.get('head_action_scale',self.asc),np.float32)
        self.head=None;self.hidden=None;self.memory=None;self.last_jaws=None;self.last_image_time=None;self.stall_count=np.zeros(2,dtype=int)
        if self.config.get('shared_head'):
            from .shared_action_head import SharedActionHead,previous_pose_baseline
            from safetensors.torch import load_file
            spec=self.config['shared_head']
            if spec.get('version',1)==4:
                from .phase_action_head import PhaseActionHead
                self.head=PhaseActionHead(spec['hidden_dim'],spec.get('memory_dim',128),len(spec['phase_names'])).to(device)
            elif spec.get('version',1)==3:
                from .recurrent_action_head import RecurrentActionHead
                self.head=RecurrentActionHead(spec['hidden_dim'],spec.get('memory_dim',128)).to(device)
            else:self.head=SharedActionHead(spec['hidden_dim'],spec.get('version',1)).to(device)
            self.head.load_state_dict(load_file(str(self.directory/spec['file']),device=device));self.head.eval()
            self.base.model.action_out_proj.register_forward_pre_hook(self._capture_hidden)
    def _capture_hidden(self,module,inputs):self.hidden=inputs[0][:,0].detach().float()
    def reset(self):self.retained[:]=False;self.memory=None;self.last_jaws=None;self.last_image_time=None;self.stall_count[:]=0
    def predict(self,observation,instruction):
        # Deliberately no task ID, object poses, contacts, success monitor, teacher, or checkpoint routing.
        started=time.monotonic();state=self.features(observation)
        if self.calibration is not None:state=self.calibration(state)
        batch=make_batch([{**observation,'q':state}],[instruction],self.tokenizer,self.device,self.ss,self.sb)
        with torch.inference_mode(),torch.autocast(self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):
            action=self.base.predict_action_chunk(batch,noise=torch.zeros((1,1,32),device=self.device))[:,0].float()
        if self.head is not None:
            with torch.inference_mode():
                from .shared_action_head import previous_pose_baseline
                raw=torch.tensor(state,device=self.device)[None,:]
                if self.config.get('head_representation')=='object_relation':
                    physical=action*torch.tensor(self.asc,device=self.device)+torch.tensor(self.ab,device=self.device)
                    physical[:,3:6]=physical[:,3:6]-physical[:,:3]+raw[:,26:29]-raw[:,3:6]
                    action=(physical-torch.tensor(self.hab,device=self.device))/torch.tensor(self.hasc,device=self.device)
                    reference=raw[:,26:29] if self.config['shared_head'].get('baseline')=='object_current' else torch.tensor(self.hab[3:6],device=self.device)[None,:]
                    baseline_raw=torch.cat([raw[:,12:15],reference,raw[:,6:10],raw[:,18:20]],dim=1)
                    baseline=(baseline_raw-torch.tensor(self.hab,device=self.device))/torch.tensor(self.hasc,device=self.device)
                else:baseline=(previous_pose_baseline(raw,self.config.get('representation')=='world')-torch.tensor(self.ab,device=self.device))/torch.tensor(self.asc,device=self.device)
                if self.config['shared_head'].get('baseline')=='coarse':baseline=action
                normalized=torch.tensor((state-self.sb)/self.ss,device=self.device)[None,:]
                if self.config['shared_head'].get('version',1)==4:
                    action,self.memory,phase_logits=self.head(self.hidden,normalized,action,baseline,self.memory)
                    self.phase_probabilities=phase_logits.softmax(-1)[0].cpu().tolist()
                elif self.config['shared_head'].get('version',1)==3:
                    action,self.memory=self.head(self.hidden,normalized,action,baseline,self.memory)
                else:action=self.head(self.hidden,normalized,action,baseline)
        action=action[0].cpu().numpy()
        delta=action*self.hasc+self.hab
        if not np.isfinite(delta).all():raise RuntimeError('Nonfinite shared policy output')
        raw_grips=delta[10:12].copy()
        if self.config.get('grasp_retention',False):
            # All supported tasks end while holding. A close request plus encoder feedback
            # latches holding effort until the next episode; no contact or object truth is used.
            close=raw_grips<.19
            jaws=np.asarray(observation['q'])[[5,11]]
            if self.config.get('retention_version',1)==1:self.retained|=close & (jaws<.15)
            else:
                stamp=float(observation.get('image_time',observation.get('sim_time',0.)))
                if self.last_image_time is not None and stamp-self.last_image_time>.04:
                    stalled=np.abs(jaws-self.last_jaws)/(stamp-self.last_image_time)<.05
                    candidate=close & stalled & (jaws>-.15) & (jaws<.2)
                    self.stall_count=np.where(candidate,self.stall_count+1,0)
                    self.retained|=self.stall_count>=2
                self.last_jaws=jaws.copy();self.last_image_time=stamp
            delta[10:12]=np.where(close|self.retained,-.17,.55)
        decoded=delta.copy()
        if self.config.get('head_representation')=='object_relation':decoded[3:6]=decoded[3:6]-state[26:29]+state[3:6]
        elif self.config.get('representation')=='world':decoded[3:6]-=decoded[:3]
        command=self.features.codec.decode(decoded,observation['q']) if self.config.get('representation','delta') in ('goal','world') else delta_decode(decoded,observation['q'],self.features.codec)
        return command,dict(delta=delta.tolist(),raw_grips=raw_grips.tolist(),retained=self.retained.tolist(),phase_probabilities=getattr(self,'phase_probabilities',None),state=state.tolist(),seconds=time.monotonic()-started)
