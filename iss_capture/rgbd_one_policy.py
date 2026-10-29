"""One SmolVLA policy with a shared RGB-D-conditioned recurrent action decoder."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from peft import PeftModel
from safetensors.torch import load_file
from .vla_policy import load_base,base_model,make_batch,BASE_REVISION,CAMERA_NAMES
from .unified_features import network_rgb
from .depth_geometry import RGBDFeatures
from .phase_action_head import PhaseActionHead
class RGBDOnePolicy:
    def __init__(self,checkpoint,device='cuda'):
        self.directory=Path(checkpoint);self.device=device;self.config=json.loads((self.directory/'capture_config.json').read_text())
        if self.config['base_revision']!=BASE_REVISION:raise ValueError('Base revision mismatch')
        for name,key in [('adapter_model.safetensors','adapter_sha256'),('shared_head.safetensors','head_sha256')]:
            if hashlib.sha256((self.directory/name).read_bytes()).hexdigest()!=self.config[key]:raise ValueError('Weight hash mismatch')
        self.policy=PeftModel.from_pretrained(load_base(device,joint_count=12,state_count=32,action_count=12),self.directory).eval()
        self.base=base_model(self.policy);self.base.config.num_steps=self.base.model.config.num_steps=1
        self.tokenizer=self.base.model.vlm_with_expert.processor.tokenizer;self.hidden=None
        self.base.model.action_out_proj.register_forward_pre_hook(self._capture)
        self.head=PhaseActionHead(self.config['hidden_dim'],128,len(self.config['phase_names']),state_dim=36).to(device)
        self.head.load_state_dict(load_file(str(self.directory/'shared_head.safetensors'),device=device));self.head.eval()
        for key in ('state_bias','state_scale','action_bias','action_scale','head_state_bias','head_state_scale','head_action_bias','head_action_scale'):
            setattr(self,key,np.array(self.config[key],np.float32))
        self.reset()
    def _capture(self,module,inputs):self.hidden=inputs[0][:,0].detach().float()
    def reset(self):
        self.features=RGBDFeatures();self.memory=None;self.retained=np.zeros(2,bool);self.stall=np.zeros(2,int)
        self.last_stamp=None;self.last_jaws=None;self.last_result=None
    def predict(self,observation,instruction):
        start=time.monotonic();stamp=float(observation['image_time'])
        if self.last_stamp is not None and abs(stamp-self.last_stamp)<1e-9:return self.last_result
        state34=self.features(observation);state=state34[:32]
        dt=.1 if self.last_stamp is None else stamp-self.last_stamp
        age=max(0.,float(observation['controller_time'])-stamp)
        full_state=np.r_[state34,dt,age].astype(np.float32)
        obs={**observation,'q':state,**{n:network_rgb(observation[n]) for n in CAMERA_NAMES}}
        batch=make_batch([obs],[instruction],self.tokenizer,self.device,self.state_scale,self.state_bias)
        with torch.inference_mode(),torch.autocast(self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):
            coarse=self.base.predict_action_chunk(batch,noise=torch.zeros((1,1,32),device=self.device))[:,0].float()
        object_axes=np.r_[state[29:31],state34[32:34]]
        raw_goal=coarse[0].cpu().numpy()*self.action_scale+self.action_bias
        raw_goal[3:6]=raw_goal[3:6]-raw_goal[:3]+state[26:29]-state[3:6]
        raw_goal[6:10]+=object_axes-state[6:10]
        coarse=(raw_goal-self.head_action_bias)/self.head_action_scale
        baseline_raw=np.r_[state[12:15],state[26:29],object_axes,state[18:20]]
        baseline=(baseline_raw-self.head_action_bias)/self.head_action_scale
        normalized=(full_state-self.head_state_bias)/self.head_state_scale
        with torch.inference_mode():
            output,self.memory,logits=self.head(self.hidden,torch.tensor(normalized,device=self.device)[None],
                torch.tensor(coarse,device=self.device)[None],torch.tensor(baseline,device=self.device)[None],self.memory)
        goal=output[0].cpu().numpy()*self.head_action_scale+self.head_action_bias
        raw_grips=goal[10:12].copy();close=raw_grips<.19;jaws=np.asarray(observation['q'])[[5,11]]
        if self.last_stamp is not None and dt>.04:
            stopped=np.abs(jaws-self.last_jaws)/dt<.05
            self.stall=np.where(close&stopped&(jaws>-.15)&(jaws<.2),self.stall+1,0)
            self.retained|=self.stall>=2
            self.retained&=jaws>-.155
        goal[10:12]=np.where(close|self.retained,-.17,.55)
        decoded=goal.copy()
        decoded[3:6]=goal[3:6]-state[26:29]+state[3:6]
        decoded[6:10]=goal[6:10]-object_axes+state[6:10]
        if not np.isfinite(decoded).all():raise RuntimeError('Nonfinite policy action')
        command=self.features.codec.decode(decoded,observation['q'])
        self.last_stamp=stamp;self.last_jaws=jaws.copy()
        info=dict(delta=goal.tolist(),state=full_state.tolist(),raw_grips=raw_grips.tolist(),retained=self.retained.tolist(),
            phase_probabilities=logits.softmax(-1)[0].cpu().tolist(),seconds=time.monotonic()-start,
            observation_source='RGB-D images and encoders only',action_representation='object-relative position and observed-object axis proxy',
            depth_points={name:g['points'] for name,g in self.features.last_geometry.items()})
        self.last_result=(command,info);return self.last_result
