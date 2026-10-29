"""One SmolVLA backbone with shared multi-task general and precision readouts."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from peft import PeftModel
from safetensors.torch import load_file
from .vla_policy import load_base,base_model,make_batch,BASE_REVISION
from .unified_features import Features
from .perception_calibration import PerceptionCalibration
from .recurrent_action_head import RecurrentActionHead
from .phase_action_head import PhaseActionHead
from .shared_action_head import previous_pose_baseline

class HierarchicalOnePolicy:
    def __init__(self,checkpoint,device='cuda'):
        self.directory=Path(checkpoint);self.device=device;self.config=json.loads((self.directory/'capture_config.json').read_text())
        if self.config['base_revision']!=BASE_REVISION:raise ValueError('Base revision mismatch')
        self.general_config=self.config['general'];self.precision_config=self.config['precision']
        self.features=Features();self.calibration=PerceptionCalibration(self.directory/'perception_calibration.npz')
        for name,key in [('adapter_model.safetensors','adapter_sha256'),('policy_heads.safetensors','heads_sha256')]:
            if hashlib.sha256((self.directory/name).read_bytes()).hexdigest()!=self.config[key]:raise ValueError('Weight hash mismatch')
        self.policy=PeftModel.from_pretrained(load_base(device,joint_count=12,state_count=32,action_count=12),self.directory).eval()
        self.base=base_model(self.policy);self.base.config.num_steps=self.base.model.config.num_steps=1
        self.tokenizer=self.base.model.vlm_with_expert.processor.tokenizer;self.hidden=None
        self.base.model.action_out_proj.register_forward_pre_hook(self._hidden)
        general=self.general_config['shared_head'];precision=self.precision_config['shared_head']
        self.heads=nn.ModuleDict(dict(general=RecurrentActionHead(general['hidden_dim'],general.get('memory_dim',128)),
            precision=PhaseActionHead(precision['hidden_dim'],precision.get('memory_dim',128),len(precision['phase_names'])))).to(device)
        self.heads.load_state_dict(load_file(str(self.directory/'policy_heads.safetensors'),device=device));self.heads.eval()
        self.sb=np.array(self.general_config['state_bias'],np.float32);self.ss=np.array(self.general_config['state_scale'],np.float32)
        self.ab=np.array(self.general_config['action_bias'],np.float32);self.asc=np.array(self.general_config['action_scale'],np.float32)
        self.pb=np.array(self.precision_config['head_action_bias'],np.float32);self.ps=np.array(self.precision_config['head_action_scale'],np.float32)
        self.reset()
    def _hidden(self,module,inputs):self.hidden=inputs[0][:,0].detach().float()
    def reset(self):
        self.memories=dict(general=None,precision=None);self.retained=np.zeros(2,bool);self.stall=np.zeros(2,int)
        self.last_jaws=None;self.last_time=None
    def predict(self,observation,instruction):
        started=time.monotonic();general_state=self.features(observation);precision_state=self.calibration(general_state)
        # The same backbone parameters process two state representations in one batch.
        states=np.stack([general_state,precision_state])
        batch=make_batch([{**observation,'q':s} for s in states],[instruction,instruction],self.tokenizer,self.device,self.ss,self.sb)
        with torch.inference_mode(),torch.autocast(self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):
            coarse=self.base.predict_action_chunk(batch,noise=torch.zeros((2,1,32),device=self.device))[:,0].float()
        raw=torch.tensor(states,device=self.device);normalized=torch.tensor((states-self.sb)/self.ss,device=self.device)
        ab=torch.tensor(self.ab,device=self.device);asc=torch.tensor(self.asc,device=self.device)
        pb=torch.tensor(self.pb,device=self.device);ps=torch.tensor(self.ps,device=self.device)
        with torch.inference_mode():
            baseline=(previous_pose_baseline(raw[:1],True)-ab)/asc
            g,self.memories['general']=self.heads['general'](self.hidden[:1],normalized[:1],coarse[:1],baseline,self.memories['general'])
            general_goal=(g[0]*asc+ab).cpu().numpy()
            object_coarse=coarse[1:]*asc+ab
            object_coarse[:,3:6]=object_coarse[:,3:6]-object_coarse[:,:3]+raw[1:,26:29]-raw[1:,3:6]
            object_coarse=(object_coarse-pb)/ps
            baseline=(torch.cat([raw[1:,12:15],raw[1:,26:29],raw[1:,6:10],raw[1:,18:20]],dim=1)-pb)/ps
            p,self.memories['precision'],logits=self.heads['precision'](self.hidden[1:],normalized[1:],object_coarse,baseline,self.memories['precision'])
            precision_goal=(p[0]*ps+pb).cpu().numpy();probabilities=logits.softmax(-1)[0].cpu().numpy()
        precision_goal[3:6]=precision_goal[:3]+precision_goal[3:6]-precision_state[26:29]+precision_state[3:6]
        jaws=np.asarray(observation['q'])[[5,11]];stamp=float(observation.get('image_time',observation['sim_time']))
        close=general_goal[10:12]<.19
        if self.last_time is not None and stamp-self.last_time>.04:
            stopped=np.abs(jaws-self.last_jaws)/(stamp-self.last_time)<.05
            self.stall=np.where(close & stopped & (jaws>-.15)&(jaws<.2),self.stall+1,0)
            self.retained|=self.stall>=2
        self.last_jaws=jaws.copy();self.last_time=stamp
        names=self.precision_config['shared_head']['phase_names']
        gate=float(sum(probabilities[names.index(name)] for name in ('orient','align','insert','hold'))) if self.retained.all() else 0.
        goal=general_goal*(1-gate)+precision_goal*gate
        raw_grips=goal[10:12].copy();goal[10:12]=np.where((raw_grips<.19)|self.retained,-.17,.55)
        if not np.isfinite(goal).all():raise RuntimeError('Nonfinite shared policy goal')
        decoded=goal.copy();decoded[3:6]-=decoded[:3]
        command=self.features.codec.decode(decoded,observation['q'])
        return command,dict(delta=goal.tolist(),state=precision_state.tolist(),raw_grips=raw_grips.tolist(),retained=self.retained.tolist(),
            phase_probabilities=probabilities.tolist(),precision_gate=gate,seconds=time.monotonic()-started,
            action_representation='shared hierarchical SmolVLA; independent world TCP goals')
