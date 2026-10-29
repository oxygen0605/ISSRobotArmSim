"""Native SmolVLA action chunks for both SO-101 arms, shared by all tasks."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from peft import PeftModel
from .vla_policy import load_base,base_model,make_batch,BASE_REVISION
from .unified_features import Features
from .perception_calibration import PerceptionCalibration
from .bimanual_config import bounded_action
MAJOR=(0,1,2,5,6,7,8,11)
def joint_state(observation,geometry):
    q=np.asarray(observation['q'],np.float32)
    previous=np.asarray(observation.get('previous_command',q),np.float32)
    return np.r_[q,previous[list(MAJOR)],np.asarray(geometry,np.float32)[20:32]].astype(np.float32)
class JointOnePolicy:
    def __init__(self,checkpoint,device='cuda'):
        self.directory=Path(checkpoint);self.device=device;self.config=json.loads((self.directory/'capture_config.json').read_text())
        if self.config['base_revision']!=BASE_REVISION:raise ValueError('Incompatible SmolVLA base')
        if hashlib.sha256((self.directory/'adapter_model.safetensors').read_bytes()).hexdigest()!=self.config['adapter_sha256']:raise ValueError('Adapter hash mismatch')
        self.features=Features();self.calibration=None
        if self.config.get('perception_calibration'):
            self.calibration=PerceptionCalibration(self.directory/self.config['perception_calibration']['file'])
        self.policy=PeftModel.from_pretrained(load_base(device,joint_count=12,state_count=32,action_count=12,chunk_size=self.config['chunk_size']),self.directory).eval()
        self.base=base_model(self.policy);self.tokenizer=self.base.model.vlm_with_expert.processor.tokenizer
        self.base.config.num_steps=self.base.model.config.num_steps=self.config.get('denoising_steps',5)
        self.sb=np.array(self.config['state_bias'],np.float32);self.ss=np.array(self.config['state_scale'],np.float32)
        self.ab=np.array(self.config['action_bias'],np.float32);self.asc=np.array(self.config['action_scale'],np.float32)
        self.reset()
    def reset(self):
        self.retained=np.zeros(2,bool);self.stall=np.zeros(2,int);self.last_jaws=None;self.last_time=None
    def predict(self,observation,instruction):
        start=time.monotonic();geometry=self.features(observation)
        if self.calibration is not None:geometry=self.calibration(geometry)
        state=joint_state(observation,geometry)
        batch=make_batch([{**observation,'q':state}],[instruction],self.tokenizer,self.device,self.ss,self.sb)
        with torch.inference_mode(),torch.autocast(self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):
            actions=self.base.predict_action_chunk(batch,noise=torch.zeros((1,self.config['chunk_size'],32),device=self.device))[0].float().cpu().numpy()
        actions=actions*self.asc+self.ab
        if not np.isfinite(actions).all():raise RuntimeError('Nonfinite action chunk')
        command=actions[0].copy();raw=command[[5,11]].copy();close=raw<.19
        jaws=np.asarray(observation['q'])[[5,11]];stamp=float(observation.get('image_time',observation['sim_time']))
        if self.last_time is not None and stamp-self.last_time>.04:
            stopped=np.abs(jaws-self.last_jaws)/(stamp-self.last_time)<.05
            self.stall=np.where(close & stopped & (jaws>-.15)&(jaws<.2),self.stall+1,0)
            self.retained|=self.stall>=2
        self.last_jaws=jaws.copy();self.last_time=stamp
        command[[5,11]]=np.where(close|self.retained,-.17,.55)
        command=bounded_action(command)
        return command,dict(joint_chunk=actions.tolist(),raw_grips=raw.tolist(),retained=self.retained.tolist(),
            state=state.tolist(),seconds=time.monotonic()-start,action_representation='absolute 12-joint targets')
