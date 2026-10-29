"""One shared SmolVLA instance; task selection enters exclusively as language."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from peft import PeftModel
from safetensors.torch import load_file
from .vla_policy import load_base,base_model,make_batch,BASE_REVISION,CAMERA_NAMES
from .unified_features import network_rgb
from .workspace_features import WorkspaceFeatures
from .scene_goal_head import SceneGoalHead
from .scene_language import language_embedding
class SceneOnePolicy:
    def __init__(self,checkpoint,device='cuda'):
        self.directory=Path(checkpoint);self.device=device;self.config=json.loads((self.directory/'capture_config.json').read_text())
        if self.config['base_revision']!=BASE_REVISION:raise ValueError('Base revision mismatch')
        for name,key in [('adapter_model.safetensors','adapter_sha256'),('shared_head.safetensors','head_sha256')]:
            if hashlib.sha256((self.directory/name).read_bytes()).hexdigest()!=self.config[key]:raise ValueError('Weight hash mismatch')
        self.policy=PeftModel.from_pretrained(load_base(device,joint_count=12,state_count=32,action_count=12),self.directory).eval()
        self.base=base_model(self.policy);self.base.config.num_steps=self.base.model.config.num_steps=1
        self.tokenizer=self.base.model.vlm_with_expert.processor.tokenizer;self.hidden=None
        self.base.model.action_out_proj.register_forward_pre_hook(self._capture)
        self.head=SceneGoalHead(self.config['hidden_dim'],moving_grasp_frame=self.config.get('moving_grasp_frame',False),physical_phase_features=self.config.get('physical_phase_features',False),alignment_refinement=self.config.get('alignment_refinement',False),grasp_refinement=self.config.get('grasp_refinement',False),carry_goal_embedding=self.config.get('carry_goal_embedding',False),water_goal_embedding=self.config.get('water_goal_embedding',False),remember_task=self.config.get('remember_task',False),language_dim=self.config['language_dim'],grip_closure_refinement=self.config.get('grip_closure_refinement',False),assembly_step_refinement=self.config.get('assembly_step_refinement',False)).to(device)
        self.head.load_state_dict(load_file(str(self.directory/'shared_head.safetensors'),device=device));self.head.eval()
        for key in ('state_bias','state_scale','head_state_bias','head_state_scale','head_action_bias','head_action_scale'):
            setattr(self,key,np.array(self.config[key],np.float32))
        self.reset()
    def _capture(self,module,inputs):self.hidden=inputs[0][:,0].detach().float()
    def reset(self):
        from .water_vision import WaterVision
        from .water_depth import WaterDepth
        self.water=(WaterDepth() if self.config.get('water_depth_sensor') else WaterVision()) if self.config.get('water_goal_embedding') else None
        self.features=WorkspaceFeatures();self.memory=None;self.retained=np.zeros(2,bool);self.stall=np.zeros(2,int)
        self.last_stamp=None;self.last_instruction=None;self.last_result=None
    def predict(self,observation,instruction):
        start=time.monotonic();stamp=float(observation['image_time'])
        if self.last_stamp is not None and stamp==self.last_stamp and instruction==self.last_instruction:return self.last_result
        if self.config.get('remember_task') and self.last_instruction is not None and instruction!=self.last_instruction:self.reset()
        full_state,rotation=self.features(observation)
        water_position=None if self.water is None else torch.tensor(self.water(observation),device=self.device,dtype=torch.float32)[None]
        obs={**observation,'q':full_state[:32],**{n:network_rgb(observation[n]) for n in CAMERA_NAMES}}
        batch=make_batch([obs],[instruction],self.tokenizer,self.device,self.state_scale,self.state_bias)
        with torch.inference_mode(),torch.autocast(self.device,dtype=torch.bfloat16,enabled=self.device=='cuda'):
            self.base.predict_action_chunk(batch,noise=torch.zeros((1,1,32),device=self.device))
        state=torch.tensor((full_state-self.head_state_bias)/self.head_state_scale,device=self.device)[None]
        rotation=torch.tensor(rotation,device=self.device)[None]
        with torch.inference_mode():output,self.memory,phase,grasp,_=self.head(self.hidden,state,rotation,self.memory,water_position=water_position,language=language_embedding(self.base,batch))
        goal=output[0].cpu().numpy()*self.head_action_scale+self.head_action_bias
        raw_grips=goal[10:12].copy();close=raw_grips<.19
        jaws=np.asarray(observation['q'])[[5,11]];velocity=np.asarray(observation.get('q_velocity',np.zeros(12)))[[5,11]]
        self.stall=np.where(close&(np.abs(velocity)<.05)&(jaws>-.15)&(jaws<.2),self.stall+1,0)
        self.retained|=self.stall>=2;self.retained&=jaws>-.155
        goal[10:12]=np.where(close|self.retained,-.17,.55)
        decoded=goal.copy();decoded[3:6]-=goal[:3]
        if not np.isfinite(decoded).all():raise RuntimeError('Nonfinite shared policy action')
        command=self.features.codec.decode(decoded,observation['q'])
        self.last_stamp=stamp;self.last_instruction=instruction
        info=dict(delta=goal.tolist(),state=full_state.tolist(),raw_grips=raw_grips.tolist(),retained=self.retained.tolist(),
            phase_probabilities=phase.softmax(-1)[0].cpu().tolist(),grasp_probabilities=grasp.softmax(-1)[0].cpu().tolist(),
            seconds=time.monotonic()-start,observation_source='synchronized RGB-D images and joint encoders only',
            action_representation='learned world TCP goals, axes and jaws')
        if self.water is not None:info['water_position']=self.water.position.tolist();info['water_detection']=self.water.last_info
        if self.config.get('remember_task'):info['language_task_probabilities']=self.memory[1][0].cpu().tolist()
        self.last_result=command,info;return self.last_result
