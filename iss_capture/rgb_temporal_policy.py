"""Learned temporal readout of SmolVLA; no object geometry, depth or contacts."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from .rgb_joint_policy import RGBJointPolicy
from .scene_language import language_embedding
VARIANT="smolvla_rgb_joint_language_temporal_v1"
PHASES=("water_approach","water_hold","pen_grasp","pen_carry","dual_grasp","dual_orient","dual_centre","dual_align","dual_insert","dual_hold")
def phase_label(task,phase):
    if task=="task1":return PHASES.index("water_hold" if phase=="hold" else "water_approach")
    if task=="task2":return PHASES.index("pen_carry" if phase=="carry" else "pen_grasp")
    return PHASES.index("dual_"+phase)

class RGBTemporalHead(torch.nn.Module):
    def __init__(self,hidden_dim,state_dim,language_dim,width=192,assembly_anchors=False,discrete_goals=False):
        super().__init__()
        self.hidden_norm=torch.nn.LayerNorm(hidden_dim)
        self.language_norm=torch.nn.LayerNorm(language_dim)
        self.input=torch.nn.Sequential(torch.nn.Linear(hidden_dim+state_dim+language_dim+12,width),torch.nn.GELU())
        self.memory=torch.nn.GRU(width,width,batch_first=True)
        self.action=torch.nn.Linear(width,12)
        self.phase=torch.nn.Linear(width,len(PHASES))
        self.assembly_anchors=assembly_anchors;self.discrete_goals=discrete_goals
        if assembly_anchors:self.anchors=torch.nn.Parameter(torch.zeros(5,12))
        torch.nn.init.zeros_(self.action.weight);torch.nn.init.zeros_(self.action.bias)
    def forward(self,hidden,state,language,core,memory=None):
        x=torch.cat([self.hidden_norm(hidden),state,self.language_norm(language),core],-1)
        x=self.input(x);x,memory=self.memory(x,memory)
        phase=self.phase(x);baseline=core;correction=self.action(x)
        if self.assembly_anchors:
            probabilities=(torch.nn.functional.one_hot(phase.argmax(-1),len(PHASES)).to(core.dtype)
                           if self.discrete_goals else phase.softmax(-1))
            assembly=probabilities[...,5:]
            baseline=(1-assembly.sum(-1,keepdim=True))*core+assembly@self.anchors
            if self.discrete_goals:correction=correction*probabilities[...,4:5]
        return baseline+correction,memory,phase

class RGBTemporalPolicy(RGBJointPolicy):
    def __init__(self,checkpoint,device="cuda"):
        super().__init__(checkpoint,device)
        if hashlib.sha256((self.directory/'rgb_temporal_head.safetensors').read_bytes()).hexdigest()!=self.config['temporal_head_sha256']:raise ValueError('Temporal head hash mismatch')
        self.temporal=RGBTemporalHead(self.config["hidden_dim"],self.config["state_dim"],self.config["language_dim"],self.config["temporal_width"],self.config.get("assembly_anchors",False),self.config.get("discrete_goals",False)).to(device)
        self.temporal.load_state_dict(load_file(str(self.directory/"rgb_temporal_head.safetensors"),device=device))
        self.temporal.eval()
        self.base.model.action_out_proj.register_forward_pre_hook(self._capture_hidden)
        self.language=None;self.last_instruction=None;self.memory=None
    def _capture_hidden(self,module,args):self.hidden=args[0][:,0].detach().float()
    def reset(self):
        super().reset();self.memory=None;self.language=None;self.last_instruction=None
    def predict(self,observation,instruction):
        if self.last_instruction is not None and instruction!=self.last_instruction:self.reset()
        _,debug=super().predict(observation,instruction)
        state=torch.tensor((self.last_state-self.state_bias)/self.state_scale,device=self.device)[None,None,:]
        core=torch.tensor(self.last_normalized_action,device=self.device)[None,None,:]
        if self.language is None or self.last_instruction!=instruction:
            self.language=language_embedding(self.base,self.last_language_batch)
            self.last_instruction=instruction
        with torch.inference_mode():
            normalized,self.memory,phase=self.temporal(self.hidden[:,None,:],state,self.language[:,None,:],core,self.memory)
        command=normalized[0,0].cpu().numpy()*self.action_scale+self.action_bias
        values=np.asarray(self.config["gripper_values"])
        for index in (5,11):command[index]=values[np.argmin(np.abs(values-command[index]))]
        command=np.clip(command,np.asarray(self.config["joint_lower"]),np.asarray(self.config["joint_upper"]))
        if not np.isfinite(command).all():raise RuntimeError("Non-finite recurrent action")
        return command,{**debug,"phase_probabilities":phase.softmax(-1)[0,0].cpu().tolist(),
            "action_representation":"learned recurrent absolute joint goals","temporal_memory":True}
