"""One learned readout shared across tasks; conditioned on SmolVLA hidden state."""
import torch
from torch import nn
class SharedActionHead(nn.Module):
    def __init__(self,hidden_dim,version=1):
        super().__init__();self.version=version
        self.visual=nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,64),nn.SiLU())
        self.state=nn.Sequential(nn.Linear(32,128),nn.SiLU())
        self.output=nn.Sequential(nn.Linear(204 if version==2 else 192,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU(),nn.Linear(128,12))
        nn.init.zeros_(self.output[-1].weight);nn.init.zeros_(self.output[-1].bias)
    def forward(self,hidden,state,coarse,baseline=None):
        values=[self.visual(hidden),self.state(state)]
        if self.version==2:values.append(coarse-baseline)
        return (baseline if self.version==2 else coarse)+self.output(torch.cat(values,dim=-1))

def previous_pose_baseline(raw_state,world=False):
    result=torch.cat([raw_state[:,12:18],raw_state[:,6:10],raw_state[:,18:20]],dim=1)
    if world:result[:,3:6]+=result[:,:3]
    return result
