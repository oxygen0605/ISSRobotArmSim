"""Shared causal actor with jointly learned phase recognition; no task routing."""
import torch
from torch import nn
class PhaseActionHead(nn.Module):
    def __init__(self,hidden_dim,memory_dim=128,phase_count=7,state_dim=32):
        super().__init__()
        self.visual=nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,64),nn.SiLU())
        self.state=nn.Sequential(nn.Linear(state_dim,128),nn.SiLU())
        self.memory=nn.GRU(204,memory_dim,batch_first=True)
        self.phase=nn.Sequential(nn.Linear(memory_dim+state_dim,64),nn.SiLU(),nn.Linear(64,phase_count))
        self.output=nn.Sequential(nn.Linear(204+memory_dim+phase_count,192),nn.SiLU(),nn.Linear(192,12))
        nn.init.zeros_(self.output[-1].weight);nn.init.zeros_(self.output[-1].bias)
    def forward(self,hidden,state,coarse,baseline,memory=None):
        step=hidden.ndim==2
        if step:hidden,state,coarse,baseline=[v[:,None,:] for v in (hidden,state,coarse,baseline)]
        values=torch.cat([self.visual(hidden),self.state(state),coarse-baseline],dim=-1)
        recurrent,memory=self.memory(values,memory)
        logits=self.phase(torch.cat([recurrent,state],dim=-1))
        action=baseline+self.output(torch.cat([values,recurrent,logits.softmax(-1)],dim=-1))
        return (action[:,0] if step else action),memory,(logits[:,0] if step else logits)
