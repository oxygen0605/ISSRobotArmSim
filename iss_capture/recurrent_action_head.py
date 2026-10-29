"""One recurrent action decoder for all instructions; no hand-coded task phases."""
import torch
from torch import nn
class RecurrentActionHead(nn.Module):
    def __init__(self,hidden_dim,memory_dim=128):
        super().__init__()
        self.visual=nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,64),nn.SiLU())
        self.state=nn.Sequential(nn.Linear(32,128),nn.SiLU())
        self.memory=nn.GRU(204,memory_dim,batch_first=True)
        self.output=nn.Sequential(nn.Linear(204+memory_dim,192),nn.SiLU(),nn.Linear(192,12))
        nn.init.zeros_(self.output[-1].weight);nn.init.zeros_(self.output[-1].bias)
    def forward(self,hidden,state,coarse,baseline,memory=None):
        step=hidden.ndim==2
        if step:hidden,state,coarse,baseline=[v[:,None,:] for v in (hidden,state,coarse,baseline)]
        values=torch.cat([self.visual(hidden),self.state(state),coarse-baseline],dim=-1)
        recurrent,memory=self.memory(values,memory)
        actions=baseline+self.output(torch.cat([values,recurrent],dim=-1))
        return (actions[:,0] if step else actions),memory
