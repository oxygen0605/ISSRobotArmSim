from types import SimpleNamespace
import torch
from train_rgb_chunks import chunk_training_loss


class SquaredError(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.4))
    def forward(self,images,masks,tokens,language_masks,state,actions,**kwargs):
        return (self.weight*state[:,None,:]-actions).square()


def test_accumulation_matches_full_batch_with_unequal_valid_chunk_lengths():
    model=SquaredError()
    base=SimpleNamespace(model=model,prepare_images=lambda b:([],[]),prepare_action=lambda b:b['action'],prepare_state=lambda b:b['observation.state'])
    batch={'action':torch.arange(4*3*12,dtype=torch.float32).reshape(4,3,12)/100,
           'action_valid':torch.tensor([[1,0,0],[1,1,0],[1,1,1],[1,0,0]],dtype=torch.bool),
           'observation.state':torch.arange(4*12,dtype=torch.float32).reshape(4,12)/10,
           'observation.language.tokens':torch.zeros(4,1),
           'observation.language.attention_mask':torch.ones(4,1)}
    axis=torch.ones(12);axis[[5,11]]=2
    chunk_training_loss(base,batch,axis,True).backward()
    full_gradient=model.weight.grad.clone();model.weight.grad=None
    for start in (0,2):
        micro={k:v[start:start+2] for k,v in batch.items()}
        fraction=micro['action_valid'].sum()/batch['action_valid'].sum()
        (chunk_training_loss(base,micro,axis,True)*fraction).backward()
    torch.testing.assert_close(model.weight.grad,full_gradient)
