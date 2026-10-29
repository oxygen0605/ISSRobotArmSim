"""One language-conditioned recurrent policy with learned goal embeddings."""
import torch
from torch import nn
class WorkspaceGoalHead(nn.Module):
    def __init__(self,hidden_dim,state_dim=53,moving_grasp_frame=False,physical_phase_features=False,alignment_refinement=False,grasp_refinement=False,carry_goal_embedding=False,water_goal_embedding=False,remember_task=False):
        super().__init__();self.moving_grasp_frame=moving_grasp_frame;self.physical_phase_features=physical_phase_features;self.alignment_refinement=alignment_refinement;self.grasp_refinement=grasp_refinement;self.carry_goal_embedding=carry_goal_embedding;self.water_goal_embedding=water_goal_embedding;self.remember_task=remember_task
        self.visual=nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,64),nn.SiLU())
        self.state=nn.Sequential(nn.Linear(state_dim,128),nn.SiLU())
        self.memory=nn.GRU(192,128,batch_first=True)
        self.phase=nn.Sequential(nn.Linear(128+state_dim,96),nn.SiLU(),nn.Linear(96,10))
        if physical_phase_features:self.physical_phase=nn.Sequential(nn.Linear(34,96),nn.SiLU(),nn.Linear(96,96),nn.SiLU(),nn.Linear(96,10))
        if alignment_refinement:self.advance=nn.Sequential(nn.Linear(6,32),nn.SiLU(),nn.Linear(32,1))
        if grasp_refinement:self.grasp_advance=nn.Sequential(nn.Linear(12,32),nn.SiLU(),nn.Linear(32,2))
        if carry_goal_embedding:
            self.carry_goal=nn.Parameter(torch.zeros(3));self.carry_axis_hold=nn.Parameter(torch.zeros(2))
        if water_goal_embedding:self.water_goal=nn.Parameter(torch.zeros(3))
        if remember_task:self.register_buffer('phase_task_ids',torch.tensor([0,0,1,1,2,2,2,2,2,2]))
        self.pregrasp=nn.Sequential(nn.Linear(128+state_dim,64),nn.SiLU(),nn.Linear(64,6))
        self.action=nn.Sequential(nn.Linear(336,192),nn.SiLU(),nn.Linear(192,12))
        self.left_hold=nn.Parameter(torch.randn(5,3)*.15)
        self.right_idle=nn.Parameter(torch.randn(4,5)*.15)
        self.right_orient=nn.Parameter(torch.randn(3)*.15)
        self.centre_relative=nn.Parameter(torch.randn(3)*.15)
        self.centre_goal=nn.Parameter(torch.randn(3)*.2)
        self.relative_goals=nn.Parameter(torch.randn(2,3)*.2)
        self.object_axis_goal=nn.Parameter(torch.randn(4)*.2)
        self.water_axis_hold=nn.Parameter(torch.randn(2)*.15)
        self.grasp_hold_offsets=nn.Parameter(torch.randn(2,5)*.1)
        self.register_buffer('state_bias',torch.zeros(state_dim));self.register_buffer('state_scale',torch.ones(state_dim))
        self.register_buffer('action_bias',torch.zeros(12));self.register_buffer('action_scale',torch.ones(12))
        self.register_buffer('relative_bias',torch.zeros(3));self.register_buffer('relative_scale',torch.ones(3))
    def forward(self,hidden,state,rotation,memory=None,water_position=None):
        task_memory=None
        if self.remember_task and isinstance(memory,tuple):memory,task_memory=memory
        step=hidden.ndim==2
        if step:
            hidden,state,rotation=hidden[:,None],state[:,None],rotation[:,None]
            if water_position is not None:water_position=water_position[:,None]
        values=torch.cat([self.visual(hidden),self.state(state)],-1);recurrent,memory=self.memory(values,memory)
        context=torch.cat([recurrent,state],-1);logits=self.phase(context);pre=self.pregrasp(context)
        if self.physical_phase_features:
            raw=state*self.state_scale+self.state_bias
            centre=self.centre_goal*self.action_scale[:3]+self.action_bias[:3]
            relatives=self.relative_goals*self.relative_scale+self.relative_bias
            previous_relative=raw[...,15:18]+raw[...,26:29]-raw[...,3:6]
            distance=torch.linalg.vector_norm(raw[...,38:47].reshape(*raw.shape[:-1],3,3),dim=-1)/.03
            axes=torch.stack([torch.linalg.vector_norm(raw[...,29:31],dim=-1),torch.linalg.vector_norm(raw[...,32:34],dim=-1)],-1)/.2
            grasp_axes=torch.stack([torch.linalg.vector_norm(raw[...,47:49],dim=-1),torch.linalg.vector_norm(raw[...,50:52],dim=-1)],-1)/.2
            old_prob=logits.softmax(-1)
            task=torch.stack([old_prob[...,:2].sum(-1),old_prob[...,2:4].sum(-1),old_prob[...,4:].sum(-1)],-1)
            if self.remember_task:
                if task_memory is None:task_memory=task[:,0].detach()
                task=task_memory[:,None].expand_as(task)
            physical=torch.cat([distance,axes,grasp_axes,(raw[...,:3]-centre)/.03,(raw[...,10:12]-.19)/.36,
                (raw[...,18:20]-.19)/.36,raw[...,36:38].clamp(-2,2),(raw[...,12:15]-raw[...,:3])/.03,
                (raw[...,26:29]-relatives[0])/.03,(raw[...,26:29]-relatives[1])/.03,
                (previous_relative-relatives[0])/.03,(previous_relative-relatives[1])/.03,task],-1)
            logits=self.physical_phase(physical)
            original_phase=logits.argmax(-1)
            if self.alignment_refinement:
                relative=raw[...,26:29]-relatives[0]
                advance_features=torch.cat([torch.linalg.vector_norm(relative[...,:2],dim=-1,keepdim=True)/.01,relative[...,2:3]/.01,
                    axes,(previous_relative[...,2:3]-relatives[0,2])/.01,torch.linalg.vector_norm(relative,dim=-1,keepdim=True)/.01],-1)
                advance=self.advance(advance_features).squeeze(-1)
                mass=torch.logsumexp(logits[...,7:9],dim=-1)
                logits=torch.cat([logits[...,:7],(mass+torch.nn.functional.logsigmoid(-advance))[...,None],
                    (mass+torch.nn.functional.logsigmoid(advance))[...,None],logits[...,9:]],-1)
            if self.grasp_refinement:
                relative_axis=torch.stack([torch.linalg.vector_norm(raw[...,47:49],dim=-1),torch.linalg.vector_norm(raw[...,50:52],dim=-1)],-1)/.2
                goal_errors=torch.stack([torch.linalg.vector_norm(raw[...,12:15]-raw[...,:3],dim=-1),torch.linalg.vector_norm(raw[...,12:15]+raw[...,15:18]-raw[...,:3]-raw[...,3:6],dim=-1)],-1)/.03
                grasp_features=torch.cat([(raw[...,10:12]-.19)/.36,(raw[...,18:20]-.19)/.36,raw[...,36:38].abs().clamp_max(2)/.2,
                    torch.linalg.vector_norm(raw[...,41:47].reshape(*raw.shape[:-1],2,3),dim=-1)/.02,relative_axis,goal_errors],-1)
                ready=self.grasp_advance(grasp_features)
                for i,(a,b) in enumerate(((2,3),(4,5))):
                    mass=torch.logsumexp(logits[...,a:b+1],dim=-1)
                    replacement=torch.stack([mass+torch.nn.functional.logsigmoid(-ready[...,i]),mass+torch.nn.functional.logsigmoid(ready[...,i])],-1)
                    active=((original_phase==a)|(original_phase==b))[...,None]
                    replacement=torch.where(active,replacement,logits[...,a:b+1])
                    logits=torch.cat([logits[...,:a],replacement,logits[...,b+1:]],-1)
        if self.remember_task:
            allowed=self.phase_task_ids[None,None,:]==task_memory.argmax(-1)[:,None,None]
            logits=logits.masked_fill(~allowed,-1e4)
            memory=(memory,task_memory)
        prob=logits.softmax(-1);grasp_prob=pre.reshape(*pre.shape[:-1],2,3).softmax(-1)
        soft_prob=prob
        hard=torch.nn.functional.one_hot(prob.argmax(-1),prob.shape[-1]).to(prob.dtype)
        prob=hard+prob-prob.detach() if self.training else hard
        output=self.action(torch.cat([values,recurrent,soft_prob,grasp_prob.flatten(-2)],-1))
        raw=state*self.state_scale+self.state_bias
        left=raw[...,:3];right=left+raw[...,3:6];previous=raw[...,12:15]
        penxy=raw[...,29:31];capxy=raw[...,32:34]
        penaxis=torch.cat([penxy,(1-penxy.square().sum(-1,keepdim=True)).clamp_min(1e-8).sqrt()],-1)
        capaxis=torch.cat([capxy,(1-capxy.square().sum(-1,keepdim=True)).clamp_min(1e-8).sqrt()],-1)
        water=raw[...,20:23];pen=raw[...,23:26]+.057*penaxis
        cap=raw[...,23:26]+raw[...,26:29]-.0275*capaxis
        local=output[...,:6].reshape(*output.shape[:-1],2,3)*.05
        if self.moving_grasp_frame:
            object_error=torch.stack([pen-left,cap-right],-2)
            gap=torch.einsum('...aji,...aj->...ai',rotation,object_error)-raw[...,41:47].reshape(*raw.shape[:-1],2,3)
            active=torch.stack([soft_prob[...,2]+soft_prob[...,4],soft_prob[...,4]],-1)
            # The moving TCP is robot geometry; the network learns its object-relative goal.
            hard_active=torch.stack([prob[...,2]+prob[...,4],prob[...,4]],-1)
            local=local-gap*hard_active[...,None]
        displacement=torch.einsum('...aij,...aj->...ai',rotation,local)
        left_hold=self.left_hold*self.action_scale[:3]
        right_idle_pos=self.right_idle[:,:3]*self.action_scale[3:6]
        right_idle_axis=self.right_idle[:,3:]*self.action_scale[8:10]
        centre=self.centre_goal*self.action_scale[:3]+self.action_bias[:3]
        relatives=self.relative_goals*self.relative_scale+self.relative_bias
        final_axes=self.object_axis_goal*self.action_scale[6:10]
        tool=raw[...,6:10];objects=torch.cat([penxy,capxy],-1)
        servo=tool-objects+final_axes
        delta_axis=output[...,6:10]*.1
        hard_grasp=torch.nn.functional.one_hot(grasp_prob.argmax(-1),3).to(grasp_prob.dtype)
        grasp_prob=hard_grasp+grasp_prob-grasp_prob.detach() if self.training else hard_grasp
        lh=grasp_prob[...,0,2,None];rh=grasp_prob[...,1,2,None]
        left_grasp=(pen+displacement[...,0,:])*(1-lh)+(left+self.grasp_hold_offsets[0,:3]*self.action_scale[:3])*lh
        right_grasp=(cap+displacement[...,1,:])*(1-rh)+(right+self.grasp_hold_offsets[1,:3]*self.action_scale[3:6])*rh
        left_grasp_axis=(penxy+delta_axis[...,:2])*(1-lh)+(tool[...,:2]+self.grasp_hold_offsets[0,3:]*self.action_scale[6:8])*lh
        right_grasp_axis=(capxy+delta_axis[...,2:])*(1-rh)+(tool[...,2:]+self.grasp_hold_offsets[1,3:]*self.action_scale[8:10])*rh
        ls=[water+displacement[...,0,:],left+left_hold[0],left_grasp,
            previous+displacement[...,0,:],left_grasp,
            left+left_hold[1],torch.zeros_like(left)+centre,left+left_hold[2],left+left_hold[3],left+left_hold[4]]
        if self.water_goal_embedding:
            observed_water=raw[...,20:23] if water_position is None else water_position
            water_target=observed_water+torch.einsum('...ij,j->...i',rotation[...,0,:,:],self.water_goal*.05)
            ls[0]=water_target;ls[1]=water_target
        if self.carry_goal_embedding:ls[3]=torch.zeros_like(left)+self.carry_goal*self.action_scale[:3]+self.action_bias[:3]
        rs=[right+right_idle_pos[i] for i in range(4)]
        rs += [right_grasp,right+self.right_orient*self.action_scale[3:6],
            right+(centre-left)+self.centre_relative*self.action_scale[3:6]]
        rs += [ls[i]+raw[...,3:6]+relatives[0 if i==7 else 1]-raw[...,26:29] for i in (7,8,9)]
        ax=[]
        ax.append(torch.cat([tool[...,:2]+delta_axis[...,:2],tool[...,2:]+right_idle_axis[0]],-1))
        ax.append(torch.cat([tool[...,:2]+self.water_axis_hold*self.action_scale[6:8],tool[...,2:]+right_idle_axis[1]],-1))
        ax.append(torch.cat([left_grasp_axis,tool[...,2:]+right_idle_axis[2]],-1))
        ax.append(torch.cat([tool[...,:2]+delta_axis[...,:2],tool[...,2:]+right_idle_axis[3]],-1))
        ax.append(torch.cat([left_grasp_axis,right_grasp_axis],-1))
        if self.water_goal_embedding:ax[0]=ax[1]
        if self.carry_goal_embedding:ax[3]=torch.cat([tool[...,:2]+self.carry_axis_hold*self.action_scale[6:8],tool[...,2:]+right_idle_axis[3]],-1)
        ax += [servo]*5
        candidates=torch.stack([torch.cat([l,r,a],-1) for l,r,a in zip(ls,rs,ax)],-2)
        goals=(candidates*prob[...,None]).sum(-2)
        jaws=self.action_bias[10:12]+output[...,10:12]*self.action_scale[10:12]
        action=(torch.cat([goals,jaws],-1)-self.action_bias)/self.action_scale
        return (action[:,0] if step else action),memory,(logits[:,0] if step else logits),(pre.reshape(*pre.shape[:-1],2,3)[:,0] if step else pre.reshape(*pre.shape[:-1],2,3)),(output[:,0] if step else output)
    def prototype_loss(self,centre_target,relative_targets,axis_target,carry_target=None,water_target=None):
        loss=(self.centre_goal-centre_target).square().mean()+(self.relative_goals-relative_targets).square().mean()
        loss+=(self.object_axis_goal-axis_target).square().mean()
        for value in (self.left_hold,self.right_idle,self.right_orient,self.centre_relative,self.water_axis_hold,self.grasp_hold_offsets):loss+=value.square().mean()
        if self.carry_goal_embedding and carry_target is not None:loss+=(self.carry_goal-carry_target).square().mean()+self.carry_axis_hold.square().mean()
        if self.water_goal_embedding and water_target is not None:loss+=(self.water_goal-water_target).square().mean()
        return loss
