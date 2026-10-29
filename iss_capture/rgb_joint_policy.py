"""SmolVLA with RGB + raw joint position/velocity + language; joint actions."""
from pathlib import Path
from functools import lru_cache
import hashlib,json,time
import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from .vla_policy import load_base,base_model,make_batch,BASE_REVISION

VARIANT="smolvla_rgb_joint_language_v1"
CAMERAS=("front","side","previous_front")
INPUT_KEYS=(*CAMERAS,"q","q_velocity")

def select_inputs(observation,positions_only=False):
    keys=(*CAMERAS,'q') if positions_only else INPUT_KEYS
    return {key:observation[key] for key in keys}

def raw_state(observation,positions_only=False):
    q=np.asarray(observation["q"],np.float32)
    if positions_only:
        if q.shape!=(12,) or not np.isfinite(q).all():raise ValueError('Expected twelve encoder angles')
        return q.copy()
    v=np.asarray(observation["q_velocity"],np.float32)
    if q.shape!=(12,) or v.shape!=(12,) or not np.isfinite(np.r_[q,v]).all():
        raise ValueError("Expected twelve encoder angles and twelve velocities")
    return np.r_[q,v].astype(np.float32)

def rgb(image,size):
    image=np.asarray(image,dtype=np.uint8)
    if image.ndim!=3 or image.shape[2]!=3:raise ValueError("RGB image required")
    if image.shape[:2]!=(size,size):
        image=np.asarray(Image.fromarray(image).resize((size,size),Image.Resampling.BILINEAR))
    return image

@lru_cache(maxsize=1024)
def read_rgb(path,size):
    with Image.open(path) as image:return rgb(np.asarray(image.convert("RGB")),size)

def load_demonstrations(directory,action_space='joint',joint_history=False,positions_only=False):
    """RGB and raw joints are inputs; optional Cartesian poses are labels only."""
    rows=[]
    if action_space=='cartesian':
        from .capping_pose import PoseCodec
        codec=PoseCodec()
    for path in sorted(Path(directory).glob("*/result.json")):
        result=json.loads(path.read_text())
        if result.get("split") not in ("train","validation"):continue
        supervision={}
        if result.get("split")=="train" and path.with_name("supervision.json").exists():
            supervision=json.loads(path.with_name("supervision.json").read_text())
        if not result.get("success") and not supervision.get("valid_prefix_frames"):continue
        records=json.loads(path.with_name("samples.json").read_text())
        if not result.get("success"):records=records[:supervision["valid_prefix_frames"]]
        last=None;history=[]
        for record in records:
            stamp=record["image_time"]
            if last is not None and stamp<=last:continue
            last=stamp
            state=raw_state(record,positions_only)
            if joint_history:
                past=[(history[-lag] if len(history)>=lag else history[0] if history else state[:12])[[5,11]] for lag in (1,2,3,4)]
                history.append(state[:12].copy());state=np.r_[state,*past].astype(np.float32)
            action=np.asarray(record["action"],np.float32)
            if action_space=='cartesian':
                if 'goal' in record:
                    pose=np.asarray(record['goal'],np.float32).copy();pose[3:6]+=pose[:3]
                else:
                    positions=codec.positions(action);axes=codec.axes(action)
                    pose=np.r_[positions.ravel(),axes[:,:2].ravel(),action[[5,11]]].astype(np.float32)
            else:pose=None
            rows.append(dict(episode=path.parent.name,task=result["task"],split=result["split"],
                phase_end_goals_valid=bool(result.get("success") or supervision.get("phase_end_goals_valid")),
                instruction=record["instruction"],phase=record.get("teacher_phase","unknown"),state=state,q=state[:12],action=action,
                delta=action-state[:12],pose=pose,
                paths={name:str(path.parent/(record["frame"]+"_"+name+".png")) for name in CAMERAS}))
    return rows

def action_target(row,mode):
    if mode=='cartesian':return row['pose'].copy()
    if mode=='absolute':return row['action'].copy()
    result=row['delta'].copy()
    if mode=='arm_delta_grip_absolute':result[[5,11]]=row['action'][[5,11]]
    return result

def decode_action(value,q,mode):
    if mode=='absolute':return value.copy()
    command=q+value
    if mode=='arm_delta_grip_absolute':command[[5,11]]=value[[5,11]]
    return command

def observations(rows,size):
    return [{**{name:read_rgb(row["paths"][name],size) for name in CAMERAS},"q":row["state"]} for row in rows]

class RGBJointPolicy:
    rgb_only=True
    def __init__(self,checkpoint,device="cuda"):
        self.directory=Path(checkpoint);self.device=device
        self.config=json.loads((self.directory/"capture_config.json").read_text())
        if self.config["base_revision"]!=BASE_REVISION:raise ValueError("Base revision mismatch")
        if hashlib.sha256((self.directory/"adapter_model.safetensors").read_bytes()).hexdigest()!=self.config["adapter_sha256"]:raise ValueError("Adapter hash mismatch")
        self.policy=PeftModel.from_pretrained(load_base(device,joint_count=12,state_count=self.config.get('state_dim',24),action_count=12,
            policy_image_size=self.config["policy_image_size"],chunk_size=self.config.get("chunk_size",1)),self.directory).eval()
        self.base=base_model(self.policy)
        self.base.config.num_steps=self.base.model.config.num_steps=self.config["denoising_steps"]
        self.tokenizer=self.base.model.vlm_with_expert.processor.tokenizer
        for key in ("state_bias","state_scale","action_bias","action_scale"):
            setattr(self,key,np.asarray(self.config[key],np.float32))
        if self.config.get('action_mode')=='cartesian':
            from .capping_pose import PoseCodec
            self.codec=PoseCodec()
        self.reset()
    def reset(self):self.joint_history=[];self.pending_actions=[]
    def predict(self,observation,instruction):
        keys=(*CAMERAS,"q") if self.config.get("positions_only") else INPUT_KEYS
        if set(observation)!=set(keys):raise ValueError("RGB-only policy accepts only images and raw joint state")
        start=time.monotonic();state=raw_state(observation,self.config.get("positions_only",False))
        if self.config.get('joint_history'):
            past=[(self.joint_history[-lag] if len(self.joint_history)>=lag else self.joint_history[0] if self.joint_history else state[:12])[[5,11]] for lag in (1,2,3,4)]
            self.joint_history.append(state[:12].copy());self.joint_history=self.joint_history[-4:]
            state=np.r_[state,*past].astype(np.float32)
        if getattr(self,"pending_actions",[]):
            command=self.pending_actions.pop(0)
            return command,dict(seconds=time.monotonic()-start,input_keys=list(keys),state_dim=len(state),
                depth_used=False,geometry_features_used=False,action_representation="cached learned action chunk",
                chunk_remaining=len(self.pending_actions))
        obs={name:rgb(observation[name],self.config["policy_image_size"]) for name in CAMERAS};obs["q"]=state
        batch=make_batch([obs],[instruction],self.tokenizer,self.device,self.state_scale,self.state_bias)
        with torch.inference_mode(),torch.autocast(self.device,dtype=torch.bfloat16,enabled=self.device=="cuda"):
            sequence=self.base.predict_action_chunk(batch,noise=torch.zeros((1,self.config.get("chunk_size",1),32),device=self.device))[0].float().cpu().numpy()
        value=sequence[0]
        if self.config.get("chunk_size",1)>1:
            if self.config.get("action_mode")!="absolute":raise ValueError("Chunks require absolute joint targets")
            future=sequence[1:self.config.get("execution_steps",1)]*self.action_scale+self.action_bias
            self.pending_actions=[]
            for row in future:
                command=row.copy()
                values=np.asarray(self.config["gripper_values"])
                for index in (5,11):command[index]=values[np.argmin(np.abs(values-command[index]))]
                self.pending_actions.append(np.clip(command,np.asarray(self.config["joint_lower"]),np.asarray(self.config["joint_upper"])))
        self.last_state=state.copy();self.last_normalized_action=value.copy()
        self.last_language_batch={key:batch[key] for key in ('observation.language.tokens','observation.language.attention_mask')}
        delta=value*self.action_scale+self.action_bias
        if self.config.get('action_mode')=='cartesian':
            goal=delta.copy();goal[3:6]-=goal[:3]
            command=self.codec.decode(goal,state[:12])
        else:command=decode_action(delta,state[:12],self.config.get('action_mode','delta'))
        if self.config.get('gripper_values'):
            values=np.asarray(self.config['gripper_values'],np.float32)
            for index in (5,11):command[index]=values[np.argmin(np.abs(values-command[index]))]
        if not np.isfinite(command).all():raise RuntimeError("Non-finite joint action")
        # Physical joint bounds only; no task-specific corrections or IK.
        command=np.clip(command,np.asarray(self.config["joint_lower"]),np.asarray(self.config["joint_upper"]))
        return command,dict(seconds=time.monotonic()-start,input_keys=list(keys),
            model_state=state.tolist(),rgb_hashes={name:hashlib.sha256(obs[name].tobytes()).hexdigest() for name in CAMERAS},
            state_dim=len(state),depth_used=False,geometry_features_used=False,
            action_representation=self.config.get("action_mode","delta"),predicted_delta=(command-state[:12]).tolist())
