"""Actual SmolVLA inference with local task-specific LoRA adapters."""
from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
import time
import re
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from peft import LoraConfig, PeftModel, get_peft_model
from .config import ROOT

BASE_ID = "lerobot/smolvla_base"
BASE_REVISION = "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"
PROCESSOR_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
BASE_DIR = ROOT / "models/smolvla_base"
PROCESSOR_DIR = ROOT / "models/smolvlm_processor"
ADAPTER_DIR = ROOT / "artifacts/smolvla_capture"
CAPTURE_TASK = "Follow the blue droplet moving upward and touch it with the green sponge."
HOLD_TASK = "Keep the robot still. Do not touch the droplet."
SCALE = np.array([.65, .30, .35, 1, 1, 1], dtype=np.float32)
CAMERA_NAMES = ("front", "side", "previous_front")


def load_base(device="cuda", chunk_size=1, policy_image_size=256, joint_count=6, state_count=None, action_count=None):
    if not (BASE_DIR / "model.safetensors").exists() or not (PROCESSOR_DIR / "config.json").exists():
        raise FileNotFoundError("先に bash scripts/setup_runtime.sh でSmolVLAを取得してください。")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPUを利用できません。GPUドライバーと実行権限を確認してください。")
    torch.set_num_threads(4)
    config = PreTrainedConfig.from_pretrained(BASE_DIR)
    config.vlm_model_name = str(PROCESSOR_DIR)
    # All VLM weights are inside the SmolVLA checkpoint. Do not download or
    # substitute a second backbone; construct its architecture before loading.
    config.load_vlm_weights = False
    config.device = "cpu"
    config.chunk_size = int(chunk_size)
    config.n_action_steps = 1
    if joint_count not in (6,12):raise ValueError("Supported joint count: 6 or 12")
    state_count=joint_count if state_count is None else int(state_count)
    action_count=joint_count if action_count is None else int(action_count)
    if not 1<=state_count<=32 or not 1<=action_count<=32:raise ValueError("Feature dimensions must fit the 32-axis base")
    config.input_features["observation.state"].shape = (state_count,)
    config.output_features["action"].shape = (action_count,)
    if policy_image_size not in (256, 512):
        raise ValueError('Policy image size must be 256 or 512')
    config.resize_imgs_with_padding = (policy_image_size, policy_image_size)
    config.num_steps = 5
    config.push_to_hub = False
    # LeRobot 0.4.4 uses a dataclass config; PEFT expects Transformers' to_dict.
    # Keep this compatibility adapter local, without editing either dependency.
    config.to_dict = lambda: asdict(config)
    base = SmolVLAPolicy.from_pretrained(BASE_DIR, config=config, strict=True)
    base.name_or_path = BASE_ID
    return base.to(device)


def new_adapter(base, vision_lora_layers=0):
    targets = r"model\.vlm_with_expert\.lm_expert\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    if vision_lora_layers:
        attention = []
        for name, module in base.named_modules():
            match = re.search(r'vision_model\.encoder\.layers\.(\d+)\.self_attn\.', name)
            if match and isinstance(module, torch.nn.Linear):
                attention.append((int(match.group(1)), name))
        layers = sorted({index for index, _ in attention})[-vision_lora_layers:]
        if len(layers) != vision_lora_layers:
            raise ValueError('Requested vision attention layers were not found')
        names = [name for index, name in attention if index in layers]
        targets = '(?:' + targets + ')|(?:' + '|'.join(re.escape(name) for name in names) + ')'
        # PEFT still freezes the base weights; only the new low-rank matrices
        # train. These flags keep the selected vision adapters in training mode.
        base.config.freeze_vision_encoder = False
        base.config.train_expert_only = False
        base.model.vlm_with_expert.freeze_vision_encoder = False
        base.model.vlm_with_expert.train_expert_only = False
    config = LoraConfig(r=16, lora_alpha=32, lora_dropout=.05, bias="none",
                        target_modules=targets,
                        modules_to_save=["state_proj", "action_in_proj", "action_out_proj",
                                         "action_time_mlp_in", "action_time_mlp_out"])
    return get_peft_model(base, config)


def base_model(policy):
    return policy.get_base_model() if isinstance(policy, PeftModel) else policy


def make_batch(observations, tasks, tokenizer, device, state_scale=SCALE, state_bias=0.):
    encoded = tokenizer(tasks, padding="max_length", max_length=48, truncation=True, return_tensors="pt")
    batch = {"observation.language.tokens": encoded["input_ids"].to(device),
             "observation.language.attention_mask": encoded["attention_mask"].bool().to(device),
             "observation.state": torch.from_numpy((np.stack([o["q"] for o in observations]).astype(np.float32) - state_bias) / state_scale).to(device)}
    for i, name in enumerate(CAMERA_NAMES, 1):
        values = np.stack([o[name] for o in observations])
        batch[f"observation.images.camera{i}"] = torch.from_numpy(values).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32) / 255
    return batch


def training_loss(policy, batch, axis_weights=None, deterministic_endpoint=False):
    """Official SmolVLA flow-matching objective restricted to the explicitly configured axes."""
    base = base_model(policy)
    images, masks = base.prepare_images(batch)
    actions = base.prepare_action(batch)
    noise = torch.zeros_like(actions) if deterministic_endpoint else None
    endpoint_time = torch.ones(actions.shape[0], device=actions.device) if deterministic_endpoint else None
    loss = base.model(images, masks, batch["observation.language.tokens"],
                      batch["observation.language.attention_mask"], base.prepare_state(batch),
                      actions, noise=noise, time=endpoint_time)
    used=loss[..., :batch["action"].shape[-1]]
    if axis_weights is not None:used=used*axis_weights/axis_weights.mean()
    return used.mean()


class VLAController:
    def __init__(self, checkpoint=ADAPTER_DIR, device="cuda"):
        checkpoint = Path(checkpoint)
        if not (checkpoint / "adapter_model.safetensors").is_file():
            raise FileNotFoundError("水滴タスク用のVLA重みがありません。train_vla.pyで追加学習してください。")
        self.metadata = json.loads((checkpoint / "capture_config.json").read_text())
        if self.metadata["base_revision"] != BASE_REVISION or self.metadata["action_scale"] != SCALE.tolist():
            raise ValueError("追加学習モデルの設定がこの実装と一致しません。")
        self.device = device
        self.policy = PeftModel.from_pretrained(load_base(device, policy_image_size=self.metadata.get('policy_image_size', 256)), checkpoint).eval()
        self.base = base_model(self.policy)
        self.tokenizer = self.base.model.vlm_with_expert.processor.tokenizer

    def predict(self, observation, instruction=CAPTURE_TASK):
        # No target coordinates, velocities, teacher policy or inverse kinematics
        # are accepted here. Vision/language/proprioception are the only inputs.
        batch = make_batch([observation], [instruction], self.tokenizer, self.device)
        started = time.monotonic()
        noise = torch.zeros((1, 1, 32), device=self.device)
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
            normalized = self.base.predict_action_chunk(batch, noise=noise)[0, 0].float().cpu().numpy()
        if not np.isfinite(normalized).all():
            raise RuntimeError("VLAが非有限値を返しました。")
        return normalized * SCALE, time.monotonic() - started
