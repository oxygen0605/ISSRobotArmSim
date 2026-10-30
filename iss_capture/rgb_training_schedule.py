"""Optimizer-step scheduling and effective-batch accounting for RGB training."""
import math
import numpy as np


def learning_rate(step, total_steps, warmup_steps, peak_lr, final_lr):
    """LR used by update `step` (one-based); step zero is the warmup origin."""
    if not 0 <= step <= total_steps or not 0 <= warmup_steps < total_steps:
        raise ValueError('Invalid optimizer step or warmup duration')
    if not 0 < final_lr <= peak_lr:
        raise ValueError('Require 0 < final_lr <= peak_lr')
    if warmup_steps and step <= warmup_steps:
        return peak_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return final_lr + .5 * (peak_lr - final_lr) * (1 + math.cos(math.pi * progress))


def batch_layout(effective_batch_size, micro_batch_size):
    if micro_batch_size < 1 or effective_batch_size < 1 or effective_batch_size % micro_batch_size:
        raise ValueError('Micro batch must divide the effective batch size')
    return micro_batch_size, effective_batch_size // micro_batch_size


def task_index(step, effective_batch_size, sample_offset):
    return ((step - 1) * effective_batch_size + sample_offset) % 3


def attach_future_targets(rows, chunk_size, target):
    """Build chunks within one episode, repeating its terminal teacher target."""
    if chunk_size < 1:raise ValueError('Chunk size must be positive')
    episodes={}
    for row in rows:episodes.setdefault(row['episode'],[]).append(row)
    for episode in episodes.values():
        actions=[target(row) for row in episode]
        for index,row in enumerate(episode):
            row['future']=np.stack([actions[min(index+j,len(actions)-1)] for j in range(chunk_size)])
            row['future_valid']=np.arange(chunk_size)<len(actions)-index


def masked_mean(values, valid):
    """Mean over real time steps and feature axes, excluding terminal padding."""
    values=np.asarray(values)
    weights=np.broadcast_to(np.asarray(valid,bool)[...,None],values.shape)
    if not weights.any():raise ValueError('No valid action targets')
    return float(values[weights].mean())
