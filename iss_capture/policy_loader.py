"""Checkpoint architecture selection, independent of the requested task."""
import json
from pathlib import Path
def load_policy(checkpoint,device='cuda'):
    variant=json.loads((Path(checkpoint)/'capture_config.json').read_text())['policy_variant']
    if variant=='shared_three_task_smolvla_task_scenes_v1':
        from .scene_one_policy import SceneOnePolicy
        return SceneOnePolicy(checkpoint,device)
    if variant=='shared_three_task_smolvla_workspace_v1':
        from .workspace_one_policy import WorkspaceOnePolicy
        return WorkspaceOnePolicy(checkpoint,device)
    if variant=='shared_three_task_smolvla_rgbd_v1':
        from .rgbd_one_policy import RGBDOnePolicy
        return RGBDOnePolicy(checkpoint,device)
    if variant=='shared_three_task_smolvla_joint_v1':
        from .joint_one_policy import JointOnePolicy
        return JointOnePolicy(checkpoint,device)
    if variant=='shared_three_task_smolvla_hierarchical_v1':
        from .hierarchical_one_policy import HierarchicalOnePolicy
        return HierarchicalOnePolicy(checkpoint,device)
    from .unified_policy import OnePolicy
    return OnePolicy(checkpoint,device)
