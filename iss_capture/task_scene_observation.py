"""Explicit absent-object encoding for the old fixed-width policy input.

This does not change learned weights, swap arm commands, or supply simulator
poses. The configured nominal positions seed image searches only. Required
objects still must be detected in RGB-D; omitted objects have zero geometry and
zero confidence. Legacy checkpoints were not trained on this missing-object encoding.
Task-scene checkpoints are explicitly trained on these measured observations.
"""
import numpy as np
from .depth_geometry import DepthGeometry
from .unified_features import Features
from .workspace_features import WorkspaceFeatures
from .water_depth import WaterDepth


def absent():
    return dict(marker=np.zeros(3), centre=np.zeros(3), axis=np.array([0., 0., 1.]),
                points=0, stale=True, present=False, radial_error_m=None,
                source='absent object: zero-filled input, not a measurement')


class OptionalGeometry(DepthGeometry):
    def __init__(self, spec, case):
        super().__init__()
        self.absent_kinds = [kind for kind in ('pen', 'cap') if 'floating_' + kind not in spec['objects']]
        for kind in ('pen', 'cap'):
            self.centres[kind] = np.array(case[kind + '_position'], float)

    def __call__(self, observation):
        for kind in self.absent_kinds:
            # Keep the strict original detector's ROI away from all scene pixels.
            # A zero-confidence record permits its existing missing-frame path.
            self.centres[kind] = np.full(3, 100.)
            self.last[kind] = absent()
            self.tool_transforms.pop(kind, None)
            self.last_jaws.pop(kind, None)
        result = super().__call__(observation)
        for kind in self.absent_kinds:
            result[kind] = absent()
            self.last[kind] = result[kind]
        return result


class OptionalWater:
    def __init__(self, spec, case):
        self.detector = WaterDepth() if 'droplet' in spec['objects'] else None
        if self.detector is not None:
            self.detector.position = np.asarray(case['water_position'], float).copy()
        self.position = np.zeros(3)
        self.last_info = dict(present=False, accepted=False, source='absent object: zero-filled input')

    def __call__(self, observation):
        if self.detector is not None:
            self.position = self.detector(observation)
            self.last_info = {**self.detector.last_info, 'present': True}
        return self.position.copy()


class SceneRGB(Features):
    def __init__(self, water):
        super().__init__()
        self.water = water

    def __call__(self, observation):
        state = super().__call__(observation)
        state[20:23] = self.water(observation)
        return state


def configure_observation(policy, spec, case):
    policy.reset()
    # Use nominal configured search regions, never jittered evaluation poses.
    search_case = {**case, **{key: spec[key] for key in ('water_position','pen_position','cap_position') if key in spec}}
    water = OptionalWater(spec, search_case)
    features = WorkspaceFeatures()
    features.depth.rgb = SceneRGB(water)
    features.depth.geometry = OptionalGeometry(spec, search_case)
    policy.features = features
    policy.water = water


def observation_audit(policy):
    geometry = policy.features.depth.last_geometry or {}
    return {kind: {'points': value.get('points', 0), 'stale': value.get('stale', False),
                   'present': value.get('present', True)} for kind, value in geometry.items()}
