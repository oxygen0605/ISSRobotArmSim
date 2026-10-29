"""Predeclared, episode-disjoint scenarios for the requested task layouts."""
import json
from pathlib import Path
import numpy as np
from .task_scenes import load_scene
from .unified_config import ROOT

SPLITS = {'train': (4100000, 16), 'validation': (4200000, 4),
          'development': (4300000, 6), 'test': (4400000, 8)}

def scenario(task, seed, jitter=True):
    spec, case = load_scene(task, seed)
    spec = dict(spec)
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(task[-1]), 9021]))
    if jitter:
        for kind, model in [('water', 'droplet'), ('pen', 'floating_pen'), ('cap', 'floating_cap')]:
            if model not in spec['objects']:
                continue
            key = kind + '_position'
            case[key] = (np.asarray(case[key]) + rng.uniform(-.005, .005, 3)).tolist()
        if 'droplet' in spec['objects']:
            case['water_velocity'] = [float(rng.uniform(-.001, .001)), float(rng.uniform(-.001, .001)), float(rng.uniform(.004, .008))]
    # Perception starts from nominal configured regions, never actual jittered poses.
    _, nominal = load_scene(task, seed)
    return spec, case, nominal

def write_splits(path=None):
    path = Path(path or ROOT/'data_splits/task_scene_cases_v1.json')
    rows = {split: {task: [dict(seed=seed, case=scenario(task, seed)[1]) for seed in range(start, start+count)]
                    for task in ('task1','task2','task3')} for split,(start,count) in SPLITS.items()}
    content = dict(version=1, position_jitter_m=.005, tasks_config=json.loads((ROOT/'config/task_scenes.json').read_text()),
                   splits=rows, test_used_for_training_or_selection=False)
    if path.exists():
        if json.loads(path.read_text()) != content:
            raise ValueError('Existing split manifest differs; preserve it and use a new version')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content, indent=2))
    return path
