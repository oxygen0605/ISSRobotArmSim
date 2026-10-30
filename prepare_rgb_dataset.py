#!/usr/bin/env python3
"""Freeze exactly 50 train / 10 validation RGB episodes per task, without copying images."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def select_episodes(source, train_count=50, validation_count=10):
    candidates = []
    for path in sorted(source.glob('*/result.json')):
        result = json.loads(path.read_text())
        if result.get('success') and result.get('split') in ('train', 'validation'):
            candidates.append((path, result))
    selected = []
    for split, count in [('train', train_count), ('validation', validation_count)]:
        for task in ('task1','task2','task3'):
            group = [(p,r) for p,r in candidates if r['split'] == split and r['task'] == task]
            if len(group) < count:
                raise ValueError(f'{split}/{task}: {len(group)} successful episodes available, {count} required')
            selected.extend(group[:count])
    return selected


def audit_episode(path, result):
    samples = json.loads(path.with_name('samples.json').read_text())
    if not samples:
        raise ValueError(f'Empty episode: {path}')
    times, phases = [], Counter()
    for sample in samples:
        state = np.asarray(sample['q'] + sample['q_velocity'], float)
        action = np.asarray(sample['action'], float)
        if state.shape != (24,) or action.shape != (12,) or not np.isfinite(np.r_[state,action]).all():
            raise ValueError(f'Invalid encoders/action: {path}')
        if sample['instruction'] != result['scene']['instruction']:
            raise ValueError(f'Instruction mismatch: {path}')
        for name in ('front','side','previous_front'):
            image = path.parent/f"{sample['frame']}_{name}.png"
            if not image.is_file() or image.stat().st_size == 0:
                raise ValueError(f'Missing RGB image: {image}')
        if not times or sample['image_time'] > times[-1]:
            times.append(sample['image_time'])
            phases[sample['teacher_phase']] += 1
    return dict(id=path.parent.name, source=str(path.parent.resolve()), task=result['task'],
        split=result['split'], seed=result['case']['seed'], frames=len(times), raw_frames=len(samples),
        phase_counts=dict(phases), case=result['case'],
        samples_sha256=hashlib.sha256(path.with_name('samples.json').read_bytes()).hexdigest(),
        result_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train-count', type=int, default=50)
    p.add_argument('--validation-count', type=int, default=10)
    a = p.parse_args()
    if a.output.exists():
        p.error('Choose a new output directory to preserve the frozen selection')
    selected = select_episodes(a.source, a.train_count, a.validation_count)
    episodes = [audit_episode(path,result) for path,result in selected]
    seen = set()
    for episode in episodes:
        key = episode['task'],episode['seed']
        if key in seen:
            raise ValueError(f'Duplicate task/seed across splits: {key}')
        seen.add(key)
    plan = json.loads((a.source/'collection_plan.json').read_text())
    used_seeds = {e['seed'] for e in episodes}
    if used_seeds & set(plan['held_out_test_seeds'] + plan['development_seeds']):
        raise ValueError('Held-out scenario leaked into training/validation')
    counts = {}
    for split in ('train','validation'):
        counts[split] = {}
        for task in ('task1','task2','task3'):
            group = [e for e in episodes if e['split'] == split and e['task'] == task]
            counts[split][task] = dict(episodes=len(group), frames=sum(e['frames'] for e in group))
    a.output.mkdir(parents=True)
    for path,_ in selected:
        (a.output/path.parent.name).symlink_to(path.parent.resolve(), target_is_directory=True)
    manifest = dict(version=1, counts=counts, episodes=episodes,
        selection='first successful episode IDs per task; unsuccessful and overshoot attempts retained in source',
        raw_source=str(a.source.resolve()), input_modalities=['RGB3','joint_position12','joint_velocity12','language'],
        depth_used_for_learning=False, ground_truth_used_as_policy_input=False,
        teacher_actions_as_labels=True, held_out_test_seeds=plan['held_out_test_seeds'])
    (a.output/'selection_manifest.json').write_text(json.dumps(manifest, indent=2))
    (a.output/'collection_plan.json').write_text(json.dumps(plan, indent=2))
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
