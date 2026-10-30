#!/usr/bin/env python3
"""Evaluate a frozen RGB-only checkpoint on predeclared, jittered episodes."""
import argparse
import hashlib
import json
from pathlib import Path

from iss_capture.config import ROOT
from iss_capture.policy_loader import load_policy
from iss_capture.scene_training_cases import scenario
from run_task_scene_preview import run, save, file_hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plan', type=Path, default=ROOT/'datasets/rgb_only_balanced50_v1/collection_plan.json')
    p.add_argument('--split', choices=['development','test'], default='test')
    p.add_argument('--tasks', nargs='+', choices=['task1','task2','task3'], default=['task1','task2','task3'])
    p.add_argument('--max-steps', type=int, default=200)
    p.add_argument('--record-first', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(a.plan.read_text())
    seeds = plan['held_out_test_seeds' if a.split == 'test' else 'development_seeds']
    sources = [Path(__file__), ROOT/'run_task_scene_preview.py', *sorted((ROOT/'iss_capture').glob('*.py'))]
    before = file_hashes([*sources, *a.checkpoint.iterdir()])
    save(a.output/'frozen_source_checkpoint_hashes.json', before)
    save(a.output/'evaluation_plan.json', dict(split=a.split, seeds=seeds, tasks=a.tasks,
         checkpoint=str(a.checkpoint.resolve()), position_jitter_m=.005,
         plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),
         teacher_intervention=False, max_steps=a.max_steps))
    policy = load_policy(a.checkpoint)
    if not getattr(policy, 'rgb_only', False):
        raise ValueError('RGB-only checkpoint required')
    results = []
    for task in a.tasks:
        for index, seed in enumerate(seeds):
            spec, case, _ = scenario(task, seed, True)
            result = run(policy, a.output/f'{task}_{seed}', task, spec, case,
                         max_steps=a.max_steps, record=a.record_first and index == 0)
            results.append(result)
            save(a.output/'results.json', results)
            aggregate = {}
            for t in a.tasks:
                entries = [r for r in results if r['task'] == t]
                successes = sum(bool(r['success']) and r['error'] is None for r in entries)
                aggregate[t] = dict(successes=successes, attempted=len(entries),
                    planned=len(seeds), errors=sum(r['error'] is not None for r in entries),
                    rate=None if not entries else successes/len(entries))
            save(a.output/'summary.json', dict(split=a.split, complete=len(results) == len(seeds)*len(a.tasks),
                 tasks=aggregate, checkpoint=str(a.checkpoint.resolve())))
    if before != file_hashes([*sources, *a.checkpoint.iterdir()]):
        raise RuntimeError('Checkpoint or policy source changed during evaluation')
    print('EVALUATION COMPLETE '+str(a.output/'summary.json'), flush=True)


if __name__ == '__main__':
    main()
