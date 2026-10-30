#!/usr/bin/env python3
"""Resume a balanced RGB-only expert dataset with disjoint episode seeds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
from iss_capture.scene_training_cases import scenario
from iss_capture.scene_teacher import SceneTeacher
from iss_capture.task_scenes import SceneSession, SceneMonitor

CAMERAS = ('front', 'side', 'previous_front')
ARM_AXES = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save(path, value):
    stage = path.with_name(path.name + f'.{os.getpid()}.tmp')
    stage.write_text(json.dumps(value, indent=2, default=serializable))
    stage.replace(path)


def completed_episodes(directory, task, split):
    entries = []
    for path in sorted(directory.glob(f'{split}_{task}_*/result.json')):
        result = json.loads(path.read_text())
        if result.get('success') and result.get('split') == split and result.get('task') == task:
            entries.append((path, result))
    return entries


def collect_episode(output, task, seed, split, max_steps, noise_probability, noise_std):
    output.mkdir(exist_ok=False)
    spec, case, nominal = scenario(task, seed, True)
    teacher, monitor = SceneTeacher(task), SceneMonitor(task)
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(task[-1]), 477]))
    rows, error, completed, termination = [], None, None, None
    started = time.monotonic()
    elapsed = None
    try:
        with SceneSession(output, case, spec, use_depth=False) as session:
            client = session.client
            begin = client.snapshot()[1]
            for index in range(max_steps):
                obs = client.camera_observation()
                goal = teacher.action(client)
                action = teacher.codec.decode(goal, obs['q'])
                command = action.copy()
                perturbed = split == 'train' and rng.random() < noise_probability
                if perturbed:
                    command[ARM_AXES] += np.clip(rng.normal(0, noise_std, len(ARM_AXES)), -3*noise_std, 3*noise_std)
                frame = f'{index:05d}'
                for name in CAMERAS:
                    Image.fromarray(obs[name]).save(output/f'{frame}_{name}.png', compress_level=1)
                rows.append(dict(frame=frame, q=obs['q'], q_velocity=obs['q_velocity'],
                    previous_command=obs['previous_command'], action=action, goal=goal,
                    instruction=spec['instruction'], teacher_phase=teacher.phase,
                    image_time=obs['image_time'], controller_time=obs['controller_time'], sim_time=obs['sim_time'],
                    behavior_action=command, behavior_perturbed=bool(perturbed)))
                client.command(command)
                time.sleep(.015)
                client.step(100)
                monitor.update(client)
                if monitor.success and completed is None:
                    completed = client.snapshot()[1]
                if completed is not None and client.snapshot()[1]-completed >= .6:
                    termination = 'success'
                    break
                if index >= 100 and teacher.phase == 'grasp' and completed is None:
                    termination = 'teacher_grasp_timeout'
                    break
                if index % 40 == 0:
                    save(output/'progress.json', dict(step=index, phase=teacher.phase, metric=monitor.last))
            elapsed = client.snapshot()[1]-begin
    except Exception as exc:
        error = dict(message=str(exc), traceback=traceback.format_exc())
    result = dict(episode=seed, task=task, split=split,
        success=completed is not None and monitor.success and error is None,
        teacher_only=True, teacher_version=teacher.version,
        data_kind='rgb_encoder_expert_demonstration', frames=len(rows), scene=spec, case=case, nominal_case=nominal,
        depth_inputs=False, depth_sensors_enabled=False, geometry_as_policy_input=False,
        privileged_information='expert labels and evaluation only',
        perturbation_probability=noise_probability if split == 'train' else 0.,
        perturbation_std_rad=noise_std if split == 'train' else 0.,
        final=monitor.last, error=error, termination=termination or ('error' if error else 'step_limit'),
        wall_seconds=time.monotonic()-started, simulated_seconds=elapsed)
    save(output/'samples.json', rows)
    save(output/'metrics.json', monitor.samples)
    save(output/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', choices=['task1', 'task2', 'task3'], required=True)
    parser.add_argument('--train-target', type=int, default=50)
    parser.add_argument('--validation-target', type=int, default=10)
    parser.add_argument('--train-seed', type=int, default=6100000)
    parser.add_argument('--validation-seed', type=int, default=6200000)
    parser.add_argument('--max-attempts', type=int, default=160)
    parser.add_argument('--max-steps', type=int, default=220)
    parser.add_argument('--noise-probability', type=float, default=.1)
    parser.add_argument('--noise-std', type=float, default=.003)
    args = parser.parse_args()
    if args.train_target < 0 or args.validation_target < 0 or args.max_attempts < 1:
        parser.error('Targets must be nonnegative and max-attempts positive')
    if abs(args.train_seed-args.validation_seed) <= args.max_attempts:
        parser.error('Training and validation seed ranges must be disjoint')
    args.output.mkdir(parents=True, exist_ok=True)
    status_file = args.output/f'collection_{args.task}.json'
    for split, target, start in [('train', args.train_target, args.train_seed),
                                  ('validation', args.validation_target, args.validation_seed)]:
        for attempt in range(args.max_attempts):
            complete = completed_episodes(args.output, args.task, split)
            if len(complete) >= target:
                break
            seed = start + attempt
            output = args.output/f'{split}_{args.task}_{seed}'
            if output.exists():
                if not (output/'result.json').exists():
                    raise RuntimeError(f'Unfinished attempt at {output}; preserve and inspect before resuming')
                continue
            print('START '+json.dumps(dict(task=args.task, split=split, seed=seed, completed=len(complete), target=target)), flush=True)
            result = collect_episode(output, args.task, seed, split, args.max_steps,
                                     args.noise_probability, args.noise_std)
            counts = {s:len(completed_episodes(args.output, args.task, s)) for s in ('train','validation')}
            save(status_file, dict(task=args.task, successful_episodes=counts, last_episode=seed,
                last_success=result['success'], last_error=result['error'],
                ros_domain_id=os.environ.get('ROS_DOMAIN_ID'), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
            print('RESULT '+json.dumps(dict(task=args.task, split=split, seed=seed, success=result['success'],
                frames=result['frames'], counts=counts, wall_seconds=result['wall_seconds'], error=result['error'])), flush=True)
        if len(completed_episodes(args.output, args.task, split)) < target:
            raise RuntimeError(f'{args.task} {split}: success target not reached in declared attempt budget')
    print('COMPLETE '+args.task, flush=True)


if __name__ == '__main__':
    main()
