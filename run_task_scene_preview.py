#!/usr/bin/env python3
"""Record task-specific layouts with the existing shared checkpoint, without training."""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np

from iss_capture.unified_config import ROOT, CHECKPOINT
from iss_capture.policy_loader import load_policy
from iss_capture.task_scenes import load_scene, SceneSession, SceneMonitor
from iss_capture.task_scene_observation import configure_observation, observation_audit
from iss_capture.task_scene_video import CameraRecording


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=serializable))


def file_hashes(paths):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.is_file()}


def run(policy, output, task, spec, case, max_steps=120, gui=False, record=True, linger=0):
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'request.json', dict(task=task, scene=spec, case=case, max_steps=max_steps))
    monitor = SceneMonitor(task)
    recording = CameraRecording(output, enabled=record)
    trace, objects, error, displays, opposite = [], [], None, [], []
    started = time.monotonic()
    configure_observation(policy, spec, case)
    begin = end = None
    try:
        with SceneSession(output, case, spec, gui=gui, record=record, use_depth=not getattr(policy,'rgb_only',False)) as session:
            client = session.client
            begin = client.snapshot()[1]
            for index in range(max_steps):
                observation = client.camera_observation()
                recording.observe(observation)
                try:
                    model_input = observation
                    if getattr(policy,'rgb_only',False):
                        from iss_capture.rgb_joint_policy import select_inputs
                        model_input = select_inputs(observation,policy.config.get('positions_only',False))
                    command, debug = policy.predict(model_input, spec['instruction'])
                except Exception as exception:
                    error = dict(kind='policy_observation_or_inference_error', message=str(exception),
                                 traceback=traceback.format_exc())
                    break
                client.command(command)
                time.sleep(.015)
                client.step(100)
                metric = monitor.update(client)
                trace.append(dict(step=index, image_time=observation['image_time'],
                    command=command, q=observation['q'], observation_presence=observation_audit(policy), **debug))
                objects.append(client.object_states())
                if index % 20 == 0:
                    save(output / 'live_trace.json', trace)
                    phase = policy.config['phase_names'][int(np.argmax(debug['phase_probabilities']))] if 'phase_probabilities' in debug else 'learned_'+policy.config.get('action_mode','joint')
                    print(json.dumps(dict(task=task, step=index, phase=phase, **metric)), flush=True)
                if monitor.success:
                    break
            # Show the final physical state for one simulated second; keep the
            # controller's previous target and do not inject teacher actions.
            for _ in range(10):
                recording.observe(client.camera_observation())
                client.step(100)
            recording.observe(client.camera_observation())
            with client.lock:
                displays = list(client.display_frames)
                opposite = list(client.opposite_frames)
            end = client.snapshot()[1]
            if gui and linger > 0:
                time.sleep(min(linger, 60))
    except Exception as exception:
        error = dict(kind='infrastructure_error', message=str(exception), traceback=traceback.format_exc())
    result = dict(task=task, success=monitor.success, instruction=spec['instruction'],
        steps=len(trace), max_steps=max_steps, termination='error' if error else ('success' if monitor.success else 'step_limit'),
        final=monitor.last, error=error, scene=spec, case=case, policy_instance=id(policy),
        checkpoint=str(policy.directory.resolve()), checkpoint_hashes=file_hashes(policy.directory.iterdir()),
        retrained=policy.config.get('policy_variant')=='shared_three_task_smolvla_task_scenes_v1', arm_command_swapping=False, teacher_used=False, task_specific_model_routing=False,
        observation_adapter='Absent object geometry is zero-filled with zero confidence; present objects use RGB-D. Nominal scene positions initialize search regions only.',
        known_limit=('Shared learned language/stage/goal head with calibrated RGB-D geometry and generic IK; idealized transparent-water depth.' if policy.config.get('policy_variant')=='shared_three_task_smolvla_task_scenes_v1' else 'Existing action head routes water goals to the left arm. Right-arm water behavior was not trained or remapped.'),
        wall_seconds=time.monotonic()-started, simulated_seconds=None if begin is None or end is None else end-begin)
    if getattr(policy,'rgb_only',False):
        result.update(retrained=True,depth_used=False,depth_sensors_enabled=False,geometry_features_used=False,
            observation_adapter='RGB images and raw encoder state only; language instruction',
            known_limit=('SmolVLA Cartesian goals and generic robot IK; no privileged geometry or contact input' if policy.config.get('action_mode')=='cartesian' else 'Direct SmolVLA joint actions; no privileged geometry or contact input'))
    if policy.config.get('policy_variant')=='smolvla_rgb_joint_language_temporal_v1':
        result['known_limit']='SmolVLA with a learned GRU action readout; RGB and encoder history only; no depth or object-geometry input'
        result['learned_temporal_readout']=True
    save(output / 'result.json', result)
    save(output / 'trace.json', trace)
    save(output / 'metrics.json', monitor.samples)
    save(output / 'evaluation_object_trace.json', objects)
    if recording.stamps:
        try:
            recording.finish(displays, opposite)
        except Exception as exception:
            result['recording_error'] = str(exception)
            save(output / 'result.json', result)
            raise
    print('RESULT ' + json.dumps(result, default=serializable), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['task1', 'task2', 'task3', 'all'], default='all')
    parser.add_argument('--seed', type=int, default=3300002)
    parser.add_argument('--scene-config', type=Path, default=ROOT/'config/task_scenes.json')
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-steps', type=int, default=120)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--no-record', action='store_true', help='Evaluate without saving camera videos')
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error('--max-steps must be positive')
    output = args.output or ROOT/'runs'/time.strftime('task_scene_preview_%Y%m%d_%H%M%S')
    if output.exists():
        parser.error('Output already exists; choose a new directory')
    output.mkdir(parents=True)
    tasks = ['task1', 'task2', 'task3'] if args.task == 'all' else [args.task]
    scenes = {task: load_scene(task, args.seed, args.scene_config) for task in tasks}
    sources = [Path(__file__), args.scene_config, ROOT/'run_biarm_policy.py', *sorted((ROOT/'iss_capture').glob('*.py'))]
    before = file_hashes([*sources, *args.checkpoint.iterdir()])
    save(output/'source_checkpoint_hashes.json', before)
    policy = load_policy(args.checkpoint)
    results = []
    for task in tasks:
        spec, case = scenes[task]
        results.append(run(policy, output/task, task, spec, case, args.max_steps, args.gui, record=not args.no_record))
        save(output/'summary.json', results)
    after = file_hashes([*sources, *args.checkpoint.iterdir()])
    if before != after:
        raise RuntimeError('Source or checkpoint changed during preview')
    print('FROZEN CHECKPOINT AND SOURCES VERIFIED', flush=True)


if __name__ == '__main__':
    main()
