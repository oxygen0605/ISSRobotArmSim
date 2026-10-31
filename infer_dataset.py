#!/usr/bin/env python3
"""Replay recorded RGB and encoder observations without moving a robot."""
import argparse, json, time
from pathlib import Path
import numpy as np
from iss_capture.config import ROOT
from iss_capture.public_dataset import ensure_dataset, load_observation
from iss_capture.policy_loader import load_policy
from iss_capture.task_scene_observation import configure_observation
from iss_capture.unified_config import CHECKPOINT

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", default="validation_task1_4200000")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--max-frames", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error("--max-frames must be positive")
    directory = ensure_dataset(args.dataset, [args.episode])
    episode = directory/args.episode
    result = json.loads((episode/"result.json").read_text())
    records = json.loads((episode/"samples.json").read_text())
    policy = load_policy(args.checkpoint)
    # Same nominal search regions as online inference; no evaluation_objects input.
    configure_observation(policy, result["scene"], result.get("nominal_case",result["case"]))
    predictions = []
    for record in records[:args.max_frames]:
        observation=load_observation(episode,record,include_depth=not getattr(policy,'rgb_only',False))
        if getattr(policy,'rgb_only',False):
            from iss_capture.rgb_joint_policy import select_inputs
            observation=select_inputs(observation,policy.config.get('positions_only',False))
        command, debug = policy.predict(observation, record["instruction"])
        if not np.isfinite(command).all():
            raise RuntimeError("Non-finite model output")
        predictions.append({"frame": record["frame"], "command": command.tolist(),
                            "phase_probabilities": debug.get("phase_probabilities")})
    output = args.output or ROOT/"runs"/("dataset_inference_"+time.strftime("%Y%m%d_%H%M%S"))/(args.episode+".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"episode":args.episode, "frames":len(predictions),
        "mode":"offline observed-input replay; no closed-loop success measurement",
        "teacher_or_object_truth_used_as_input":False, "predictions":predictions},indent=2)+"\n")
    print(output)

if __name__ == "__main__":
    main()
