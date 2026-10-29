#!/usr/bin/env python3
"""Run one isolated Gazebo episode using learned inverse kinematics."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
import numpy as np

from iss_capture.config import ROOT, POLICY, SCENARIOS, JOINTS
from iss_capture.kinematics import Arm
from iss_capture.policy import Policy
from iss_capture.world import build


def smoothstep(fraction):
    x = float(np.clip(fraction, 0, 1))
    return x * x * (3 - 2 * x)


def stop_process(process):
    """Stop only the process group created by this run, including the GUI."""
    if process is None:
        return
    for sig, timeout in [(signal.SIGINT, 5), (signal.SIGTERM, 3), (signal.SIGKILL, 2)]:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=timeout)
            # The launcher can exit before its children; terminate residual group.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            break
        except subprocess.TimeoutExpired:
            continue


def run(args):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / "runs" / f"{stamp}_{args.scenario}_{uuid.uuid4().hex[:6]}"
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {"kind": "gazebo_contact_episode", "status": "error", "success": False,
              "started_utc": stamp, "scenario": args.scenario if args.target is None else "custom",
              "headless": args.headless, "success_source": "/capture/touched",
              "policy_sha256": hashlib.sha256(POLICY.read_bytes()).hexdigest() if POLICY.exists() else None}
    client, process, log = None, None, None
    old_partition = os.environ.get("GZ_PARTITION")
    try:
        arm, policy = Arm(), Policy()
        target = np.asarray(args.target) if args.target is not None else arm.forward(SCENARIOS[args.scenario])
        plan = policy.plan(arm, target, refine=not args.no_refine)
        result["plan"] = plan
        print(f"AI推論: 水滴XYZ = {target.round(4).tolist()} m", flush=True)
        print(f"AI単体の位置誤差 {plan['neural_error_mm']:.3f} mm / "
              f"実行計画の位置誤差 {plan['planned_error_mm']:.3f} mm", flush=True)
        path, scene = build(output / "scene", args.scenario, args.target)
        result["scene"] = scene
        # A fresh transport partition prevents events from a different Gazebo run
        # being counted as this episode's success.
        os.environ["GZ_PARTITION"] = "iss_capture_" + uuid.uuid4().hex
        from iss_capture.transport import GazeboClient
        client = GazeboClient()
        log = (output / "gazebo.log").open("w")
        command = ["gz", "sim", "-r", "-v", "3"]
        if args.headless:
            command.append("-s")
        command.append(str(path))
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + args.startup_timeout
        print("Gazebo起動中。接触イベントと関節状態の購読は開始済みです。", flush=True)
        while True:
            q, sim_time, updated = client.snapshot()
            if process.poll() is not None:
                raise RuntimeError("Gazeboが終了しました。gazebo.logを確認してください。")
            if q is not None and sim_time >= 0.5:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("関節状態を受信できません。Gazeboログと通信設定を確認してください。")
            time.sleep(0.05)
        if client.touched.is_set():
            raise RuntimeError("動作開始前に接触しています。初期配置を修正してください。")
        if np.max(np.abs(q)) > 0.03:
            raise RuntimeError("初期姿勢がゼロ姿勢からずれています。初期化を確認してください。")
        start_q = q.copy()
        goal_q = np.r_[plan["command_q_rad"], [0, 0, 0]]
        # Cubic interpolation's maximum derivative is 1.5. Respect actuator speed.
        duration = max(args.duration, 1.5 * float(np.max(np.abs(goal_q - q))) / 0.30)
        sim_start, wall_start = sim_time, time.monotonic()
        wall_deadline = wall_start + (duration + args.timeout) * 10
        result.update({"motion_duration_sim_s": duration, "motion_start_sim_s": sim_start})
        print("回収動作を開始します。成功は物理接触イベントだけで判定します。", flush=True)
        with (output / "telemetry.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sim_seconds", "wall_seconds", "fk_distance_mm"] +
                            [f"actual_{n}_rad" for n in JOINTS] + [f"command_{n}_rad" for n in JOINTS])
            last_print = -10.0
            while True:
                q, sim_time, updated = client.snapshot()
                elapsed = sim_time - sim_start
                wall_elapsed = time.monotonic() - wall_start
                if client.touched.is_set():
                    result.update({"status": "success", "success": True,
                                   "touch_event_count": client.touch_count,
                                   "capture_sim_seconds": client.touch_sim_time - sim_start,
                                   "capture_wall_seconds": client.touch_wall_time - wall_start})
                    client.command(q)  # Hold current measured joints after contact.
                    break
                if process.poll() is not None:
                    raise RuntimeError("動作中にGazeboが終了しました。")
                if time.monotonic() - updated > 3:
                    raise TimeoutError("関節状態の受信が3秒以上途切れました。")
                if (elapsed > duration / 2 and np.max(np.abs(goal_q - start_q)) > 0.03
                        and np.max(np.abs(q - start_q)) < 0.002):
                    raise RuntimeError("指令を送っても関節が動きません。物理エンジンと制御プラグインの読込を確認してください。")
                if elapsed > duration + args.timeout or time.monotonic() > wall_deadline:
                    result.update({"status": "timeout", "success": False, "touch_event_count": 0})
                    client.command(q)
                    break
                command_q = start_q + smoothstep(elapsed / duration) * (goal_q - start_q)
                client.command(command_q)
                distance = float(np.linalg.norm(arm.forward(q[:3]) - target) * 1000)
                writer.writerow([elapsed, wall_elapsed, distance, *q, *command_q])
                if wall_elapsed - last_print >= 1:
                    print(f"  sim {elapsed:5.1f} s / 手先中心まで {distance:6.1f} mm", flush=True)
                    last_print = wall_elapsed
                time.sleep(0.05)
        result["physics_paused"] = client.pause()
        result["final_joint_rad"] = dict(zip(JOINTS, q.tolist()))
        if result["success"]:
            print(f"CAPTURE SUCCESS / data: true / {result['capture_sim_seconds']:.2f} sim s", flush=True)
        else:
            print("接触イベントを確認できず、タイムアウトしました。", flush=True)
        # Keep the final scene visible for recording; do not falsify extra motion.
        hold_until = time.monotonic() + args.hold
        while time.monotonic() < hold_until and process.poll() is None:
            time.sleep(0.1)
    except KeyboardInterrupt:
        if not result["success"]:
            result.update({"status": "interrupted", "error": "ユーザーによる中断"})
    except Exception as exc:
        result.update({"status": "error", "success": False, "error": str(exc)})
        print(f"実行エラー: {exc}", flush=True)
    finally:
        stop_process(process)
        if client is not None:
            client.close()
            (output / "contacts.txt").write_text("\n---\n".join(client.contacts))
        if log is not None:
            log.close()
        if old_partition is None:
            os.environ.pop("GZ_PARTITION", None)
        else:
            os.environ["GZ_PARTITION"] = old_partition
        (output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(f"実行記録: {output / 'result.json'}", flush=True)
    return 0 if result["success"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--scenario", choices=SCENARIOS, default="center")
    group.add_argument("--target", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--headless", action="store_true", help="GUIなしで物理シミュレーション")
    parser.add_argument("--no-refine", action="store_true", help="AI単体の角度推定を使用")
    parser.add_argument("--duration", type=float, default=6.0, help="動作時間（シミュレーション秒）")
    parser.add_argument("--timeout", type=float, default=8.0, help="動作後の接触待ち時間")
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    parser.add_argument("--hold", type=float, default=5.0, help="終了場面の表示時間（実時間秒）")
    parser.add_argument("--output", type=Path, help="未作成の実行記録フォルダー")
    args = parser.parse_args()
    for name in ("duration", "timeout", "startup_timeout", "hold"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name != "hold" and value == 0):
            parser.error(f"{name} の値が不正です。")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
