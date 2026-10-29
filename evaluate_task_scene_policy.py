#!/usr/bin/env python3
"""Frozen, one-instance closed-loop evaluation on the predeclared scene split."""
import argparse,hashlib,json,os
from pathlib import Path
from iss_capture.unified_config import ROOT
from iss_capture.policy_loader import load_policy
from iss_capture.scene_training_cases import scenario,write_splits
from run_task_scene_preview import run

def hashes(paths):
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--split',choices=['development','test'],default='development')
    p.add_argument('--offset',type=int,default=0);p.add_argument('--limit',type=int,default=1)
    p.add_argument('--tasks',nargs='+',choices=['task1','task2','task3'],default=['task1','task2','task3'])
    p.add_argument('--max-steps',type=int,default=220);p.add_argument('--gui',action='store_true');a=p.parse_args()
    if a.output.exists():p.error('Refusing to overwrite an evaluation')
    manifest=write_splits();splits=json.loads(manifest.read_text())['splits'][a.split]
    jobs=[(task,row['seed'],row['case']) for task in a.tasks for row in splits[task][a.offset:a.offset+a.limit]]
    if not jobs:raise ValueError('Empty evaluation slice')
    a.output.mkdir(parents=True)
    sources=[Path(__file__),ROOT/'run_task_scene_preview.py',manifest,ROOT/'config/task_scenes.json',
             *sorted((ROOT/'iss_capture').glob('*.py')),*sorted(a.checkpoint.iterdir()),
             ROOT/'run_biarm_policy.py',ROOT/'run_demo.py',ROOT/'scripts/ros_env.sh',ROOT/'scripts/water_env.sh',
             *sorted((ROOT/'plugins/water').glob('*.cc')),*sorted((ROOT/'plugins/water').glob('*.hh')),
             *sorted((ROOT/'build/water_plugin').glob('*.so'))]
    before=hashes(sources)
    (a.output/'frozen_sources.json').write_text(json.dumps(before,indent=2))
    policy=load_policy(a.checkpoint);results=[]
    for index,(task,seed,case) in enumerate(jobs):
        os.environ['ROS_DOMAIN_ID']=str(160+index)
        spec,generated,nominal=scenario(task,seed)
        if generated!=case:raise ValueError('Evaluation case changed since split declaration')
        row=run(policy,a.output/f'{task}_{seed}',task,spec,case,a.max_steps,a.gui)
        results.append(row)
        summary={t:dict(attempts=sum(r['task']==t for r in results),successes=sum(r['success'] for r in results if r['task']==t),
                        errors=sum(r['error'] is not None for r in results if r['task']==t)) for t in a.tasks}
        (a.output/'summary.json').write_text(json.dumps(dict(split=a.split,tasks=summary,results=results),indent=2))
        print('EVALUATION '+json.dumps(summary),flush=True)
    if before!=hashes(sources):raise RuntimeError('Checkpoint or source changed during evaluation')
    (a.output/'verified.json').write_text(json.dumps(dict(frozen_sources_verified=True,one_policy_instance=len({r['policy_instance'] for r in results})==1),indent=2))
    print('FROZEN MODEL AND SOURCES VERIFIED',flush=True)

if __name__=='__main__':main()
