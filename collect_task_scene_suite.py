#!/usr/bin/env python3
"""Collect the predeclared training/validation tasks; never open test seeds."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from iss_capture.unified_config import ROOT
from iss_capture.scene_training_cases import SPLITS,write_splits

p=argparse.ArgumentParser();p.add_argument('--train-count',type=int,default=8)
p.add_argument('--validation-count',type=int,default=3)
p.add_argument('--output-root',type=Path,default=ROOT/'datasets/task_scenes_v1',help='Directory for newly collected episodes')
a=p.parse_args()
assert 0<a.train_count<=SPLITS['train'][1] and 0<a.validation_count<=SPLITS['validation'][1]
write_splits();root=a.output_root.resolve();root.mkdir(parents=True,exist_ok=True)
(ROOT/'progress').mkdir(parents=True,exist_ok=True)
jobs=[(split,task,seed) for split,count in [('validation',a.validation_count),('train',a.train_count)]
      for seed in range(SPLITS[split][0],SPLITS[split][0]+count) for task in ('task1','task2','task3')]
results=[]
for job_index,(split,task,seed) in enumerate(jobs):
    for attempt in range(2):
        name=f'{split}_{task}_{seed}'+('' if attempt==0 else '_retry1')
        destination=root/name
        result=destination/'result.json'
        if not result.exists():
            if destination.exists():raise RuntimeError('Incomplete existing episode: '+str(destination))
            log=root/(name+'.log')
            print('START '+json.dumps(dict(split=split,task=task,seed=seed,attempt=attempt)),flush=True)
            with log.open('w') as stream:
                completed=subprocess.run([sys.executable,str(ROOT/'collect_task_scene.py'),'--task',task,
                    '--seed',str(seed),'--split',split,'--output',str(destination)],
                    env={**os.environ,'ROS_DOMAIN_ID':str(110+(job_index*2+attempt)%90)},
                    stdout=stream,stderr=subprocess.STDOUT)
            if not result.exists():
                destination.mkdir(exist_ok=True)
                result.write_text(json.dumps(dict(task=task,split=split,episode=seed,success=False,
                    error={'kind':'process_failure','returncode':completed.returncode}),indent=2))
        row=json.loads(result.read_text());results.append({**row,'source':str(destination)})
        if row.get('error') is None:break
    summary={s:{t:dict(attempts=sum(r['split']==s and r['task']==t for r in results),
                successes=sum(bool(r['success']) and r.get('error') is None for r in results if r['split']==s and r['task']==t))
                 for t in ('task1','task2','task3')} for s in ('train','validation')}
    (root/'collection_summary.json').write_text(json.dumps(results,indent=2))
    status=dict(stage='collecting measured RGB-D demonstrations',complete=False,jobs_done=job_index+1,
        jobs_total=len(jobs),summary=summary,active_training=None,test_used=False)
    (ROOT/'progress/task_scene_training.json').write_text(json.dumps(status,indent=2))
    print('PROGRESS '+json.dumps(status),flush=True)
print('COLLECTION DONE',flush=True)
