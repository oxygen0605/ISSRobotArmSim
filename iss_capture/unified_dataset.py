"""Episode-disjoint offline demonstrations and RGB-only feature cache."""
import hashlib,json
from pathlib import Path
from functools import lru_cache
import numpy as np
from PIL import Image
from .unified_features import Features,delta_target,network_rgb
@lru_cache(maxsize=512)
def rgb(path):
    with Image.open(path) as im:return np.asarray(im.convert('RGB'))
def observations(items):
    return [{**{name:network_rgb(rgb(path)) for name,path in r['paths'].items()},'q':r['state']} for r in items]
def load(directory,representation='delta'):
    features=Features();items=[]
    source=''.join((Path(__file__).parent/p).read_text() for p in ('unified_features.py','unified_config.py','bimanual_config.py','capping_pose.py'))
    for path in sorted(Path(directory).glob('*/result.json')):
        result=json.loads(path.read_text())
        if result['split'] not in ('train','validation'):continue
        prefix_file=path.with_name('supervision.json')
        prefix=None
        if not result.get('success'):
            if result['split']!='train' or not prefix_file.exists():continue
            prefix=json.loads(prefix_file.read_text()).get('valid_prefix_frames',0)
            if prefix<6:continue
        records=json.loads(path.with_name('samples.json').read_text())
        if prefix is not None:records=records[:prefix]
        digest=hashlib.sha256((source+path.with_name('samples.json').read_text()).encode()).hexdigest()[:16]
        cache=path.with_name('features_'+digest+'.npy');cached=np.load(cache) if cache.exists() else None;states=[]
        for i,r in enumerate(records):
            paths={n:str(path.parent/f'{r["frame"]}_{n}.png') for n in ('front','side','previous_front')}
            obs=dict(q=np.asarray(r['q']),previous_command=np.asarray(r['previous_command']))
            if cached is None:obs.update({n:rgb(p) for n,p in paths.items()})
            state=features(obs) if cached is None else cached[i];states.append(state)
            items.append(dict(episode=result['episode'],task=result['task'],split=result['split'],instruction=r['instruction'],
                state=state,q=obs['q'],goal=np.asarray(r['goal'],np.float32) if representation in ('goal','world') else delta_target(r['goal'],obs['q'],features.codec),paths=paths,
                phase=r['teacher_phase'],initial=i<3,source=str(path.parent),successful_rollout=bool(result['success']),supervision_prefix=prefix is not None))
            if representation=='world':items[-1]['goal'][3:6]+=items[-1]['goal'][:3]
        if cached is None:np.save(cache,np.stack(states))
    return items
