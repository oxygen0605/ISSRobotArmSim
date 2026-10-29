"""Only measured, synchronized RGB-D demonstrations in the final workspace."""
import json
from pathlib import Path
import numpy as np
from .workspace_features import enrich,unit_axis,phase_name,PHASES
from .capping_pose import PoseCodec
def load_workspace(directory):
    codec=PoseCodec();items=[];episodes=[]
    for path in sorted(Path(directory).glob('*/result.json')):
        result=json.loads(path.read_text())
        if not result.get('success') or result['split'] not in ('train','validation'):continue
        records=json.loads(path.with_name('samples.json').read_text());indices=[];last=None
        contact_path=path.with_name('contacts.json')
        contact_log=json.loads(contact_path.read_text()) if contact_path.exists() else []
        contact_times={(side,pad):np.array([c['time'] for c in contact_log if c['side']==side and c['pad']==pad]) for side in ('left','right') for pad in ('fixed','moving')}
        for index,record in enumerate(records):
            stamp=record['image_time']
            if last is not None and stamp<=last:continue
            dt=.1 if last is None else stamp-last;last=stamp
            age=max(0.,record.get('controller_time',stamp)-stamp)
            state34=np.asarray(record['depth_state'],np.float32)
            velocity=np.asarray(record.get('q_velocity',np.zeros(12)),np.float32)
            state,rotations=enrich(state34,record['q'],velocity,dt,age,codec)
            goal=np.array(record['goal'],np.float32);goal[3:6]+=goal[:3]
            phase=phase_name(result['task'],record['teacher_phase'])
            pen=state[23:26]+.057*unit_axis(state[29:31])
            cap=state[23:26]+state[26:29]-.0275*unit_axis(state[32:34])
            pre=np.ones(2,np.int64);pre_mask=np.zeros(2,np.float32)
            contact=record.get('contact_labels',{})
            for side in ('left','right'):
                if side not in contact:
                    hits=[]
                    for pad in ('fixed','moving'):
                        ts=contact_times[side,pad];past=ts[ts<=stamp+.002]
                        hits.append(len(past)>0 and stamp-past[-1]<.025)
                    contact[side]=all(hits)
            if phase in ('pen_grasp','dual_grasp'):
                pre[0]=0 if (rotations[0].T@(goal[:3]-pen))[2]>.0175 and goal[10]>.35 else 2 if contact.get('left',False) and goal[10]<.19 else 1;pre_mask[0]=1
            if phase=='dual_grasp':
                pre[1]=0 if (rotations[1].T@(goal[3:6]-cap))[2]>.0175 and goal[11]>.35 else 2 if contact.get('right',False) and goal[11]<.19 else 1;pre_mask[1]=1
            row=dict(episode=result['episode'],task=result['task'],split=result['split'],source=str(path.parent),
                instruction=record['instruction'],state=state[:32],head_state=state,rotation=rotations,
                q=np.asarray(record['q'],np.float32),goal=goal,phase=phase,phase_id=PHASES.index(phase),
                pre=pre,pre_mask=pre_mask,initial=len(indices)<3,time=stamp,
                paths={n:str(path.parent/(record['frame']+'_'+n+'.png')) for n in ('front','side','previous_front')})
            indices.append(len(items));items.append(row)
        episodes.append(dict(source=str(path.parent),task=result['task'],split=result['split'],indices=indices))
    return items,episodes

def goal_targets(items,episodes):
    centre=[];clearance=[];insert=[];axes=[];carry=[];water=[]
    for episode in episodes:
        if episode['split']=='train' and episode['task']=='task1':
            water.extend(items[i]['rotation'][1].T@(items[i]['goal'][3:6]-items[i].get('water_position',items[i]['head_state'][20:23])) for i in episode['indices'] if items[i]['phase']=='water_approach')
        if episode['split']=='train' and episode['task']=='task2':
            candidates=[items[i] for i in episode['indices'] if items[i]['phase']=='pen_carry']
            if candidates:carry.append(candidates[-1]['goal'][:3])
        if episode['split']!='train' or episode['task']!='task3':continue
        rows=[items[i] for i in episode['indices']]
        c=[r for r in rows if r['phase']=='dual_centre']
        a=[r for r in rows if r['phase']=='dual_align']
        h=[r for r in rows if r['phase']=='dual_hold']
        if c:centre.append(np.stack([r['goal'][:3] for r in c]).mean(0))
        if a:
            r=a[-1];s=r['head_state'];clearance.append(s[26:29]+r['goal'][3:6]-r['goal'][:3]-s[3:6])
        if h:
            for r in h[-3:]:
                s=r['head_state'];insert.append(s[26:29]);axes.append(np.r_[s[29:31],s[32:34]])
    if not all((centre,clearance,insert,axes)):raise ValueError('Successful assembly training episodes with all phases are required')
    return dict(centre=np.mean(centre,0).astype(np.float32),relative=np.stack([np.mean(clearance,0),np.mean(insert,0)]).astype(np.float32),
        axes=np.mean(axes,0).astype(np.float32),carry=np.mean(carry,0).astype(np.float32),water=np.mean(water,0).astype(np.float32))

def supervised_targets(items,prototypes,moving_grasp_frame=False,water_goal_embedding=False,assembly_step_refinement=False):
    result=[];local=[];axis_delta=[];local_mask=[];axis_mask=[]
    for r in items:
        s=r['head_state'];p=r['phase'];g=r['goal'].copy();l=s[:3];right=l+s[3:6]
        tool=s[6:10];objects=np.r_[s[29:31],s[32:34]]
        pen=s[23:26]+.057*unit_axis(s[29:31]);cap=s[23:26]+s[26:29]-.0275*unit_axis(s[32:34])
        if p.startswith('water'):
            g[:3]=l;g[6:8]=tool[:2]
            if water_goal_embedding:g[3:6]=r['water_position']+r['rotation'][1]@prototypes['water']
            g[8:10]=tool[2:];g[10:12]=[.55,-.17]
        if p.startswith('pen'):
            g[3:6]=right;g[8:10]=tool[2:]
            if p=='pen_carry':g[:3]=l;g[6:8]=tool[:2]
        if p.startswith('dual_') and p!='dual_grasp':
            g[:3]=prototypes['centre'] if p=='dual_centre' else l
            if p=='dual_orient':g[3:6]=right
            elif p=='dual_centre':g[3:6]=right+g[:3]-l
            elif assembly_step_refinement:g[3:6]=r['goal'][3:6]
            else:g[3:6]=g[:3]+s[3:6]+prototypes['relative'][0 if p=='dual_align' else 1]-s[26:29]
            g[6:10]=tool-objects+prototypes['axes']
        if p in ('pen_grasp','dual_grasp') and r['pre'][0]==2:g[:3]=l;g[6:8]=tool[:2]
        if p=='dual_grasp' and r['pre'][1]==2:g[3:6]=right;g[8:10]=tool[2:]
        lm=np.zeros(6,np.float32);am=np.zeros(4,np.float32);lo=np.zeros(6,np.float32);ad=np.zeros(4,np.float32)
        if p in ('pen_grasp','dual_grasp') and r['pre'][0]!=2:
            anchor=s[20:23] if p=='water_approach' else s[12:15] if p=='pen_carry' else pen
            lo[:3]=r['rotation'][0].T@(g[:3]-anchor)/.05;lm[:3]=1
            reference=objects[:2] if p in ('pen_grasp','dual_grasp') else tool[:2]
            ad[:2]=(g[6:8]-reference)/.1;am[:2]=1
        if p.startswith('water'):
            lo[3:]=r['rotation'][1].T@(g[3:6]-r['water_position'])/.05;lm[3:]=1
            ad[2:]=(g[8:10]-tool[2:])/.1;am[2:]=1
        if p=='dual_grasp' and r['pre'][1]!=2:
            lo[3:]=r['rotation'][1].T@(g[3:6]-cap)/.05;lm[3:]=1
            ad[2:]=(g[8:10]-objects[2:])/.1;am[2:]=1
        if moving_grasp_frame:
            if p in ('pen_grasp','dual_grasp') and r['pre'][0]!=2:lo[:3]+=(r['rotation'][0].T@(pen-l)-s[41:44])/.05
            if p=='dual_grasp' and r['pre'][1]!=2:lo[3:]+=(r['rotation'][1].T@(cap-right)-s[44:47])/.05
        result.append(g);local.append(lo);axis_delta.append(ad);local_mask.append(lm);axis_mask.append(am)
    return tuple(np.stack(v) for v in (result,local,axis_delta,local_mask,axis_mask))

def attach_water_positions(items,episodes,use_depth=False):
    """The collector records image-derived depth estimates; no truth is used here."""
    for row in items:row['water_position']=row['head_state'][20:23].copy()
