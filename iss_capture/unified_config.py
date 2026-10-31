"""Common embodiment and language contract for one multi-task SmolVLA."""
import numpy as np
from .config import ROOT
TASKS = {
    'task1': 'Use the left green sponge to catch the rising water droplet. Keep the right arm still.',
    'task2': 'Catch the rotating orange pen with the left gripper, lift it and hold it. Keep the right arm still.',
    'task3': 'Catch the orange pen with the left gripper and the blue cap with the right gripper. Align and insert the cap onto the pen tip, then hold both.'
}
SPONGE = np.array([.025,-.045,-.065])
SPONGE_SIZE = [.026,.025,.030]
CAMERAS = {
    'front': dict(eye=[-.08,0,.48], target=[.31,0,.28], fov=1.25),
    'side': dict(eye=[.37,-.50,.39], target=[.31,0,.28], fov=.95)
}
CHECKPOINT = ROOT/'artifacts/smolvla_rgb_only'
DT = .1

def case_from_seed(seed):
    # Independent streams: object position must not encode the motion parameters.
    streams = np.random.SeedSequence(int(seed)).spawn(4)
    pos, tilt, motion, water = [np.random.default_rng(s) for s in streams]
    direction=motion.uniform(-np.pi,np.pi)
    spin=motion.uniform(.15,.30)
    velocity=motion.normal(size=3);velocity*=motion.uniform(.002,.005)/np.linalg.norm(velocity)
    return dict(seed=int(seed),
        pen_position=(np.array([.29,.035,.215])+pos.uniform(-.006,.006,3)).tolist(),
        cap_position=(np.array([.29,-.060,.325])+pos.uniform(-.006,.006,3)).tolist(),
        pen_tilt=tilt.uniform(-.035,.035,2).tolist(),cap_tilt=tilt.uniform(-.02,.02,2).tolist(),
        linear_velocity=velocity.tolist(),angular_velocity=[spin*np.cos(direction),spin*np.sin(direction),0],
        water_position=(np.array([.275,.160,.245])+water.uniform(-.008,.008,3)).tolist(),
        water_velocity=[0,0,float(water.uniform(.004,.010))])

def prepare_splits(path, ntrain=16, nval=4, ndev=6, ntest=12):
    import json
    path=ROOT/path
    if path.exists():raise FileExistsError(path)
    rows={}
    for split,n,start in [('train',ntrain,3100000),('validation',nval,3200000),('development',ndev,3300000),('test',ntest,3400000)]:
        rows[split]=[case_from_seed(start+i) for i in range(n)]
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(version=3,splits=rows,shared_scenes_across_tasks=True,
        independence='SeedSequence separate position/tilt/motion/water streams',test_used_for_selection=False),indent=2))
    return rows

ASSEMBLY_POSITION = np.array([.28,0.,.22])
