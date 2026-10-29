import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from PIL import Image
from iss_capture.rgb_joint_policy import INPUT_KEYS,RGBJointPolicy,select_inputs,raw_state,load_demonstrations,observations
from iss_capture.public_dataset import load_observation

def measured():
    return {**{n:np.full((16,16,3),80,np.uint8) for n in ("front","side","previous_front")},
            "q":np.linspace(-.5,.5,12,dtype=np.float32),"q_velocity":np.linspace(-.1,.1,12,dtype=np.float32)}

def test_boundary_removes_depth_truth_and_derived_geometry():
    raw={**measured(),"front_depth":object(),"side_depth":object(),"evaluation_objects":object(),
         "depth_state":object(),"goal":object(),"contact_labels":object(),"previous_command":object()}
    selected=select_inputs(raw)
    assert set(selected)==set(INPUT_KEYS)
    np.testing.assert_array_equal(raw_state(selected),np.r_[raw["q"],raw["q_velocity"]])

def test_network_sees_only_raw_encoder_state_and_rgb():
    captured={}
    def predict(batch,noise):
        captured.update(batch);return torch.zeros((1,1,12))
    policy=RGBJointPolicy.__new__(RGBJointPolicy)
    policy.device="cpu";policy.base=SimpleNamespace(predict_action_chunk=predict)
    policy.tokenizer=lambda tasks,**kw:{"input_ids":torch.ones((1,48),dtype=torch.long),"attention_mask":torch.ones((1,48),dtype=torch.long)}
    policy.state_bias=np.zeros(24,np.float32);policy.state_scale=np.ones(24,np.float32)
    policy.action_bias=np.zeros(12,np.float32);policy.action_scale=np.ones(12,np.float32)
    policy.config={"policy_image_size":256,"joint_lower":[-10]*12,"joint_upper":[10]*12}
    obs=measured();command,info=policy.predict(obs,"Catch the pen.")
    np.testing.assert_array_equal(captured["observation.state"][0].numpy(),np.r_[obs["q"],obs["q_velocity"]])
    assert set(captured)=={"observation.state","observation.language.tokens","observation.language.attention_mask",
                          "observation.images.camera1","observation.images.camera2","observation.images.camera3"}
    np.testing.assert_allclose(command,obs["q"])
    with pytest.raises(ValueError,match="only images"):
        policy.predict({**obs,"front_depth":np.zeros((16,16))},"Catch the pen.")

def test_dataset_loader_and_replay_need_no_depth_files(tmp_path):
    episode=tmp_path/"train_task1_1";episode.mkdir()
    for n in ("front","side","previous_front"):
        Image.fromarray(np.zeros((16,16,3),np.uint8)).save(episode/f"00000_{n}.png")
    record={"frame":"00000","q":[.1]*12,"q_velocity":[.2]*12,"action":[.3]*12,
            "instruction":"Catch the droplet.","image_time":1.,"sim_time":1.,"controller_time":1.,
            "previous_command":[.1]*12,"depth_state":"DO NOT READ","evaluation_objects":"DO NOT READ",
            "goal":"DO NOT READ","contact_labels":"DO NOT READ"}
    (episode/"samples.json").write_text(json.dumps([record]))
    (episode/"result.json").write_text(json.dumps({"success":True,"split":"train","task":"task1"}))
    rows=load_demonstrations(tmp_path);assert len(rows)==1 and rows[0]["state"].shape==(24,)
    np.testing.assert_allclose(rows[0]["delta"],np.full(12,.2),atol=1e-6)
    assert observations(rows,256)[0]["front"].shape==(256,256,3)
    replay=load_observation(episode,record,include_depth=False)
    assert set(select_inputs(replay))==set(INPUT_KEYS)


def test_rgb_only_recording_has_no_depth_outputs(tmp_path):
    from iss_capture.task_scene_video import CameraRecording
    recording=CameraRecording(tmp_path)
    for stamp in (1.,1.1):
        recording.observe({"image_time":stamp,"front":np.zeros((16,16,3),np.uint8),
                           "side":np.full((16,16,3),20,np.uint8)})
    metadata=recording.finish([(1.,np.zeros((16,16,3),np.uint8)),(1.1,np.zeros((16,16,3),np.uint8))])
    assert metadata["depth_topics"]==[]
    assert "front_depth" not in metadata["files"] and "side_depth" not in metadata["files"]
    assert (tmp_path/"policy_front.mp4").is_file()
    assert not (tmp_path/"camera_frames/front_depth").exists()

def test_action_chunk_execution_uses_absolute_targets():
    calls=[]
    policy=RGBJointPolicy.__new__(RGBJointPolicy);policy.device='cpu'
    def predict(batch,noise):
        calls.append(1);out=torch.zeros((1,10,12));out[0,:,0]=torch.arange(1,11)/10
        out[0,:,[5,11]]=.55;return out
    policy.base=SimpleNamespace(predict_action_chunk=predict)
    policy.tokenizer=lambda tasks,**kw:{"input_ids":torch.ones((1,48),dtype=torch.long),"attention_mask":torch.ones((1,48),dtype=torch.long)}
    policy.state_bias=np.zeros(24,np.float32);policy.state_scale=np.ones(24,np.float32)
    policy.action_bias=np.zeros(12,np.float32);policy.action_scale=np.ones(12,np.float32)
    policy.config={"policy_image_size":256,"chunk_size":10,"execution_steps":3,"action_mode":"absolute",
                   "joint_lower":[-10]*12,"joint_upper":[10]*12,"gripper_values":[-.17,.55]}
    policy.reset()
    outputs=[policy.predict(measured(),'Catch the pen.')[0][0] for _ in range(4)]
    np.testing.assert_allclose(outputs,[.1,.2,.3,.1])
    assert len(calls)==2


def test_positions_only_excludes_velocity_and_extra_signals():
    raw={**measured(),"front_depth":object(),"evaluation_objects":object()}
    selected=select_inputs(raw,positions_only=True)
    assert set(selected)=={"front","side","previous_front","q"}
    np.testing.assert_array_equal(raw_state(selected,positions_only=True),raw["q"])
