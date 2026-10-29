from pathlib import Path
import hashlib, io, json, tarfile
import numpy as np
import pytest
from PIL import Image
from iss_capture.public_dataset import extract_episode, load_observation

def bundle(tmp_path, name="episode/result.json", content=b"{}"):
    archive=tmp_path/"episode.tar"
    with tarfile.open(archive,"w") as tar:
        info=tarfile.TarInfo(name);info.size=len(content)
        tar.addfile(info,io.BytesIO(content))
    row={"path":name,"bytes":len(content),"sha256":hashlib.sha256(content).hexdigest()}
    spec={"id":"episode","archive":"episode.tar","sha256":hashlib.sha256(archive.read_bytes()).hexdigest()}
    dest=tmp_path/"destination";dest.mkdir()
    return archive,dest,spec,[row]

def test_extract_and_reverify(tmp_path):
    archive,dest,spec,rows=bundle(tmp_path)
    extract_episode(archive,dest,spec,rows)
    extract_episode(archive,dest,spec,rows)
    assert (dest/"episode/result.json").read_bytes()==b"{}"

def test_corrupted_archive_rejected(tmp_path):
    archive,dest,spec,rows=bundle(tmp_path)
    archive.write_bytes(archive.read_bytes()+b"corrupt")
    with pytest.raises(RuntimeError,match="Archive hash"):extract_episode(archive,dest,spec,rows)
    assert not (dest/"episode").exists()

def test_path_traversal_rejected(tmp_path):
    archive,dest,spec,rows=bundle(tmp_path,"../outside.json")
    with pytest.raises(ValueError,match="Unexpected archive member"):extract_episode(archive,dest,spec,rows)
    assert not (tmp_path/"outside.json").exists()

def test_existing_modified_data_not_overwritten(tmp_path):
    archive,dest,spec,rows=bundle(tmp_path)
    extract_episode(archive,dest,spec,rows)
    p=dest/"episode/result.json";p.write_text("edited")
    with pytest.raises(RuntimeError,match="verification failed"):extract_episode(archive,dest,spec,rows)
    assert p.read_text()=="edited"

def test_replay_input_excludes_privileged_labels(tmp_path):
    for name in ("front","side","previous_front"):
        Image.fromarray(np.zeros((3,4,3),dtype=np.uint8)).save(tmp_path/f"00000_{name}.png")
    np.savez(tmp_path/"00000_depth.npz",front=np.ones((3,4),np.float32),side=np.ones((3,4),np.float32))
    record={"frame":"00000","q":[0]*12,"q_velocity":[0]*12,"previous_command":[0]*12,
            "image_time":1.0,"controller_time":1.0,"sim_time":1.0,
            "evaluation_objects":{"secret":"privileged"},"goal":[9]*12,"action":[9]*12,"contact_labels":{"left":True}}
    obs=load_observation(tmp_path,record)
    assert obs["front"].shape==(3,4,3) and obs["front_depth"].dtype==np.float32
    assert not {"evaluation_objects","goal","action","contact_labels"} & obs.keys()
