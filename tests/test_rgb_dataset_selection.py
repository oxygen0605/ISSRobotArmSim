import json
from pathlib import Path
import pytest
from prepare_rgb_dataset import select_episodes, audit_episode

def result_file(root, task, split, seed, success):
    directory=root/f"{split}_{task}_{seed}"
    directory.mkdir()
    result={"task":task,"split":split,"success":success}
    path=directory/"result.json"
    path.write_text(json.dumps(result))
    return path

def test_selection_excludes_failed_trials_and_keeps_splits(tmp_path):
    for task in ("task1","task2","task3"):
        result_file(tmp_path,task,"train",100,False)
        result_file(tmp_path,task,"train",101,True)
        result_file(tmp_path,task,"train",102,True)
        result_file(tmp_path,task,"validation",200,True)
    selected=select_episodes(tmp_path,train_count=1,validation_count=1)
    assert len(selected)==6
    assert all(result["success"] for _,result in selected)
    assert {path.parent.name.rsplit("_",1)[1] for path,_ in selected}=={"101","200"}
    assert sum(result["split"]=="validation" for _,result in selected)==3

def test_selection_refuses_to_count_failed_trials_toward_target(tmp_path):
    for task in ("task1","task2","task3"):
        result_file(tmp_path,task,"train",100,False)
    with pytest.raises(ValueError,match="successful episodes"):
        select_episodes(tmp_path,train_count=1,validation_count=0)

def test_audit_rejects_missing_rgb_even_if_result_says_success(tmp_path):
    path=result_file(tmp_path,"task1","train",101,True)
    record={"q":[0.]*12,"q_velocity":[0.]*12,"action":[0.]*12,
            "instruction":"catch water","frame":"00000"}
    path.with_name("samples.json").write_text(json.dumps([record]))
    with pytest.raises(ValueError,match="Missing RGB"):
        audit_episode(path,{"scene":{"instruction":"catch water"}})
