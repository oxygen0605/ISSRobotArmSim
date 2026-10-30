from pathlib import Path
import evaluate_rgb_policy as evaluation

def invoke(monkeypatch,tmp_path,responses,retries=2):
    calls=[]
    def fake_run(policy,path,task,spec,case,**kw):
        calls.append(path)
        return {"task":task,**responses[len(calls)-1]}
    monkeypatch.setattr(evaluation,"run",fake_run)
    result=evaluation.evaluate_case(None,tmp_path,"task1",123,{}, {},200,False,retries)
    return result,calls

def test_policy_failure_is_not_retried(monkeypatch,tmp_path):
    result,calls=invoke(monkeypatch,tmp_path,[{"success":False,"error":None}])
    assert len(calls)==1 and not result["success"]
    summary=evaluation.summarize([result],["task1"],1)["task1"]
    assert summary["valid_trials"]==1 and summary["rate"]==0

def test_infrastructure_retry_is_preserved(monkeypatch,tmp_path):
    result,calls=invoke(monkeypatch,tmp_path,[
        {"success":False,"error":{"kind":"infrastructure_error"}},
        {"success":True,"error":None}])
    assert len(calls)==2 and result["infrastructure_retries"]==1
    assert calls[0].name=="task1_123" and calls[1].name=="task1_123_infra_retry_1"
    summary=evaluation.summarize([result],["task1"],1)["task1"]
    assert summary["attempted"]==1 and summary["valid_trials"]==1
    assert summary["infrastructure_retries"]==1 and summary["rate"]==1

def test_unresolved_infrastructure_error_is_separate_from_policy_rate(monkeypatch,tmp_path):
    result,calls=invoke(monkeypatch,tmp_path,[
        {"success":False,"error":{"kind":"infrastructure_error"}},
        {"success":False,"error":{"kind":"infrastructure_error"}}],retries=1)
    assert len(calls)==2
    summary=evaluation.summarize([result],["task1"],1)["task1"]
    assert summary["errors"]==1 and summary["valid_trials"]==0 and summary["rate"] is None
