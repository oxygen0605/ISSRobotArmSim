import numpy as np
import pytest
from iss_capture.rgb_training_schedule import learning_rate,batch_layout,task_index,attach_future_targets,masked_mean


def test_warmup_cosine_hits_requested_peak_and_final_update():
    rates=np.array([learning_rate(i,30000,1000,3e-4,3e-5) for i in range(30001)])
    assert rates[0]==0
    assert rates[1]==pytest.approx(3e-7)
    assert rates[1000]==pytest.approx(3e-4)
    assert rates[15500]==pytest.approx(1.65e-4)
    assert rates[30000]==pytest.approx(3e-5)
    assert (np.diff(rates[:1001])>0).all()
    assert (np.diff(rates[1000:])<=0).all()


@pytest.mark.parametrize('micro',[1,2,4,8,16,32,64])
def test_effective_batch_and_task_balance_survive_accumulation(micro):
    size,accumulation=batch_layout(64,micro)
    totals=np.zeros(3,int)
    for step in (1,2,3):
        tasks=[task_index(step,64,part*size+j) for part in range(accumulation) for j in range(size)]
        assert len(tasks)==64
        counts=np.bincount(tasks,minlength=3)
        assert counts.max()-counts.min()==1
        totals+=counts
    np.testing.assert_array_equal(totals,[64,64,64])


def test_invalid_microbatch_is_not_silently_rounded():
    with pytest.raises(ValueError):batch_layout(64,12)


def test_fifty_action_targets_never_cross_episode_boundary():
    rows=[dict(episode=ep,action=np.full(12,value)) for ep,value in [('a',1),('a',2),('b',99)]]
    attach_future_targets(rows,50,lambda row:row['action'])
    assert all(row['future'].shape==(50,12) for row in rows)
    np.testing.assert_array_equal(rows[0]['future'][:,0],[1,*([2]*49)])
    assert (rows[1]['future']==2).all()
    assert (rows[2]['future']==99).all()
    assert [r['future_valid'].sum() for r in rows]==[2,1,1]


def test_padded_target_errors_do_not_affect_reported_error():
    values=np.array([[[1,3],[1e9,1e9]]])
    assert masked_mean(values,[[True,False]])==2
