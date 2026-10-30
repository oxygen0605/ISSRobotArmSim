import torch
from train_rgb_only import endpoint_actions
from iss_capture.capping_kinematic_loss import KinematicLoss

def test_endpoint_decoder_matches_lerobot_velocity_sign_and_preserves_gradient():
    bias=torch.linspace(-.2,.2,12)
    scale=torch.linspace(.1,.3,12)
    target=torch.linspace(-.5,.5,12)[None,None,:]
    velocity=(-(target-bias)/scale).detach().requires_grad_()
    actual=endpoint_actions(velocity,bias,scale)
    torch.testing.assert_close(actual,target)
    actual.sum().backward()
    torch.testing.assert_close(velocity.grad,-scale[None,None,:])

def test_kinematic_loss_is_zero_at_teacher_action_and_differentiable():
    metric=KinematicLoss(device="cpu")
    target=torch.tensor([[.4,-1.2,1.1,.1,1.6,-.17,-.2,-.7,-.07,.8,1.6,-.17]])
    assert metric(target,target).item()==0.
    predicted=target.clone();predicted[0,1]+=.01;predicted.requires_grad_()
    loss=metric(predicted,target)
    assert loss.item()>0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad.abs().sum().item()>0
