import numpy as np
import pytest
from iss_capture.capping_monitor import CappingMonitor
from iss_capture.bimanual_config import PEN_TIP,CAP_LENGTH

class Client:
    def __init__(self,cap_jaw=.159,closed=True,depth=.032,lateral=0.,angle=0.,contacts=True):
        self.q=np.zeros(12);self.q[5]=-.1;self.q[11]=cap_jaw
        self.previous_command=np.zeros(12);self.previous_command[[5,11]]=-.17 if closed else .55
        self.t=0.;self.depth=depth;self.lateral=lateral;self.angle=angle;self.contacts=contacts
    def snapshot(self):return self.q.copy(),self.t,0.
    def bilateral(self,side):return self.contacts
    def object_states(self):
        angle=np.deg2rad(self.angle);axis=np.array([np.sin(angle),0.,np.cos(angle)])
        perpendicular=np.array([np.cos(angle),0.,-np.sin(angle)])
        tip=np.array([0.,0.,PEN_TIP])
        opening=tip-self.depth*axis+self.lateral*perpendicular
        return {"floating_pen":{"position":np.zeros(3),"axis":np.array([0.,0.,-1.]),"stamp":self.t},
                "floating_cap":{"position":opening+CAP_LENGTH/2*axis,"axis":axis,"stamp":self.t}}

def evaluate(client):
    monitor=CappingMonitor(require_carry=False)
    for i in range(12):
        client.t=i*.1;monitor.update(client)
    return monitor

def test_wide_cap_grasp_requires_real_contact_and_closing_command():
    monitor=evaluate(Client(cap_jaw=.159))
    assert monitor.success
    assert not monitor.legacy_success
    assert not monitor.last["legacy_grip_angle_check"]
    assert all(monitor.last["closing_grip_command"].values())

def test_original_body_grasp_still_passes_both_criteria():
    monitor=evaluate(Client(cap_jaw=.024))
    assert monitor.success and monitor.legacy_success

@pytest.mark.parametrize("changes",[
    {"closed":False},{"contacts":False},{"depth":.001},
    {"depth":.045},{"lateral":.003},{"angle":12.}])
def test_contact_change_does_not_relax_geometry_or_allow_open_jaws(changes):
    assert not evaluate(Client(**changes)).success

def test_six_tenths_stability_is_still_required():
    client=Client();monitor=CappingMonitor(require_carry=False)
    for i in range(6):
        client.t=i*.1;monitor.update(client)
    assert not monitor.success
