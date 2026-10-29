import numpy as np
from iss_capture.encoder_alignment import aligned_encoders

def test_position_and_velocity_use_the_same_timestamp():
    history = [(1., np.array([0., 2.]), np.array([10., 20.])),
               (1.005, np.array([1., 3.]), np.array([30., 40.]))]
    q, dq = aligned_encoders(history, 1.0025)
    np.testing.assert_allclose(q, [.5, 2.5])
    np.testing.assert_allclose(dq, [20., 30.])

def test_stale_encoders_are_not_reported_as_synchronized():
    history = [(1., np.zeros(12), np.ones(12))]
    assert aligned_encoders(history, 1.02) is None
    assert aligned_encoders(history, .98) is None
    assert aligned_encoders(history, 1.002) is not None
    assert aligned_encoders([], 1.) is None
