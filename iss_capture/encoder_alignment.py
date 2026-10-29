"""Align raw encoder position and velocity to the same camera timestamp."""
import numpy as np

def aligned_encoders(history, stamp, max_extrapolation=.0050001):
    if not history:
        return None
    times = np.asarray([sample[0] for sample in history])
    if stamp < times[0] - max_extrapolation or stamp > times[-1] + max_extrapolation:
        return None
    hi = min(len(history) - 1, int(np.searchsorted(times, stamp)))
    lo = max(0, hi - 1)
    weight = np.clip((stamp-times[lo]) / max(times[hi]-times[lo], 1e-9), 0, 1)
    return tuple(history[lo][k]*(1-weight) + history[hi][k]*weight for k in (1, 2))
