"""
Rolling shutter model: image rows are read out sequentially, so a corner
in row v is observed at frame_time + (v - image_height / 2) * line_delay.
The frame time refers to the middle row.

Within one frame the relative motion between camera and body is modeled
with constant velocity, the twist estimated from the neighboring frames'
poses (central difference, one-sided at gaps).
"""
from typing import List
import numpy as np
import gtsam


def corner_time_offsets(corners: np.ndarray, image_height: int, line_delay: float) -> np.ndarray:
    """
    (4,2) corner pixels -> (4,) observation time relative to frame time (sec).
    """
    return (corners[:, 1] - image_height / 2) * line_delay


def relative_twists(T_cam_body: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Body-frame twist (GTSAM Pose3 local coordinates per second) of the
    camera-from-body pose at every frame, from neighboring frame estimates.
    T_cam_body: (N,4,4), NaN where not estimated
    Returns (N,6), NaN where no neighbor is estimated.
    """
    n = len(T_cam_body)
    valid = ~np.isnan(T_cam_body[:, 0, 0])
    twists = np.full((n, 6), np.nan)
    for i in range(n):
        if not valid[i]:
            continue
        prev_i = i - 1 if (i > 0 and valid[i - 1]) else i
        next_i = i + 1 if (i < n - 1 and valid[i + 1]) else i
        if prev_i == next_i:
            continue
        T_prev = gtsam.Pose3(T_cam_body[prev_i])
        T_next = gtsam.Pose3(T_cam_body[next_i])
        twists[i] = gtsam.Pose3.Logmap(T_prev.between(T_next)) / (times[next_i] - times[prev_i])
    return twists


def corner_motions(twist: np.ndarray, time_offsets: np.ndarray) -> List[gtsam.Pose3]:
    """
    Pose increment of the body from frame time to each corner's observation time.
    """
    if np.any(np.isnan(twist)):
        return [gtsam.Pose3() for _ in time_offsets]
    return [gtsam.Pose3.Expmap(twist * dt) for dt in time_offsets]
