"""
Gripper pose by visual-inertial smoothing of each gripper's wrist camera
(umi.tag_tracking.visual_inertial.fuse_camera_trajectory), per episode window:
- wrist IMU preintegration between consecutive wrist frames;
- environment tags seen by the wrist camera (corner projection factors, known
  tag poses from the setup calibration);
- the wrist camera pose measured by the head camera: its cube tracked in the
  world (03_track_bodies.py / 03b_fuse_head_imu.py) times T_cube_wrist (setup
  calibration), interpolated to the wrist frame time, with the cube's world
  covariance carried to the wrist camera.
Where the head camera does not see the cube, the wrist IMU and environment
tags bridge the gap; the smoothed covariance grows with the gap, so validity
(accuracy limits) reflects how well the gap is bridged.

All inputs are on the wrist video's clock (its frames and IMU); head times
are converted with the audio synchronized start times.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import gtsam
from umi.tag_tracking.setup_calibration import interpolate_pose


def adjoint_inverse(T: np.ndarray) -> np.ndarray:
    return gtsam.Pose3(np.linalg.inv(T)).AdjointMap()


def wrist_measurements(wrist_times: np.ndarray, head_times: np.ndarray, T_world_cube: np.ndarray,
        world_covariance: np.ndarray, cube_valid: np.ndarray, T_cube_wrist: np.ndarray,
        max_head_gap: float) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Wrist camera pose measurements at wrist frames, times on one clock.
    A wrist frame gets a measurement if the head frames before and after it,
    at most max_head_gap apart, both have a valid cube pose: the cube pose is
    interpolated to the wrist frame time, the covariance taken from the nearer
    head frame and carried to the wrist camera:
        T_world_wrist = T_world_cube * T_cube_wrist,
        perturbation xi of the cube -> Ad(T_cube_wrist^-1) xi of the wrist.
    Returns wrist frame index -> (T_world_wrist, covariance).
    """
    Ad = adjoint_inverse(T_cube_wrist)
    k1 = np.searchsorted(head_times, wrist_times, side='right')
    out = dict()
    for i, (t, b) in enumerate(zip(wrist_times, k1)):
        if b <= 0 or b >= len(head_times):
            continue
        a = b - 1
        if not (cube_valid[a] and cube_valid[b]) or head_times[b] - head_times[a] > max_head_gap:
            continue
        alpha = (t - head_times[a]) / (head_times[b] - head_times[a])
        T_cube = interpolate_pose(T_world_cube[a], T_world_cube[b], alpha)
        covariance = world_covariance[a if alpha < 0.5 else b]
        out[i] = (T_cube @ T_cube_wrist, Ad @ covariance @ Ad.T)
    return out


def init_poses(n: int, measurements: Dict[int, Tuple[np.ndarray, np.ndarray]],
        tag_poses: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Initial wrist camera poses: head measurements, else wrist tag-only poses,
    NaN elsewhere (filled by the smoother); None if there is no pose at all.
    """
    T = np.full((n, 4, 4), np.nan) if tag_poses is None else tag_poses.copy()
    for i, (T_world_wrist, _) in measurements.items():
        T[i] = T_world_wrist
    return None if np.all(np.isnan(T[:, 0, 0])) else T
