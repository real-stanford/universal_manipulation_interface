"""
umi.tag_tracking.delivery: alignment to the primary camera frames without
filling lost data.

python tests/test_delivery.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial.transform import Rotation
from umi.tag_tracking.delivery import align_poses, align_values, episode_frames, pose7

FPS = 60000 / 1001


def trajectory(times):
    """
    Smooth pose: position and rotation known at any time.
    """
    T = np.tile(np.eye(4), (len(times), 1, 1))
    T[:, :3, 3] = np.stack([0.3 * np.sin(times), 0.2 * np.cos(0.7 * times), 0.1 + 0.05 * times], axis=1)
    T[:, :3, :3] = Rotation.from_rotvec(np.stack([0.5 * np.sin(times), 0.3 * times, 0.2 * np.cos(times)], axis=1)).as_matrix()
    return T


def test_align():
    wrist_t = np.arange(0.0, 10.0, 1 / FPS) + 0.004      # another camera's phase
    head_t = np.arange(0.5, 9.5, 1 / FPS)
    T = trajectory(wrist_t)
    valid = np.ones(len(wrist_t), dtype=bool)
    valid[200:215] = False                                # lost tracking (0.25 s)
    T[200:215] = np.nan
    pose, ok = align_poses(wrist_t, T, valid, head_t, 1.5 / FPS)
    # never aligned across the gap: head frames between wrist samples 199 and 215 are None
    in_gap = (head_t > wrist_t[199]) & (head_t < wrist_t[215])
    assert in_gap.any() and not ok[in_gap].any() and np.all(np.isnan(pose[in_gap]))
    assert ok[~in_gap].all()
    # aligned values close to the true trajectory (interpolation error over one frame)
    truth = pose7(trajectory(head_t))
    position_error = np.linalg.norm(pose[ok, :3] - truth[ok, :3], axis=1)
    rotation_error = (Rotation.from_quat(pose[ok, 3:]).inv() * Rotation.from_quat(truth[ok, 3:])).magnitude()
    assert position_error.max() < 2e-5, position_error.max()
    assert rotation_error.max() < 1e-4, rotation_error.max()
    assert np.allclose(np.linalg.norm(pose[ok, 3:], axis=1), 1.0)
    # exact hits: query at a sample time
    p, o = align_poses(wrist_t, T, valid, wrist_t[[10, 199]], 1.5 / FPS)
    assert o.all() and np.allclose(p, pose7(T[[10, 199]]))


def test_values():
    t = np.arange(0.0, 2.0, 1 / FPS)
    w = 0.05 + 0.02 * t
    w[30:33] = np.nan                                      # finger tags not seen
    q = np.arange(0.1, 1.9, 1 / FPS) + 0.003
    out = align_values(t, w, q, 1.5 / FPS)
    gap = (q > t[29]) & (q < t[33])
    assert np.all(np.isnan(out[gap])) and not np.any(np.isnan(out[~gap]))
    assert np.allclose(out[~gap], 0.05 + 0.02 * q[~gap], atol=1e-9)


def test_episode_frames():
    head_t = np.arange(0.0, 20.0, 1 / FPS)
    head_T = trajectory(head_t)
    head_valid = np.ones(len(head_t), dtype=bool)
    head_valid[100:110] = False
    wrist_t = np.arange(0.0, 20.0, 1 / FPS) + 0.007
    T = trajectory(wrist_t)
    valid = np.ones(len(wrist_t), dtype=bool)
    valid[400:420] = False
    widths = np.full(len(wrist_t), 0.08)
    grippers = [{'side': side, 'pose_times': wrist_t, 'T_world_tcp': T, 'pose_valid': valid,
        'width_times': wrist_t, 'widths': widths, 'width_tags': np.full(len(wrist_t), 2)} for side in ('right', 'left')]
    out = episode_frames(head_t, head_T, head_valid, 2.0, 12.0, grippers)
    f = out['frames']
    rows = np.flatnonzero((head_t >= 2.0) & (head_t < 12.0))
    assert np.array_equal(f['head_frame_index'], rows) and np.all(np.diff(f['time']) > 0)
    assert not f['camera_pose_head_valid'][np.isin(rows, np.arange(100, 110))].any()
    assert f['camera_pose_head_valid'][~np.isin(rows, np.arange(100, 110))].all()
    lost = (f['time'] > wrist_t[399]) & (f['time'] < wrist_t[420])
    assert not f['pose_valid'][lost].any() and f['pose_valid'][~lost].all()
    assert np.all(np.isnan(f['ee_pose_left'][lost])) and np.all(np.isnan(f['ee_pose_right'][lost]))
    native = out['native']['right']
    assert np.all(native['pose_time'] >= 2.0) and np.all(native['pose_time'] < 12.0)
    assert np.isnan(native['pose'][~native['pose_valid']]).all()


if __name__ == '__main__':
    test_align()
    test_values()
    test_episode_frames()
    print("all passed")
