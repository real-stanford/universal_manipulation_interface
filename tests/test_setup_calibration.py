# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import numpy as np
from scipy.spatial.transform import Rotation
from umi.tag_tracking.camera_model import FisheyeCamera
from umi.tag_tracking.noise_model import TagNoiseModel
from umi.tag_tracking.tag_detection import get_tag_object_points
from umi.tag_tracking.setup_calibration import (frames_with_neighbors, select_cube_frames, robust_mean_transform,
    transform_residuals, interpolate_pose, wrist_to_tcp_umi, cube_to_wrist)

CAMERA = FisheyeCamera('example/calibration/gopro_intrinsics_2_7k.json')
TAG_SIZE = 0.06
ENV_TAG_SIZE = 0.08


def pose(rotvec_deg, t):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.radians(rotvec_deg)).as_matrix()
    T[:3, 3] = t
    return T


def _facing(normal_rotvec_deg, t):
    T = pose(normal_rotvec_deg, t) @ pose([180, 0, 0], [0, 0, 0])
    return T


def observe(T_cam_tags, sizes, rng, noise_px=0.2):
    corners = dict()
    for k, T in T_cam_tags.items():
        points = (T[:3, :3] @ get_tag_object_points(sizes[k]).T).T + T[:3, 3]
        if np.any(points[:, 2] < 0.05):
            continue
        px = CAMERA.project(points)
        if np.all((px >= 0) & (px < np.array(CAMERA.image_size))):
            corners[k] = px + rng.normal(scale=noise_px, size=px.shape)
    return corners


def test_frames_with_neighbors():
    frames = [{'time': i / 60, 'corners': {}} for i in range(20)]
    out, position = frames_with_neighbors(frames, [5, 6, 15], 'v')
    assert sorted(position) == [4, 5, 6, 7, 14, 15, 16]
    assert out[position[4]]['sequence'] == out[position[7]]['sequence'] != out[position[15]]['sequence']


def test_select_cube_frames():
    frames = [{'corners': {2: 0, 3: 0}} for _ in range(1000)] + [{'corners': {4: 0, 5: 0}} for _ in range(5)]
    idx = select_cube_frames(frames, [2, 3, 4, 5], per_pair=50)
    # common pair capped, rare pair kept entirely
    assert sum(i < 1000 for i in idx) == 50 and sum(i >= 1000 for i in idx) == 5


def test_robust_mean_and_tcp():
    rng = np.random.default_rng(0)
    T_true = pose([10, -20, 5], [0.01, 0.08, -0.05])
    samples = np.array([T_true @ pose(rng.normal(scale=0.2, size=3), rng.normal(scale=0.001, size=3)) for _ in range(200)]
        + [pose([60, 0, 0], [0.3, 0.3, 0.3]) for _ in range(20)])
    T_mean, inliers = robust_mean_transform(samples)
    d_t, d_r = transform_residuals(T_mean[None], T_true)
    assert d_t[0] < 0.0005 and np.degrees(d_r[0]) < 0.1 and not inliers[200:].any()
    assert np.allclose(interpolate_pose(pose([0, 0, 0], [0, 0, 0]), pose([0, 0, 90], [1, 0, 0]), 0.5),
        pose([0, 0, 45], [0.5, 0, 0]))
    assert np.allclose(wrist_to_tcp_umi()[:3, 3], [0, 0.086, 0.21965])


def make_scene():
    rng = np.random.default_rng(1)
    # head camera = world frame (looking +z); environment tags in front of the
    # head (z = 1.5) and to its right (x = 0.9, facing -x) for the wrist camera
    T_env_tags = {20 + i: _facing([0, 0, 0], [x, y, 1.5]) for i, (x, y) in enumerate([(-0.3, -0.2), (0.0, 0.25), (0.3, -0.1)])}
    T_env_tags.update({30 + i: _facing([0, -90, 0], [0.9, y, z]) for i, (y, z) in enumerate([(-0.15, 0.5), (0.1, 0.65), (-0.05, 0.85)])})
    # cube: two tags facing the head, tilted for a 3D layout
    T_cube_tags = {2: _facing([0, 25, 0], [-0.03, 0, -0.035]), 3: _facing([0, -25, 0], [0.03, 0, -0.035])}
    # wrist camera rigidly on the cube, looking +x (toward the x = 0.9 tags)
    T_cube_wrist_true = pose([0, 90, 0], [0.02, 0.07, 0.04]) @ pose([0, 0, 3], [0, 0, 0])
    sizes = {**{k: ENV_TAG_SIZE for k in T_env_tags}, **{k: TAG_SIZE for k in T_cube_tags}}

    def T_world_cube(t):
        return pose([5 * np.sin(t), 8 * np.sin(0.7 * t), 3 * np.cos(t)],
            [0.05 * np.sin(0.5 * t), 0.03 * np.cos(0.4 * t), 0.6 + 0.02 * np.sin(0.3 * t)])

    head_times = np.arange(0, 6, 1 / 60)
    wrist_times = head_times[:-1] + 0.005
    head_frames = [{'time': t, 'corners': observe({**T_env_tags, **{k: T_world_cube(t) @ v for k, v in T_cube_tags.items()}},
        sizes, rng)} for t in head_times]
    wrist_frames = list()
    for t in wrist_times:
        T_wrist_world = np.linalg.inv(T_world_cube(t) @ T_cube_wrist_true)
        wrist_frames.append({'time': t, 'corners': observe({k: T_wrist_world @ v for k, v in T_env_tags.items()}, sizes, rng)})
    assert np.mean([len(f['corners']) >= 2 for f in wrist_frames]) > 0.9, "wrist must see env tags in the synthetic scene"
    return T_env_tags, T_cube_tags, T_cube_wrist_true, sizes, head_times, head_frames, wrist_times, wrist_frames


def test_cube_to_wrist(executor=None):
    T_env_tags, T_cube_tags, T_cube_wrist_true, sizes, head_times, head_frames, wrist_times, wrist_frames = make_scene()

    noise = TagNoiseModel(0.0, {k: 0.3 for k in sizes})
    accuracy = {'position_std': 0.01, 'rotation_std_rad': np.radians(3.0)}
    result = cube_to_wrist(CAMERA, head_frames, head_times, T_env_tags, T_cube_tags, sizes, noise, 0.0,
        CAMERA, wrist_frames, wrist_times, 0.3, accuracy, max_samples=120, max_speed=0.5,
        executor=executor, n_chunks=7)
    assert result is not None and result['T_cube_wrist'] is not None, result
    d_t, d_r = transform_residuals(result['T_cube_wrist'][None], T_cube_wrist_true)
    print(f"  cube to wrist error {d_t[0] * 1000:.2f} mm {np.degrees(d_r[0]):.3f} deg, stats {result['stats']}")
    assert d_t[0] < 0.002 and np.degrees(d_r[0]) < 0.3


def test_parallel_tracking_identical(executor):
    from umi.tag_tracking.rigid_body import track_rigid_body
    from umi.tag_tracking.parallel import parallel_track_rigid_body
    T_env_tags, T_cube_tags, _, sizes, _, head_frames, _, _ = make_scene()
    frames = [{**f, 'sequence': 0 if i < 200 else 1} for i, f in enumerate(head_frames)]
    noise = TagNoiseModel(0.0, {k: 0.3 for k in sizes})
    for T_tags in (T_env_tags, T_cube_tags):
        full = track_rigid_body(CAMERA, frames, T_tags, sizes, noise, 5e-6)
        chunked = parallel_track_rigid_body(CAMERA, frames, T_tags, sizes, noise, 5e-6, executor, n_chunks=7)
        for k in full:
            assert np.allclose(full[k], chunked[k], equal_nan=True, atol=1e-9), k


def test():
    test_frames_with_neighbors()
    test_select_cube_frames()
    test_robust_mean_and_tcp()
    test_cube_to_wrist()
    from umi.tag_tracking.parallel import make_executor
    with make_executor(4) as executor:
        test_parallel_tracking_identical(executor)
        test_cube_to_wrist(executor)
    print('all passed')


if __name__ == '__main__':
    test()
