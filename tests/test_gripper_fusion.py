# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import numpy as np
import gtsam
from scipy.spatial.transform import Rotation
from umi.tag_tracking.camera_model import FisheyeCamera
from umi.tag_tracking.noise_model import TagNoiseModel
from umi.tag_tracking.visual_inertial import fuse_camera_trajectory
from umi.tag_tracking.gripper_fusion import wrist_measurements, init_poses, adjoint_inverse
from umi.tag_tracking.setup_calibration import transform_residuals

CAMERA = FisheyeCamera('example/calibration/gopro_intrinsics_2_7k.json')
GRAVITY = 9.80665


def pose(rotvec_deg, t):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.radians(rotvec_deg)).as_matrix()
    T[:3, 3] = t
    return T


def T_world_imu(t):
    # smooth hand-like motion, world z up (gravity -z)
    return pose([20 * np.sin(0.9 * t), 15 * np.sin(0.6 * t + 1), 30 * np.sin(0.4 * t)],
        [0.15 * np.sin(1.1 * t), 0.1 * np.cos(0.8 * t), 1.0 + 0.05 * np.sin(1.5 * t)])


def synthetic_imu(times):
    """
    Exact gyro (body rate) and accelerometer (specific force) in the IMU frame.
    """
    h = 1e-4
    gyro, accel = list(), list()
    for t in times:
        R0, R1 = T_world_imu(t - h)[:3, :3], T_world_imu(t + h)[:3, :3]
        R = T_world_imu(t)[:3, :3]
        gyro.append(Rotation.from_matrix(R0.T @ R1).as_rotvec() / (2 * h))
        p = lambda s: T_world_imu(s)[:3, 3]
        a_world = (p(t + h) - 2 * p(t) + p(t - h)) / h ** 2
        accel.append(R.T @ (a_world + np.array([0, 0, GRAVITY])))
    return {'time': np.array(times), 'accel': np.array(accel), 'gyro': np.array(gyro)}


def test_measurement_covariance_transform():
    # a small cube perturbation xi maps to Ad(C^-1) xi on the wrist
    rng = np.random.default_rng(0)
    T_cube = pose([10, 20, 30], [0.1, 0.2, 0.3])
    C = pose([0, 90, 5], [0.02, 0.07, 0.04])
    xi = rng.normal(scale=1e-4, size=6)
    lhs = gtsam.Pose3(T_cube).compose(gtsam.Pose3.Expmap(xi)).compose(gtsam.Pose3(C)).matrix()
    rhs = gtsam.Pose3(T_cube @ C).compose(gtsam.Pose3.Expmap(adjoint_inverse(C) @ xi)).matrix()
    assert np.allclose(lhs, rhs, atol=1e-9)


def test_wrist_measurements():
    head_times = np.arange(0, 1, 1 / 60)
    T_cube = np.array([pose([0, 0, 0], [t, 0, 0]) for t in head_times])
    cov = np.tile(np.eye(6) * 1e-6, (len(head_times), 1, 1))
    valid = np.ones(len(head_times), dtype=bool)
    valid[30] = False
    C = pose([0, 0, 0], [0, 0, 0.1])
    wrist_times = head_times[:-1] + 0.005
    m = wrist_measurements(wrist_times, head_times, T_cube, cov, valid, C, 0.05)
    assert 29 not in m and 30 not in m and 10 in m
    assert np.allclose(m[10][0][:3, 3], [wrist_times[10], 0, 0.1])
    assert init_poses(5, {}) is None


def test_fusion_bridges_gap():
    rng = np.random.default_rng(1)
    E = pose([0, 0, 90], [0.0, 0.01, 0.02])   # T_imu_cam
    imu_times = np.arange(0, 6, 1 / 200)
    imu = synthetic_imu(imu_times)
    frame_times = np.arange(0.1, 5.9, 1 / 60)
    truth = np.array([T_world_imu(t) @ E for t in frame_times])
    sigma_t, sigma_r = 0.001, np.radians(0.2)
    covariance = np.diag([sigma_r ** 2] * 3 + [sigma_t ** 2] * 3)
    gap = (frame_times > 2.5) & (frame_times < 3.5)
    measurements = dict()
    for i in np.flatnonzero(~gap):
        noise = np.concatenate([rng.normal(scale=sigma_r, size=3), rng.normal(scale=sigma_t, size=3)])
        measurements[int(i)] = (gtsam.Pose3(truth[i]).compose(gtsam.Pose3.Expmap(noise)).matrix(), covariance)
    frames = [{'time': t, 'corners': {}} for t in frame_times]
    noise_setting = {'NoiseGyro': 0.0015, 'NoiseAcc': 0.017, 'GyroWalk': 5e-05, 'AccWalk': 0.0055}
    fused = fuse_camera_trajectory(CAMERA, frames, {}, {}, TagNoiseModel(0.0, {}), imu, E, 0.0,
        noise_setting, init_poses(len(frames), measurements), pose_measurements=measurements)
    err_t = np.array([np.linalg.norm(fused['T_world_cam'][i][:3, 3] - truth[i][:3, 3]) for i in range(len(frames))])
    err_r = np.array([transform_residuals(fused['T_world_cam'][i][None], truth[i])[1][0] for i in range(len(frames))])
    print(f"  measured frames: {err_t[~gap].max() * 1000:.2f} mm {np.degrees(err_r[~gap].max()):.3f} deg max error; "
        f"1 s gap: {err_t[gap].max() * 1000:.2f} mm {np.degrees(err_r[gap].max()):.3f} deg max error, "
        f"std in gap up to {np.nanmax(fused['position_std'][gap]) * 1000:.1f} mm")
    assert err_t[~gap].max() < 0.002 and np.degrees(err_r[~gap].max()) < 0.3
    # the IMU bridges the 1 s gap (needs a negligible bias init covariance, see visual_inertial)
    assert err_t[gap].max() < 0.005 and np.degrees(err_r[gap].max()) < 0.3
    # uncertainty grows inside the gap, but stays realistic (above the error, not inflated)
    std_gap = np.nanmax(fused['position_std'][gap])
    assert 2 * np.nanmedian(fused['position_std'][~gap]) < std_gap < 0.02
    assert std_gap > err_t[gap].max()


def test():
    test_measurement_covariance_transform()
    test_wrist_measurements()
    test_fusion_bridges_gap()
    print('all passed')


if __name__ == '__main__':
    test()
