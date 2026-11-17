"""
GoPro IMU data and calibration, in the raw GPMF axis convention used by
UMI's ORB_SLAM3 (gopro_slam reads ACCL / GYRO values as stored).
"""
from typing import Dict
import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

# OpenImuCameraCalibrator IMU axes to raw GPMF axes, same permutation as
# scripts/gen_orb_slam_setting.py (OpenICC reads accl as [v1, v2, v0])
OPENICC_TO_GPMF = np.array([
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=np.float64)


def load_imu(imu_json: str) -> Dict[str, np.ndarray]:
    """
    imu_data.json from scripts_slam_pipeline/01_extract_gopro_imu.py.
    Returns times (sec, IMU clock), accel (m/s^2), gyro (rad/s), raw GPMF axes.
    """
    streams = json.load(open(imu_json, 'r'))['1']['streams']
    accl = streams['ACCL']['samples']
    gyro = streams['GYRO']['samples']
    n = min(len(accl), len(gyro))
    return {
        'time': np.array([x['cts'] for x in accl[:n]]) / 1000.0,
        'accel': np.array([x['value'] for x in accl[:n]], dtype=np.float64),
        'gyro': np.array([x['value'] for x in gyro[:n]], dtype=np.float64)
    }


def load_cam_imu_calibration(cam_imu_json: str) -> Dict:
    """
    OpenICC cam_imu_calib_result json.
    Returns T_imu_cam in raw GPMF IMU axes, and the time offset:
    camera time = IMU time + time_offset (OpenICC imu_camera_calibrator.cc).
    """
    data = json.load(open(cam_imu_json, 'r'))
    q = data['q_i_c']
    t = data['t_i_c']
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q['x'], q['y'], q['z'], q['w']]).as_matrix()
    T[:3, 3] = [t['x'], t['y'], t['z']]
    return {
        'T_imu_cam': OPENICC_TO_GPMF @ T,
        'time_offset': float(data['time_offset_imu_to_cam_s'])
    }


def load_imu_noise(orb_slam_setting_yaml: str) -> Dict[str, float]:
    """
    Continuous-time IMU noise densities and bias random walks from an
    ORB_SLAM3 setting yaml (IMU.NoiseGyro, IMU.NoiseAcc, IMU.GyroWalk, IMU.AccWalk).
    """
    fs = cv2.FileStorage(orb_slam_setting_yaml, cv2.FILE_STORAGE_READ)
    out = {k: fs.getNode('IMU.' + k).real() for k in ['NoiseGyro', 'NoiseAcc', 'GyroWalk', 'AccWalk']}
    fs.release()
    assert all(v > 0 for v in out.values()), f"IMU noise missing in {orb_slam_setting_yaml}"
    return out
