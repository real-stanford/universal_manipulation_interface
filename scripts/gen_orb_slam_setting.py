"""
Generate an ORB_SLAM3 monocular-inertial setting yaml from
OpenImuCameraCalibrator (OpenICC) results, using an existing setting
yaml as template for all other parameters (IMU noise, ORB extractor).

python scripts/gen_orb_slam_setting.py \
-i openicc_calib/gripper1/cam/cam_calib_GX019254_fi_1.0.json \
-ci openicc_calib/gripper1/cam_imu/cam_imu_calib_result_GX019255.json \
-o umi_calib/gopro12_wide_fisheye_setting_720.yaml
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import re
import json
import pathlib
import click
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    convert_fisheye_intrinsics_resolution
)

# OpenICC IMU axes to ORB_SLAM3 body axes,
# same permutation as used for the template IMU.T_b_c1
OPENICC_TO_ORB_SLAM_BODY = np.array([
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=np.float64)


# %%
def get_tx_imu_cam(cam_imu_result):
    """
    Camera to IMU transform from OpenICC cam_imu_calib_result json.
    """
    q = cam_imu_result['q_i_c']
    t = cam_imu_result['t_i_c']
    tx = np.eye(4)
    tx[:3, :3] = Rotation.from_quat([q['x'], q['y'], q['z'], q['w']]).as_matrix()
    tx[:3, 3] = [t['x'], t['y'], t['z']]
    return tx

def replace_scalar(text, key, value):
    pattern = re.compile(r'^(' + re.escape(key) + r':[ \t]*)[^#\n]*?([ \t]*(#.*)?)$', re.MULTILINE)
    text, n = pattern.subn(lambda m: m.group(1) + str(value) + m.group(2), text)
    assert n == 1, f"{key} not found in template"
    return text

def replace_matrix_data(text, key, mat):
    pattern = re.compile(r'(^' + re.escape(key) + r':.*?^\s*data:\s*)\[[^\]]*\]', re.MULTILINE | re.DOTALL)
    data = ', '.join(repr(float(x)) for x in mat.flatten())
    text, n = pattern.subn(lambda m: m.group(1) + '[ ' + data + ' ]', text)
    assert n == 1, f"{key} not found in template"
    return text


# %%
@click.command()
@click.option('-i', '--intrinsics_json', required=True, help='OpenICC camera calibration json (FISHEYE)')
@click.option('-ci', '--cam_imu_json', required=True, help='OpenICC cam_imu_calib_result json')
@click.option('-t', '--template', default='example/calibration/gopro10_maxlens_fisheye_setting_v1_720.yaml',
    help='ORB_SLAM3 setting yaml providing all other parameters')
@click.option('-o', '--output', required=True, help='Output setting yaml')
@click.option('-r', '--resolution', type=str, default='960,720', help='SLAM input resolution, must match template')
def main(intrinsics_json, cam_imu_json, template, output, resolution):
    resolution = tuple(int(x) for x in resolution.split(','))
    template = pathlib.Path(os.path.expanduser(template)).absolute()
    output = pathlib.Path(os.path.expanduser(output)).absolute()

    # camera intrinsics at SLAM resolution
    intr = parse_fisheye_intrinsics(json.load(open(os.path.expanduser(intrinsics_json), 'r')))
    intr = convert_fisheye_intrinsics_resolution(intr, target_resolution=resolution)
    K = intr['K']
    D = intr['D'].flatten()

    # camera to body transform
    cam_imu_result = json.load(open(os.path.expanduser(cam_imu_json), 'r'))
    tx_body_cam = OPENICC_TO_ORB_SLAM_BODY @ get_tx_imu_cam(cam_imu_result)

    text = template.read_text()
    values = {
        'Camera1.fx': K[0, 0],
        'Camera1.fy': K[1, 1],
        'Camera1.cx': K[0, 2],
        'Camera1.cy': K[1, 2],
        'Camera1.k1': D[0],
        'Camera1.k2': D[1],
        'Camera1.k3': D[2],
        'Camera1.k4': D[3]
    }
    for key, value in values.items():
        text = replace_scalar(text, key, repr(float(value)))
    text = replace_matrix_data(text, 'IMU.T_b_c1', tx_body_cam)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)

    # verify the written file parses to the intended values
    fs = cv2.FileStorage(str(output), cv2.FILE_STORAGE_READ)
    assert (fs.getNode('Camera.width').real(), fs.getNode('Camera.height').real()) == resolution, \
        "resolution does not match template Camera.width/height"
    for key, value in values.items():
        assert np.isclose(fs.getNode(key).real(), value)
    assert np.allclose(fs.getNode('IMU.T_b_c1').mat(), tx_body_cam, atol=1e-6)
    fs.release()

    print(f"Camera1 fx {K[0,0]:.3f} fy {K[1,1]:.3f} cx {K[0,2]:.3f} cy {K[1,2]:.3f}, k {D}")
    print(f"IMU.T_b_c1\n{np.round(tx_body_cam, 6)}")
    print(f"IMU to camera time offset {cam_imu_result['time_offset_imu_to_cam_s'] * 1000:.1f}ms "
        "is not used by ORB_SLAM3 settings, same as the template")
    print(f"Saved to {output}")

# %%
if __name__ == "__main__":
    main()
