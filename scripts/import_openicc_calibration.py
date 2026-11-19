"""
Bring one camera's OpenImuCameraCalibrator result into the repo calibration
(example/tag_tracking by default):

  <calib_dir>/openicc/<name>/cam/cam_calib_*.json             intrinsics (copied)
  <calib_dir>/openicc/<name>/cam_imu/cam_imu_calib_result_*.json  camera-IMU, fitted trajectory removed
  <calib_dir>/gopro12_wide_fisheye_setting_720_<serial>.yaml   IMU noise setting (scripts/gen_orb_slam_setting.py)

Prints the quality numbers to check and the lines for the tag tracking config.

python scripts/import_openicc_calibration.py -i ~/openicc_calib/set2_head -n set2_head -s C3501326495617
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import json
import pathlib
import shutil
import subprocess
import click
import numpy as np


# %%
@click.command()
@click.option('-i', '--input', required=True, help='OpenICC dataset directory (with cam/ and cam_imu/ results)')
@click.option('-n', '--name', required=True, help='Name of the calibration in the repo, e.g. set2_head')
@click.option('-s', '--serial', required=True, help='Camera serial number')
@click.option('-c', '--calib_dir', default='example/tag_tracking', help='Repo calibration directory')
@click.option('-t', '--template', default='example/tag_tracking/gopro12_wide_fisheye_setting_720.yaml',
    help='IMU noise setting to start from')
def main(input, name, serial, calib_dir, template):
    source = pathlib.Path(os.path.expanduser(input)).absolute()
    calib_dir = pathlib.Path(calib_dir).absolute()
    cams = sorted(source.glob('cam/cam_calib_*.json'))
    cam_imus = sorted(source.glob('cam_imu/cam_imu_calib_result_*.json'))
    assert len(cams) == 1 and len(cam_imus) == 1, f"one result each expected in {source}: {cams}, {cam_imus}"
    target = calib_dir.joinpath('openicc', name)
    target.joinpath('cam').mkdir(parents=True, exist_ok=True)
    target.joinpath('cam_imu').mkdir(parents=True, exist_ok=True)
    cam_out = target.joinpath('cam', cams[0].name)
    shutil.copy2(cams[0], cam_out)
    cam_imu = json.load(cam_imus[0].open('r'))
    cam_imu.pop('trajectory', None)
    cam_imu_out = target.joinpath('cam_imu', cam_imus[0].name)
    json.dump(cam_imu, cam_imu_out.open('w'), indent=4)
    setting = calib_dir.joinpath(f'gopro12_wide_fisheye_setting_720_{serial}.yaml')
    subprocess.run([sys.executable, 'scripts/gen_orb_slam_setting.py', '-i', str(cam_out), '-ci', str(cam_imu_out),
        '-t', template, '-o', str(setting)], check=True, stdout=subprocess.DEVNULL)

    intr = json.load(cam_out.open('r'))
    t = cam_imu['t_i_c']
    t_mm = np.array([t['x'], t['y'], t['z']]) * 1000
    print(f"{name} ({serial})")
    print(f"  intrinsics: {intr['nr_calib_images']} views, reprojection error {intr['final_reproj_error']:.3f} px")
    print(f"  camera-IMU: reprojection error {cam_imu['final_reproj_error']:.3f} px, "
        f"translation {np.round(t_mm, 1).tolist()} mm")
    if np.linalg.norm(t_mm) < 5.0:
        print("  WARNING: camera-IMU translation near 0: not estimated (too little motion in the cam_imu video)")
    print(f"Saved {cam_out}\n      {cam_imu_out}\n      {setting}")
    rel = lambda p: os.path.relpath(p, calib_dir)
    print("Tag tracking config:")
    print(f"  cameras:\n    {serial}: {rel(cam_out)}")
    print(f"  gripper_imu (wrist camera) or head_imu (head camera):\n"
        f"    cam_imu_calibration: {rel(cam_imu_out)}\n    noise_setting: {rel(setting)}\n    imu_time_offset: gpmf")

# %%
if __name__ == "__main__":
    main()
