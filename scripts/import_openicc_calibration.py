"""
Bring one camera's OpenImuCameraCalibrator result into the repo calibration
(example/tag_tracking by default):

  <calib_dir>/openicc/<name>/cam/cam_calib_*.json             intrinsics (copied)
  <calib_dir>/openicc/<name>/cam_imu/cam_imu_calib_result_*.json  camera-IMU, fitted trajectory removed
  <calib_dir>/gopro12_wide_fisheye_setting_720_<serial>.yaml   IMU noise setting (scripts/gen_orb_slam_setting.py)

Prints the quality numbers to check and the lines for the tag tracking config.

python scripts/import_openicc_calibration.py -i ~/openicc_calib/set2_head -n set2_head -s C3501326495617
Translation not estimated (too little rotation in the cam_imu video): HERO12 mean of the good calibrations
python scripts/import_openicc_calibration.py -i ~/openicc_calib/set2_left -n set2_left -s C3501326501951 --translation_mm=-0.2,-22.2,4.4
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
@click.option('--translation_mm', default=None,
    help='x,y,z: replace the estimated camera-IMU translation (t_i_c, mm), e.g. the HERO12 mean '
    '-0.2,-22.2,4.4 when the cam_imu video had too little rotation; the estimate is kept as t_i_c_estimated')
def main(input, name, serial, calib_dir, template, translation_mm):
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
    if translation_mm is not None:
        x, y, z = (float(v) / 1000 for v in translation_mm.split(','))
        cam_imu['t_i_c_estimated'] = cam_imu['t_i_c']
        cam_imu['t_i_c'] = {'x': x, 'y': y, 'z': z}
        cam_imu['t_i_c_source'] = f'set by import_openicc_calibration.py --translation_mm {translation_mm} (not this camera\'s estimate)'
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
    if translation_mm is not None:
        e = cam_imu['t_i_c_estimated']
        print(f"  translation set by --translation_mm (estimated {np.round(np.array([e['x'], e['y'], e['z']]) * 1000, 1).tolist()} mm)")
    elif np.linalg.norm(t_mm) < 5.0:
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
