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
Rotation far from the other HERO12 cameras (same reason): the mean of good ones, also on an imported calibration
python scripts/import_openicc_calibration.py -i example/tag_tracking/openicc/station4_head -n station4_head -s C3501326498323 \\
    --rotation_from station2_left,station2_right,station4_left,station4_right,station5_head,station5_left,station5_right
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


def mean_quaternion(qs: list) -> np.ndarray:
    """
    Mean of unit quaternions (w, x, y, z) close to each other: signs aligned to the first, averaged, normalized.
    """
    q0 = qs[0] / np.linalg.norm(qs[0])
    m = np.sum([q if np.dot(q, q0) >= 0 else -q for q in (q / np.linalg.norm(q) for q in qs)], axis=0)
    return m / np.linalg.norm(m)


def quaternion_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    d = abs(float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))))
    return float(np.degrees(2 * np.arccos(min(1.0, d))))


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
@click.option('--rotation_from', default=None,
    help='name,name,...: replace the estimated camera-IMU rotation (q_i_c) by the mean of these calibrations in '
    '<calib_dir>/openicc/ (good HERO12 ones), when this camera\'s is far from them; the estimate is kept as q_i_c_estimated')
def main(input, name, serial, calib_dir, template, translation_mm, rotation_from):
    source = pathlib.Path(os.path.expanduser(input)).absolute()
    calib_dir = pathlib.Path(calib_dir).absolute()
    cams = sorted(source.glob('cam/cam_calib_*.json'))
    cam_imus = sorted(source.glob('cam_imu/cam_imu_calib_result_*.json'))
    assert len(cams) == 1 and len(cam_imus) == 1, f"one result each expected in {source}: {cams}, {cam_imus}"
    target = calib_dir.joinpath('openicc', name)
    target.joinpath('cam').mkdir(parents=True, exist_ok=True)
    target.joinpath('cam_imu').mkdir(parents=True, exist_ok=True)
    cam_out = target.joinpath('cam', cams[0].name)
    if cams[0].resolve() != cam_out.resolve():     # -i may be the imported calibration itself (re-import)
        shutil.copy2(cams[0], cam_out)
    cam_imu = json.load(cam_imus[0].open('r'))
    cam_imu.pop('trajectory', None)
    if translation_mm is not None:
        x, y, z = (float(v) / 1000 for v in translation_mm.split(','))
        cam_imu.setdefault('t_i_c_estimated', cam_imu['t_i_c'])     # a re-import keeps the original estimate
        cam_imu['t_i_c'] = {'x': x, 'y': y, 'z': z}
        cam_imu['t_i_c_source'] = f'set by import_openicc_calibration.py --translation_mm {translation_mm} (not this camera\'s estimate)'
    if rotation_from is not None:
        names = [n.strip() for n in rotation_from.split(',') if n.strip()]
        qs = list()
        for n in names:
            f = sorted(calib_dir.joinpath('openicc', n).glob('cam_imu/cam_imu_calib_result_*.json'))
            assert len(f) == 1, f"--rotation_from {n}: one cam_imu result expected, found {f}"
            q = json.load(f[0].open('r'))['q_i_c']
            qs.append(np.array([q['w'], q['x'], q['y'], q['z']]))
        mean = mean_quaternion(qs)
        cam_imu.setdefault('q_i_c_estimated', cam_imu['q_i_c'])
        est = cam_imu['q_i_c_estimated']
        print(f"  rotation: estimate {quaternion_angle_deg(np.array([est['w'], est['x'], est['y'], est['z']]), mean):.1f} deg "
            f"from the mean of {len(qs)}; they spread up to {max(quaternion_angle_deg(q, mean) for q in qs):.1f} deg from it")
        cam_imu['q_i_c'] = dict(zip('wxyz', mean.tolist()))
        cam_imu['q_i_c_source'] = f'set by import_openicc_calibration.py --rotation_from {",".join(names)} ' \
            "(their mean, not this camera's estimate)"
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
