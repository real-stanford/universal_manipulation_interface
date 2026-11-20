"""
Sort one camera's calibration videos into the folder layout of OpenImuCameraCalibrator
(run_gopro_calibration.py --path_calib_dataset <output>/<name>):

  <output>/<name>/cam/       slow board video     (intrinsics)
  <output>/<name>/imu_bias/  still video          (IMU bias)
  <output>/<name>/cam_imu/   faster board video   (camera-IMU)

Roles from each video's GPMF gyro: the still one has the camera still (< 2 deg/s)
for most samples; of the two others, the one with more rotation is cam_imu, or the
one given with --cam_imu (e.g. a new camera-IMU video next to an older board video).

The table also gives each video's angular acceleration (90th percentile): the
camera-IMU translation shows only while the rotation speeds up or slows down (the
IMU, ~2 cm from the camera, swings on an arc). In our HERO12 calibrations the
translation was estimated with >= ~4.5 rad/s^2 and stayed at ~0 with 3.2-3.9:
record the cam_imu video with sharp back-and-forth rotations about every axis.
All videos must come from one camera. Files are hard linked (copied if on
another file system). --dry_run only prints the table.

python scripts/prepare_openicc_dataset.py -o ~/openicc_calib -n head GX010157.MP4 GX010158.MP4 GX010159.MP4
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# %%
import pathlib
import shutil
from typing import Dict, List, Optional
import click
import numpy as np
from umi.common.gpmf_util import gyro_vectors
from umi.common.exiftool_util import ExifTool

STILL_RAD_S = np.radians(2.0)


def motion_stats(rates: np.ndarray, angular_acceleration: Optional[np.ndarray] = None) -> Dict[str, float]:
    deg = np.degrees(rates)
    out = {'median_deg_s': float(np.median(deg)), 'p90_deg_s': float(np.percentile(deg, 90)),
        'p99_deg_s': float(np.percentile(deg, 99)), 'still_share': float(np.mean(rates < STILL_RAD_S))}
    if angular_acceleration is not None and len(angular_acceleration):
        out['p90_ang_acc'] = float(np.percentile(angular_acceleration, 90))
    return out


def angular_acceleration(gyro: np.ndarray, duration: float, smooth: int = 9) -> np.ndarray:
    """
    Angular acceleration magnitude (rad/s^2) from gyro samples evenly spread over duration (s),
    smoothed over `smooth` samples (sensor noise).
    """
    if len(gyro) < 2 * smooth or duration <= 0:
        return np.zeros(0)
    kernel = np.ones(smooth) / smooth
    g = np.stack([np.convolve(gyro[:, k], kernel, mode='valid') for k in range(3)], axis=1)
    return np.linalg.norm(np.diff(g, axis=0), axis=1) * len(gyro) / duration


def assign_roles(stats: Dict[str, Dict[str, float]], cam_imu=None) -> Dict[str, str]:
    """
    Video -> 'imu_bias' (most still), 'cam' / 'cam_imu' (less / more rotation, by the 90th percentile;
    cam_imu: the video to use as cam_imu instead).
    """
    assert len(stats) == 3, f"three videos expected, got {len(stats)}"
    still = max(stats, key=lambda k: stats[k]['still_share'])
    assert stats[still]['still_share'] > 0.8, f"no still video (most still: {still}, {stats[still]['still_share']:.0%})"
    moving = sorted((k for k in stats if k != still), key=lambda k: stats[k]['p90_deg_s'])
    if cam_imu is not None:
        assert cam_imu in moving, f"--cam_imu {cam_imu} is not one of the moving videos {moving}"
        moving = [k for k in moving if k != cam_imu] + [cam_imu]
    return {still: 'imu_bias', moving[0]: 'cam', moving[1]: 'cam_imu'}


def place(src: pathlib.Path, dst_dir: pathlib.Path) -> pathlib.Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir.joinpath(src.name)
    if not dst.exists():
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    return dst


# %%
@click.command()
@click.argument('videos', nargs=-1, required=True)
@click.option('-o', '--output', required=True, help='OpenICC calibration root')
@click.option('-n', '--name', required=True, help='Dataset (camera) name, e.g. head / left / right')
@click.option('--cam_imu', default=None, help='The video to use as cam_imu (default: the one with more rotation)')
@click.option('--dry_run', is_flag=True, default=False, help='Only print the table')
def main(videos, output, name, cam_imu, dry_run):
    paths = [pathlib.Path(os.path.expanduser(v)).absolute() for v in videos]
    with ExifTool() as et:
        meta = {p: list(et.get_metadata(str(p)))[0] for p in paths}
    serials = {meta[p].get('QuickTime:CameraSerialNumber') for p in paths}
    assert len(serials) == 1, f"videos from several cameras: {serials}"
    stats = dict()
    for p in paths:
        gyro = gyro_vectors(str(p))
        stats[p] = motion_stats(np.linalg.norm(gyro, axis=1),
            angular_acceleration(gyro, float(meta[p].get('QuickTime:Duration') or 0)))
    forced = None if cam_imu is None else pathlib.Path(os.path.expanduser(cam_imu)).absolute()
    roles = assign_roles(stats, forced)
    print(f"camera {serials.pop()}")
    for p in paths:
        s = stats[p]
        print(f"  {p.name}: {meta[p].get('QuickTime:Duration')} s, gyro median {s['median_deg_s']:.1f} deg/s, "
            f"90% {s['p90_deg_s']:.1f}, 99% {s['p99_deg_s']:.1f}, angular acceleration 90% "
            f"{s.get('p90_ang_acc', float('nan')):.1f} rad/s^2, still {s['still_share']:.0%} -> {roles[p]}")
    if dry_run:
        return
    root = pathlib.Path(os.path.expanduser(output)).absolute().joinpath(name)
    for p in paths:
        print(f"  {place(p, root.joinpath(roles[p]))}")

# %%
if __name__ == "__main__":
    main()
