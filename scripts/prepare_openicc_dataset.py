"""
Sort one camera's calibration videos into the folder layout of OpenImuCameraCalibrator
(run_gopro_calibration.py --path_calib_dataset <output>/<name>):

  <output>/<name>/cam/       slow board video     (intrinsics)
  <output>/<name>/imu_bias/  still video          (IMU bias)
  <output>/<name>/cam_imu/   faster board video   (camera-IMU)

Roles from each video's GPMF gyro: the still one has the camera still (< 2 deg/s)
for most samples; of the two others, the one with more rotation is cam_imu.
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
from typing import Dict, List
import click
import numpy as np
from umi.common.gpmf_util import gyro_rates
from umi.common.exiftool_util import ExifTool

STILL_RAD_S = np.radians(2.0)


def motion_stats(rates: np.ndarray) -> Dict[str, float]:
    deg = np.degrees(rates)
    return {'median_deg_s': float(np.median(deg)), 'p90_deg_s': float(np.percentile(deg, 90)),
        'p99_deg_s': float(np.percentile(deg, 99)), 'still_share': float(np.mean(rates < STILL_RAD_S))}


def assign_roles(stats: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """
    Video -> 'imu_bias' (most still), 'cam' / 'cam_imu' (less / more rotation, by the 90th percentile).
    """
    assert len(stats) == 3, f"three videos expected, got {len(stats)}"
    still = max(stats, key=lambda k: stats[k]['still_share'])
    assert stats[still]['still_share'] > 0.8, f"no still video (most still: {still}, {stats[still]['still_share']:.0%})"
    moving = sorted((k for k in stats if k != still), key=lambda k: stats[k]['p90_deg_s'])
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
@click.option('--dry_run', is_flag=True, default=False, help='Only print the table')
def main(videos, output, name, dry_run):
    paths = [pathlib.Path(os.path.expanduser(v)).absolute() for v in videos]
    with ExifTool() as et:
        meta = {p: list(et.get_metadata(str(p)))[0] for p in paths}
    serials = {meta[p].get('QuickTime:CameraSerialNumber') for p in paths}
    assert len(serials) == 1, f"videos from several cameras: {serials}"
    stats = {p: motion_stats(gyro_rates(str(p))) for p in paths}
    roles = assign_roles(stats)
    print(f"camera {serials.pop()}")
    for p in paths:
        s = stats[p]
        print(f"  {p.name}: {meta[p].get('QuickTime:Duration')} s, gyro median {s['median_deg_s']:.1f} deg/s, "
            f"90% {s['p90_deg_s']:.1f}, 99% {s['p99_deg_s']:.1f}, still {s['still_share']:.0%} -> {roles[p]}")
    if dry_run:
        return
    root = pathlib.Path(os.path.expanduser(output)).absolute().joinpath(name)
    for p in paths:
        print(f"  {place(p, root.joinpath(roles[p]))}")

# %%
if __name__ == "__main__":
    main()
