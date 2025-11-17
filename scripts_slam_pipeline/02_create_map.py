"""
python scripts_slam_pipeline/00_process_videos.py -i data_workspace/toss_objects/20231113/mapping
"""

# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import subprocess
import multiprocessing
import concurrent.futures
from tqdm import tqdm
import numpy as np
import cv2
# from umi.common.cv_util import draw_predefined_mask
from umi.common.cv_util_realsense import (
    draw_rgb_predefined_mask,
    RGB_IMG_SHAPE
)

# %%
@click.command()
@click.option('-i', '--input_dir', required=True, help='Directory for mapping video')
@click.option('-m', '--map_path', default=None, help='ORB_SLAM3 *.osa map atlas file')
@click.option('-d', '--docker_image', default="chicheng/orb_slam3:latest")
@click.option('-np', '--no_docker_pull', is_flag=True, default=False, help="pull docker image from docker hub")
@click.option('-nm', '--no_mask', is_flag=True, default=False, help="Whether to mask out gripper and mirrors. Set if map is created with bare GoPro no on gripper.")
def main(input_dir, map_path, docker_image, no_docker_pull, no_mask):
    video_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()
    for fn in ['raw_video.mp4', 'imu_data.json']:
        assert video_dir.joinpath(fn).is_file()

    if map_path is None:
        map_path = video_dir.joinpath('map_atlas.osa')
    else:
        map_path = pathlib.Path(os.path.expanduser(map_path)).absolute()
    map_path.parent.mkdir(parents=True, exist_ok=True)

    csv_path = video_dir.joinpath('mapping_camera_trajectory.csv')
    video_path = video_dir.joinpath('raw_video.mp4')
    json_path = video_dir.joinpath('imu_data.json')
    mask_path = video_dir.joinpath('slam_mask.png')
    
    if not no_mask:
        mask_write_path = video_dir.joinpath('slam_mask.png')
        slam_mask = np.zeros(RGB_IMG_SHAPE, dtype=np.uint8)
        slam_mask = draw_rgb_predefined_mask(slam_mask, color=255)
        cv2.imwrite(str(mask_write_path.absolute()), slam_mask)

    map_mount_source = pathlib.Path(map_path)

    # print(f"[INFO]: path\n{video_path=}\n{json_path=}\n{csv_path=}\n{map_mount_source=}")

    ORB_SLAM3_ROOT = pathlib.Path("~/Desktop/study/ORB_SLAM3").expanduser()
    binary_path = ORB_SLAM3_ROOT.joinpath("Examples/RGB-D-Inertial/realsense_slam")
    setting_path = ORB_SLAM3_ROOT.joinpath("Examples/RGB-D-Inertial/RealSense_D435i.yaml")
    voca_path = ORB_SLAM3_ROOT.joinpath("Vocabulary/ORBvoc.txt")

    cmd = [
        str(binary_path),
        "--setting", str(setting_path), 
        "--vocabulary", str(voca_path),
        "--input_data_dir", str(video_dir),
        '--output_trajectory_csv', str(csv_path),
        '--save_map', str(map_mount_source),
    ]

    if not no_mask:
        cmd.extend([
            '--mask_img', str(mask_path)
        ])

    stdout_path = video_dir.joinpath('slam_stdout.txt')
    stderr_path = video_dir.joinpath('slam_stderr.txt')

    result = subprocess.run(
        cmd,
        cwd=str(video_dir),
        stdout=stdout_path.open('w'),
        stderr=stderr_path.open('w')
    )
    print(f"[INFO] create map {result=}")


# %%
if __name__ == "__main__":
    main()
