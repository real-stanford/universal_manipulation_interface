"""
python scripts_slam_pipeline/03_batch_slam.py -i data_workspace/fold_cloth_20231214/demos
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
import cv2
import av
import numpy as np
# from umi.common.cv_util import draw_predefined_mask
from umi.common.cv_util_realsense import (
    draw_rgb_predefined_mask,
    draw_im_l_infrared_mask,
    draw_im_r_infrared_mask,
    RGB_IMG_SHAPE,
    IR_IMG_SHAPE
)


# %%
def runner(cmd, cwd, stdout_path, stderr_path, timeout, **kwargs):
    try:
        return subprocess.run(cmd,                       
            cwd=str(cwd),
            stdout=stdout_path.open('w'),
            stderr=stderr_path.open('w'),
            timeout=timeout,
            **kwargs)
    except subprocess.TimeoutExpired as e:
        return e


# %%
@click.command()
@click.option('-i', '--input_dir', required=True, help='Directory for demos folder')
@click.option('-m', '--map_path', default=None, help='ORB_SLAM3 *.osa map atlas file')
@click.option('-d', '--docker_image', default="chicheng/orb_slam3:latest")
@click.option('-n', '--num_workers', type=int, default=None)
@click.option('-ml', '--max_lost_frames', type=int, default=60)
@click.option('-tm', '--timeout_multiple', type=float, default=16, help='timeout_multiple * duration = timeout')
@click.option('-np', '--no_docker_pull', is_flag=True, default=False, help="pull docker image from docker hub")
@click.option('-nm', '--no_mask', is_flag=True, default=False, help="Whether to mask out gripper and mirrors. Set if map is created with bare GoPro no on gripper.")
def main(input_dir, 
    map_path, 
    docker_image, 
    num_workers, 
    max_lost_frames, 
    timeout_multiple, 
    no_docker_pull,
    no_mask,
):
    input_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()
    input_bag_dirs = [x.parent for x in input_dir.glob('demo*/raw_video.mp4')]
    input_bag_dirs += [x.parent for x in input_dir.glob('map*/raw_video.mp4')]
    print(f'Found {len(input_bag_dirs)} video dirs')
    
    if map_path is None:
        map_path = input_dir.joinpath('mapping', 'map_atlas.osa')
    else:
        map_path = pathlib.Path(os.path.expanduser(map_path)).absolute()
    assert map_path.is_file()

    if num_workers is None:
        num_workers = multiprocessing.cpu_count() // 2

    # pull docker
    if not no_docker_pull:
        print(f"Pulling docker image {docker_image}")
        cmd = [
            'docker',
            'pull',
            docker_image
        ]
        p = subprocess.run(cmd)
        if p.returncode != 0:
            print("Docker pull failed!")
            exit(1)

    with tqdm(total=len(input_bag_dirs)) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for bag_dir in tqdm(input_bag_dirs):
                bag_dir = bag_dir.absolute()
                print(f"[INFO] bag_dir={str(bag_dir)}")
                # if bag_dir.joinpath('camera_trajectory.csv').is_file():
                #     print(f"camera_trajectory.csv already exists, skipping {bag_dir.name}")
                #     continue
                
                # softlink won't work in bind volume
                # mount_target = pathlib.Path('/data')
                mount_target = bag_dir
                csv_path = mount_target.joinpath('camera_trajectory.csv')
                video_path = mount_target.joinpath('raw_video.mp4')
                json_path = mount_target.joinpath('imu_data.json')
                bag_path = bag_dir.joinpath("raw_bag.bag")

                # NOTE
                mask_path = mount_target.joinpath('slam_mask.png')
                mask_write_path = bag_dir.joinpath('slam_mask.png')
                
                # find video duration
                with av.open(str(bag_dir.joinpath('raw_video.mp4').absolute())) as container:
                    video = container.streams.video[0]
                    duration_sec = float(video.duration * video.time_base)
                timeout = duration_sec * timeout_multiple
                
                # slam_mask = np.zeros(RGB_IMG_SHAPE, dtype=np.uint8)
                # slam_mask = draw_rgb_predefined_mask(slam_mask, color=255)
                # cv2.imwrite(str(mask_write_path.absolute()), slam_mask)

                if not no_mask:
                    # left, right
                    ir_l_slam_mask_path = bag_dir.joinpath('ir_l_slam_mask.png')
                    ir_r_slam_mask_path = bag_dir.joinpath('ir_r_slam_mask.png')

                    slam_mask = np.zeros(IR_IMG_SHAPE, dtype=np.uint8)
                    slam_mask = draw_im_l_infrared_mask(slam_mask, color=255)
                    cv2.imwrite(str(ir_l_slam_mask_path.absolute()), slam_mask)

                    slam_mask = np.zeros(IR_IMG_SHAPE, dtype=np.uint8)
                    slam_mask = draw_im_r_infrared_mask(slam_mask, color=255)
                    cv2.imwrite(str(ir_r_slam_mask_path.absolute()), slam_mask)

                map_mount_source = map_path
                map_mount_target = pathlib.Path('/map').joinpath(map_mount_source.name)

                ORB_SLAM3_ROOT = pathlib.Path("~/Desktop/study/ORB_SLAM3").expanduser()
                binary_path = ORB_SLAM3_ROOT.joinpath("Examples/Stereo-Inertial/stereo_inertial_realsense_D435i")
                setting_path = ORB_SLAM3_ROOT.joinpath("Examples/Stereo-Inertial/RealSense_D435i.yaml")
                voca_path = ORB_SLAM3_ROOT.joinpath("Vocabulary/ORBvoc.txt")

                
                # run SLAM
                cmd = [
                    str(binary_path),
                    "--setting", str(setting_path), 
                    "--vocabulary", str(voca_path),
                    "--bag_path", str(bag_path),
                    '--output_trajectory_csv', str(csv_path),
                    '--load_map', str(map_mount_source),
                    # '--max_lost_frames', str(max_lost_frames)
                ]

                if not no_mask:
                    cmd.extend([
                        # '--mask_img', str(mask_path),
                        "--ir_l_mask", str(ir_l_slam_mask_path),
                        "--ir_r_mask", str(ir_r_slam_mask_path),
                    ])

                stdout_path = bag_dir.joinpath('slam_stdout.txt')
                stderr_path = bag_dir.joinpath('slam_stderr.txt')

                if len(futures) >= num_workers:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))

                futures.add(executor.submit(runner,
                    cmd, str(bag_dir), stdout_path, stderr_path, timeout))
                # print(' '.join(cmd))

            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    print("Done! Result:")
    print([x.result() for x in completed])

# %%
if __name__ == "__main__":
    main()
