"""
python scripts_slam_pipeline/03_batch_slam.py -i data_workspace/fold_cloth_20231214/demos

Several cameras: --camera_setting SERIAL=setting.yaml per camera (others use
--setting); --camera_serial to process only these cameras (e.g. wrist cameras,
not a head camera). Long continuous takes: --max_lost_frames 0, since
gopro_slam counts lost frames cumulatively over the whole video and
terminates once the count reaches max_lost_frames.
Prints per video: frames, lost frames and whether SLAM terminated early.
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
from umi.common.exiftool_util import ExifTool
from umi.common.cv_util import draw_predefined_mask


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
@click.option('-s', '--setting', default='example/calibration/gopro10_maxlens_fisheye_setting_v1_720.yaml', help='ORB_SLAM3 setting yaml, mounted into docker')
@click.option('-mk', '--mask', default=None, help='SLAM mask png (255 = ignored), default predefined UMI gripper mask')
@click.option('-nm', '--no_mask', is_flag=True, default=False, help='No mask, e.g. for a camera without gripper in view')
@click.option('-n', '--num_workers', type=int, default=None)
@click.option('-cs', '--camera_setting', multiple=True, help='SERIAL=setting.yaml, setting of one camera (repeatable)')
@click.option('-cam', '--camera_serial', multiple=True, help='Only videos of these cameras (repeatable), default all')
@click.option('-ml', '--max_lost_frames', type=int, default=60, help='Terminate after this many lost frames in total, 0: never')
@click.option('-tm', '--timeout_multiple', type=float, default=16, help='timeout_multiple * duration = timeout')
@click.option('-np', '--no_docker_pull', is_flag=True, default=False, help="pull docker image from docker hub")
def main(input_dir, map_path, docker_image, setting, mask, no_mask, camera_setting, camera_serial,
        num_workers, max_lost_frames, timeout_multiple, no_docker_pull):
    input_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()
    input_video_dirs = [x.parent for x in input_dir.glob('demo*/raw_video.mp4')]
    input_video_dirs += [x.parent for x in input_dir.glob('map*/raw_video.mp4')]
    print(f'Found {len(input_video_dirs)} video dirs')

    # setting per video by camera serial
    serial_settings = dict()
    for item in camera_setting:
        serial, path = item.split('=', 1)
        serial_settings[serial] = pathlib.Path(os.path.expanduser(path)).absolute()
        assert serial_settings[serial].is_file(), serial_settings[serial]
    default_setting = pathlib.Path(os.path.expanduser(setting)).absolute()
    assert default_setting.is_file()
    video_settings = dict()
    with ExifTool() as et:
        for video_dir in list(input_video_dirs):
            serial = list(et.get_metadata(str(video_dir.joinpath('raw_video.mp4'))))[0]['QuickTime:CameraSerialNumber']
            if len(camera_serial) > 0 and serial not in camera_serial:
                print(f"{video_dir.name}: skipped, camera {serial} not selected")
                input_video_dirs.remove(video_dir)
                continue
            video_settings[video_dir.absolute()] = serial_settings.get(serial, default_setting)
            print(f"{video_dir.name}: setting {video_settings[video_dir.absolute()].name}")
    
    if map_path is None:
        map_path = input_dir.joinpath('mapping', 'map_atlas.osa')
    else:
        map_path = pathlib.Path(os.path.expanduser(map_path)).absolute()
    assert map_path.is_file()

    if mask is None:
        slam_mask = np.zeros((2028, 2704), dtype=np.uint8)
        slam_mask = draw_predefined_mask(
            slam_mask, color=255, mirror=True, gripper=False, finger=True)
    else:
        slam_mask = cv2.imread(os.path.expanduser(mask), cv2.IMREAD_GRAYSCALE)
        assert slam_mask is not None and slam_mask.shape == (2028, 2704)

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

    with tqdm(total=len(input_video_dirs)) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for video_dir in tqdm(input_video_dirs):
                video_dir = video_dir.absolute()
                if video_dir.joinpath('camera_trajectory.csv').is_file():
                    print(f"camera_trajectory.csv already exists, skipping {video_dir.name}")
                    continue
                
                # softlink won't work in bind volume
                mount_target = pathlib.Path('/data')
                csv_path = mount_target.joinpath('camera_trajectory.csv')
                video_path = mount_target.joinpath('raw_video.mp4')
                json_path = mount_target.joinpath('imu_data.json')
                mask_path = mount_target.joinpath('slam_mask.png')
                mask_write_path = video_dir.joinpath('slam_mask.png')
                
                # find video duration
                with av.open(str(video_dir.joinpath('raw_video.mp4').absolute())) as container:
                    video = container.streams.video[0]
                    duration_sec = float(video.duration * video.time_base)
                timeout = duration_sec * timeout_multiple
                
                if not no_mask:
                    cv2.imwrite(str(mask_write_path.absolute()), slam_mask)

                map_mount_source = map_path
                map_mount_target = pathlib.Path('/map').joinpath(map_mount_source.name)
                setting_mount_source = video_settings[video_dir]
                setting_mount_target = pathlib.Path('/setting').joinpath(setting_mount_source.name)

                # run SLAM
                cmd = [
                    'docker',
                    'run',
                    '--rm', # delete after finish
                    '--volume', str(video_dir) + ':' + '/data',
                    '--volume', str(map_mount_source.parent) + ':' + str(map_mount_target.parent),
                    '--volume', str(setting_mount_source.parent) + ':' + str(setting_mount_target.parent),
                    docker_image,
                    '/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam',
                    '--vocabulary', '/ORB_SLAM3/Vocabulary/ORBvoc.txt',
                    '--setting', str(setting_mount_target),
                    '--input_video', str(video_path),
                    '--input_imu_json', str(json_path),
                    '--output_trajectory_csv', str(csv_path),
                    '--load_map', str(map_mount_target),
                    '--max_lost_frames', str(max_lost_frames)
                ]
                if not no_mask:
                    cmd.extend(['--mask_img', str(mask_path)])

                stdout_path = video_dir.joinpath('slam_stdout.txt')
                stderr_path = video_dir.joinpath('slam_stderr.txt')

                if len(futures) >= num_workers:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))

                futures.add(executor.submit(runner,
                    cmd, str(video_dir), stdout_path, stderr_path, timeout))
                # print(' '.join(cmd))

            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    # per video: frames, lost frames, early termination (gopro_slam stdout)
    print("Done! Result:")
    for video_dir in input_video_dirs:
        csv_path = video_dir.joinpath('camera_trajectory.csv')
        stdout_path = video_dir.joinpath('slam_stdout.txt')
        terminated = stdout_path.is_file() and 'Terminating!' in stdout_path.read_text(errors='ignore')
        with av.open(str(video_dir.joinpath('raw_video.mp4'))) as container:
            n_video = container.streams.video[0].frames
        if not csv_path.is_file():
            print(f"  {video_dir.name}: FAILED, no camera_trajectory.csv (see {stdout_path.name})")
            continue
        lines = csv_path.read_text().strip().split('\n')
        header = lines[0].split(',')
        rows = [x.split(',') for x in lines[1:]]
        lost_idx = header.index('is_lost')
        n_lost = sum(r[lost_idx].strip().lower() in ('true', '1') for r in rows)
        print(f"  {video_dir.name}: {len(rows)} of {n_video} frames in trajectory, "
            f"{n_lost} lost ({n_lost / max(len(rows), 1) * 100:.1f}%)"
            + (", TERMINATED early (max_lost_frames)" if terminated else ''))

# %%
if __name__ == "__main__":
    main()
