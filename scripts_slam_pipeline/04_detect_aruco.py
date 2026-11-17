"""
Detect ArUco tags (corners and single-tag pose) in every <demos>/*/raw_video.mp4,
writes tag_detection.pkl next to each video (skipped if it exists).

python scripts_slam_pipeline/04_detect_aruco.py \
-i data_workspace/cup_in_the_wild/20240105_zhenjia_packard_2nd_conference_room/demos \
-ci data_workspace/toss_objects/20231113/calibration/gopro_intrinsics_2_7k.json \
-ac data_workspace/toss_objects/20231113/calibration/aruco_config.yaml

Videos are processed one at a time, each in parallel time chunks on all cores.
With --tag_tracking_config, each camera uses its intrinsics from the config's
cameras section (camera_intrinsics for cameras not listed). With --wrist_only
as well, only wrist (gripper) camera videos are processed, without the SLAM
mapping video: the tag tracking pipeline detects head camera tags itself
(scripts_tag_tracking/00_detect_tag_corners.py, tag_corners.pkl).
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
import yaml
from umi.common.exiftool_util import ExifTool
import multiprocessing
import subprocess

# %%
@click.command()
@click.option('-i', '--input_dir', required=True, help='Directory for demos folder')
@click.option('-ci', '--camera_intrinsics', required=True, help='Camera intrinsics json file (2.7k)')
@click.option('-ac', '--aruco_yaml', required=True, help='Aruco config yaml file')
@click.option('-tc', '--tag_tracking_config', default=None,
    help='Optional tag tracking config: per camera serial intrinsics from its cameras section, '
    'camera_intrinsics for cameras not listed')
@click.option('-wo', '--wrist_only', is_flag=True, default=False,
    help='With --tag_tracking_config: only videos of gripper cameras, not the mapping video')
@click.option('-n', '--num_workers', type=int, default=None, help='Worker processes per video, default all cores')
def main(input_dir, camera_intrinsics, aruco_yaml, tag_tracking_config, wrist_only, num_workers):
    input_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()
    video_dirs = sorted(x.parent for x in input_dir.glob('*/raw_video.mp4'))
    print(f'Found {len(video_dirs)} video dirs')

    assert os.path.isfile(camera_intrinsics)
    assert os.path.isfile(aruco_yaml)
    assert (not wrist_only) or (tag_tracking_config is not None), "--wrist_only needs --tag_tracking_config"

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    script_path = pathlib.Path(__file__).parent.parent.joinpath('scripts', 'detect_aruco.py')

    # per camera serial intrinsics and wrist cameras from the tag tracking config
    serial_intrinsics = dict()
    wrist_serials = set()
    if tag_tracking_config is not None:
        config_path = pathlib.Path(os.path.expanduser(tag_tracking_config)).absolute()
        config = yaml.safe_load(config_path.open('r'))
        serial_intrinsics = {str(k): str(config_path.parent.joinpath(v)) for k, v in config['cameras'].items()}
        for path in serial_intrinsics.values():
            assert os.path.isfile(path), path
        wrist_serials = {str(b['gripper_camera']) for b in config['bodies'].values() if 'gripper_camera' in b}

    # videos to process and their intrinsics
    video_intrinsics = dict()
    with ExifTool() as et:
        for video_dir in video_dirs:
            serial = list(et.get_metadata(str(video_dir.joinpath('raw_video.mp4'))))[0]['QuickTime:CameraSerialNumber']
            if wrist_only and (serial not in wrist_serials or video_dir.name == 'mapping'):
                print(f"{video_dir.name}: skipped (--wrist_only)")
                continue
            video_intrinsics[video_dir] = serial_intrinsics.get(serial, camera_intrinsics)
            print(f"{video_dir.name}: intrinsics {os.path.basename(video_intrinsics[video_dir])}")

    todo = list()
    for video_dir in video_intrinsics:
        if video_dir.joinpath('tag_detection.pkl').is_file():
            print(f"tag_detection.pkl already exists, skipping {video_dir.name}")
        else:
            todo.append(video_dir)

    # one video at a time, each split into time chunks on all workers (scripts/detect_aruco.py)
    results = dict()
    for k, video_dir in enumerate(todo):
        print(f"[{k + 1}/{len(todo)}] {video_dir.name}")
        cmd = [
            'python', str(script_path),
            '--input', str(video_dir.joinpath('raw_video.mp4')),
            '--output', str(video_dir.joinpath('tag_detection.pkl')),
            '--intrinsics_json', video_intrinsics[video_dir],
            '--aruco_yaml', aruco_yaml,
            '--num_workers', str(num_workers)
        ]
        results[video_dir] = subprocess.run(cmd)

    # every selected video must have its detections
    failed = [x for x in video_intrinsics if not x.joinpath('tag_detection.pkl').is_file()]
    for video_dir in failed:
        result = results.get(video_dir)
        print(f"FAILED {video_dir.name}" + (f" (exit code {result.returncode})" if result is not None else ''))
    assert len(failed) == 0, f"tag detection failed for {len(failed)} videos"
    print(f"Done, {len(video_intrinsics)} videos have tag_detection.pkl")

# %%
if __name__ == "__main__":
    main()
