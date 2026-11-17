"""
Tag tracking (GTSAM) pipeline: gripper poses from tag cubes observed by a
head camera, instead of wrist camera SLAM. Counterpart of run_slam_pipeline_synced.py.

Session layout:
<session>/raw_videos/*.MP4                     head and wrist camera task videos
<session>/raw_videos/gripper_calibration/*.MP4 one open/close video per gripper camera
<session>/raw_videos/head_mapping/*.MP4        head camera walk past all environment tags
<session>/raw_videos/mapping.mp4               optional, not used by tag tracking

Steps (name: script -> output), each reading the outputs of the previous ones:
sort_videos:       scripts_slam_pipeline/00_process_videos.py --strict  -> demos/*/raw_video.mp4
imu:               scripts_slam_pipeline/01_extract_gopro_imu.py        -> demos/*/imu_data.json
wrist_tags:        scripts_slam_pipeline/04_detect_aruco.py --wrist_only -> wrist demos/*/tag_detection.pkl
gripper_range:     scripts_slam_pipeline/05_run_calibrations.py --skip_slam_tag -> demos/gripper_calibration*/gripper_range.json
audio_sync:        scripts_slam_pipeline/06a_estimate_audio_sync.py     -> audio_sync_offsets.json
head_tags:         scripts_tag_tracking/00_detect_tag_corners.py        -> head demos/*/tag_corners.pkl
segment_episodes:  scripts_tag_tracking/00b_segment_episodes.py         -> tag_tracking/episodes.json (if episodes configured)
use_setup:         scripts_tag_tracking/00c_use_setup.py                -> tag_tracking/bodies.json, gripper_tcp.json (if setup_calibration configured)
calibrate_bodies:  scripts_tag_tracking/02_calibrate_bodies.py          -> tag_tracking/bodies.json (without setup_calibration)
track_bodies:      scripts_tag_tracking/03_track_bodies.py              -> head demos/*/tag_tracking.pkl (episodes +- 0.5 s)
fuse_head_imu:     scripts_tag_tracking/03b_fuse_head_imu.py            -> tag_tracking.pkl refined per episode (if head_imu configured)
fuse_grippers:     scripts_tag_tracking/03c_fuse_gripper_imu.py         -> tag_tracking.pkl gripper poses: wrist IMU + wrist env tags + head (if gripper_imu configured)
gripper_tcp:       scripts_tag_tracking/05_calibrate_gripper_tcp.py     -> tag_tracking/gripper_tcp.json (without setup_calibration)
render:            scripts_tag_tracking/04_render_tracking.py           -> head demos/*/tag_tracking_render.mp4
export_plan:       scripts_tag_tracking/06_export_dataset_plan.py       -> dataset_plan.pkl

Each step's output is shown and written to <session>/logs/<nn>_<name>.log.
sort_videos, imu, wrist_tags and head_tags skip videos whose output exists;
the other steps always rerun. --start_from / --stop_after select a range of steps.

Then, as for the SLAM pipeline:
python scripts_slam_pipeline/07_generate_replay_buffer.py --no_mirror -o <dataset.zarr.zip> <session> ...

python run_tag_tracking_pipeline.py -c umi_calib/tag_tracking.yaml <session_dir>
python run_tag_tracking_pipeline.py -c umi_calib/tag_tracking.yaml --stop_after calibrate_bodies <session_dir>
python run_tag_tracking_pipeline.py -c umi_calib/tag_tracking.yaml -j 16 <session_dir>
"""

import sys
import os

ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import subprocess
import yaml

STEP_NAMES = ['sort_videos', 'imu', 'wrist_tags', 'gripper_range', 'audio_sync', 'head_tags',
    'segment_episodes', 'use_setup', 'calibrate_bodies', 'track_bodies', 'fuse_head_imu', 'gripper_tcp', 'fuse_grippers',
    'render', 'export_plan']


# %%
def run(script_path, args, log_path):
    """
    Run a pipeline script, showing its output and writing it to log_path.
    """
    print(f"############# {script_path.parent.name}/{script_path.name} -> {log_path} ###########")
    assert script_path.is_file(), script_path
    cmd = ['python', str(script_path)] + [str(x) for x in args]
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    with log_path.open('wb') as log:
        log.write((' '.join(cmd) + '\n').encode())
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        # raw chunks, so progress bars (carriage returns) show live
        while True:
            chunk = os.read(process.stdout.fileno(), 4096)
            if len(chunk) == 0:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
            log.write(chunk)
            log.flush()
        returncode = process.wait()
    assert returncode == 0, f"{script_path.name} failed, see {log_path}"


# %%
@click.command()
@click.argument('session_dir', nargs=-1)
@click.option('-c', '--config', required=True, help='Tag tracking config yaml, see umi/tag_tracking/config.py')
@click.option('-ci', '--camera_intrinsics', default=None,
    help='Fallback intrinsics json for 04_detect_aruco.py (cameras not in the config), default gopro_intrinsics_2_7k.json next to the config')
@click.option('-sf', '--start_from', type=click.Choice(STEP_NAMES), default=STEP_NAMES[0], help='First step to run')
@click.option('-sa', '--stop_after', type=click.Choice(STEP_NAMES), default=STEP_NAMES[-1], help='Last step to run')
@click.option('-nr', '--no_render', is_flag=True, default=False, help='Skip the render step')
@click.option('-j', '--num_workers', type=int, default=None,
    help='Parallel processes for imu, wrist_tags, audio_sync, head_tags, track_bodies and fuse_head_imu, default all cores')
def main(session_dir, config, camera_intrinsics, start_from, stop_after, no_render, num_workers):
    root_dir = pathlib.Path(__file__).parent
    slam_dir = root_dir.joinpath('scripts_slam_pipeline')
    tag_dir = root_dir.joinpath('scripts_tag_tracking')
    first, last = STEP_NAMES.index(start_from), STEP_NAMES.index(stop_after)
    assert first <= last, "--start_from must not come after --stop_after"

    config_path = pathlib.Path(os.path.expanduser(config)).absolute()
    config_dir = config_path.parent
    config_yaml = yaml.safe_load(config_path.open('r'))
    aruco_config = config_dir.joinpath(config_yaml['aruco_config'])
    assert 'gripper_config' in config_yaml, "tag tracking config needs gripper_config"
    gripper_config = config_dir.joinpath(config_yaml['gripper_config'])
    if camera_intrinsics is None:
        camera_intrinsics = config_dir.joinpath('gopro_intrinsics_2_7k.json')
    camera_intrinsics = pathlib.Path(os.path.expanduser(camera_intrinsics)).absolute()
    for path in [aruco_config, gripper_config, camera_intrinsics]:
        assert path.is_file(), path

    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session)).absolute()
        print(f"============= {session} =============")
        demos_dir = session.joinpath('demos')
        log_dir = session.joinpath('logs')
        log_dir.mkdir(exist_ok=True)
        tag_args = ['-i', session, '-c', config_path]
        workers = [] if num_workers is None else ['--num_workers', num_workers]

        steps = {
            'sort_videos': (slam_dir.joinpath('00_process_videos.py'), [session, '--strict']),
            'imu': (slam_dir.joinpath('01_extract_gopro_imu.py'), [session] + workers),
            'wrist_tags': (slam_dir.joinpath('04_detect_aruco.py'), ['--input_dir', demos_dir,
                '--camera_intrinsics', camera_intrinsics, '--aruco_yaml', aruco_config,
                '--tag_tracking_config', config_path, '--wrist_only'] + workers),
            'gripper_range': (slam_dir.joinpath('05_run_calibrations.py'),
                [session, '--gripper_config', gripper_config, '--skip_slam_tag']),
            'audio_sync': (slam_dir.joinpath('06a_estimate_audio_sync.py'), ['--input', session] + workers),
            'head_tags': (tag_dir.joinpath('00_detect_tag_corners.py'), tag_args + workers),
            'segment_episodes': (tag_dir.joinpath('00b_segment_episodes.py'), tag_args),
            'use_setup': (tag_dir.joinpath('00c_use_setup.py'), tag_args),
            'calibrate_bodies': (tag_dir.joinpath('02_calibrate_bodies.py'), tag_args),
            'track_bodies': (tag_dir.joinpath('03_track_bodies.py'), tag_args + workers),
            'fuse_head_imu': (tag_dir.joinpath('03b_fuse_head_imu.py'), tag_args + workers),
            'fuse_grippers': (tag_dir.joinpath('03c_fuse_gripper_imu.py'), tag_args + workers),
            'gripper_tcp': (tag_dir.joinpath('05_calibrate_gripper_tcp.py'), tag_args),
            'render': (tag_dir.joinpath('04_render_tracking.py'), tag_args),
            'export_plan': (tag_dir.joinpath('06_export_dataset_plan.py'), tag_args),
        }
        assert set(steps) == set(STEP_NAMES)  # steps run in STEP_NAMES order
        skip = {
            'fuse_head_imu': None if 'head_imu' in config_yaml else "no head_imu in config",
            'segment_episodes': None if 'episodes' in config_yaml else "no episodes in config (one take = one demo)",
            'render': "--no_render" if no_render else None,
            'fuse_grippers': None if 'gripper_imu' in config_yaml else "no gripper_imu in config",
            # setup calibration (calibrate_setup.py) replaces per-session calibration
            'use_setup': None if 'setup_calibration' in config_yaml else "no setup_calibration in config",
            'calibrate_bodies': "using setup_calibration" if 'setup_calibration' in config_yaml else None,
            'gripper_tcp': "using setup_calibration" if 'setup_calibration' in config_yaml else None,
        }

        for i, name in enumerate(STEP_NAMES[first:last + 1], start=first):
            if skip.get(name) is not None:
                print(f"Skipping {name}: {skip[name]}")
                continue
            script_path, args = steps[name]
            run(script_path, args, log_dir.joinpath(f"{i:02d}_{name}.log"))
        print(f"Done {session.name}: steps {start_from} to {stop_after}, logs in {log_dir}")

# %%
if __name__ == "__main__":
    main()
