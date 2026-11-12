"""
UMI SLAM pipeline with audio-based multi-camera sync.
Runs run_slam_pipeline.py, then re-generates dataset_plan.pkl
with per-video clock offsets estimated from shared sounds
(grippers banged together at the start and end of each recording).
python run_slam_pipeline_synced.py <session_dir>
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

# %%
@click.command()
@click.argument('session_dir', nargs=-1)
@click.option('-c', '--calibration_dir', type=str, default=None)
@click.option('-s', '--slam_setting', type=str, default=None, help='ORB_SLAM3 setting yaml, passed to run_slam_pipeline.py')
@click.option('-m', '--slam_mask', type=str, default=None, help='SLAM mask png, passed to run_slam_pipeline.py')
def main(session_dir, calibration_dir, slam_setting, slam_mask):
    root_dir = pathlib.Path(__file__).parent
    script_dir = root_dir.joinpath('scripts_slam_pipeline')

    print("############## run_slam_pipeline #############")
    script_path = root_dir.joinpath("run_slam_pipeline.py")
    assert script_path.is_file()
    cmd = ['python', str(script_path)] + list(session_dir)
    if calibration_dir is not None:
        cmd += ['--calibration_dir', calibration_dir]
    if slam_setting is not None:
        cmd += ['--slam_setting', slam_setting]
    if slam_mask is not None:
        cmd += ['--slam_mask', slam_mask]
    result = subprocess.run(cmd)
    assert result.returncode == 0

    if calibration_dir is None:
        calibration_dir = root_dir.joinpath('example', 'calibration')
    # optional, default UMI gripper convention
    gripper_config = pathlib.Path(calibration_dir).joinpath('gripper_config.yaml')

    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session)).absolute()

        print("############# 06a_estimate_audio_sync ###########")
        script_path = script_dir.joinpath("06a_estimate_audio_sync.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            '--input', str(session)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 06b_generate_dataset_plan_synced ###########")
        script_path = script_dir.joinpath("06b_generate_dataset_plan_synced.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            '--input', str(session)
        ]
        if gripper_config.is_file():
            cmd += ['--gripper_config', str(gripper_config)]
        result = subprocess.run(cmd)
        assert result.returncode == 0

## %%
if __name__ == "__main__":
    main()
