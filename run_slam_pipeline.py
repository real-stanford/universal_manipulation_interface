"""
Main script for UMI SLAM pipeline.
python run_slam_pipeline.py <session_dir>
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
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
def main(session_dir, calibration_dir):
    script_dir = pathlib.Path(__file__).parent.joinpath('scripts_slam_pipeline')
    if calibration_dir is None:
        calibration_dir = pathlib.Path(__file__).parent.joinpath('example', 'calibration')
    else:
        calibration_dir = pathlib.Path(calibration_dir)
    assert calibration_dir.is_dir()

    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session)).absolute()

        print("############## 00_process_videos #############")
        script_path = script_dir.joinpath("00_process_videos.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            str(session)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 01_extract_gopro_imu ###########")
        script_path = script_dir.joinpath("01_extract_gopro_imu.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            str(session)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 02_create_map ###########")
        script_path = script_dir.joinpath("02_create_map.py")
        assert script_path.is_file()
        demo_dir = session.joinpath('demos')
        mapping_dir = demo_dir.joinpath('mapping')
        assert mapping_dir.is_dir()
        map_path = mapping_dir.joinpath('map_atlas.osa')
        if not map_path.is_file():
            cmd = [
                'python', str(script_path),
                '--input_dir', str(mapping_dir),
                '--map_path', str(map_path)
            ]
            result = subprocess.run(cmd)
            assert result.returncode == 0
            assert map_path.is_file()

        print("############# 03_batch_slam ###########")
        script_path = script_dir.joinpath("03_batch_slam.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            '--input_dir', str(demo_dir),
            '--map_path', str(map_path)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 04_detect_aruco ###########")
        script_path = script_dir.joinpath("04_detect_aruco.py")
        assert script_path.is_file()
        camera_intrinsics = calibration_dir.joinpath('gopro_intrinsics_2_7k.json')
        aruco_config = calibration_dir.joinpath('aruco_config.yaml')
        assert camera_intrinsics.is_file()
        assert aruco_config.is_file()

        cmd = [
            'python', str(script_path),
            '--input_dir', str(demo_dir),
            '--camera_intrinsics', str(camera_intrinsics),
            '--aruco_yaml', str(aruco_config)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 05_run_calibrations ###########")
        script_path = script_dir.joinpath("05_run_calibrations.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            str(session)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 06_generate_dataset_plan ###########")
        script_path = script_dir.joinpath("06_generate_dataset_plan.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            '--input', str(session)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

## %%
if __name__ == "__main__":
    main()
