"""
python scripts_slam_pipeline/05_run_calibrations.py data_workspace/cup_in_the_wild/20240105_zhenjia_packard_2nd_conference_room
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
from exiftool import ExifToolHelper
from umi.common.gripper_config_util import load_gripper_config

# %%
@click.command()
@click.argument('session_dir', nargs=-1)
@click.option('-gc', '--gripper_config', default=None, help='Gripper config yaml, default UMI finger tag convention')
def main(session_dir, gripper_config):
    gripper_config = load_gripper_config(gripper_config)
    script_dir = pathlib.Path(__file__).parent.parent.joinpath('scripts')
    
    for session in session_dir:
        session = pathlib.Path(session)
        demos_dir = session.joinpath('demos')
        mapping_dir = demos_dir.joinpath('mapping')
        slam_tag_path = mapping_dir.joinpath('tx_slam_tag.json')
            
        # run slam tag calibration
        script_path = script_dir.joinpath('calibrate_slam_tag.py')
        assert script_path.is_file()
        tag_path = mapping_dir.joinpath('tag_detection.pkl')
        assert tag_path.is_file()
        csv_path = mapping_dir.joinpath('camera_trajectory.csv')
        if not csv_path.is_file():
            csv_path = mapping_dir.joinpath('mapping_camera_trajectory.csv')
            print("camera_trajectory.csv not found! using mapping_camera_trajectory.csv")
        assert csv_path.is_file()
        
        cmd = [
            'python', str(script_path),
            '--tag_detection', str(tag_path),
            '--csv_trajectory', str(csv_path),
            '--output', str(slam_tag_path),
            '--keyframe_only'
        ]
        subprocess.run(cmd)
        
        # run gripper range calibration
        script_path = script_dir.joinpath('calibrate_gripper_range.py')
        assert script_path.is_file()
        
        for gripper_dir in demos_dir.glob("gripper_calibration*"):
            gripper_range_path = gripper_dir.joinpath('gripper_range.json')
            tag_path = gripper_dir.joinpath('tag_detection.pkl')
            assert tag_path.is_file()
            cmd = [
                'python', str(script_path),
                '--input', str(tag_path),
                '--output', str(gripper_range_path)
            ]
            if gripper_config is not None:
                with ExifToolHelper() as et:
                    meta = list(et.get_metadata(str(gripper_dir.joinpath('raw_video.mp4'))))[0]
                cam_serial = meta['QuickTime:CameraSerialNumber']
                if cam_serial not in gripper_config['grippers']:
                    print(f"Skipping {gripper_dir.name}, camera {cam_serial} not in gripper config")
                    continue
                gripper = gripper_config['grippers'][cam_serial]
                cmd += [
                    '--gripper_id', str(gripper['gripper_id']),
                    '--left_id', str(gripper['left_finger_tag_id']),
                    '--right_id', str(gripper['right_finger_tag_id'])
                ]
                if 'nominal_z' in gripper_config:
                    cmd += ['--nominal_z', str(gripper_config['nominal_z'])]
            subprocess.run(cmd)

            
# %%
if __name__ == "__main__":
    main()
