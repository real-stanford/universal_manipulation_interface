import sys
import os
import pathlib
import click
import shutil
import datetime

# --- Environment Setup (Kept for Context) ---
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)


def bag_get_start_datetime(file_path):
    """
    Returns the file's modification time (mtime) as a proxy for the start date/time.
    """
    try:
        mtime = pathlib.Path(file_path).stat().st_mtime
        return datetime.datetime.fromtimestamp(mtime)
    except Exception:
        return datetime.datetime.now()

def bag_get_camera_serial(file_path):
    """
    Returns a hardcoded serial number for classification purposes.
    """
    return "135122070988"


# %%
@click.command(help='Session directories. Assumming bag videos are in <session_dir>/raw_bags')
@click.argument('session_dir', nargs=-1)
def main(session_dir):
    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session)).absolute()
        
        # Hardcode subdirs
        input_dir = session.joinpath('raw_bags')
        output_dir = session.joinpath('demos')
        
        # Create raw_bags if don't exist
        if not input_dir.is_dir():
            input_dir.mkdir()
            print(f"{input_dir.name} subdir don't exist! Creating one and assuming files will be moved here externally.")
            return

        # Create output dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check for required mapping file (The simplified design requires this file to exist)
        mapping_vid_path = input_dir.joinpath('mapping.bag')
        if (not mapping_vid_path.exists()) and not(mapping_vid_path.is_symlink()):
            print(f"raw_bags/mapping.bag don't exist!")
            return
        
        # Check for required gripper calibration directory (The simplified design requires this directory to exist)
        gripper_cal_dir = input_dir.joinpath('gripper_calibration')
        if not gripper_cal_dir.is_dir():
            gripper_cal_dir.mkdir()
            print("raw_bags/gripper_calibration don't exist! Creating one.")
            return

        # Look for bag video in all subdirectories in input_dir
        input_bag_paths = list(input_dir.glob('**/*.BAG')) + list(input_dir.glob('**/*.bag'))
        print(f'Found {len(input_bag_paths)} bag videos')

        # --- Process and Structure Files ---
        for bag_path in input_bag_paths:
            print(f"[INFO] {bag_path=}")
            if bag_path.is_symlink():
                print(f"Skipping {bag_path.name}, already moved.")
                continue

            # Retrieve metadata using simplified helpers
            start_date = bag_get_start_datetime(str(bag_path))
            cam_serial = bag_get_camera_serial(str(bag_path))
            
            # Default output directory name
            out_dname = 'demo_' + cam_serial + '_' + start_date.strftime(r"%Y.%m.%d_%H.%M.%S.%f")

            # Special folders (Checking file/parent name against required structure)
            if bag_path.name.startswith('mapping'):
                out_dname = "mapping"
            elif bag_path.name.startswith('gripper_cal') or bag_path.parent.name.startswith('gripper_cal'):
                # Calibration structure: gripper_calibration_<serial>_<time>
                out_dname = "gripper_calibration_" + cam_serial + '_' + start_date.strftime(r"%Y.%m.%d_%H.%M.%S.%f")
            
            # Create directory
            this_out_dir = output_dir.joinpath(out_dname)
            this_out_dir.mkdir(parents=True, exist_ok=True)
            
            # Move/Copy videos
            vfname = 'raw_bag.bag' # Renamed to raw_bag.bag for consistency with previous scripts
            out_video_path = this_out_dir.joinpath(vfname)
            shutil.copy(bag_path, out_video_path)
            print(f"Copied {bag_path.name} to {out_video_path.relative_to(session)}")

            # Create symlink back from original location (Logic is kept but commented out)
            # dots = os.path.join(*['..'] * len(bag_path.parent.relative_to(session).parts))
            # rel_path = str(out_video_path.relative_to(session))
            # symlink_path = os.path.join(dots, rel_path)                
            # bag_path.symlink_to(symlink_path)

# %%
if __name__ == '__main__':
    if len(sys.argv) == 1:
        main.main(['--help'])
    else:
        main()