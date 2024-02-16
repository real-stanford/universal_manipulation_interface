# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import numpy as np
import json
import pathlib
import pickle
import collections
import click
import pandas as pd
from scipy.spatial.transform import Rotation

# %%
def pose_interp_from_df(df, tx_base_slam=None):    
    cam_pos = df[['x', 'y', 'z']].to_numpy()
    cam_rot_quat_xyzw = df[['q_x', 'q_y', 'q_z', 'q_w']].to_numpy()
    cam_rot = Rotation.from_quat(cam_rot_quat_xyzw)
    cam_pose = np.zeros((cam_pos.shape[0], 4, 4), dtype=np.float32)
    cam_pose[:,3,3] = 1
    cam_pose[:,:3,3] = cam_pos
    cam_pose[:,:3,:3] = cam_rot.as_matrix()
    tx_slam_cam = cam_pose
    tx_base_cam = tx_slam_cam
    if tx_base_slam is not None:
        tx_base_cam = tx_base_slam @ tx_slam_cam
    return tx_base_cam

# %%
@click.command(help='Session directories. Assumming mp4 videos are in <session_dir>/raw_videos')
@click.argument('session_dir', nargs=-1)
def main(session_dir):
    for session in session_dir:
        session = pathlib.Path(session)
        demos_dir = session.joinpath('demos')
        path = demos_dir.joinpath('mapping', 'tx_slam_tag.json')
        if not path.is_file():
            continue
        
        tx_slam_tag = str(path)
        tx_slam_tag = np.array(json.load(
            open(os.path.expanduser(tx_slam_tag), 'r')
            )['tx_slam_tag']
        )
        tx_tag_slam = np.linalg.inv(tx_slam_tag)
        
        # find videos
        video_dirs = sorted([x.parent for x in demos_dir.glob('demo_*/raw_video.mp4')])
        
        for vid_dir in video_dirs:
            csv_path = vid_dir.joinpath('camera_trajectory.csv')
            if not csv_path.is_file():
                # no tracking data
                continue
            
            csv_df = pd.read_csv(csv_path)
            
            if csv_df['is_lost'].sum() > 10:
                # drop episode if too many lost frames
                # unreliable tracking
                continue
            
            if (~csv_df['is_lost']).sum() < 60:
                continue
            
            df = csv_df.loc[~csv_df['is_lost']]
            tx_tag_tcp = pose_interp_from_df(df, 
                # build pose in tag frame (z-up)
                tx_base_slam=tx_tag_slam)
            tx_tag_tcp0 = tx_tag_tcp[0]
            tx_tcp0_tag = np.linalg.inv(tx_tag_tcp0)
            tx_tcp0_tcp = tx_tcp0_tag @ tx_tag_tcp
            
            rot = Rotation.from_matrix(tx_tcp0_tcp[:,:3,:3])
            euler = rot.as_euler('zyx')
            yaw = euler[:,0]
            max_yaw = np.max(np.abs(yaw))
            v = max_yaw / np.pi * 180
            if v > 40:
                print(vid_dir)
                print(v)
            
# %%
if __name__ == "__main__":
    main()
