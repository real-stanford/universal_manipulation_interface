import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

def load_tum_trajectory(tum_txt_path):
    tum_traj_raw = np.loadtxt(tum_txt_path, delimiter=' ', dtype=np.float32)
    if len(tum_traj_raw) == 0:
        return {
            'timestamp': np.array([]),
            'pose': np.array([]),
        }

    timestamp_sec = tum_traj_raw[:,0]
    cam_pos = tum_traj_raw[:,1:4]
    cam_rot_quat_xyzw = tum_traj_raw[:,4:8]
    cam_rot = Rotation.from_quat(cam_rot_quat_xyzw)

    cam_pose = np.zeros((cam_pos.shape[0], 4, 4), dtype=np.float32)
    cam_pose[:,3,3] = 1
    cam_pose[:,:3,3] = cam_pos
    cam_pose[:,:3,:3] = cam_rot.as_matrix()

    result = {
        'timestamp': timestamp_sec,
        'pose': cam_pose
    }
    return result

def load_csv_trajectory(csv_path):
    df = pd.read_csv(csv_path)
    if (~df.is_lost).sum() == 0:
        return {
            'raw_data': df
        }
    
    valid_df = df.loc[~df.is_lost]
    
    timestamp_sec = valid_df['timestamp'].to_numpy()
    cam_pos = valid_df[['x', 'y', 'z']].to_numpy()
    cam_rot_quat_xyzw = valid_df[['q_x', 'q_y', 'q_z', 'q_w']].to_numpy()
    cam_rot = Rotation.from_quat(cam_rot_quat_xyzw)

    cam_pose = np.zeros((cam_pos.shape[0], 4, 4), dtype=np.float32)
    cam_pose[:,3,3] = 1
    cam_pose[:,:3,3] = cam_pos
    cam_pose[:,:3,:3] = cam_rot.as_matrix()

    result = {
        'timestamp': timestamp_sec,
        'pose': cam_pose,
        'raw_data': df
    }
    return result
