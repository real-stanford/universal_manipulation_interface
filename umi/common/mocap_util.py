from datetime import datetime
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation

def get_mocap_start_datetime(csv_path):
    with open(csv_path, 'r') as f:
        first_row = f.readline().split(',')
    meta_dict = dict()
    for i in range(len(first_row) // 2):
        start = i * 2
        end = start + 1
        meta_dict[first_row[start]] = first_row[end]
    start_timestamp_str = meta_dict['Capture Start Time']
    start_date = datetime.strptime(start_timestamp_str, r"%Y-%m-%d %I.%M.%S.%f %p")
    return start_date

def get_mocap_data(csv_path, rigid_body_name):
    mocap_df = pd.read_csv(csv_path, skiprows=2, index_col=0, header=[1,3,4])
    assert mocap_df.index[0] == 0
    assert mocap_df.index[-1] == (len(mocap_df) - 1)
    assert mocap_df.columns[0][-1] == 'Time (Seconds)'

    time_since_start = mocap_df.iloc[:,0].to_numpy()
    pos = np.zeros((len(mocap_df), 3))
    pos[:,0] = mocap_df[(rigid_body_name, 'Position', 'X')]
    pos[:,1] = mocap_df[(rigid_body_name, 'Position', 'Y')]
    pos[:,2] = mocap_df[(rigid_body_name, 'Position', 'Z')]

    rot_quat = np.zeros((len(mocap_df), 4))
    rot_quat[:,0] = mocap_df[(rigid_body_name, 'Rotation', 'X')]
    rot_quat[:,1] = mocap_df[(rigid_body_name, 'Rotation', 'Y')]
    rot_quat[:,2] = mocap_df[(rigid_body_name, 'Rotation', 'Z')]
    rot_quat[:,3] = mocap_df[(rigid_body_name, 'Rotation', 'W')]
    rot = Rotation.from_quat(rot_quat)

    pose = np.zeros((pos.shape[0], 4, 4), dtype=pos.dtype)
    pose[:,3,3] = 1
    pose[:,:3,:3] = rot.as_matrix()
    pose[:,:3,3] = pos

    result = {
        'time_since_start': time_since_start,
        'pose': pose,
    }
    return result
