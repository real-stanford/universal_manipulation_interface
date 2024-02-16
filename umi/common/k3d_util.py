import numpy as np
import numba

@numba.jit()
def k3d_get_pose_axis(poses, axis_size_m = 0.1):
    # points in camera frame
    points = np.zeros((4,3), dtype=poses.dtype)
    points[1,0] = axis_size_m
    points[2,1] = axis_size_m
    points[3,2] = axis_size_m

    n_poses = poses.shape[0]
    out_verts = np.zeros((n_poses * 4, 3), dtype=poses.dtype)
    out_idxs = np.zeros((n_poses * 3, 2), dtype=np.int64)
    out_colors = np.zeros((n_poses * 4,), dtype=np.int64)
    for i in range(n_poses):
        this_pose = poses[i]
        # convert points to world frame
        this_verts = points @ this_pose[:3,:3].T + this_pose[:3,3]
        # fill in vert array
        vert_idx_start = i * 4
        out_verts[vert_idx_start:vert_idx_start+4] = this_verts
        # draw 3 lines for x,y,z axis
        this_idxs = out_idxs[i*3:(i+1)*3]
        this_idxs[0] = [0,1]
        this_idxs[1] = [0,2]
        this_idxs[2] = [0,3]
        this_idxs += vert_idx_start
        # fill out vertex colors, rgb for xyz
        out_colors[i*4:(i+1)*4] = [0xffffff, 0xff0000, 0x00ff00, 0x0000ff]
    
    return out_verts, out_idxs, out_colors