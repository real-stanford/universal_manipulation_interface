import numpy as np


def compute_relative_pose(pos, rot, base_pos, base_rot_mat,
                          rot_transformer_to_mat,
                          rot_transformer_to_target,
                          backward=False,
                          delta=False):
    if not backward:
        # forward pass
        if not delta:
            output_pos = pos if base_pos is None else pos - base_pos
            output_rot = rot_transformer_to_target.forward(
                rot_transformer_to_mat.forward(rot) @ np.linalg.inv(base_rot_mat))
            return output_pos, output_rot
        else:
            all_pos = np.concatenate([base_pos[None,...], pos], axis=0)
            output_pos = np.diff(all_pos, axis=0)
            
            rot_mat = rot_transformer_to_mat.forward(rot)
            all_rot_mat = np.concatenate([base_rot_mat[None,...], rot_mat], axis=0)
            prev_rot = np.linalg.inv(all_rot_mat[:-1])
            curr_rot = all_rot_mat[1:]
            rot = np.matmul(curr_rot, prev_rot)
            output_rot = rot_transformer_to_target.forward(rot)
            return output_pos, output_rot
            
    else:
        # backward pass
        if not delta:
            output_pos = pos if base_pos is None else pos + base_pos
            output_rot = rot_transformer_to_mat.inverse(
                rot_transformer_to_target.inverse(rot) @ base_rot_mat)
            return output_pos, output_rot
        else:
            output_pos = np.cumsum(pos, axis=0) + base_pos
            
            rot_mat = rot_transformer_to_target.inverse(rot)
            output_rot_mat = np.zeros_like(rot_mat)
            curr_rot = base_rot_mat
            for i in range(len(rot_mat)):
                curr_rot = rot_mat[i] @ curr_rot
                output_rot_mat[i] = curr_rot
            output_rot = rot_transformer_to_mat.inverse(rot)
            return output_pos, output_rot


def convert_pose_mat_rep(pose_mat, base_pose_mat, pose_rep='abs', backward=False):
    if not backward:
        # training transform
        if pose_rep == 'abs':
            return pose_mat
        elif pose_rep == 'rel':
            # legacy buggy implementation
            # for compatibility
            pos = pose_mat[...,:3,3] - base_pose_mat[:3,3]
            rot = pose_mat[...,:3,:3] @ np.linalg.inv(base_pose_mat[:3,:3])
            out = np.copy(pose_mat)
            out[...,:3,:3] = rot
            out[...,:3,3] = pos
            return out
        elif pose_rep == 'relative':
            out = np.linalg.inv(base_pose_mat) @ pose_mat
            return out
        elif pose_rep == 'delta':
            all_pos = np.concatenate([base_pose_mat[None,:3,3], pose_mat[...,:3,3]], axis=0)
            out_pos = np.diff(all_pos, axis=0)
            
            all_rot_mat = np.concatenate([base_pose_mat[None,:3,:3], pose_mat[...,:3,:3]], axis=0)
            prev_rot = np.linalg.inv(all_rot_mat[:-1])
            curr_rot = all_rot_mat[1:]
            out_rot = np.matmul(curr_rot, prev_rot)
            
            out = np.copy(pose_mat)
            out[...,:3,:3] = out_rot
            out[...,:3,3] = out_pos
            return out
        else:
            raise RuntimeError(f"Unsupported pose_rep: {pose_rep}")

    else:
        # eval transform
        if pose_rep == 'abs':
            return pose_mat
        elif pose_rep == 'rel':
            # legacy buggy implementation
            # for compatibility
            pos = pose_mat[...,:3,3] + base_pose_mat[:3,3]
            rot = pose_mat[...,:3,:3] @ base_pose_mat[:3,:3]
            out = np.copy(pose_mat)
            out[...,:3,:3] = rot
            out[...,:3,3] = pos
            return out
        elif pose_rep == 'relative':
            out = base_pose_mat @ pose_mat
            return out
        elif pose_rep == 'delta':
            output_pos = np.cumsum(pose_mat[...,:3,3], axis=0) + base_pose_mat[:3,3]
            
            output_rot_mat = np.zeros_like(pose_mat[...,:3,:3])
            curr_rot = base_pose_mat[:3,:3]
            for i in range(len(pose_mat)):
                curr_rot = pose_mat[i,:3,:3] @ curr_rot
                output_rot_mat[i] = curr_rot
            
            out = np.copy(pose_mat)
            out[...,:3,:3] = output_rot_mat
            out[...,:3,3] = output_pos
            return out
        else:
            raise RuntimeError(f"Unsupported pose_rep: {pose_rep}")
            