import time
import cv2
import numpy as np
import torch
import dill
import hydra
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import h5py
import vedo
from copy import deepcopy
import copy
from diffusion_policy.common.pose_repr_util import compute_relative_pose
from diffusion_policy.common.pose_util import rot6d_to_mat, mat_to_rot6d
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.model.common.rotation_transformer import RotationTransformer, RotationTransformerUMI


rot_quat2mat = RotationTransformerUMI(from_rep='quaternion', to_rep='matrix')
rot_rot6d2mat = RotationTransformerUMI(from_rep='rotation_6d', to_rep='matrix')
rot_mat2rot6d = RotationTransformerUMI(from_rep='matrix', to_rep='rotation_6d')
rot_aa2mat = RotationTransformerUMI(from_rep='axis_angle',to_rep='matrix')


def get_frame_axes_vedo (pose, axes_length):

    x_axes = (pose @ (np.array([axes_length, 0, 0, 1]).reshape((4, 1))))[:3].reshape((3,))
    y_axes = (pose @ (np.array([0, axes_length, 0, 1]).reshape((4, 1))))[:3].reshape((3,))
    z_axes = (pose @ (np.array([0, 0, axes_length, 1]).reshape((4, 1))))[:3].reshape((3,))

    ret = [vedo.Line([pose[:3, 3].reshape((3,)), x_axes], c=(255, 0, 0), lw=6),
           vedo.Line([pose[:3, 3].reshape((3,)), y_axes], c=(0, 255, 0), lw=6),
           vedo.Line([pose[:3, 3].reshape((3,)), z_axes], c=(0, 0, 255), lw=6)]
    
    return ret


def predict_from_train_dataset (filename, episode_id, timestep):

    with h5py.File(filename, 'r') as f:
        
        wrist_images = f['data']['episode_{}'.format(episode_id)]['obs']['wrist_camera'][()]
        eef_pos = f['data']['episode_{}'.format(episode_id)]['obs']['eef_pos'][()]
        eef_quat = f['data']['episode_{}'.format(episode_id)]['obs']['eef_quat'][()]
        gripper_qpos = f['data']['episode_{}'.format(episode_id)]['obs']['gripper_qpos'][()]
        eef_action = f['data']['episode_{}'.format(episode_id)]['eef_action'][()]

    # create obs dict
    np_obs_dict = dict()

    # fill in all input required by dp
    wrist_image_old = np.swapaxes(np.swapaxes(wrist_images[timestep-1], 1, 2), 0, 1).astype(np.float32)/255.
    wrist_image = np.swapaxes(np.swapaxes(wrist_images[timestep], 1, 2), 0, 1).astype(np.float32)/255.
    np_obs_dict['wrist_camera'] = np.stack((wrist_image_old, wrist_image), axis=0)

    np_obs_dict['eef_pos'] = np.stack((eef_pos[timestep-1], eef_pos[timestep]), axis=0)
    np_obs_dict['eef_quat'] = np.stack((eef_quat[timestep-1], eef_quat[timestep]), axis=0)
    np_obs_dict['gripper_qpos'] = np.stack((gripper_qpos[timestep-1], gripper_qpos[timestep]), axis=0)

    # using relative action
    current_pos = copy.copy(np_obs_dict['eef_pos'][-1])
    current_rot_mat = copy.copy(rot_quat2mat.forward(np_obs_dict['eef_quat'][-1]))
    T_world_baseframe = np.eye(4)
    T_world_baseframe[:3, :3] = current_rot_mat.copy()
    T_world_baseframe[:3, 3] = current_pos.copy()
    # print(T_world_baseframe)

    # for key in np_obs_dict:
    #     print(key, np_obs_dict[key].shape)

    np_obs_dict['eef_pos'], np_obs_dict['eef_quat'] = compute_relative_pose(
        pos=np_obs_dict['eef_pos'],
        rot=np_obs_dict['eef_quat'],
        base_pos=current_pos,
        base_rot_mat=current_rot_mat,
        rot_transformer_to_mat=rot_quat2mat,
        rot_transformer_to_target=rot_mat2rot6d
    )

    np_obs_dict_stacked = {
        key: value[np.newaxis, :] for key, value in np_obs_dict.items()
    }

    # device transfer
    obs_dict = dict_apply(np_obs_dict_stacked, 
        lambda x: torch.from_numpy(x).to(
            device='cuda'))
    
    # run inference
    with torch.no_grad():
        action_dict = policy.predict_action(obs_dict)

    # device_transfer
    np_action_dict = dict_apply(action_dict,
        lambda x: x.detach().to('cpu').numpy())

    action_chunk = np_action_dict['action'][0]
    print(action_chunk.shape)

    # action rotation transformer
    action_pos, action_rot = compute_relative_pose(
        pos=action_chunk[..., :3],
        rot=action_chunk[..., 3: -1],
        base_pos=current_pos,
        base_rot_mat=current_rot_mat,
        rot_transformer_to_mat=rot_aa2mat,
        rot_transformer_to_target=rot_mat2rot6d,
        backward=True
    )
    action_gripper = action_chunk[..., -1:]
    action_chunk_converted = np.concatenate([action_pos, action_rot, action_gripper], axis=-1)
    print(action_gripper)

    # action_chunk_converted = []
    # for action in action_chunk:
    #     action_pos = action[0:3]
    #     action_rot6d = action[3:-1]
    #     action_gripper = action[-1]

    #     action_mat = rot6d_to_mat(action_rot6d)

    #     T_baseframe_action = np.eye(4)
    #     T_baseframe_action[:3, :3] = action_mat
    #     T_baseframe_action[:3, 3] = action_pos
    #     T_world_action = T_world_baseframe @ T_baseframe_action

    #     action_pos_converted = T_world_action[:3, 3]
    #     # action_quat_converted = R.from_matrix(T_world_action[:3, :3]).as_rotvec()
    #     action_quat_converted = mat_to_rot6d(T_world_action[:3, :3])
    #     action_converted = action_pos_converted.tolist() + action_quat_converted.tolist() + [action_gripper]
    #     action_chunk_converted.append(action_converted)

    # action_chunk_converted = np.array(action_chunk_converted)

    action_chunk_gt = []
    for i in range (len(action_chunk_converted)):
        action_chunk_gt.append(eef_action[timestep + i][0:3].tolist() + mat_to_rot6d(R.from_rotvec(eef_action[timestep + i][3:-1]).as_matrix()).tolist() + [eef_action[timestep + i][-1]])
    action_chunk_gt = np.array(action_chunk_gt)

    cv2.imshow('frame', wrist_images[timestep])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plotter = vedo.Plotter(axes=1)
    visualizations = []
    action_chunk_vis = vedo.Points(action_chunk[:, 0:3], r=6, c='red')
    visualizations.append(action_chunk_vis)
    plotter.show(action_chunk_vis)

    plotter = vedo.Plotter(axes=1)
    print(current_pos)
    cur_pos_vis = vedo.Point(current_pos, r=10, c='blue')
    action_chunk_vis = vedo.Points(action_chunk_converted[:, 0:3], r=6, c='red')
    action_chunk_gt_vis = vedo.Points(action_chunk_gt[:, 0:3], r=6, c='green')
    plotter.show(cur_pos_vis, action_chunk_vis, action_chunk_gt_vis)
    
    np.set_printoptions(suppress=True, precision=10)
    print(np.round(action_chunk, 4))
    

if __name__ == '__main__':

    ckpt_path = '/home/jingyixiang/diffusion_policy/data/outputs/2026.01.22/14.29.22_train_diffusion_unet_image_grasp_cube_umi/checkpoints/latest.ckpt'

    # load checkpoint
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)

    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    policy.to(torch.device('cuda:0'))
    policy.eval()
    device = policy.device

    # run tests
    filename = '/home/jingyixiang/dp_ws/src/rocky_scripts/data/demonstrations/grasp_cube_umi_3.hdf5'
    predict_from_train_dataset(filename, 0, 50)