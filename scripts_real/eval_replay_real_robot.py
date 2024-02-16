"""
Usage:
(robodiff)$ python eval_real_robot.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_ur5>

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import os
import pathlib
import time
from multiprocessing.managers import SharedMemoryManager

import click
import cv2
import dill
import hydra
import numpy as np
import scipy.spatial.transform as st
import skvideo.io
import torch
import zarr
from omegaconf import OmegaConf

from umi.common.interpolation_util import get_interp1d, PoseInterpolator
from diffusion_policy.common.pose_trajectory_interpolator import pose_distance
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.precise_sleep import precise_wait
from umi.real_world.umi_env import UmiEnv
from umi.real_world.real_inference_util import (get_real_obs_dict,
                                                get_real_obs_resolution)
from umi.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()

OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--input', '-i', required=True, help='Path to dataset')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--replay_episode', '-re', type=int, default=0)
# @click.option('--robot_ip', '-ri', default='172.24.95.9')
# @click.option('--gripper_ip', '-gi', default='172.24.95.17')
@click.option('--robot_ip', default='172.24.95.8')
@click.option('--gripper_ip', default='172.24.95.18')
@click.option('--camera_reorder', '-cr', default='120')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=60, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
def main(input, output, replay_episode, robot_ip, gripper_ip,
    camera_reorder,
    vis_camera_idx, init_joints, 
    frequency, command_latency):
    max_gripper_width = 0.09
    gripper_speed = 0.2

    # load replay buffer
    with zarr.ZipStore(input, mode='r') as zip_store:
        replay_buffer = ReplayBuffer.copy_from_store(
            src_store=zip_store, store=zarr.MemoryStore())
    obs_res = replay_buffer['camera0_rgb'].shape[1:-1][::-1]
    episode_data = None

    # setup experiment
    dt = 1/frequency
    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager) as sm, UmiEnv(
            output_dir=output, 
            robot_ip=robot_ip,
            gripper_ip=gripper_ip,
            frequency=frequency,
            obs_image_resolution=obs_res,
            obs_float32=True,
            camera_reorder=[int(x) for x in camera_reorder],
            init_joints=init_joints,
            enable_multi_cam_vis=True,
            # latency
            camera_obs_latency=0.125,
            robot_obs_latency=0.0001,
            gripper_obs_latency=0.01,
            robot_action_latency=0.0,
            gripper_action_latency=0.0,
            # obs
            # action
            max_pos_speed=2.0,
            max_rot_speed=6.0,
            shm_manager=shm_manager) as env:
            cv2.setNumThreads(2)
            print("Waiting for camera")
            time.sleep(1.0)

            print('Ready!')
            while True:
                # ========= human control loop ==========
                print("Human in control!")
                state = env.get_robot_state()
                target_pose = state['TargetTCPPose']
                gripper_target_pos = max_gripper_width
                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # pump obs
                    obs = env.get_obs()

                    # visualize
                    vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1]
                    episode_slice = replay_buffer.get_episode_slice(replay_episode)
                    start_idx = episode_slice.start
                    match_img = replay_buffer['camera0_rgb'][start_idx]
                    start_pos = replay_buffer['robot0_eef_pos'][start_idx]
                    start_rot = replay_buffer['robot0_eef_rot_axis_angle'][start_idx]
                    start_pose = np.concatenate([start_pos, start_rot])
                    start_gripper_width = replay_buffer['robot0_gripper_width'][start_idx]
                    match_img = match_img.astype(np.float32) / 255
                    avg_img = (vis_img + match_img) / 2
                    vis_img = np.concatenate([vis_img, avg_img, match_img], axis=1)

                    text = f'Episode: {replay_episode}'
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255,255,255)
                    )
                    cv2.imshow('default', vis_img[...,::-1])
                    key_stroke = cv2.pollKey()
                    if key_stroke == ord('q'):
                        # Exit program
                        env.end_episode()
                        exit(0)
                    elif key_stroke == ord('c'):
                        # Exit human control loop
                        pos = obs['robot0_eef_pos'][-1]
                        rot = obs['robot0_eef_rot_axis_angle'][-1]
                        pose = np.concatenate([pos, rot])
                        pos_dist, rot_dist = pose_distance(start_pose, pose)
                        if pos_dist < 0.01 and rot_dist < 0.05:
                            # hand control over to the policy
                            break
                        else:
                            print('Have not reached start pose yet! Press "M" to move to start pose.')
                    elif key_stroke == ord('e'):
                        # Next episode
                        replay_episode = min(replay_episode + 1, replay_buffer.n_episodes-1)
                    elif key_stroke == ord('w'):
                        # Prev episode
                        replay_episode = max(replay_episode - 1, 0)
                    elif key_stroke == ord('m'):
                        # move the robot
                        duration = 3.0
                        env.robot.servoL(start_pose, duration=duration)
                        gripper_target_pos = start_gripper_width
                        start_t = time.time()
                        episode_data = replay_buffer.get_episode(replay_episode)
                        time.sleep(max(duration - (time.time() - start_t), 0))
                        target_pose = start_pose
                        
                    precise_wait(t_sample)
                    # get teleop command
                    sm_state = sm.get_motion_state_transformed()
                    # print(sm_state)
                    dpos = sm_state[:3] * (0.5 / frequency)
                    drot_xyz = sm_state[3:] * (2.0 / frequency)

                    drot = st.Rotation.from_euler('xyz', drot_xyz)
                    target_pose[:3] += dpos
                    target_pose[3:] = (drot * st.Rotation.from_rotvec(
                        target_pose[3:])).as_rotvec()
                    
                    dpos = 0
                    if sm.is_button_pressed(0):
                        # close gripper
                        dpos = -gripper_speed / frequency
                    if sm.is_button_pressed(1):
                        dpos = gripper_speed / frequency
                    gripper_target_pos = np.clip(gripper_target_pos + dpos, 0, max_gripper_width)

                    action = np.zeros((7,))
                    action[:6] = target_pose
                    action[-1] = gripper_target_pos                    
                    
                    # execute teleop command
                    if t_command_target > (time.monotonic() + 0.001):
                        env.exec_actions(
                            actions=[action], 
                            timestamps=[t_command_target-time.monotonic()+time.time()])
                    precise_wait(t_cycle_end)
                    iter_idx += 1
                
                # ========== policy control loop ==============
                try:

                    # pre-compute interpolation
                    data_frequency = 59.94
                    n_data_samples = len(episode_data['robot0_eef_pos'])
                    data_timestamps = np.arange(n_data_samples).astype(np.float32) / data_frequency
                    data_pose = np.concatenate([episode_data['robot0_eef_pos'], episode_data['robot0_eef_rot_axis_angle']], axis=-1)
                    data_pose_interpolator = PoseInterpolator(data_timestamps, data_pose)
                    data_gripper_interpolator = get_interp1d(data_timestamps + 0.05, episode_data['robot0_gripper_width'])
                    # camera
                    exec_timestamps = np.arange(int(np.floor(data_timestamps[-1] * frequency))) / frequency
                    exec_data_idxs = np.round(np.clip(exec_timestamps, 0, data_timestamps[-1]) * data_frequency).astype(np.int32)

                    # start episode
                    start_delay = 1.0 
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")

                    for iter_idx, t in enumerate(exec_timestamps):
                        t_cycle_start = t_start + t
                        t_cycle_end = t_cycle_start + 1/frequency
                        data_idx = exec_data_idxs[iter_idx]

                        obs = env.get_obs()
                        pose = data_pose_interpolator(t)
                        pose[2] -= 0.02
                        grip = data_gripper_interpolator(t) - 0.005
                        action = np.concatenate([pose, grip], axis=-1)

                        env.exec_actions(
                            actions=[action], 
                            timestamps=[t_cycle_end-time.monotonic()+time.time()])

                        # plot image overlay
                        img = episode_data['camera0_rgb'][data_idx]
                        vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1]
                        match_img = img.astype(np.float32) / 255
                        avg_img = (vis_img + match_img) / 2
                        vis_img = np.concatenate([vis_img, avg_img, match_img], axis=1)

                        cv2.imshow('default', vis_img[...,::-1])
                        key_stroke = cv2.pollKey()
                        if key_stroke == ord('s'):
                            # Stop episode
                            # Hand control back to human
                            env.end_episode()
                            print('Stopped.')
                            break
                            
                        precise_wait(t_cycle_end)
                    env.end_episode()

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                
                print("Stopped.")



# %%
if __name__ == '__main__':
    main()