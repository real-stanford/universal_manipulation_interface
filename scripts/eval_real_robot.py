# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import numpy as np
import cv2
import hydra
import torch
import dill
import time
import json
import pathlib
from multiprocessing.managers import SharedMemoryManager
import scipy.spatial.transform as st
from umi.real_world.spacemouse_shared_memory import Spacemouse
from umi.real_world.uvc_camera import UvcCamera
from umi.real_world.rtde_interpolation_controller import RTDEInterpolationController
from umi.real_world.wsg_controller import WSGController
from umi.real_world.keystroke_counter import KeystrokeCounter, KeyCode, Key
from umi.common.usb_util import create_usb_list
from umi.common.precise_sleep import precise_wait
from umi.common.cv_util import get_image_transform
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from umi.common.interpolation_util import get_interp1d, PoseInterpolator
from umi.common.pose_util import mat_to_pose, pose10d_to_mat, mat_to_pose

# %%
def get_obs_dict(camera, controller, gripper, n_obs_steps, img_tf):
    camera_data = camera.get(k=n_obs_steps)
    robot_data = controller.get_all_state()
    gripper_data = gripper.get_all_state()

    raw_imgs = camera_data['color']
    img_timestamps = camera_data['camera_capture_timestamp']
    imgs = list()
    for x in raw_imgs:
        # bgr to rgb
        imgs.append(img_tf(x)[...,::-1])
    imgs = np.array(imgs)
    # T,H,W,C to T,C,H,W
    imgs = np.moveaxis(imgs,-1,1).astype(np.float32) / 255.

    gripper_interp = get_interp1d(
        t=gripper_data['gripper_receive_timestamp'],
        x=gripper_data['gripper_position'] / 1000) # mm to meters
    robot_interp = PoseInterpolator(
        t=robot_data['robot_receive_timestamp'],
        x=robot_data['ActualTCPPose'])

    robot_eef_pose = robot_interp(img_timestamps).astype(np.float32)
    gripper_width = gripper_interp(img_timestamps).astype(np.float32)

    obs_dict = {
        'img': imgs,
        'robot_eef_pose': robot_eef_pose,
        'gripper_width': gripper_width
    }
    return obs_dict, img_timestamps

    
# %%
@click.command()
@click.option('-i', '--input', required=True)
@click.option('-rh', '--robot_hostname', default='ur-2017356986.internal.tri.global')
@click.option('-gh', '--gripper_hostname', default='wsg50-00004544.internal.tri.global')
@click.option('-gp', '--gripper_port', type=int, default=1000)
@click.option('-f', '--frequency', type=float, default=30)
@click.option('-v', '--video_path', default='/dev/video0')
@click.option('-s', '--steps_per_inference', type=int, default=16)
def main(input, robot_hostname, gripper_hostname, gripper_port, frequency, video_path, steps_per_inference):
    cv2.setNumThreads(1)

    # load checkpoint
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # diffusion model
    policy: BaseImagePolicy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device('cuda:0')
    policy.eval().to(device)

    # set inference params
    policy.num_inference_steps = 16
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    n_obs_steps = policy.n_obs_steps

    # setup env
    max_pos_speed = 0.25
    max_rot_speed = 0.6
    cube_diag = np.linalg.norm([1,1,1])

    # load tcp offset
    tx_cam_gripper = np.array(
        json.load(open('data/calibration/robot_world_hand_eye.json', 'r')
                  )['tx_gripper2camera'])
    tx_gripper_cam = np.linalg.inv(tx_cam_gripper)
    tcp_offset_pose = mat_to_pose(tx_gripper_cam)

    # tcp_offset = 0
    dt = 1/frequency
    command_latency = dt / 2

    dev_video_path = video_path
    # enumerate UBS device to find Elgato Capture Card
    device_list = create_usb_list()
    dev_usb_path = None
    for dev in device_list:
        if 'Elgato' in dev['description']:
            dev_usb_path = dev['path']
            print('Found :', dev['description'])
            break
    
    capture_res = (1280, 720)
    out_res = (224, 224)
    img_tf = get_image_transform(in_res=capture_res, 
        out_res=out_res, crop_ratio=0.65)

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter(
        ) as key_counter,\
        Spacemouse(shm_manager=shm_manager
        ) as sm,\
        UvcCamera(
            shm_manager=shm_manager,
            dev_video_path=dev_video_path,
            dev_usb_path=dev_usb_path,
            resolution=capture_res
        ) as camera,\
        WSGController(
            shm_manager=shm_manager,
            hostname=gripper_hostname,
            port=gripper_port,
            frequency=frequency,
            move_max_speed=400.0,
            verbose=False
        ) as gripper,\
        RTDEInterpolationController(
            shm_manager=shm_manager,
            robot_ip=robot_hostname,
            lookahead_time=0.05,
            gain=1000,
            max_pos_speed=max_pos_speed*cube_diag,
            max_rot_speed=max_rot_speed*cube_diag,
            tcp_offset_pose=tcp_offset_pose,
            verbose=False
        ) as controller:
            time.sleep(1.0)

            # policy warmup
            obs_dict_np, obs_timestamps = get_obs_dict(
                camera=camera, 
                controller=controller, 
                gripper=gripper, 
                n_obs_steps=n_obs_steps,
                img_tf=img_tf)
            with torch.no_grad():
                print("Warming up policy")
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action'][0].detach().to('cpu').numpy()

            print('Ready!')
            while True:
                # ========= human control loop ==========
                print("Human in control!")

                gripper_speed = 200.
                state = controller.get_state()
                target_pose = state['TargetTCPPose']
                gripper_target_pos = gripper.get_state()['gripper_position']
                t_start = time.monotonic()
                gripper.restart_put(t_start-time.monotonic() + time.time())

                iter_idx = 0
                while True:
                    s = time.time()
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    vis_img = camera.get()['color']
                    cv2.imshow('main camera', vis_img)
                    key_stroke = cv2.pollKey()
                    if key_stroke != -1:
                        print(key_stroke)
                    if key_stroke == ord('q'):
                        exit(0)
                    if key_stroke == ord('c'):
                        break

                    precise_wait(t_sample)
                    sm_state = sm.get_motion_state_transformed()
                    # print(sm_state)
                    dpos = sm_state[:3] * (0.25 / frequency)
                    drot_xyz = sm_state[3:] * (0.6 / frequency)

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
                    gripper_target_pos = np.clip(gripper_target_pos + dpos, 0, 90.)
    
                    if t_command_target > time.monotonic():
                        # skip outdated command
                        controller.schedule_waypoint(target_pose, 
                            t_command_target-time.monotonic()+time.time())
                        gripper.schedule_waypoint(gripper_target_pos, 
                            t_command_target-time.monotonic()+time.time())

                    precise_wait(t_cycle_end)
                    iter_idx += 1

                # ========== policy control loop ==============
                print("Robot in control!")
                try:
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/59
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        obs_dict_np, obs_timestamps = get_obs_dict(
                            camera=camera, 
                            controller=controller, 
                            gripper=gripper, 
                            n_obs_steps=n_obs_steps,
                            img_tf=img_tf)
                        with torch.no_grad():
                            s = time.time()
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            action = result['action'][0].detach().to('cpu').numpy()
                            print('Inference latency:', time.time() - s)
                        
                        # action conversion
                        action_pose10d = action[:,:9]
                        action_grip = action[:,9:]
                        action_pose = mat_to_pose(pose10d_to_mat(action_pose10d))
                        action = np.concatenate([action_pose, action_grip], axis=-1)

                        # deal with timing
                        # the same step actions are always the target for
                        action_timestamps = (np.arange(len(action), dtype=np.float64)
                            ) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            # exceeded time budget, still do something
                            action = action[[-1]]
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                        else:
                            action = action[is_new]
                            action_timestamps = action_timestamps[is_new]

                        # execute actions
                        robot_action = action[:,:6]
                        gripper_action = action[:,-1] * 1000 - 4 # m to mm

                        for i in range(len(action_timestamps)):
                            controller.schedule_waypoint(robot_action[i], 
                                action_timestamps[i])
                            gripper.schedule_waypoint(gripper_action[i], 
                                action_timestamps[i])

                        # visualize
                        vis_img = camera.get()['color']
                        cv2.imshow('main camera', vis_img)

                        key_stroke = cv2.pollKey()
                        if key_stroke == ord('s'):
                            print('Stopped.')
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                
                print("Stopped.")



            

            

# %%
if __name__ == "__main__":
    main()
