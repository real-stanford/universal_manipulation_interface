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
import time
import pickle
import pathlib
from multiprocessing.managers import SharedMemoryManager
import scipy.spatial.transform as st
from umi.real_world.spacemouse_shared_memory import Spacemouse
from umi.real_world.uvc_camera import UvcCamera
from umi.real_world.rtde_interpolation_controller import RTDEInterpolationController
from umi.real_world.keystroke_counter import KeystrokeCounter, KeyCode, Key
from umi.common.usb_util import reset_all_elgato_devices, get_sorted_v4l_paths
from umi.common.precise_sleep import precise_wait

# %%
@click.command()
@click.option('-o', '--output', required=True, type=str)
@click.option('-r', '--robot_ip', default='172.24.95.8')
@click.option('-v', '--v4l_idx', type=int, default=0)
def main(output, robot_ip, v4l_idx):
    cv2.setNumThreads(4)

    max_pos_speed = 0.25
    max_rot_speed = 0.6
    frequency = 10
    cube_diag = np.linalg.norm([1,1,1])
    dt = 1 / frequency

    # Find and reset all Elgato capture cards.
    # Required to workaround a firmware bug.
    reset_all_elgato_devices()

    # Wait for all v4l cameras to be back online
    time.sleep(0.1)
    v4l_paths = get_sorted_v4l_paths()
    v4l_path = v4l_paths[v4l_idx]

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter,\
            Spacemouse(shm_manager=shm_manager) as sm,\
            UvcCamera(
                shm_manager=shm_manager,
                dev_video_path=v4l_path,
                resolution=(1920,1080)) as camera,\
            RTDEInterpolationController(
                shm_manager=shm_manager,
                robot_ip=robot_ip,
                lookahead_time=0.1,
                max_pos_speed=max_pos_speed*cube_diag,
                max_rot_speed=max_rot_speed*cube_diag,
                tcp_offset_pose=[0,0,0,0,0,0],
                # joints_init=j_init,
                verbose=False) as controller:
            # warm up GUI
            data = camera.get()
            img = data['color']
            cv2.imshow('frame', img)
            cv2.pollKey()

            print('Ready!')
            # to account for recever interfance latency, use target pose
            # to init buffer.
            record_data_buffer = list()
            state = controller.get_state()
            target_pose = state['TargetTCPPose']
            stop = False
            while not stop:
                t_cycle_start = time.time()
                # handle image stuff
                data = camera.get()
                img = data['color']
                curr_pose = controller.get_state()['ActualTCPPose']

                # Display the resulting frame
                vis_img = img.copy()
                text = f'Num frames saved: {len(record_data_buffer)}'
                cv2.putText(
                    vis_img,
                    text,
                    (10,30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(255,255,255)
                )
                cv2.imshow('frame', vis_img)
                cv2.pollKey()

                # handle key presses
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        stop = True
                    elif key_stroke == Key.space:
                        # record image
                        record_data_buffer.append({
                            'img': img[...,::-1],
                            'tcp_pose': curr_pose
                        })
                        print(curr_pose)
                    elif key_stroke == Key.backspace:
                        # delete latest
                        if len(record_data_buffer) > 0:
                            record_data_buffer.pop()
                    elif key_stroke == KeyCode(char='s'):
                        # save
                        out_path = pathlib.Path(output)
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        pickle.dump(record_data_buffer, file=out_path.open('wb'))
                        print(f"Saved data to {output}")
                        
                # handle robot stuff
                sm_state = sm.get_motion_state_transformed()
                dpos = sm_state[:3] * (max_pos_speed / frequency)
                drot_xyz = sm_state[3:] * (max_rot_speed / frequency)

                drot = st.Rotation.from_euler('xyz', drot_xyz)
                target_pose[:3] += dpos
                target_pose[3:] = (drot * st.Rotation.from_rotvec(
                    target_pose[3:])).as_rotvec()

                controller.schedule_waypoint(target_pose, time.time() + dt)
                # print(f"Running at {1 / (time.time() - t_cycle_start)}")

# %%
if __name__ == '__main__':
    main()
