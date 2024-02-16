# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
print(ROOT_DIR)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import time
import numpy as np
from multiprocessing.managers import SharedMemoryManager
import scipy.spatial.transform as st
from umi.real_world.spacemouse_shared_memory import Spacemouse
from umi.real_world.rtde_interpolation_controller import RTDEInterpolationController
from umi.real_world.franka_interpolation_controller import FrankaInterpolationController
from umi.common.precise_sleep import precise_wait
from umi.common.latency_util import get_latency
from matplotlib import pyplot as plt

# %%
@click.command()
@click.option('-rh', '--robot_hostname', default='172.16.0.3')
@click.option('-f', '--frequency', type=float, default=30)
def main(robot_hostname, frequency):
    max_pos_speed = 0.5
    max_rot_speed = 1.2
    cube_diag = np.linalg.norm([1,1,1])
    tcp_offset = 0.21
    # tcp_offset = 0
    dt = 1/frequency
    command_latency = dt / 2

    with SharedMemoryManager() as shm_manager:
        # with RTDEInterpolationController(
        #     shm_manager=shm_manager,
        #     robot_ip=robot_hostname,
        #     frequency=500,
        #     lookahead_time=0.1,
        #     gain=300,
        #     max_pos_speed=max_pos_speed*cube_diag,
        #     max_rot_speed=max_rot_speed*cube_diag,
        #     tcp_offset_pose=[0,0,tcp_offset,0,0,0],
        #     get_max_k=10000,
        #     verbose=False
        with FrankaInterpolationController(
            shm_manager=shm_manager,
            robot_ip=robot_hostname,
            frequency=200,
            Kx_scale=np.array([0.8,0.8,1.2,3.0,3.0,3.0]),
            Kxd_scale=np.array([2.0,2.0,2.0,2.0,2.0,2.0]),
            verbose=False
        ) as controller,\
        Spacemouse(
            shm_manager=shm_manager
        ) as sm:
            print('Ready!')
            # to account for recever interfance latency, use target pose
            # to init buffer.
            state = controller.get_state()
            target_pose = state['ActualTCPPose']
            t_start = time.time()
            
            t_target = list()
            x_target = list()

            iter_idx = 0
            while True:
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                precise_wait(t_sample, time_func=time.time)
                sm_state = sm.get_motion_state_transformed()
                # print(sm_state)
                dpos = sm_state[:3] * (max_pos_speed / frequency)
                drot_xyz = sm_state[3:] * (max_rot_speed / frequency)

                drot = st.Rotation.from_euler('xyz', drot_xyz)
                target_pose[:3] += dpos
                target_pose[3:] = (drot * st.Rotation.from_rotvec(
                    target_pose[3:])).as_rotvec()
                
                t_target.append(t_command_target)
                x_target.append(target_pose.copy())

                controller.schedule_waypoint(target_pose, 
                    t_command_target)
                
                if sm.is_button_pressed(0) or sm.is_button_pressed(1):
                    # close gripper
                    break

                precise_wait(t_cycle_end, time_func=time.time)
                iter_idx += 1

            states = controller.get_all_state()

    t_target = np.array(t_target)
    x_target = np.array(x_target)
    t_actual = states['robot_receive_timestamp']
    x_actual = states['ActualTCPPose']
    n_dims = 6
    fig, axes = plt.subplots(n_dims, 3)
    fig.set_size_inches(15, 15, forward=True)

    for i in range(n_dims):
        latency, info = get_latency(x_target[...,i], t_target, x_actual[...,i], t_actual)

        row = axes[i]
        ax = row[0]
        ax.plot(info['lags'], info['correlation'])
        ax.set_xlabel('lag')
        ax.set_ylabel('cross-correlation')
        ax.set_title(f"Action Dim {i} Cross Correlation")

        ax = row[1]
        ax.plot(t_target, x_target[...,i], label='target')
        ax.plot(t_actual, x_actual[...,i], label='actual')
        ax.set_xlabel('time')
        ax.set_ylabel('gripper-width')
        ax.legend()
        ax.set_title(f"Action Dim {i} Raw observation")

        ax = row[2]
        t_samples = info['t_samples'] - info['t_samples'][0]
        ax.plot(t_samples, info['x_target'], label='target')
        ax.plot(t_samples-latency, info['x_actual'], label='actual-latency')
        ax.set_xlabel('time')
        ax.set_ylabel('gripper-width')
        ax.legend()
        ax.set_title(f"Action Dim {i} Aligned with latency={latency}")

    fig.tight_layout()
    plt.show()

# %%
if __name__ == '__main__':
    main()
