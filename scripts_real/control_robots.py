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
from umi.real_world.rtde_interpolation_controller import RTDEInterpolationController
from umi.real_world.dh_controller import DHController
from umi.common.precise_sleep import precise_wait

# %%
@click.command()
@click.option('-rh', '--robot_hostname', default='192.168.2.152')
# @click.option('-gh', '--gripper_hostname', default='172.24.95.27')
# @click.option('-gp', '--gripper_port', type=int, default=1000)
@click.option('-f', '--frequency', type=float, default=30)
@click.option('-gs', '--gripper_speed', type=float, default=0.07273)
def main(robot_hostname, frequency, gripper_speed):
    max_pos_speed = 0.1
    max_rot_speed = 0.6
    max_gripper_width = 0.08
    cube_diag = np.linalg.norm([1,1,1])
    tcp_offset = 0.13
    dt = 1/frequency
    command_latency = dt / 2

    with SharedMemoryManager() as shm_manager:
        with DHController(
            shm_manager=shm_manager,
            port="/dev/ttyUSBDH_",
            receive_latency=0.01,
            use_meters=True
        ) as gripper,\
        RTDEInterpolationController(
            shm_manager=shm_manager,
            robot_ip=robot_hostname,
            frequency=125,
            lookahead_time=0.1,
            gain=300,
            max_pos_speed=max_pos_speed*cube_diag,
            max_rot_speed=max_rot_speed*cube_diag,
            tcp_offset_pose=[0,0,tcp_offset,0,0,0],
            verbose=False
        ) as controller:
            print('Ready!')
            # to account for recever interfance latency, use target pose
            # to init buffer.
            state = controller.get_state()
            target_pose = state['TargetTCPPose']
            gripper_target_pos = gripper.get_state()['gripper_position']
            t_start = time.monotonic()
            gripper.restart_put(t_start-time.monotonic() + time.time())
            
            ur_start_time = time.time()
            dh_start_time = time.time()
            iter_idx = 0
            while True:
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                precise_wait(t_sample)
                # sm_state = [0, 0, 0, 0, 0, 0]
                dpos = 0.0
                ur_time = 2
                if time.time() - ur_start_time < ur_time:
                    dpos = -max_pos_speed / frequency
                if time.time() - ur_start_time > ur_time and time.time() - ur_start_time < 2 * ur_time:
                    dpos = max_pos_speed / frequency
                if time.time() - ur_start_time > 2 * ur_time:
                    ur_start_time = time.time()
                target_pose[2] = target_pose[2] + dpos

                # dpos = sm_state[:3] * (max_pos_speed / frequency)
                # drot_xyz = sm_state[3:] * (max_rot_speed / frequency)

                # drot = st.Rotation.from_euler('xyz', drot_xyz)
                # target_pose[:3] += dpos
                # target_pose[3:] = (drot * st.Rotation.from_rotvec(
                #     target_pose[3:])).as_rotvec()
                    
                dpos = 0
                dh_time = 3
                if time.time() - dh_start_time < dh_time:
                    dpos = -gripper_speed / frequency
                if time.time() - dh_start_time > dh_time and time.time() - dh_start_time < 2 * dh_time:
                    dpos = gripper_speed / frequency
                if time.time() - dh_start_time > 2 * dh_time:
                    dh_start_time = time.time()
                gripper_target_pos = np.clip(gripper_target_pos + dpos, 0, max_gripper_width)
 
                controller.schedule_waypoint(target_pose, 
                    t_command_target-time.monotonic()+time.time())
                gripper.schedule_waypoint(gripper_target_pos, 
                    t_command_target-time.monotonic()+time.time())

                precise_wait(t_cycle_end)
                iter_idx += 1

# %%
if __name__ == '__main__':
    main()