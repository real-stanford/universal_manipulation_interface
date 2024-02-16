# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import time
import numpy as np
from multiprocessing.managers import SharedMemoryManager
import scipy.spatial.transform as st
from umi.real_world.spacemouse_shared_memory import Spacemouse
from umi.real_world.rtde_interpolation_controller import RTDEInterpolationController
from umi.common.precise_sleep import precise_wait

# %%
def main():
    robot_ip = 'ur-2017356986.internal.tri.global'
    max_pos_speed = 0.25
    max_rot_speed = 0.6
    frequency = 30
    cube_diag = np.linalg.norm([1,1,1])
    j_init = None
    j_init = np.array([0,-90,-90,-90,90,0]) / 180 * np.pi
    tcp_offset = 0.13
    # tcp_offset = 0
    command_latency = 1/100
    dt = 1/frequency

    with SharedMemoryManager() as shm_manager:
        with RTDEInterpolationController(
            shm_manager=shm_manager,
            robot_ip=robot_ip,
            lookahead_time=0.1,
            max_pos_speed=max_pos_speed*cube_diag,
            max_rot_speed=max_rot_speed*cube_diag,
            tcp_offset_pose=[0,0,tcp_offset,0,0,0],
            # joints_init=j_init,
            verbose=False) as controller:
            with Spacemouse(shm_manager=shm_manager) as sm:
                print('Ready!')
                # to account for recever interfance latency, use target pose
                # to init buffer.
                state = controller.get_state()
                target_pose = state['TargetTCPPose']
                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    precise_wait(t_sample)
                    sm_state = sm.get_motion_state_transformed()
                    # print(sm_state)
                    dpos = sm_state[:3] * (max_pos_speed / frequency)
                    drot_xyz = sm_state[3:] * (max_rot_speed / frequency)
                    
                    # if not sm.is_button_pressed(0):
                    #     # translation mode
                    #     drot_xyz[:] = 0
                    # else:
                    #     dpos[:] = 0
                    # if not sm.is_button_pressed(1):
                    #     # 2D translation mode
                    #     dpos[2] = 0    

                    drot = st.Rotation.from_euler('xyz', drot_xyz)
                    target_pose[:3] += dpos
                    target_pose[3:] = (drot * st.Rotation.from_rotvec(
                        target_pose[3:])).as_rotvec()

                    controller.schedule_waypoint(target_pose, 
                        t_command_target-time.monotonic()+time.time())
                    precise_wait(t_cycle_end)
                    iter_idx += 1


# %%
if __name__ == '__main__':
    main()
