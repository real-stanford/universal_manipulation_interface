# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import os
import click
import pickle
import numpy as np
import time

from multiprocessing.managers import SharedMemoryManager
from umi.common.precise_sleep import precise_sleep, precise_wait
from umi.real_world.spacemouse_shared_memory import Spacemouse
from umi.real_world.rtde_interpolation_controller import RTDEInterpolationController

# %%
@click.command()
@click.option('-i', '--input', type=str)
@click.option('-o', '--output', type=str)
@click.option('-s', '--init_pose_sec', type=float, default=3.0)
@click.option('--slow_down', type=float, default=1.0)
def main(input, output, init_pose_sec, slow_down):
    # load data
    input_data = pickle.load(open(os.path.expanduser(input), 'rb'))
    tcp_pose = input_data['tcp_pose']
    timestamp = input_data['timestamp'] * slow_down
    
    duration_sec = timestamp[-1]
    avg_fps = 1 / np.mean(np.diff(timestamp))
    print(f'Input total {duration_sec}sec @ {avg_fps}fps')

    robot_ip = 'ur-2017356986.internal.tri.global'
    max_pos_speed = 0.25 * 10
    max_rot_speed = 0.6 * 10
    cube_diag = np.linalg.norm([1,1,1])
    dt = 1 / avg_fps
    print(f'Robot controlling at {avg_fps}fps')
    robot_state_freq = 125
    get_max_k = int(robot_state_freq * duration_sec * 1.5)

    with SharedMemoryManager() as shm_manager:
        with RTDEInterpolationController(
                shm_manager=shm_manager,
                robot_ip=robot_ip,
                # lookahead_time=0.1,
                lookahead_time=0.05,
                gain=1000,
                max_pos_speed=max_pos_speed*cube_diag,
                max_rot_speed=max_rot_speed*cube_diag,
                tcp_offset_pose=[0,0,0,0,0,0],
                get_max_k=get_max_k,
                verbose=False) as controller:
            
            click.confirm('Start init pose?', abort=True)
            # move to first pose
            init_pose = tcp_pose[0]
            target_timestamp = time.time() + init_pose_sec
            controller.schedule_waypoint(init_pose, target_timestamp)
            precise_sleep(init_pose_sec+0.1)
            click.confirm('Is the robot ready?', abort=True)

            # move the robot
            command_lookahead = 0.1
            t_start = time.time()
            target_pose_traj = list()
            target_timestamps = list()

            for iter_idx in range(len(timestamp)):
                t_cycle_end = t_start + (iter_idx + 1) * dt
                
                this_t = timestamp[iter_idx] - timestamp[0]
                this_t_target = this_t + command_lookahead + t_start

                this_pose = tcp_pose[iter_idx]
                controller.schedule_waypoint(
                    this_pose,
                    this_t_target
                )
                target_pose_traj.append(this_pose)
                target_timestamps.append(this_t_target)

                # sleep
                precise_wait(t_cycle_end, time_func=time.time)

            robot_state = controller.get_all_state()

            result = {
                'target_tcp_pose': np.array(target_pose_traj),
                'target_timestamp': np.array(target_timestamps),
                'actual_tcp_pose': robot_state['ActualTCPPose'],
                'actual_timestamp': robot_state['robot_receive_timestamp']
            }
            print(f'Saving results to {output}')
            pickle.dump(result, open(output, 'wb'))            

# %%
if __name__ == '__main__':
    main()
