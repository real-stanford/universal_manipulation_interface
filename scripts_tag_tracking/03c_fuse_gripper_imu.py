"""
Gripper (wrist camera) poses by visual-inertial smoothing per episode window,
see umi.tag_tracking.gripper_fusion: wrist IMU + environment tags seen by the
wrist camera + the wrist pose measured by the head camera through the cube.
Bridges frames where the head camera does not see the cube.

Requires: tag_tracking.pkl with world cube poses (03_track_bodies.py,
03b_fuse_head_imu.py), tag_tracking/bodies.json and gripper_tcp.json
(T_cube_wrist, setup calibration), wrist demos/*/imu_data.json and
tag_detection.pkl, audio_sync_offsets.json, and gripper_imu (camera-IMU
calibration and IMU noise per wrist camera) in the tag tracking config.
Wrist clocks: the audio sync start times corrected by tag_tracking/wrist_sync.json
(03s_sync_wrists.py) when present.
Adds to each head task video's tag_tracking.pkl, per gripper body:
grippers[name] = {'time' (head video clock), 'T_world_wrist', 'covariance',
'position_std', 'rotation_std', 'is_valid', 'head_measured', 'n_wrist_tags'}
over the wrist frames in the tracked windows; 06_export_dataset_plan.py uses it.

python scripts_tag_tracking/03c_fuse_gripper_imu.py -i data_workspace/session -c umi_calib/tag_tracking.yaml
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import json
import pathlib
import pickle
import click
import numpy as np
from umi.tag_tracking.config import load_tag_tracking_config
from umi.tag_tracking.noise_model import TagNoiseModel
from umi.tag_tracking.imu import load_imu, load_cam_imu_calibration, load_imu_noise
from umi.tag_tracking.session import get_session_videos, overlapping
from umi.tag_tracking.parallel import make_executor, submit_fuse
from umi.tag_tracking.gripper_fusion import wrist_measurements, init_poses
from umi.tag_tracking.wrist_sync import wrist_corrections, correction_seconds


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Session directory')
@click.option('-c', '--config', required=True, help='Tag tracking config yaml with gripper_imu')
@click.option('--max_head_gap', type=float, default=0.05, help='Head frames around a wrist frame at most this apart (s)')
@click.option('-j', '--num_workers', type=int, default=None, help='Worker processes, default all cores')
def main(input, config, max_head_gap, num_workers):
    session = pathlib.Path(os.path.expanduser(input)).absolute()
    config = load_tag_tracking_config(config)
    assert 'gripper_imu' in config, "tag tracking config needs gripper_imu"
    assert 'sigma_px' in config, "run 01_measure_corner_noise.py first (wrist corner noise)"
    accuracy = config['accuracy']
    is_accurate = lambda p, r: (p <= accuracy['position_std']) & (r <= accuracy['rotation_std_rad'])
    calibration = json.load(session.joinpath('tag_tracking', 'bodies.json').open('r'))
    tcp = json.load(session.joinpath('tag_tracking', 'gripper_tcp.json').open('r'))
    static_name = [k for k, v in calibration['bodies'].items() if v['static']][0]
    T_env_tags = {int(k): np.array(v) for k, v in calibration['bodies'][static_name]['T_body_tags'].items()}
    wrist_noise = TagNoiseModel(0.0, {t: config['sigma_px'] for t in T_env_tags})
    names = [name for _, name in sorted((b['robot_index'], name)
        for name, b in config['bodies'].items() if 'gripper_camera' in b)]
    videos = get_session_videos(session)
    executor = make_executor(num_workers)
    sync_path = session.joinpath('tag_tracking', 'wrist_sync.json')
    sync = json.load(sync_path.open('r')) if sync_path.is_file() else None
    if sync is None:
        print("no tag_tracking/wrist_sync.json (03s_sync_wrists.py): wrist clocks from the audio sync only")

    for head in [x for x in videos if x['camera_serial'] == config['head_camera']]:
        track_path = head['dir'].joinpath('tag_tracking.pkl')
        track = pickle.load(track_path.open('rb'))
        head_times = track['time']
        windows = track.get('windows', [(0, len(head_times))])
        track['grippers'] = dict()
        corrections = wrist_corrections(sync, head['name'])
        for name in names:
            serial = config['bodies'][name]['gripper_camera']
            assert serial in config['gripper_imu'], f"no gripper_imu for camera {serial} ({name})"
            camera = config['cameras'][serial]
            cam_imu = load_cam_imu_calibration(config['gripper_imu'][serial]['cam_imu_calibration'])
            imu_noise = load_imu_noise(config['gripper_imu'][serial]['noise_setting'])
            T_cube_wrist = np.array(tcp[name]['T_cube_wrist'])
            body = track['bodies'][name]

            jobs = list()
            for wrist in overlapping(videos, head, serial):
                detections = pickle.load(wrist['dir'].joinpath('tag_detection.pkl').open('rb'))
                wrist_times = np.array([d['time'] for d in detections])
                # wrist video frames on the head video clock: audio sync start + wrist sync correction
                on_head = wrist_times + wrist['start_timestamp'] - head['start_timestamp']
                if wrist['name'] in corrections:
                    on_head = on_head + correction_seconds(corrections[wrist['name']], on_head)
                elif sync is not None:
                    print(f"  {name} {wrist['name']}: no wrist sync correction, audio sync only")
                imu = load_imu(str(wrist['dir'].joinpath('imu_data.json')))
                for a, b in windows:
                    inside = np.flatnonzero((on_head >= head_times[a]) & (on_head <= head_times[b - 1]))
                    if len(inside) < 2:
                        continue
                    frames = [{'time': float(wrist_times[i]), 'corners': {k: v['corners']
                        for k, v in detections[i]['tag_dict'].items() if k in T_env_tags}} for i in inside]
                    measurements = wrist_measurements(on_head[inside], head_times, body['T_world_body'],
                        body['world_covariance'], body['is_valid_world'], T_cube_wrist, max_head_gap)
                    init = init_poses(len(inside), measurements)
                    if init is None:
                        print(f"  {name} {wrist['name']} window {head_times[a]:.1f}-{head_times[b - 1]:.1f}s: "
                            f"no head measurement, skipped")
                        continue
                    future = submit_fuse(executor, camera, frames, T_env_tags, config['tag_sizes'], wrist_noise,
                        imu, cam_imu['T_imu_cam'], cam_imu['time_offset'], imu_noise, init, measurements)
                    jobs.append((on_head[inside], measurements, frames, future))

            parts = list()
            n_failed = 0
            for times, measurements, frames, future in jobs:
                try:
                    fused = future.result()
                except Exception as e:
                    n_failed += 1
                    print(f"  {name} window {times[0]:.1f}-{times[-1]:.1f}s: smoothing failed ({e})")
                    continue
                measured = np.zeros(len(times), dtype=bool)
                measured[list(measurements)] = True
                parts.append({
                    'time': times,
                    'T_world_wrist': fused['T_world_cam'],
                    'covariance': fused['covariance'],
                    'position_std': fused['position_std'],
                    'rotation_std': fused['rotation_std'],
                    'head_measured': measured,
                    'n_wrist_tags': np.array([len(f['corners']) for f in frames])
                })
            result = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else None
            if result is None:
                print(f"{head['name']} {name}: no fused windows")
                continue
            order = np.argsort(result['time'])
            result = {k: v[order] for k, v in result.items()}
            result['is_valid'] = is_accurate(result['position_std'], result['rotation_std'])
            track['grippers'][name] = result
            print(f"{head['name']} {name} (camera {serial}): {len(parts)} windows ({n_failed} failed), "
                f"{len(result['time'])} wrist frames: head measured {result['head_measured'].mean() * 100:.0f}%, "
                f"wrist sees >= 1 env tag {(result['n_wrist_tags'] >= 1).mean() * 100:.0f}% -> fused valid "
                f"{result['is_valid'].mean() * 100:.0f}% (median std {np.nanmedian(result['position_std']) * 1000:.1f} mm, "
                f"{np.degrees(np.nanmedian(result['rotation_std'])):.2f} deg)")
        pickle.dump(track, track_path.open('wb'))
        print(f"Saved to {track_path}")
    executor.shutdown()

# %%
if __name__ == "__main__":
    main()
