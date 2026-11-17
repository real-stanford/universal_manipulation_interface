"""
Head camera trajectory by visual-inertial smoothing of the static body's
tags (known poses from 02_calibrate_bodies.py) and the head camera's IMU,
see umi.tag_tracking.visual_inertial. Replaces the per-frame head poses of
03_track_bodies.py in tag_tracking.pkl (kept as 'head_tags_only') and
recomputes the moving bodies in the world frame.
Requires the head camera's imu_data.json (01_extract_gopro_imu.py) and
head_imu in the tag tracking config.
Task videos only (demos/demo_*). Smoothing runs per tracked window of 03_track_bodies.py (the episodes +-
margin, or the whole video), windows in parallel on --num_workers processes.

python scripts_tag_tracking/03b_fuse_head_imu.py -i data_workspace/session -c umi_calib/tag_tracking.yaml
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
from umi.tag_tracking.parallel import make_executor, submit_fuse
from umi.tag_tracking.windows import window_frames, scatter
from umi.tag_tracking.uncertainty import compose_covariance, pose_std


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Session directory')
@click.option('-c', '--config', required=True, help='Tag tracking config yaml')
@click.option('-j', '--num_workers', type=int, default=None, help='Worker processes, default all cores')
def main(input, config, num_workers):
    session = pathlib.Path(os.path.expanduser(input)).absolute()
    config = load_tag_tracking_config(config)
    assert 'head_imu' in config, "tag tracking config needs head_imu"
    camera = config['cameras'][config['head_camera']]
    calibration = json.load(session.joinpath('tag_tracking', 'bodies.json').open('r'))
    noise_model = TagNoiseModel.from_dict(calibration['noise_model'])
    static_name = [k for k, v in calibration['bodies'].items() if v['static']][0]
    T_world_tags = {int(k): np.array(v) for k, v in calibration['bodies'][static_name]['T_body_tags'].items()}
    cam_imu = load_cam_imu_calibration(config['head_imu']['cam_imu_calibration'])
    imu_noise = load_imu_noise(config['head_imu']['noise_setting'])
    accuracy = config['accuracy']
    is_accurate = lambda p, r: (p <= accuracy['position_std']) & (r <= accuracy['rotation_std_rad'])

    executor = make_executor(num_workers)
    for track_path in sorted(session.glob('demos/demo_*/tag_tracking.pkl')):
        video_dir = track_path.parent
        track = pickle.load(track_path.open('rb'))
        corners = pickle.load(video_dir.joinpath('tag_corners.pkl').open('rb'))
        frames = [{'time': x['time'], 'corners': x['corners']} for x in corners]
        imu = load_imu(str(video_dir.joinpath('imu_data.json')))
        # per-frame tag poses of 03_track_bodies.py, also when rerun
        tags_only = track.get('head_tags_only', track['head'])
        windows = track.get('windows', [(0, len(frames))])

        # one smoothing problem per window, windows in parallel
        futures = [submit_fuse(executor, camera, frames[a:b], T_world_tags, config['tag_sizes'],
            noise_model, imu, cam_imu['T_imu_cam'], cam_imu['time_offset'], imu_noise,
            tags_only['T_world_cam'][a:b]) for a, b in windows]
        parts = list()
        for (a, b), future in zip(windows, futures):
            try:
                parts.append(future.result())
            except Exception as e:
                # e.g. too few tags in the window: no fused head pose there
                print(f"  window {frames[a]['time']:.1f}-{frames[b - 1]['time']:.1f}s: smoothing failed ({e}), left invalid")
                parts.append({'T_world_cam': np.full((b - a, 4, 4), np.nan), 'covariance': np.full((b - a, 6, 6), np.nan),
                    'position_std': np.full(b - a, np.nan), 'rotation_std': np.full(b - a, np.nan),
                    'n_tags': np.zeros(b - a, dtype=int), 'gyro_bias': np.full((b - a, 3), np.nan),
                    'accel_bias': np.full((b - a, 3), np.nan), 'variance_factor': np.nan, 'gravity_alignment_deg': np.nan})
        _, idx = window_frames(frames, windows)
        per_frame = ['T_world_cam', 'covariance', 'position_std', 'rotation_std', 'n_tags', 'gyro_bias', 'accel_bias']
        fused = scatter(len(frames), idx, {k: np.concatenate([x[k] for x in parts]) for k in per_frame})
        fused['window_stats'] = [{'variance_factor': x['variance_factor'],
            'gravity_alignment_deg': x['gravity_alignment_deg']} for x in parts]
        fused['is_valid'] = is_accurate(fused['position_std'], fused['rotation_std'])
        variance_factors = [x['variance_factor'] for x in parts]
        gravity = [x['gravity_alignment_deg'] for x in parts]
        print(f"{video_dir.name}: {len(windows)} windows, head valid {tags_only['is_valid'][idx].mean() * 100:.0f}% tags only -> "
            f"{fused['is_valid'][idx].mean() * 100:.0f}% fused (median std {np.nanmedian(fused['position_std']) * 1000:.1f} mm, "
            f"{np.degrees(np.nanmedian(fused['rotation_std'])):.2f} deg), tag variance factor median {np.nanmedian(variance_factors):.2f}, "
            f"world-gravity alignment median {np.nanmedian(gravity):.2f} deg")
        track['head_tags_only'] = tags_only
        track['head'] = fused

        # moving bodies in the world with the fused head poses
        for name, body in track['bodies'].items():
            if name == static_name:
                continue
            n = len(frames)
            for key in ['T_world_body', 'world_covariance']:
                body[key] = np.full((n, 4, 4) if key == 'T_world_body' else (n, 6, 6), np.nan)
            body['world_position_std'] = np.full(n, np.nan)
            body['world_rotation_std'] = np.full(n, np.nan)
            for k in np.flatnonzero(~np.isnan(body['T_cam_body'][:, 0, 0])):
                body['T_world_body'][k] = fused['T_world_cam'][k] @ body['T_cam_body'][k]
                body['world_covariance'][k] = compose_covariance(fused['T_world_cam'][k],
                    fused['covariance'][k], body['T_cam_body'][k], body['covariance'][k])
                body['world_position_std'][k], body['world_rotation_std'][k] = pose_std(
                    body['T_world_body'][k], body['world_covariance'][k])
            body['is_valid_world'] = is_accurate(body['world_position_std'], body['world_rotation_std'])
            print(f"   {name}: valid in world {body['is_valid_world'][idx].mean() * 100:.0f}% "
                f"(valid in head frame {body['is_valid'][idx].mean() * 100:.0f}%)")
        pickle.dump(track, track_path.open('wb'))
        print(f"Saved to {track_path}")
    executor.shutdown()

# %%
if __name__ == "__main__":
    main()
