"""
Per frame pose of every calibrated body in each head camera task video
(demos/demo_*; the head_mapping walk is only used for calibration).
The static body (environment) gives the head camera pose in the world;
moving bodies (gripper cubes) are expressed in the head camera frame
and, where the head pose is known, in the world frame. Covariances are
propagated to the world frame; a pose is valid if its uncertainty meets
the accuracy in the config.
With episodes (tag_tracking/episodes.json from 00b_segment_episodes.py), only
frames within --margin seconds of an episode are tracked (others NaN), unless
--whole_take. Frames are tracked in parallel on --num_workers processes.
Writes tag_tracking.pkl next to each tag_corners.pkl, with the tracked frame
windows ('windows', used by 03b_fuse_head_imu.py).

python scripts_tag_tracking/03_track_bodies.py -i data_workspace/session -c umi_calib/tag_tracking.yaml
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
from umi.tag_tracking.parallel import make_executor, parallel_track_rigid_body, CHUNKS_PER_WORKER
from umi.tag_tracking.windows import episode_windows, window_frames, scatter
from umi.tag_tracking.noise_model import TagNoiseModel
from umi.tag_tracking.uncertainty import (
    inverse_covariance,
    compose_covariance,
    pose_std
)


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Session directory')
@click.option('-c', '--config', required=True, help='Tag tracking config yaml')
@click.option('-m', '--margin', type=float, default=0.5, help='Track this long before / after each episode (s)')
@click.option('-wt', '--whole_take', is_flag=True, default=False, help='Track all frames, ignore episodes.json')
@click.option('-j', '--num_workers', type=int, default=None, help='Worker processes, default all cores')
def main(input, config, margin, whole_take, num_workers):
    session = pathlib.Path(os.path.expanduser(input)).absolute()
    config = load_tag_tracking_config(config)
    camera = config['cameras'][config['head_camera']]
    calibration = json.load(session.joinpath('tag_tracking', 'bodies.json').open('r'))
    noise_model = TagNoiseModel.from_dict(calibration['noise_model'])
    line_delay = calibration['line_delay']

    episodes = None
    episodes_path = session.joinpath('tag_tracking', 'episodes.json')
    if not whole_take and 'episodes' in config:
        assert episodes_path.is_file(), "run 00b_segment_episodes.py first, or --whole_take"
        episodes = json.load(episodes_path.open('r'))['videos']
    executor = make_executor(num_workers)
    n_chunks = executor._max_workers * CHUNKS_PER_WORKER

    for corner_path in sorted(session.glob('demos/demo_*/tag_corners.pkl')):
        detections = pickle.load(corner_path.open('rb'))
        frames = [{'time': x['time'], 'corners': x['corners'], 'sequence': 0} for x in detections]
        times = np.array([x['time'] for x in detections])
        # frames to track: windows around the episodes of this video (whole video if none)
        video_episodes = None
        if episodes is not None and corner_path.parent.name in episodes:
            video_episodes = episodes[corner_path.parent.name]['episodes']
        windows = episode_windows(times, video_episodes, margin)
        selected, idx = window_frames(frames, windows)
        print(f"{corner_path.parent.name}: tracking {len(selected)} of {len(frames)} frames "
            f"in {len(windows)} windows")
        result = {
            'time': times,
            'camera_serial': config['head_camera'],
            'windows': windows,
            'bodies': dict()
        }
        accuracy = config['accuracy']
        is_accurate = lambda position_std, rotation_std: (position_std <= accuracy['position_std']) \
            & (rotation_std <= accuracy['rotation_std_rad'])
        n = len(frames)

        static_name = None
        for name, body in calibration['bodies'].items():
            T_body_tags = {int(k): np.array(v) for k, v in body['T_body_tags'].items()}
            track = scatter(len(frames), idx, parallel_track_rigid_body(camera, selected, T_body_tags,
                config['tag_sizes'], noise_model, line_delay, executor, n_chunks))
            # body pose in the camera frame accepted if its uncertainty meets the accuracy
            track['is_valid'] = is_accurate(track['position_std'], track['rotation_std'])
            result['bodies'][name] = track
            if body['static']:
                static_name = name

        if static_name is not None:
            # head camera pose in the world (static body frame)
            env = result['bodies'][static_name]
            head = {
                'T_world_cam': np.full((n, 4, 4), np.nan),
                'covariance': np.full((n, 6, 6), np.nan),
                'position_std': np.full(n, np.nan),
                'rotation_std': np.full(n, np.nan)
            }
            for i in np.flatnonzero(~np.isnan(env['T_cam_body'][:, 0, 0])):
                head['T_world_cam'][i] = np.linalg.inv(env['T_cam_body'][i])
                head['covariance'][i] = inverse_covariance(env['T_cam_body'][i], env['covariance'][i])
                head['position_std'][i], head['rotation_std'][i] = pose_std(
                    head['T_world_cam'][i], head['covariance'][i])
            head['is_valid'] = is_accurate(head['position_std'], head['rotation_std'])
            result['head'] = head

            # moving bodies in the world, head and body uncertainty combined
            for name, track in result['bodies'].items():
                if name == static_name:
                    continue
                track['T_world_body'] = np.full((n, 4, 4), np.nan)
                track['world_covariance'] = np.full((n, 6, 6), np.nan)
                track['world_position_std'] = np.full(n, np.nan)
                track['world_rotation_std'] = np.full(n, np.nan)
                both = ~np.isnan(head['T_world_cam'][:, 0, 0]) & ~np.isnan(track['T_cam_body'][:, 0, 0])
                for i in np.flatnonzero(both):
                    track['T_world_body'][i] = head['T_world_cam'][i] @ track['T_cam_body'][i]
                    track['world_covariance'][i] = compose_covariance(head['T_world_cam'][i],
                        head['covariance'][i], track['T_cam_body'][i], track['covariance'][i])
                    track['world_position_std'][i], track['world_rotation_std'][i] = pose_std(
                        track['T_world_body'][i], track['world_covariance'][i])
                track['is_valid_world'] = is_accurate(track['world_position_std'], track['world_rotation_std'])

        tag_ids = lambda name: {int(k) for k in calibration['bodies'][name]['T_body_tags']}
        # statistics over the tracked frames
        tag_visible = lambda name: np.mean([len(tag_ids(name) & set(f['corners'])) > 0 for f in selected]) * 100
        for name, track in result['bodies'].items():
            if name == static_name:
                continue
            estimated = ~np.isnan(track['T_cam_body'][:, 0, 0])
            print(f"{corner_path.parent.name} {name}: tag visible {tag_visible(name):.0f}%, "
                f"estimated {estimated[idx].mean() * 100:.0f}%, valid in head frame {track['is_valid'][idx].mean() * 100:.0f}% "
                f"(median std {np.nanmedian(track['position_std']) * 1000:.1f} mm, "
                f"{np.degrees(np.nanmedian(track['rotation_std'])):.2f} deg)"
                + (f", valid in world {track['is_valid_world'][idx].mean() * 100:.0f}%" if 'is_valid_world' in track else ''))
        if static_name is not None:
            head = result['head']
            print(f"{corner_path.parent.name} head in world: tag visible {tag_visible(static_name):.0f}%, "
                f"estimated {(~np.isnan(head['T_world_cam'][idx, 0, 0])).mean() * 100:.0f}%, valid {head['is_valid'][idx].mean() * 100:.0f}% "
                f"(median std {np.nanmedian(head['position_std']) * 1000:.1f} mm, "
                f"{np.degrees(np.nanmedian(head['rotation_std'])):.2f} deg)")
        output = corner_path.parent.joinpath('tag_tracking.pkl')
        pickle.dump(result, output.open('wb'))
        print(f"Saved to {output}")
    executor.shutdown()

# %%
if __name__ == "__main__":
    main()
