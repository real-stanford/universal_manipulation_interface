"""
Clock sync of the wrist camera videos to the head camera video from motion
(umi.tag_tracking.wrist_sync): per episode, the cube rotation tracked by the
head camera against the wrist gyro; per wrist video, a drift line through
the episodes' corrections. 03c_fuse_gripper_imu.py, 06_export_dataset_plan.py
and 07_export_mcap.py add the line's correction to the wrist videos' times.

Head rolling shutter readout: the head video's SensorReadoutTime (GoPro
metadata), or --readout_ms. Also fitted from the episodes (drift lines + cube
image rows) as a check; its uncertainty enters the head-to-wrist accuracy.

Requires: tag_tracking/episodes.json (00b), tag_tracking.pkl (03, 03b),
tag_tracking/gripper_tcp.json, wrist demos/*/imu_data.json, gripper_imu in the config.
Writes <session>/tag_tracking/wrist_sync.json:
{
  "parameters": {...},
  "videos": {"<head video>": {"<wrist video>": {"gripper", "robot_index",
      "line": {"offset_ms", "drift_ms_per_min", "residual_ms", "n_episodes", "n_inliers",
               "mean_row", "episodes": [{"start", "end", "correction_ms", "sigma_ms", ...}]} or null,
      "accuracy_ms"}}},
  "readout": {"<head video>": {"used_ms", "source", "fit": {"readout_ms", "sigma_ms", ...} or null}},
  "pairs": {"<head video>": [{"a", "b", "accuracy_ms"}]}
}
correction(t) = offset_ms + drift_ms_per_min * t / 60 (t: head video seconds), added to the wrist video's times.
accuracy: head to wrist sqrt(residual^2 + (readout sigma * mean cube row)^2);
wrist to wrist sqrt(residual_a^2 + residual_b^2 + (readout sigma * row difference)^2).

python scripts_tag_tracking/03s_sync_wrists.py -i data_workspace/session -c umi_calib/tag_tracking.yaml
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
from umi.tag_tracking.session import get_session_videos, overlapping
from umi.tag_tracking.imu import load_imu, load_cam_imu_calibration, wrist_imu_time_offset
from umi.tag_tracking.wrist_sync import (DEFAULTS, cube_image_rows, sync_episodes, drift_line, fit_readout,
    observation_times, head_readout)


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Session directory')
@click.option('-c', '--config', required=True, help='Tag tracking config yaml')
@click.option('-o', '--output', default=None, help='Default <input>/tag_tracking/wrist_sync.json')
@click.option('--readout_ms', type=float, default=None, help='Head rolling shutter readout (ms), default from the metadata')
@click.option('--interval_frames', type=int, default=DEFAULTS['interval_frames'], help='Head frames per rotation rate')
@click.option('--search_ms', type=float, default=DEFAULTS['search_ms'], help='Correction search range +- (ms)')
def main(input, config, output, readout_ms, interval_frames, search_ms):
    session = pathlib.Path(os.path.expanduser(input)).absolute()
    config = load_tag_tracking_config(config)
    assert 'gripper_imu' in config, "wrist sync needs gripper_imu (camera-IMU calibration) in the config"
    head_camera = config['cameras'][config['head_camera']]
    height = head_camera.image_size[1]
    params = {**DEFAULTS, 'interval_frames': interval_frames, 'search_ms': search_ms}
    tcp = json.load(session.joinpath('tag_tracking', 'gripper_tcp.json').open('r'))
    episodes = json.load(session.joinpath('tag_tracking', 'episodes.json').open('r'))['videos']
    names = [name for _, name in sorted((b['robot_index'], name)
        for name, b in config['bodies'].items() if 'gripper_camera' in b)]
    videos = get_session_videos(session)
    # line delay the head tracking corrected (setup calibration): fixes when a tracked pose was observed
    line_delay = json.load(session.joinpath('tag_tracking', 'bodies.json').open('r'))['line_delay']

    out = {'parameters': params, 'videos': dict(), 'readout': dict(), 'pairs': dict()}
    # head videos 00b_segment_episodes.py skipped (no wrist video overlapping) have no episodes
    for head in [x for x in videos if x['camera_serial'] == config['head_camera'] and x['name'] in episodes]:
        track = pickle.load(head['dir'].joinpath('tag_tracking.pkl').open('rb'))
        times = track['time']
        head_episodes = episodes[head['name']]['episodes']
        readout = head_readout(head) * 1e3 if readout_ms is None else readout_ms
        source = 'metadata SensorReadoutTime' if readout_ms is None else '--readout_ms'
        print(f"{head['name']}: {len(head_episodes)} episodes, head readout {readout:.1f} ms ({source}), "
            f"tracking line delay {line_delay * 1e6:.3f} us")

        results, readout_samples = dict(), list()
        for name in names:
            body = track['bodies'][name]
            serial = config['bodies'][name]['gripper_camera']
            cam_imu = load_cam_imu_calibration(config['gripper_imu'][serial]['cam_imu_calibration'])
            # gyro (IMU frame) -> wrist camera frame -> cube frame
            R_cube_imu = np.array(tcp[name]['T_cube_wrist'])[:3, :3] @ cam_imu['T_imu_cam'][:3, :3].T
            T = body['T_world_body']
            valid = body['is_valid_world'] & ~np.isnan(T[:, 0, 0])
            rows = cube_image_rows(body['T_cam_body'], head_camera, height)
            obs = observation_times(times, rows, readout * 1e-3, line_delay, height)
            obs0 = observation_times(times, rows, 0.0, line_delay, height)
            for wrist in overlapping(videos, head, serial):
                imu = load_imu(str(wrist['dir'].joinpath('imu_data.json')))
                # wrist IMU -> wrist video (OpenICC offset) -> head video clock (audio sync start times)
                time_offset = wrist_imu_time_offset(config['gripper_imu'][serial], cam_imu,
                    str(wrist['dir'].joinpath('raw_video.mp4')), float(1 / wrist['fps']))
                print(f"  {name} {wrist['name']}: IMU time offset {time_offset * 1e3:+.2f} ms "
                    f"({config['gripper_imu'][serial].get('imu_time_offset', 'openicc')})")
                gyro_times = imu['time'] + time_offset + wrist['start_timestamp'] - head['start_timestamp']
                gyro_cube = imu['gyro'] @ R_cube_imu.T
                fits = sync_episodes(times, obs, rows, T, valid, head_episodes, gyro_times, gyro_cube, params)
                line = drift_line(fits)
                results[wrist['name']] = {'gripper': name, 'robot_index': config['bodies'][name]['robot_index'],
                    'line': line, 'imu_time_offset_ms': time_offset * 1e3, 'start': wrist['start_timestamp'], 'end': wrist['end_timestamp']}
                # readout check: the same episodes without the rolling shutter
                for e, f in sync_episodes(times, obs0, rows, T, valid, head_episodes, gyro_times, gyro_cube, params):
                    if f is not None:
                        readout_samples.append((wrist['name'], 0.5 * (e['start'] + e['end']), f['correction'], f['row']))
                if line is None:
                    print(f"  {name} {wrist['name']}: no episode synchronized, no correction")
                    continue
                print(f"  {name} {wrist['name']}: correction {line['offset_ms']:+.2f} ms {line['drift_ms_per_min']:+.3f} ms/min "
                    f"from {line['n_inliers']} of {line['n_episodes']} episodes, residual RMS {line['residual_ms']:.2f} ms"
                    + (" (constant: fewer than 3 episodes, no drift fitted)" if line.get('constant') else ''))
                print('    per episode (ms, sigma): ' + ', '.join(f"{x['correction_ms']:+.1f}({x['sigma_ms']:.1f})"
                    + ('' if x['inlier'] else '*') for x in line['episodes']))

        fit = fit_readout(readout_samples)
        readout_sigma = None if fit is None else fit['sigma_ms']
        if fit is not None:
            print(f"  readout check: fitted {fit['readout_ms']:+.1f} +- {fit['sigma_ms']:.1f} ms from {fit['n_episodes']} "
                f"episodes (cube rows {fit['row_min']:.2f}-{fit['row_max']:.2f}), used {readout:.1f} ms")
        # accuracy
        for name, r in results.items():
            line = r['line']
            r['accuracy_ms'] = None if line is None or readout_sigma is None else \
                float(np.hypot(line['residual_ms'], readout_sigma * line['mean_row']))
        pairs = list()
        wrist_names = [n for n, r in results.items() if r['line'] is not None]
        for i, a in enumerate(wrist_names):
            for b in wrist_names[i + 1:]:
                ra, rb = results[a], results[b]
                if ra['gripper'] == rb['gripper'] or ra['end'] <= rb['start'] or rb['end'] <= ra['start']:
                    continue
                row_diff = abs(ra['line']['mean_row'] - rb['line']['mean_row'])
                pairs.append({'a': a, 'b': b, 'accuracy_ms': None if readout_sigma is None else float(np.sqrt(
                    ra['line']['residual_ms'] ** 2 + rb['line']['residual_ms'] ** 2 + (readout_sigma * row_diff) ** 2))})
        for r in results.values():
            r.pop('start')
            r.pop('end')
        print("  accuracy: " + ', '.join([f"head-{r['gripper']} {r['accuracy_ms']:.2f} ms ({n.split('_')[1][-6:]})"
            for n, r in results.items() if r['accuracy_ms'] is not None]
            + [f"{results[p['a']]['gripper']}-{results[p['b']]['gripper']} {p['accuracy_ms']:.2f} ms"
            for p in pairs if p['accuracy_ms'] is not None]))
        out['videos'][head['name']] = results
        out['readout'][head['name']] = {'used_ms': readout, 'source': source, 'fit': fit,
            'tracking_line_delay_us': line_delay * 1e6}
        out['pairs'][head['name']] = pairs

    output = session.joinpath('tag_tracking', 'wrist_sync.json') if output is None else pathlib.Path(output)
    json.dump(out, output.open('w'), indent=2)
    print(f"Saved to {output}")

# %%
if __name__ == "__main__":
    main()
