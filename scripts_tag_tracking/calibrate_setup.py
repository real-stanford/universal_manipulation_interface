"""
Once per setup (same tags, cubes, grippers, camera settings): calibrate
everything sessions share, see umi.tag_tracking.setup_calibration.

From one session of the setup, processed up to segment_episodes
(run_tag_tracking_pipeline.py --stop_after segment_episodes):
1. environment tag map from the head camera walk past the tags
   (demos/head_mapping_*/tag_corners.pkl, all frames);
2. cube shapes from the head task video(s), at most per_pair frames per pair
   of cube tags seen together (rarely seen pairs keep all their frames);
3. corner noise model and rolling shutter line delay from those frames
   (readout time from the head video's GoPro metadata SensorReadoutTime, only
   the readout direction is estimated);
4. each cube to its wrist camera: wrist frames with >= 2 environment tags
   (wrist demos/*/tag_detection.pkl) and the cube tracked by the head camera
   at the same time (audio_sync_offsets.json);
5. wrist camera to TCP from the gripper geometry (gripper config
   cam_to_center_height / cam_to_mount_offset / tcp_offset, per gripper under
   grippers.<camera serial> or shared at the top level, defaults as UMI;
   --tcp_offset overrides the config for all grippers).

--tcp_only: only redo 5. in an existing setup calibration (<output>/gripper_tcp.json,
report.json), e.g. after measuring the gripper geometry; no session needed.

Writes <output>/bodies.json (as 02_calibrate_bodies.py), <output>/gripper_tcp.json
(T_cube_tcp per gripper, as 05_calibrate_gripper_tcp.py) and <output>/report.json.
Sessions use it with 'setup_calibration: <output>' in the tag tracking config
(run_tag_tracking_pipeline.py step use_setup copies both files into the session).

Bodies are calibrated in parallel and tracking passes are split over
--num_workers processes (default all cores).

python scripts_tag_tracking/calibrate_setup.py -i data_workspace/session -c umi_calib/tag_tracking.yaml -o umi_calib/setup_20261116 -j 24
python scripts_tag_tracking/calibrate_setup.py -c umi_calib/tag_tracking.yaml -o umi_calib/setup_20261116 --tcp_only
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import json
import multiprocessing
import pathlib
import pickle
import time
import click
import numpy as np
from umi.common.exiftool_util import ExifTool
from umi.tag_tracking.config import load_tag_tracking_config
from umi.tag_tracking.session import get_session_videos, overlapping
from umi.tag_tracking.parallel import make_executor, CHUNKS_PER_WORKER
from umi.tag_tracking.setup_calibration import (frames_with_neighbors, select_cube_frames, calibrate_bodies,
    cube_to_wrist, wrist_to_tcp_umi, UMI_CAM_TO_CENTER_HEIGHT, UMI_CAM_TO_MOUNT_OFFSET, UMI_TCP_OFFSET)


# %%
@click.command()
@click.option('-i', '--input', default=None, help='Session directory, processed up to segment_episodes')
@click.option('-c', '--config', required=True, help='Tag tracking config yaml')
@click.option('-o', '--output', required=True, help='Setup calibration directory to write')
@click.option('--per_pair', type=int, default=300, help='Max head frames per pair of cube tags')
@click.option('--max_samples', type=int, default=600, help='Max wrist frames per gripper for cube to wrist')
@click.option('--max_speed', type=float, default=0.3, help='Max cube speed (m/s) for cube to wrist samples')
@click.option('--tcp_offset', type=float, default=None,
    help='Gripper tip to mounting screw (m), default gripper config tcp_offset, else UMI 0.205')
@click.option('--tcp_only', is_flag=True, default=False, help='Only recompute wrist to TCP in an existing calibration')
@click.option('--readout_ms', type=float, default=None, help='Sensor readout time (ms), default from the head video metadata')
@click.option('--search_line_delay', is_flag=True, default=False, help='Ignore the readout time, search the line delay')
@click.option('-j', '--num_workers', type=int, default=None, help='Worker processes, default all cores')
def main(input, config, output, per_pair, max_samples, max_speed, tcp_offset, tcp_only, readout_ms,
        search_line_delay, num_workers):
    t_start = time.time()
    output = pathlib.Path(os.path.expanduser(output)).absolute()
    config = load_tag_tracking_config(config)
    assert 'gripper_hardware' in config, "tag tracking config needs gripper_config"
    hardware = config['gripper_hardware']
    bodies = config['bodies']
    gripper_names = [name for _, name in sorted((b['robot_index'], name) for name, b in bodies.items() if 'gripper_camera' in b)]
    # wrist camera to TCP per gripper: its own geometry, else the shared one, else UMI
    T_wrist_tcps = dict()
    for name in gripper_names:
        gripper = hardware['grippers'].get(bodies[name]['gripper_camera'], dict())
        get = lambda key, default: gripper.get(key, hardware.get(key, default))
        T_wrist_tcps[name] = wrist_to_tcp_umi(get('cam_to_center_height', UMI_CAM_TO_CENTER_HEIGHT),
            get('cam_to_mount_offset', UMI_CAM_TO_MOUNT_OFFSET),
            get('tcp_offset', UMI_TCP_OFFSET) if tcp_offset is None else tcp_offset)
        print(f"{name}: TCP in the wrist camera frame {np.round(T_wrist_tcps[name][:3, 3], 5).tolist()} m")
    if tcp_only:
        tcp_path, report_path = output.joinpath('gripper_tcp.json'), output.joinpath('report.json')
        assert tcp_path.is_file(), f"{tcp_path} missing: run the full setup calibration first"
        tcp_output = json.load(tcp_path.open('r'))
        for name, x in tcp_output.items():
            x['T_wrist_tcp'] = T_wrist_tcps[name].tolist()
            x['T_cube_tcp'] = (np.array(x['T_cube_wrist']) @ T_wrist_tcps[name]).tolist()
        json.dump(tcp_output, tcp_path.open('w'), indent=2)
        if report_path.is_file():
            report = json.load(report_path.open('r'))
            report['T_wrist_tcp'] = {name: T.tolist() for name, T in T_wrist_tcps.items()}
            json.dump(report, report_path.open('w'), indent=2)
        print(f"Updated T_wrist_tcp / T_cube_tcp of {', '.join(tcp_output)} in {tcp_path}")
        return

    assert input is not None, "--input session required (unless --tcp_only)"
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    n_chunks = num_workers * CHUNKS_PER_WORKER
    session = pathlib.Path(os.path.expanduser(input)).absolute()
    assert 'sigma_px' in config, "run 01_measure_corner_noise.py first (wrist corner noise)"
    head_camera = config['cameras'][config['head_camera']]
    static_name = [k for k, v in bodies.items() if v['static']][0]

    # 1. environment: head camera walk past the tags, all frames
    mapping_paths = sorted(session.glob('demos/head_mapping_*/tag_corners.pkl'))
    assert len(mapping_paths) > 0, "no demos/head_mapping_*/tag_corners.pkl (raw_videos/head_mapping/ + head_tags step)"
    env_frames = list()
    for path in mapping_paths:
        frames = pickle.load(path.open('rb'))
        env_frames += frames_with_neighbors(frames, range(len(frames)), path.parent.name)[0]
    print(f"environment: {len(env_frames)} frames from {[p.parent.name for p in mapping_paths]}")

    # 2. cubes: head task videos, balanced over tag pairs
    videos = get_session_videos(session)
    heads = [x for x in videos if x['camera_serial'] == config['head_camera']]
    assert len(heads) == 1, f"one head task video expected for the setup calibration, found {len(heads)}"
    head = heads[0]
    head_frames = pickle.load(head['dir'].joinpath('tag_corners.pkl').open('rb'))
    head_times = np.array([f['time'] for f in head_frames])
    body_frames = {static_name: env_frames}
    for name in gripper_names:
        idx = select_cube_frames(head_frames, bodies[name]['tags'], per_pair)
        body_frames[name] = frames_with_neighbors(head_frames, idx, head['name'])[0]
        print(f"{name}: {len(idx)} frames with >= 2 cube tags (+ neighbors: {len(body_frames[name])})")

    # 3. bodies, noise model, line delay
    readout_time = None
    if not search_line_delay:
        if readout_ms is None:
            with ExifTool() as et:
                meta = list(et.get_metadata(str(head['dir'].joinpath('raw_video.mp4'))))[0]
            readout_ms = meta.get('QuickTime:SensorReadoutTime')
            print(f"head camera sensor readout time from metadata: {readout_ms} ms")
        if readout_ms not in (None, ''):
            readout_time = float(readout_ms) / 1000
    print(f"calibrating bodies with {num_workers} workers")
    executor = make_executor(num_workers)
    noise_model, line_delay, results = calibrate_bodies(head_camera, body_frames,
        {name: bodies[name] for name in body_frames}, config['tag_sizes'], executor, n_chunks, readout_time)
    for name, x in results.items():
        print(f"  {name}: anchor {x['anchor_tag']}, tags {sorted(x['T_body_tags'])}, missing {x['missing_tags']}, "
            f"{x['n_frames_consistent']}/{x['n_frames_used']} multi-tag frames consistent, "
            f"variance factor {x['variance_factor']:.2f}")
    output.mkdir(parents=True, exist_ok=True)
    json.dump({
        'noise_model': noise_model.to_dict(),
        'line_delay': line_delay,
        'bodies': {name: {
            'static': bodies[name]['static'],
            'anchor_tag': x['anchor_tag'],
            'T_body_tags': {str(k): v.tolist() for k, v in x['T_body_tags'].items()},
            'missing_tags': x['missing_tags'],
            'n_frames_used': x['n_frames_used'],
            'n_frames_consistent': x['n_frames_consistent'],
            'variance_factor': x['variance_factor'],
            'tag_rms_px': {str(k): v for k, v in x['tag_rms_px'].items()}
        } for name, x in results.items()}
    }, output.joinpath('bodies.json').open('w'), indent=2)

    # 4. + 5. cube -> wrist camera -> TCP
    T_env_tags = results[static_name]['T_body_tags']
    tcp_output, report_grippers = dict(), dict()
    for name in gripper_names:
        serial = bodies[name]['gripper_camera']
        wrists = overlapping(videos, head, serial)
        assert len(wrists) > 0, f"no {name} wrist video overlapping {head['name']}"
        wrist_frames, wrist_times = list(), list()
        for wrist in wrists:
            detections = pickle.load(wrist['dir'].joinpath('tag_detection.pkl').open('rb'))
            wrist_frames += [{'time': d['time'], 'corners': {k: v['corners'] for k, v in d['tag_dict'].items()}}
                for d in detections]
            wrist_times += [d['time'] + wrist['start_timestamp'] - head['start_timestamp'] for d in detections]
        result = cube_to_wrist(head_camera, head_frames, head_times, T_env_tags, results[name]['T_body_tags'],
            config['tag_sizes'], noise_model, line_delay, config['cameras'][serial], wrist_frames,
            np.array(wrist_times), config['sigma_px'], config['accuracy'], max_samples, max_speed,
            executor=executor, n_chunks=n_chunks)
        assert result is not None and result['T_cube_wrist'] is not None, \
            f"{name}: too few synchronized frames for cube to wrist ({None if result is None else result['stats']})"
        stats = result['stats']
        T_cube_wrist = result['T_cube_wrist']
        print(f"{name} (camera {serial}): cube to wrist from {stats['n_inliers']}/{stats['n_estimates']} frames "
            f"(of {stats['n_candidates']} wrist frames with >= 2 environment tags), spread {stats['spread_mm']:.1f} mm "
            f"{stats['spread_deg']:.2f} deg, standard error {stats['std_error_mm']:.2f} mm {stats['std_error_deg']:.3f} deg, "
            f"wrist camera {stats['distance_cube_to_wrist_mm']:.0f} mm from cube origin")
        tcp_output[name] = {
            'gripper_camera': serial,
            'robot_index': bodies[name]['robot_index'],
            'method': 'cube to wrist from environment tags seen by the wrist camera; wrist to TCP from gripper geometry',
            'T_cube_wrist': T_cube_wrist.tolist(),
            'T_wrist_tcp': T_wrist_tcps[name].tolist(),
            'T_cube_tcp': (T_cube_wrist @ T_wrist_tcps[name]).tolist(),
            'stats': stats
        }
        report_grippers[name] = stats
    json.dump(tcp_output, output.joinpath('gripper_tcp.json').open('w'), indent=2)

    json.dump({
        'source_session': str(session),
        'head_mapping_videos': [p.parent.name for p in mapping_paths],
        'head_task_video': head['name'],
        'n_frames': {name: len(f) for name, f in body_frames.items()},
        'missing_tags': {name: x['missing_tags'] for name, x in results.items()},
        'variance_factor': {name: x['variance_factor'] for name, x in results.items()},
        'line_delay_us': line_delay * 1e6,
        'readout_ms': None if readout_time is None else readout_time * 1000,
        'T_wrist_tcp': {name: T.tolist() for name, T in T_wrist_tcps.items()},
        'cube_to_wrist': report_grippers
    }, output.joinpath('report.json').open('w'), indent=2)
    executor.shutdown()
    print(f"Saved {output}/bodies.json, gripper_tcp.json, report.json ({time.time() - t_start:.0f}s)")

# %%
if __name__ == "__main__":
    main()
