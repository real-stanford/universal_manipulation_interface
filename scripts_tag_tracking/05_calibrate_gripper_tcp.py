"""
Calibrate each gripper's TCP relative to its tag cube, without CAD,
see umi.tag_tracking.gripper_calibration:
    T_cube_tcp = T_cube_wrist * T_wrist_tcp
T_cube_wrist from the gripper's finger tags seen by both the head camera
and the gripper's wrist camera in audio synchronized frames; only finger
tags unique to the gripper are used (a tag id on two grippers is ambiguous
in the head view). T_wrist_tcp from the two finger tags in the wrist camera.

Requires: 00-03 (head tag corners, body calibration, tracking),
04_detect_aruco.py of the SLAM pipeline (wrist camera tag detections),
06a_estimate_audio_sync.py (audio_sync_offsets.json), and a gripper
config referenced by the tag tracking config.
Writes <session>/tag_tracking/gripper_tcp.json

python scripts_tag_tracking/05_calibrate_gripper_tcp.py -i data_workspace/session -c umi_calib/tag_tracking.yaml
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
import collections
import click
import numpy as np
from umi.tag_tracking.config import load_tag_tracking_config
from umi.tag_tracking.noise_model import TagNoiseModel
from umi.tag_tracking.session import get_session_videos, overlapping
from umi.tag_tracking.uncertainty import pose_std
from umi.tag_tracking.gripper_calibration import (
    wrist_tag_poses,
    calibrate_cube_to_wrist,
    wrist_to_tcp
)


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Session directory')
@click.option('-c', '--config', required=True, help='Tag tracking config yaml')
def main(input, config):
    session = pathlib.Path(os.path.expanduser(input)).absolute()
    config = load_tag_tracking_config(config)
    assert 'gripper_hardware' in config, "tag tracking config needs gripper_config"
    assert 'sigma_px' in config, "run 01_measure_corner_noise.py first (wrist corner noise)"
    hardware = config['gripper_hardware']
    head_camera = config['cameras'][config['head_camera']]
    calibration = json.load(session.joinpath('tag_tracking', 'bodies.json').open('r'))
    noise_model = TagNoiseModel.from_dict(calibration['noise_model'])
    videos = get_session_videos(session)
    head_videos = [x for x in videos if x['camera_serial'] == config['head_camera']]

    # finger tags used by exactly one gripper
    finger_tag_count = collections.Counter(t for g in hardware['grippers'].values()
        for t in (g['left_finger_tag_id'], g['right_finger_tag_id']))

    output = dict()
    for name, body in config['bodies'].items():
        if 'gripper_camera' not in body:
            continue
        serial = body['gripper_camera']
        gripper = hardware['grippers'][serial]
        wrist_camera = config['cameras'][serial]
        finger_tags = (gripper['left_finger_tag_id'], gripper['right_finger_tag_id'])
        unique_tags = [t for t in finger_tags if finger_tag_count[t] == 1]
        assert len(unique_tags) > 0, f"{name}: no finger tag unique to this gripper"

        # synchronized pairs: head frame with tracked cube + finger tag, wrist frame with the tag
        pairs = collections.defaultdict(list)
        wrist_poses_all = {t: list() for t in finger_tags}
        for head in head_videos:
            corners = pickle.load(head['dir'].joinpath('tag_corners.pkl').open('rb'))
            track = pickle.load(head['dir'].joinpath('tag_tracking.pkl').open('rb'))['bodies'][name]
            for wrist in overlapping(videos, head, serial):
                detections = pickle.load(wrist['dir'].joinpath('tag_detection.pkl').open('rb'))
                wrist_times = wrist['start_timestamp'] + np.array([x['time'] for x in detections])
                wrist_period = float(1 / wrist['fps'])
                for tag_id in finger_tags:
                    poses = wrist_tag_poses(wrist_camera, detections, tag_id,
                        config['tag_sizes'][tag_id], config['sigma_px'])
                    wrist_poses_all[tag_id].append(poses)
                    if tag_id not in unique_tags:
                        continue
                    for i, frame in enumerate(corners):
                        if (not track['is_valid'][i]) or (tag_id not in frame['corners']):
                            continue
                        t = head['start_timestamp'] + frame['time']
                        j = int(np.argmin(np.abs(wrist_times - t)))
                        if abs(wrist_times[j] - t) > wrist_period / 2 or np.isnan(poses[j, 0, 0]):
                            continue
                        pairs[tag_id].append({
                            'T_head_cube': track['T_cam_body'][i],
                            'cube_covariance': track['covariance'][i],
                            'head_corners': frame['corners'][tag_id],
                            'T_wrist_tag': poses[j]
                        })

        # finger tags not in the noise model get their scale with the shared exponent
        for tag_id in unique_tags:
            if tag_id not in noise_model.tag_scales:
                noise_model.tag_scales[tag_id] = noise_model.fit_tag_scale(head_camera,
                    [x['head_corners'] for x in pairs[tag_id]], config['tag_sizes'][tag_id])

        # use the unique finger tag with most synchronized observations
        tag_id = max(unique_tags, key=lambda x: len(pairs[x]))
        result = calibrate_cube_to_wrist(head_camera, pairs[tag_id],
            config['tag_sizes'][tag_id], noise_model, tag_id)
        position_std, rotation_std = pose_std(result['T_cube_wrist'], result['covariance'])

        tcp = wrist_to_tcp(
            np.concatenate(wrist_poses_all[gripper['left_finger_tag_id']]),
            np.concatenate(wrist_poses_all[gripper['right_finger_tag_id']]),
            gripper.get('finger_tag_to_tip'), hardware['nominal_z'])
        T_cube_tcp = result['T_cube_wrist'] @ tcp['T_wrist_tcp']

        print(f"{name} (camera {serial}): finger tag {tag_id}, {result['n_pairs']} synchronized frames, "
            f"{result['fraction_consistent'] * 100:.0f}% consistent, variance factor {result['variance_factor']:.2f}")
        print(f"  T_cube_wrist std {position_std * 1000:.2f} mm, {np.degrees(rotation_std):.3f} deg, "
            f"wrist camera {np.linalg.norm(result['T_cube_wrist'][:3, 3]) * 1000:.1f} mm from cube origin")
        print(f"  TCP in wrist camera {np.round(tcp['T_wrist_tcp'][:3, 3] * 1000, 1)} mm "
            f"from {tcp['n_frames']} frames (MAD {np.round(np.array(tcp['midpoint_mad_m']) * 1000, 2)} mm)")
        if tcp['finger_tag_to_tip'] is None:
            print("  finger_tag_to_tip not in gripper config: TCP at the finger tags, not the fingertips")

        output[name] = {
            'gripper_camera': serial,
            'robot_index': body['robot_index'],
            'finger_tag': tag_id,
            'T_cube_wrist': result['T_cube_wrist'].tolist(),
            'T_cube_wrist_covariance': result['covariance'].tolist(),
            'T_wrist_tcp': tcp['T_wrist_tcp'].tolist(),
            'T_cube_tcp': T_cube_tcp.tolist(),
            'finger_tag_to_tip': tcp['finger_tag_to_tip'],
            'stats': {k: v for k, v in result.items() if k not in ('T_cube_wrist', 'covariance')}
        }

    path = session.joinpath('tag_tracking', 'gripper_tcp.json')
    json.dump(output, path.open('w'), indent=2)
    print(f"Saved to {path}")

# %%
if __name__ == "__main__":
    main()
