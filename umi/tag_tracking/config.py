"""
Tag tracking config, all paths relative to the config file:

aruco_config: aruco_config.yaml       # dictionary and tag sizes
corner_noise: corner_noise.json       # from scripts_tag_tracking/01_measure_corner_noise.py
gripper_config: gripper_config.yaml   # optional, finger tags per wrist camera (umi.common.gripper_config_util)
head_camera: C3501326502664           # serial of the camera observing the bodies
setup_calibration: setup_20261116      # optional, bodies.json + gripper_tcp.json from
                                      # scripts_tag_tracking/calibrate_setup.py, used by every session
head_imu:                             # optional, for 03b_fuse_head_imu.py
  cam_imu_calibration: cam_imu_calib_result.json   # OpenICC result of the head camera
  noise_setting: head_setting.yaml    # ORB_SLAM3 setting yaml with IMU.Noise*/IMU.*Walk
gripper_imu:                          # optional, per wrist camera, for 03c_fuse_gripper_imu.py
  C3501326029433:
    cam_imu_calibration: right_cam_imu_calib_result.json
    noise_setting: right_setting.yaml
accuracy:                             # pose accepted if its std is within these
  position_std: 0.0061                # m
  rotation_std: 3.5                   # deg
cameras:                              # serial -> OpenICC intrinsics json
  C3501326502664: head_intrinsics.json
bodies:
  environment:
    static: true                      # fixed in the world, defines the world frame
    tags: [13, 14, 15]
    anchor_tag: 13                    # optional, body frame origin
  left_cube:
    tags: [2, 3, 4, 5]
    gripper_camera: C3501326029433    # optional, wrist camera of this gripper
    robot_index: 1                    # with gripper_camera, UMI order: 0 right, 1 left
episodes:                             # optional, continuous takes, see umi.tag_tracking.wrist_events
  dock_tags: [46, 47]                 # environment tags next to the docked grippers
  max_gap_sec: 0.25                   # longest tracking gap inside an exported episode
  # detection settings, defaults in umi.tag_tracking.wrist_events.DEFAULTS, e.g.
  # gyro_still: 0.04                  # rad/s, gripper still (on the dock)
  # bang_accel: 5.0                   # m/s^2, weaker wrist spike of a bang
  # audio_ratio: 5.0                  # head audio click of a bang
"""
import os
import json
import yaml
import numpy as np
from umi.common.cv_util import parse_aruco_config
from umi.common.gripper_config_util import load_gripper_config
from umi.tag_tracking.camera_model import FisheyeCamera
from umi.tag_tracking.wrist_events import DEFAULTS as WRIST_EVENT_DEFAULTS


# max_gap_sec: export; dock_exit_distance: 03t_trim_docking.py (cube to nearest dock tag, m)
EPISODE_DEFAULTS = {**WRIST_EVENT_DEFAULTS, 'max_gap_sec': 0.25, 'dock_exit_distance': 0.45}


def load_tag_tracking_config(path: str) -> dict:
    path = os.path.abspath(os.path.expanduser(path))
    root = os.path.dirname(path)
    resolve = lambda x: os.path.join(root, os.path.expanduser(x))
    config = yaml.safe_load(open(path, 'r'))

    aruco_yaml = yaml.safe_load(open(resolve(config['aruco_config']), 'r'))
    aruco_config = parse_aruco_config(aruco_yaml)
    config['aruco_dict'] = aruco_config['aruco_dict']
    # tags with an explicitly configured size, others are not in the setup
    config['known_tag_ids'] = {int(k) for k in aruco_yaml['marker_size_map'] if k != 'default'}
    config['cameras'] = {str(k): FisheyeCamera(resolve(v)) for k, v in config['cameras'].items()}
    config['head_camera'] = str(config['head_camera'])
    config['accuracy']['rotation_std_rad'] = float(np.radians(config['accuracy']['rotation_std']))
    assert config['head_camera'] in config['cameras']

    all_tags = list()
    for name, body in config['bodies'].items():
        body.setdefault('static', False)
        body.setdefault('anchor_tag', None)
        body['tags'] = [int(x) for x in body['tags']]
        all_tags.extend(body['tags'])
    assert len(all_tags) == len(set(all_tags)), "each tag id must belong to one body"
    assert set(all_tags) <= config['known_tag_ids'], "body tags need a size in aruco_config"
    assert sum(b['static'] for b in config['bodies'].values()) <= 1, "at most one static body"
    config['tag_sizes'] = {x: aruco_config['marker_size_map'][x] for x in config['known_tag_ids']}

    if 'gripper_config' in config:
        config['gripper_hardware'] = load_gripper_config(resolve(config['gripper_config']))
    gripper_bodies = {k: v for k, v in config['bodies'].items() if 'gripper_camera' in v}
    for name, body in gripper_bodies.items():
        body['gripper_camera'] = str(body['gripper_camera'])
        assert body['gripper_camera'] in config['cameras'], f"{name}: no intrinsics for its camera"
        assert 'robot_index' in body, f"{name}: robot_index required with gripper_camera"
        if 'gripper_hardware' in config:
            assert body['gripper_camera'] in config['gripper_hardware']['grippers'], \
                f"{name}: camera not in gripper config"
    robot_indices = sorted(b['robot_index'] for b in gripper_bodies.values())
    assert robot_indices == list(range(len(robot_indices))), "robot_index must be 0..n-1"

    if 'head_imu' in config:
        config['head_imu'] = {k: resolve(v) for k, v in config['head_imu'].items()}

    if 'gripper_imu' in config:
        config['gripper_imu'] = {str(serial): {k: resolve(v) for k, v in x.items()}
            for serial, x in config['gripper_imu'].items()}
        wrist_serials = {b['gripper_camera'] for b in gripper_bodies.values()}
        assert set(config['gripper_imu']) <= wrist_serials, "gripper_imu cameras must be gripper cameras"

    if 'setup_calibration' in config:
        config['setup_calibration'] = resolve(config['setup_calibration'])

    if 'episodes' in config:
        episodes = config['episodes']
        assert 'dock_tags' in episodes, "episodes needs dock_tags"
        for key, value in EPISODE_DEFAULTS.items():
            episodes.setdefault(key, value)
        unknown = set(episodes) - set(EPISODE_DEFAULTS) - {'dock_tags'}
        assert len(unknown) == 0, f"unknown episodes settings {unknown}"
        episodes['dock_tags'] = [int(x) for x in episodes['dock_tags']]
        static = [b for b in config['bodies'].values() if b['static']]
        assert len(static) == 1, "episodes need a static environment body"
        assert set(episodes['dock_tags']) <= set(static[0]['tags']), \
            "episodes dock_tags must be tags of the static environment body"
        assert len(gripper_bodies) == 2, "episodes (bang detection) need two gripper bodies"

    config['corner_noise_path'] = resolve(config['corner_noise'])
    if os.path.isfile(config['corner_noise_path']):
        config['sigma_px'] = json.load(open(config['corner_noise_path'], 'r'))['sigma_px']
    return config
