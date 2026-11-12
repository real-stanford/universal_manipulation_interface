"""
Tag tracking config, all paths relative to the config file:

aruco_config: aruco_config.yaml       # dictionary and tag sizes
corner_noise: corner_noise.json       # from scripts_tag_tracking/01_measure_corner_noise.py
head_camera: C3501326502664           # serial of the camera observing the bodies
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
"""
import os
import json
import yaml
import numpy as np
from umi.common.cv_util import parse_aruco_config
from umi.tag_tracking.camera_model import FisheyeCamera


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
    config['tag_sizes'] = {x: aruco_config['marker_size_map'][x] for x in all_tags}

    config['corner_noise_path'] = resolve(config['corner_noise'])
    if os.path.isfile(config['corner_noise_path']):
        config['sigma_px'] = json.load(open(config['corner_noise_path'], 'r'))['sigma_px']
    return config
