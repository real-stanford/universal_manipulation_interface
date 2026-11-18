from typing import Optional
import yaml


def load_gripper_config(path: Optional[str]) -> Optional[dict]:
    """
    Gripper hardware config, replacing the default UMI conventions
    (gripper g uses finger tags 6g (left) and 6g+1 (right), UMI geometry).

    Example:
    nominal_z: 0.079                # finger tag z distance from camera [m]
    cam_to_center_height: 0.086     # optional, see 06_generate_dataset_plan.py
    cam_to_mount_offset: 0.01465    # optional, see 06_generate_dataset_plan.py
    tcp_offset: 0.205               # optional, mount to fingertips (tag tracking calibrate_setup.py)
    grippers:                       # by wrist camera serial
      C3501326029433:
        gripper_id: 0
        left_finger_tag_id: 6       # tag on the image-left finger
        right_finger_tag_id: 1      # tag on the image-right finger
        finger_tag_to_tip: 0.02     # optional, finger tag centers to fingertips
                                    # along the camera optical axis [m], measured
        cam_to_center_height: 0.075 # optional, this gripper's TCP geometry, overrides the
        tcp_offset: 0.177           # shared values above (tag tracking calibrate_setup.py)
        range_camera: C3501325427851  # optional: the opening range (gripper_range.json) is from
                                    # this camera's gripper calibration video (the finger tag
                                    # distance is the gripper's, the camera then needs its own
                                    # entry with the same finger tags)
    """
    if path is None:
        return None
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    assert len(config['grippers']) > 0
    gripper_ids = list()
    for serial, gripper in config['grippers'].items():
        for key in ['gripper_id', 'left_finger_tag_id', 'right_finger_tag_id']:
            assert key in gripper, f"{key} missing for camera {serial}"
        assert gripper['left_finger_tag_id'] != gripper['right_finger_tag_id']
        if 'range_camera' in gripper:
            source = config['grippers'].get(gripper['range_camera'])
            assert source is not None, f"{serial}: range_camera {gripper['range_camera']} needs its own entry"
            assert (source['left_finger_tag_id'], source['right_finger_tag_id']) == \
                (gripper['left_finger_tag_id'], gripper['right_finger_tag_id']), \
                f"{serial}: range_camera {gripper['range_camera']} has other finger tags"
        gripper_ids.append(gripper['gripper_id'])
    assert len(set(gripper_ids)) == len(gripper_ids), "gripper_id must be unique"
    return config

