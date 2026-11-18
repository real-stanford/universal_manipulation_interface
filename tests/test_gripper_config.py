"""
umi.common.gripper_config_util.load_gripper_config: range_camera.

python tests/test_gripper_config.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
from umi.common.gripper_config_util import load_gripper_config

BASE = """nominal_z: 0.079
grippers:
  A:
    gripper_id: 0
    left_finger_tag_id: 6
    right_finger_tag_id: 7
    range_camera: B
"""


def load(text):
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
        f.write(text)
    try:
        return load_gripper_config(f.name)
    finally:
        os.unlink(f.name)


def expect_error(text, words):
    try:
        load(text)
        assert False, 'no error'
    except AssertionError as e:
        assert words in str(e), str(e)


def test_range_camera():
    ok = load(BASE + "  B:\n    gripper_id: 1\n    left_finger_tag_id: 6\n    right_finger_tag_id: 7\n")
    assert ok['grippers']['A']['range_camera'] == 'B'
    expect_error(BASE, 'needs its own entry')
    expect_error(BASE + "  B:\n    gripper_id: 1\n    left_finger_tag_id: 0\n    right_finger_tag_id: 1\n", 'other finger tags')


if __name__ == '__main__':
    test_range_camera()
    print("all passed")
