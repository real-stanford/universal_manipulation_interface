"""
scripts/prepare_openicc_dataset.py: role assignment from gyro statistics.

python tests/test_prepare_openicc_dataset.py
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from prepare_openicc_dataset import motion_stats, assign_roles


def test_roles():
    rng = np.random.default_rng(0)
    still = motion_stats(np.abs(rng.normal(0, 0.005, 2000)))
    slow = motion_stats(np.radians(np.abs(rng.normal(8, 6, 2000))))
    fast = motion_stats(np.radians(np.abs(rng.normal(15, 15, 2000))))
    assert still['still_share'] > 0.95 and slow['still_share'] < 0.3
    roles = assign_roles({'a': fast, 'b': still, 'c': slow})
    assert roles == {'b': 'imu_bias', 'c': 'cam', 'a': 'cam_imu'}
    # forced cam_imu: the slower moving video
    assert assign_roles({'a': fast, 'b': still, 'c': slow}, 'c') == {'b': 'imu_bias', 'a': 'cam', 'c': 'cam_imu'}
    from prepare_openicc_dataset import angular_acceleration
    t = np.arange(2000) / 200.0  # 10 s at 200 Hz, rotation rate 2 sin(2 pi t): peak acceleration 4 pi
    gyro = np.stack([2 * np.sin(2 * np.pi * t), 0 * t, 0 * t], axis=1)
    acc = angular_acceleration(gyro, 10.0)
    assert abs(acc.max() - 4 * np.pi) < 0.2
    for bad in ({'a': fast, 'c': slow}, {'a': fast, 'b': slow, 'c': slow}):
        try:
            assign_roles(bad)
            assert False
        except AssertionError as e:
            assert 'expected' in str(e) or 'no still video' in str(e)


if __name__ == '__main__':
    test_roles()
    print("all passed")
