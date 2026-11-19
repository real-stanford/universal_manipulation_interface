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
    for bad in ({'a': fast, 'c': slow}, {'a': fast, 'b': slow, 'c': slow}):
        try:
            assign_roles(bad)
            assert False
        except AssertionError as e:
            assert 'expected' in str(e) or 'no still video' in str(e)


if __name__ == '__main__':
    test_roles()
    print("all passed")
