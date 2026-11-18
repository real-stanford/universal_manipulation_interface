"""
umi.tag_tracking.wrist_sync.drift_line: drift line from 3+ episodes, constant from 1-2.

python tests/test_wrist_sync_line.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial.transform import Rotation
from umi.tag_tracking.wrist_sync import drift_line, correction_seconds, sync_episodes


def fit(c, sigma=0.0003):
    return {'correction': c, 'sigma': sigma, 'rms': 0.05, 'motion_rms': 0.4, 'n_pairs': 500, 'row': 0.6}


def test_drift_line():
    assert drift_line([]) is None and drift_line([({'start': 0, 'end': 10}, None)]) is None
    one = drift_line([({'start': 0.0, 'end': 60.0}, fit(0.015))])
    assert one['constant'] and abs(one['offset_ms'] - 15.0) < 1e-9 and one['drift_ms_per_min'] == 0.0
    assert abs(one['residual_ms'] - 0.3) < 1e-9  # the fit's own sigma, not 0
    assert abs(float(correction_seconds(one, 30.0)) - 0.015) < 1e-12
    two = drift_line([({'start': 0.0, 'end': 60.0}, fit(0.014)), ({'start': 70.0, 'end': 130.0}, fit(0.016))])
    assert two['constant'] and abs(two['offset_ms'] - 15.0) < 1e-9 and abs(two['residual_ms'] - 1.0) < 1e-9
    many = drift_line([({'start': 60.0 * k, 'end': 60.0 * k + 30}, fit(0.010 + 0.001 * k)) for k in range(6)])
    assert not many['constant'] and abs(many['drift_ms_per_min'] - 1.0) < 1e-6


def rotations(t):
    return Rotation.from_rotvec(np.stack([0.4 * np.sin(1.3 * t), 0.3 * np.sin(0.7 * t + 1), 0.5 * np.sin(1.9 * t + 2)], axis=1))


def test_partly_covered_episode():
    # head-tracked cube over 0-60 s, the wrist gyro (12 ms late) only over 10-50 s
    t = np.arange(0, 60, 1 / 60.0)
    T = np.tile(np.eye(4), (len(t), 1, 1))
    T[:, :3, :3] = rotations(t).as_matrix()
    shift, dt = 0.012, 1e-4
    tg = np.arange(10, 50, 1 / 200.0)
    omega = (rotations(tg - shift).inv() * rotations(tg + dt - shift)).as_rotvec() / dt
    (e, f), = sync_episodes(t, t, np.full(len(t), 0.5), T, np.ones(len(t), bool), [{'start': 0.0, 'end': 60.0}], tg, omega, {})
    assert 10.0 < e['start'] < 10.2 and 49.8 < e['end'] < 50.0
    assert abs(f['correction'] + shift) < 0.0005
    # covered for less than min_cover_sec: not fitted
    assert sync_episodes(t, t, np.full(len(t), 0.5), T, np.ones(len(t), bool), [{'start': 0.0, 'end': 13.0}], tg, omega, {}) == []


if __name__ == '__main__':
    test_drift_line()
    test_partly_covered_episode()
    print("all passed")
