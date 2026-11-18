"""
umi.tag_tracking.wrist_sync on a synthetic take: a cube rotating smoothly,
observed by a rolling-shutter head camera (with pose noise) and by a gyro
whose clock is off by a drifting error. The episode corrections and their
drift line recover the clock error; the readout fit recovers the readout.

python tests/test_wrist_sync.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from umi.tag_tracking.wrist_sync import sync_episodes, drift_line, fit_readout, correction_seconds

FPS = 60000 / 1001
GYRO_RATE = 197.13
READOUT = 0.011
# wrist clock error: the wrist reports an event this much earlier than the head (s)
ERROR = lambda t: -0.020 + 0.0007 * t / 60.0


def synthetic_take(duration, seed):
    rng = np.random.default_rng(seed)
    t_fine = np.arange(0.0, duration, 0.001)
    # smooth body-frame angular velocity, ~0.4 rad/s
    omega = np.zeros((len(t_fine), 3))
    for j in range(3):
        for _ in range(6):
            f, phase, amp = rng.uniform(0.1, 1.5), rng.uniform(0, 2 * np.pi), rng.uniform(0.1, 0.3)
            omega[:, j] += amp * np.sin(2 * np.pi * f * t_fine + phase)
    rotations = [Rotation.identity()]
    for k in range(1, len(t_fine)):
        # midpoint rule: a left-endpoint step would delay the orientation by half a step
        rotations.append(rotations[-1] * Rotation.from_rotvec(0.5 * (omega[k - 1] + omega[k]) * 0.001))
    slerp = Slerp(t_fine, Rotation.from_quat([r.as_quat() for r in rotations]))
    return t_fine, omega, slerp, rng


def test_sync_and_readout():
    duration = 300.0
    t_fine, omega, slerp, rng = synthetic_take(duration, 1)
    times = np.arange(0.0, duration - 1.0, 1 / FPS)
    # cube image row drifts across the frame, different per episode
    rows = np.clip(0.5 + 0.3 * np.sin(2 * np.pi * times / 70.0), 0, 1)
    observed = slerp(times + READOUT * rows) * Rotation.from_rotvec(rng.normal(scale=np.radians(0.15), size=(len(times), 3)))
    T = np.tile(np.eye(4), (len(times), 1, 1))
    T[:, :3, :3] = observed.as_matrix()
    valid = rng.uniform(size=len(times)) > 0.05
    episodes = [{'start': s, 'end': s + 15.0} for s in np.arange(10.0, duration - 30.0, 25.0)]

    true_gyro_times = np.arange(0.5, duration - 0.5, 1 / GYRO_RATE)
    gyro = np.stack([np.interp(true_gyro_times, t_fine, omega[:, j]) for j in range(3)], axis=1)
    gyro += rng.normal(scale=0.005, size=gyro.shape)
    reported = true_gyro_times - ERROR(true_gyro_times)

    fits = sync_episodes(times, rows, T, valid, episodes, reported, gyro, READOUT, {})
    assert len(fits) == len(episodes) and all(f is not None for _, f in fits)
    errors = np.array([f['correction'] - ERROR(0.5 * (e['start'] + e['end'])) for e, f in fits])
    sigmas = np.array([f['sigma'] for _, f in fits])
    assert np.all(np.abs(errors) < 3 * sigmas + 0.0002), (errors, sigmas)
    assert abs(errors.mean()) < 0.0003, errors.mean()   # no bias
    line = drift_line(fits)
    assert line['n_inliers'] == len(episodes)
    for t in [0.0, 150.0, 300.0]:
        assert abs(correction_seconds(line, t) - ERROR(t)) < 0.0003, (t, correction_seconds(line, t), ERROR(t))
    assert line['residual_ms'] < 0.3, line['residual_ms']

    # without the rolling shutter the corrections shift by -readout * row; the readout fit recovers it
    fits0 = sync_episodes(times, rows, T, valid, episodes, reported, gyro, 0.0, {})
    samples = [('wrist', 0.5 * (e['start'] + e['end']), f['correction'], f['row']) for e, f in fits0]
    readout = fit_readout(samples)
    assert abs(readout['readout_ms'] - READOUT * 1e3) < 2.0, readout


def test_no_motion():
    times = np.arange(0.0, 60.0, 1 / FPS)
    T = np.tile(np.eye(4), (len(times), 1, 1))
    valid = np.ones(len(times), dtype=bool)
    gyro_times = np.arange(0.0, 60.0, 1 / GYRO_RATE)
    gyro = np.zeros((len(gyro_times), 3))
    fits = sync_episodes(times, np.full(len(times), 0.5), T, valid, [{'start': 10.0, 'end': 25.0}],
        gyro_times, gyro, READOUT, {})
    # flat cost: no minimum inside the search range, or a meaningless one with infinite sigma
    assert fits[0][1] is None or not np.isfinite(fits[0][1]['sigma']) or fits[0][1]['motion_rms'] == 0.0
    assert drift_line(fits) is None


if __name__ == '__main__':
    test_sync_and_readout()
    test_no_motion()
    print("all passed")
