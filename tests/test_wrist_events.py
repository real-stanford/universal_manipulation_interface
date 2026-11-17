# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import numpy as np
from umi.tag_tracking.wrist_events import (DEFAULTS, runs, still_mask, accel_highpass, accel_peaks,
    joint_spikes, merge_bangs, audio_onset_ratio, max_near, docked_view, idle_periods, episodes_from_events)

RATE = 200.0


def synthetic_imu(segments, bang_times, seed):
    """
    segments: (duration, 'dock' | 'hand'); docked: gyro noise only, in hand:
    tremor and motion. Bangs: 1-sample 80 m/s^2 accel spikes.
    """
    rng = np.random.default_rng(seed)
    gyro, accel = list(), list()
    for duration, kind in segments:
        n = int(duration * RATE)
        scale = 0.005 if kind == 'dock' else 0.4
        gyro.append(rng.normal(scale=scale, size=(n, 3)))
        a = np.tile([0, 0, 9.81], (n, 1)) + rng.normal(scale=0.05 if kind == 'dock' else 1.0, size=(n, 3))
        accel.append(a)
    gyro, accel = np.concatenate(gyro), np.concatenate(accel)
    times = np.arange(len(gyro)) / RATE
    for t in bang_times:
        accel[int(t * RATE)] += [80, 0, 0]
    return times, gyro, accel


def tag(center, side):
    c = np.array(center, dtype=float)
    h = side / 2
    return np.array([c + [-h, -h], c + [h, -h], c + [h, h], c + [-h, h]])


def test_runs():
    assert runs(np.array([0, 1, 1, 0, 1], dtype=bool)) == [(1, 3), (4, 5)]


def test_still_and_bangs():
    segments = [(5, 'dock'), (20, 'hand'), (5, 'dock')]
    bangs_true = [6.0, 23.0]
    t, g0, a0 = synthetic_imu(segments, bangs_true, 0)
    _, g1, a1 = synthetic_imu(segments, [x + 0.01 for x in bangs_true], 1)
    # a big jolt in one gripper only, 0.2 s before the first bang, must not hide it
    a0[int(5.8 * RATE)] += [0, 150, 0]
    still = still_mask(t, g0, DEFAULTS['still_sec'], DEFAULTS['gyro_still'])
    assert still[int(2 * RATE)] and not still[int(15 * RATE)] and still[int(28 * RATE)]
    p0 = accel_peaks(t, accel_highpass(t, a0), 5.0)
    p1 = accel_peaks(t, accel_highpass(t, a1), 5.0)
    joint = joint_spikes(p0[0], p0[1], p1[0], p1[1], DEFAULTS['bang_tolerance'])
    bangs, strengths = merge_bangs(joint['time'], joint['strength'], DEFAULTS['bang_accel'], DEFAULTS['bang_merge_sec'])
    assert np.allclose(bangs, [6.005, 23.005], atol=0.01), bangs
    assert np.all(strengths > 50)
    # not simultaneous within tolerance: no bang
    late = joint_spikes(p0[0], p0[1], p1[0] + 0.1, p1[1], DEFAULTS['bang_tolerance'])
    assert len(late['time']) == 0


def test_audio_onset():
    rate = 48000
    rng = np.random.default_rng(0)
    audio = rng.normal(scale=0.01, size=rate * 4)
    # 2 ms broadband click at 2.5 s
    click = int(2.5 * rate)
    audio[click:click + 96] += rng.normal(scale=0.5, size=96)
    times, ratio = audio_onset_ratio(audio, rate, start_time=10.0)
    assert max_near(times, ratio, 12.5, 0.01) > 5 * np.median(ratio)
    assert max_near(times, ratio, 11.0, 0.01) < 2


def test_docked_view():
    from scipy.spatial.transform import Rotation
    from umi.tag_tracking.camera_model import FisheyeCamera
    from umi.tag_tracking.tag_detection import get_tag_object_points
    camera = FisheyeCamera('example/calibration/gopro_intrinsics_2_7k.json')
    sizes = {46: 0.08, 2: 0.06, 8: 0.06}
    cubes = [[2, 3, 4, 5], [8, 9, 10, 11]]
    # head camera looking down at the table from 0.9 m; dock tag 46 on the table
    # (tag z up = toward the camera); poses are camera-from-tag
    R_down = Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()
    def T(R, t):
        out = np.eye(4); out[:3, :3] = R; out[:3, 3] = t
        return out
    T_cam_dock = T(R_down, [0.0, 0.1, 0.9])

    def corners(T_cam_tag, size):
        pts = (T_cam_tag[:3, :3] @ get_tag_object_points(size).T).T + T_cam_tag[:3, 3]
        return camera.project(pts)

    def cube_tag(center_in_dock):
        # top face tag of a cube whose center is at center_in_dock (dock frame), tag facing up
        T_dock_tag = T(np.eye(3), np.asarray(center_in_dock) + [0, 0, 0.035])
        return corners(T_cam_dock @ T_dock_tag, 0.06)

    frame = {46: corners(T_cam_dock, 0.08), 2: cube_tag([-0.08, -0.10, 0.13]), 8: cube_tag([0.08, -0.10, 0.13])}
    ok, distance = docked_view(frame, cubes, [46, 47], sizes, camera, 0.35)
    # single small tags: ~1-2 cm position error (orientation ambiguity), far below the docked / held-above gap
    assert ok and abs(distance - np.linalg.norm([0.08, 0.10, 0.13])) < 0.03, distance
    # held above the dock (a bang over it): not docked
    above = {**frame, 2: cube_tag([-0.05, -0.10, 0.6]), 8: cube_tag([0.05, -0.10, 0.6])}
    ok, distance = docked_view(above, cubes, [46, 47], sizes, camera, 0.35)
    assert not ok and abs(distance - np.linalg.norm([0.05, 0.10, 0.6])) < 0.05, distance
    # one cube far away on the table
    far = {**frame, 8: cube_tag([0.5, -0.10, 0.13])}
    assert not docked_view(far, cubes, [46, 47], sizes, camera, 0.35)[0]
    # cube or dock tags not seen
    assert docked_view({46: frame[46], 2: frame[2]}, cubes, [46, 47], sizes, camera, 0.35) == (False, None)
    assert docked_view({2: frame[2], 8: frame[8]}, cubes, [46, 47], sizes, camera, 0.35) == (False, None)


def test_episodes():
    # dock 0-5 (seen), episode 5-25 with bangs at 6 and 23, dock 25-30 (seen),
    # still 40-45 away from the dock (not seen), bangs 50 and 60, dock 65-70 (seen)
    grid = np.arange(0, 70, 0.01)
    still = ((grid < 5) | ((grid >= 25) & (grid < 30)) | ((grid >= 40) & (grid < 45)) | (grid >= 65))
    frame_times = np.arange(0, 70, 1 / 60)
    frame_docked = (frame_times < 5) | ((frame_times >= 25) & (frame_times < 30)) | (frame_times >= 65)
    idle = idle_periods(grid, still, frame_times, frame_docked, DEFAULTS['min_dock_frames'])
    assert [p['idle'] for p in idle] == [True, True, False, True]
    # 25.5 (setting down) and 30.4 (lifting off) are within idle_margin of idle: not bangs
    episodes = episodes_from_events(idle, np.array([6.0, 23.0, 25.5, 30.4, 50.0, 60.0]), 70.0,
        DEFAULTS['idle_margin'], DEFAULTS['min_episode_sec'])
    ok = [e for e in episodes if e['status'] == 'ok']
    # the still period without docked view does not split the second episode
    assert [(e['start'], e['end']) for e in ok] == [(6.0, 23.0), (50.0, 60.0)]


def test_frame_labels():
    from umi.tag_tracking.episodes import frame_labels
    result = {
        'idle': [{'start': 0.0, 'end': 4.0, 'idle': True}, {'start': 40.0, 'end': 45.0, 'idle': False}],
        'rejected': [{'active_start': 30.0, 'active_end': 50.0, 'status': 'no_bang', 'n_bangs': 0}],
        'episodes': [{'start': 6.0, 'end': 23.0, 'active_start': 4.0, 'active_end': 25.0, 'n_bangs': 2}],
        'bangs': [6.0, 23.0]
    }
    times = np.array([1.0, 6.0, 10.0, 23.0, 24.0, 42.0])
    assert frame_labels(times, result) == ['IDLE', 'BANG', 'EPISODE 1', 'BANG', '', 'REJECTED (no_bang)']


def test():
    test_runs()
    test_still_and_bangs()
    test_audio_onset()
    test_docked_view()
    test_episodes()
    test_frame_labels()
    print('all passed')


if __name__ == '__main__':
    test()
