# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import numpy as np
from umi.tag_tracking.wrist_events import (DEFAULTS, runs, still_mask, accel_highpass, accel_peaks,
    joint_spikes, merge_bangs, audio_onset_ratio, max_near, docked_view, idle_periods, episodes_from_events,
    bang_click, click_offset, wrist_clock_line, wrist_clock_offset_ms, episodes_from_idle)

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


def test_wrist_clock_sync():
    """
    Head and wrist microphones hear the same bang clicks; the wrist recording's
    clock drifts. Near every click the wrist also hears a louder, different
    sound (fingers, handling). Per-bang offsets by envelope cross-correlation
    and the drift line recover the clock error; a wrong bang is an outlier.
    """
    rate = 48000
    rng = np.random.default_rng(3)
    duration = 120.0
    bangs = np.arange(5.0, 116.0, 10.0)
    # wrist clock error (s): the wrist recording places a sound error(t) earlier than the head
    error = lambda t: 0.014 + 0.0007 * t / 60.0
    n = int(duration * rate)
    head = rng.normal(scale=0.01, size=n).astype(np.float32)
    wrist = rng.normal(scale=0.01, size=n).astype(np.float32)
    decay = np.exp(-np.arange(240) / 40.0)
    for t in bangs:
        click = rng.normal(size=240) * decay   # each bang sounds different
        i = int(round(t * rate))
        head[i:i + 240] += 0.5 * click        # farther from the head microphone
        j = int(round((t - error(t)) * rate))
        wrist[j:j + 240] += click
        # louder handling sound 40 ms later at the wrist
        k = j + int(0.04 * rate)
        wrist[k:k + 2400] += 3.0 * rng.normal(size=2400) * np.exp(-np.arange(2400) / 800.0)
    head_times, head_ratio = audio_onset_ratio(head, rate, 0.0)
    head_clicks = [bang_click(head_times, head_ratio, t, 0.015, 5.0) for t in bangs]
    assert all(h is not None and abs(h - t) < 0.004 for h, t in zip(head_clicks, bangs)), head_clicks

    measured = [click_offset(head, 0.0, wrist, 0.0, rate, h, 0.1) for h in head_clicks]
    for t, (offset, correlation) in zip(bangs, measured):
        assert abs(offset - error(t)) < 0.0002, (t, offset, error(t))
        assert correlation > 0.8, correlation
    offsets = [m[0] for m in measured]
    correlations = [m[1] for m in measured]
    # one bang matched to another sound
    offsets[3] += 0.03
    line = wrist_clock_line(bangs, offsets, correlations, 0.5)
    assert line['n_inliers'] == len(bangs) - 1 and not line['bangs'][3]['inlier'], line
    for t in [0.0, 60.0, 120.0]:
        assert abs(wrist_clock_offset_ms(line, t) - error(t) * 1e3) < 0.2, (t, wrist_clock_offset_ms(line, t))
    assert line['residual_ms'] < 0.2, line['residual_ms']
    # short video with 30% wrong matches (task3 first left video): still the true line
    t_short = np.arange(10.0, 420.0, 20.0)
    y_short = 0.003 + np.random.default_rng(5).normal(scale=0.001, size=len(t_short))
    wrong = [1, 4, 6, 9, 13, 17]
    y_short[wrong] = [0.085, -0.078, -0.035, -0.023, -0.017, 0.029]
    line = wrist_clock_line(t_short, y_short, np.ones(len(t_short)), 0.5)
    assert set(np.flatnonzero([not b['inlier'] for b in line['bangs']])) == set(wrong), line['bangs']
    assert abs(line['drift_ms_per_min']) < 0.3 and abs(line['offset_ms'] - 3.0) < 1.0, line
    assert line['residual_ms'] < 1.5, line['residual_ms']
    # low correlation bangs are not used; fewer than 2 usable: no line
    assert wrist_clock_line(bangs[:2], offsets[:2], [0.9, 0.1], 0.5) is None
    # outside the recording: no measurement
    assert click_offset(head, 0.0, wrist, 0.0, rate, duration - 0.01, 0.1) is None


def test_dock_to_dock():
    """
    Episodes between dock periods without bangs; the task interval removes the docking.
    """
    from umi.tag_tracking.episodes import task_interval
    idle = [{'start': 0.0, 'end': 5.0, 'idle': True}, {'start': 30.0, 'end': 35.0, 'idle': True},
        {'start': 36.0, 'end': 37.0, 'idle': False}, {'start': 60.0, 'end': 61.0, 'idle': True}]
    out = episodes_from_idle(idle, 62.0, 3.0)
    # the take starts docked (no stretch before the first dock period); 61-62 s too short
    assert [x['status'] for x in out] == ['ok', 'ok', 'too_short'], out
    assert (out[0]['start'], out[0]['end']) == (5.0, 30.0) and (out[1]['start'], out[1]['end']) == (35.0, 60.0)

    times = np.arange(5.0, 30.0, 0.1)
    dock = np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]])
    # cube A leaves at 7 s, cube B at 8 s (not tracked 9-12 s), both back by 27 / 28 s
    def path(leave, back):
        p = np.zeros((len(times), 3))
        p[:, 0] = np.where((times > leave) & (times < back), 0.8, 0.1)
        return p
    a, b = path(7.0, 27.0), path(8.0, 28.0)
    b[(times > 9.0) & (times < 12.0)] = np.nan
    interval = task_interval(times, [a, b], dock, 5.0, 30.0, 0.45)
    assert abs(interval[0] - 8.1) < 0.051 and abs(interval[1] - 27.0) < 0.051, interval
    # cubes never leave the dock
    assert task_interval(times, [path(50.0, 51.0)] * 2, dock, 5.0, 30.0, 0.45) is None
    # bang ritual: the task runs between the bangs; a side without a bang keeps the dock boundary
    from umi.tag_tracking.episodes import between_bangs
    assert between_bangs((8.0, 27.0), [2.0, 9.5, 10.0, 16.0, 19.0, 25.0, 40.0]) == ((9.5, 25.0), ['bang', 'bang'])
    assert between_bangs((8.0, 27.0), [25.0]) == ((8.0, 25.0), ['dock', 'bang'])
    assert between_bangs((8.0, 27.0), []) == ((8.0, 27.0), ['dock', 'dock'])
    # the start bang before the dock exit (work close to the dock): found in the dock-to-dock episode
    assert between_bangs((19.0, 27.0), [8.0, 25.0], episode=(5.0, 30.0)) == ((8.0, 25.0), ['bang', 'bang'])


def test():
    test_runs()
    test_still_and_bangs()
    test_audio_onset()
    test_docked_view()
    test_episodes()
    test_frame_labels()
    test_wrist_clock_sync()
    test_dock_to_dock()
    print('all passed')


if __name__ == '__main__':
    test()
