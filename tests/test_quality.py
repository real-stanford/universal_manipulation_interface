"""
umi.tag_tracking.quality checks and umi.common.image_stats on synthetic data.

python tests/test_quality.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import av
import numpy as np
from umi.tag_tracking import quality as q
from umi.common.image_stats import video_image_stats, stream_facts

FPS = 60.0


def status(checks, name, scope=None):
    return [c['status'] for c in checks if c['check'] == name and (scope is None or c['scope'] == scope)][0]


def test_runs_and_worst():
    assert q.longest_run([0, 1, 1, 0, 1, 1, 1, 0]) == 3 and q.longest_run([]) == 0
    assert q.worst([q.record('a', 'x', 'info')]) == 'pass'
    assert q.worst([q.record('a', 'x', 'pass'), q.record('b', 'x', 'warn')]) == 'warn'
    assert q.worst([q.record('a', 'x', 'fail'), q.record('b', 'x', 'warn')]) == 'fail'


def test_media():
    facts = {'width': 2704, 'height': 2028, 'container_rate': 59.94, 'codec': 'hevc', 'profile': 'Main',
        'pix_fmt': 'yuvj420p', 'color_range': 'JPEG', 'bit_rate': 60_000_000}
    n = 600
    times = np.arange(n) / 59.94
    key = np.zeros(n, bool)
    key[::30] = True
    timing = {'rate_hz': 59.9401, 'max_us': 200.0, 'rms_us': 50.0, 'n_frames': n, 'skipped_mskp': 0,
        'skipped_lskp': 0, 'has_skip_counters': True}
    c = q.media_checks(facts, timing, times, key, 'head')
    assert status(c, 'resolution') == 'pass' and status(c, 'frame_timing') == 'pass'
    assert status(c, 'frame_rate') == 'warn'  # 59.94 is not an integer rate
    assert status(c, 'dropped_frames') == 'pass' and status(c, 'keyframe_interval') == 'pass'
    assert status(q.media_checks(facts, {**timing, 'rate_hz': 60.0, 'n_frames': n + 3}, times, key, 'head'), 'dropped_frames') == 'fail'
    assert status(q.media_checks(facts, {**timing, 'rate_hz': 59.0}, times, key, 'head'), 'frame_rate') == 'fail'
    key2 = np.zeros(n, bool)
    key2[[0, 200]] = True  # 200 frames apart: 3.3 s
    assert status(q.media_checks(facts, timing, times, key2, 'head'), 'keyframe_interval') == 'fail'
    small = {**facts, 'width': 1280, 'height': 600}
    assert status(q.media_checks(small, timing, times, key, 'head'), 'resolution') == 'fail'
    assert status(q.media_checks(small, timing, times, key, 'wrist'), 'resolution') == 'pass'
    gap = np.delete(times, 100)
    assert status(q.media_checks(facts, timing, gap, key[:-1], 'head'), 'frame_intervals') == 'fail'


def test_images():
    n = 300
    rng = np.random.default_rng(0)
    stats = {'mean': 100 + rng.normal(0, 1, n), 'std': np.full(n, 40.0), 'saturated': np.zeros(n),
        'diff': np.concatenate([[np.nan], 2 + rng.random(n - 1)])}
    c = q.image_checks(stats, FPS, 'head')
    assert all(x['status'] == 'pass' for x in c), c
    frozen = {**stats, 'diff': stats['diff'].copy()}
    frozen['diff'][100:170] = 0.0  # 70 frames identical: > 1 s, also duplicates
    c = q.image_checks(frozen, FPS, 'head')
    assert status(c, 'frozen') == 'fail' and status(c, 'duplicate_frames') == 'warn'
    short = {**stats, 'diff': stats['diff'].copy(), 'mean': stats['mean'].copy()}
    short['diff'][10] = 0.0
    short['mean'][50:80] = 5.0  # dark for 0.5 s only
    c = q.image_checks(short, FPS, 'head')
    assert status(c, 'frozen') == 'pass' and status(c, 'dark') == 'pass' and status(c, 'duplicate_frames') == 'warn'
    dark = {**stats, 'mean': np.full(n, 5.0)}
    assert status(q.image_checks(dark, FPS, 'head'), 'dark') == 'fail'
    assert q.image_checks({k: v[:0] for k, v in stats.items()}, FPS, 'head')[0]['status'] == 'fail'


def frames_fixture(n=200):
    t = np.arange(n) / FPS
    pose = np.zeros((n, 7))
    pose[:, 0] = np.linspace(0, 0.3, n)
    pose[:, 6] = 1.0
    pose[50:60] = np.nan
    return {'time': t, 'head_frame_index': np.arange(1000, 1000 + n), 'ee_pose_right': pose}


def test_frame_table():
    f = frames_fixture()
    c = q.frame_table_checks(f, 200, FPS, ['right'])
    assert all(x['status'] == 'pass' for x in c), c
    assert status(q.frame_table_checks(f, 201, FPS, ['right']), 'rows') == 'fail'
    bad = {**f, 'ee_pose_right': f['ee_pose_right'].copy()}
    bad['ee_pose_right'][5, 3] = np.nan  # partly null row
    assert status(q.frame_table_checks(bad, 200, FPS, ['right']), 'pose_values') == 'fail'
    skip = {**f, 'head_frame_index': np.delete(np.arange(1000, 1201), 7)}
    assert status(q.frame_table_checks(skip, 200, FPS, ['right']), 'rows') == 'fail'


def test_tracking_gripper_motion():
    f = frames_fixture()
    valid = ~np.isnan(f['ee_pose_right'][:, 0])
    c = q.tracking_checks(valid, FPS, 'right')
    assert c[0]['status'] == 'pass' and c[0]['value'] == 0.95
    assert q.tracking_checks(valid[:20] & False, FPS, 'right')[0]['status'] == 'fail'
    width = np.full(200, 0.04)
    width[:10] = np.nan
    c = q.gripper_checks(width, 0.08, 2, 'right')
    assert status(c, 'width_valid') == 'pass' and status(c, 'width_range') == 'pass'
    width[100] = 0.2
    assert status(q.gripper_checks(width, 0.08, 2, 'right'), 'width_range') == 'fail'
    pose = f['ee_pose_right']
    m = q.motion_checks(pose, np.full(200, 0.04), FPS, 'right')
    assert status(m, 'tcp_speed') == 'pass' and status(m, 'pose_jumps') == 'pass'
    path = [x['value'] for x in m if x['check'] == 'tcp_path'][0]
    assert abs(path - 0.3 * (188 / 199)) < 1e-3  # the 11 steps touching the gap are not counted
    spike = pose.copy()
    spike[120, 1] = 0.2
    m = q.motion_checks(spike, np.full(200, 0.04), FPS, 'right')
    assert status(m, 'pose_jumps') == 'fail' and status(m, 'tcp_speed') == 'fail'
    assert q.static_check([0.01, 0.02])['status'] == 'fail' and q.static_check([0.3, 0.0])['status'] == 'pass'
    assert q.visibility_check(np.array([1, 1, 0, 0, 0], bool), 'right')['status'] == 'warn'


def test_sync():
    line = {'offset_ms': 16.0, 'drift_ms_per_min': -0.3, 'episodes': [
        {'start': 60.0, 'end': 80.0, 'correction_ms': 15.6, 'sigma_ms': 0.3, 'inlier': True},
        {'start': 120.0, 'end': 140.0, 'correction_ms': 10.0, 'sigma_ms': 0.3, 'inlier': True},
        {'start': 200.0, 'end': 220.0, 'correction_ms': 15.0, 'sigma_ms': 0.3, 'inlier': False}]}
    c = q.sync_check(line, 2.0, 60.0, 'right')
    assert c['status'] == 'pass' and abs(c['value'] - (15.6 - (16.0 - 0.3 * 70 / 60))) < 1e-3
    assert q.sync_check(line, 2.0, 120.0, 'right')['status'] == 'warn'  # 5.35 ms off: > 2 ms, < 3 x 2 ms
    assert q.sync_check(line, 1.0, 120.0, 'right')['status'] == 'fail'
    assert q.sync_check(line, 2.0, 200.0, 'right')['status'] == 'warn'
    assert q.sync_check(None, None, 60.0, 'right')['status'] == 'fail'


def test_timeline():
    ann = {'events': [
        {'arm': 'right', 'start_s': 0.0, 'end_s': 2.0, 'contribution': 'idle'},
        {'arm': 'right', 'start_s': 2.0, 'end_s': 10.0, 'contribution': 'advancing'},
        {'arm': 'right', 'start_s': 10.0, 'end_s': 15.0, 'contribution': 'idle'}],
        'key_events': [{'t_s': 4.0, 'kind': 'contact', 'measured_kind': 'close'},
            {'t_s': 9.0, 'kind': 'subgoal_complete'}],
        'recoveries': [], 'state_changes': [{'t_s': 4.0}], 'issues': [],
        'tasks': [{'outcome': 'success'}], 'annotation_source': 'model'}
    c = q.timeline_checks(ann, 15.0)
    assert q.worst(c) == 'pass', [x for x in c if x['status'] != 'pass']
    assert [x['value'] for x in c if x['check'] == 'idle_time'] == [7.0]
    gap = {**ann, 'events': [ann['events'][0], ann['events'][2]]}
    assert status(q.timeline_checks(gap, 15.0), 'event_coverage') == 'fail'
    late = {**ann, 'state_changes': [{'t_s': 16.0}]}
    assert status(q.timeline_checks(late, 15.0), 'event_times') == 'fail'
    failed = {**ann, 'recoveries': [{'failure_t_s': 5.0, 'recovered': False, 'recovered_at_s': 0.0}]}
    assert status(q.timeline_checks(failed, 15.0), 'labels_consistent') == 'warn'
    assert q.timeline_checks(None, 15.0)[0]['status'] == 'warn'


def test_image_stats_video():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'v.mp4')
        with av.open(path, 'w') as out:
            s = out.add_stream('libx264', rate=30)
            s.width, s.height, s.pix_fmt = 320, 240, 'yuv420p'
            s.options = {'g': '15'}
            for i in range(60):
                img = np.full((240, 320, 3), 0 if i >= 50 else 128, np.uint8)
                if i < 30:
                    img[:, i * 5:i * 5 + 40] = 255  # moving bar
                frame = av.VideoFrame.from_ndarray(img, format='rgb24')
                for p in s.encode(frame):
                    out.mux(p)
            for p in s.encode():
                out.mux(p)
        st = video_image_stats(path)
        facts = stream_facts(path)
        assert facts['width'] == 320 and facts['n_frames'] == 60 and len(st['time']) == 60
        assert np.all(np.diff(st['time']) > 0) and st['keyframe'][0]
        assert np.isnan(st['diff'][0]) and st['diff'][5] > 1 and st['diff'][40] < 0.5
        assert st['mean'][55] < 16 and st['dark'][55] > 0.9


if __name__ == '__main__':
    test_runs_and_worst()
    test_media()
    test_images()
    test_frame_table()
    test_tracking_gripper_motion()
    test_sync()
    test_timeline()
    test_image_stats_video()
    print("all passed")
