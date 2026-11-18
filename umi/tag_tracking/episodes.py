"""
Episode segmentation of continuous takes (docked -> bang -> task -> bang ->
docked) from the wrist camera IMUs, the head camera audio and the head
camera's raw tag detections, before any tag tracking; see
umi.tag_tracking.wrist_events for the rules and validated defaults.

segment_take: one head camera take -> idle periods, bangs, episodes.
frame_labels: per frame label for rendering.
All times are seconds on the head video's clock (its frame times in tag_corners.pkl).
"""
from typing import Dict, List, Optional, Sequence, Tuple
import pickle
import numpy as np
from umi.common.audio_sync_util import read_audio_mono
from umi.tag_tracking.imu import load_imu
from umi.tag_tracking.wrist_events import (DEFAULTS, gyro_rms, still_mask, accel_highpass, accel_peaks, robust_line,
    joint_spikes, docked_near, merge_bangs, audio_onset_ratio, audio_clicks, estimate_audio_lag, max_near,
    docked_view, idle_periods, episodes_from_events, episodes_from_idle, bang_click, click_offset, wrist_clock_line, wrist_clock_offset_ms, robust_drift_line)

GRID_DT = 0.01
# joint spikes kept for diagnostics down to this strength (m/s^2)
REPORT_MIN_STRENGTH = 5.0


def segment_take(head: dict, gripper_wrists: Sequence[List[dict]], cube_tags: Sequence[Sequence[int]],
        dock_tags: Sequence[int], tag_sizes: Dict[int, float], head_camera, params: Dict) -> dict:
    """
    head: session video dict of the head camera take (umi.tag_tracking.session),
        with tag_corners.pkl in its dir.
    gripper_wrists: per gripper, its wrist camera videos overlapping the take
        (session video dicts, imu_data.json in their dirs), in the order of cube_tags.
    params: DEFAULTS keys (the episodes section of the tag tracking config).

    Returns:
        episodes: [{'start', 'end', 'active_start', 'active_end', 'n_bangs', 'audio_clock'}] (status ok);
            audio_clock: diagnostic, per wrist video name recording the episode,
            {'gripper', 'offset_ms', 'residual_ms', 'bang_offsets_ms'} or None
        audio_clock: diagnostic, per wrist video name, its audio clock against the
            head audio from the bang clicks (wrist_events.wrist_clock_line) or None.
            GoPro audio is not locked to the video (task3: drifting up to 0.8 ms/min
            against the camera's own IMU), so this is not applied: the wrist video
            clocks are corrected from motion (03s_sync_wrists.py).
        rejected: [{'active_start', 'active_end', 'status', 'n_bangs'}]
        idle: [{'start', 'end', 'n_docked_frames', 'idle'}] all still periods of both grippers
        bangs, audio_lag, n_lag_pairs, n_clicks
        diagnostics: per gripper and per frame arrays for checking (see check_wrist_events.py)
    """
    params = {**DEFAULTS, **params}
    assert len(gripper_wrists) == 2 and len(cube_tags) == 2, "bang detection needs two grippers"

    # head view: both cubes in front of the dock tags
    corners = pickle.load(head['dir'].joinpath('tag_corners.pkl').open('rb'))
    frame_times = np.array([f['time'] for f in corners])
    views = [docked_view(f['corners'], cube_tags, dock_tags, tag_sizes, head_camera, params['dock_distance'])
        for f in corners]
    frame_docked = np.array([v[0] for v in views])
    frame_dock_distance = np.array([np.nan if v[1] is None else v[1] for v in views])
    take_end = float(frame_times[-1])
    grid = np.arange(0.0, take_end, GRID_DT)

    # head audio on the head video clock (audio_offset: audio start relative to the first frame)
    audio, sample_rate, audio_offset = read_audio_mono(str(head['dir'].joinpath('raw_video.mp4')))
    head_audio_start = frame_times[0] + audio_offset
    audio_times, audio_ratio = audio_onset_ratio(audio, sample_rate, head_audio_start)

    # wrist IMUs on the head video clock (audio synchronized start times)
    rms_on_grid, still_on_grid, accel_times, accel_high, peaks = list(), list(), list(), list(), list()
    for wrists in gripper_wrists:
        assert len(wrists) > 0, f"no wrist video overlapping {head['name']}"
        rms_g = np.full(len(grid), np.nan)
        still_g = np.zeros(len(grid), dtype=bool)
        g_times, g_high = list(), list()
        for wrist in wrists:
            imu = load_imu(str(wrist['dir'].joinpath('imu_data.json')))
            times = imu['time'] + wrist['start_timestamp'] - head['start_timestamp']
            rms = gyro_rms(times, imu['gyro'], params['still_sec'])
            still = still_mask(times, imu['gyro'], params['still_sec'], params['gyro_still'])
            inside = (grid >= times[0]) & (grid <= times[-1])
            idx = np.clip(np.searchsorted(times, grid[inside]), 0, len(times) - 1)
            rms_g[inside] = rms[idx]
            still_g[inside] = still[idx]
            g_times.append(times)
            g_high.append(accel_highpass(times, imu['accel']))
        rms_on_grid.append(rms_g)
        still_on_grid.append(still_g)
        accel_times.append(np.concatenate(g_times))
        accel_high.append(np.concatenate(g_high))
        peaks.append(accel_peaks(accel_times[-1], accel_high[-1], min(REPORT_MIN_STRENGTH, params['bang_accel'])))

    # bangs: joint wrist spikes, confirmed by the head audio click at spike + lag, not at the dock.
    # Clocks drift (wrist-to-wrist and wrist-to-head, ~1 ms/min): spikes are paired widely, then
    # bangs are found iteratively: confirmed bangs (initially with constant timing, which holds
    # early in the take) give robust line fits of the wrist-to-wrist dt(t) and the click lag(t),
    # which re-check every pair; until the confirmed set no longer changes.
    joint = joint_spikes(peaks[0][0], peaks[0][1], peaks[1][0], peaks[1][1], params['bang_search_tolerance'])
    clicks = audio_clicks(audio_times, audio_ratio, params['audio_ratio'] if params['audio_ratio'] > 0 else 5.0)
    strong = (joint['strength'] >= params['lag_accel']) & (np.abs(joint['dt']) <= params['bang_tolerance'])
    lag_line, n_pairs = estimate_audio_lag(joint['time'][strong], clicks, params['audio_lag_search'])
    lag_a, lag_b = (0.0, 0.0) if lag_line is None else lag_line
    dt_a, dt_b = 0.0, 0.0
    joint['docked'] = docked_near(joint['time'], frame_times, frame_docked, params['dock_window'])
    previous = None
    for _ in range(10):
        on_time = np.abs(joint['dt'] - (dt_a + dt_b * joint['time'])) <= params['bang_tolerance']
        joint['audio'] = np.array([max_near(audio_times, audio_ratio, t + lag_a + lag_b * t, params['audio_tolerance'])
            for t in joint['time']])
        candidate = on_time & (joint['strength'] >= params['bang_accel']) \
            & (joint['audio'] >= params['audio_ratio']) & ~joint['docked']
        if previous is not None and np.array_equal(candidate, previous):
            break
        previous = candidate
        if candidate.sum() < 5:
            break
        # drift fits from the confirmed bangs only
        t_c = joint['time'][candidate]
        dt_a, dt_b, _ = robust_line(t_c, joint['dt'][candidate])
        lags = list()
        for t in t_c:
            j = int(np.argmin(np.abs(clicks - (t + lag_a + lag_b * t)))) if len(clicks) else None
            if j is not None and abs(clicks[j] - (t + lag_a + lag_b * t)) <= params['audio_tolerance']:
                lags.append((t, clicks[j] - t))
        if len(lags) >= 5:
            lag_a, lag_b, n_pairs = robust_line(np.array([x for x, _ in lags]), np.array([y for _, y in lags]))
            lag_line = (lag_a, lag_b)
    bangs, bang_strengths = merge_bangs(joint['time'][candidate], joint['strength'][candidate],
        params['bang_accel'], params['bang_merge_sec'])
    # per bang: wrist-to-wrist spike time difference (gripper 1 - gripper 0) and the head click
    bang_wrist_dt = np.array([joint['dt'][np.argmin(np.abs(joint['time'] - t))] for t in bangs])
    head_clicks = [bang_click(audio_times, audio_ratio, t + lag_a + lag_b * t, params['audio_tolerance'],
        params['audio_ratio']) for t in bangs]

    idle = idle_periods(grid, still_on_grid[0] & still_on_grid[1], frame_times, frame_docked, params['min_dock_frames'])
    if params['boundary'] == 'dock':
        stretches = episodes_from_idle(idle, take_end, params['min_episode_sec'])
        for e in stretches:
            e['n_bangs'] = int(np.sum((bangs >= e['active_start']) & (bangs <= e['active_end'])))
    elif params['boundary'] == 'video':
        # one episode per head video (recordings started / stopped per episode, continuous
        # work inside): the whole video; the task interval (docking, first / last bang)
        # comes from 03t_trim_docking.py
        stretches = [{'start': 0.0, 'end': take_end, 'active_start': 0.0, 'active_end': take_end,
            'n_bangs': int(len(bangs)), 'status': 'ok' if take_end >= params['min_episode_sec'] else 'too_short'}]
    else:
        assert params['boundary'] == 'bang', f"unknown episode boundary {params['boundary']}"
        stretches = episodes_from_events(idle, bangs, take_end, params['idle_margin'], params['min_episode_sec'])
    episodes = [{k: e[k] for k in ['start', 'end', 'active_start', 'active_end', 'n_bangs']}
        for e in stretches if e['status'] == 'ok']

    # diagnostic: per wrist video, its audio clock against the head audio at every bang it
    # recorded (head click matched in its audio), a drift line over the take
    wrist_sync = dict()
    for e in episodes:
        e['audio_clock'] = dict()
    for g, wrists in enumerate(gripper_wrists):
        for wrist in wrists:
            offset = wrist['start_timestamp'] - head['start_timestamp']
            w_audio, w_rate, w_audio_offset = read_audio_mono(str(wrist['dir'].joinpath('raw_video.mp4')))
            assert w_rate == sample_rate, f"{wrist['name']}: audio {w_rate} Hz, head {sample_rate} Hz"
            measured = [(t, click_offset(audio, head_audio_start, w_audio, offset + w_audio_offset, sample_rate,
                h, params['sync_search'])) for t, h in zip(bangs, head_clicks) if h is not None]
            del w_audio
            measured = [(t, m) for t, m in measured if m is not None]
            line = wrist_clock_line([t for t, _ in measured], [m[0] for t, m in measured],
                [m[1] for t, m in measured], params['sync_min_correlation'])
            if line is not None:
                # check: this camera's own IMU spike to its own audio click at the matched bangs;
                # constant over the take if its audio and IMU (video) clocks run together
                for item, (t, (offset_s, _)) in zip(line['bangs'], measured):
                    k = int(np.argmin(np.abs(joint['time'] - t)))
                    spike = joint['time'][k] + (0.5 if g == 1 else -0.5) * joint['dt'][k]
                    h = head_clicks[int(np.argmin(np.abs(bangs - t)))]
                    item['imu_to_audio_ms'] = float(((h - offset_s) - spike) * 1e3)
                used = np.array([b['inlier'] for b in line['bangs']])
                if used.sum() >= 3:
                    a, b, ok = robust_drift_line(np.array([b['time'] for b in line['bangs']]),
                        np.array([b['imu_to_audio_ms'] for b in line['bangs']]) * 1e-3, used)
                    line['imu_to_audio'] = {'offset_ms': a * 1e3, 'drift_ms_per_min': b * 60e3, 'n_inliers': int(ok.sum()),
                        'residual_ms': float(np.sqrt(np.mean((np.array([x['imu_to_audio_ms'] for x in line['bangs']])[ok]
                            - (a + b * np.array([x['time'] for x in line['bangs']])[ok]) * 1e3) ** 2)))}
            wrist_sync[wrist['name']] = None if line is None else {'gripper': g, **line}
            # episodes this wrist video records entirely
            for e in episodes:
                if not (wrist['start_timestamp'] <= head['start_timestamp'] + e['start']
                        and head['start_timestamp'] + e['end'] <= wrist['end_timestamp']):
                    continue
                e['audio_clock'][wrist['name']] = None if line is None else {
                    'gripper': g,
                    'offset_ms': wrist_clock_offset_ms(line, 0.5 * (e['start'] + e['end'])),
                    'residual_ms': line['residual_ms'],
                    'bang_offsets_ms': [b['offset_ms'] for b in line['bangs'] if e['start'] <= b['time'] <= e['end']]
                }
    del audio
    rejected = [{k: e[k] for k in ['active_start', 'active_end', 'status', 'n_bangs']}
        for e in stretches if e['status'] != 'ok']

    return {
        'episodes': episodes,
        'rejected': rejected,
        'idle': idle,
        'bangs': bangs.tolist(),
        'bang_strengths': bang_strengths.tolist(),
        'bang_wrist_dt_ms': (bang_wrist_dt * 1e3).tolist(),
        'audio_clock': wrist_sync,
        # lag(t) = a + b * t (head video seconds), wrist-to-wrist dt(t) likewise
        'audio_lag': None if lag_line is None else lag_a,
        'audio_lag_drift_ms_per_min': None if lag_line is None else lag_b * 60e3,
        'wrist_dt_ms': [dt_a * 1e3, dt_b * 60e3],
        'n_lag_pairs': n_pairs,
        'n_clicks': int(len(clicks)),
        'diagnostics': {
            'grid': grid, 'gyro_rms': rms_on_grid,
            'accel_times': accel_times, 'accel_high': accel_high,
            'audio_times': audio_times, 'audio_ratio': audio_ratio,
            'frame_times': frame_times, 'frame_docked': frame_docked, 'frame_dock_distance': frame_dock_distance,
            'joint': joint, 'stretches': stretches, 'take_end': take_end
        }
    }


def frame_labels(times: np.ndarray, video_result: dict, bang_window: float = 0.15) -> List[str]:
    """
    Per frame label for rendering: EPISODE k, BANG, IDLE, REJECTED (reason) or ''.
    video_result: one video's entry of episodes.json (segment_take without diagnostics).
    """
    labels = np.full(len(times), '', dtype=object)
    for p in video_result['idle']:
        if p['idle']:
            labels[(times >= p['start']) & (times <= p['end'])] = 'IDLE'
    for r in video_result['rejected']:
        labels[(times >= r['active_start']) & (times <= r['active_end'])] = f"REJECTED ({r['status']})"
    for k, e in enumerate(video_result['episodes']):
        labels[(times >= e['start']) & (times <= e['end'])] = f"EPISODE {k + 1}"
    for t in video_result['bangs']:
        labels[np.abs(times - t) <= bang_window] = 'BANG'
    return labels.tolist()


def task_interval(times: np.ndarray, cube_positions: Sequence[np.ndarray], dock_positions: np.ndarray,
        start: float, end: float, exit_distance: float) -> Optional[Tuple[float, float]]:
    """
    The task part of a dock-to-dock episode, without the docking: from the
    first head frame where every cube is farther than exit_distance (m) from
    the nearest dock tag, to the last such frame. A cube not tracked in a
    frame keeps its last known state (at the start of the episode: docked).
    cube_positions: per cube (N,3) world positions per head frame, NaN where
    not tracked; dock_positions: (M,3) world positions of the dock tags.
    None if the cubes never all leave the dock area.
    """
    inside = np.flatnonzero((times >= start) & (times <= end))
    if len(inside) == 0:
        return None
    out_all = np.ones(len(inside), dtype=bool)
    for positions in cube_positions:
        p = positions[inside]
        distance = np.min(np.linalg.norm(p[:, None, :] - dock_positions[None, :, :], axis=2), axis=1)
        known = ~np.isnan(distance)
        state = np.zeros(len(inside), dtype=bool)   # out of the dock area
        last = False
        for k in range(len(inside)):
            if known[k]:
                last = distance[k] > exit_distance
            state[k] = last
        out_all &= state
    if not out_all.any():
        return None
    k = np.flatnonzero(out_all)
    return float(times[inside[k[0]]]), float(times[inside[k[-1]]])


def between_bangs(interval: Tuple[float, float], bangs: Sequence[float],
        episode: Optional[Tuple[float, float]] = None) -> Tuple[Tuple[float, float], List[str]]:
    """
    Remove the bang ritual from a task interval: bangs are made right after
    leaving the dock and right before returning (both grippers held up to the
    head camera's microphone, where the cubes leave its view). Bangs are looked
    for in the whole dock-to-dock episode (default: the interval): the work and
    the bangs can be close to the dock, still within dock_exit_distance. The
    task starts at the first bang in the episode's first half and ends at the
    last bang in its second half (first to last bang, as the bang boundary of
    00b, so bangs inside the task do not cut it); a side without a bang keeps
    the interval's boundary (dock exit / return). Returns the interval and,
    per side, 'bang' or 'dock'.
    """
    start, end = interval
    lo, hi = episode if episode is not None else interval
    middle = 0.5 * (lo + hi)
    inside = [b for b in bangs if lo <= b <= hi]
    early = [b for b in inside if b < middle]
    late = [b for b in inside if b >= middle]
    new_start = min(early) if early else start
    new_end = max(late) if late else end
    return (new_start, new_end), ['bang' if early else 'dock', 'bang' if late else 'dock']
