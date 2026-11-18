"""
Quality checks of the delivery (Odyssey spec 1.4, 1.9, 2.10), as our own
validation before Odyssey re-runs theirs. Every check gives a record
{'check', 'scope', 'status', 'value', 'limit', 'detail'}; status is 'pass',
'warn' (to look at, does not block), 'fail' (the spec's unusable / rejected
cases) or 'info' (reported only, e.g. idle time, which Odyssey also reports
without rejecting). An episode's validation_status is its worst check.

Covered here (the rest is checked by the delivery export: manifest
completeness, sha256, schema, required fields, LeRobot validator):
- media per video: native resolution (R2), frame rate measured on the camera
  clock (GPMF) against the container rate and an integer rate (R3), frame
  times uniform and no skipped / missing frames, keyframe interval <= 2 s (R7);
- images per camera and episode: no sustained frozen, near-black, saturated
  or low-contrast stretches, no duplicate adjacent frames;
- per-frame table (R8, R13): one row per head frame, contiguous indices,
  strictly increasing times, unit quaternions, null exactly where invalid;
- rig tracking: pose_valid share of the task frames per arm and the longest gap;
- gripper: width populated, inside the calibrated range, closes where the task grasps;
- motion plausibility: TCP speed, pose jumps that return, gripper speed,
  mostly static episodes;
- cross-stream sync: per wrist, the episode's own motion-sync correction
  against the session drift line, within the declared accuracy (R4);
- visibility: each gripper seen by the head camera during the task;
- episode length (R5) and the event timeline of annotations.json (2.10).
"""
from typing import Dict, List, Optional, Sequence
import numpy as np

DEFAULTS = {
    'min_episode_sec': 10.0,
    'min_size': {'head': (1280, 720), 'wrist': (640, 480)},
    'min_rate_hz': 15.0,
    'rate_tolerance': 1e-3,          # relative, measured vs container rate
    'integer_rate_tolerance_hz': 0.01,
    'frame_time_max_frac': 0.25,     # camera-clock frame time residual, fraction of a frame
    'max_keyframe_interval_sec': 2.0,
    'sustained_sec': 1.0,            # frozen / dark / exposure / contrast stretches
    'frozen_diff': 0.1,              # mean abs difference of adjacent frames (0..255, 160 px wide gray)
    'dark_mean': 16.0,
    'saturated_share': 0.5,
    'low_contrast_std': 6.0,
    'pose_valid_warn': 0.9,          # share of task frames
    'pose_valid_fail': 0.5,
    'width_valid_warn': 0.9,
    'width_valid_fail': 0.5,
    'width_margin': 0.005,           # m outside the calibrated range
    'max_tcp_speed': 3.0,            # m/s
    'jump': 0.05,                    # m, a step this large that comes back is a jump
    'max_width_speed': 1.0,          # m/s
    'static_path': 0.05,             # m, TCP path of both arms in the task below this: static episode
    'visible_warn': 0.5,             # share of task frames with the gripper seen by the head camera
    'sync_fail_factor': 3.0,         # x declared accuracy
}
ORDER = {'info': 0, 'pass': 1, 'warn': 2, 'fail': 3}


def record(check: str, scope: str, status: str, value=None, limit=None, detail: str = '') -> dict:
    return {'check': check, 'scope': scope, 'status': status, 'value': value, 'limit': limit, 'detail': detail}


def worst(checks: Sequence[dict]) -> str:
    """
    validation_status: worst of pass / warn / fail ('pass' when there are only info records).
    """
    s = max((ORDER[c['status']] for c in checks), default=ORDER['pass'])
    return [k for k, v in ORDER.items() if v == max(s, ORDER['pass'])][0]


def longest_run(mask: np.ndarray) -> int:
    """
    Length of the longest run of True.
    """
    best = run = 0
    for m in np.asarray(mask, dtype=bool):
        run = run + 1 if m else 0
        best = max(best, run)
    return best


# ---------------------------------------------------------------- media, per video
def media_checks(facts: dict, timing: Optional[dict], times: np.ndarray, keyframes: np.ndarray,
        role: str, params: Optional[Dict] = None) -> List[dict]:
    """
    One video: facts = umi.common.image_stats.stream_facts, timing =
    umi.common.gpmf_util.frame_timing, times / keyframes per decoded frame.
    role 'head' or 'wrist'.
    """
    p = {**DEFAULTS, **(params or {})}
    out = list()
    w, h = p['min_size'][role]
    ok = facts['width'] >= w and facts['height'] >= h
    out.append(record('resolution', role, 'pass' if ok else 'fail', f"{facts['width']}x{facts['height']}", f">= {w}x{h}"))
    period = 1.0 / facts['container_rate']
    if timing is None:
        out.append(record('frame_rate', role, 'warn', None, None, 'no per-frame GPMF stream: rate not measured'))
    else:
        rate = timing['rate_hz']
        relative = abs(rate / facts['container_rate'] - 1)
        status = 'fail' if rate < p['min_rate_hz'] or relative > p['rate_tolerance'] else \
            'warn' if abs(rate - round(rate)) > p['integer_rate_tolerance_hz'] else 'pass'
        out.append(record('frame_rate', role, status, round(rate, 4), f">= {p['min_rate_hz']}, integer, = container",
            f"camera clock {rate:.4f} Hz, container {facts['container_rate']:.4f} Hz"
            + (' (not an integer rate)' if status == 'warn' else '')))
        limit = p['frame_time_max_frac'] * period * 1e6
        out.append(record('frame_timing', role, 'pass' if timing['max_us'] <= limit else 'fail',
            round(timing['max_us'], 1), round(limit, 1), f"camera-clock frame time residual max (us), RMS {timing['rms_us']:.1f}"))
        missing = timing['n_frames'] - len(times)
        skipped = timing['skipped_mskp'] + timing['skipped_lskp']
        status = 'fail' if skipped or missing else 'pass'
        out.append(record('dropped_frames', role, status, {'skipped': skipped, 'camera_minus_decoded': int(missing)}, 0,
            ('skip counters ' + ('MSKP/LSKP' if timing['has_skip_counters'] else 'absent (not reported by the camera)'))))
    if len(times) > 1:
        dev = float(np.max(np.abs(np.diff(times) - period)))
        out.append(record('frame_intervals', role, 'pass' if dev <= 0.1 * period else 'fail',
            round(dev * 1e6, 1), round(0.1 * period * 1e6, 1), 'max deviation of the container frame interval (us)'))
    k = np.flatnonzero(keyframes)
    gaps = np.diff(np.concatenate([times[k], [times[-1] + period]])) if len(k) else np.array([np.inf])
    gap = float(gaps.max())
    out.append(record('keyframe_interval', role, 'pass' if gap <= p['max_keyframe_interval_sec'] else 'fail',
        round(gap, 3), p['max_keyframe_interval_sec'], 's'))
    out.append(record('encoding', role, 'info', {k: facts[k] for k in ('codec', 'profile', 'pix_fmt', 'color_range', 'bit_rate')}))
    return out


# ---------------------------------------------------------------- images, per camera and episode
def image_checks(stats: Dict[str, np.ndarray], fps: float, scope: str, params: Optional[Dict] = None) -> List[dict]:
    """
    stats = umi.common.image_stats.video_image_stats restricted to the episode's frames of one camera.
    """
    p = {**DEFAULTS, **(params or {})}
    n = int(round(p['sustained_sec'] * fps))
    if len(stats['mean']) == 0:
        return [record('images', scope, 'fail', 0, None, 'no frames of this camera in the episode')]
    diff = np.nan_to_num(stats['diff'][1:], nan=np.inf)
    out = list()
    for check, mask, detail in (
            ('frozen', diff < p['frozen_diff'], f"adjacent frames differing by < {p['frozen_diff']} levels"),
            ('dark', stats['mean'] < p['dark_mean'], f"mean brightness < {p['dark_mean']}"),
            ('overexposed', stats['saturated'] > p['saturated_share'], f"> {p['saturated_share']:.0%} saturated pixels"),
            ('low_contrast', stats['std'] < p['low_contrast_std'], f"brightness std < {p['low_contrast_std']}")):
        run = longest_run(mask)
        out.append(record(check, scope, 'fail' if run >= n else 'pass', round(run / fps, 3), p['sustained_sec'],
            f"longest stretch (s) of {detail}"))
    duplicates = int(np.sum(diff == 0))
    out.append(record('duplicate_frames', scope, 'warn' if duplicates else 'pass', duplicates, 0, 'identical adjacent frames'))
    return out


# ---------------------------------------------------------------- per-frame table
def frame_table_checks(frames: Dict[str, np.ndarray], expected_rows: int, fps: float, sides: Sequence[str]) -> List[dict]:
    """
    frames = umi.tag_tracking.delivery.episode_frames(...)['frames'] of the episode.
    """
    out = list()
    n = len(frames['time'])
    idx = frames['head_frame_index']
    contiguous = n == 0 or bool(np.all(np.diff(idx) == 1))
    out.append(record('rows', 'frames', 'pass' if n == expected_rows and contiguous else 'fail', n, expected_rows,
        'one row per head frame, contiguous frame indices'))
    dt = np.diff(frames['time'])
    ok = bool(np.all(dt > 0))
    dev = float(np.max(np.abs(dt - 1 / fps))) if len(dt) else 0.0
    out.append(record('timestamps', 'frames', 'pass' if ok and dev <= 0.1 / fps else 'fail', round(dev * 1e6, 1),
        round(0.1 / fps * 1e6, 1), 'strictly increasing, max frame interval deviation (us)'))
    for side in sides:
        pose = frames[f'ee_pose_{side}']
        valid = ~np.isnan(pose[:, 0])
        pattern = bool(np.all(np.isnan(pose).any(axis=1) == ~valid)) and bool(np.all(np.isnan(pose).all(axis=1) == ~valid))
        q = np.abs(np.linalg.norm(pose[valid, 3:], axis=1) - 1).max() if valid.any() else 0.0
        out.append(record('pose_values', side, 'pass' if pattern and q < 1e-6 else 'fail', float(q), 1e-6,
            'null exactly where invalid, unit quaternions (max norm error)'))
    return out


# ---------------------------------------------------------------- tracking, gripper, motion (task frames)
def tracking_checks(valid: np.ndarray, fps: float, side: str, params: Optional[Dict] = None) -> List[dict]:
    p = {**DEFAULTS, **(params or {})}
    share = float(valid.mean()) if len(valid) else 0.0
    status = 'fail' if share < p['pose_valid_fail'] else 'warn' if share < p['pose_valid_warn'] else 'pass'
    return [record('pose_valid', side, status, round(share, 4), p['pose_valid_warn'],
        f"share of task frames, longest gap {longest_run(~valid) / fps:.2f} s")]


def gripper_checks(width: np.ndarray, open_width: float, n_closes: int, side: str,
        params: Optional[Dict] = None) -> List[dict]:
    """
    width: aligned gripper opening (m, relative to fully closed) on the task
    frames, NaN where not measured; open_width: calibrated fully open opening;
    n_closes: measured closes of this gripper in the task.
    """
    p = {**DEFAULTS, **(params or {})}
    measured = ~np.isnan(width)
    share = float(measured.mean()) if len(width) else 0.0
    status = 'fail' if share < p['width_valid_fail'] else 'warn' if share < p['width_valid_warn'] else 'pass'
    out = [record('width_valid', side, status, round(share, 4), p['width_valid_warn'], 'share of task frames')]
    w = width[measured]
    outside = int(np.sum((w < -p['width_margin']) | (w > open_width + p['width_margin'])))
    out.append(record('width_range', side, 'pass' if outside == 0 else 'fail', outside, 0,
        f"frames outside 0..{open_width * 1000:.1f} mm (+-{p['width_margin'] * 1000:.0f} mm)"))
    out.append(record('gripper_closes', side, 'info', n_closes, None, 'measured closes in the task'))
    return out


def motion_checks(pose: np.ndarray, width: np.ndarray, fps: float, side: str, params: Optional[Dict] = None) -> List[dict]:
    """
    pose (n, 7) and width (n,) aligned on the task frames, NaN where invalid.
    Only consecutive valid frames are compared.
    """
    p = {**DEFAULTS, **(params or {})}
    out = list()
    valid = ~np.isnan(pose[:, 0])
    step = valid[1:] & valid[:-1]
    d = np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1)
    speed = float(np.max(d[step]) * fps) if step.any() else 0.0
    out.append(record('tcp_speed', side, 'pass' if speed <= p['max_tcp_speed'] else 'fail', round(speed, 3),
        p['max_tcp_speed'], 'max between consecutive frames (m/s)'))
    # a jump that returns: two large steps whose ends are close together
    three = valid[2:] & valid[1:-1] & valid[:-2]
    back = np.linalg.norm(pose[2:, :3] - pose[:-2, :3], axis=1)
    jumps = int(np.sum(three & (d[:-1] > p['jump']) & (d[1:] > p['jump']) & (back < 0.5 * p['jump'])))
    out.append(record('pose_jumps', side, 'pass' if jumps == 0 else 'fail', jumps, 0,
        f"steps > {p['jump'] * 100:.0f} cm that return the next frame"))
    ws = np.abs(np.diff(width)) * fps
    ws = ws[~np.isnan(ws)]
    top = float(ws.max()) if len(ws) else 0.0
    out.append(record('gripper_speed', side, 'pass' if top <= p['max_width_speed'] else 'warn', round(top, 3),
        p['max_width_speed'], 'max opening speed between consecutive frames (m/s)'))
    path = float(np.sum(d[step]))
    out.append(record('tcp_path', side, 'info', round(path, 3), None, 'TCP path length in the task (m)'))
    return out


def static_check(paths: Sequence[float], params: Optional[Dict] = None) -> dict:
    p = {**DEFAULTS, **(params or {})}
    total = float(sum(paths))
    return record('static_episode', 'episode', 'fail' if total < p['static_path'] else 'pass', round(total, 3),
        p['static_path'], 'TCP path of all arms in the task (m)')


def visibility_check(seen: np.ndarray, side: str, params: Optional[Dict] = None) -> dict:
    p = {**DEFAULTS, **(params or {})}
    share = float(seen.mean()) if len(seen) else 0.0
    return record('visible_in_scene', side, 'pass' if share >= p['visible_warn'] else 'warn', round(share, 4),
        p['visible_warn'], 'share of task frames with the gripper seen by the head camera')


# ---------------------------------------------------------------- sync, length, timeline
def sync_check(line: Optional[dict], accuracy_ms: Optional[float], start: float, side: str,
        params: Optional[Dict] = None) -> dict:
    """
    line: the wrist's drift line (wrist_sync.json videos/<head>/<wrist>/line)
    with its per-episode fits; the episode's own correction against the line,
    within the declared accuracy.
    """
    p = {**DEFAULTS, **(params or {})}
    if line is None:
        return record('sync', side, 'fail', None, None, 'no wrist sync for this wrist video')
    fits = [x for x in line['episodes'] if abs(x['start'] - start) < 0.05]
    if not fits or not fits[0].get('inlier', False):
        return record('sync', side, 'warn', None, accuracy_ms,
            'no motion-sync fit of its own in this episode: corrected from the session drift line')
    x = fits[0]
    mid = 0.5 * (x['start'] + x['end'])
    residual = x['correction_ms'] - (line['offset_ms'] + line['drift_ms_per_min'] * mid / 60.0)
    status = 'pass' if abs(residual) <= accuracy_ms else \
        'warn' if abs(residual) <= p['sync_fail_factor'] * accuracy_ms else 'fail'
    return record('sync', side, status, round(residual, 3), round(accuracy_ms, 3),
        f"episode correction minus drift line (ms), fit sigma {x['sigma_ms']:.2f} ms")


def length_checks(episode_s: float, task_s: float, params: Optional[Dict] = None) -> List[dict]:
    p = {**DEFAULTS, **(params or {})}
    return [record('episode_length', 'episode', 'pass' if episode_s >= p['min_episode_sec'] else 'fail',
                round(episode_s, 2), p['min_episode_sec'], 'dock to dock (s)'),
            record('task_length', 'episode', 'info', round(task_s, 2), p['min_episode_sec'], 'task part (s)')]


def timeline_checks(annotation: Optional[dict], episode_s: float) -> List[dict]:
    """
    annotations.json layers (08 / 09): per arm events cover the episode without
    overlap, every time inside the episode, outcome present, failures
    consistent with the outcome, contact events from measured gripper events,
    idle time.
    """
    if annotation is None:
        return [record('annotations', 'episode', 'warn', None, None, 'not annotated')]
    out = list()
    eps = 1e-6
    by_arm = dict()
    for e in annotation['events']:
        by_arm.setdefault(e['arm'], list()).append(e)
    problems = list()
    for arm, events in by_arm.items():
        events = sorted(events, key=lambda e: e['start_s'])
        if abs(events[0]['start_s']) > eps or abs(events[-1]['end_s'] - episode_s) > 1e-3:
            problems.append(f"{arm}: covers {events[0]['start_s']:.3f}-{events[-1]['end_s']:.3f} s")
        for a, b in zip(events[:-1], events[1:]):
            if abs(a['end_s'] - b['start_s']) > eps:
                problems.append(f"{arm}: {'overlap' if b['start_s'] < a['end_s'] else 'gap'} at {a['end_s']:.3f} s")
    out.append(record('event_coverage', 'episode', 'fail' if problems else 'pass', len(problems), 0,
        '; '.join(problems) or 'events cover the episode per arm without overlap'))
    times = [('key_event', k['t_s']) for k in annotation['key_events']]
    times += [('recovery', r['failure_t_s']) for r in annotation['recoveries']]
    times += [('recovered', r['recovered_at_s']) for r in annotation['recoveries'] if r.get('recovered')]
    times += [('state_change', s['t_s']) for s in annotation.get('state_changes', [])]
    times += [('issue', s['t_s']) for s in annotation.get('issues', [])]
    outside = [f"{name} {t:.2f}" for name, t in times if t is None or t < 0 or t > episode_s]
    out.append(record('event_times', 'episode', 'fail' if outside else 'pass', len(outside), 0,
        ', '.join(outside) or 'all times inside the episode'))
    task = annotation['tasks'][0] if annotation['tasks'] else dict()
    outcome = task.get('outcome')
    out.append(record('outcome', 'episode', 'pass' if outcome in ('success', 'failure', 'partial', 'aborted', 'unknown')
        else 'fail', outcome))
    unrecovered = [r for r in annotation['recoveries'] if not r.get('recovered')]
    consistent = not (outcome == 'success' and unrecovered)
    out.append(record('labels_consistent', 'episode', 'pass' if consistent else 'warn', len(unrecovered), 0,
        'unrecovered failures in a successful episode' if not consistent else 'failures consistent with the outcome'))
    contact = [k for k in annotation['key_events'] if k['kind'] == 'contact']
    unmeasured = sum(1 for k in contact if k.get('measured_kind') is None)
    out.append(record('contacts_measured', 'episode', 'pass' if unmeasured == 0 else 'warn', unmeasured, 0,
        'contact events not from a measured gripper close / open'))
    for arm, events in by_arm.items():
        idle = sum(e['end_s'] - e['start_s'] for e in events if e.get('contribution') == 'idle')
        out.append(record('idle_time', arm, 'info', round(idle, 2), None, 'idle and reset spans (s)'))
    out.append(record('annotation_source', 'episode', 'info', annotation.get('annotation_source')))
    return out
