"""
Clock sync of wrist camera videos to the head camera video from motion.

The audio sync (06a) gives each video one start time for the whole take; the
GoPro clocks drift apart (task1/task3: -0.35..-0.45 ms/min right wrist,
+0.62..+0.75 ms/min left wrist, against the head), and a camera's audio is
not locked to its video, so audio cannot correct it. Motion can: during an
episode the head camera tracks the cube (tag_tracking.pkl, head video
clock) while the wrist gyro measures the same rotation (imu_data.json,
whose times follow the wrist video clock: GPMF STMP; the OpenICC camera-IMU
offset maps them to the wrist video).

Per episode and wrist video, the cube's mean body-frame angular velocity
between head frames k and k + interval (both tracked) is compared with the
gyro (rotated into the cube frame) averaged over the same interval shifted
by a time shift; the shift minimizing the squared difference (grid +
parabola) gives the episode's correction: the time to add to the wrist
video's times (relative to the audio sync start times) to put them on the
head video clock. Head rolling shutter: a cube observation is timed at
frame time + readout * (cube image row / image height), top row first
(SensorReadoutTime, confirmed on task1/task3: +9.3 +- 2.7 and +12.4 +- 3.6 ms
against 11 ms metadata).

Per wrist video, a robust drift line through the episodes' corrections
(GoPro clocks drift smoothly); the line gives the correction at any head
time, and the RMS of its inlier episodes is the measured accuracy
(task1/task3: 0.36-0.70 ms).
"""
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.spatial.transform import Rotation
from umi.tag_tracking.wrist_events import robust_drift_line

DEFAULTS = {
    'interval_frames': 6,      # head frames between the two poses of a rate (0.1 s at 59.94 fps)
    'search_ms': 100.0,        # +- range of the shift search around the audio sync
    'step_ms': 0.25,           # shift grid step, refined by a parabola
    'min_pairs': 30,           # rate pairs needed in an episode
}


def head_rotation_rates(times: np.ndarray, rows: np.ndarray, T_world_body: np.ndarray, valid: np.ndarray,
        interval: int, readout: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Mean body-frame angular velocity of the cube between head frames k and
    k + interval, both valid: (t0, t1, omega (N,3), mean image row in [0, 1]);
    frame times plus readout * row (rolling shutter).
    """
    k0 = np.flatnonzero(valid[:-interval] & valid[interval:])
    k1 = k0 + interval
    t0 = times[k0] + readout * rows[k0]
    t1 = times[k1] + readout * rows[k1]
    R0 = Rotation.from_matrix(T_world_body[k0, :3, :3])
    R1 = Rotation.from_matrix(T_world_body[k1, :3, :3])
    rotvec = (R0.inv() * R1).as_rotvec()
    return t0, t1, rotvec / (t1 - t0)[:, None], 0.5 * (rows[k0] + rows[k1])


def gyro_integral(times: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Cumulative integral of an angular velocity (trapezoid), for interval means.
    """
    dt = np.diff(times)
    steps = 0.5 * (omega[1:] + omega[:-1]) * dt[:, None]
    return np.vstack([np.zeros((1, 3)), np.cumsum(steps, axis=0)])


def mean_rate(times: np.ndarray, integral: np.ndarray, t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    a = np.stack([np.interp(t0, times, integral[:, j]) for j in range(3)], axis=1)
    b = np.stack([np.interp(t1, times, integral[:, j]) for j in range(3)], axis=1)
    return (b - a) / (t1 - t0)[:, None]


def fit_shift(t0: np.ndarray, t1: np.ndarray, omega_head: np.ndarray, rows: np.ndarray, gyro_times: np.ndarray,
        integral: np.ndarray, shifts: np.ndarray, min_pairs: int) -> Optional[dict]:
    """
    Correction of one episode: the shift of the gyro (head time t <-> gyro
    time t + shift) minimizing the summed squared rate difference, on the
    grid shifts (s) refined by a parabola. Returns {'correction' (s, to add to
    the wrist's times = -shift), 'sigma' (s, from the cost curvature), 'rms'
    (rad/s rate residual), 'motion_rms' (rad/s), 'n_pairs', 'row' (mean
    cube image row)}; None with too few pairs or the minimum at the grid edge.
    """
    inside = (t0 + shifts[0] > gyro_times[0]) & (t1 + shifts[-1] < gyro_times[-1])
    t0, t1, omega_head, rows = t0[inside], t1[inside], omega_head[inside], rows[inside]
    if len(t0) < min_pairs:
        return None
    cost = np.array([np.sum((omega_head - mean_rate(gyro_times, integral, t0 + s, t1 + s)) ** 2) for s in shifts])
    k = int(np.argmin(cost))
    if k == 0 or k == len(shifts) - 1:
        return None
    step = shifts[1] - shifts[0]
    c0, c1, c2 = cost[k - 1], cost[k], cost[k + 1]
    second = c0 - 2 * c1 + c2
    delta = 0.5 * (c0 - c2) / second if second > 0 else 0.0
    n = 3 * len(t0)
    sigma = np.sqrt(2 * (c1 / max(n - 1, 1)) / (second / step ** 2)) if second > 0 else np.inf
    return {'correction': float(-(shifts[k] + delta * step)), 'sigma': float(sigma),
        'rms': float(np.sqrt(c1 / n)), 'motion_rms': float(np.sqrt(np.mean(omega_head ** 2))),
        'n_pairs': int(len(t0)), 'row': float(np.mean(rows))}


def cube_image_rows(T_cam_body: np.ndarray, camera, image_height: int) -> np.ndarray:
    """
    Image row (0 top .. 1 bottom) of the cube body origin per head frame, 0.5 where not tracked.
    """
    rows = np.full(len(T_cam_body), 0.5)
    ok = ~np.isnan(T_cam_body[:, 0, 0])
    if ok.any():
        rows[ok] = camera.project(T_cam_body[ok, :3, 3])[:, 1] / image_height
    return np.clip(rows, 0.0, 1.0)


def sync_episodes(times: np.ndarray, rows: np.ndarray, T_world_body: np.ndarray, valid: np.ndarray,
        episodes: Sequence[dict], gyro_times: np.ndarray, gyro_cube: np.ndarray, readout: float,
        params: Dict) -> List[Tuple[dict, Optional[dict]]]:
    """
    fit_shift per episode ({'start', 'end'} head video seconds) the gyro covers.
    gyro_times: head video clock under the current sync; gyro_cube: (N,3) rad/s in the cube frame.
    """
    params = {**DEFAULTS, **params}
    shifts = np.arange(-params['search_ms'], params['search_ms'] + 0.5 * params['step_ms'], params['step_ms']) * 1e-3
    integral = gyro_integral(gyro_times, gyro_cube)
    out = list()
    for e in episodes:
        if not (gyro_times[0] < e['start'] and e['end'] < gyro_times[-1]):
            continue
        in_episode = valid & (times >= e['start']) & (times <= e['end'])
        t0, t1, omega, pair_rows = head_rotation_rates(times, rows, T_world_body, in_episode,
            params['interval_frames'], readout)
        out.append((e, fit_shift(t0, t1, omega, pair_rows, gyro_times, integral, shifts, params['min_pairs'])))
    return out


def drift_line(episode_fits: Sequence[Tuple[dict, Optional[dict]]]) -> Optional[dict]:
    """
    Robust line correction(t) = offset + drift * t (head video seconds) through
    the episodes' corrections. None with fewer than 3 fitted episodes.
    """
    fits = [(e, f) for e, f in episode_fits if f is not None]
    if len(fits) < 3:
        return None
    t = np.array([0.5 * (e['start'] + e['end']) for e, _ in fits])
    c = np.array([f['correction'] for _, f in fits])
    a, b, inlier = robust_drift_line(t, c, np.ones(len(t), dtype=bool))
    residual = c - (a + b * t)
    return {
        'offset_ms': float(a * 1e3),
        'drift_ms_per_min': float(b * 60e3),
        'residual_ms': float(np.sqrt(np.mean(residual[inlier] ** 2)) * 1e3),
        'n_episodes': int(len(t)),
        'n_inliers': int(inlier.sum()),
        'mean_row': float(np.mean([f['row'] for _, f in fits])),
        'episodes': [{'start': float(e['start']), 'end': float(e['end']), 'correction_ms': f['correction'] * 1e3,
            'sigma_ms': f['sigma'] * 1e3, 'rate_rms': f['rms'], 'motion_rms': f['motion_rms'],
            'n_pairs': f['n_pairs'], 'row': f['row'], 'inlier': bool(ok)}
            for (e, f), ok in zip(fits, inlier)]
    }


def fit_readout(samples: Sequence[Tuple[str, float, float, float]], n_sigma: float = 3.0) -> Optional[dict]:
    """
    Head rolling shutter readout from episode corrections computed with
    readout 0: correction0 = a_v + b_v * t - readout * row, one line per
    wrist video v (samples: (video, head time, correction0 s, mean row)),
    one readout; least squares with rejection beyond n_sigma robust standard deviations.
    """
    videos = sorted({v for v, *_ in samples})
    n = len(samples)
    if n < 2 * len(videos) + 3:
        return None
    A = np.zeros((n, 2 * len(videos) + 1))
    y = np.zeros(n)
    for i, (v, t, c, row) in enumerate(samples):
        j = videos.index(v)
        A[i, 2 * j], A[i, 2 * j + 1], A[i, -1] = 1.0, t / 60.0, -row
        y[i] = c * 1e3
    rows = -A[:, -1]
    keep = np.ones(n, dtype=bool)
    for _ in range(5):
        x, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
        residual = y - A @ x
        sigma = 1.4826 * np.median(np.abs(residual[keep]))
        new_keep = np.abs(residual) <= n_sigma * max(sigma, 0.05)
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    dof = max(int(keep.sum()) - A.shape[1], 1)
    covariance = np.sum(residual[keep] ** 2) / dof * np.linalg.inv(A[keep].T @ A[keep])
    return {'readout_ms': float(x[-1]), 'sigma_ms': float(np.sqrt(covariance[-1, -1])),
        'n_episodes': int(keep.sum()), 'row_min': float(rows[keep].min()), 'row_max': float(rows[keep].max())}


def correction_seconds(line: dict, head_time) -> np.ndarray:
    """
    Correction (s) to add to a wrist video's times at head video time(s).
    """
    return (line['offset_ms'] + line['drift_ms_per_min'] * np.asarray(head_time) / 60.0) * 1e-3


def wrist_corrections(sync: Optional[dict], head_name: str) -> Dict[str, dict]:
    """
    Drift lines per wrist video name of one head video from wrist_sync.json ({} without it).
    """
    if sync is None:
        return dict()
    return {name: x['line'] for name, x in sync['videos'].get(head_name, dict()).items() if x.get('line') is not None}
