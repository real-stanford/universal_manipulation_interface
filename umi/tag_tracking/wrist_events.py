"""
Episode events of a continuous take from the wrist camera IMUs and the head
camera's raw tag detections, without tag tracking:

- still: a gripper's gyro RMS over still_sec stays below gyro_still (rad/s);
  a gripper resting on the dock is motionless, a hand-held one never is.
- docked view: in a head frame, both cubes are in the dock: from single-tag 3D
  positions (no tag map), each cube center is within dock_distance of a dock
  tag (e.g. 46, 47). In 3D, grippers held above the dock (a bang right over it,
  task2: 55-71 cm above) are far from docked ones (~12-14 cm above, ~20 cm away).
- idle: both grippers still and, within that still period, at least
  min_dock_frames head frames with the docked view. Still periods without
  it (e.g. a gripper put down elsewhere) are reported, not idle.
- bang: acceleration spikes (gravity and hand motion removed by a short
  rolling median) in both wrist IMUs within bang_tolerance seconds of each
  other (audio synchronized clocks), the weaker one at least bang_accel (m/s^2).
  The wrist cameras' clocks drift apart slowly (task2: 1.5 ms/min, ~22 ms over
  20 min), so spikes are first paired within bang_search_tolerance, the
  wrist-to-wrist time difference of strong pairs (>= lag_accel) is fitted as a
  robust line over the take, and a pair is joint if within bang_tolerance of it.
  Simultaneity, not height, separates bangs from handling: a bang can be
  smaller than the jolt of lifting a gripper off the dock. All spikes of each
  gripper are kept (at least 50 ms apart) before matching, so a larger jolt
  nearby cannot hide a bang. Joint spikes closer than bang_merge_sec are one bang.
- bang sound: the click of a bang is heard by the head camera. audio_onset_ratio
  is the head audio's short-time (5 ms) high-frequency (> 2 kHz) level over
  its 1 s average; a click gives a high ratio at the bang time.
  The click follows the wrist spike by a lag (task1: 44 ms, IQR 41-50 ms;
  wrist IMU vs head audio timing, drifting slowly with the clocks), estimated
  per take by estimate_audio_lag as a robust line over time from strong joint
  spikes (>= lag_accel, independent of bang_accel: weak spikes pair with
  unrelated clicks) to the nearest click.
  A bang needs a click of at least audio_ratio at spike time + lag, within
  audio_tolerance; audio_ratio 0: not required.
  Lifting the grippers off the dock or setting them down also gives joint
  spikes with a clunk; those happen at the dock: a joint spike is not a bang
  if the head sees the docked view within dock_window seconds (task1: all
  such jolts had the docked view, cubes 0.13-0.20 m from the dock tags; no
  bang had it, cubes >= 0.5 m away or not in view).
- episode: between two idle periods, from the first to the last bang that is
  at least idle_margin seconds away from both idle periods (lifting off and
  setting down on the dock happen at the idle boundaries).
- wrist clock sync: the audio sync gives each video one offset for the
  whole take, but the GoPro clocks drift (task3: wrist to wrist 14 -> 34 ms
  over 28 min). Every wrist camera also hears each bang's click: per bang,
  click_offset locates the head audio around the head click in the wrist
  audio by envelope cross-correlation (the same sound by its shape, not the
  loudest one: the wrist microphone also hears the fingers and handling).
  Audio against audio, as the audio sync: the cameras' fixed audio delays
  cancel. Per wrist video, wrist_clock_line fits a robust line through these
  offsets over the take (bangs correlating below sync_min_correlation are not
  used, mismatches are outliers); its value at an episode is the episode's
  clock correction, the RMS of the inliers around it the sync accuracy.

"""
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.stats

DEFAULTS = {
    'still_sec': 1.0,
    'gyro_still': 0.04,
    'bang_accel': 5.0,
    'bang_tolerance': 0.02,
    'bang_search_tolerance': 0.06,
    'bang_merge_sec': 0.5,
    'idle_margin': 1.0,
    'audio_tolerance': 0.015,
    'audio_ratio': 5.0,
    'audio_lag_search': 0.3,
    'lag_accel': 10.0,
    'min_episode_sec': 3.0,
    'dock_window': 0.25,
    'dock_distance': 0.35,
    'min_dock_frames': 15,
    'sync_search': 0.1,
    # episode boundaries: 'dock' (between dock periods, docking trimmed after tracking) or 'bang'
    'boundary': 'dock',
    'sync_min_correlation': 0.5,
}


def runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    (start, stop) index ranges of the True runs of a boolean array.
    """
    edges = np.diff(np.concatenate([[0], np.asarray(mask, dtype=np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def sample_period(times: np.ndarray) -> float:
    return float(np.median(np.diff(times)))


def gyro_rms(times: np.ndarray, gyro: np.ndarray, window_sec: float) -> np.ndarray:
    """
    Centered rolling RMS of the gyro magnitude (rad/s).
    """
    n = max(1, int(round(window_sec / sample_period(times))))
    return np.sqrt(scipy.ndimage.uniform_filter1d(np.sum(gyro ** 2, axis=1), n, mode='nearest'))


def still_mask(times: np.ndarray, gyro: np.ndarray, still_sec: float, gyro_still: float) -> np.ndarray:
    return gyro_rms(times, gyro, still_sec) <= gyro_still


def accel_highpass(times: np.ndarray, accel: np.ndarray) -> np.ndarray:
    """
    Magnitude (m/s^2) of the acceleration minus its 0.1 s rolling median,
    which removes gravity and hand motion and keeps impacts.
    """
    n = max(3, int(round(0.1 / sample_period(times))) | 1)
    baseline = scipy.ndimage.median_filter(accel, size=(n, 1), mode='nearest')
    return np.linalg.norm(accel - baseline, axis=1)


SPIKE_MIN_DISTANCE_SEC = 0.05
# UMI gripper cube: 70 mm (CAD cube_size), its center 35 mm behind a face tag
UMI_CUBE_HALF_SIZE = 0.035
# corner noise assumed for the single-tag pose ambiguity test of the docked view
DOCK_POSE_SIGMA_PX = 0.5
AUDIO_HIGHPASS_HZ = 2000.0
AUDIO_HOP_SEC = 0.001
AUDIO_ENVELOPE_SEC = 0.005
AUDIO_BACKGROUND_SEC = 1.0


def audio_onset_ratio(audio: np.ndarray, sample_rate: int, start_time: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Times (1 ms steps, from start_time) and ratio of the high-pass audio level
    over AUDIO_ENVELOPE_SEC to its centered AUDIO_BACKGROUND_SEC average.
    """
    sos = scipy.signal.butter(4, AUDIO_HIGHPASS_HZ, 'highpass', fs=sample_rate, output='sos')
    x = scipy.signal.sosfilt(sos, audio)
    hop = max(1, int(round(AUDIO_HOP_SEC * sample_rate)))
    n = len(x) // hop
    level = np.sqrt((x[:n * hop] ** 2).reshape(n, hop).mean(axis=1))
    level = scipy.ndimage.uniform_filter1d(level, max(1, int(round(AUDIO_ENVELOPE_SEC / AUDIO_HOP_SEC))))
    background = scipy.ndimage.uniform_filter1d(level, int(round(AUDIO_BACKGROUND_SEC / AUDIO_HOP_SEC)))
    times = start_time + (np.arange(n) + 0.5) * hop / sample_rate
    return times, level / (background + 1e-12)


def audio_clicks(times: np.ndarray, ratio: np.ndarray, audio_ratio: float) -> np.ndarray:
    """
    Times of click peaks (ratio >= audio_ratio), at least 0.2 s apart.
    """
    distance = max(1, int(round(0.2 / AUDIO_HOP_SEC)))
    peaks, _ = scipy.signal.find_peaks(ratio, height=audio_ratio, distance=distance)
    return times[peaks]


def robust_line(t: np.ndarray, y: np.ndarray, min_points: int = 5, n_mad: float = 3.0,
        iterations: int = 3) -> Tuple[float, float, int]:
    """
    Line y = a + b * t fitted by least squares with repeated rejection of points
    beyond n_mad median absolute deviations: (a, b, inliers). Fewer than
    min_points: the median as a constant (b = 0); none: (0, 0, 0).
    """
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    if len(y) == 0:
        return 0.0, 0.0, 0
    if len(y) < min_points:
        return float(np.median(y)), 0.0, len(y)
    keep = np.ones(len(y), dtype=bool)
    for _ in range(iterations):
        b, a = np.polyfit(t[keep], y[keep], 1)
        residual = y - (a + b * t)
        mad = np.median(np.abs(residual[keep])) * 1.4826 + 1e-6
        keep = np.abs(residual) <= n_mad * mad
        if keep.sum() < min_points:
            return float(np.median(y)), 0.0, len(y)
    b, a = np.polyfit(t[keep], y[keep], 1)
    return float(a), float(b), int(keep.sum())


def estimate_audio_lag(spike_times: np.ndarray, click_times: np.ndarray, search: float,
        min_pairs: int = 5) -> Tuple[Optional[Tuple[float, float]], int]:
    """
    Time from joint wrist spikes to the nearest head audio click within
    +-search seconds, as a robust line over the take (clocks drift: the lag
    changes slowly): ((a, b), pairs) with lag(t) = a + b * t; None if fewer
    than min_pairs pairs.
    """
    times, lags = list(), list()
    if len(click_times) > 0:
        for t in spike_times:
            j = int(np.argmin(np.abs(click_times - t)))
            if abs(click_times[j] - t) <= search:
                times.append(t)
                lags.append(click_times[j] - t)
    if len(lags) < min_pairs:
        return None, len(lags)
    a, b, _ = robust_line(np.array(times), np.array(lags), min_points=min_pairs)
    return (a, b), len(lags)


def bang_click(times: np.ndarray, ratio: np.ndarray, t: float, search: float, min_ratio: float) -> Optional[float]:
    """
    Time of the strongest audio click within +-search of t (ratio >= min_ratio), None if none.
    """
    a, b = np.searchsorted(times, [t - search, t + search])
    if b <= a:
        return None
    k = a + int(np.argmax(ratio[a:b]))
    return float(times[k]) if ratio[k] >= min_ratio else None


# head audio around the head click used to find the same sound in a wrist recording
CLICK_TEMPLATE_SEC = (-0.01, 0.03)
CLICK_ENVELOPE_SEC = 0.0005
CLICK_FILTER_PAD_SEC = 0.01


def click_envelope(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    High-pass (> AUDIO_HIGHPASS_HZ) amplitude envelope, smoothed over CLICK_ENVELOPE_SEC.
    """
    sos = scipy.signal.butter(4, AUDIO_HIGHPASS_HZ, 'highpass', fs=sample_rate, output='sos')
    envelope = np.abs(scipy.signal.sosfilt(sos, audio.astype(np.float64)))
    return scipy.ndimage.uniform_filter1d(envelope, max(1, int(round(CLICK_ENVELOPE_SEC * sample_rate))))


def click_offset(head_audio: np.ndarray, head_audio_start: float, wrist_audio: np.ndarray,
        wrist_audio_start: float, sample_rate: int, t_click: float, search: float) -> Optional[Tuple[float, float]]:
    """
    Clock offset of a wrist recording at one bang: the head audio envelope
    around the head click (CLICK_TEMPLATE_SEC) is located in the wrist audio
    envelope within +-search by normalized cross-correlation (sub-sample peak).
    *_audio_start: head video clock time of each recording's first sample.
    Returns (offset, correlation): offset (s) = head clock time - wrist clock
    time of the same sound, i.e. what to add to the wrist video's times; None
    if the windows are outside a recording.
    """
    pad = CLICK_FILTER_PAD_SEC

    def segment(audio, audio_start, t0, t1):
        i0 = int(round((t0 - pad - audio_start) * sample_rate))
        i1 = int(round((t1 + pad - audio_start) * sample_rate))
        if i0 < 0 or i1 > len(audio):
            return None
        n = int(round(pad * sample_rate))
        envelope = click_envelope(audio[i0:i1], sample_rate)
        return envelope[n:len(envelope) - n], audio_start + (i0 + n) / sample_rate

    template = segment(head_audio, head_audio_start, t_click + CLICK_TEMPLATE_SEC[0], t_click + CLICK_TEMPLATE_SEC[1])
    window = segment(wrist_audio, wrist_audio_start, t_click + CLICK_TEMPLATE_SEC[0] - search,
        t_click + CLICK_TEMPLATE_SEC[1] + search)
    if template is None or window is None:
        return None
    (tpl, tpl_start), (win, win_start) = template, window
    tpl = tpl - tpl.mean()
    n = len(tpl)
    if len(win) < n + 2 or not np.any(tpl):
        return None
    corr = scipy.signal.correlate(win, tpl, mode='valid')
    sums = np.cumsum(np.concatenate([[0.0], win]))
    sums2 = np.cumsum(np.concatenate([[0.0], win ** 2]))
    s, s2 = sums[n:] - sums[:-n], sums2[n:] - sums2[:-n]
    ncc = corr / (np.sqrt(np.maximum(s2 - s * s / n, 1e-30)) * np.linalg.norm(tpl))
    k = int(np.argmax(ncc))
    delta = 0.0
    if 0 < k < len(ncc) - 1:
        denominator = ncc[k - 1] - 2 * ncc[k] + ncc[k + 1]
        if denominator < 0:
            delta = 0.5 * (ncc[k - 1] - ncc[k + 1]) / denominator
    wrist_time = win_start + (k + delta) / sample_rate
    return float(tpl_start - wrist_time), float(ncc[k])


def robust_drift_line(t: np.ndarray, y: np.ndarray, use: np.ndarray, n_mad: float = 3.0,
        min_sigma: float = 1e-4) -> Tuple[float, float, np.ndarray]:
    """
    Line y = a + b * t through the used points, robust to a large share of
    outliers (other sounds matched): Theil-Sen start (median of pairwise
    slopes), then points within n_mad robust standard deviations are refitted
    by least squares, twice. Returns (a, b, inlier mask).
    """
    if use.sum() >= 3:
        b, a, _, _ = scipy.stats.theilslopes(y[use], t[use])
    else:
        b, a = 0.0, float(np.median(y[use]))
    inlier = use.copy()
    for _ in range(2):
        residual = y - (a + b * t)
        sigma = 1.4826 * np.median(np.abs(residual[use]))
        inlier = use & (np.abs(residual) <= n_mad * max(sigma, min_sigma))
        if inlier.sum() >= 3:
            b, a = np.polyfit(t[inlier], y[inlier], 1)
        elif inlier.sum() > 0:
            b, a = 0.0, float(np.median(y[inlier]))
    residual = y - (a + b * t)
    sigma = 1.4826 * np.median(np.abs(residual[use]))
    inlier = use & (np.abs(residual) <= n_mad * max(sigma, min_sigma))
    return float(a), float(b), inlier


def wrist_clock_line(times: Sequence[float], offsets: Sequence[float], correlations: Sequence[float],
        min_correlation: float, n_mad: float = 3.0) -> Optional[dict]:
    """
    Clock offset of one wrist video over the take from its per-bang offsets
    (click_offset): a robust line offset(t) = a + b * t (GoPro clocks drift
    smoothly) through the bangs with correlation >= min_correlation; bangs
    beyond n_mad robust standard deviations (another sound matched) are outliers.
    residual_ms: RMS of the inliers around the line, the sync accuracy.
    None with fewer than 2 usable bangs.
    """
    t, y, c = (np.asarray(x, dtype=float) for x in (times, offsets, correlations))
    use = c >= min_correlation
    if use.sum() < 2:
        return None
    a, b, inlier = robust_drift_line(t, y, use, n_mad)
    residual = y - (a + b * t)
    return {
        'offset_ms': float(a * 1e3),
        'drift_ms_per_min': float(b * 60e3),
        'n_bangs': int(len(t)),
        'n_used': int(use.sum()),
        'n_inliers': int(inlier.sum()),
        'residual_ms': float(np.sqrt(np.mean(residual[inlier] ** 2)) * 1e3),
        'bangs': [{'time': float(ti), 'offset_ms': float(yi * 1e3), 'correlation': float(ci), 'inlier': bool(ii)}
            for ti, yi, ci, ii in zip(t, y, c, inlier)]
    }


def wrist_clock_offset_ms(line: dict, t: float) -> float:
    """
    Offset (ms) of the wrist video at head video time t from wrist_clock_line.
    """
    return line['offset_ms'] + line['drift_ms_per_min'] * t / 60.0


def max_near(times: np.ndarray, values: np.ndarray, t: float, tolerance: float) -> float:
    """
    Largest value within tolerance of t, 0 if none.
    """
    a, b = np.searchsorted(times, [t - tolerance, t + tolerance])
    return float(values[a:b].max()) if b > a else 0.0


def accel_peaks(times: np.ndarray, high: np.ndarray, min_height: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Times and heights of all peaks of accel_highpass above min_height,
    at least SPIKE_MIN_DISTANCE_SEC apart.
    """
    distance = max(1, int(round(SPIKE_MIN_DISTANCE_SEC / sample_period(times))))
    peaks, props = scipy.signal.find_peaks(high, height=min_height, distance=distance)
    return times[peaks], props['peak_heights']


def joint_spikes(times_a: np.ndarray, heights_a: np.ndarray, times_b: np.ndarray, heights_b: np.ndarray,
        tolerance: float) -> Dict[str, np.ndarray]:
    """
    Peaks of gripper a with a peak of gripper b within tolerance (s):
    time (mean of both), strength (the weaker height) and dt (b - a).
    """
    out = {'time': list(), 'strength': list(), 'dt': list()}
    if len(times_b) > 0:
        for t, h in zip(times_a, heights_a):
            j = int(np.argmin(np.abs(times_b - t)))
            if abs(times_b[j] - t) <= tolerance:
                out['time'].append((t + times_b[j]) / 2)
                out['strength'].append(min(h, heights_b[j]))
                out['dt'].append(times_b[j] - t)
    return {k: np.array(v) for k, v in out.items()}


def docked_near(times: np.ndarray, frame_times: np.ndarray, frame_docked: np.ndarray, window: float) -> np.ndarray:
    """
    Per time: whether any head frame within +-window shows the docked view.
    """
    docked_times = frame_times[frame_docked]
    out = np.zeros(len(times), dtype=bool)
    if len(docked_times) > 0:
        j = np.clip(np.searchsorted(docked_times, times), 1, len(docked_times) - 1)
        out = np.minimum(np.abs(docked_times[j] - times), np.abs(docked_times[j - 1] - times)) <= window
        if len(docked_times) == 1:
            out = np.abs(docked_times[0] - times) <= window
    return out


def merge_bangs(times: np.ndarray, strengths: np.ndarray, bang_accel: float, merge_sec: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Joint spikes with strength >= bang_accel; of spikes closer than merge_sec
    (one bang, several hits) the strongest is kept.
    """
    keep_t, keep_s = list(), list()
    for t, s in sorted(zip(times[strengths >= bang_accel], strengths[strengths >= bang_accel])):
        if len(keep_t) > 0 and t - keep_t[-1] < merge_sec:
            if s > keep_s[-1]:
                keep_t[-1], keep_s[-1] = t, s
            continue
        keep_t.append(t)
        keep_s.append(s)
    return np.array(keep_t), np.array(keep_s)


def docked_view(frame_corners: Dict[int, np.ndarray], cube_tags: Sequence[Sequence[int]],
        dock_tags: Sequence[int], tag_sizes: Dict[int, float], camera, dock_distance: float,
        cube_half_size: float = UMI_CUBE_HALF_SIZE) -> Tuple[bool, Optional[float]]:
    """
    Whether all cubes are in the dock, from single-tag 3D positions in the head
    camera (no tag map): each cube's center (cube_half_size behind a visible
    tag) within dock_distance (m) of the nearest visible dock tag. Positions
    only: a small single tag's orientation is ambiguous by ~10 deg, its
    position is accurate. Grippers held above the dock (e.g. a bang right over
    it) are farther and not docked. Also returns the largest cube distance (m);
    None if a cube or all dock tags are not seen (or their pose is ambiguous).
    """
    from umi.tag_tracking.initialization import unambiguous_tag_pose
    docks = list()
    for t in dock_tags:
        if t in frame_corners:
            T = unambiguous_tag_pose(camera, frame_corners[t], tag_sizes[t], DOCK_POSE_SIGMA_PX)
            if T is not None:
                docks.append(T[:3, 3])
    if len(docks) == 0:
        return False, None
    distances = list()
    for tags in cube_tags:
        centers = list()
        for t in tags:
            if t in frame_corners:
                T = unambiguous_tag_pose(camera, frame_corners[t], tag_sizes[t], DOCK_POSE_SIGMA_PX)
                if T is not None:
                    centers.append(T[:3, 3] - cube_half_size * T[:3, 2])
        if len(centers) == 0:
            return False, None
        c = np.median(centers, axis=0)
        distances.append(min(float(np.linalg.norm(c - d)) for d in docks))
    return max(distances) <= dock_distance, max(distances)


def idle_periods(times: np.ndarray, still_both: np.ndarray, frame_times: np.ndarray,
        frame_docked: np.ndarray, min_dock_frames: int) -> List[dict]:
    """
    Still periods of both grippers (IMU time base), each with the number of
    head frames showing the docked view and whether that confirms it as idle.
    """
    out = list()
    for a, b in runs(still_both):
        start, end = float(times[a]), float(times[b - 1])
        in_period = (frame_times >= start) & (frame_times <= end)
        n_docked = int(np.sum(frame_docked[in_period]))
        out.append({'start': start, 'end': end, 'n_docked_frames': n_docked,
            'idle': n_docked >= min_dock_frames})
    return out


def episodes_from_idle(idle: List[dict], take_end: float, min_episode_sec: float) -> List[dict]:
    """
    Dock to dock: every stretch between consecutive confirmed idle periods
    (and before the first / after the last) is an episode, no bangs needed;
    shorter than min_episode_sec: rejected (too_short). The docking motions
    at both ends are removed after tracking (03t_trim_docking.py).
    """
    confirmed = [x for x in idle if x['idle']]
    bounds = [-np.inf] + [x for p in confirmed for x in (p['start'], p['end'])] + [take_end]
    out = list()
    for start, end in zip(bounds[0::2], bounds[1::2]):
        start = float(max(start, 0.0))
        if end <= start:
            continue  # the take starts or ends docked
        item = {'active_start': start, 'active_end': float(end), 'n_bangs': 0}
        if end - start >= min_episode_sec:
            item.update({'start': start, 'end': float(end), 'status': 'ok'})
        else:
            item['status'] = 'too_short'
        out.append(item)
    return out


def episodes_from_events(idle: List[dict], bangs: np.ndarray, take_end: float, idle_margin: float,
        min_episode_sec: float) -> List[dict]:
    """
    Between consecutive confirmed idle periods (and before the first / after
    the last), the episode runs from the first to the last bang at least
    idle_margin away from the idle periods; fewer than two bangs or shorter
    than min_episode_sec (e.g. the start bang missed): rejected.
    """
    confirmed = [x for x in idle if x['idle']]
    bounds = [-np.inf] + [x for p in confirmed for x in (p['start'], p['end'])] + [take_end]
    out = list()
    for start, end in zip(bounds[0::2], bounds[1::2]):
        inside = bangs[(bangs > start + idle_margin) & (bangs < end - idle_margin)]
        item = {'active_start': float(max(start, 0.0)), 'active_end': float(end), 'n_bangs': int(len(inside))}
        if len(inside) >= 2 and inside[-1] - inside[0] >= min_episode_sec:
            item.update({'start': float(inside[0]), 'end': float(inside[-1]), 'status': 'ok'})
        elif len(inside) >= 2:
            item['status'] = 'too_short'
        else:
            item['status'] = 'no_bang' if len(inside) == 0 else 'one_bang'
        out.append(item)
    return out
