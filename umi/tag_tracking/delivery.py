"""
Per-frame data of an episode for delivery, without filling lost data.

- Native streams, per gripper: TCP poses at its own wrist camera frames (the
  fused trajectory of 03c_fuse_gripper_imu.py, head video clock with the wrist
  sync correction) and gripper widths at the wrist frames where finger tags
  were detected. Nothing interpolated.
- Aligned table: one row per head camera frame (the primary camera) in the
  episode's task interval. A gripper's pose / width at the head frame time is
  interpolated only between its two neighboring samples if both are valid and
  consecutive (at most max_spacing apart, 1.5 sample periods): alignment
  between frames of different cameras. Across lost samples the value is
  None and the row stays (R13 of the Odyssey spec). The head camera pose is
  its own, measured at that frame.

Poses: position (m) + quaternion xyzw, world frame (environment tag map).
"""
from typing import Dict, List, Optional, Sequence, Tuple
import json
import pathlib
import numpy as np
from scipy.spatial.transform import Rotation
from umi.common.cv_util import get_gripper_width, get_finger_center_x
from umi.common.exiftool_util import ExifTool
from umi.common.interpolation_util import get_gripper_calibration_interpolator

SPACING_FACTOR = 1.5


def pose7(T: np.ndarray) -> np.ndarray:
    """
    (N,4,4) -> (N,7) position xyz + quaternion xyzw; NaN rows stay NaN.
    """
    out = np.full((len(T), 7), np.nan)
    ok = ~np.isnan(T[:, 0, 0])
    if ok.any():
        out[ok, :3] = T[ok, :3, 3]
        out[ok, 3:] = Rotation.from_matrix(T[ok, :3, :3]).as_quat()
    return out


def align_poses(times: np.ndarray, T: np.ndarray, valid: np.ndarray, query: np.ndarray,
        max_spacing: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Poses at query times between the two neighboring samples (linear position,
    slerp rotation), only where both are valid and at most max_spacing apart
    (or the query hits a valid sample within 1 us). Returns (N,7) pose7
    (NaN where not aligned) and the aligned mask.
    """
    out = np.full((len(query), 7), np.nan)
    ok = np.zeros(len(query), dtype=bool)
    if len(times) < 2:
        return out, ok
    j = np.clip(np.searchsorted(times, query, side='right'), 1, len(times) - 1)
    i = j - 1
    exact = (np.abs(times[i] - query) < 1e-6) & valid[i]
    between = (times[i] <= query) & (query <= times[j]) & valid[i] & valid[j] & (times[j] - times[i] <= max_spacing)
    ok = exact | between
    # exact hits: the sample itself (its neighbor may be lost)
    e = np.flatnonzero(exact)
    out[e] = pose7(T[i[e]])
    k = np.flatnonzero(between & ~exact)
    if len(k):
        f = (query[k] - times[i[k]]) / np.maximum(times[j[k]] - times[i[k]], 1e-12)
        R0 = Rotation.from_matrix(T[i[k], :3, :3])
        R1 = Rotation.from_matrix(T[j[k], :3, :3])
        out[k, :3] = T[i[k], :3, 3] + f[:, None] * (T[j[k], :3, 3] - T[i[k], :3, 3])
        out[k, 3:] = (R0 * Rotation.from_rotvec(f[:, None] * (R0.inv() * R1).as_rotvec())).as_quat()
    return out, ok


def align_values(times: np.ndarray, values: np.ndarray, query: np.ndarray, max_spacing: float) -> np.ndarray:
    """
    Scalar samples (NaN = not measured) at query times, linear between two
    measured neighbors at most max_spacing apart; NaN elsewhere.
    """
    measured = ~np.isnan(values)
    t, v = times[measured], values[measured]
    out = np.full(len(query), np.nan)
    if len(t) < 1:
        return out
    j = np.clip(np.searchsorted(t, query, side='right'), 1, max(len(t) - 1, 1))
    i = j - 1
    if len(t) == 1:
        hit = np.abs(t[0] - query) < 1e-6
        out[hit] = v[0]
        return out
    exact = np.abs(t[i] - query) < 1e-6
    between = (t[i] <= query) & (query <= t[j]) & (t[j] - t[i] <= max_spacing)
    ok = exact | between
    f = np.where(exact, 0.0, (query - t[i]) / np.maximum(t[j] - t[i], 1e-12))
    out[ok] = (v[i] + f * (v[j] - v[i]))[ok]
    return out


def load_gripper_range(session: pathlib.Path, serial: str, gripper: Optional[Dict] = None) -> Dict:
    """
    gripper_range.json (05_run_calibrations.py) of a wrist camera: finger tag distance min / max (m).
    With the gripper's config entry: from its range_camera's calibration video if it has one.
    """
    serial = (gripper or dict()).get('range_camera', serial)
    with ExifTool() as et:
        for path in session.glob('demos/gripper_calibration*/gripper_range.json'):
            meta = list(et.get_metadata(str(path.parent.joinpath('raw_video.mp4'))))[0]
            if meta['QuickTime:CameraSerialNumber'] == serial:
                return json.load(path.open('r'))
    raise RuntimeError(f"no gripper_range.json for camera {serial}, run 05_run_calibrations.py")


def finger_widths(detections: Sequence[dict], left_tag: int, right_tag: int, nominal_z: float,
        gripper_range: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per wrist frame: gripper width (m, NaN if no finger tag detected), the
    number of finger tags it is from (2: both; 1: twice the tag's distance from
    the fingers' midpoint finger_center_x of the gripper range calibration, the
    camera axis if the calibration has none, as in UMI) and the midpoint of
    the two finger tags where both are detected (NaN elsewhere; for checking
    the calibrated midpoint). Width as in UMI: the finger opening relative to
    fully closed, finger tag distance - its closed value, 0 (closed) ..
    max_width - min_width of gripper_range.json (fully open).
    """
    center_x = gripper_range.get('finger_center_x', 0.0)
    calibrate = get_gripper_calibration_interpolator(
        aruco_measured_width=[gripper_range['min_width'], gripper_range['max_width']],
        aruco_actual_width=[gripper_range['min_width'], gripper_range['max_width']])
    widths = np.full(len(detections), np.nan)
    n_tags = np.zeros(len(detections), dtype=int)
    centers = np.full(len(detections), np.nan)
    for k, d in enumerate(detections):
        width = get_gripper_width(d['tag_dict'], left_tag, right_tag, nominal_z=nominal_z, center_x=center_x)
        if width is not None:
            widths[k] = float(calibrate(width))
            n_tags[k] = int(left_tag in d['tag_dict']) + int(right_tag in d['tag_dict'])
        c = get_finger_center_x(d['tag_dict'], left_tag, right_tag, nominal_z=nominal_z)
        if c is not None:
            centers[k] = c
    return widths, n_tags, centers


def episode_frames(head_times: np.ndarray, head_T_world_cam: np.ndarray, head_valid: np.ndarray,
        start: float, end: float, grippers: Sequence[dict], row_time_offset: float = 0.0) -> Dict:
    """
    One episode's native streams and aligned table, times in head video seconds.
    head_*: per head frame (tag_tracking.pkl); start / end: task interval (head video seconds).
    row_time_offset: a row's time after its head frame time, the instant its head pose
        refers to (rolling shutter corrected tracking: readout / 2, the middle row).
    grippers: per gripper {'side', 'pose_times', 'T_world_tcp', 'pose_valid', 'width_times',
        'widths' (NaN = not measured), 'width_tags'}, times on the head video clock.
    Returns {'frames': {'head_frame_index', 'time', 'camera_pose_head', 'camera_pose_head_valid',
        'ee_pose_<side>', 'gripper_<side>', 'pose_valid'}, 'native': {side: {'pose_time', 'pose', 'pose_valid',
        'width_time', 'width', 'width_tags'}}, 'stats': {...}}.
    """
    rows = np.flatnonzero((head_times >= start) & (head_times < end))
    query = head_times[rows] + row_time_offset
    head_T = head_T_world_cam[rows]
    head_ok = head_valid[rows] & ~np.isnan(head_T[:, 0, 0])
    frames = {
        'head_frame_index': rows,
        'time': query,
        'camera_pose_head': np.where(head_ok[:, None], pose7(head_T), np.nan),
        'camera_pose_head_valid': head_ok,
    }
    native, stats = dict(), dict()
    pose_valid = np.ones(len(rows), dtype=bool)
    for g in grippers:
        side = g['side']
        pt, pv = g['pose_times'], g['pose_valid'] & ~np.isnan(g['T_world_tcp'][:, 0, 0])
        period = float(np.median(np.diff(pt))) if len(pt) > 1 else np.inf
        pose, ok = align_poses(pt, g['T_world_tcp'], pv, query, SPACING_FACTOR * period)
        wt = g['width_times']
        w_period = float(np.median(np.diff(wt))) if len(wt) > 1 else np.inf
        width = align_values(wt, g['widths'], query, SPACING_FACTOR * w_period)
        frames[f'ee_pose_{side}'] = pose
        frames[f'gripper_{side}'] = width
        pose_valid &= ok
        in_pose = (pt >= start) & (pt < end)
        in_width = (wt >= start) & (wt < end)
        native[side] = {
            'pose_time': pt[in_pose], 'pose': pose7(np.where(pv[in_pose, None, None], g['T_world_tcp'][in_pose], np.nan)),
            'pose_valid': pv[in_pose],
            'width_time': wt[in_width], 'width': g['widths'][in_width], 'width_tags': g['width_tags'][in_width],
        }
        stats[side] = {'aligned_pose': float(ok.mean()) if len(ok) else 0.0,
            'aligned_width': float(np.mean(~np.isnan(width))) if len(width) else 0.0,
            'native_pose_valid': float(pv[in_pose].mean()) if in_pose.any() else 0.0,
            'native_width_measured': float(np.mean(~np.isnan(g['widths'][in_width]))) if in_width.any() else 0.0}
    frames['pose_valid'] = pose_valid
    stats['rows'] = int(len(rows))
    stats['pose_valid'] = float(pose_valid.mean()) if len(rows) else 0.0
    stats['head_pose_valid'] = float(head_ok.mean()) if len(rows) else 0.0
    return {'frames': frames, 'native': native, 'stats': stats}


def gripper_streams(session: pathlib.Path, config: Dict, videos: List[dict], head: dict, track: Dict,
        tcp: Dict, corrections: Dict[str, dict]) -> List[dict]:
    """
    Per gripper (robot order) the inputs of episode_frames from a processed session:
    TCP poses from the fused trajectory (tag_tracking.pkl grippers, 03c; head-only
    cube poses x T_cube_tcp where there is no fusion) and finger-tag widths of all
    its wrist videos, on the head video clock with the wrist sync correction.
    """
    import pickle
    from umi.tag_tracking.session import overlapping
    from umi.tag_tracking.wrist_sync import correction_seconds
    hardware = config['gripper_hardware']
    names = [name for _, name in sorted((b['robot_index'], name)
        for name, b in config['bodies'].items() if 'gripper_camera' in b)]
    sides = {0: 'right', 1: 'left'}
    out = list()
    for name in names:
        body = config['bodies'][name]
        serial = body['gripper_camera']
        T_wrist_tcp = np.array(tcp[name]['T_wrist_tcp'])
        if name in track.get('grippers', dict()):
            fused = track['grippers'][name]
            pose_times, T_world_tcp = fused['time'], fused['T_world_wrist'] @ T_wrist_tcp
            pose_valid, source = fused['is_valid'], 'visual_inertial'
        else:
            b = track['bodies'][name]
            pose_times, T_world_tcp = track['time'], b['T_world_body'] @ np.array(tcp[name]['T_cube_tcp'])
            pose_valid, source = b['is_valid_world'], 'head_camera_tags'
        gripper = hardware['grippers'][serial]
        gripper_range = load_gripper_range(session, serial, gripper)
        width_times, widths, width_tags, centers = list(), list(), list(), list()
        for wrist in overlapping(videos, head, serial):
            detections = pickle.load(wrist['dir'].joinpath('tag_detection.pkl').open('rb'))
            t = np.array([d['time'] for d in detections]) + wrist['start_timestamp'] - head['start_timestamp']
            if wrist['name'] in corrections:
                t = t + correction_seconds(corrections[wrist['name']], t)
            w, n, c = finger_widths(detections, gripper['left_finger_tag_id'], gripper['right_finger_tag_id'],
                hardware['nominal_z'], gripper_range)
            width_times.append(t)
            widths.append(w)
            width_tags.append(n)
            centers.append(c)
        order = np.argsort(np.concatenate(width_times))
        out.append({
            'name': name, 'side': sides.get(body['robot_index'], str(body['robot_index'])), 'pose_source': source,
            'pose_times': pose_times, 'T_world_tcp': T_world_tcp, 'pose_valid': pose_valid,
            'width_times': np.concatenate(width_times)[order], 'widths': np.concatenate(widths)[order],
            'width_tags': np.concatenate(width_tags)[order], 'gripper_range': gripper_range,
            # finger midpoint in this session's frames with both tags, against the calibrated one
            'session_center_x': float(np.nanmedian(np.concatenate(centers))) if np.any(~np.isnan(np.concatenate(centers))) else None,
            'T_cube_tcp': np.array(tcp[name]['T_cube_tcp']),
        })
    return out
