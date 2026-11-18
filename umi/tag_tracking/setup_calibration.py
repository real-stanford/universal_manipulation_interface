"""
Calibration of everything that is fixed for a setup (same tags, cubes,
grippers and camera settings), done once and reused by every session:

- environment tag map, gripper cube shapes, corner noise model and rolling
  shutter line delay (calibrate_bodies), from head camera tag corners;
- each gripper cube to its wrist camera (cube_to_wrist): in synchronized
  frames the head camera tracks the cube in the world (environment tags) and
  the wrist camera sees environment tags, so both are known in the world:
      T_cube_wrist = inv(T_world_cube) * T_world_wrist
  This transform is rigid, so a few hundred frames anywhere in a session
  suffice; per frame estimates are averaged robustly and their spread reported;
- wrist camera to TCP from the UMI gripper geometry (wrist_to_tcp_umi), as in
  scripts_slam_pipeline/06_generate_dataset_plan.py.
"""
from typing import Dict, List, Optional, Sequence, Tuple
import itertools
import time
import numpy as np
import scipy.optimize
from scipy.spatial.transform import Rotation, Slerp
from umi.tag_tracking.camera_model import FisheyeCamera
from umi.tag_tracking.noise_model import TagNoiseModel
from umi.tag_tracking.rigid_body import track_rigid_body
from umi.tag_tracking.parallel import parallel_calibrate_bodies, parallel_track_rigid_body, submit_track, gather_track
from umi.tag_tracking.initialization import mean_transform

# UMI gripper geometry (scripts_slam_pipeline/06_generate_dataset_plan.py)
UMI_CAM_TO_CENTER_HEIGHT = 0.086
UMI_CAM_TO_MOUNT_OFFSET = 0.01465
UMI_TCP_OFFSET = 0.205


def frames_with_neighbors(frames: List[dict], indices: Sequence[int], video_id) -> Tuple[List[dict], Dict[int, int]]:
    """
    Sampled frames with their immediate neighbors in the video. Each run of
    consecutive frames gets its own 'sequence', so the rolling shutter motion
    (umi.tag_tracking.rigid_body.get_frame_motions, from neighboring frames of
    a sequence) uses truly consecutive frames, never frames seconds apart.
    Returns the frames and original index -> position in the returned list.
    """
    keep = sorted({j for i in indices for j in (i - 1, i, i + 1) if 0 <= j < len(frames)})
    out, position = list(), dict()
    run = 0
    for n, j in enumerate(keep):
        if n > 0 and j != keep[n - 1] + 1:
            run += 1
        position[j] = len(out)
        out.append({'time': frames[j]['time'], 'corners': frames[j]['corners'], 'sequence': f"{video_id}:{run}"})
    return out, position


def select_cube_frames(frames: List[dict], tags: Sequence[int], per_pair: int) -> List[int]:
    """
    Indices of frames with at least two tags of a cube, at most per_pair
    evenly spread frames per tag pair, so rarely seen pairs (e.g. the face
    turned away from the head) are not outnumbered.
    """
    by_pair = {pair: list() for pair in itertools.combinations(sorted(tags), 2)}
    for i, frame in enumerate(frames):
        seen = sorted(t for t in tags if t in frame['corners'])
        for pair in itertools.combinations(seen, 2):
            by_pair[pair].append(i)
    selected = set()
    for indices in by_pair.values():
        if len(indices) > per_pair:
            indices = [indices[k] for k in np.linspace(0, len(indices) - 1, per_pair).round().astype(int)]
        selected.update(indices)
    return sorted(selected)


def calibrate_bodies(camera: FisheyeCamera, body_frames: Dict[str, List[dict]], body_configs: Dict[str, dict],
        tag_sizes: Dict[int, float], executor, n_chunks: int, readout_time: Optional[float] = None,
        log=print) -> Tuple[TagNoiseModel, float, Dict[str, dict]]:
    """
    Corner noise model from all frames, tag poses on each body from its own
    frames (bodies in parallel), rolling shutter line delay, then tag poses
    again with the line delay.
    Line delay: with the sensor readout_time (s, GoPro metadata
    SensorReadoutTime), line_delay = +readout_time / image rows, top row first:
    confirmed on the 2026-11-16 head camera by timing (wrist sync, 03s: fitted
    readout +9.3 +- 2.7 and +12.4 +- 3.6 ms against 11 ms), and the cubes'
    reprojection error on their fastest frames (check_line_delay.py). It is not
    chosen by the tracking error over all frames with bodies calibrated at
    delay 0: the tag geometry (above all the environment map from a steady
    mapping walk) then has absorbed the rolling shutter distortion and delay 0
    always looks best. Without readout_time: bounded search over |line_delay|
    <= frame period / image rows, the bodies recalibrated at every candidate.
    Tracking passes are split over the executor's workers (umi.tag_tracking.parallel).
    body_frames: body name -> frames ({'time', 'corners', 'sequence'})
    """
    t0 = time.time()
    all_frames = [f for frames in body_frames.values() for f in frames]
    noise_model = TagNoiseModel.fit(camera, all_frames, tag_sizes)
    log(f"  noise model: sigma = s_tag * cos(theta)^-{noise_model.exponent:.2f}, s_tag px "
        f"{ {k: round(v, 3) for k, v in sorted(noise_model.tag_scales.items())} } ({time.time() - t0:.0f}s)")

    def calibrate(line_delay):
        t = time.time()
        out = parallel_calibrate_bodies(camera, body_frames, body_configs, tag_sizes, noise_model, line_delay, executor)
        log(f"  bodies calibrated with line delay {line_delay * 1e6:.3f} us ({time.time() - t:.0f}s)")
        return out

    results = None
    image_rows = camera.image_size[1]

    def tracking_errors(line_delays):
        # all bodies and candidates submitted at once, sharing the workers
        t = time.time()
        submitted = {(k, name): submit_track(executor, camera, body_frames[name], x['T_body_tags'], tag_sizes,
            noise_model, d, n_chunks) for k, d in enumerate(line_delays) for name, x in results.items()}
        errors = [sum(gather_track(submitted[(k, name)])['error'].sum() for name in results)
            for k in range(len(line_delays))]
        for d, e in zip(line_delays, errors):
            log(f"  line delay {d * 1e6:7.3f} us: tracking error {e:.1f}")
        log(f"  ({len(line_delays)} tracking passes, {time.time() - t:.0f}s)")
        return errors

    if readout_time is not None:
        line_delay = readout_time / image_rows
        log(f"  readout {readout_time * 1e3:.1f} ms, top row first -> line delay {line_delay * 1e6:.3f} us")
    else:
        results = calibrate(0.0)
        diffs = [b['time'] - a['time'] for a, b in zip(all_frames[:-1], all_frames[1:]) if a['sequence'] == b['sequence']]
        max_line_delay = float(np.median(diffs)) / image_rows

        def error_at(r):
            # bodies calibrated at this line delay, then tracked with it
            nonlocal results
            results = calibrate(r * max_line_delay)
            return tracking_errors([r * max_line_delay])[0]

        opt = scipy.optimize.minimize_scalar(error_at, bounds=(-1, 1), method='bounded', options={'maxiter': 8})
        line_delay = float(opt.x * max_line_delay)
        log(f"  line delay {line_delay * 1e6:.3f} us (readout {abs(line_delay) * image_rows * 1e3:.2f} ms)")
    return noise_model, line_delay, calibrate(line_delay)


def wrist_to_tcp_umi(cam_to_center_height: float = UMI_CAM_TO_CENTER_HEIGHT,
        cam_to_mount_offset: float = UMI_CAM_TO_MOUNT_OFFSET, tcp_offset: float = UMI_TCP_OFFSET) -> np.ndarray:
    """
    T_wrist_tcp of the UMI gripper: TCP with the camera's orientation at
    (0, cam_to_center_height, cam_to_mount_offset + tcp_offset) in the camera frame.
    """
    T = np.eye(4)
    T[:3, 3] = [0.0, cam_to_center_height, cam_to_mount_offset + tcp_offset]
    return T


def interpolate_pose(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = (1 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([T0[:3, :3], T1[:3, :3]])))
    T[:3, :3] = slerp([alpha]).as_matrix()[0]
    return T


def transform_residuals(transforms: np.ndarray, T_mean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Translation (m) and rotation (rad) differences of each transform to T_mean.
    """
    d_t = np.linalg.norm(transforms[:, :3, 3] - T_mean[:3, 3], axis=1)
    d_r = Rotation.from_matrix(np.einsum('ji,njk->nik', T_mean[:3, :3], transforms[:, :3, :3])).magnitude()
    return d_t, d_r


def robust_mean_transform(transforms: np.ndarray, n_mad: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean transform after dropping estimates beyond n_mad median absolute
    deviations (translation or rotation) of the first mean. Returns the mean
    and the inlier mask.
    """
    T = mean_transform(transforms)
    d_t, d_r = transform_residuals(transforms, T)
    mad_t = np.median(d_t) * 1.4826 + 1e-6
    mad_r = np.median(d_r) * 1.4826 + 1e-6
    inliers = (d_t <= n_mad * mad_t) & (d_r <= n_mad * mad_r)
    return mean_transform(transforms[inliers]), inliers


def cube_to_wrist(head_camera: FisheyeCamera, head_frames: List[dict], head_times: np.ndarray,
        T_env_tags: Dict[int, np.ndarray], T_cube_tags: Dict[int, np.ndarray],
        tag_sizes: Dict[int, float], head_noise: TagNoiseModel, line_delay: float,
        wrist_camera: FisheyeCamera, wrist_frames: List[dict], wrist_times: np.ndarray,
        wrist_sigma_px: float, accuracy: dict, max_samples: int, max_speed: float,
        executor=None, n_chunks: int = 1, min_wrist_tags: int = 2) -> Optional[dict]:
    """
    T_cube_wrist from synchronized frames, all times on the head video clock.
    head_frames / wrist_frames: {'time', 'corners'} per frame (corners: tag id -> (4,2)).
    Samples: wrist frames with >= min_wrist_tags environment tags (at most
    max_samples, evenly spread); the cube pose in the world is interpolated
    between the two head frames around the wrist frame time, both valid, and
    the cube must move slower than max_speed (m/s) there.
    """
    is_accurate = lambda p, r: (p <= accuracy['position_std']) & (r <= accuracy['rotation_std_rad'])
    env_tags = set(T_env_tags)
    candidates = [i for i, f in enumerate(wrist_frames)
        if len(env_tags & set(f['corners'])) >= min_wrist_tags
        and head_times[0] <= wrist_times[i] < head_times[-1]]
    n_candidates = len(candidates)
    if len(candidates) > max_samples:
        candidates = [candidates[k] for k in np.linspace(0, len(candidates) - 1, max_samples).round().astype(int)]
    if len(candidates) == 0:
        return None

    # wrist camera pose from environment tags, constant corner noise of the wrist camera
    wrist_noise = TagNoiseModel(0.0, {t: wrist_sigma_px for t in env_tags})
    def track(camera, frames, T_tags, noise, delay):
        if executor is None:
            return track_rigid_body(camera, frames, T_tags, tag_sizes, noise, delay)
        return parallel_track_rigid_body(camera, frames, T_tags, tag_sizes, noise, delay, executor, n_chunks)

    wrist = track(wrist_camera, [wrist_frames[i] for i in candidates], T_env_tags, wrist_noise, 0.0)
    wrist_valid = is_accurate(wrist['position_std'], wrist['rotation_std']) & wrist['is_consistent']

    # head frames around each sample time (with neighbors for the rolling
    # shutter motion): environment (head pose) and cube
    k1 = np.clip(np.searchsorted(head_times, wrist_times[candidates], side='right'), 1, len(head_times) - 1)
    k0 = k1 - 1
    subset, position = frames_with_neighbors(head_frames, sorted(set(k0.tolist()) | set(k1.tolist())), 'head')
    env = track(head_camera, subset, T_env_tags, head_noise, line_delay)
    cube = track(head_camera, subset, T_cube_tags, head_noise, line_delay)

    def world_cube(k):
        j = position[k]
        if not (is_accurate(env['position_std'][j], env['rotation_std'][j])
                and is_accurate(cube['position_std'][j], cube['rotation_std'][j])):
            return None
        return np.linalg.inv(env['T_cam_body'][j]) @ cube['T_cam_body'][j]

    estimates, speeds = list(), list()
    n_wrist_invalid = n_head_invalid = n_fast = 0
    for s, i in enumerate(candidates):
        if not wrist_valid[s]:
            n_wrist_invalid += 1
            continue
        T0, T1 = world_cube(k0[s]), world_cube(k1[s])
        if T0 is None or T1 is None:
            n_head_invalid += 1
            continue
        dt = head_times[k1[s]] - head_times[k0[s]]
        speed = np.linalg.norm(T1[:3, 3] - T0[:3, 3]) / dt
        if speed > max_speed:
            n_fast += 1
            continue
        alpha = (wrist_times[i] - head_times[k0[s]]) / dt
        T_world_cube = interpolate_pose(T0, T1, alpha)
        T_world_wrist = np.linalg.inv(wrist['T_cam_body'][s])
        estimates.append(np.linalg.inv(T_world_cube) @ T_world_wrist)
        speeds.append(speed)
    stats = {'n_candidates': n_candidates, 'n_sampled': len(candidates), 'n_wrist_invalid': n_wrist_invalid,
        'n_head_invalid': n_head_invalid, 'n_too_fast': n_fast, 'n_estimates': len(estimates)}
    if len(estimates) < 10:
        return {'T_cube_wrist': None, 'stats': stats}

    estimates = np.array(estimates)
    T_cube_wrist, inliers = robust_mean_transform(estimates)
    d_t, d_r = transform_residuals(estimates[inliers], T_cube_wrist)
    n = int(inliers.sum())
    stats.update({
        'n_inliers': n,
        'spread_mm': float(np.median(d_t) * 1000),
        'spread_deg': float(np.degrees(np.median(d_r))),
        # standard error of the mean, from the inlier scatter
        'std_error_mm': float(np.sqrt(np.mean(d_t ** 2) / n) * 1000),
        'std_error_deg': float(np.degrees(np.sqrt(np.mean(d_r ** 2) / n))),
        'median_cube_speed': float(np.median(speeds)),
        'distance_cube_to_wrist_mm': float(np.linalg.norm(T_cube_wrist[:3, 3]) * 1000)
    })
    return {'T_cube_wrist': T_cube_wrist, 'stats': stats}
