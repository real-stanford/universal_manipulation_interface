"""
Rigid bodies carrying several tags (TagSLAM bodies, Pfrommer & Daniilidis 2019),
e.g. a gripper's tag cube or the static environment.

Calibration estimates each tag's pose on the body by bundle adjustment
over all frames of a camera observing the body. Tracking estimates the
body pose in the camera frame for every frame from all visible tags jointly.

Frames are dicts with 'time' (sec), 'corners' (tag id -> (4,2) pixels)
and 'sequence' (video index; motion is never estimated across videos).
Every tag observation is weighted by its own corner noise from the
TagNoiseModel (viewing angle and tag dependent).
With a nonzero line delay, both run a second pass with rolling shutter
correction, the within-frame motion estimated from the first pass.
"""
from typing import Dict, List, Optional
import collections
import numpy as np
import gtsam
from umi.tag_tracking.camera_model import FisheyeCamera
from umi.tag_tracking.tag_detection import get_tag_object_points
from umi.tag_tracking.noise_model import TagNoiseModel, single_tag_view
from umi.tag_tracking.factors import (
    make_tag_factor,
    get_corner_noise_model,
    tag_corner_residual
)
from umi.tag_tracking.initialization import (
    chi2_threshold,
    unambiguous_tag_pose,
    mean_transform,
    body_pose_from_tags
)
from umi.tag_tracking.uncertainty import pose_std
from umi.tag_tracking.rolling_shutter import (
    corner_time_offsets,
    relative_twists,
    corner_motions
)

X = lambda i: gtsam.symbol('x', i)
L = lambda i: gtsam.symbol('l', i)


def get_observations(camera: FisheyeCamera, frames: List[dict], tag_ids, tag_sizes,
        noise_model: TagNoiseModel) -> List[Dict[int, dict]]:
    """
    Per frame: tag id -> {'corners', 'sigma', 'single_sse'} for tags of the body,
    gross errors dropped (see drop_gross_errors).
    """
    out = list()
    for frame in frames:
        obs = dict()
        for tag_id, corners in frame['corners'].items():
            if (tag_id in tag_ids) and (tag_id in noise_model.tag_scales):
                sse, view_cosine = single_tag_view(camera, corners, tag_sizes[tag_id])
                obs[tag_id] = {
                    'corners': corners,
                    'sigma': noise_model.sigma(tag_id, view_cosine),
                    'single_sse': sse
                }
        out.append(drop_gross_errors(obs))
    return out


def drop_gross_errors(obs: Dict[int, dict]) -> Dict[int, dict]:
    """
    In frames with several tags, drop tags whose single-tag fit is
    inconsistent with their own corner noise (chi-square, 8 - 6 dof),
    e.g. partially occluded tags, as long as another tag remains.
    A single tag is kept; its pose uncertainty decides its validity.
    """
    if len(obs) < 2:
        return obs
    consistent = {k: o for k, o in obs.items()
        if o['single_sse'] <= chi2_threshold(8 - 6) * o['sigma'] ** 2}
    return consistent if len(consistent) > 0 else obs


def get_frame_motions(camera: FisheyeCamera, frames: List[dict], T_cam_body: np.ndarray,
        line_delay: float) -> List[Optional[Dict[int, list]]]:
    """
    Rolling shutter motion of every tag corner in every frame.
    T_cam_body: (N,4,4) first pass estimates, NaN where missing.
    Returns per frame: tag id -> 4 Pose3 increments, None without correction.
    """
    motions = [None] * len(frames)
    if line_delay == 0:
        return motions
    sequences = collections.defaultdict(list)
    for i, frame in enumerate(frames):
        sequences[frame['sequence']].append(i)
    image_height = camera.image_size[1]
    for idxs in sequences.values():
        idxs = np.array(idxs)
        twists = relative_twists(T_cam_body[idxs], np.array([frames[i]['time'] for i in idxs]))
        for i, twist in zip(idxs, twists):
            motions[i] = {k: corner_motions(twist, corner_time_offsets(v, image_height, line_delay))
                for k, v in frames[i]['corners'].items()}
    return motions


def tag_observation_residual(camera, T_cam_body, T_body_tag, tag_size, corners, motions=None):
    return tag_corner_residual(gtsam.Pose3(), gtsam.Pose3(T_cam_body), gtsam.Pose3(T_body_tag),
        get_tag_object_points(tag_size), corners, camera.calibration, corner_motions=motions)


def initialize_tag_poses(camera: FisheyeCamera, observations: List[Dict[int, dict]],
        tag_sizes: Dict[int, float], anchor_tag: Optional[int]) -> Dict:
    """
    Tag poses on the body from relative poses between tags observed in
    the same frame, each unambiguous and fitting its corner noise, chained
    along the maximum co-observation spanning tree from the anchor tag.
    """
    relative = collections.defaultdict(list)
    n_seed = collections.Counter()
    for obs in observations:
        poses = dict()
        for tag_id, o in obs.items():
            T = unambiguous_tag_pose(camera, o['corners'], tag_sizes[tag_id], o['sigma'])
            if (T is not None) and (o['single_sse'] <= chi2_threshold(8 - 6) * o['sigma'] ** 2):
                poses[tag_id] = T
                n_seed[tag_id] += 1
        for a in poses:
            for b in poses:
                if a != b:
                    relative[(a, b)].append(np.linalg.inv(poses[a]) @ poses[b])
    if anchor_tag is None:
        # deterministic default: tag with most seed observations
        anchor_tag = max(n_seed, key=lambda x: (n_seed[x], -x))

    T_body_tags = {anchor_tag: np.eye(4)}
    while True:
        edges = [(len(v), a, b) for (a, b), v in relative.items()
            if (a in T_body_tags) and (b not in T_body_tags)]
        if len(edges) == 0:
            break
        _, a, b = max(edges)
        T_body_tags[b] = T_body_tags[a] @ mean_transform(np.array(relative[(a, b)]))
    return {'anchor_tag': anchor_tag, 'T_body_tags': T_body_tags, 'n_seed': dict(n_seed)}


def calibrate_rigid_body(camera: FisheyeCamera, frames: List[dict],
        tag_ids: List[int], tag_sizes: Dict[int, float], noise_model: TagNoiseModel,
        anchor_tag: Optional[int] = None, line_delay: float = 0.0) -> Dict:
    """
    Bundle adjustment of tag poses on a rigid body.
    The anchor tag defines the body frame (gauge). Only frames with at
    least two tags of the body constrain the relative tag poses.
    """
    all_observations = get_observations(camera, frames, tag_ids, tag_sizes, noise_model)
    init = initialize_tag_poses(camera, all_observations, tag_sizes, anchor_tag)
    anchor_tag = init['anchor_tag']
    calibrated_ids = sorted(init['T_body_tags'].keys())

    observations = dict()
    values = gtsam.Values()
    for tag_id in calibrated_ids:
        if tag_id != anchor_tag:
            values.insert(L(tag_id), gtsam.Pose3(init['T_body_tags'][tag_id]))
    for i, obs in enumerate(all_observations):
        obs = {k: v for k, v in obs.items() if k in init['T_body_tags']}
        if len(obs) < 2:
            continue
        T_cam_body, _ = body_pose_from_tags(camera, {k: v['corners'] for k, v in obs.items()},
            init['T_body_tags'], tag_sizes, {k: v['sigma'] for k, v in obs.items()})
        if T_cam_body is None:
            continue
        observations[i] = obs
        values.insert(X(i), gtsam.Pose3(T_cam_body))

    def optimize(values, motions):
        graph = gtsam.NonlinearFactorGraph()
        for i, obs in observations.items():
            for tag_id, o in obs.items():
                T_body_tag = gtsam.Pose3() if tag_id == anchor_tag else L(tag_id)
                graph.add(make_tag_factor(gtsam.Pose3(), X(i), T_body_tag,
                    get_tag_object_points(tag_sizes[tag_id]), o['corners'], camera.calibration,
                    get_corner_noise_model(o['sigma']),
                    None if motions[i] is None else motions[i][tag_id]))
        return gtsam.LevenbergMarquardtOptimizer(graph, values).optimize()

    def frame_poses(result):
        T_cam_body = np.full((len(frames), 4, 4), np.nan)
        for i in observations:
            T_cam_body[i] = result.atPose3(X(i)).matrix()
        return T_cam_body

    motions = [None] * len(frames)
    result = optimize(values, motions)
    if line_delay != 0:
        motions = get_frame_motions(camera, frames, frame_poses(result), line_delay)
        result = optimize(result, motions)
    T_body_tags = {k: (np.eye(4) if k == anchor_tag else result.atPose3(L(k)).matrix())
        for k in calibrated_ids}

    # normalized residuals and per frame consistency
    T_cam_body = frame_poses(result)
    normalized_sse = 0.0
    n_residuals = 0
    n_consistent = 0
    per_tag = collections.defaultdict(list)
    for i, obs in observations.items():
        frame_chi2 = 0.0
        for tag_id, o in obs.items():
            r = tag_observation_residual(camera, T_cam_body[i], T_body_tags[tag_id],
                tag_sizes[tag_id], o['corners'], None if motions[i] is None else motions[i][tag_id])
            frame_chi2 += np.sum(np.square(r)) / o['sigma'] ** 2
            per_tag[tag_id].append(np.sqrt(np.mean(np.square(r))))
        normalized_sse += frame_chi2
        n_residuals += 8 * len(obs)
        n_consistent += frame_chi2 <= chi2_threshold(8 * len(obs) - 6)
    n_params = 6 * len(observations) + 6 * (len(calibrated_ids) - 1)
    return {
        'anchor_tag': anchor_tag,
        'T_body_tags': T_body_tags,
        'missing_tags': sorted(set(tag_ids) - set(calibrated_ids)),
        'n_frames_used': len(observations),
        'n_frames_consistent': int(n_consistent),
        'variance_factor': float(normalized_sse / max(n_residuals - n_params, 1)),
        'tag_rms_px': {k: float(np.median(v)) for k, v in per_tag.items()},
        'n_seed': init['n_seed']
    }


def track_rigid_body(camera: FisheyeCamera, frames: List[dict],
        T_body_tags: Dict[int, np.ndarray], tag_sizes: Dict[int, float],
        noise_model: TagNoiseModel, line_delay: float = 0.0) -> Dict[str, np.ndarray]:
    """
    Per frame pose of a calibrated rigid body in the camera frame,
    estimated from all visible tags of the body jointly, each weighted
    by its own corner noise.

    The covariance is scaled by the a posteriori variance factor
    sum(SSE_k / sigma_k^2) / dof where it exceeds 1, so the actual
    reprojection residual, not the noise model alone, sets the uncertainty.

    Returns arrays over frames:
        T_cam_body: (N,4,4), NaN if not estimated
        covariance: (N,6,6) scaled, in GTSAM Pose3 local coordinates [rotation, translation]
        position_std: (N,) body origin position std in camera frame (m)
        rotation_std: (N,) rotation std (rad)
        variance_factor: (N,) a posteriori variance factor
        n_tags: (N,) tags of the body used
        is_consistent: (N,) chi-square test of the normalized residual passed (diagnostic)
        error: (N,) robust graph error, 0 if not estimated
    """
    n = len(frames)
    observations = get_observations(camera, frames, set(T_body_tags), tag_sizes, noise_model)

    def track(T_init, motions):
        out = {
            'T_cam_body': np.full((n, 4, 4), np.nan),
            'covariance': np.full((n, 6, 6), np.nan),
            'position_std': np.full(n, np.nan),
            'rotation_std': np.full(n, np.nan),
            'variance_factor': np.full(n, np.nan),
            'n_tags': np.array([len(x) for x in observations]),
            'is_consistent': np.zeros(n, dtype=bool),
            'error': np.zeros(n)
        }
        for i, obs in enumerate(observations):
            if np.isnan(T_init[i, 0, 0]):
                continue
            graph = gtsam.NonlinearFactorGraph()
            for tag_id, o in obs.items():
                graph.add(make_tag_factor(gtsam.Pose3(), X(0), gtsam.Pose3(T_body_tags[tag_id]),
                    get_tag_object_points(tag_sizes[tag_id]), o['corners'], camera.calibration,
                    get_corner_noise_model(o['sigma']),
                    None if motions[i] is None else motions[i][tag_id]))
            values = gtsam.Values()
            values.insert(X(0), gtsam.Pose3(T_init[i]))
            result = gtsam.LevenbergMarquardtOptimizer(graph, values).optimize()
            T = result.atPose3(X(0)).matrix()
            chi2 = sum(np.sum(np.square(tag_observation_residual(camera, T, T_body_tags[k],
                tag_sizes[k], o['corners'], None if motions[i] is None else motions[i][k]))) / o['sigma'] ** 2
                for k, o in obs.items())
            dof = 8 * len(obs) - 6
            variance_factor = chi2 / dof
            covariance = gtsam.Marginals(graph, result).marginalCovariance(X(0)) * max(variance_factor, 1.0)
            out['T_cam_body'][i] = T
            out['covariance'][i] = covariance
            out['position_std'][i], out['rotation_std'][i] = pose_std(T, covariance)
            out['variance_factor'][i] = variance_factor
            out['is_consistent'][i] = chi2 <= chi2_threshold(dof)
            out['error'][i] = graph.error(result)
        return out

    T_init = np.full((n, 4, 4), np.nan)
    for i, obs in enumerate(observations):
        T, _ = body_pose_from_tags(camera, {k: v['corners'] for k, v in obs.items()},
            T_body_tags, tag_sizes, {k: v['sigma'] for k, v in obs.items()})
        if T is not None:
            T_init[i] = T
    out = track(T_init, [None] * n)
    if line_delay != 0:
        motions = get_frame_motions(camera, frames, out['T_cam_body'], line_delay)
        out = track(out['T_cam_body'], motions)
    return out
