"""
Gripper tool center point (TCP) from tag observations, without CAD.

T_world_tcp = T_world_cube * T_cube_wrist * T_wrist_tcp

T_cube_wrist: the head camera observes a finger tag of the gripper while
the gripper's wrist camera observes the same tag at the same time (audio
synchronized). The finger tag pose in the wrist camera is measured close up;
the head's observation of its corners constrains the single unknown
T_cube_wrist through project(T_head_cube * T_cube_wrist * T_wrist_tag * corner),
solved by least squares over all synchronized frames.

T_wrist_tcp: UMI convention, TCP frame with the wrist camera's orientation.
Its origin is the midpoint of the gripper's two finger tag centers measured
by the wrist camera, moved along the camera's optical axis by the measured
distance from finger tags to fingertips (finger_tag_to_tip, gripper config).
"""
from typing import Dict, List, Optional
import inspect
import numpy as np
import gtsam
from umi.common.cv_util import get_gripper_width
from umi.tag_tracking.camera_model import FisheyeCamera
from umi.tag_tracking.tag_detection import get_tag_object_points
from umi.tag_tracking.factors import make_tag_factor, tag_corner_residual, HUBER_K
from umi.tag_tracking.uncertainty import inverse_covariance
from umi.tag_tracking.initialization import unambiguous_tag_pose, mean_transform, chi2_threshold
from umi.tag_tracking.noise_model import TagNoiseModel, single_tag_view, is_usable_view

W = gtsam.symbol('w', 0)


def wrist_tag_poses(camera: FisheyeCamera, detections: List[dict], tag_id: int,
        tag_size: float, sigma_px: float) -> np.ndarray:
    """
    (N,4,4) pose of a finger tag in the wrist camera per frame,
    NaN where not detected or ambiguous.
    detections: per frame dict with 'tag_dict' or 'corners' (tag id -> (4,2) corners)
    """
    poses = np.full((len(detections), 4, 4), np.nan)
    for i, frame in enumerate(detections):
        tags = frame['corners'] if 'corners' in frame else \
            {k: v['corners'] for k, v in frame['tag_dict'].items()}
        if tag_id in tags:
            T = unambiguous_tag_pose(camera, tags[tag_id], tag_size, sigma_px)
            if T is not None:
                poses[i] = T
    return poses


def calibrate_cube_to_wrist(head_camera: FisheyeCamera, pairs: List[dict],
        tag_size: float, noise_model: TagNoiseModel, tag_id: int) -> Dict:
    """
    pairs: synchronized frames with
        'T_head_cube': (4,4) tracked cube pose in the head camera
        'cube_covariance': (6,6) its covariance (GTSAM Pose3 local coordinates)
        'head_corners': (4,2) finger tag corners seen by the head
        'T_wrist_tag': (4,4) finger tag pose in the wrist camera

    The tracked cube pose is not exact: each frame's residual covariance is
    sigma_tag^2 I + J Sigma_cube J^T, J the residual Jacobian w.r.t. the cube pose.
    """
    object_points = get_tag_object_points(tag_size)
    # head views whose single-tag fit fails (e.g. fisheye image edge) have no usable sigma
    pairs = [p for p in pairs if is_usable_view(*single_tag_view(head_camera, p['head_corners'], tag_size))]
    assert len(pairs) > 0, "no usable head observation of the finger tag"

    # initial value: chordal mean of per frame estimates from unambiguous head views
    estimates = list()
    sigmas = list()
    for pair in pairs:
        sigma = noise_model.observation_sigma(head_camera, tag_id, pair['head_corners'], tag_size)
        sigmas.append(sigma)
        T_head_tag = unambiguous_tag_pose(head_camera, pair['head_corners'], tag_size, sigma)
        if T_head_tag is not None:
            estimates.append(np.linalg.inv(pair['T_head_cube']) @ T_head_tag
                @ np.linalg.inv(pair['T_wrist_tag']))
    assert len(estimates) > 0, "no unambiguous head observation of the finger tag"
    T_init = mean_transform(np.array(estimates))

    covariances = list()
    for pair, sigma in zip(pairs, sigmas):
        T_cube_head = np.linalg.inv(pair['T_head_cube'])
        _, J = tag_corner_residual(gtsam.Pose3(T_cube_head), gtsam.Pose3(T_init),
            gtsam.Pose3(pair['T_wrist_tag']), object_points, pair['head_corners'],
            head_camera.calibration, compute_jacobians=True)
        cov_cube_head = inverse_covariance(pair['T_head_cube'], pair['cube_covariance'])
        covariances.append(sigma ** 2 * np.eye(8) + J[0] @ cov_cube_head @ J[0].T)

    graph = gtsam.NonlinearFactorGraph()
    for pair, covariance in zip(pairs, covariances):
        noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(HUBER_K),
            gtsam.noiseModel.Gaussian.Covariance(covariance))
        graph.add(make_tag_factor(
            gtsam.Pose3(np.linalg.inv(pair['T_head_cube'])), W, gtsam.Pose3(pair['T_wrist_tag']),
            object_points, pair['head_corners'], head_camera.calibration, noise))
    values = gtsam.Values()
    values.insert(W, gtsam.Pose3(T_init))
    result = gtsam.LevenbergMarquardtOptimizer(graph, values).optimize()
    T_cube_wrist = result.atPose3(W).matrix()

    # normalized residuals, variance factor scaled covariance
    chi2 = list()
    for p, covariance in zip(pairs, covariances):
        r = tag_corner_residual(gtsam.Pose3(np.linalg.inv(p['T_head_cube'])), gtsam.Pose3(T_cube_wrist),
            gtsam.Pose3(p['T_wrist_tag']), object_points, p['head_corners'], head_camera.calibration)
        chi2.append(r @ np.linalg.solve(covariance, r))
    chi2 = np.array(chi2)
    variance_factor = float(chi2.sum() / (8 * len(pairs) - 6))
    covariance = gtsam.Marginals(graph, result).marginalCovariance(W) * max(variance_factor, 1.0)
    return {
        'T_cube_wrist': T_cube_wrist,
        'covariance': covariance,
        'variance_factor': variance_factor,
        'n_pairs': len(pairs),
        'n_init': len(estimates),
        'fraction_consistent': float(np.mean(chi2 <= chi2_threshold(8))),
        'init_change_deg': float(np.degrees(np.linalg.norm(gtsam.Rot3.Logmap(
            gtsam.Rot3(T_init[:3, :3]).between(gtsam.Rot3(T_cube_wrist[:3, :3])))))),
        'init_change_m': float(np.linalg.norm(T_init[:3, 3] - T_cube_wrist[:3, 3]))
    }


def wrist_to_tcp(left_tag_poses: np.ndarray, right_tag_poses: np.ndarray,
        finger_tag_to_tip: Optional[float], nominal_z: float,
        z_tolerance: float = inspect.signature(get_gripper_width).parameters['z_tolerance'].default) -> Dict:
    """
    T_wrist_tcp from the finger tag poses (N,4,4) seen by the wrist camera.
    Own finger tags are identified by depth within nominal_z +- z_tolerance,
    as in umi.common.cv_util.get_gripper_width (same tolerance), since a
    tag id can also appear on another gripper in view.
    """
    is_own = lambda poses: ~np.isnan(poses[:, 0, 0]) & (np.abs(poses[:, 2, 3] - nominal_z) <= z_tolerance)
    both = is_own(left_tag_poses) & is_own(right_tag_poses)
    assert both.any(), "both finger tags never seen together by the wrist camera"
    midpoints = (left_tag_poses[both, :3, 3] + right_tag_poses[both, :3, 3]) / 2
    origin = np.median(midpoints, axis=0)
    if finger_tag_to_tip is not None:
        origin = origin + np.array([0, 0, finger_tag_to_tip])
    T = np.eye(4)
    T[:3, 3] = origin
    return {
        'T_wrist_tcp': T,
        'n_frames': int(both.sum()),
        'midpoint_mad_m': (np.median(np.abs(midpoints - np.median(midpoints, axis=0)), axis=0)).tolist(),
        'finger_tag_to_tip': finger_tag_to_tip
    }
