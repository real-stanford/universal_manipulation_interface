from typing import List, Tuple, Optional
import numpy as np
import cv2
import scipy.stats
from umi.tag_tracking.camera_model import FisheyeCamera
from umi.tag_tracking.tag_detection import get_tag_object_points

# confidence level of all chi-square consistency tests
CHI2_CONFIDENCE = 0.95


def chi2_threshold(dof: int) -> float:
    return scipy.stats.chi2.ppf(CHI2_CONFIDENCE, dof)


def to_mat(rvec, tvec) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
    T[:3, 3] = np.asarray(tvec).flatten()
    return T


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]


def tag_reprojection_sse(camera: FisheyeCamera, T_cam_tag: np.ndarray,
        tag_size: float, corners: np.ndarray) -> float:
    points_cam = transform_points(T_cam_tag, get_tag_object_points(tag_size))
    if np.any(points_cam[:, 2] <= 0):
        return np.inf
    return float(np.sum(np.square(camera.project(points_cam) - corners)))


def tag_pose_candidates(camera: FisheyeCamera, corners: np.ndarray,
        tag_size: float) -> List[Tuple[np.ndarray, float]]:
    """
    Both IPPE solutions of a single planar tag (Collins & Bartoli 2014),
    sorted by reprojection sum of squared errors in pixels.
    """
    normalized = camera.undistort_normalized(corners)
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        get_tag_object_points(tag_size), normalized, np.eye(3), None,
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    candidates = list()
    for rvec, tvec in zip(rvecs, tvecs):
        T = to_mat(rvec, tvec)
        candidates.append((T, tag_reprojection_sse(camera, T, tag_size, corners)))
    return sorted(candidates, key=lambda x: x[1])


def unambiguous_tag_pose(camera: FisheyeCamera, corners: np.ndarray,
        tag_size: float, sigma_px: float) -> Optional[np.ndarray]:
    """
    Single-tag pose only if the planar ambiguity is resolved: the
    likelihood of the second IPPE solution, exp(-SSE / (2 sigma^2)), is
    at most (1 - CHI2_CONFIDENCE) / CHI2_CONFIDENCE of the best solution's.
    This is a relative test; the absolute fit of physical tags can exceed
    the corner noise without making the pose ambiguous.
    """
    candidates = tag_pose_candidates(camera, corners, tag_size)
    if len(candidates) < 2:
        return candidates[0][0] if len(candidates) == 1 else None
    (T_best, sse_best), (_, sse_second) = candidates[0], candidates[1]
    min_log_likelihood_ratio = np.log(CHI2_CONFIDENCE / (1 - CHI2_CONFIDENCE))
    if (sse_second - sse_best) / (2 * sigma_px ** 2) >= min_log_likelihood_ratio:
        return T_best
    return None


def mean_rotation(rotations: np.ndarray) -> np.ndarray:
    """
    Chordal L2 mean of rotation matrices (Hartley et al. 2013).
    """
    U, _, Vt = np.linalg.svd(np.sum(rotations, axis=0))
    S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ S @ Vt


def mean_transform(transforms: np.ndarray) -> np.ndarray:
    """
    Chordal mean rotation and median translation of (N,4,4) transforms.
    """
    T = np.eye(4)
    T[:3, :3] = mean_rotation(transforms[:, :3, :3])
    T[:3, 3] = np.median(transforms[:, :3, 3], axis=0)
    return T


def body_pose_from_tags(camera: FisheyeCamera, observations: dict,
        T_body_tags: dict, tag_sizes: dict, sigmas: dict
        ) -> Tuple[Optional[np.ndarray], int]:
    """
    Initial camera-from-body pose from all visible tags with known
    pose on the body. Candidate poses come from both IPPE solutions of
    each visible tag (largest tags first, as in TagSLAM); the candidate
    with the lowest total reprojection error over all visible tags is
    kept. A single visible tag is only used if unambiguous. Multi-tag
    consistency is tested after joint refinement.

    observations: tag id -> (4,2) corners
    sigmas: tag id -> corner noise of the observation (px)
    Returns: T_cam_body or None, number of tags used
    """
    tag_ids = [x for x in observations if x in T_body_tags]
    if len(tag_ids) == 0:
        return None, 0
    areas = {x: cv2.contourArea(observations[x].astype(np.float32)) for x in tag_ids}
    tag_ids = sorted(tag_ids, key=lambda x: -areas[x])

    def total_sse(T_cam_body):
        return sum(tag_reprojection_sse(camera, T_cam_body @ T_body_tags[x],
            tag_sizes[x], observations[x]) for x in tag_ids)

    best = None
    best_sse = np.inf
    for tag_id in tag_ids:
        for T_cam_tag, _ in tag_pose_candidates(camera, observations[tag_id], tag_sizes[tag_id]):
            T_cam_body = T_cam_tag @ np.linalg.inv(T_body_tags[tag_id])
            sse = total_sse(T_cam_body)
            if sse < best_sse:
                best, best_sse = T_cam_body, sse
    if len(tag_ids) == 1:
        # single tag: require the solution to be unambiguous
        T_cam_tag = unambiguous_tag_pose(camera, observations[tag_ids[0]],
            tag_sizes[tag_ids[0]], sigmas[tag_ids[0]])
        if T_cam_tag is None:
            return None, 1
        return T_cam_tag @ np.linalg.inv(T_body_tags[tag_ids[0]]), 1
    # multiple tags: consistency is tested after joint refinement
    return best, len(tag_ids)
