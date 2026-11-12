from typing import Union, Optional, List
import numpy as np
import gtsam

# Huber threshold in units of sigma with 95% asymptotic efficiency
# for Gaussian noise (Huber 1981), standard choice for M-estimators
HUBER_K = 1.345

PoseOrKey = Union[int, gtsam.Pose3]


def get_corner_noise_model(sigma_px: float, robust: bool = True):
    """
    Isotropic pixel noise on the 8 coordinates of one tag's 4 corners.
    """
    noise = gtsam.noiseModel.Isotropic.Sigma(8, sigma_px)
    if robust:
        noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(HUBER_K), noise)
    return noise


def tag_corner_residual(
        T_ref_cam: gtsam.Pose3,
        T_ref_body: gtsam.Pose3,
        T_body_tag: gtsam.Pose3,
        object_points: np.ndarray,
        measured: np.ndarray,
        calibration: gtsam.Cal3Fisheye,
        compute_jacobians: bool = False,
        corner_motions: Optional[List[gtsam.Pose3]] = None):
    """
    Reprojection residual of 4 tag corners observed by a camera:
    pixel = project(T_ref_cam^-1 * T_ref_body * E_c * T_body_tag * corner)
    E_c: known body motion from frame time to the corner's observation time
    (rolling shutter, see umi.tag_tracking.rolling_shutter), identity if None.

    Returns:
        residual: (8,) predicted minus measured pixels
        jacobians: (8,6) w.r.t. local coordinates of
            T_ref_cam, T_ref_body, T_body_tag, if compute_jacobians
    """
    camera = gtsam.PinholeCameraCal3Fisheye(gtsam.Pose3(), calibration)
    residual = np.zeros(8)
    J = [np.zeros((8, 6)) for _ in range(3)]
    for i, point_tag in enumerate(object_points):
        H_tag = np.zeros((3, 6), order='F')
        H_body = np.zeros((3, 6), order='F')
        H_pbody = np.zeros((3, 3), order='F')
        H_cam = np.zeros((3, 6), order='F')
        H_pref = np.zeros((3, 3), order='F')
        H_proj = np.zeros((2, 3), order='F')
        point_body = T_body_tag.transformFrom(point_tag, H_tag, np.zeros((3, 3), order='F'))
        H_compose = np.eye(6)
        T_ref_body_c = T_ref_body
        if corner_motions is not None:
            H_compose = np.zeros((6, 6), order='F')
            T_ref_body_c = T_ref_body.compose(corner_motions[i], H_compose, np.zeros((6, 6), order='F'))
        point_ref = T_ref_body_c.transformFrom(point_body, H_body, H_pbody)
        point_cam = T_ref_cam.transformTo(point_ref, H_cam, H_pref)
        rows = slice(2 * i, 2 * i + 2)
        try:
            pixel = camera.project(point_cam, np.zeros((2, 6), order='F'),
                H_proj, np.zeros((2, 9), order='F'))
        except RuntimeError:
            # corner behind the camera: same as GTSAM GenericProjectionFactor
            # with throwCheirality=false, constant error and zero Jacobian
            residual[rows] = 2.0 * calibration.fx()
            continue
        residual[rows] = pixel - measured[i]
        if compute_jacobians:
            J[0][rows] = H_proj @ H_cam
            J[1][rows] = H_proj @ H_pref @ H_body @ H_compose
            J[2][rows] = H_proj @ H_pref @ H_pbody @ H_tag
    if compute_jacobians:
        return residual, J
    return residual


def make_tag_factor(
        T_ref_cam: PoseOrKey,
        T_ref_body: PoseOrKey,
        T_body_tag: PoseOrKey,
        object_points: np.ndarray,
        measured: np.ndarray,
        calibration: gtsam.Cal3Fisheye,
        noise_model,
        corner_motions: Optional[List[gtsam.Pose3]] = None) -> gtsam.CustomFactor:
    """
    Tag corner projection factor (TagSLAM, Pfrommer & Daniilidis 2019).
    Each pose is either a variable key (int) or a known gtsam.Pose3.
    corner_motions: rolling shutter motion per corner, see tag_corner_residual.
    """
    poses = [T_ref_cam, T_ref_body, T_body_tag]
    is_key = [not isinstance(x, gtsam.Pose3) for x in poses]
    keys = [x for x, k in zip(poses, is_key) if k]
    assert len(keys) > 0

    def error_func(this: gtsam.CustomFactor, values: gtsam.Values, H):
        current = [values.atPose3(x) if k else x for x, k in zip(poses, is_key)]
        if H is None:
            return tag_corner_residual(*current, object_points, measured, calibration,
                corner_motions=corner_motions)
        residual, J = tag_corner_residual(*current, object_points, measured,
            calibration, compute_jacobians=True, corner_motions=corner_motions)
        for i, Ji in enumerate([J[j] for j in range(3) if is_key[j]]):
            H[i] = Ji
        return residual

    return gtsam.CustomFactor(noise_model, keys, error_func)
