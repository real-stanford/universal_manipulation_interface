"""
Visual-inertial camera trajectory from tags with known poses (marker map)
and the camera's IMU, by batch smoothing in GTSAM: tag corner projection
factors (TagSLAM) and IMU preintegration factors with bias random walk
(Forster et al. 2017, CombinedImuFactor), as in fiducial visual-inertial
motion capture (Neunert et al. 2016; He et al. 2018).

Variables per frame k: IMU pose X_k and velocity V_k in a gravity aligned
navigation frame, IMU bias B_k; and one pose G = T_nav_world aligning the
tag world frame with gravity (only its roll and pitch are observable,
yaw and translation are gauge and held by a prior).

Optionally, pose measurements of the camera in the world (e.g. a gripper's
wrist camera from its cube tracked by the head camera) add BetweenFactor(G, X_k)
with the measured T_world_imu, robust (Huber) with the measurement covariance.

Optionally, point tracks of static scene content (umi.tag_tracking.feature_tracks)
add landmarks L_j with projection factors, constraining the trajectory where
few tags are visible. Tracks are tested against the tag-inertial trajectory:
a track is static if its triangulated point reprojects within the point
noise (chi-square); tracks on moving content (grippers, hands, the wearer)
are rejected. The point noise is estimated from the accepted tracks.
"""
from typing import Dict, Optional, Tuple
import numpy as np
import scipy.stats
import gtsam
from umi.tag_tracking.camera_model import FisheyeCamera
from umi.tag_tracking.noise_model import TagNoiseModel
from umi.tag_tracking.tag_detection import get_tag_object_points
from umi.tag_tracking.factors import make_tag_factor, get_corner_noise_model, tag_corner_residual, HUBER_K
from umi.tag_tracking.rigid_body import get_observations
from umi.tag_tracking.uncertainty import pose_std
from umi.tag_tracking.initialization import chi2_threshold
from umi.tag_tracking.rolling_shutter import relative_twists, corner_motions

X = lambda k: gtsam.symbol('x', k)
V = lambda k: gtsam.symbol('v', k)
B = lambda k: gtsam.symbol('b', k)
L = lambda j: gtsam.symbol('l', j)
G = gtsam.symbol('g', 0)

STANDARD_GRAVITY = 9.80665
# ORB_SLAM3 LocalMapping::CreateNewMapPoints: rays must differ for a
# triangulated point (cosine of the parallax angle below this)
ORB_SLAM3_MAX_PARALLAX_COS = 0.9998


def preintegrate(params, imu_times, accel, gyro, t0, t1, bias):
    """
    Preintegrated IMU measurement over [t0, t1); each sample holds
    until the next sample time.
    """
    pim = gtsam.PreintegratedCombinedMeasurements(params, bias)
    i = max(int(np.searchsorted(imu_times, t0, side='right')) - 1, 0)
    while i < len(imu_times) and imu_times[i] < t1:
        start = max(imu_times[i], t0)
        end = min(imu_times[i + 1] if i + 1 < len(imu_times) else t1, t1)
        if end > start:
            pim.integrateMeasurement(accel[i], gyro[i], end - start)
        i += 1
    return pim


def triangulate(T_cams_nav: np.ndarray, normalized: np.ndarray) -> Optional[np.ndarray]:
    """
    Linear (DLT) triangulation from (m,4,4) camera-from-nav poses and (m,2)
    normalized image coordinates. None if the point is not in front of all cameras.
    """
    P = T_cams_nav[:, :3, :]
    A = np.concatenate([
        normalized[:, :1] * P[:, 2] - P[:, 0],
        normalized[:, 1:] * P[:, 2] - P[:, 1]])
    X = np.linalg.svd(A)[2][-1]
    if X[3] == 0:
        return None
    point = X[:3] / X[3]
    depth = np.einsum('ij,j->i', P[:, 2, :3], point) + P[:, 2, 3]
    return point if np.all(depth > 0) else None


def triangulate_tracks(camera: FisheyeCamera, tracks: Dict[str, np.ndarray],
        T_nav_cams: np.ndarray) -> Dict[int, Tuple[np.ndarray, float, int]]:
    """
    tracks: 'k' (M,) frame index into T_nav_cams, 'track' (M,), 'pixel' (M,2).
    Points are triangulated if their viewing rays have parallax, as in
    ORB_SLAM3 (ORB_SLAM3_MAX_PARALLAX_COS); without it the depth is undetermined.
    Returns track id -> (point in nav frame, reprojection sse in px^2, dof 2m - 3).
    """
    T_cams_nav = np.linalg.inv(T_nav_cams)
    order = np.argsort(tracks['track'], kind='stable')
    ids, starts = np.unique(tracks['track'][order], return_index=True)
    out = dict()
    for track_id, obs in zip(ids, np.split(order, starts[1:])):
        if len(obs) < 2:
            continue
        k = tracks['k'][obs]
        normalized = camera.undistort_normalized(tracks['pixel'][obs])
        rays = np.einsum('mij,mj->mi', T_nav_cams[k, :3, :3], np.hstack([normalized, np.ones((len(obs), 1))]))
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        if np.min(rays @ rays.T) >= ORB_SLAM3_MAX_PARALLAX_COS:
            continue
        point = triangulate(T_cams_nav[k], normalized)
        if point is None:
            continue
        points_cam = np.einsum('mij,j->mi', T_cams_nav[k, :3, :3], point) + T_cams_nav[k, :3, 3]
        sse = float(np.sum(np.square(camera.project(points_cam) - tracks['pixel'][obs])))
        out[int(track_id)] = (point, sse, 2 * len(obs) - 3)
    return out


def classify_static(sse: np.ndarray, dof: np.ndarray, initial_sigma: float) -> Tuple[np.ndarray, float]:
    """
    Static / moving classification of tracks by a two class mixture fitted
    with EM (as MLESAC, Torr and Zisserman 2000): static tracks' sse is
    sigma^2 chi-square(dof); moving tracks' sse has no scale, a log-uniform
    density over the observed range. Estimates sigma and the static
    fraction jointly; moving content can be the majority.
    Returns the static posterior per track and sigma.
    """
    log_range = np.log(sse.max()) - np.log(sse.min())
    p_moving = 1.0 / (sse * log_range)
    sigma2, static_fraction = initial_sigma ** 2, 0.5
    while True:
        p_static = scipy.stats.chi2.pdf(sse / sigma2, dof) / sigma2
        w = static_fraction * p_static / (static_fraction * p_static + (1 - static_fraction) * p_moving)
        new_sigma2 = np.sum(w * sse) / np.sum(w * dof)
        static_fraction = float(np.mean(w))
        if np.isclose(new_sigma2, sigma2):
            return w, float(np.sqrt(new_sigma2))
        sigma2 = new_sigma2


def fill_missing_poses(T: np.ndarray) -> np.ndarray:
    """
    Initial values: nearest estimated pose for frames without an estimate.
    """
    valid = np.flatnonzero(~np.isnan(T[:, 0, 0]))
    assert len(valid) > 0
    idx = valid[np.argmin(np.abs(np.arange(len(T))[:, None] - valid[None, :]), axis=1)]
    return T[idx]


def fuse_camera_trajectory(camera: FisheyeCamera, frames: list,
        T_world_tags: Dict[int, np.ndarray], tag_sizes: Dict[int, float],
        noise_model: TagNoiseModel, imu: Dict[str, np.ndarray], T_imu_cam: np.ndarray,
        imu_time_offset: float, imu_noise: Dict[str, float],
        T_world_cam_init: np.ndarray, compute_covariance: bool = True,
        feature_tracks: Optional[Dict[str, np.ndarray]] = None,
        pose_measurements: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
        line_delay: float = 0.0, reference_row: float = 0.0) -> Dict[str, np.ndarray]:
    """
    line_delay (s per image row): rolling shutter of the camera; a tag corner in
        row v is observed at frame time + (v - reference_row) * line_delay, the
        camera moving with its velocity at the frame (from the neighboring frames
        of T_world_cam_init). reference_row: the row read at the frame's time
        (0: frame times at the first row; image height / 2: at the middle row).
    pose_measurements: frame index -> (T_world_cam, 6x6 covariance in GTSAM
        Pose3 local coordinates [rotation, translation], right perturbation)
    frames: per frame dict with 'time' (camera clock, sec) and 'corners'
    T_world_cam_init: (N,4,4) per-frame tag poses, NaN where missing
    imu_time_offset: camera time = IMU time + offset
    feature_tracks: optional point tracks (umi.tag_tracking.feature_tracks)
    Returns per frame T_world_cam, covariance (Pose3 local coordinates),
    position_std, rotation_std, and fit statistics.
    """
    n_frames = len(frames)
    all_times = np.array([f['time'] for f in frames])
    imu_times = imu['time'] + imu_time_offset
    # frames covered by IMU data
    used = np.flatnonzero((all_times >= imu_times[0]) & (all_times <= imu_times[-1]))
    frames = [frames[i] for i in used]
    T_world_cam_init = T_world_cam_init[used]
    n = len(frames)
    times = all_times[used]
    E = gtsam.Pose3(T_imu_cam)
    observations = get_observations(camera, frames, set(T_world_tags), tag_sizes, noise_model)
    # rolling shutter: per observation, the camera's motion from the frame time to each corner's time
    motions = [dict() for _ in range(n)]
    if line_delay != 0:
        twists = relative_twists(T_world_cam_init, times)
        for k, obs in enumerate(observations):
            for tag_id, o in obs.items():
                motions[k][tag_id] = corner_motions(twists[k], (o['corners'][:, 1] - reference_row) * line_delay)

    params = gtsam.PreintegrationCombinedParams.MakeSharedU(STANDARD_GRAVITY)
    params.setGyroscopeCovariance(imu_noise['NoiseGyro'] ** 2 * np.eye(3))
    params.setAccelerometerCovariance(imu_noise['NoiseAcc'] ** 2 * np.eye(3))
    params.setBiasOmegaCovariance(imu_noise['GyroWalk'] ** 2 * np.eye(3))
    params.setBiasAccCovariance(imu_noise['AccWalk'] ** 2 * np.eye(3))
    params.setIntegrationCovariance(np.zeros((3, 3)))
    # uncertainty of the bias the measurements are preintegrated with: the bias
    # is estimated in the graph (B_k), so negligible. GTSAM's default is the
    # identity (1 m/s^2, 1 rad/s), which drowns the IMU noise above and makes
    # the IMU factors nearly uninformative (inflated covariances in tag gaps).
    params.setBiasAccOmegaInit(np.eye(6) * 1e-10)

    # initial values from per-frame tag poses, world frame as navigation frame
    T_world_imu = fill_missing_poses(T_world_cam_init) @ np.linalg.inv(T_imu_cam)
    velocities = np.gradient(T_world_imu[:, :3, 3], times, axis=0)
    values = gtsam.Values()
    values.insert(G, gtsam.Pose3())
    for k in range(n):
        values.insert(X(k), gtsam.Pose3(T_world_imu[k]))
        values.insert(V(k), velocities[k])
        values.insert(B(k), gtsam.imuBias.ConstantBias())

    graph = gtsam.NonlinearFactorGraph()
    # gauge: yaw and translation of the navigation frame are unobservable,
    # any prior holds them without affecting the solution
    gauge = np.array([np.inf, np.inf, 1.0, 1.0, 1.0, 1.0])
    graph.add(gtsam.PriorFactorPose3(G, gtsam.Pose3(),
        gtsam.noiseModel.Diagonal.Precisions(np.where(np.isinf(gauge), 0.0, 1.0 / gauge ** 2))))
    n_tag_factors = 0
    for k, obs in enumerate(observations):
        for tag_id, o in obs.items():
            # camera pose in nav frame = X_k * T_imu_cam, tags given in world: nav <- world via G
            graph.add(make_tag_factor(X(k), G, gtsam.Pose3(T_world_tags[tag_id]),
                get_tag_object_points(tag_sizes[tag_id]), o['corners'], camera.calibration,
                get_corner_noise_model(o['sigma']), camera_offset=E, camera_motions=motions[k].get(tag_id)))
            n_tag_factors += 1
    n_pose_factors = 0
    if pose_measurements is not None:
        # camera pose -> IMU pose: T_world_imu = T_world_cam * E^-1, covariance by Ad_E
        Ad_E = E.AdjointMap()
        k_of_used = {int(i): k for k, i in enumerate(used)}
        for i, (T_world_cam, covariance) in pose_measurements.items():
            if i not in k_of_used:
                continue
            noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(HUBER_K),
                gtsam.noiseModel.Gaussian.Covariance(Ad_E @ covariance @ Ad_E.T))
            # G = T_nav_world, X_k = T_nav_imu: G^-1 X_k = T_world_imu
            graph.add(gtsam.BetweenFactorPose3(G, X(k_of_used[i]),
                gtsam.Pose3(T_world_cam).compose(E.inverse()), noise))
            n_pose_factors += 1
    for k in range(n - 1):
        pim = preintegrate(params, imu_times, imu['accel'], imu['gyro'], times[k], times[k + 1],
            gtsam.imuBias.ConstantBias())
        graph.add(gtsam.CombinedImuFactor(X(k), V(k), X(k + 1), V(k + 1), B(k), B(k + 1), pim))

    result = gtsam.LevenbergMarquardtOptimizer(graph, values).optimize()

    feature_stats = None
    if feature_tracks is not None:
        # static scene points: alternate classifying tracks against the trajectory
        # and smoothing with the static points, until the static set repeats
        k_of_frame = np.full(n_frames, -1)
        k_of_frame[used] = np.arange(n)
        inside = k_of_frame[feature_tracks['frame']] >= 0
        tracks = {
            'k': k_of_frame[feature_tracks['frame'][inside]],
            'track': feature_tracks['track'][inside],
            'pixel': feature_tracks['pixel'][inside]
        }
        base_graph = graph
        sigma = feature_tracks['flow_back_dist']
        static, seen = None, list()
        while static not in seen:
            seen.append(static)
            T_nav_cams = np.array([result.atPose3(X(k)).compose(E).matrix() for k in range(n)])
            candidates = triangulate_tracks(camera, tracks, T_nav_cams)
            track_ids = np.array(list(candidates))
            posterior, sigma = classify_static(np.array([candidates[j][1] for j in track_ids]),
                np.array([candidates[j][2] for j in track_ids]), sigma)
            static = frozenset(track_ids[posterior > 0.5].tolist())

            point_noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(HUBER_K),
                gtsam.noiseModel.Isotropic.Sigma(2, sigma))
            graph = gtsam.NonlinearFactorGraph(base_graph)
            values = gtsam.Values(result)
            for key in result.keys():
                if gtsam.Symbol(key).chr() == ord('l'):
                    values.erase(key)
            n_point_factors = 0
            for k, track_id, pixel in zip(tracks['k'], tracks['track'], tracks['pixel']):
                if track_id in static:
                    graph.add(gtsam.GenericProjectionFactorCal3Fisheye(pixel.astype(np.float64),
                        point_noise, X(k), L(track_id), camera.calibration, E))
                    n_point_factors += 1
            for track_id in static:
                values.insert(L(track_id), candidates[track_id][0])
            result = gtsam.LevenbergMarquardtOptimizer(graph, values).optimize()
            feature_stats = {
                'n_tracks': len(np.unique(tracks['track'])),
                'n_triangulated': len(candidates),   # with parallax, in front of the cameras
                'n_static': len(static),
                'n_factors': n_point_factors,
                'sigma_px': sigma,
                'iterations': len(seen)
            }
            print(f"   points: {feature_stats}", flush=True)

    # tag fit: a posteriori variance factor of the normalized corner residuals
    T_nav_world = result.atPose3(G)
    # robust: median of per observation chi-square (8 residuals) over its median
    chi2 = list()
    for k, obs in enumerate(observations):
        T_nav_cam = result.atPose3(X(k)).compose(E)
        for tag_id, o in obs.items():
            r = tag_corner_residual(T_nav_cam, T_nav_world, gtsam.Pose3(T_world_tags[tag_id]),
                get_tag_object_points(tag_sizes[tag_id]), o['corners'], camera.calibration,
                camera_motions=motions[k].get(tag_id))
            chi2.append(np.sum(np.square(r)) / o['sigma'] ** 2)
    # no tag observations (e.g. pose measurements only): no tag fit to scale by
    variance_factor = float(np.median(chi2) / scipy.stats.chi2.median(8)) if len(chi2) > 0 else 1.0

    def to_frames(x, fill=np.nan):
        out = np.full((n_frames,) + x.shape[1:], fill, dtype=x.dtype)
        out[used] = x
        return out

    if not compute_covariance:
        T_world_nav = T_nav_world.inverse()
        return {
            'T_world_cam': to_frames(np.array([T_world_nav.compose(result.atPose3(X(k)).compose(E)).matrix()
                for k in range(n)])),
            'variance_factor': float(variance_factor),
            'gravity_alignment_deg': float(np.degrees(np.linalg.norm(gtsam.Rot3.Logmap(T_nav_world.rotation())))),
            'feature_stats': feature_stats
        }
    marginals = gtsam.Marginals(graph, result)
    Ad_inv_E = E.inverse().AdjointMap()
    out = {
        'T_world_cam': np.zeros((n, 4, 4)),
        'covariance': np.zeros((n, 6, 6)),
        'position_std': np.zeros(n),
        'rotation_std': np.zeros(n),
        'n_tags': np.array([len(x) for x in observations]),
        'gyro_bias': np.zeros((n, 3)),
        'accel_bias': np.zeros((n, 3))
    }
    T_world_nav = T_nav_world.inverse()
    for k in range(n):
        T_nav_imu = result.atPose3(X(k))
        T_nav_cam = T_nav_imu.compose(E)
        out['T_world_cam'][k] = T_world_nav.compose(T_nav_cam).matrix()
        # T_world_imu = G^-1 X_k is gauge invariant: its covariance from the joint
        # marginal of (G, X_k); camera pose covariance (right perturbation) via E
        H_g = np.zeros((6, 6), order='F')
        H_x = np.zeros((6, 6), order='F')
        T_nav_world.between(T_nav_imu, H_g, H_x)
        joint = marginals.jointMarginalCovariance(gtsam.KeyVector([G, X(k)]))
        J = np.hstack([H_g, H_x])
        cov_world_imu = J @ np.block([[joint.at(G, G), joint.at(G, X(k))],
            [joint.at(X(k), G), joint.at(X(k), X(k))]]) @ J.T
        covariance = Ad_inv_E @ cov_world_imu @ Ad_inv_E.T * max(variance_factor, 1.0)
        out['covariance'][k] = covariance
        out['position_std'][k], out['rotation_std'][k] = pose_std(out['T_world_cam'][k], covariance)
        bias = result.atConstantBias(B(k))
        out['gyro_bias'][k] = bias.gyroscope()
        out['accel_bias'][k] = bias.accelerometer()
    out = {k: to_frames(v, 0 if v.dtype.kind == 'i' else np.nan) for k, v in out.items()}
    out['variance_factor'] = float(variance_factor)
    out['gravity_alignment_deg'] = float(np.degrees(np.linalg.norm(gtsam.Rot3.Logmap(T_nav_world.rotation()))))
    out['feature_stats'] = feature_stats
    out['n_tag_factors'] = n_tag_factors
    out['n_pose_factors'] = n_pose_factors
    return out
