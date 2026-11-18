"""
umi.tag_tracking.factors.tag_corner_residual with camera_motions (rolling
shutter of a moving camera): analytic Jacobians against numerical ones.

python tests/test_factors_rolling_shutter.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gtsam
from umi.tag_tracking.factors import tag_corner_residual
from umi.tag_tracking.tag_detection import get_tag_object_points

CALIBRATION = gtsam.Cal3Fisheye(1200.0, 1200.0, 0.0, 1352.0, 1014.0, 0.05, 0.02, -0.02, 0.008)


def numerical(f, poses, k, eps=1e-6):
    base = f(poses)
    J = np.zeros((len(base), 6))
    for j in range(6):
        d = np.zeros(6)
        d[j] = eps
        p = list(poses)
        p[k] = poses[k].retract(d)
        J[:, j] = (f(p) - base) / eps
    return J


def test_camera_motions_jacobians():
    T_ref_cam = gtsam.Pose3(gtsam.Rot3.RzRyRx(0.1, -0.2, 0.05), gtsam.Point3(0.02, -0.01, -0.6))
    T_ref_body = gtsam.Pose3(gtsam.Rot3.RzRyRx(-0.05, 0.1, 0.2), gtsam.Point3(0.01, 0.03, 0.05))
    T_body_tag = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.0, 0.0, 0.0))
    points = get_tag_object_points(0.08)
    twist = np.array([0.3, -0.5, 0.8, 0.2, 0.1, -0.3])
    motions = [gtsam.Pose3.Expmap(twist * dt) for dt in (0.001, 0.003, 0.006, 0.009)]
    measured = np.zeros((4, 2))
    f = lambda p: tag_corner_residual(p[0], p[1], p[2], points, measured, CALIBRATION, camera_motions=motions)
    poses = [T_ref_cam, T_ref_body, T_body_tag]
    residual, J = tag_corner_residual(*poses, points, measured, CALIBRATION, compute_jacobians=True,
        camera_motions=motions)
    for k in range(3):
        assert np.allclose(J[k], numerical(f, poses, k), atol=1e-3 * max(1.0, np.abs(J[k]).max())), k
    # identity motions: same as without
    still = [gtsam.Pose3() for _ in range(4)]
    assert np.allclose(tag_corner_residual(*poses, points, measured, CALIBRATION, camera_motions=still),
        tag_corner_residual(*poses, points, measured, CALIBRATION))


if __name__ == '__main__':
    test_camera_motions_jacobians()
    print("all passed")
