"""
First-order propagation of pose covariances in GTSAM Pose3 local
(right-perturbation) coordinates [rotation, translation]: T * Exp(xi).
"""
from typing import Tuple
import numpy as np
import gtsam


def inverse_covariance(T: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """
    Covariance of T^-1: (T Exp(xi))^-1 = T^-1 Exp(-Ad_T xi).
    """
    Ad = gtsam.Pose3(T).AdjointMap()
    return Ad @ covariance @ Ad.T


def compose_covariance(T_a: np.ndarray, cov_a: np.ndarray,
        T_b: np.ndarray, cov_b: np.ndarray) -> np.ndarray:
    """
    Covariance of T_a * T_b for independent T_a, T_b:
    T_a Exp(a) T_b Exp(b) = T_a T_b Exp(Ad_{T_b^-1} a) Exp(b).
    """
    Ad = gtsam.Pose3(T_b).inverse().AdjointMap()
    return Ad @ cov_a @ Ad.T + cov_b


def pose_std(T: np.ndarray, covariance: np.ndarray) -> Tuple[float, float]:
    """
    Position std (m) of the pose origin in the parent frame and rotation std (rad).
    """
    R = T[:3, :3]
    position_std = float(np.sqrt(np.trace(R @ covariance[3:, 3:] @ R.T)))
    rotation_std = float(np.sqrt(np.trace(covariance[:3, :3])))
    return position_std, rotation_std
