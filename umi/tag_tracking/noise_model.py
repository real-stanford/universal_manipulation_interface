"""
Corner noise model of tag observations:
    sigma = s_tag * cos(theta)^-p
theta: angle between the tag normal and the camera ray to the tag.
Foreshortened tags localize corners worse; the exponent p is shared,
the scale s_tag absorbs each physical tag's quality (print, flatness).

Fitted from single-tag fits alone: under Gaussian corner noise the
best single-tag reprojection SSE / sigma^2 follows chi-square with
8 - 6 = 2 dof. Real tag errors are heavy tailed, so the model is fitted
by median regression of log SSE (exact L1 fit by linear programming),
which is unaffected by the tail.
"""
from typing import Dict, List
import numpy as np
import scipy.stats
import scipy.optimize
import scipy.sparse
from umi.tag_tracking.camera_model import FisheyeCamera
from umi.tag_tracking.initialization import tag_pose_candidates

CHI2_2_MEDIAN = scipy.stats.chi2.median(8 - 6)


def single_tag_view(camera: FisheyeCamera, corners: np.ndarray, tag_size: float):
    """
    Best single-tag fit: (SSE px^2, cosine of viewing angle).
    """
    T_cam_tag, sse = tag_pose_candidates(camera, corners, tag_size)[0]
    ray = T_cam_tag[:3, 3] / np.linalg.norm(T_cam_tag[:3, 3])
    return sse, float(abs(ray @ T_cam_tag[:3, 2]))


class TagNoiseModel:
    def __init__(self, exponent: float, tag_scales: Dict[int, float]):
        self.exponent = exponent
        self.tag_scales = tag_scales

    def sigma(self, tag_id: int, view_cosine: float) -> float:
        return self.tag_scales[tag_id] * view_cosine ** -self.exponent

    def observation_sigma(self, camera: FisheyeCamera, tag_id: int,
            corners: np.ndarray, tag_size: float) -> float:
        _, view_cosine = single_tag_view(camera, corners, tag_size)
        return self.sigma(tag_id, view_cosine)

    def to_dict(self) -> dict:
        return {'exponent': self.exponent,
            'tag_scales': {str(k): v for k, v in self.tag_scales.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> 'TagNoiseModel':
        return cls(data['exponent'], {int(k): v for k, v in data['tag_scales'].items()})

    @classmethod
    def fit(cls, camera: FisheyeCamera, frames: List[dict],
            tag_sizes: Dict[int, float]) -> 'TagNoiseModel':
        """
        Median regression: log SSE = log(s_tag^2 * median(chi2_2)) + 2p * (-log cos theta)
        """
        tag_ids = sorted({k for f in frames for k in f['corners'] if k in tag_sizes})
        column = {k: i for i, k in enumerate(tag_ids)}
        rows = list()
        y = list()
        for frame in frames:
            for tag_id, corners in frame['corners'].items():
                if tag_id not in column:
                    continue
                sse, view_cosine = single_tag_view(camera, corners, tag_sizes[tag_id])
                if sse <= 0 or view_cosine <= 0:
                    continue
                x = np.zeros(len(tag_ids) + 1)
                x[column[tag_id]] = 1
                x[-1] = -np.log(view_cosine)
                rows.append(x)
                y.append(np.log(sse))
        X = np.array(rows)
        y = np.array(y)

        # L1 regression as linear program: min sum(u + v) s.t. X b + u - v = y, u, v >= 0
        n, m = X.shape
        result = scipy.optimize.linprog(
            c=np.concatenate([np.zeros(m), np.ones(2 * n)]),
            A_eq=scipy.sparse.hstack([scipy.sparse.csr_matrix(X),
                scipy.sparse.identity(n), -scipy.sparse.identity(n)], format='csr'), b_eq=y,
            bounds=[(None, None)] * m + [(0, None)] * (2 * n),
            method='highs')
        assert result.success, result.message
        b = result.x[:m]
        tag_scales = {k: float(np.sqrt(np.exp(b[column[k]]) / CHI2_2_MEDIAN)) for k in tag_ids}
        return cls(exponent=float(b[-1] / 2), tag_scales=tag_scales)
