import json
import numpy as np
import cv2
import gtsam
from umi.common.cv_util import parse_fisheye_intrinsics


class FisheyeCamera:
    """
    Kannala-Brandt fisheye camera from an OpenImuCameraCalibrator json,
    usable both with OpenCV (cv2.fisheye) and GTSAM (Cal3Fisheye).
    Both implement the same projection model.
    """
    def __init__(self, intrinsics_json: str):
        # kept so worker processes can rebuild the camera (gtsam objects are not picklable)
        self.intrinsics_json = intrinsics_json
        intr = parse_fisheye_intrinsics(json.load(open(intrinsics_json, 'r')))
        self.image_size = tuple(int(x) for x in intr['DIM'])
        self.K = intr['K']
        self.D = intr['D']
        k = self.D.flatten()
        self.calibration = gtsam.Cal3Fisheye(
            self.K[0, 0], self.K[1, 1], 0.0, self.K[0, 2], self.K[1, 2],
            k[0], k[1], k[2], k[3])

    def undistort_normalized(self, pixels: np.ndarray) -> np.ndarray:
        """
        (N,2) pixel coordinates to (N,2) normalized image coordinates.
        """
        return cv2.fisheye.undistortPoints(
            pixels.reshape(-1, 1, 2).astype(np.float64), self.K, self.D).reshape(-1, 2)

    def project(self, points_cam: np.ndarray) -> np.ndarray:
        """
        (N,3) points in camera frame to (N,2) pixel coordinates.
        """
        pixels, _ = cv2.fisheye.projectPoints(
            points_cam.reshape(-1, 1, 3).astype(np.float64),
            np.zeros(3), np.zeros(3), self.K, self.D)
        return pixels.reshape(-1, 2)
