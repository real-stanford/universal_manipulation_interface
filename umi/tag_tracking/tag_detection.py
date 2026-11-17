from typing import Dict, Optional, Set
import os
import cv2
import numpy as np
from umi.common.video_chunk_util import process_video_chunked


def get_tag_object_points(tag_size: float) -> np.ndarray:
    """
    (4,3) tag corners in the tag frame, in ArUco detection order,
    same convention as cv2.SOLVEPNP_IPPE_SQUARE (z out of the tag).
    """
    h = tag_size / 2
    return np.array([
        [-h, h, 0],
        [h, h, 0],
        [h, -h, 0],
        [-h, -h, 0]
    ], dtype=np.float64)


def detect_tag_corners(
        img: np.ndarray,
        aruco_dict: cv2.aruco.Dictionary,
        tag_ids: Optional[Set[int]] = None
        ) -> Dict[int, np.ndarray]:
    """
    Detect ArUco tags with sub-pixel corner refinement, same detector
    settings as umi.common.cv_util.detect_localize_aruco_tags.
    No single-tag pose is computed here: a single planar tag has
    two ambiguous poses, poses are estimated jointly downstream.

    Returns: tag id -> (4,2) corner pixel coordinates
    """
    param = cv2.aruco.DetectorParameters()
    param.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.detectMarkers(
        image=img, dictionary=aruco_dict, parameters=param)
    result = dict()
    if ids is None:
        return result
    for tag_id, tag_corners in zip(ids.flatten(), corners):
        tag_id = int(tag_id)
        if (tag_ids is None) or (tag_id in tag_ids):
            result[tag_id] = tag_corners.reshape(4, 2).astype(np.float64)
    return result


def make_corner_detector(dict_bytes: np.ndarray, marker_size: int, max_correction_bits: int,
        tag_ids: Optional[Set[int]]):
    """
    Per frame corner detector, built in each worker process
    (cv2.aruco.Dictionary is not picklable, it is rebuilt from its parts).
    """
    cv2.setNumThreads(1)
    aruco_dict = cv2.aruco.Dictionary(dict_bytes, marker_size, max_correction_bits)

    def detect(frame, time):
        return {'time': time, 'corners': detect_tag_corners(frame.to_ndarray(format='gray'), aruco_dict, tag_ids)}
    return detect


def detect_video_tag_corners(
        video_path: str,
        aruco_dict: cv2.aruco.Dictionary,
        tag_ids: Optional[Set[int]] = None,
        num_workers: Optional[int] = None
        ) -> list:
    """
    Tag corners for every frame of a video, on raw decoded frames
    (no rotation metadata applied, matching the camera calibration).
    Frames are processed in parallel time chunks on num_workers processes,
    default all cores (umi.common.video_chunk_util).
    Returns per frame {'frame_idx', 'time', 'corners'}.
    """
    video_path = os.path.expanduser(video_path)
    return process_video_chunked(video_path, make_corner_detector,
        (aruco_dict.bytesList, aruco_dict.markerSize, aruco_dict.maxCorrectionBits, tag_ids),
        num_workers=num_workers, desc=os.path.basename(os.path.dirname(os.path.abspath(video_path))))
