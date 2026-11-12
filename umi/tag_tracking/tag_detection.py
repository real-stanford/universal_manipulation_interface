from typing import Dict, Optional, Set
import os
import av
import cv2
import numpy as np
from tqdm import tqdm


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


def detect_video_tag_corners(
        video_path: str,
        aruco_dict: cv2.aruco.Dictionary,
        tag_ids: Optional[Set[int]] = None,
        num_threads: int = 1
        ) -> list:
    """
    Tag corners for every frame of a video, on raw decoded frames
    (no rotation metadata applied, matching the camera calibration).
    """
    cv2.setNumThreads(num_threads)
    results = list()
    with av.open(os.path.expanduser(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        for i, frame in tqdm(enumerate(container.decode(stream)), total=stream.frames,
                desc=os.path.basename(os.path.dirname(video_path)), leave=False):
            img = frame.to_ndarray(format='gray')
            results.append({
                'frame_idx': i,
                'time': float(frame.pts * stream.time_base),
                'corners': detect_tag_corners(img, aruco_dict, tag_ids)
            })
    return results
