from typing import Dict, Optional, Sequence, Set
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


def tag_inventory(detections: Sequence[Dict[int, np.ndarray]], image_size: Sequence[int]) -> Dict[int, dict]:
    """
    Which tags a video shows, from per-frame detections (detect_tag_corners of
    sampled frames): per tag id the share of frames it is in, its median side
    length (px) and its median center in the image (0..1, x right, y down).
    Finger tags sit at a fixed spot near the bottom of a wrist image, cube and
    environment tags move around.
    """
    w, h = image_size
    seen = dict()
    for corners in detections:
        for tag_id, c in corners.items():
            side = float(np.mean(np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1)))
            center = c.mean(axis=0) / np.array([w, h])
            seen.setdefault(int(tag_id), list()).append((side, center[0], center[1]))
    out = dict()
    for tag_id in sorted(seen):
        x = np.array(seen[tag_id])
        out[tag_id] = {'share': len(x) / max(len(detections), 1), 'side_px': float(np.median(x[:, 0])),
            'center': (float(np.median(x[:, 1])), float(np.median(x[:, 2])))}
    return out


def sample_detections(path: str, predefined: str, every_sec: float, max_sec: Optional[float] = None) -> dict:
    """
    Tag corners (detect_tag_corners) of one frame every every_sec seconds of a
    video, up to max_sec. Returns {'duration': s, 'size': (w, h), 'detections': [...]},
    detections[i] at i * every_sec.
    """
    import av
    import av.logging
    av.logging.set_level(av.logging.ERROR)   # swscaler 'deprecated pixel format' per frame
    cv2.setNumThreads(1)
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, predefined))
    detections = list()
    # URLs (presigned S3): reconnect when the connection drops mid-read; a read that gets no data
    # for 30 s fails (rw_timeout, microseconds) instead of waiting forever, the caller retries
    options = {'reconnect': '1', 'reconnect_on_network_error': '1', 'reconnect_delay_max': '10',
        'rw_timeout': '30000000'} if str(path).startswith('http') else None
    with av.open(str(path), options=options) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        duration = float(stream.frames / stream.average_rate)
        size = (stream.codec_context.width, stream.codec_context.height)
        end = duration if max_sec is None else min(duration, max_sec)
        for t in np.arange(0.0, end, every_sec):
            container.seek(int(t / stream.time_base), stream=stream)
            for frame in container.decode(stream):
                if float(frame.pts * stream.time_base) >= t - 0.5 / float(stream.average_rate):
                    detections.append(detect_tag_corners(frame.to_ndarray(format='gray'), aruco_dict))
                    break
    return {'duration': duration, 'size': size, 'detections': detections}
