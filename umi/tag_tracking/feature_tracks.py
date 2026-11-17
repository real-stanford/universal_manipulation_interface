"""
Point tracks through a video for visual-inertial smoothing: Shi-Tomasi
corners tracked frame to frame by pyramidal Lucas-Kanade with a
forward-backward check, as the VINS-Mono / VINS-Fusion feature tracker
(Qin et al. 2018, feature_tracker.cpp). Its parameters are used as
published (EuRoC config), pixel distances scaled from its image width
to the tracking resolution.

Tracks are not filtered for static scene content here; moving points
(grippers, hands, the wearer) are rejected against the trajectory in
umi.tag_tracking.visual_inertial.
"""
from typing import Dict
import av
import cv2
import numpy as np
from tqdm import tqdm

# VINS-Fusion feature tracker (config/euroc, feature_tracker.cpp)
VINS_IMAGE_WIDTH = 752      # EuRoC image width the pixel parameters refer to
VINS_MAX_CNT = 150          # max_cnt: features kept per frame
VINS_MIN_DIST = 30          # min_dist: px between features
VINS_FLOW_BACK_DIST = 0.5   # px, forward-backward consistency
VINS_QUALITY_LEVEL = 0.01   # goodFeaturesToTrack quality level
VINS_WIN_SIZE = 21          # calcOpticalFlowPyrLK window
VINS_MAX_LEVEL = 3          # calcOpticalFlowPyrLK pyramid levels


def track_features(video_path: str, tracking_width: int = VINS_IMAGE_WIDTH) -> Dict[str, np.ndarray]:
    """
    Frames are tracked at tracking_width (aspect kept); pixel coordinates
    are returned at the video's resolution.
    Returns observations as flat arrays: 'frame' (M,), 'track' (M,),
    'pixel' (M,2); 'image_size' (w, h); and 'flow_back_dist' (px at video
    resolution), the forward-backward bound tracks were accepted with.
    """
    scale_px = tracking_width / VINS_IMAGE_WIDTH
    min_dist = int(round(VINS_MIN_DIST * scale_px))
    flow_back_dist = VINS_FLOW_BACK_DIST * scale_px
    lk = dict(winSize=(VINS_WIN_SIZE, VINS_WIN_SIZE), maxLevel=VINS_MAX_LEVEL)

    frames, tracks, pixels = list(), list(), list()
    prev_img = None
    prev_pts = np.zeros((0, 2), dtype=np.float32)
    prev_ids = np.zeros(0, dtype=np.int64)
    next_id = 0
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        w, h = stream.codec_context.width, stream.codec_context.height
        size = (tracking_width, int(round(h * tracking_width / w)))
        to_video = np.array([w / size[0], h / size[1]])
        for i, frame in tqdm(enumerate(container.decode(stream)), total=stream.frames, desc='tracking'):
            img = cv2.resize(frame.to_ndarray(format='gray'), size, interpolation=cv2.INTER_AREA)
            pts, ids = np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.int64)
            if len(prev_pts) > 0:
                fwd, status, _ = cv2.calcOpticalFlowPyrLK(prev_img, img, prev_pts, None, **lk)
                back, status_back, _ = cv2.calcOpticalFlowPyrLK(img, prev_img, fwd, prev_pts.copy(),
                    flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
                inside = (fwd[:, 0] >= 0) & (fwd[:, 0] <= size[0] - 1) & (fwd[:, 1] >= 0) & (fwd[:, 1] <= size[1] - 1)
                keep = (status[:, 0] == 1) & (status_back[:, 0] == 1) & inside \
                    & (np.linalg.norm(back - prev_pts, axis=1) <= flow_back_dist)
                pts, ids = fwd[keep], prev_ids[keep]
            # new features away from tracked ones, as VINS setMask
            n_new = VINS_MAX_CNT - len(pts)
            if n_new > 0:
                mask = np.full(img.shape, 255, dtype=np.uint8)
                for p in pts:
                    cv2.circle(mask, (int(round(p[0])), int(round(p[1]))), min_dist, 0, -1)
                new = cv2.goodFeaturesToTrack(img, n_new, VINS_QUALITY_LEVEL, min_dist, mask=mask)
                if new is not None:
                    new = new.reshape(-1, 2)
                    pts = np.concatenate([pts, new]).astype(np.float32)
                    ids = np.concatenate([ids, np.arange(next_id, next_id + len(new))])
                    next_id += len(new)
            frames.append(np.full(len(pts), i))
            tracks.append(ids)
            # pixel centers: x_video + 0.5 = (x + 0.5) * scale
            pixels.append((pts + 0.5) * to_video - 0.5)
            prev_img, prev_pts, prev_ids = img, pts, ids
    return {
        'frame': np.concatenate(frames),
        'track': np.concatenate(tracks),
        'pixel': np.concatenate(pixels),
        'image_size': (w, h),
        'flow_back_dist': float(flow_back_dist * w / size[0])
    }
