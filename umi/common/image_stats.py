"""
Per-frame image statistics and stream facts of a video, for the media quality
checks (Odyssey spec 1.9 / 2.10: frozen, dark, exposure, contrast, duplicate
frames, encoding, keyframe interval). One decode of every frame, scaled down to
a small gray image: mean and standard deviation of the brightness, share of
near-black and saturated pixels, mean absolute difference to the previous
frame (0..255 levels).
"""
from typing import Dict
import pathlib
import json
import av
import numpy as np

STATS_WIDTH = 160
VERSION = 1


def stream_facts(path) -> Dict:
    """
    Codec, geometry, pixel format, color range, bitrate and container rate of the first video stream.
    """
    with av.open(str(path)) as container:
        s = container.streams.video[0]
        cc = s.codec_context
        return {
            'codec': cc.name, 'profile': cc.profile, 'width': cc.width, 'height': cc.height,
            'pix_fmt': cc.pix_fmt, 'color_range': str(getattr(cc, 'color_range', None)),
            'bit_rate': int(s.bit_rate or cc.bit_rate or 0),
            'container_rate': float(s.average_rate), 'n_frames': int(s.frames),
            'time_base': float(s.time_base),
        }


def video_image_stats(path, width: int = STATS_WIDTH) -> Dict[str, np.ndarray]:
    """
    Per decoded frame: 'time' (s, pts), 'keyframe', 'mean', 'std', 'dark'
    (share < 16), 'saturated' (share > 239), 'diff' (mean absolute difference
    to the previous frame, NaN for the first).
    """
    out = {k: list() for k in ('time', 'keyframe', 'mean', 'std', 'dark', 'saturated', 'diff')}
    previous = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        height = int(round(stream.codec_context.height * width / stream.codec_context.width / 2) * 2)
        for frame in container.decode(stream):
            g = frame.reformat(width=width, height=height, format='gray').to_ndarray().astype(np.float32)
            out['time'].append(float(frame.pts * stream.time_base))
            out['keyframe'].append(bool(frame.key_frame))
            out['mean'].append(float(g.mean()))
            out['std'].append(float(g.std()))
            out['dark'].append(float(np.mean(g < 16)))
            out['saturated'].append(float(np.mean(g > 239)))
            out['diff'].append(float(np.mean(np.abs(g - previous))) if previous is not None else np.nan)
            previous = g
    return {k: np.array(v) for k, v in out.items()}


def cached_image_stats(video_path) -> Dict:
    """
    video_image_stats and stream_facts, cached next to the video (image_stats.npz / image_stats.json).
    """
    video_path = pathlib.Path(video_path)
    npz, meta = video_path.with_name('image_stats.npz'), video_path.with_name('image_stats.json')
    if npz.is_file() and meta.is_file():
        facts = json.load(meta.open('r'))
        if facts.get('version') == VERSION and facts.get('video_size') == video_path.stat().st_size:
            return {'facts': facts, 'stats': dict(np.load(npz))}
    stats = video_image_stats(video_path)
    facts = {**stream_facts(video_path), 'version': VERSION, 'video_size': video_path.stat().st_size,
        'stats_width': STATS_WIDTH}
    np.savez_compressed(npz, **stats)
    json.dump(facts, meta.open('w'), indent=2)
    return {'facts': facts, 'stats': stats}
