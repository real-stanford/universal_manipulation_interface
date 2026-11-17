"""
Videos of a session with synchronized start times: GoPro timecode
corrected by the audio sync offsets of
scripts_slam_pipeline/06a_estimate_audio_sync.py (audio_sync_offsets.json).
"""
from typing import List
import json
import pathlib
import av
from umi.common.exiftool_util import ExifTool
from umi.common.timecode_util import mp4_get_start_datetime


def get_session_videos(session: pathlib.Path) -> List[dict]:
    """
    Per demo video dir: name, path, camera serial, fps, number of frames
    and synchronized start timestamp (sec).
    """
    offsets_path = session.joinpath('audio_sync_offsets.json')
    assert offsets_path.is_file(), "run 06a_estimate_audio_sync.py first"
    offsets = json.load(offsets_path.open('r'))['videos']

    videos = list()
    with ExifTool() as et:
        for video_path in sorted(session.glob('demos/demo_*/raw_video.mp4')):
            name = video_path.parent.name
            assert offsets.get(name, {}).get('offset_sec') is not None, \
                f"{name} has no audio sync offset ({offsets.get(name, {}).get('status')})"
            meta = list(et.get_metadata(str(video_path)))[0]
            with av.open(str(video_path)) as container:
                stream = container.streams.video[0]
                fps = stream.average_rate
                n_frames = stream.frames
            start = mp4_get_start_datetime(str(video_path)).timestamp() + offsets[name]['offset_sec']
            videos.append({
                'name': name,
                'dir': video_path.parent,
                'camera_serial': meta['QuickTime:CameraSerialNumber'],
                'fps': fps,
                'n_frames': n_frames,
                'start_timestamp': start,
                'end_timestamp': start + float(n_frames / fps)
            })
    return videos


def overlapping(videos: List[dict], reference: dict, camera_serial: str) -> List[dict]:
    """
    Videos of a camera overlapping the reference video in time.
    """
    return [x for x in videos if x['camera_serial'] == camera_serial
        and x['start_timestamp'] < reference['end_timestamp']
        and x['end_timestamp'] > reference['start_timestamp']]
