from typing import Union
from fractions import Fraction
import datetime
import av


def timecode_to_seconds(
        timecode: str, frame_rate: Union[int, float, Fraction]
        ) -> Union[float, Fraction]:
    """
    Convert non-skip frame timecode into seconds since midnight
    """
    # calculate whole frame rate
    # 29.97 -> 30, 59.94 -> 60
    int_frame_rate = round(frame_rate)

    # parse timecode string
    h, m, s, f = [int(x) for x in timecode.split(':')]

    # calculate frames assuming whole frame rate (i.e. non-drop frame)
    frames = (3600 * h + 60 * m + s) * int_frame_rate + f

    # convert to seconds
    seconds = frames / frame_rate
    return seconds


def stream_get_start_datetime(stream: av.stream.Stream) -> datetime.datetime:
    """
    Combines creation time and timecode to get high-precision
    time for the first frame of a video.
    """
    # read metadata
    frame_rate = stream.average_rate
    tc = stream.metadata['timecode']
    creation_time = stream.metadata['creation_time']
    
    # get time within the day
    seconds_since_midnight = float(timecode_to_seconds(timecode=tc, frame_rate=frame_rate))
    delta = datetime.timedelta(seconds=seconds_since_midnight)
    
    # get dates
    create_datetime = datetime.datetime.strptime(creation_time, r"%Y-%m-%dT%H:%M:%S.%fZ")
    create_datetime = create_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
    start_datetime = create_datetime + delta
    return start_datetime


def mp4_get_start_datetime(mp4_path: str) -> datetime.datetime:
    with av.open(mp4_path) as container:
        stream = container.streams.video[0]
        return stream_get_start_datetime(stream=stream)


# UTC offsets are multiples of 15 min; the timecode and creation_time of a
# GoPro video agree to about a second
UTC_OFFSET_STEP_SEC = 900
MAX_CREATION_TIMECODE_DIFF_SEC = 300
DAY_SEC = 86400


def start_utc_from_metadata(creation_time: str, timecode: str,
        frame_rate: Union[int, float, Fraction]) -> datetime.datetime:
    """
    UTC time of the first frame (timezone-aware), independent of the time
    zone of the machine processing the video.
    GoPro: creation_time is UTC with whole seconds, the timecode is local
    time (non-drop, frames counted at the whole rate) with frame precision.
    The camera's UTC offset is the difference of the two, rounded to 15 min.
    """
    creation = datetime.datetime.strptime(creation_time, r"%Y-%m-%dT%H:%M:%S.%fZ").replace(
        tzinfo=datetime.timezone.utc)
    local_sec = float(timecode_to_seconds(timecode=timecode, frame_rate=frame_rate))
    creation_sec = creation.hour * 3600 + creation.minute * 60 + creation.second + creation.microsecond * 1e-6
    # local - UTC within (-12 h, 12 h]
    diff = (local_sec - creation_sec + DAY_SEC / 2) % DAY_SEC - DAY_SEC / 2
    utc_offset = round(diff / UTC_OFFSET_STEP_SEC) * UTC_OFFSET_STEP_SEC
    residual = diff - utc_offset
    assert abs(residual) <= MAX_CREATION_TIMECODE_DIFF_SEC, \
        f"timecode {timecode} and creation_time {creation_time} differ by {residual:.0f} s beyond a UTC offset"
    # first frame = creation_time corrected by the sub-offset residual (frame precision)
    return creation + datetime.timedelta(seconds=residual)


def mp4_get_start_utc(mp4_path: str) -> datetime.datetime:
    """
    UTC time of the first frame of a GoPro video, see start_utc_from_metadata.
    """
    with av.open(mp4_path) as container:
        stream = container.streams.video[0]
        return start_utc_from_metadata(stream.metadata['creation_time'], stream.metadata['timecode'],
            stream.average_rate)
