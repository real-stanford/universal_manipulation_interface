"""
umi.common.timecode_util.start_utc_from_metadata: GoPro creation_time (UTC)
+ local timecode -> UTC of the first frame, for any camera time zone.

python tests/test_timecode_utc.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
from fractions import Fraction
from umi.common.timecode_util import start_utc_from_metadata, timecode_to_seconds

NTSC_60 = Fraction(60000, 1001)
UTC = datetime.timezone.utc


def timecode_for(local: datetime.datetime, frame_rate) -> str:
    """
    Non-drop timecode (frames counted at the whole rate) of a local wall-clock time.
    """
    sec = local.hour * 3600 + local.minute * 60 + local.second + local.microsecond * 1e-6
    frames = int(round(sec * float(frame_rate)))
    n = round(frame_rate)
    s, f = divmod(frames, n)
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}:{f:02d}"


def check(first_frame_utc: datetime.datetime, utc_offset_hours: float, frame_rate=NTSC_60):
    local = first_frame_utc + datetime.timedelta(hours=utc_offset_hours)
    tc = timecode_for(local.replace(tzinfo=None), frame_rate)
    # creation_time: whole seconds, UTC
    creation = first_frame_utc.replace(microsecond=0).strftime(r"%Y-%m-%dT%H:%M:%S.000000Z")
    result = start_utc_from_metadata(creation, tc, frame_rate)
    error = abs((result - first_frame_utc).total_seconds())
    assert result.tzinfo is not None
    assert error < 1.5 / float(frame_rate), f"{first_frame_utc} +{utc_offset_hours}h: got {result}, error {error:.4f}s"


def test_recorded_head_video():
    # 2026-11-16 task1 head camera (IST): creation_time 11:54:06Z, timecode 17:23:03:58
    result = start_utc_from_metadata('2026-11-16T11:54:06.000000Z', '17:23:03:58', NTSC_60)
    expected_local = float(timecode_to_seconds('17:23:03:58', NTSC_60))  # 17:24:06.6 local
    expected = datetime.datetime(2026, 11, 16, tzinfo=UTC) + datetime.timedelta(seconds=expected_local - 5.5 * 3600)
    assert abs((result - expected).total_seconds()) < 1e-6, (result, expected)


def test_time_zones():
    t = datetime.datetime(2026, 11, 16, 11, 54, 6, 600000, tzinfo=UTC)
    for hours in [5.5, 0, -5, 9, 5.75, -3.5, 12, -11]:
        check(t, hours)


def test_day_wrap():
    # local date differs from the UTC date
    check(datetime.datetime(2026, 11, 16, 18, 35, 0, 250000, tzinfo=UTC), 5.5)    # 00:05 next day IST
    check(datetime.datetime(2026, 11, 17, 2, 0, 0, 500000, tzinfo=UTC), -5)       # 21:00 previous day
    check(datetime.datetime(2026, 12, 31, 23, 59, 59, 900000, tzinfo=UTC), 1)     # new year


def test_frame_rates():
    t = datetime.datetime(2026, 11, 16, 11, 54, 6, 300000, tzinfo=UTC)
    for rate in [NTSC_60, Fraction(30000, 1001), 50, 25, 60]:
        check(t, 5.5, rate)


def test_inconsistent_metadata():
    try:
        start_utc_from_metadata('2026-11-16T11:54:06.000000Z', '17:01:00:00', NTSC_60)
    except AssertionError:
        return
    raise AssertionError("timecode 7.5 min off a 15 min offset must be rejected")


if __name__ == '__main__':
    test_recorded_head_video()
    test_time_zones()
    test_day_wrap()
    test_frame_rates()
    test_inconsistent_metadata()
    print("all passed")
