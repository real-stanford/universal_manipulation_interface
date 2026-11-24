"""
H.265 Annex B access units (one per frame, as in the MCAP video messages): read from a GoPro MP4
(video_messages, video_messages_exact), keyframe test, and the cut of a stream at any frame with
only the first GOP re-encoded (07_export_mcap.py --exact_start, 14_export_rhoda.py,
15_export_episode_videos.py, exact_start_mcaps.py).
"""
from fractions import Fraction
import av

START_CODE = b'\x00\x00\x00\x01'
HEVC_AUD = 35

IRAP_AND_VPS = {16, 17, 18, 19, 20, 21, 32}   # IRAP pictures, VPS


def nal_types(data: bytes) -> set:
    """
    H.265 NAL unit types in an Annex B access unit.
    """
    out, i = set(), data.find(b'\x00\x00\x01')
    while i >= 0 and i + 3 < len(data):
        out.add((data[i + 3] >> 1) & 0x3f)
        i = data.find(b'\x00\x00\x01', i + 3)
    return out


def is_keyframe(data: bytes) -> bool:
    """
    An access unit a decoder can start at: an IRAP picture (with its parameter sets).
    """
    return bool(nal_types(data) & IRAP_AND_VPS)


def cut_from(frames, start_frame: int, rate, crf: int = 16, threads: int = 0):
    """
    (frame index, Annex B data) of the frames from start_frame on, exactly. frames: (frame index,
    Annex B data) in order, starting at a keyframe at or before start_frame, no B-frames. The frames
    from start_frame up to the next keyframe are decoded (from the first keyframe) and re-encoded as
    H.265 (libx265, first frame a keyframe, no B-frames, parameter sets in the stream, full range
    like the GoPro stream); from the next keyframe on the original data is passed through. If
    start_frame is a keyframe nothing is re-encoded. At GoPro's one keyframe per second at most one
    second is re-encoded. threads: CPU threads for decoder and encoder (0: all cores).
    """
    decoder = av.CodecContext.create('hevc', 'r')
    if threads:
        decoder.thread_count = threads
    state = {'encoder': None, 'first': None, 'sent': 0, 'decoded': 0, 'encoded': 0}
    pending = list()

    def encode(frame):
        if state['encoder'] is None:
            encoder = av.CodecContext.create('libx265', 'w')
            encoder.width, encoder.height = frame.width, frame.height
            encoder.pix_fmt = frame.format.name
            encoder.time_base = 1 / Fraction(rate)
            encoder.options = {'preset': 'fast', 'crf': str(crf), 'x265-params':
                'bframes=0:repeat-headers=1:aud=1:open-gop=0:keyint=600:range=full:log-level=error'
                + (f':pools={threads}:frame-threads={min(threads, 4)}' if threads else '')}
            state['encoder'] = encoder
        frame.pts = state['encoded']
        state['encoded'] += 1
        pending.extend(bytes(p) for p in state['encoder'].encode(frame))

    def decoded(frames_out):
        for frame in frames_out:
            index = state['first'] + state['decoded']   # no B-frames: decoded in input order
            state['decoded'] += 1
            if index >= start_frame:
                encode(frame)

    def flush():
        if state['sent']:
            decoded(decoder.decode(None))
        if state['encoder'] is not None:
            pending.extend(bytes(p) for p in state['encoder'].encode(None))
        out = [(start_frame + i, data) for i, data in enumerate(pending)]
        pending.clear()
        return out

    copying = False
    for index, data in frames:
        if state['first'] is None:
            assert is_keyframe(data), f"the frames start at frame {index}, not a keyframe"
            assert index <= start_frame, f"first keyframe {index} after the cut frame {start_frame}"
            state['first'] = index
        if not copying and index >= start_frame and is_keyframe(data):
            copying = True                  # from here the original stream
            yield from flush()
        if copying:
            yield index, data
            continue
        state['sent'] += 1
        decoded(decoder.decode(av.Packet(data)))
    if not copying:                         # the frames ended before the next keyframe
        yield from flush()


def cut_to_steps(times, frames: list, steps, rate, crf: int = 16, threads: int = 0):
    """
    One camera's episode video from an MCAP: times (ns) and Annex B data of its video messages (in
    order, from a keyframe), steps (ns) the episode's step times. Returns (start index in the
    messages, start re-encoded, (message index, data) of exactly one frame per step from the frame
    nearest the first step). The frames must be evenly spaced and at the steps.
    """
    import numpy as np
    times, steps = np.asarray(times, dtype=np.int64), np.asarray(steps, dtype=np.int64)
    n, dt_ns = len(steps), 1e9 / float(rate)
    assert len(times) < 2 or np.all(np.abs(np.diff(times) - dt_ns) < 0.25 * dt_ns), "frames not evenly spaced (dropped frames?)"
    start = int(np.argmin(np.abs(times - steps[0])))
    assert len(times) >= start + n and np.all(np.abs(times[start:start + n] - steps) < 0.5 * dt_ns), \
        f"{len(times) - start} frames from the episode start, not one per step for {n} steps"
    out = list(cut_from(((i, frames[i]) for i in range(start + n)), start, rate, crf, threads))
    assert len(out) == n, f"{len(out)} frames after the cut, {n} steps"
    return start, not is_keyframe(frames[start]), out


def hevc_parameter_sets(extradata: bytes) -> list:
    """
    VPS, SPS, PPS NAL units from an MP4 hvcC record.
    """
    assert extradata[0] == 1, "expected an hvcC record (MP4 HEVC)"
    out, i = list(), 23
    for _ in range(extradata[22]):
        count = int.from_bytes(extradata[i + 1:i + 3], 'big')
        i += 3
        for _ in range(count):
            length = int.from_bytes(extradata[i:i + 2], 'big')
            out.append(extradata[i + 2:i + 2 + length])
            i += 2 + length
    return out


def to_annexb(packet: bytes, length_size: int, parameter_sets: list, keyframe: bool) -> bytes:
    """
    MP4 length-prefixed NAL units -> Annex B; on keyframes the parameter sets
    follow the access unit delimiter (or lead if there is none).
    """
    nals, i = list(), 0
    while i + length_size <= len(packet):
        length = int.from_bytes(packet[i:i + length_size], 'big')
        nals.append(packet[i + length_size:i + length_size + length])
        i += length_size + length
    if keyframe:
        at = 1 if nals and ((nals[0][0] >> 1) & 0x3f) == HEVC_AUD else 0
        nals = nals[:at] + parameter_sets + nals[at:]
    return b''.join(START_CODE + n for n in nals)


def video_messages(video_path: str, start_frame: int, stop_frame: int):
    """
    (frame index, Annex B data) of frames [keyframe before start_frame, stop_frame).
    """
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == 'hevc', f"{video_path}: {stream.codec_context.name}, expected hevc"
        extradata = bytes(stream.codec_context.extradata)
        length_size = (extradata[21] & 3) + 1
        parameter_sets = hevc_parameter_sets(extradata)
        fps = float(stream.average_rate)
        frame_time = lambda pts: float(pts * stream.time_base) - float((stream.start_time or 0) * stream.time_base)
        container.seek(int(max(0.0, start_frame / fps - 0.001) / stream.time_base), stream=stream,
            backward=True, any_frame=False)
        started = False
        for packet in container.demux(stream):
            if packet.size == 0 or packet.pts is None:
                continue
            index = int(round(frame_time(packet.pts) * fps))
            if index >= stop_frame:
                break
            if not started:
                if not packet.is_keyframe:
                    continue
                assert index <= start_frame, f"{video_path}: first keyframe {index} after episode start {start_frame}"
                started = True
            yield index, to_annexb(bytes(packet), length_size, parameter_sets, packet.is_keyframe)


def video_messages_exact(video_path: str, start_frame: int, stop_frame: int, crf: int = 16):
    """
    (frame index, Annex B data) of frames [start_frame, stop_frame) exactly, no frames before
    start_frame: the frames from start_frame up to the next keyframe re-encoded (libx265, first frame
    a keyframe, no B-frames, full range like the GoPro stream), the rest the original stream (from
    its next keyframe, which carries the original VPS / SPS / PPS); umi.common.hevc_util.cut_from.
    Only the first GOP is re-encoded (at most one second at GoPro's one keyframe per second).
    """
    with av.open(video_path) as container:
        rate = container.streams.video[0].average_rate
    yield from cut_from(video_messages(video_path, start_frame, stop_frame), start_frame, rate, crf)
