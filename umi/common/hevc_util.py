"""
H.265 Annex B access units (one per frame, as in the MCAP video messages): keyframe test and the
cut of a stream at any frame with only the first GOP re-encoded (07_export_mcap.py --exact_start,
exact_start_mcaps.py).
"""
from fractions import Fraction
import av

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


def cut_from(frames, start_frame: int, rate, crf: int = 16):
    """
    (frame index, Annex B data) of the frames from start_frame on, exactly. frames: (frame index,
    Annex B data) in order, starting at a keyframe at or before start_frame, no B-frames. The frames
    from start_frame up to the next keyframe are decoded (from the first keyframe) and re-encoded as
    H.265 (libx265, first frame a keyframe, no B-frames, parameter sets in the stream, full range
    like the GoPro stream); from the next keyframe on the original data is passed through. If
    start_frame is a keyframe nothing is re-encoded. At GoPro's one keyframe per second at most one
    second is re-encoded.
    """
    decoder = av.CodecContext.create('hevc', 'r')
    state = {'encoder': None, 'first': None, 'sent': 0, 'decoded': 0, 'encoded': 0}
    pending = list()

    def encode(frame):
        if state['encoder'] is None:
            encoder = av.CodecContext.create('libx265', 'w')
            encoder.width, encoder.height = frame.width, frame.height
            encoder.pix_fmt = frame.format.name
            encoder.time_base = 1 / Fraction(rate)
            encoder.options = {'preset': 'fast', 'crf': str(crf), 'x265-params':
                'bframes=0:repeat-headers=1:aud=1:open-gop=0:keyint=600:range=full:log-level=error'}
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


def cut_to_steps(times, frames: list, steps, rate, crf: int = 16):
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
    out = list(cut_from(((i, frames[i]) for i in range(start + n)), start, rate, crf))
    assert len(out) == n, f"{len(out)} frames after the cut, {n} steps"
    return start, not is_keyframe(frames[start]), out
