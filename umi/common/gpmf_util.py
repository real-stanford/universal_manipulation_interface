"""
GoPro GPMF metadata from the MP4 'GoPro MET' track, read with PyAV (no
external tools). Format: https://github.com/gopro/gpmf-parser (README):
KLV entries of an 8 byte header (FourCC, type char, structure size uint8,
repeat uint16 big-endian) and size * repeat bytes of data padded to 4 bytes;
type 0 is a nested container (DEVC device, STRM stream).

Timing per payload (one MP4 sample of the metadata track, ~1 s):
- MP4: pts / duration of the metadata track sample (the MP4 timeline,
  shared with the video and audio tracks);
- STMP: microsecond timestamp of the stream's first sample in the payload
  (the camera's sensor clock);
- TSMP: total samples of the stream delivered since the start of the recording;
- TIMO: optional delay of the stream data (s).
"""
from typing import Dict, Iterator, List, Optional
import struct
import av
import numpy as np

TYPE_FORMATS = {
    'b': 'b', 'B': 'B', 'c': 'c', 'd': 'd', 'f': 'f', 'j': 'q', 'J': 'Q',
    'l': 'i', 'L': 'I', 's': 'h', 'S': 'H',
}


def parse_klv(data: bytes) -> List[dict]:
    """
    Entries of a GPMF buffer: {'key', 'type', 'size', 'repeat', 'value' or 'children'}.
    Numeric values as numpy arrays (repeat, size / item size); strings as str.
    """
    out = list()
    i = 0
    while i + 8 <= len(data):
        key = data[i:i + 4].decode('latin-1')
        type_char = chr(data[i + 4])
        size = data[i + 5]
        repeat = struct.unpack('>H', data[i + 6:i + 8])[0]
        length = size * repeat
        body = data[i + 8:i + 8 + length]
        item = {'key': key, 'type': type_char, 'size': size, 'repeat': repeat}
        if data[i + 4] == 0:
            item['children'] = parse_klv(body)
        elif type_char == 'c':
            item['value'] = body.decode('latin-1').rstrip('\x00')
        elif type_char in TYPE_FORMATS:
            fmt = TYPE_FORMATS[type_char]
            n = length // struct.calcsize(fmt)
            values = np.array(struct.unpack('>' + fmt * n, body[:n * struct.calcsize(fmt)]))
            per_item = size // struct.calcsize(fmt)
            item['value'] = values.reshape(repeat, per_item) if per_item > 1 else values
        else:
            item['raw'] = body
        out.append(item)
        i += 8 + (length + 3) // 4 * 4
    return out


def mp4_gpmf_payloads(mp4_path: str) -> Iterator[dict]:
    """
    Per metadata track sample: {'time' (s, MP4 timeline), 'duration' (s), 'klv'}.
    """
    with av.open(mp4_path) as container:
        streams = [s for s in container.streams if s.type == 'data'
            and s.metadata.get('handler_name', '').strip().startswith('GoPro MET')]
        assert len(streams) == 1, f"{mp4_path}: no GoPro MET track"
        stream = streams[0]
        tb = float(stream.time_base)
        for packet in container.demux(stream):
            if packet.size == 0 or packet.pts is None:
                continue
            yield {'time': packet.pts * tb, 'duration': (packet.duration or 0) * tb, 'klv': parse_klv(bytes(packet))}


def stream_timing(mp4_path: str, keys=('ACCL', 'GYRO', 'CORI')) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Per sensor stream (FourCC in keys), per payload: MP4 time and duration,
    STMP (us, None if absent), TSMP, number of samples, TIMO (s, 0 if absent).
    """
    out = {k: {'mp4_time': [], 'mp4_duration': [], 'stmp_us': [], 'tsmp': [], 'n': [], 'timo': []} for k in keys}
    for payload in mp4_gpmf_payloads(mp4_path):
        for devc in payload['klv']:
            for strm in devc.get('children', []):
                if strm['key'] != 'STRM':
                    continue
                entries = {c['key']: c for c in strm['children']}
                for k in keys:
                    if k not in entries:
                        continue
                    d = out[k]
                    d['mp4_time'].append(payload['time'])
                    d['mp4_duration'].append(payload['duration'])
                    d['stmp_us'].append(int(entries['STMP']['value'][0]) if 'STMP' in entries else -1)
                    d['tsmp'].append(int(entries['TSMP']['value'][0]) if 'TSMP' in entries else -1)
                    d['n'].append(entries[k]['repeat'])
                    d['timo'].append(float(entries['TIMO']['value'][0]) if 'TIMO' in entries else 0.0)
    return {k: {a: np.array(b) for a, b in v.items()} for k, v in out.items() if len(v['n'])}


def stream_keys(mp4_path: str, n_payloads: int = 1) -> List[List[str]]:
    """
    FourCC keys of each stream in the first payloads, for inspection.
    """
    out = list()
    for i, payload in enumerate(mp4_gpmf_payloads(mp4_path)):
        if i >= n_payloads:
            break
        for devc in payload['klv']:
            for strm in devc.get('children', []):
                if strm['key'] == 'STRM':
                    out.append([c['key'] for c in strm['children']])
    return out


def video_frame0_time(mp4_path: str, frame_duration: float) -> float:
    """
    Camera clock time (s, the STMP clock of the sensor streams) of the video's
    first frame, from a per-frame stream (CORI, else MSKP): its first payload's
    STMP minus the frames before it. imu_data.json times are this clock (STMP),
    so an IMU sample at time c is at video time c - video_frame0_time.
    """
    timing = stream_timing(mp4_path, ('CORI', 'MSKP'))
    d = timing.get('CORI', timing.get('MSKP'))
    assert d is not None and d['stmp_us'][0] >= 0, f"{mp4_path}: no per-frame GPMF stream with STMP"
    return d['stmp_us'][0] * 1e-6 - (d['tsmp'][0] - d['n'][0]) * frame_duration


def fit_stream_timing(d: Dict[str, np.ndarray]) -> dict:
    """
    Line fit STMP = start + period * (samples before the payload) over the
    payloads of one stream (stream_timing): sample period and start on the
    camera clock (us), residual RMS / max (us).
    """
    before = d['tsmp'] - d['n']
    ok = d['stmp_us'] >= 0
    period, start = np.polyfit(before[ok], d['stmp_us'][ok], 1)
    residual = d['stmp_us'][ok] - (start + period * before[ok])
    return {'period_us': float(period), 'start_us': float(start), 'rms_us': float(np.sqrt(np.mean(residual ** 2))),
        'max_us': float(np.max(np.abs(residual))), 'n_payloads': int(ok.sum()), 'n_samples': int(d['tsmp'][-1])}


def frame_timing(mp4_path: str) -> Optional[dict]:
    """
    Video frames on the camera clock, from the per-frame GPMF stream (CORI,
    else MSKP): measured frame rate (Hz), how uniform the frame times are
    (payload residual RMS / max, us), number of frame samples, and the frames
    the camera reports skipped (nonzero MSKP / LSKP values). None without a
    per-frame stream.
    """
    timing = stream_timing(mp4_path, ('CORI', 'MSKP'))
    d = timing.get('CORI', timing.get('MSKP'))
    if d is None or not np.any(d['stmp_us'] >= 0):
        return None
    fit = fit_stream_timing(d)
    skipped = {'MSKP': 0, 'LSKP': 0}
    for payload in mp4_gpmf_payloads(mp4_path):
        for devc in payload['klv']:
            for strm in devc.get('children', []):
                for c in strm.get('children', []):
                    if c['key'] in skipped and 'value' in c:
                        skipped[c['key']] += int(np.count_nonzero(c['value']))
    return {'stream': 'CORI' if 'CORI' in timing else 'MSKP', 'rate_hz': 1e6 / fit['period_us'],
        'rms_us': fit['rms_us'], 'max_us': fit['max_us'], 'n_frames': fit['n_samples'],
        'skipped_mskp': skipped['MSKP'], 'skipped_lskp': skipped['LSKP'], 'has_skip_counters': 'MSKP' in timing}


def gyro_rates(mp4_path: str) -> np.ndarray:
    """
    Angular rate magnitude (rad/s) of every GYRO sample of the video (GPMF, SCAL applied).
    """
    rates = list()
    for payload in mp4_gpmf_payloads(mp4_path):
        for devc in payload['klv']:
            for strm in devc.get('children', []):
                entries = {c['key']: c for c in strm.get('children', [])}
                if 'GYRO' in entries and 'value' in entries['GYRO']:
                    scale = float(np.ravel(entries['SCAL']['value'])[0]) if 'SCAL' in entries else 1.0
                    gyro = np.asarray(entries['GYRO']['value'], dtype=float).reshape(-1, 3) / scale
                    rates.append(np.linalg.norm(gyro, axis=1))
    return np.concatenate(rates) if rates else np.zeros(0)
