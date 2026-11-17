"""
Per-frame processing of a long video on all CPU cores.

The video is split into time chunks, one worker process per chunk. Each worker
seeks to the keyframe before its chunk, decodes, skips frames before the chunk
and stops at the chunk end, so every frame is processed exactly once, by
time (frame.pts * time_base, as in sequential decoding). Results are returned
in frame order; the frame count is checked against the stream's frame count.

Workers are started with 'spawn' (fresh interpreters), not 'fork': a forked
worker can inherit a lock held by another thread of the parent (e.g. from an
earlier pool or progress bar) and wait on it forever. With spawn, the factory
must be importable (module level, also in a script run as __main__).

FFmpeg warnings are silenced in the workers (e.g. 'deprecated pixel format
used' for every frame of full-range GoPro video, harmless for detection).

The per-frame function is created inside each worker by a picklable factory,
e.g. to build OpenCV objects that cannot be sent between processes:

    def make_processor(arg1, arg2):
        ...setup...
        def process(frame: av.VideoFrame, time: float) -> dict: ...
        return process

    results = process_video_chunked(path, make_processor, (arg1, arg2), num_workers=32)
"""
from typing import Callable, List, Optional, Tuple
import concurrent.futures
import math
import multiprocessing
import av
import av.logging
from tqdm import tqdm

# seek this much before a chunk start, the keyframe before it is decoded from there
SEEK_MARGIN_SEC = 1.0


def get_chunk_bounds(video_path: str, n_chunks: int) -> Tuple[List[Tuple[float, float]], int]:
    """
    n_chunks time intervals [start, end) covering all frames, with boundaries
    halfway between frames, and the stream's frame count.
    """
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        n_frames = stream.frames
        fps = float(stream.average_rate)
        first_time = float(next(container.decode(stream)).pts * stream.time_base)
    assert n_frames > 0, f"{video_path}: frame count unknown, use one chunk"
    n_chunks = max(1, min(n_chunks, n_frames))
    starts = [first_time + (k * n_frames // n_chunks - 0.5) / fps for k in range(n_chunks)]
    starts[0] = -math.inf
    ends = starts[1:] + [math.inf]
    return list(zip(starts, ends)), n_frames


def _process_chunk(video_path: str, start: float, end: float,
        make_processor: Callable, processor_args: tuple) -> List[dict]:
    av.logging.set_level(av.logging.ERROR)
    process = make_processor(*processor_args)
    results = list()
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        if start > -math.inf:
            container.seek(max(0, int((start - SEEK_MARGIN_SEC) / stream.time_base)),
                stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            time = float(frame.pts * stream.time_base)
            if time < start:
                continue
            if time >= end:
                break
            results.append(process(frame, time))
    return results


def process_video_chunked(video_path: str, make_processor: Callable, processor_args: tuple = (),
        num_workers: Optional[int] = None, desc: Optional[str] = None) -> List[dict]:
    """
    Results of process(frame, time) for every frame in order, each with
    'frame_idx' and 'time' set. process must return a dict.
    """
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    bounds, n_frames = get_chunk_bounds(video_path, num_workers)
    chunks = [None] * len(bounds)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers,
            mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = {executor.submit(_process_chunk, video_path, start, end, make_processor, processor_args): k
            for k, (start, end) in enumerate(bounds)}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                desc=f"{desc or video_path} ({len(bounds)} chunks)"):
            chunks[futures[future]] = future.result()

    results = [x for chunk in chunks for x in chunk]
    times = [x['time'] for x in results]
    assert len(results) == n_frames, \
        f"{video_path}: {len(results)} frames processed, stream has {n_frames}"
    assert all(a < b for a, b in zip(times[:-1], times[1:])), f"{video_path}: frames out of order"
    for i, (x, time) in enumerate(zip(results, times)):
        x['frame_idx'] = i
        x['time'] = time
    return results
