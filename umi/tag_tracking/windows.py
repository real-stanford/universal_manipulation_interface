"""
Processing only the episodes of a continuous take: frame windows around the
episodes of episodes.json (00b_segment_episodes.py), frames of the windows as
separate sequences (rolling shutter motion and smoothing never span a gap),
and results scattered back into arrays over all frames of the take.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np


def episode_windows(times: np.ndarray, episodes: Optional[List[dict]], margin: float) -> List[Tuple[int, int]]:
    """
    (start, stop) frame index ranges covering each episode +- margin seconds,
    overlapping ranges merged. episodes None: the whole take.
    """
    if episodes is None:
        return [(0, len(times))]
    ranges = sorted((int(np.searchsorted(times, e['start'] - margin)),
        int(np.searchsorted(times, e['end'] + margin, side='right'))) for e in episodes)
    merged = list()
    for a, b in ranges:
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def window_frames(frames: List[dict], windows: List[Tuple[int, int]]) -> Tuple[List[dict], np.ndarray]:
    """
    Frames of all windows, each window its own sequence, and their indices in the take.
    """
    out, idx = list(), list()
    for w, (a, b) in enumerate(windows):
        for i in range(a, b):
            out.append({'time': frames[i]['time'], 'corners': frames[i]['corners'], 'sequence': w})
            idx.append(i)
    return out, np.array(idx, dtype=int)


def scatter(n: int, idx: np.ndarray, values: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Per frame arrays over the selected frames -> arrays over all n frames:
    NaN for floats, 0 for integers, False for booleans outside the selection.
    """
    out = dict()
    for k, v in values.items():
        v = np.asarray(v)
        fill = False if v.dtype == bool else (0 if v.dtype.kind in 'iu' else np.nan)
        full = np.full((n,) + v.shape[1:], fill, dtype=v.dtype)
        full[idx] = v
        out[k] = full
    return out


def split_window(times: np.ndarray, a: int, b: int, chunk_sec: float, context_sec: float) -> List[Tuple[int, int, int, int]]:
    """
    A long window [a, b) as chunks solved separately (in parallel): per chunk
    (lo, hi, keep_lo, keep_hi) frame indices; the kept parts [keep_lo, keep_hi)
    tile [a, b) without overlap, each extended by context_sec on both sides
    (within the window) so every kept frame has motion context around it.
    A window up to chunk_sec long is one chunk.
    """
    duration = times[b - 1] - times[a]
    n = max(1, int(np.ceil(duration / chunk_sec)))
    if n == 1:
        return [(a, b, a, b)]
    cuts = [a] + [int(np.searchsorted(times, times[a] + k * duration / n, side='left')) for k in range(1, n)] + [b]
    out = list()
    for keep_lo, keep_hi in zip(cuts[:-1], cuts[1:]):
        if keep_hi <= keep_lo:
            continue
        lo = max(a, int(np.searchsorted(times, times[keep_lo] - context_sec, side='left')))
        hi = min(b, int(np.searchsorted(times, times[keep_hi - 1] + context_sec, side='right')))
        out.append((lo, hi, keep_lo, keep_hi))
    return out
