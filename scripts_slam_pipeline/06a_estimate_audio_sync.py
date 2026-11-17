"""
Estimate per-video clock offsets from sounds shared by all cameras,
such as the grippers banged together at the start and end of each recording.

GoPro timecode alone can be off by many frames
(rounded to whole frames, failed QR clock sync, clock drift).
Timecode is only used to find which videos were recorded together.
Within each recording, the relative start time of every pair of videos
from different cameras is measured by cross-correlating their audio.

Acceptance follows KIWI (arXiv 2609.22809, Sec. 4.1) and needs no tuned thresholds:
1. A pair is accepted only if each half of the overlap, correlated on its own,
   reproduces the full-overlap estimate within one video frame.
   This rejects matches supported by a single ambiguous sound or by drift.
2. Per-video start times are solved by least squares over accepted pairs,
   anchored at the reference camera's timecode. With 3+ cameras
   (e.g. two grippers and a head camera) the pair measurements must
   close around every cycle within one video frame.

Output: <session>/audio_sync_offsets.json, consumed by
06b_generate_dataset_plan_synced.py
corrected start time = timecode start time + offset_sec

python scripts_slam_pipeline/06a_estimate_audio_sync.py -i data_workspace/cup_in_the_wild/20240105_zhenjia_packard_2nd_conference_room
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import json
import collections
import multiprocessing
import concurrent.futures
import numpy as np
import av
from tqdm import tqdm
from umi.common.exiftool_util import ExifTool
from umi.common.timecode_util import mp4_get_start_datetime
from umi.common.audio_sync_util import (
    read_audio_mono,
    cross_correlation_lag
)


# %%
def get_video_meta(mp4_path):
    with av.open(mp4_path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        n_frames = stream.frames
    start_timestamp = mp4_get_start_datetime(mp4_path).timestamp()
    return {
        'fps': fps,
        'start_timestamp': start_timestamp,
        'end_timestamp': start_timestamp + n_frames / fps
    }

def estimate_start_diff(ref_mp4_path, query_mp4_path):
    """
    Start time of query video minus start time of ref video (sec),
    from the full overlap and from each half of it.
    """
    ref_audio, sample_rate, ref_audio_offset = read_audio_mono(ref_mp4_path)
    query_audio, query_sample_rate, query_audio_offset = read_audio_mono(query_mp4_path)
    assert sample_rate == query_sample_rate

    def to_start_diff(lag):
        # a sound at ref video time t appears at query video time t + lag / sample_rate
        return ref_audio_offset - query_audio_offset - lag / sample_rate

    lag, standard_score = cross_correlation_lag(ref_audio, query_audio)

    # ref samples that overlap query at this lag
    overlap_start = max(0, -lag)
    overlap_end = min(len(ref_audio), len(query_audio) - lag)
    overlap_mid = (overlap_start + overlap_end) // 2
    half_start_diffs = list()
    half_standard_scores = list()
    for start, end in [(overlap_start, overlap_mid), (overlap_mid, overlap_end)]:
        half_lag, half_score = cross_correlation_lag(ref_audio[start:end], query_audio)
        half_start_diffs.append(to_start_diff(half_lag - start))
        half_standard_scores.append(half_score)

    return {
        'start_diff_sec': to_start_diff(lag),
        'standard_score': standard_score,
        'half_start_diffs_sec': half_start_diffs,
        'half_standard_scores': half_standard_scores
    }

def get_recording_groups(video_meta):
    """
    Group videos whose timecode intervals overlap.
    """
    names = sorted(video_meta.keys(), key=lambda x: video_meta[x]['start_timestamp'])
    groups = list()
    group_end = -np.inf
    for name in names:
        if video_meta[name]['start_timestamp'] >= group_end:
            groups.append(list())
        groups[-1].append(name)
        group_end = max(group_end, video_meta[name]['end_timestamp'])
    return groups

def solve_start_timestamps(anchor, anchor_start_timestamp, edges):
    """
    Start times from pairwise start differences, for videos connected to anchor.
    edges: list of (ref_name, query_name, query_start - ref_start)
    Returns:
        start_timestamps: name -> least squares start timestamp
        closure_errors: (len(edges),) loop closure error of each edge,
            measured difference minus the difference along a
            spanning tree from anchor. Zero for tree edges.
    """
    neighbors = collections.defaultdict(list)
    for ref_name, query_name, start_diff in edges:
        neighbors[ref_name].append((query_name, start_diff))
        neighbors[query_name].append((ref_name, -start_diff))

    # breadth first spanning tree
    tree_starts = {anchor: 0.0}
    queue = collections.deque([anchor])
    while len(queue) > 0:
        name = queue.popleft()
        for other, start_diff in neighbors[name]:
            if other not in tree_starts:
                tree_starts[other] = tree_starts[name] + start_diff
                queue.append(other)
    edges = [e for e in edges if e[0] in tree_starts]
    closure_errors = np.array([start_diff - (tree_starts[query_name] - tree_starts[ref_name])
        for ref_name, query_name, start_diff in edges])

    # least squares relative to anchor, rows: one per edge, plus anchor at 0
    names = sorted(tree_starts.keys())
    idx = {x: i for i, x in enumerate(names)}
    A = np.zeros((len(edges) + 1, len(names)))
    b = np.zeros(len(edges) + 1)
    for i, (ref_name, query_name, start_diff) in enumerate(edges):
        A[i, idx[query_name]] = 1
        A[i, idx[ref_name]] = -1
        b[i] = start_diff
    A[-1, idx[anchor]] = 1
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    start_timestamps = {name: anchor_start_timestamp + float(x[idx[name]]) for name in names}
    return start_timestamps, closure_errors


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Session directory')
@click.option('-o', '--output', default=None, help='Default <input>/audio_sync_offsets.json')
@click.option('-rs', '--reference_serial', default=None, help='Reference camera serial, default first sorted serial')
@click.option('-n', '--num_workers', type=int, default=None)
def main(input, output, reference_serial, num_workers):
    input_path = pathlib.Path(os.path.expanduser(input)).absolute()
    demos_dir = input_path.joinpath('demos')
    if output is None:
        output = input_path.joinpath('audio_sync_offsets.json')
    output = pathlib.Path(os.path.expanduser(output)).absolute()
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    video_dirs = sorted([x.parent for x in demos_dir.glob('demo_*/raw_video.mp4')])
    print(f'Found {len(video_dirs)} demo videos')
    if len(video_dirs) == 0:
        exit(1)
    mp4_paths = {x.name: str(x.joinpath('raw_video.mp4')) for x in video_dirs}

    serials = dict()
    video_meta = dict()
    with ExifTool() as et:
        for name, mp4_path in mp4_paths.items():
            meta = list(et.get_metadata(mp4_path))[0]
            serials[name] = meta['QuickTime:CameraSerialNumber']
            video_meta[name] = get_video_meta(mp4_path)

    if reference_serial is None:
        reference_serial = sorted(set(serials.values()))[0]
    print(f"Reference camera: {reference_serial}")

    # pairs of overlapping videos from different cameras
    groups = get_recording_groups(video_meta)
    group_anchors = list()
    pairs = list()
    for group in groups:
        # anchor on the reference camera, else the first sorted serial
        group_serials = sorted(set(serials[x] for x in group))
        anchor_serial = reference_serial if reference_serial in group_serials else group_serials[0]
        anchor = min([x for x in group if serials[x] == anchor_serial],
            key=lambda x: video_meta[x]['start_timestamp'])
        group_anchors.append(anchor)
        for i, ref_name in enumerate(group):
            for query_name in group[i+1:]:
                if serials[ref_name] == serials[query_name]:
                    continue
                overlap = min(video_meta[ref_name]['end_timestamp'], video_meta[query_name]['end_timestamp']) \
                    - max(video_meta[ref_name]['start_timestamp'], video_meta[query_name]['start_timestamp'])
                if overlap > 0:
                    pairs.append((ref_name, query_name))

    pair_results = dict()
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(estimate_start_diff,
            mp4_paths[ref_name], mp4_paths[query_name]): (ref_name, query_name)
            for ref_name, query_name in pairs}
        for future in tqdm(concurrent.futures.as_completed(futures),
                total=len(futures), desc='Correlating audio'):
            pair_results[futures[future]] = future.result()

    # accept a pair if both halves of the overlap agree with the full estimate
    for (ref_name, query_name), result in pair_results.items():
        frame_period = 1 / video_meta[query_name]['fps']
        half_errors = np.abs(np.array(result['half_start_diffs_sec']) - result['start_diff_sec'])
        result['is_consistent'] = bool(np.all(half_errors <= frame_period))

    video_results = dict()
    for group, anchor in zip(groups, group_anchors):
        for name in group:
            video_results[name] = {
                'camera_serial': serials[name],
                'status': 'failed',
                'offset_sec': None
            }
        if len(set(serials[x] for x in group)) < 2:
            for name in group:
                video_results[name]['status'] = 'single_camera'
                video_results[name]['offset_sec'] = 0.0
            continue

        edges = [(ref_name, query_name, result['start_diff_sec'])
            for (ref_name, query_name), result in pair_results.items()
            if result['is_consistent'] and (ref_name in group)]
        start_timestamps, closure_errors = solve_start_timestamps(
            anchor, video_meta[anchor]['start_timestamp'], edges)

        # pairwise measurements must close around cycles within one frame
        frame_period = 1 / video_meta[anchor]['fps']
        if len(closure_errors) > 0 and np.max(np.abs(closure_errors)) > frame_period:
            for name in start_timestamps:
                video_results[name]['status'] = 'inconsistent'
            continue
        for name, start_timestamp in start_timestamps.items():
            video_results[name]['offset_sec'] = start_timestamp - video_meta[name]['start_timestamp']
            video_results[name]['status'] = 'reference' if name == anchor else 'ok'

    # report
    for (ref_name, query_name), result in sorted(pair_results.items()):
        timecode_diff = video_meta[query_name]['start_timestamp'] - video_meta[ref_name]['start_timestamp']
        print(f"{ref_name} -> {query_name}: audio-timecode "
            f"{(result['start_diff_sec'] - timecode_diff) * 1000:.1f}ms, "
            f"halves {', '.join(f'{(x - timecode_diff) * 1000:.1f}ms' for x in result['half_start_diffs_sec'])}, "
            f"score {result['standard_score']:.1f}, "
            f"{'consistent' if result['is_consistent'] else 'REJECTED'}")
    for name, result in sorted(video_results.items()):
        if result['offset_sec'] is None:
            print(f"{name}: {result['status']}")
        else:
            offset = result['offset_sec']
            print(f"{name}: {result['status']} offset {offset * 1000:.1f}ms "
                f"({offset * video_meta[name]['fps']:.2f} frames)")

    n_unsynced = sum(r['offset_sec'] is None for r in video_results.values())
    if n_unsynced > 0:
        print(f"Warning: {n_unsynced} videos could not be synced, timecode will be used as-is.")

    json.dump({
        'reference_serial': reference_serial,
        'videos': video_results,
        'pairs': [{'ref_video': ref_name, 'query_video': query_name, **result}
            for (ref_name, query_name), result in sorted(pair_results.items())]
    }, output.open('w'), indent=2)
    print(f"Saved to {output}")

# %%
if __name__ == "__main__":
    main()
