"""
Rhoda delivery from episode MCAP files (07_export_mcap.py, with or without --exact_start), in one
pass: one folder per recording (episode) with the three camera videos, per-frame data and metadata,
in the order of the data PRD's UMI delivery:

  <out>/<rec>/
    <rec>_head.mp4, <rec>_left.mp4, <rec>_right.mp4
        the camera videos of the MCAP (/cameras/camera{j}/video), each cut to exactly the episode:
        frame 0 = the episode's first step, one frame per step, so all three files cover the same span
        with the same frame count and frame i is the same instant in all of them (PRD U2). MCAP videos
        that start at the keyframe before the episode (07 without --exact_start): the frames from the
        episode start to the next keyframe (at most 1 s) are re-encoded in memory (07's --exact_start
        re-encode, libx265, no B-frames), the rest is the GoPro's own stream, stream-copied into the mp4
        (faststart). Tag hev1 when the file has a re-encoded start (two encodings: the parameter sets
        stay in the stream), else hvc1.
    <rec>_meta.json
        cameras: per camera model, lens, field of view, resolution, frame rate, intrinsics (the MCAP's
        camera{j} metadata, 07 with -c); sync: method and the frame index of the sync event in each
        file (0: every file starts at it)
    <rec>_right_gripper.csv, <rec>_left_gripper.csv
        gripper width per frame: frame (index in every file of the recording), width_m (0 = fully closed)
    <rec>_mosaic.mp4 (optional, --mosaic N per task folder or --mosaic_files)
        review video: head on top, right and left wrist below, frame by frame at the cameras' frame rate; H.264 yuv420p,
        any player.

Sync: the sync event is the episode's first step (the first robot0/gripper/state message); in each
camera's MCAP video the frame nearest to it (the MCAP's synchronized message times), within half a
frame, and the following frames at the following steps (checked).

Input: an MCAP file, a folder or s3:// prefix with MCAP files below it, or a paths CSV of
describe_mcaps.py --list_only. S3 files are downloaded one at a time into the output folder and
deleted after. Recording name: task folder + file name (station3/2026-11-21/task01/episode_000012.mcap
-> station3_2026-11-21_task01_episode_000012).

python scripts_tag_tracking/14_export_rhoda.py <mcap / folder / s3:// / paths.csv> -o <out> [--mosaic 3]
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# %%
import base64
import collections
import csv
import json
import pathlib
import subprocess
from fractions import Fraction
import click
import numpy as np
from mcap.reader import make_reader
from umi.common.hevc_util import cut_to_steps

ROBOT_SIDES = ('right', 'left')   # UMI: robot0 right, robot1 left


def read_mcap(path: str) -> dict:
    """
    Everything the delivery needs from one episode MCAP: metadata records, per camera topic its
    (time ns, Annex B data) messages, per robot its gripper widths by time.
    """
    out = {'metadata': {}, 'video': collections.defaultdict(list), 'width': collections.defaultdict(dict)}
    with open(path, 'rb') as f:
        reader = make_reader(f)
        out['metadata'] = {m.name: dict(m.metadata) for m in reader.iter_metadata()}
        for _, channel, message in reader.iter_messages():
            topic = channel.topic
            if topic.startswith('/cameras/') and topic.endswith('/video'):
                out['video'][topic.split('/')[2]].append(
                    (message.log_time, base64.b64decode(json.loads(message.data)['data'])))
            elif topic.endswith('/gripper/state'):
                out['width'][int(topic.split('/')[0][5:])][message.log_time] = json.loads(message.data)['width']
    return out


def write_video(frames: list, fps: float, out_path: str, keep_parameter_sets: bool):
    """
    H.265 Annex B frames (MCAP video messages) into an mp4 by ffmpeg stream copy, constant fps.
    The parameter sets go into the file header (hvc1) unless keep_parameter_sets (video spliced
    from two encodings, --exact_start MCAPs: they stay in the stream, tag hev1).
    """
    rate = Fraction(fps).limit_denominator(1001)
    bsf = [] if keep_parameter_sets else ['-bsf:v', 'filter_units=remove_types=32|33|34']
    tag = 'hev1' if keep_parameter_sets else 'hvc1'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'hevc', '-framerate', f"{rate.numerator}/{rate.denominator}",
        '-i', 'pipe:0', '-c', 'copy', *bsf, '-tag:v', tag, '-movflags', '+faststart', str(out_path)],
        input=b''.join(data for _, data in frames), check=True, capture_output=True)


def mosaic(head: tuple, right: tuple, left: tuple, n_frames: int, out_path: str, fps: float,
        encoder: str = 'libx264', width: int = 1280):
    """
    Review mosaic (ffmpeg): head on top (width x 3/4 width), right and left wrist below (half width
    each), each input from its sync frame on ((file, sync_frame_index)), n_frames frames at the
    cameras' frame rate fps (every frame, none dropped or repeated), H.264 yuv420p faststart. Each
    input's frames are timed by their number from the sync frame (frame k at k / fps), so the three
    views are stacked frame by frame, not by their file timestamps (which differ by rounding).
    """
    w2 = width // 2
    h1, h2 = width * 3 // 4, w2 * 3 // 4
    inputs = list()
    for path, _ in (head, right, left):
        inputs += ['-reinit_filter', '0', '-i', str(path)]   # one filter graph even if the start's encoding differs
    rate = Fraction(fps).limit_denominator(1001)
    trims = [f"[{k}:v]trim=start_frame={sync}:end_frame={sync + n_frames},setpts=N*{rate.denominator}/({rate.numerator}*TB),"
        f"scale={size}:out_range=tv,setsar=1[v{k}]" for k, ((_, sync), size) in
        enumerate(zip((head, right, left), (f'{width}:{h1}', f'{w2}:{h2}', f'{w2}:{h2}')))]
    graph = ';'.join(trims) + ";[v1][v2]hstack=inputs=2[bottom];[v0][bottom]vstack=inputs=2,format=yuv420p[out]"
    codec = ['-c:v', encoder] + (['-preset', 'veryfast', '-crf', '23'] if encoder == 'libx264' else ['-preset', 'p4', '-cq', '23'])
    subprocess.run(['ffmpeg', '-v', 'error', '-y', *inputs, '-filter_complex', graph, '-map', '[out]', *codec,
        '-r', f"{rate.numerator}/{rate.denominator}", '-frames:v', str(n_frames),
        '-movflags', '+faststart', '-an', str(out_path)], check=True)


def export_mcap(path: str, rec: str, out_dir: pathlib.Path) -> dict:
    """
    One recording folder from one episode MCAP. Returns the files, sync frame indices and frame count.
    """
    data = read_mcap(path)
    recording = data['metadata'].get('recording', {})
    fps = float(recording.get('video_fps', 0)) or None
    roles = {k: v for k, v in recording.items() if k.startswith('camera')}   # camera0 -> robot0_wrist, ...
    role_of = lambda cam: {'robot0_wrist': 'right', 'robot1_wrist': 'left', 'head': 'head'}.get(roles.get(cam), cam)
    steps = sorted(data['width'][0])
    assert steps, f"{path}: no robot0/gripper/state messages"
    assert data['video'], f"{path}: no camera video"
    dt_ns = 1e9 / fps if fps else float(np.median(np.diff(steps)))
    folder = out_dir.joinpath(rec)
    folder.mkdir(parents=True, exist_ok=True)
    n = len(steps)
    exact = recording.get('video_start') == 'exact'
    rate = Fraction(fps or 1e9 / dt_ns).limit_denominator(1001)

    # every file exactly the episode: from this camera's frame at the first step, one frame per step.
    # An MCAP without --exact_start starts at the keyframe before the episode: the frames from the
    # episode start to the next keyframe are re-encoded here (07's --exact_start re-encode), the rest
    # is copied; an --exact_start MCAP is copied as it is.
    files, reencoded = dict(), dict()
    for cam, frames in sorted(data['video'].items()):
        role = role_of(cam)
        frames.sort(key=lambda x: x[0])
        try:
            _, reencoded[role], cut = cut_to_steps([t for t, _ in frames], [d for _, d in frames], steps, rate)
        except AssertionError as e:
            raise AssertionError(f"{path} {cam}: {e}") from None
        out = folder.joinpath(f"{rec}_{role}.mp4")
        # two encodings in one file (re-encoded start, here or by 07): parameter sets stay in the stream (hev1)
        write_video(cut, float(rate), str(out), keep_parameter_sets=exact or reencoded[role])
        files[role] = out.name

    roles_order = [r for r in ('head', 'left', 'right') if r in files]

    # gripper width per frame of that gripper's own video file (robot0 right wrist, robot1 left wrist):
    # frame i = episode step i
    for g, side in enumerate(ROBOT_SIDES):
        if side not in files or g not in data['width']:
            continue
        widths = data['width'][g]
        assert sorted(widths) == steps, f"{path}: robot{g} gripper widths not at the episode steps"
        with folder.joinpath(f"{rec}_{side}_gripper.csv").open('w', newline='') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerow(['frame', 'width_m'])
            for frame, t in enumerate(steps):
                w.writerow([frame, f"{widths[t]:.5f}"])

    camera_keys = ('model', 'lens', 'fov_horizontal_deg', 'fov_vertical_deg', 'fov_diagonal_deg', 'resolution',
        'frame_rate_hz', 'intrinsics_model', 'fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4')
    cameras = {role_of(k): {key: v[key] for key in camera_keys if key in v}
        for k, v in data['metadata'].items() if k.startswith('camera') and k[6:].isdigit()}
    meta = {
        'files': {r: files[r] for r in roles_order},
        'cameras': {r: cameras[r] for r in roles_order if r in cameras} or None,
        'sync': {
            'method': data['metadata'].get('sync', {}).get('method'),
            'event': "the episode's first step: frame 0 of every file (all files start at it, same frame count)",
            'frame_index': {r: 0 for r in roles_order},
        },
        'gripper_width': {side: f"{rec}_{side}_gripper.csv" for side in ROBOT_SIDES if side in files},
    }
    folder.joinpath(f"{rec}_meta.json").write_text(json.dumps(meta, indent=2))
    return {'recording': rec, 'folder': folder, 'files': files, 'sync_index': {r: 0 for r in files}, 'n_frames': n, 'reencoded': reencoded,
        'fps': fps}


def recording_name(task: str, path: str) -> str:
    stem = os.path.basename(path).rsplit('.', 1)[0]
    parts = [p for p in (task or '').split('/') if p and p not in ('.', 'dataset_mcap')]
    return '_'.join(parts + [stem])


# %%
@click.command()
@click.argument('source')
@click.option('-o', '--output', required=True, help='Output directory, one folder per recording')
@click.option('--mosaic', 'n_mosaic', type=int, default=0, help='Also a mosaic for this many recordings per task folder')
@click.option('--mosaic_files', default=None, help='Comma separated MCAP file names that get a mosaic (instead of --mosaic)')
@click.option('--mosaic_encoder', default='libx264', type=click.Choice(['libx264', 'h264_nvenc']))
@click.option('--overwrite', is_flag=True, default=False, help='Redo recordings whose folder has a meta.json')
def main(source, output, n_mosaic, mosaic_files, mosaic_encoder, overwrite):
    from report_episodes import list_mcaps
    from describe_mcaps import read_paths_csv
    out_dir = pathlib.Path(os.path.expanduser(output)).absolute()
    if source.endswith('.csv'):
        files = read_paths_csv(source)
    elif source.endswith('.mcap') and not source.startswith('s3://'):
        files = [(os.path.abspath(os.path.expanduser(source)), os.path.getsize(os.path.expanduser(source)), '.')]
    else:
        files = list_mcaps(source)
    assert files, f"no MCAP files in {source}"
    by_task = collections.defaultdict(list)
    for path, size, task in files:
        by_task[task or '.'].append(path)
    chosen = set()
    if mosaic_files:
        chosen = set(mosaic_files.split(','))
    elif n_mosaic > 0:
        for task, paths in by_task.items():
            paths = sorted(paths)
            chosen |= {os.path.basename(paths[int(round(i))]) for i in np.linspace(0, len(paths) - 1, min(n_mosaic, len(paths)))}
    print(f"{len(files)} MCAP files in {len(by_task)} task folders -> {out_dir}", flush=True)

    done, failed = 0, 0
    for n, (path, size, task) in enumerate(sorted(files, key=lambda f: (f[2], f[0])), 1):
        rec = recording_name(task, path)
        if not overwrite and out_dir.joinpath(rec, f"{rec}_meta.json").is_file():
            continue
        local = path
        try:
            if path.startswith('s3://'):
                local = str(out_dir.joinpath(f".{rec}.mcap"))
                out_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(['aws', 's3', 'cp', path, local, '--only-show-errors'], check=True)
            r = export_mcap(local, rec, out_dir)
            msg = f"{n}/{len(files)} {rec}: {r['n_frames']} frames in every file, start re-encoded: " + \
                (', '.join(k for k, v in sorted(r['reencoded'].items()) if v) or 'none')
            if os.path.basename(path) in chosen and all(k in r['files'] for k in ('head', 'right', 'left')):
                mosaic(*((r['folder'].joinpath(r['files'][k]), r['sync_index'][k]) for k in ('head', 'right', 'left')),
                    r['n_frames'], r['folder'].joinpath(f"{rec}_mosaic.mp4"), r['fps'], mosaic_encoder)
                msg += ', mosaic'
            print(msg, flush=True)
            done += 1
        except Exception as e:
            failed += 1
            print(f"{n}/{len(files)} {rec}: FAILED {type(e).__name__}: {e}", flush=True)
        finally:
            if local != path and os.path.exists(local):
                os.remove(local)
    print(f"done: {done} exported, {failed} failed (rerun to retry; existing recordings are skipped)")

# %%
if __name__ == "__main__":
    main()
