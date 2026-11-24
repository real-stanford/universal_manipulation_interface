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
        review video: head on top, left wrist bottom left, right wrist bottom right, frame by frame at the cameras' frame rate; H.264 yuv420p,
        any player.

Sync: the sync event is the episode's first step (the first robot0/gripper/state message); in each
camera's MCAP video the frame nearest to it (the MCAP's synchronized message times), within half a
frame, and the following frames at the following steps (checked).

Input: an MCAP file, a folder or s3:// prefix with MCAP files below it, or a paths CSV of
describe_mcaps.py --list_only. Output: a local folder or an s3:// prefix with the station / date /
task folders of the input (from the stationN folder in its path on, whatever folder is the input),
one folder per recording named as its MCAP file, the files in it named <rec>_head.mp4, ... with rec
the date / task + file name, no station (.../station2/2026-11-22/task01/episode_000004.mcap ->
<output>/station2/2026-11-22/task01/episode_000004/2026-11-22_task01_episode_000004_head.mp4). -j recordings at the same time (each job: download,
export, mosaic, upload), the CPU cores split between them. S3 input is downloaded into -w and deleted
after the export; with S3 output each recording folder is written in -w, uploaded (meta.json last,
sizes checked) and deleted, so the disk holds only the recordings in progress. Recordings whose
meta.json is in the output are skipped (a stopped run continues). Each recording's files are checked
before the upload (frame counts, codecs, frame rate, gripper widths, meta); a failed recording (frame
drop, too few frames, check, download, upload) is not written. report.json per task folder in
<output>/<task>/: per episode ok (frames, mosaic) / failed (reason, detail) / skipped.

python scripts_tag_tracking/14_export_rhoda.py <mcap / folder / s3:// / paths.csv> -o <out> [--mosaic 3]
python scripts_tag_tracking/14_export_rhoda.py s3://bucket/mcaps/station1 -o s3://bucket/rhoda/station1 -w ~/rhoda_work -j 8 --mosaic 3
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
import concurrent.futures
import csv
import datetime
import json
import pathlib
import re
import shutil
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


def write_video(frames, fps: float, out_path: str, keep_parameter_sets: bool):
    """
    H.265 Annex B frames ((index or time, data), e.g. MCAP video messages) into an mp4 by ffmpeg
    stream copy, constant fps; streamed, not held in memory. The parameter sets go into the file
    header (hvc1) unless keep_parameter_sets (video spliced from two encodings: they stay in the
    stream, tag hev1).
    """
    rate = Fraction(fps).limit_denominator(1001)
    bsf = [] if keep_parameter_sets else ['-bsf:v', 'filter_units=remove_types=32|33|34']
    tag = 'hev1' if keep_parameter_sets else 'hvc1'
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'hevc', '-framerate', f"{rate.numerator}/{rate.denominator}",
        '-i', 'pipe:0', '-c', 'copy', *bsf, '-tag:v', tag, '-movflags', '+faststart', str(out_path)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for _, data in frames:
            p.stdin.write(data)
        p.stdin.close()
    except BrokenPipeError:
        pass
    err = p.stderr.read()
    if p.wait() != 0:
        raise subprocess.CalledProcessError(p.returncode, 'ffmpeg', stderr=err)


CAMERA_KEYS = ('model', 'lens', 'fov_horizontal_deg', 'fov_vertical_deg', 'fov_diagonal_deg', 'resolution',
    'frame_rate_hz', 'intrinsics_model', 'fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4')


def write_widths_and_meta(folder: pathlib.Path, rec: str, files: dict, widths: dict, cameras: dict, sync_method):
    """
    The recording's gripper CSVs (<rec>_<side>_gripper.csv: frame, width_m, frame i = episode step i)
    and <rec>_meta.json. files: role (head / left / right) -> mp4 name; widths: side -> width per
    frame; cameras: role -> camera record (07's camera{j} metadata, only CAMERA_KEYS kept).
    """
    for side in ROBOT_SIDES:
        if side not in files or side not in widths:
            continue
        with folder.joinpath(f"{rec}_{side}_gripper.csv").open('w', newline='') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerow(['frame', 'width_m'])
            for frame, width in enumerate(widths[side]):
                w.writerow([frame, f"{width:.5f}"])
    roles_order = [r for r in ('head', 'left', 'right') if r in files]
    meta = {
        'files': {r: files[r] for r in roles_order},
        'cameras': {r: {k: cameras[r][k] for k in CAMERA_KEYS if k in cameras[r]} for r in roles_order
            if r in cameras} or None,
        'sync': {
            'method': sync_method,
            'event': "the episode's first step: frame 0 of every file (all files start at it, same frame count)",
            'frame_index': {r: 0 for r in roles_order},
        },
        'gripper_width': {side: f"{rec}_{side}_gripper.csv" for side in ROBOT_SIDES if side in files and side in widths},
    }
    folder.joinpath(f"{rec}_meta.json").write_text(json.dumps(meta, indent=2))


def upload_recording(folder: pathlib.Path, target: str):
    """
    A recording folder to target (s3:// prefix of the recording), meta.json last (a recording with
    meta.json on S3 is complete), every file's size checked.
    """
    names = sorted(p.name for p in folder.iterdir())
    names = [n for n in names if not n.endswith('_meta.json')] + [n for n in names if n.endswith('_meta.json')]
    for name in names:
        url = f"{target}/{name}"
        subprocess.run(['aws', 's3', 'cp', str(folder.joinpath(name)), url, '--only-show-errors'],
            check=True, capture_output=True)
        assert s3_size(url) == folder.joinpath(name).stat().st_size, f"{url}: uploaded size differs"


def mosaic(head: tuple, right: tuple, left: tuple, n_frames: int, out_path: str, fps: float,
        encoder: str = 'libx264', width: int = 1280, threads: int = 0):
    """
    Review mosaic (ffmpeg): head on top (width x 3/4 width), left wrist bottom left, right wrist bottom right (half width
    each), each input from its sync frame on ((file, sync_frame_index)), n_frames frames at the
    cameras' frame rate fps (every frame, none dropped or repeated), H.264 yuv420p faststart. Each
    input's frames are timed by their number from the sync frame (frame k at k / fps), so the three
    views are stacked frame by frame, not by their file timestamps (which differ by rounding).
    threads: CPU threads per decoder, filter graph and encoder (0: ffmpeg's default, all cores).
    """
    w2 = width // 2
    h1, h2 = width * 3 // 4, w2 * 3 // 4
    inputs = list()
    for path, _ in (head, right, left):
        inputs += ['-reinit_filter', '0', *(['-threads', str(threads)] if threads else []), '-i', str(path)]   # one filter graph even if the start's encoding differs
    rate = Fraction(fps).limit_denominator(1001)
    trims = [f"[{k}:v]trim=start_frame={sync}:end_frame={sync + n_frames},setpts=N*{rate.denominator}/({rate.numerator}*TB),"
        f"scale={size}:out_range=tv,setsar=1[v{k}]" for k, ((_, sync), size) in
        enumerate(zip((head, right, left), (f'{width}:{h1}', f'{w2}:{h2}', f'{w2}:{h2}')))]
    graph = ';'.join(trims) + ";[v2][v1]hstack=inputs=2[bottom];[v0][bottom]vstack=inputs=2,format=yuv420p[out]"
    codec = ['-c:v', encoder] + (['-preset', 'veryfast', '-crf', '23'] if encoder == 'libx264' else ['-preset', 'p4', '-cq', '23'])
    limit = ['-filter_complex_threads', str(threads), '-threads', str(threads)] if threads else []
    subprocess.run(['ffmpeg', '-v', 'error', '-y', *inputs, '-filter_complex', graph, '-map', '[out]', *codec, *limit,
        '-r', f"{rate.numerator}/{rate.denominator}", '-frames:v', str(n_frames),
        '-movflags', '+faststart', '-an', str(out_path)], check=True)


def export_mcap(path: str, rec: str, out_dir: pathlib.Path, threads: int = 0, folder_name: str = None) -> dict:
    """
    One recording folder (out_dir/<folder_name or rec>, files <rec>_head.mp4, ...) from one episode
    MCAP. Returns the files, sync frame indices and frame count.
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
    folder = out_dir.joinpath(folder_name or rec)
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
            _, reencoded[role], cut = cut_to_steps([t for t, _ in frames], [d for _, d in frames], steps, rate,
                threads=threads)
        except AssertionError as e:
            raise AssertionError(f"{path} {cam}: {e}") from None
        out = folder.joinpath(f"{rec}_{role}.mp4")
        # two encodings in one file (re-encoded start, here or by 07): parameter sets stay in the stream (hev1)
        write_video(cut, float(rate), str(out), keep_parameter_sets=exact or reencoded[role])
        files[role] = out.name

    # gripper width per frame of that gripper's own video file (robot0 right wrist, robot1 left wrist):
    # frame i = episode step i
    widths = dict()
    for g, side in enumerate(ROBOT_SIDES):
        if g in data['width']:
            assert sorted(data['width'][g]) == steps, f"{path}: robot{g} gripper widths not at the episode steps"
            widths[side] = [data['width'][g][t] for t in steps]
    cameras = {role_of(k): v for k, v in data['metadata'].items() if k.startswith('camera') and k[6:].isdigit()}
    write_widths_and_meta(folder, rec, files, widths, cameras, data['metadata'].get('sync', {}).get('method'))
    return {'recording': rec, 'folder': folder, 'files': files, 'sync_index': {r: 0 for r in files}, 'n_frames': n, 'reencoded': reencoded,
        'fps': fps}


def task_folder(path: str, task: str) -> str:
    """
    The recording's folder path in the output: from the station folder (stationN) of its path on, so
    the output has station / date / task whether the input is a station, a date or a task folder
    (s3://b/umi_data/industrial_data/station2/2026-11-22/task01/episode_000004.mcap ->
    station2/2026-11-22/task01); without a station folder in the path: task (relative to the input).
    """
    parts = [p for p in os.path.dirname(path.split('://', 1)[-1]).split('/') if p and p != 'dataset_mcap']
    station = next((k for k, p in enumerate(parts) if re.fullmatch(r'station\d+', p)), None)
    return '/'.join(parts[station:]) if station is not None else task


def recording_name(task: str, path: str) -> str:
    """
    File name prefix of a recording: date / task folders + file name, without the station folder
    (station2/2026-11-22/task01, episode_000004.mcap -> 2026-11-22_task01_episode_000004).
    """
    stem = os.path.basename(path).rsplit('.', 1)[0]
    parts = [p for p in (task or '').split('/') if p and p not in ('.', 'dataset_mcap') and not re.fullmatch(r'station\d+', p)]
    return '_'.join(parts + [stem])


def s3_size(url: str):
    bucket, _, key = url[len('s3://'):].partition('/')
    out = subprocess.run(['aws', 's3api', 'head-object', '--bucket', bucket, '--key', key, '--query', 'ContentLength'],
        capture_output=True, text=True)
    return int(out.stdout) if out.returncode == 0 else None


class OutputCheckError(Exception):
    pass


def check_outputs(folder: pathlib.Path, rec: str, r: dict, with_mosaic: bool):
    """
    The written recording before it is uploaded: every camera mp4 H.265 with one frame per step at
    the cameras' frame rate and one resolution, the mosaic H.264 with the same frame count, the
    gripper CSVs one width per frame, meta.json with sync frame index 0. Raises OutputCheckError.
    """
    n = r['n_frames']

    def probe(path):
        out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets', '-show_entries',
            'stream=codec_name,width,height,r_frame_rate,nb_read_packets', '-of', 'json', str(path)],
            capture_output=True, text=True)
        streams = json.loads(out.stdout or '{}').get('streams') if out.returncode == 0 else None
        if not streams:
            raise OutputCheckError(f"{path.name}: not readable")
        return streams[0]

    sizes = set()
    for role, name in r['files'].items():
        p = probe(folder.joinpath(name))
        if p['codec_name'] != 'hevc' or int(p['nb_read_packets']) != n:
            raise OutputCheckError(f"{name}: {p['codec_name']}, {p['nb_read_packets']} frames, {n} expected")
        if r['fps'] and abs(float(Fraction(p['r_frame_rate'])) - r['fps']) > 0.01:
            raise OutputCheckError(f"{name}: {p['r_frame_rate']} fps, {r['fps']} expected")
        sizes.add((p['width'], p['height']))
    if len(sizes) > 1:
        raise OutputCheckError(f"camera resolutions differ: {sorted(sizes)}")
    if with_mosaic:
        p = probe(folder.joinpath(f"{rec}_mosaic.mp4"))
        if p['codec_name'] != 'h264' or int(p['nb_read_packets']) != n:
            raise OutputCheckError(f"mosaic: {p['codec_name']}, {p['nb_read_packets']} frames, {n} expected")
    for side in ROBOT_SIDES:
        path = folder.joinpath(f"{rec}_{side}_gripper.csv")
        if side in r['files'] and path.is_file():
            rows = list(csv.reader(path.open()))[1:]
            if len(rows) != n or any(len(row) < 2 or row[1] in ('', 'nan') for row in rows):
                raise OutputCheckError(f"{path.name}: {len(rows)} rows, {n} expected, or widths missing")
    meta = json.loads(folder.joinpath(f"{rec}_meta.json").read_text())
    if set(meta['files']) != set(r['files']) or any(v != 0 for v in meta['sync']['frame_index'].values()):
        raise OutputCheckError("meta.json: files or sync frame index wrong")


def failure_reason(stage: str, e: Exception) -> str:
    text = str(e)
    if isinstance(e, OutputCheckError):
        return 'output_check'
    if stage in ('download', 'upload'):
        return stage
    if 'not evenly spaced' in text:
        return 'frame_drop'
    if 'not one per step' in text or 'frames after the cut' in text:
        return 'frame_count'
    return 'other'


def export_one(path: str, task: str, output: str, workdir: pathlib.Path, make_mosaic: bool, encoder: str,
        threads: int, overwrite: bool) -> dict:
    """
    One recording into <output>/<task>/<episode>/ (the folder named as the MCAP file), files
    <rec>_head.mp4, ... (rec: date / task + file name): download (S3 input) into workdir, export,
    mosaic, check (check_outputs), upload (S3 output; meta.json last, every file's size checked),
    then delete the download and, with S3 output, the local folder. A failed recording is not
    uploaded and its local folder is deleted. Returns its report entry (task, episode, status,
    frames, mosaic / reason, detail).
    """
    rec = recording_name(task, path)
    episode = os.path.basename(path).rsplit('.', 1)[0]
    s3_out = output.startswith('s3://')
    rel = '/'.join(p for p in (task or '').split('/') if p and p != '.')
    target = '/'.join(x for x in (output.rstrip('/'), rel, episode) if x)    # the recording's folder
    entry = {'task': rel, 'episode': episode}
    if not overwrite and (s3_size(f"{target}/{rec}_meta.json") is not None if s3_out
            else os.path.isfile(f"{target}/{rec}_meta.json")):
        return {**entry, 'status': 'skipped'}
    # S3 output: written in workdir/<task>/<episode> (episodes of the same name in other tasks run in parallel)
    out_dir = workdir.joinpath(rel) if s3_out else pathlib.Path(target).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    local = str(workdir.joinpath(f".{rec}.mcap")) if path.startswith('s3://') else path
    folder = out_dir.joinpath(episode)
    stage, ok = 'download', False
    try:
        if local != path:
            subprocess.run(['aws', 's3', 'cp', path, local, '--only-show-errors'], check=True, capture_output=True)
        stage = 'export'
        r = export_mcap(local, rec, out_dir, threads, folder_name=episode)
        if local != path:
            os.remove(local)                # the MCAP is not needed for the mosaic
        with_mosaic = make_mosaic and all(k in r['files'] for k in ('head', 'right', 'left'))
        if with_mosaic:
            mosaic(*((folder.joinpath(r['files'][k]), r['sync_index'][k]) for k in ('head', 'right', 'left')),
                r['n_frames'], folder.joinpath(f"{rec}_mosaic.mp4"), r['fps'], encoder, threads=threads)
        check_outputs(folder, rec, r, with_mosaic)
        if s3_out:
            stage = 'upload'
            upload_recording(folder, target)
        ok = True
        return {**entry, 'status': 'ok', 'frames': r['n_frames'], 'mosaic': with_mosaic}
    except Exception as e:
        stderr = getattr(e, 'stderr', None)
        stderr = stderr.decode(errors='replace') if isinstance(stderr, bytes) else stderr
        detail = (stderr.strip() if stderr else str(e)).replace(str(local), '').strip(' :')
        return {**entry, 'status': 'failed', 'reason': failure_reason(stage, e), 'detail': detail.splitlines()[-1][:300]
            if detail else type(e).__name__}
    finally:
        if local != path and os.path.exists(local):
            os.remove(local)
        if (s3_out or not ok) and folder.exists():
            shutil.rmtree(folder)           # S3: uploaded or failed; local: a failed recording is not kept


def write_reports(entries: list, source: str, output: str, workdir: pathlib.Path):
    """
    report.json per task folder, in <output>/<task>/ (uploaded with S3 output): this run's time,
    input and output, counts, and per episode its status (ok: frames, mosaic; failed: reason,
    detail). Episodes skipped in this run keep their entry of the previous report.json.
    """
    by_task = collections.defaultdict(list)
    for e in entries:
        by_task[e['task']].append(e)
    for task, items in sorted(by_task.items()):
        folder = '/'.join(x for x in (output.rstrip('/'), task) if x)
        url = f"{folder}/report.json"
        local = workdir.joinpath(task, 'report.json') if output.startswith('s3://') else pathlib.Path(url)
        local.parent.mkdir(parents=True, exist_ok=True)
        previous = dict()
        if output.startswith('s3://'):
            if subprocess.run(['aws', 's3', 'cp', url, str(local), '--only-show-errors'], capture_output=True).returncode != 0:
                local.unlink(missing_ok=True)
        if local.is_file():
            previous = {e['episode']: e for e in json.loads(local.read_text()).get('episodes', [])}
        episodes = list()
        for e in sorted(items, key=lambda e: e['episode']):
            e = {k: v for k, v in e.items() if k != 'task'}
            if e['status'] == 'skipped' and previous.get(e['episode'], {}).get('status') == 'ok':
                e = previous[e['episode']]
            episodes.append(e)
        report = {
            'run': {'time_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                'input': source, 'output': folder},
            'summary': {'total': len(episodes), **{s: sum(e['status'] == s for e in episodes)
                for s in ('ok', 'failed', 'skipped')}},
            'episodes': episodes,
        }
        local.write_text(json.dumps(report, indent=2))
        if output.startswith('s3://'):
            subprocess.run(['aws', 's3', 'cp', str(local), url, '--only-show-errors'], check=True)
            local.unlink()
        print(f"report: {url} ({report['summary']})", flush=True)


# %%
@click.command()
@click.argument('source')
@click.option('-o', '--output', required=True, help='Local folder or s3:// prefix: the input folder structure, one folder per recording')
@click.option('-w', '--workdir', default=None,
    help='Local folder for downloads and, with S3 output, the recordings until uploaded (deleted after)')
@click.option('-j', '--jobs', type=int, default=1, help='Recordings exported at the same time')
@click.option('--mosaic', 'n_mosaic', type=int, default=0, help='Also a mosaic for this many recordings per task folder')
@click.option('--mosaic_files', default=None, help='Comma separated MCAP file names that get a mosaic (instead of --mosaic)')
@click.option('--mosaic_encoder', default='libx264', type=click.Choice(['libx264', 'h264_nvenc']))
@click.option('--overwrite', is_flag=True, default=False, help='Redo recordings whose meta.json is in the output')
def main(source, output, workdir, jobs, n_mosaic, mosaic_files, mosaic_encoder, overwrite):
    from report_episodes import list_mcaps
    from describe_mcaps import read_paths_csv
    if not output.startswith('s3://'):
        output = str(pathlib.Path(os.path.expanduser(output)).absolute())
    if source.endswith('.csv'):
        files = read_paths_csv(source)
    elif source.endswith('.mcap') and not source.startswith('s3://'):
        files = [(os.path.abspath(os.path.expanduser(source)), os.path.getsize(os.path.expanduser(source)), '.')]
    elif source.endswith('.mcap'):
        files = [(source, s3_size(source), '.')]
    else:
        files = list_mcaps(source)
    assert files, f"no MCAP files in {source}"
    assert workdir or not output.startswith('s3://'), "S3 output: give a local --workdir"
    workdir = pathlib.Path(os.path.expanduser(workdir or output)).absolute()
    workdir.mkdir(parents=True, exist_ok=True)
    files = [(path, size, task_folder(path, task)) for path, size, task in files]
    by_task = collections.defaultdict(list)
    for path, size, task in files:
        by_task[task or '.'].append(path)
    # mosaics: --mosaic_files by file name (in every task folder), --mosaic N per task folder by path
    chosen = set()
    if mosaic_files:
        names = set(mosaic_files.split(','))
        chosen = {path for path, _, _ in files if os.path.basename(path) in names}
    elif n_mosaic > 0:
        for task, paths in by_task.items():
            paths = sorted(paths)
            chosen |= {paths[int(round(i))] for i in np.linspace(0, len(paths) - 1, min(n_mosaic, len(paths)))}
    # CPU threads per job: the cores shared by the jobs (one job: all cores, ffmpeg / x265 default)
    threads = 0 if jobs <= 1 else max(1, (os.cpu_count() or jobs) // jobs)
    print(f"{len(files)} MCAP files in {len(by_task)} task folders -> {output}, {jobs} jobs"
        + (f" x {threads} threads" if threads else ''), flush=True)

    entries = list()
    with concurrent.futures.ProcessPoolExecutor(jobs) as pool:
        futures = [pool.submit(export_one, path, task, output, workdir, path in chosen, mosaic_encoder, threads, overwrite)
            for path, _, task in sorted(files, key=lambda f: (f[2], f[0]))]
        for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
            e = future.result()
            entries.append(e)
            name = '/'.join(x for x in (e['task'], e['episode']) if x)
            what = {'ok': lambda: f"ok, {e['frames']} frames" + (', mosaic' if e['mosaic'] else ''),
                'skipped': lambda: 'in the output, skipped',
                'failed': lambda: f"FAILED {e['reason']}: {e['detail']}"}[e['status']]()
            print(f"{n}/{len(files)} {name}: {what}", flush=True)
    write_reports(entries, source, output, workdir)
    counts = collections.Counter(e['status'] for e in entries)
    print(f"done: {counts['ok']} ok, {counts['failed']} failed, {counts['skipped']} skipped "
        f"(rerun to retry the failed; recordings in the output are skipped)")

# %%
if __name__ == "__main__":
    main()
