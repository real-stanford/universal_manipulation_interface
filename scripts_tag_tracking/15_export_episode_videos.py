"""
Episode videos of a processed session, without poses or MCAP files: the data PRD's UMI delivery as
14_export_rhoda.py writes it, straight from the session's GoPro videos. For sessions processed with
run_tag_tracking_pipeline.py --videos_only (episodes, head tracking of the cubes for the motion sync,
wrist clock sync, gripper widths, dataset_plan.pkl with --ignore_poses), or any dataset_plan.pkl.

Per plan episode k, in <output>/<task>/episode_<k:06d>/ (files <rec>_..., rec: date / task +
episode, no station, as 14_export_rhoda.py):
  <rec>_head.mp4, <rec>_left.mp4, <rec>_right.mp4
      each camera's frames of the episode as the plan has them (audio sync start times + the motion
      sync correction of 03s_sync_wrists.py, whole frames): frame 0 = the episode's first step, one
      frame per step, the same count in every file. The frames from the start to the camera's next
      keyframe (at most 1 s) are re-encoded (07_export_mcap.py --exact_start), the rest is the GoPro
      stream, stream-copied (tag hev1 with a re-encoded start, else hvc1).
  <rec>_meta.json      per camera model, lens, field of view, resolution, frame rate, intrinsics;
                       sync method, frame index of the sync event in each file (0)
  <rec>_right_gripper.csv, <rec>_left_gripper.csv   gripper width per frame
  <rec>_mosaic.mp4     optional (--mosaic N per task, --mosaic_files episode_000003,...)
<output>/<task>/report.json: per episode ok / failed (reason, detail) / skipped.
Every recording is checked before the upload (14_export_rhoda.check_outputs); a failed one is not
written. With an s3:// output each recording is written in -w, uploaded (meta.json last, sizes
checked) and deleted. Recordings whose meta.json is in the output are skipped (--overwrite to redo).

--task: the recording's folders in the output, from the station on, e.g. station3/2026-11-22/task01.

python scripts_tag_tracking/15_export_episode_videos.py -i <session> -c <config> --task station3/2026-11-22/task01 \\
    -o s3://bucket/umi_data/industrial_data_mp4 -w ~/umi_data_mp4 -j 4 --mosaic 1000
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# %%
import concurrent.futures
import importlib
import itertools
import pathlib
import pickle
import shutil
import av
import click
import numpy as np
from umi.common.exiftool_util import ExifTool
from umi.common.hevc_util import cut_from, is_keyframe, video_messages
from umi.tag_tracking.camera_metadata import camera_record
from umi.tag_tracking.config import load_tag_tracking_config

rhoda = importlib.import_module('14_export_rhoda')
SYNC_METHOD = 'audio cross-correlation, wrist motion refinement'     # as 07_export_mcap.py


def episode_cameras(plan_episode: dict) -> list:
    """
    (role, camera) of an episode: the plan's wrist cameras in robot order (robot0 right, robot1 left),
    then the head camera.
    """
    n_grippers = len(plan_episode['grippers'])
    return [(rhoda.ROBOT_SIDES[c] if c < n_grippers else 'head', camera)
        for c, camera in enumerate(plan_episode['cameras'])]


def camera_records(plan: list, demos_dir: pathlib.Path, config: dict) -> dict:
    """
    video_path -> camera record (07's camera{j} metadata: model, lens, field of view, resolution,
    frame rate, intrinsics), read once per video.
    """
    paths = sorted({c['video_path'] for e in plan for c in e['cameras']})
    roles = {c['video_path']: role for e in plan for role, c in episode_cameras(e)}
    out = dict()
    with ExifTool() as et:
        for path in paths:
            info = list(et.get_metadata(str(demos_dir.joinpath(path))))[0]
            serial = info['QuickTime:CameraSerialNumber']
            with av.open(str(demos_dir.joinpath(path))) as container:
                fps = float(container.streams.video[0].average_rate)
            role = roles[path] if roles[path] == 'head' else f"{roles[path]}_wrist"
            out[path] = camera_record(role, serial, info, fps, config['cameras'][serial])
    return out


def write_camera_video(video_path: str, start: int, stop: int, out_path: str, threads: int) -> bool:
    """
    Frames [start, stop) of a GoPro video into an mp4, exactly: the frames up to the next keyframe
    re-encoded, the rest stream-copied. Returns whether the start was re-encoded.
    """
    with av.open(video_path) as container:
        rate = container.streams.video[0].average_rate
    # the start a keyframe: nothing re-encoded, one encoding, parameter sets into the header (hvc1)
    reencoded = not next((is_keyframe(data) for index, data in video_messages(video_path, start, start + 1)
        if index == start), False)
    frames = cut_from(video_messages(video_path, start, stop), start, rate, threads=threads)
    rhoda.write_video(frames, float(rate), out_path, keep_parameter_sets=reencoded)
    return reencoded


def export_episode(k: int, plan_episode: dict, demos_dir: pathlib.Path, task: str, output: str,
        workdir: pathlib.Path, records: dict, make_mosaic: bool, encoder: str, threads: int, overwrite: bool) -> dict:
    """
    One episode into <output>/<task>/episode_<k:06d>/: the three mp4s, gripper CSVs, meta.json,
    mosaic, checked, uploaded (s3:// output). Returns its report entry.
    """
    episode = f"episode_{k:06d}"
    rec = rhoda.recording_name(task, episode)
    s3_out = output.startswith('s3://')
    target = '/'.join(x for x in (output.rstrip('/'), task, episode) if x)
    entry = {'task': task, 'episode': episode}
    meta = f"{target}/{rec}_meta.json"
    if not overwrite and (rhoda.s3_size(meta) is not None if s3_out else os.path.isfile(meta)):
        return {**entry, 'status': 'skipped'}
    folder = workdir.joinpath(task, episode) if s3_out else pathlib.Path(target)
    stage, ok = 'export', False
    try:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        cameras = episode_cameras(plan_episode)
        ranges = [tuple(int(v) for v in c['video_start_end']) for _, c in cameras]
        n = ranges[0][1] - ranges[0][0]
        if any(b - a != n for a, b in ranges):
            raise AssertionError(f"frame counts differ between cameras: {ranges}")
        if len(plan_episode['episode_timestamps']) != n:
            raise AssertionError(f"{n} frames per camera, {len(plan_episode['episode_timestamps'])} steps")
        files, reencoded, fps = dict(), dict(), None
        for (role, camera), (a, b) in zip(cameras, ranges):
            path = str(demos_dir.joinpath(camera['video_path']))
            out = folder.joinpath(f"{rec}_{role}.mp4")
            reencoded[role] = write_camera_video(path, a, b, str(out), threads)
            files[role] = out.name
            fps = fps or float(records[camera['video_path']]['frame_rate_hz'])
        widths = {rhoda.ROBOT_SIDES[g]: np.asarray(gripper['gripper_width'], dtype=float)
            for g, gripper in enumerate(plan_episode['grippers'])}
        rhoda.write_widths_and_meta(folder, rec, files, widths,
            {role: records[c['video_path']] for role, c in cameras},
            SYNC_METHOD if any(c.get('clock_correction_ms') for _, c in cameras[:len(plan_episode['grippers'])])
            else 'audio cross-correlation')
        r = {'files': files, 'n_frames': n, 'fps': fps, 'sync_index': {role: 0 for role in files}}
        with_mosaic = make_mosaic and all(x in files for x in ('head', 'right', 'left'))
        if with_mosaic:
            rhoda.mosaic(*((folder.joinpath(files[x]), 0) for x in ('head', 'right', 'left')), n,
                folder.joinpath(f"{rec}_mosaic.mp4"), fps, encoder, threads=threads)
        rhoda.check_outputs(folder, rec, r, with_mosaic)
        if s3_out:
            stage = 'upload'
            rhoda.upload_recording(folder, target)
        ok = True
        return {**entry, 'status': 'ok', 'frames': n, 'mosaic': with_mosaic}
    except Exception as e:
        stderr = getattr(e, 'stderr', None)
        stderr = stderr.decode(errors='replace') if isinstance(stderr, bytes) else stderr
        detail = (stderr.strip() if stderr else str(e)) or type(e).__name__
        return {**entry, 'status': 'failed', 'reason': rhoda.failure_reason(stage, e), 'detail': detail.splitlines()[-1][:300]}
    finally:
        if (s3_out or not ok) and folder.exists():
            shutil.rmtree(folder)       # S3: uploaded or failed; local: a failed recording is not kept


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Session directory with dataset_plan.pkl')
@click.option('-c', '--config', required=True, help="The station's tag tracking config (camera intrinsics)")
@click.option('--task', required=True, help='Folders of the recordings in the output, e.g. station3/2026-11-22/task01')
@click.option('-o', '--output', required=True, help='Local folder or s3:// prefix')
@click.option('-w', '--workdir', default=None, help='Local folder for the recordings until uploaded (S3 output)')
@click.option('-j', '--jobs', type=int, default=1, help='Episodes exported at the same time')
@click.option('--mosaic', 'n_mosaic', type=int, default=0, help='Also a mosaic for this many episodes (spread evenly)')
@click.option('--mosaic_files', default=None, help='Comma separated episode names that get a mosaic (instead of --mosaic)')
@click.option('--mosaic_encoder', default='libx264', type=click.Choice(['libx264', 'h264_nvenc']))
@click.option('--overwrite', is_flag=True, default=False, help='Redo episodes whose meta.json is in the output')
def main(input, config, task, output, workdir, jobs, n_mosaic, mosaic_files, mosaic_encoder, overwrite):
    session = pathlib.Path(os.path.expanduser(input)).absolute()
    demos_dir = session.joinpath('demos')
    plan = pickle.load(session.joinpath('dataset_plan.pkl').open('rb'))
    assert plan, f"{session}: no episodes in dataset_plan.pkl"
    config = load_tag_tracking_config(config)
    task = task.strip('/')
    if not output.startswith('s3://'):
        output = str(pathlib.Path(os.path.expanduser(output)).absolute())
    assert workdir or not output.startswith('s3://'), "S3 output: give a local --workdir"
    workdir = pathlib.Path(os.path.expanduser(workdir or output)).absolute()
    workdir.mkdir(parents=True, exist_ok=True)
    names = [f"episode_{k:06d}" for k in range(len(plan))]
    if mosaic_files:
        chosen = set(mosaic_files.replace('.mcap', '').split(','))
    else:
        chosen = {names[int(round(i))] for i in np.linspace(0, len(names) - 1, min(n_mosaic, len(names)))} \
            if n_mosaic > 0 else set()
    records = camera_records(plan, demos_dir, config)
    threads = 0 if jobs <= 1 else max(1, (os.cpu_count() or jobs) // jobs)
    print(f"{session}: {len(plan)} episodes -> {output}/{task}, {jobs} jobs"
        + (f" x {threads} threads" if threads else ''), flush=True)

    entries = list()
    with concurrent.futures.ProcessPoolExecutor(jobs) as pool:
        futures = [pool.submit(export_episode, k, e, demos_dir, task, output, workdir, records,
            names[k] in chosen, mosaic_encoder, threads, overwrite) for k, e in enumerate(plan)]
        for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
            e = future.result()
            entries.append(e)
            what = {'ok': lambda: f"ok, {e['frames']} frames" + (', mosaic' if e['mosaic'] else ''),
                'skipped': lambda: 'in the output, skipped',
                'failed': lambda: f"FAILED {e['reason']}: {e['detail']}"}[e['status']]()
            print(f"{n}/{len(plan)} {task}/{e['episode']}: {what}", flush=True)
    rhoda.write_reports(entries, str(session), output, workdir)
    counts = {s: sum(e['status'] == s for e in entries) for s in ('ok', 'failed', 'skipped')}
    print(f"done: {counts['ok']} ok, {counts['failed']} failed, {counts['skipped']} skipped "
        f"(rerun to retry the failed; episodes in the output are skipped)")
    return counts

# %%
if __name__ == "__main__":
    main()
