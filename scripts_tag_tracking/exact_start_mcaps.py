"""
Episode MCAP files (07_export_mcap.py without --exact_start: each camera's video starts at the
keyframe before the episode) rewritten as with --exact_start, from the MCAP alone (no session, no
GoPro videos): every camera's video starts exactly at its frame of the episode's first step and has
one frame per step.

Per camera, the frame nearest the first step (robot0/gripper/state) is the start; the frames from it
to the next keyframe are re-encoded (07's re-encode, umi.common.hevc_util.cut_from: libx265, no
B-frames, full range), the rest is the original stream; frames after the episode's last step are
dropped. Message times and every other topic stay as they are. Metadata: recording video_start
exact; sync without camera{j}_sync_frame (every video starts at the episode start); source with
camera{j}_source_first_frame (the old camera{j}_sync_frame) if the MCAP has no source record.
MCAP files already exported with --exact_start are copied as they are.

Input: an MCAP file, a folder or s3:// prefix with MCAP files below it, or a paths CSV of
describe_mcaps.py --list_only. Output: a local folder or an s3:// prefix, with the input's task
folders below it (s3://in/station1/task02/episode_000003.mcap -> <output>/station1/task02/...).
S3 files are downloaded to --workdir one at a time per job and deleted after; S3 output is uploaded,
its size checked, then the local file deleted. Files already in the output are skipped (--overwrite
to redo).

python scripts_tag_tracking/exact_start_mcaps.py s3://bucket/umi_data/industrial_data/station1 -o s3://bucket/umi_data/industrial_data_exact/station1 -w ~/exact_work
python scripts_tag_tracking/exact_start_mcaps.py ~/data/station1/task02/dataset_mcap/episode_000003.mcap -o ~/data/exact
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# %%
import base64
import concurrent.futures
import json
import pathlib
import shutil
import subprocess
from fractions import Fraction
import click
import numpy as np
from mcap.reader import make_reader
from mcap.writer import Writer, CompressionType
from umi.common.hevc_util import cut_to_steps


def exact_start(in_path: str, out_path: str) -> str:
    """
    Rewrite one episode MCAP with every camera video starting at the episode's first step.
    Returns a one-line summary.
    """
    with open(in_path, 'rb') as f:
        reader = make_reader(f)
        header = reader.get_header()
        summary = reader.get_summary()
        metadata = {m.name: dict(m.metadata) for m in reader.iter_metadata()}
        attachments = list(reader.iter_attachments())
        messages = [(channel.id, message.log_time, message.publish_time, message.data)
            for _, channel, message in reader.iter_messages(log_time_order=True)]
    recording = metadata.get('recording', {})
    if recording.get('video_start') == 'exact':
        shutil.copyfile(in_path, out_path)
        return 'already exact start, copied'

    channels = summary.channels
    topic_id = {c.topic: i for i, c in channels.items()}
    gripper = topic_id.get('robot0/gripper/state')
    assert gripper is not None, f"{in_path}: no robot0/gripper/state"
    steps = np.array([t for c, t, _, _ in messages if c == gripper], dtype=np.int64)
    n = len(steps)
    fps = float(recording['video_fps']) if recording.get('video_fps') else 1e9 / float(np.median(np.diff(steps)))
    rate = Fraction(fps).limit_denominator(1001)

    video_ids = sorted(i for i, c in channels.items() if c.topic.startswith('/cameras/') and c.topic.endswith('/video'))
    assert video_ids, f"{in_path}: no camera video"
    new_video, starts, reencoded = dict(), dict(), dict()
    for cid in video_ids:
        cam = channels[cid].topic.split('/')[2]
        frames = [(t, p, json.loads(d)) for c, t, p, d in messages if c == cid]
        try:
            start, reencoded[cam], cut = cut_to_steps([t for t, _, _ in frames],
                [base64.b64decode(m['data']) for _, _, m in frames], steps, rate)
        except AssertionError as e:
            raise AssertionError(f"{in_path} {cam}: {e}") from None
        new_video[cid] = [(cid, frames[i][0], frames[i][1],
            json.dumps({**frames[i][2], 'data': base64.b64encode(annexb).decode()}).encode()) for i, annexb in cut]
        starts[cam] = start

    others = [m for m in messages if m[0] not in new_video]
    merged = sorted(others + [m for out in new_video.values() for m in out], key=lambda m: m[1])

    sync_record = metadata.get('sync')
    old_sync_frames = dict()
    if sync_record is not None:
        old_sync_frames = {k[:-len('_sync_frame')]: v for k, v in sync_record.items() if k.endswith('_sync_frame')}
        metadata['sync'] = {k: v for k, v in sync_record.items() if not k.endswith('_sync_frame')}
    metadata['recording'] = {**recording, 'video_start': 'exact'}
    if 'source' not in metadata and old_sync_frames:
        metadata['source'] = {
            **{f'{cam}_source_first_frame': v for cam, v in old_sync_frames.items()},
            **{f'{cam}_source_end_frame': str(int(v) + n) for cam, v in old_sync_frames.items()},
        }

    with open(out_path, 'wb') as f:
        writer = Writer(f, compression=CompressionType.ZSTD)
        writer.start(profile=header.profile, library=header.library)
        schema_ids = {i: writer.register_schema(name=s.name, encoding=s.encoding, data=s.data)
            for i, s in summary.schemas.items()}
        channel_ids = {i: writer.register_channel(topic=c.topic, message_encoding=c.message_encoding,
            schema_id=schema_ids.get(c.schema_id, 0), metadata=dict(c.metadata)) for i, c in channels.items()}
        sequence = dict()
        for cid, t, p, d in merged:
            sequence[cid] = sequence.get(cid, -1) + 1
            writer.add_message(channel_ids[cid], log_time=t, publish_time=p, data=d, sequence=sequence[cid])
        for name, record in metadata.items():
            writer.add_metadata(name, record)
        for a in attachments:
            writer.add_attachment(create_time=a.create_time, log_time=a.log_time, name=a.name,
                media_type=a.media_type, data=a.data)
        writer.finish()
    return f"{n} frames per camera from the episode start; start frame " + \
        ', '.join(f"{cam} {starts[cam]}" + (' (re-encoded to the next keyframe)' if reencoded[cam] else ' (keyframe)')
            for cam in sorted(starts))


def s3_size(url: str):
    bucket, _, key = url[len('s3://'):].partition('/')
    out = subprocess.run(['aws', 's3api', 'head-object', '--bucket', bucket, '--key', key, '--query', 'ContentLength'],
        capture_output=True, text=True)
    return int(out.stdout) if out.returncode == 0 else None


def process(path: str, task: str, output: str, workdir: pathlib.Path, overwrite: bool) -> str:
    name = os.path.basename(path)
    rel = '/'.join(p for p in (task or '').split('/') if p and p != '.')
    target = '/'.join(x for x in (output.rstrip('/'), rel, name) if x)
    if not overwrite and (s3_size(target) is not None if target.startswith('s3://') else os.path.isfile(target)):
        return f"{rel}/{name}: in the output, skipped"
    stem = (rel.replace('/', '_') + '_' if rel else '') + name
    local_in, local_out = path, target
    temporary = list()                      # downloads and staged uploads, deleted at the end
    if path.startswith('s3://'):
        local_in = str(workdir.joinpath(f".{stem}.in"))
        temporary.append(local_in)
    if target.startswith('s3://'):
        local_out = str(workdir.joinpath(f".{stem}.exact"))
        temporary.append(local_out)
    temporary.append(local_out + '.part')
    try:
        if path.startswith('s3://'):
            subprocess.run(['aws', 's3', 'cp', path, local_in, '--only-show-errors'], check=True)
        os.makedirs(os.path.dirname(local_out), exist_ok=True)
        done = exact_start(local_in, local_out + '.part')
        os.replace(local_out + '.part', local_out)
        if target.startswith('s3://'):
            subprocess.run(['aws', 's3', 'cp', local_out, target, '--only-show-errors'], check=True)
            assert s3_size(target) == os.path.getsize(local_out), f"{target}: uploaded size differs"
        return f"{rel}/{name}: {done}"
    finally:
        for p in temporary:
            if os.path.exists(p):
                os.remove(p)


# %%
@click.command()
@click.argument('source')
@click.option('-o', '--output', required=True, help='Local folder or s3:// prefix; task folders of the input below it')
@click.option('-w', '--workdir', default=None, help='Local folder for downloads and uploads (needed with S3)')
@click.option('-j', '--jobs', type=int, default=2, help='MCAP files at a time (each holds one MCAP in memory)')
@click.option('--overwrite', is_flag=True, default=False, help='Redo files already in the output')
def main(source, output, workdir, jobs, overwrite):
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
    needs_workdir = output.startswith('s3://') or any(p.startswith('s3://') for p, _, _ in files)
    assert workdir or not needs_workdir, "S3 input or output: give a local --workdir"
    workdir = pathlib.Path(os.path.expanduser(workdir or output)).absolute()
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} MCAP files -> {output}", flush=True)

    failed = 0
    with concurrent.futures.ProcessPoolExecutor(jobs) as pool:
        futures = {pool.submit(process, path, task, output, workdir, overwrite): (path, task)
            for path, _, task in sorted(files, key=lambda f: (f[2], f[0]))}
        for k, future in enumerate(concurrent.futures.as_completed(futures), 1):
            path, task = futures[future]
            try:
                print(f"{k}/{len(files)} {future.result()}", flush=True)
            except Exception as e:
                failed += 1
                print(f"{k}/{len(files)} {task}/{os.path.basename(path)}: FAILED {type(e).__name__}: {e}", flush=True)
    print(f"done: {len(files) - failed} ok, {failed} failed (rerun to retry; files in the output are skipped)")

# %%
if __name__ == "__main__":
    main()
