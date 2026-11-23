"""
One CSV row per episode MCAP (07_export_mcap.py) saying what happens in it: --frames head camera
frames (default 6, evenly over the episode) are decoded from the MCAP and Claude describes the
task from them (structured JSON: task sentence, objects, hands used, visible outcome, notes).

Input: a folder or s3:// prefix with MCAP files anywhere below it (as report_episodes.py). On S3
only the parts of each file holding those frames and the metadata are read (ranged reads, aws
CLI), not the whole file.

Output CSV columns: task (folder of the MCAP relative to the input), file, duration_s, good (a
pose for every step, report_episodes.py), task_description, objects, hands, outcome, notes,
model. Rows already in the CSV are skipped, so a run can be stopped and continued.
--sheets <dir>: also save the frames Claude saw, one image per MCAP, for checking.
--dry_run: frames and sheets only, no API call.

Needs the anthropic package and Anthropic credentials (ANTHROPIC_API_KEY or `ant auth login`).
Cost: about 6 images + 300 words of answer per MCAP.

python scripts_tag_tracking/describe_mcaps.py s3://my-bucket/umi_data/industrial_data/ -o mcap_tasks.csv [--sheets sheets/]
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
import csv
import json
import pathlib
import threading
import click
import av
import av.logging
import cv2
import numpy as np
from mcap.reader import make_reader
from report_episodes import S3File, list_mcaps, episode_info

COLUMNS = ['task', 'file', 'duration_s', 'good', 'task_description', 'objects', 'hands', 'outcome', 'notes', 'model']
IMAGE_WIDTH = 768

SYSTEM = """You look at frames from the head camera of a person doing a manipulation task with two
handheld grippers (white and red, with black-and-white tag cubes on them; the cubes and the tags on
the table and walls are tracking markers, not task objects). The frames are in time order across
one episode. Describe only what the frames show; where they do not show something clearly, say so
instead of guessing."""

PROMPT = """These {n} frames are from one episode ({duration:.0f} s), at the times given above each frame.
Say what task the person does in this episode."""

SCHEMA = {
    'type': 'object',
    'properties': {
        'task_description': {'type': 'string',
            'description': 'One sentence: what the person does in this episode, with the objects and where they go'},
        'objects': {'type': 'array', 'items': {'type': 'string'},
            'description': 'Task objects handled or moved (not the grippers, cubes or tags)'},
        'hands': {'type': 'string', 'enum': ['both', 'right', 'left', 'unclear'],
            'description': 'Which grippers handle objects (right / left as seen from the head camera)'},
        'outcome': {'type': 'string', 'enum': ['success', 'failure', 'partial', 'unclear'],
            'description': 'Whether the last frames show the task done'},
        'notes': {'type': 'string',
            'description': 'Anything that makes the episode doubtful (nothing happening, objects dropped, '
            'grippers out of view, blur), else empty'},
    },
    'required': ['task_description', 'objects', 'hands', 'outcome', 'notes'],
    'additionalProperties': False,
}


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
    # 07_export_mcap.py puts VPS / SPS / PPS (32-34) into every keyframe; IRAP pictures are 16-23
    return bool(nal_types(data) & {32, 16, 17, 18, 19, 20, 21})


def open_mcap(path: str, size: int):
    return S3File(path, size) if path.startswith('s3://') else open(path, 'rb')


def head_frames(path: str, size: int, n: int):
    """
    n head camera frames evenly over the episode, with their seconds from the first frame, and
    the episode's recording metadata.
    """
    with open_mcap(path, size) as stream:
        reader = make_reader(stream)
        recording = {m.name: m.metadata for m in reader.iter_metadata()}.get('recording', {})
        cameras = [k for k, v in recording.items() if k.startswith('camera') and v == 'head']
        assert cameras, f"{path}: no head camera in the MCAP metadata"
        topic = f"/cameras/{cameras[0]}/video"
        summary = reader.get_summary()
        channel = [c for c in summary.channels.values() if c.topic == topic]
        assert channel, f"{path}: no {topic}"
        stats = summary.statistics
        t0, t1 = stats.message_start_time, stats.message_end_time
        # the videos start at the keyframe up to 1 s before the episode: frames from 1 s in
        start = t0 + int(1e9)
        targets = [int(start + (t1 - 0.3e9 - start) * k / max(n - 1, 1)) for k in range(n)]
        frames = list()
        for target in targets:
            # decode from the last keyframe before the target (GoPro: one per second) up to the target
            packets = [base64.b64decode(json.loads(message.data)['data']) for _, _, message in
                reader.iter_messages(topics=[topic], starting_at=target - int(1.5e9), ending_at=target)]
            keys = [k for k, data in enumerate(packets) if is_keyframe(data)]
            if not keys:
                continue
            codec = av.CodecContext.create('hevc', 'r')
            last = None
            for data in packets[keys[-1]:]:
                for packet in codec.parse(data):
                    for frame in codec.decode(packet):
                        last = frame
            for packet in codec.parse(b''):
                for frame in codec.decode(packet):
                    last = frame
            for frame in codec.decode(None):
                last = frame
            if last is not None:
                img = last.to_ndarray(format='bgr24')
                h, w = img.shape[:2]
                frames.append(((target - t0) / 1e9 - 1.0,
                    cv2.resize(img, (IMAGE_WIDTH, int(round(h * IMAGE_WIDTH / w))), interpolation=cv2.INTER_AREA)))
    return frames, recording


def sheet(frames: list) -> np.ndarray:
    """
    The frames side by side, two rows, each with its time.
    """
    tiles = list()
    for t, img in frames:
        img = img.copy()
        cv2.putText(img, f"{t:.1f} s", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
        cv2.putText(img, f"{t:.1f} s", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        tiles.append(img)
    cols = (len(tiles) + 1) // 2
    blank = np.zeros_like(tiles[0])
    rows = [np.hstack(tiles[r * cols:(r + 1) * cols] + [blank] * (cols - len(tiles[r * cols:(r + 1) * cols])))
        for r in range(2) if tiles[r * cols:(r + 1) * cols]]
    return np.vstack(rows)


def describe(client, model: str, effort: str, frames: list, duration: float) -> tuple:
    """
    Claude's description of the episode from its frames; (answer dict, model that answered).
    """
    content = list()
    for t, img in frames:
        jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]
        content.append({'type': 'text', 'text': f"{t:.1f} s"})
        content.append({'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg',
            'data': base64.standard_b64encode(jpg.tobytes()).decode('ascii')}})
    content.append({'type': 'text', 'text': PROMPT.format(n=len(frames), duration=duration)})
    response = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        betas=['server-side-fallback-2026-07-01'],
        fallbacks='default',
        output_config={'effort': effort, 'format': {'type': 'json_schema', 'schema': SCHEMA}},
        system=SYSTEM,
        messages=[{'role': 'user', 'content': content}],
    )
    if response.stop_reason != 'end_turn':
        raise RuntimeError(f"stop reason {response.stop_reason} {response.stop_details or ''}")
    return json.loads(next(b.text for b in response.content if b.type == 'text')), response.model


# %%
@click.command()
@click.argument('root')
@click.option('-o', '--output', required=True, help='CSV file (rows already in it are skipped)')
@click.option('--frames', 'n_frames', type=int, default=6, help='Head camera frames per MCAP')
@click.option('--sheets', default=None, help='Also save the frames Claude saw here, one image per MCAP')
@click.option('--model', default='claude-opus-5-5')
@click.option('--effort', default='low', type=click.Choice(['low', 'medium', 'high']),
    help='Describing a few frames needs little reasoning')
@click.option('--dry_run', is_flag=True, default=False, help='Frames and sheets only, no API call, no CSV')
@click.option('-j', '--jobs', type=int, default=4, help='MCAP files in parallel')
def main(root, output, n_frames, sheets, model, effort, dry_run, jobs):
    av.logging.set_level(av.logging.ERROR)   # swscaler 'deprecated pixel format' per frame
    files = list_mcaps(root)
    assert files, f"no .mcap files under {root}"
    output = pathlib.Path(os.path.expanduser(output))
    done = set()
    if output.is_file():
        done = {(r['task'], r['file']) for r in csv.DictReader(output.open())}
    todo = [f for f in files if (f[2] or '.', f[0].rsplit('/', 1)[-1]) not in done]
    print(f"{len(files)} MCAP files under {root}, {len(files) - len(todo)} already in {output}, {len(todo)} to do",
        flush=True)
    if sheets:
        pathlib.Path(os.path.expanduser(sheets)).mkdir(parents=True, exist_ok=True)
    client = None
    if not dry_run:
        import anthropic
        client = anthropic.Anthropic()
        if not output.is_file():
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open('w', newline='') as f:
                csv.writer(f).writerow(COLUMNS)
    lock = threading.Lock()

    def one(item):
        path, size, task = item
        task = task or '.'
        name = path.rsplit('/', 1)[-1]
        frames, recording = head_frames(path, size, n_frames)
        assert frames, "no head frames decoded"
        info = episode_info(path, size)
        if sheets:
            cv2.imwrite(str(pathlib.Path(os.path.expanduser(sheets)).joinpath(
                ('' if task == '.' else task.replace('/', '_') + '_') + name.rsplit('.', 1)[0] + '.jpg')), sheet(frames))
        if dry_run:
            return f"{task}/{name}: {len(frames)} frames"
        answer, answered_by = describe(client, model, effort, frames, info['duration_s'])
        row = [task, name, round(info['duration_s'], 1), info['good'], answer['task_description'],
            '; '.join(answer['objects']), answer['hands'], answer['outcome'], answer['notes'], answered_by]
        with lock, output.open('a', newline='') as f:
            csv.writer(f).writerow(row)
        return f"{task}/{name}: {answer['task_description']}"

    failed = 0
    pool = concurrent.futures.ThreadPoolExecutor(jobs)
    futures = {pool.submit(one, item): item for item in todo}
    try:
        for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
            item = futures[future]
            try:
                print(f"{n}/{len(todo)} {future.result()}", flush=True)
            except Exception as e:
                failed += 1
                print(f"{n}/{len(todo)} {item[2]}/{item[0].rsplit('/', 1)[-1]}: FAILED {type(e).__name__}: {e}",
                    flush=True)
    except KeyboardInterrupt:
        # Ctrl-C: drop the files not started, the ones running finish; their rows are kept
        pool.shutdown(wait=False, cancel_futures=True)
        print("stopped; rerun to continue (rows already written are skipped)")
        raise SystemExit(1)
    pool.shutdown()
    print(f"done, {len(todo) - failed} described, {failed} failed (rerun to retry them)" +
        ('' if dry_run else f"; {output}"))

# %%
if __name__ == "__main__":
    main()
