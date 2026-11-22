"""
Sort a recording day's GoPro videos into tag tracking sessions.

1. Inventory (default): every video under the given folders with its folder
   (head / left / right as copied from the cameras), camera serial, start time
   (camera timecode, local), length and size; videos recorded at the same
   time (overlapping in time across cameras) form a recording. Prints the
   recordings in time order with the gap before each, so task blocks stand
   out, and flags what does not fit: a folder with several camera serials,
   recordings without all three folders, very short files (< --min_sec).
   --save writes the inventory as json for step 2.

2. Sessions (--tasks tasks.yaml): links the videos into
   <output>/<session>/raw_videos/ as run_tag_tracking_pipeline.py expects:

     session_name:
       recordings: [3, 4, 5]                  # recording numbers of the inventory
       gripper_calibration: [right/GX010199.MP4, left/GX010284.MP4]
       head_mapping: [head/GX019288.MP4]

   Hard links (copies if on another file system); existing files are kept.
   Each session also gets raw_videos/pairs.csv: which videos were recorded
   together (one row per recording: start, per camera folder its file and
   length, the time all cameras recorded), and the gripper calibration and
   head mapping files. Rerunning on built sessions only (re)writes the sheets.

3. Proposed tasks (--propose tasks.yaml): writes the tasks.yaml of step 2 from the
   inventory and what the short recordings (< --take_sec, every camera) show, read
   from their tags (umi.tag_tracking.station_config.clip_role, sizes from --aruco_config):
   - gripper calibration: a wrist camera sees its finger tags open and close fully,
     at least twice (that wrist is calibrated by it; one recording may calibrate both);
   - head mapping: otherwise, the head sees environment tags;
   - anything else is left out, with what was seen.
   Recordings >= --take_sec with every camera are takes. A task starts at the first take
   after a head mapping not used yet, with every wrist calibrated before it; its gripper
   calibration is the latest per wrist and its head mapping the latest before that take.
   A recalibration without a new head mapping stays in the task (noted). Left out, listed
   in the file: recordings missing a camera or with a video < --min_sec, takes before a
   calibration and mapping. Every short clip's role and evidence is listed in the file.
   Tasks are named task01, task02, ... in time order. On S3 the short clips are read
   through presigned URLs (nothing downloaded).

S3: root and output can be s3://bucket/prefix (aws CLI, its credentials). The
inventory reads only each video's MP4 header (camera serial from the first
64 KB, start time and length from the movie header 'mvhd': creation time UTC,
whole seconds, shown in local time); sessions are server-side copies.

python scripts_tag_tracking/sort_recordings.py <day_folder> --save <day_folder>/inventory.json --propose tasks.yaml
python scripts_tag_tracking/sort_recordings.py <day_folder> --tasks tasks.yaml -o <sessions_folder>
python scripts_tag_tracking/sort_recordings.py s3://<bucket>/<prefix>/<station>/<date> --save inventory.json --propose tasks.yaml
python scripts_tag_tracking/sort_recordings.py s3://<bucket>/<prefix>/<station>/<date> --tasks tasks.yaml -o s3://<bucket>/<prefix>/<station>/<sessions>
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# %%
import collections
import concurrent.futures
import datetime
import json
import pathlib
import re
import shutil
import subprocess
import tempfile
import time
from typing import Dict, List, Optional, Tuple
import click
import av
import yaml
from umi.common.exiftool_util import ExifTool
from umi.common.timecode_util import mp4_get_start_datetime


def video_info(path: pathlib.Path, root: pathlib.Path, et) -> dict:
    meta = list(et.get_metadata(str(path)))[0]
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration = float(stream.frames / stream.average_rate) if stream.frames else float(container.duration / 1e6)
    start = mp4_get_start_datetime(str(path))
    return {'file': str(path.relative_to(root)), 'folder': path.relative_to(root).parts[0],
        'serial': meta.get('QuickTime:CameraSerialNumber'), 'start': start.isoformat(),
        'duration': duration, 'size_mb': path.stat().st_size / 1e6}


MP4_EPOCH = datetime.datetime(1904, 1, 1, tzinfo=datetime.timezone.utc)


def top_level_atoms(data: bytes) -> List[Tuple[bytes, int, int]]:
    """
    (type, offset, size) of the top-level MP4 atoms whose headers lie in data (the file's first
    bytes); the last one may extend past data. size 0 = to the end of the file.
    """
    atoms, offset = list(), 0
    while offset + 8 <= len(data):
        size = int.from_bytes(data[offset:offset + 4], 'big')
        kind = data[offset + 4:offset + 8]
        if size == 1:
            if offset + 16 > len(data):
                break
            size = int.from_bytes(data[offset + 8:offset + 16], 'big')
        atoms.append((kind, offset, size))
        if size == 0:
            break
        offset += size
    return atoms


def parse_mvhd(data: bytes) -> Optional[Tuple[datetime.datetime, float]]:
    """
    (creation time UTC, duration s) from the bytes of a moov atom (its 'mvhd' near the start).
    """
    i = data.find(b'mvhd')
    if i < 0:
        return None
    version = data[i + 4]
    if version == 1:
        created = int.from_bytes(data[i + 8:i + 16], 'big')
        scale = int.from_bytes(data[i + 24:i + 28], 'big')
        duration = int.from_bytes(data[i + 28:i + 36], 'big')
    else:
        created = int.from_bytes(data[i + 8:i + 12], 'big')
        scale = int.from_bytes(data[i + 16:i + 20], 'big')
        duration = int.from_bytes(data[i + 20:i + 24], 'big')
    return MP4_EPOCH + datetime.timedelta(seconds=created), duration / scale if scale else 0.0


def split_s3(url: str) -> Tuple[str, str]:
    bucket, _, prefix = url[len('s3://'):].partition('/')
    return bucket, prefix.rstrip('/')


def s3_list(url: str) -> List[Tuple[str, int]]:
    bucket, prefix = split_s3(url)
    out = subprocess.run(['aws', 's3api', 'list-objects-v2', '--bucket', bucket, '--prefix', prefix + '/',
        '--output', 'json'], check=True, capture_output=True, text=True).stdout
    return [(o['Key'], o['Size']) for o in (json.loads(out) if out.strip() else {}).get('Contents', [])]


def s3_range(bucket: str, key: str, start: int, end: int) -> bytes:
    with tempfile.NamedTemporaryFile() as f:
        subprocess.run(['aws', 's3api', 'get-object', '--bucket', bucket, '--key', key,
            '--range', f'bytes={start}-{end}', f.name], check=True, capture_output=True)
        return pathlib.Path(f.name).read_bytes()


def s3_video_info(bucket: str, prefix: str, key: str, size: int) -> dict:
    head = s3_range(bucket, key, 0, 65535)
    serial = re.search(rb'C\d{13}', head)
    moov = [a for a in top_level_atoms(head) if a[0] == b'moov']
    if not moov:
        # moov after the media data: the atom right after the last one whose header is in head
        last = top_level_atoms(head)[-1]
        moov_offset = last[1] + last[2]
    else:
        moov_offset = moov[0][1]
    parsed = parse_mvhd(s3_range(bucket, key, moov_offset, min(moov_offset + 4095, size - 1)))
    assert parsed is not None, f"s3://{bucket}/{key}: no mvhd at {moov_offset}"
    created, duration = parsed
    rel = key[len(prefix) + 1:]
    return {'file': rel, 'folder': rel.split('/')[0], 'serial': serial.group().decode() if serial else None,
        'start': created.astimezone().replace(tzinfo=None).isoformat(), 'duration': duration, 'size_mb': size / 1e6}


def s3_size(url: str) -> Optional[int]:
    bucket, key = split_s3(url)
    r = subprocess.run(['aws', 's3api', 'head-object', '--bucket', bucket, '--key', key, '--query', 'ContentLength',
        '--output', 'text'], capture_output=True, text=True)
    return int(r.stdout.strip()) if r.returncode == 0 else None


def s3_place(src: str, dst_dir: str, name: str):
    """
    Server-side copy of src to dst_dir/name; skipped if the same file (same size) is already
    there, an error if another file is.
    """
    dst = f"{dst_dir}/{name}"
    existing = s3_size(dst)
    if existing is not None:
        assert existing == s3_size(src), f"{dst} exists with another size than {src}: not overwritten"
        return dst
    subprocess.run(['aws', 's3', 'cp', src, dst, '--only-show-errors'], check=True)
    return dst


def session_files(groups: List[List[dict]], spec: dict) -> List[Tuple[str, str]]:
    """
    (file relative to the day folder, sub folder of raw_videos/) of a session (tasks.yaml entry):
    every file of its recordings (a camera can have several in one recording), then the gripper
    calibration and head mapping files; without the files listed under exclude (e.g. a camera's
    file copied into another camera's folder).
    """
    exclude = set(spec.get('exclude', []))
    out = [(v['file'], '') for k in spec.get('recordings', []) for v in groups[k] if v['file'] not in exclude]
    out += [(f, 'gripper_calibration') for f in spec.get('gripper_calibration', []) if f not in exclude]
    out += [(f, 'head_mapping') for f in spec.get('head_mapping', []) if f not in exclude]
    return out


def placed_names(files: List[Tuple[str, str]]) -> Dict[str, str]:
    """
    File -> its name in the session: the camera's file name, prefixed with its camera folder
    (left_GX010136.MP4) where files of two cameras with the same name would land in the same
    raw_videos folder (each GoPro numbers its own files). The pipeline reads camera and time
    from the video, not from the name.
    """
    count = collections.Counter((sub, f.split('/')[-1]) for f, sub in set(files))
    return {f: f.split('/')[-1] if count[(sub, f.split('/')[-1])] == 1 else f"{f.split('/')[0]}_{f.split('/')[-1]}"
        for f, sub in files}


def pairs_rows(groups: List[List[dict]], spec: dict, folders: List[str],
        names: Optional[Dict[str, str]] = None) -> List[dict]:
    """
    Rows of a session's pairs.csv: per recording of spec (tasks.yaml entry) its start / end,
    per camera folder its file(s) (as named in raw_videos/, names: placed_names; several
    joined with '; ' when the camera split or restarted the recording) and their total length,
    the time all folders recorded together, and a note on missing cameras, cameras stopping
    early, several files of a camera and gaps between them; then the gripper calibration and
    head mapping files.
    """
    name = lambda f: (names or {}).get(f, f.split('/')[-1])
    exclude = set(spec.get('exclude', []))
    rows = list()
    for k in spec.get('recordings', []):
        group = [v for v in groups[k] if v['file'] not in exclude]
        start, end = span(group)
        by_folder = dict()
        for v in sorted(group, key=lambda v: v['start']):
            by_folder.setdefault(v['folder'], list()).append(v)
        row = {'type': 'take', 'recording': k, 'start': f"{start:%Y-%m-%d %H:%M:%S}", 'end': f"{end:%H:%M:%S}"}
        notes, starts, ends = list(), list(), list()
        for f in folders:
            vs = by_folder.get(f, [])
            row[f] = '; '.join(name(v['file']) for v in vs)
            row[f'{f}_sec'] = round(sum(v['duration'] for v in vs), 1) if vs else ''
            if not vs:
                continue
            s0 = [datetime.datetime.fromisoformat(v['start']) for v in vs]
            e0 = [s + datetime.timedelta(seconds=v['duration']) for s, v in zip(s0, vs)]
            starts.append(min(s0))
            ends.append(max(e0))
            if len(vs) > 1:
                gaps = [(s0[i + 1] - e0[i]).total_seconds() for i in range(len(vs) - 1)]
                notes.append(f"{f}: {len(vs)} files" + (f", gaps {', '.join(f'{g:.0f}' for g in gaps)} s"
                    if max(gaps) > 1.5 else ' (split while recording)'))
            if (max(e0) - min(s0)).total_seconds() < (end - start).total_seconds() - 30:
                notes.append(f"{f} stops after {(max(e0) - min(s0)).total_seconds() / 60:.1f} min")
        shared = (min(ends) - max(starts)).total_seconds() if len(starts) == len(folders) else 0.0
        row['shared_sec'] = round(max(shared, 0.0), 1)
        missing = [f for f in folders if f not in by_folder]
        row['note'] = '; '.join((['missing: ' + ', '.join(missing)] if missing else []) + notes)
        rows.append(row)
    for kind, files in (('gripper_calibration', spec.get('gripper_calibration', [])), ('head_mapping', spec.get('head_mapping', []))):
        for path in files:
            row = {'type': kind, 'recording': '', 'start': '', 'end': '', 'shared_sec': '', 'note': f"{kind}/{name(path)}"}
            for f in folders:
                row[f] = name(path) if path.split('/')[0] == f else ''
                row[f'{f}_sec'] = ''
            rows.append(row)
    return rows


def pairs_csv(rows: List[dict], folders: List[str]) -> str:
    import csv
    import io
    columns = ['type', 'recording', 'start', 'end'] + [c for f in folders for c in (f, f'{f}_sec')] + ['shared_sec', 'note']
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=columns, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def is_complete(group: List[dict], folders: List[str], min_sec: float) -> bool:
    """
    Every camera folder recorded at least min_sec in the recording (several files of a camera summed).
    """
    total = collections.Counter()
    for v in group:
        total[v['folder']] += v['duration']
    return all(total[f] >= min_sec for f in folders)


def recording_length(group: List[dict]) -> float:
    start, end = span(group)
    return (end - start).total_seconds()


def propose_tasks(groups: List[List[dict]], folders: List[str], min_sec: float, take_sec: float,
        roles: Dict[int, tuple]):
    """
    Tasks from the recordings (see the module docstring, step 3). roles: per short complete
    recording (role, calibrated wrist folders, evidence) from what it shows (clip_roles).
    Returns ({name: {'recordings', 'gripper_calibration', 'head_mapping'}}, {name: comment},
    [(recording, reason) left out]).
    """
    wrists = [f for f in folders if f != 'head']
    tasks, comments, left_out = dict(), dict(), list()
    current, calibration, mapping, mapping_used = None, dict(), None, None
    recalibrations, used = list(), set()   # calibrations within a task; calibrations of a task
    for k, g in enumerate(groups):
        if not is_complete(g, folders, min_sec):
            missing = [f for f in folders if f not in {v['folder'] for v in g}]
            left_out.append((k, f"missing {', '.join(missing)}" if missing else f"a video shorter than {min_sec:g} s"))
            continue
        if recording_length(g) < take_sec:
            role, calibrated, evidence = roles[k]
            if role == 'gripper_calibration':
                calibration.update({w: k for w in calibrated})
                if current is not None and tasks[current]['recordings']:
                    recalibrations.append((current, k, calibrated))
            elif role == 'head_mapping':
                mapping = k
            else:
                left_out.append((k, f"short recording, neither gripper calibration nor head mapping ({evidence})"))
            continue
        if mapping is not None and mapping != mapping_used and all(w in calibration for w in wrists):
            current = f"task{len(tasks) + 1:02d}"
            tasks[current] = {'recordings': [],
                'gripper_calibration': [v['file'] for w in wrists for v in groups[calibration[w]] if v['folder'] == w],
                'head_mapping': [v['file'] for v in groups[mapping] if v['folder'] == 'head']}
            comments[current] = 'gripper calibration = ' + ', '.join(f"{w} recording {calibration[w]}" for w in wrists) \
                + f", head mapping = recording {mapping}"
            mapping_used = mapping
            used.update(calibration[w] for w in wrists)
        if current is None:
            have = ([] if mapping is None else ['head mapping']) + [f"{w} calibration" for w in wrists if w in calibration]
            left_out.append((k, f"take before a gripper calibration and head mapping (have: {', '.join(have) or 'none'})"))
            continue
        tasks[current]['recordings'].append(k)
    for name, k, calibrated in recalibrations:
        if k not in used:
            comments[name] += f"; recalibration {', '.join(calibrated)} = recording {k} (not used)"
    return tasks, comments, sorted(left_out)


def sample_or_none(url: str, predefined: str, every_sec: float, attempts: int = 3) -> Optional[dict]:
    """
    sample_detections, retried; None if the video cannot be read (the error without the URL,
    a presigned URL carries a temporary credential).
    """
    from umi.tag_tracking.tag_detection import sample_detections
    error = None
    for _ in range(attempts):
        try:
            return sample_detections(url, predefined, every_sec)
        except Exception as e:
            error = f"{type(e).__name__}: {str(e).split(': ')[0]}"
    print(f"{url.split('?')[0]}: unreadable after {attempts} attempts ({error})", flush=True)
    return None


def clip_roles(groups: List[List[dict]], folders: List[str], min_sec: float, take_sec: float, source,
        aruco_config: str, finger_size: float, cube_size: float, every_sec: float, jobs: int) -> Dict[int, tuple]:
    """
    Per short complete recording what it shows (umi.tag_tracking.station_config.clip_role),
    from its videos' tags sampled every every_sec. source(file) -> path or URL of a video.
    """
    from umi.tag_tracking.tag_detection import tag_inventory
    from umi.tag_tracking.station_config import tag_classes, finger_pair, finger_opening, clip_role
    aruco = yaml.safe_load(open(aruco_config, 'r'))
    sizes = {int(k): float(v) for k, v in aruco['marker_size_map'].items() if k != 'default'}
    classes = tag_classes(sizes, finger_size, cube_size)
    short = [k for k, g in enumerate(groups) if is_complete(g, folders, min_sec) and recording_length(g) < take_sec]
    videos = [v for k in short for v in groups[k]]
    print(f"\nreading {len(videos)} videos of {len(short)} short recordings "
        f"({sum(v['duration'] for v in videos):.0f} s, {sum(v.get('size_mb', 0) for v in videos):.0f} MB), "
        f"a frame every {every_sec:g} s", flush=True)
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(sample_or_none, source(v['file']), aruco['aruco_dict']['predefined'], every_sec): i
            for i, v in enumerate(videos)}
        results = [None] * len(videos)
        for n, f in enumerate(concurrent.futures.as_completed(futures), 1):
            i = futures[f]
            results[i] = f.result()
            print(f"  {n}/{len(videos)} {videos[i]['file']} ({time.time() - t0:.0f} s)", flush=True)
    by_file = {v['file']: r for v, r in zip(videos, results)}
    roles = dict()
    for k in short:
        unreadable = [v['file'] for v in groups[k] if by_file[v['file']] is None]
        if unreadable:
            roles[k] = ('unknown', [], f"unreadable: {', '.join(unreadable)}")
            continue
        head = [by_file[v['file']] for v in groups[k] if v['folder'] == 'head'][0]
        openings = dict()
        for v in groups[k]:
            if v['folder'] != 'head':
                r = by_file[v['file']]
                pair = finger_pair(tag_inventory(r['detections'], r['size']), classes['finger'], min_share=0.3)
                openings[v['folder']] = finger_opening(r['detections'], r['size'], pair)
        roles[k] = clip_role(tag_inventory(head['detections'], head['size']), openings, classes['environment'])
    return roles


def tasks_yaml(tasks: dict, comments: dict, left_out: list, groups: List[List[dict]], root: str,
        roles: Dict[int, tuple] = None) -> str:
    """
    The proposed tasks as tasks.yaml text, with times, notes and the short clips' roles as comments.
    """
    lines = [f"# proposed from {root} by sort_recordings.py --propose",
        "# recording numbers as printed by the inventory"]
    for k, (role, calibrated, evidence) in sorted((roles or {}).items()):
        lines.append(f"# recording {k} ({span(groups[k])[0]:%H:%M:%S}): {role}"
            + (f" {', '.join(calibrated)}" if calibrated else '') + f" ({evidence})")
    for name, t in tasks.items():
        start = span(groups[t['recordings'][0]])[0] if t['recordings'] else None
        end = span(groups[t['recordings'][-1]])[1] if t['recordings'] else None
        lines.append(f"{name}:   # {start:%H:%M}-{end:%H:%M}, {len(t['recordings'])} recordings; {comments[name]}"
            if start else f"{name}:   # no takes; {comments[name]}")
        lines.append(f"  recordings: [{', '.join(str(k) for k in t['recordings'])}]")
        lines.append(f"  gripper_calibration: [{', '.join(t['gripper_calibration'])}]")
        lines.append(f"  head_mapping: [{', '.join(t['head_mapping'])}]")
    if left_out:
        lines.append("# left out:")
        lines += [f"#   recording {k} ({span(groups[k])[0]:%H:%M:%S}): {why}" for k, why in left_out]
    return '\n'.join(lines) + '\n'


def group_recordings(videos: List[dict]) -> List[List[dict]]:
    """
    Videos overlapping in time (any camera) in one recording, in start order.
    """
    items = sorted(videos, key=lambda v: v['start'])
    groups, end = list(), None
    for v in items:
        s = datetime.datetime.fromisoformat(v['start'])
        e = s + datetime.timedelta(seconds=v['duration'])
        if groups and s < end:
            groups[-1].append(v)
            end = max(end, e)
        else:
            groups.append([v])
            end = e
    return groups


def span(group: List[dict]):
    starts = [datetime.datetime.fromisoformat(v['start']) for v in group]
    ends = [s + datetime.timedelta(seconds=v['duration']) for s, v in zip(starts, group)]
    return min(starts), max(ends)


def place(src: pathlib.Path, dst_dir: pathlib.Path, name: str):
    """
    Hard link (copy on another file system) of src as dst_dir/name; skipped if the same file
    (same size) is already there, an error if another file is.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir.joinpath(name)
    if dst.exists():
        assert dst.stat().st_size == src.stat().st_size, f"{dst} exists with another size than {src}: not overwritten"
        return dst
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


# %%
@click.command()
@click.argument('root')  # local folder or s3://bucket/prefix
@click.option('--min_sec', type=float, default=5.0, help='Flag videos shorter than this')
@click.option('--save', default=None, help='Write the inventory json here')
@click.option('--propose', default=None, help='Write proposed tasks (tasks.yaml) here')
@click.option('--take_sec', type=float, default=60.0, help='--propose: recordings at least this long are takes')
@click.option('--tasks', default=None, help='tasks.yaml: build the sessions')
@click.option('-o', '--output', default=None, help='Sessions directory (with --tasks)')
@click.option('-a', '--aruco_config', default=os.path.join(ROOT_DIR, 'example/tag_tracking/aruco_config.yaml'),
    help='--propose: tag sizes (finger, cube, environment tags)')
@click.option('--finger_size', type=float, default=0.016, help='--propose: finger tag side [m]')
@click.option('--cube_size', type=float, default=0.06, help='--propose: cube tag side [m]')
@click.option('--every_sec', type=float, default=0.5, help='--propose: short clips sampled every this many seconds')
@click.option('-j', '--jobs', type=int, default=8, help='--propose: videos read in parallel')
def main(root, min_sec, save, propose, take_sec, tasks, output, aruco_config, finger_size, cube_size, every_sec, jobs):
    s3 = root.startswith('s3://')
    if s3:
        root = root.rstrip('/')
        bucket, prefix = split_s3(root)
        objects = [(k, n) for k, n in s3_list(root) if k.lower().endswith('.mp4') and '/sessions/' not in k]
        with concurrent.futures.ThreadPoolExecutor(16) as executor:
            videos = list(executor.map(lambda o: s3_video_info(bucket, prefix, *o), objects))
    else:
        root = pathlib.Path(os.path.expanduser(root)).absolute()
        paths = sorted(p for p in root.rglob('*') if p.suffix.lower() == '.mp4' and 'sessions' not in p.parts)
        with ExifTool() as et:
            videos = [video_info(p, root, et) for p in paths]
    groups = group_recordings(videos)

    serials: Dict[str, set] = dict()
    for v in videos:
        serials.setdefault(v['folder'], set()).add(v['serial'])
    print(f"{len(videos)} videos in {root}")
    for folder, s in sorted(serials.items()):
        print(f"  {folder}: cameras {sorted(map(str, s))}" + ('  <- several cameras' if len(s) > 1 else ''))
    folders = sorted(serials)
    previous_end = None
    for k, group in enumerate(groups):
        start, end = span(group)
        gap = '' if previous_end is None else f", {(start - previous_end).total_seconds():.0f} s after the previous"
        missing = [f for f in folders if f not in {v['folder'] for v in group}]
        print(f"\nrecording {k}: {start:%H:%M:%S}-{end:%H:%M:%S} ({(end - start).total_seconds():.0f} s{gap})"
            + (f"  missing: {', '.join(missing)}" if missing else ''))
        for v in sorted(group, key=lambda v: (v['folder'], v['start'])):
            flag = '  <- short' if v['duration'] < min_sec else ''
            print(f"  {v['file']:<24} {v['serial']}  {v['start'][11:19]}  {v['duration']:7.1f} s  {v['size_mb']:8.0f} MB{flag}")
        previous_end = end
    if save:
        pathlib.Path(os.path.expanduser(save)).parent.mkdir(parents=True, exist_ok=True)
        json.dump({'root': str(root), 'videos': videos, 'recordings': [[v['file'] for v in g] for g in groups]},
            open(os.path.expanduser(save), 'w'), indent=2)
        print(f"\nSaved {save}")
    if propose:
        if s3:
            source = lambda f: subprocess.run(['aws', 's3', 'presign', f"{root}/{f}", '--expires-in', '21600'],
                check=True, capture_output=True, text=True).stdout.strip()
        else:
            source = lambda f: str(root.joinpath(f))
        roles = clip_roles(groups, folders, min_sec, take_sec, source, os.path.expanduser(aruco_config),
            finger_size, cube_size, every_sec, jobs)
        proposed, comments, left_out = propose_tasks(groups, folders, min_sec, take_sec, roles)
        text = tasks_yaml(proposed, comments, left_out, groups, str(root), roles)
        pathlib.Path(os.path.expanduser(propose)).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.expanduser(propose)).write_text(text)
        print(f"\n{text}\nSaved {propose}")

    if tasks:
        assert output, "-o sessions directory required with --tasks"
        spec = yaml.safe_load(open(os.path.expanduser(tasks), 'r'))
        if s3:
            assert output.startswith('s3://'), "an S3 root needs an S3 output (-o s3://...)"
            output = output.rstrip('/')
            for name, s in spec.items():
                raw = f"{output}/{name}/raw_videos"
                files = session_files(groups, s)
                names = placed_names(files)
                jobs = [(f"{root}/{f}", f"{raw}/{sub}".rstrip('/'), names[f]) for f, sub in files]
                with concurrent.futures.ThreadPoolExecutor(8) as executor:
                    placed = list(executor.map(lambda j: s3_place(*j), jobs))
                renamed = sorted(f for f in names if names[f] != f.split('/')[-1])
                if renamed:
                    print(f"{name}: same file name from two cameras, placed as " + ', '.join(names[f] for f in renamed))
                sheet = pairs_csv(pairs_rows(groups, s, folders, names), folders)
                b, key = split_s3(f"{raw}/pairs.csv")
                with tempfile.NamedTemporaryFile('w', suffix='.csv') as f:
                    f.write(sheet)
                    f.flush()
                    subprocess.run(['aws', 's3', 'cp', f.name, f"s3://{b}/{key}", '--only-show-errors'], check=True)
                print(f"{name}: {len(placed)} videos and pairs.csv in {raw}")
            return
        output = pathlib.Path(os.path.expanduser(output)).absolute()
        for name, s in spec.items():
            raw = output.joinpath(name, 'raw_videos')
            files = session_files(groups, s)
            names = placed_names(files)
            for f, sub in files:
                place(root.joinpath(f), raw.joinpath(sub) if sub else raw, names[f])
            renamed = sorted(f for f in names if names[f] != f.split('/')[-1])
            if renamed:
                print(f"{name}: same file name from two cameras, placed as " + ', '.join(names[f] for f in renamed))
            raw.joinpath('pairs.csv').write_text(pairs_csv(pairs_rows(groups, s, folders, names), folders))
            n = sum(1 for _ in raw.rglob('*.MP4'))
            print(f"{name}: {n} videos and pairs.csv in {raw}")

# %%
if __name__ == "__main__":
    main()
