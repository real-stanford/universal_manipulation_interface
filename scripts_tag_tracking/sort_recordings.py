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

python scripts_tag_tracking/sort_recordings.py ~/Videos/station_2026-11-18 --save ~/Videos/station_2026-11-18/inventory.json
python scripts_tag_tracking/sort_recordings.py ~/Videos/station_2026-11-18 --tasks tasks.yaml -o ~/Videos/station_2026-11-18/sessions
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# %%
import datetime
import json
import pathlib
import shutil
from typing import Dict, List
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


def place(src: pathlib.Path, dst_dir: pathlib.Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir.joinpath(src.name)
    if not dst.exists():
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    return dst


# %%
@click.command()
@click.argument('root')
@click.option('--min_sec', type=float, default=5.0, help='Flag videos shorter than this')
@click.option('--save', default=None, help='Write the inventory json here')
@click.option('--tasks', default=None, help='tasks.yaml: build the sessions')
@click.option('-o', '--output', default=None, help='Sessions directory (with --tasks)')
def main(root, min_sec, save, tasks, output):
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
        json.dump({'root': str(root), 'videos': videos, 'recordings': [[v['file'] for v in g] for g in groups]},
            open(os.path.expanduser(save), 'w'), indent=2)
        print(f"\nSaved {save}")

    if tasks:
        assert output, "-o sessions directory required with --tasks"
        output = pathlib.Path(os.path.expanduser(output)).absolute()
        spec = yaml.safe_load(open(os.path.expanduser(tasks), 'r'))
        for name, s in spec.items():
            raw = output.joinpath(name, 'raw_videos')
            for k in s.get('recordings', []):
                for v in groups[k]:
                    place(root.joinpath(v['file']), raw)
            for f in s.get('gripper_calibration', []):
                place(root.joinpath(f), raw.joinpath('gripper_calibration'))
            for f in s.get('head_mapping', []):
                place(root.joinpath(f), raw.joinpath('head_mapping'))
            n = sum(1 for _ in raw.rglob('*.MP4'))
            print(f"{name}: {n} videos in {raw}")

# %%
if __name__ == "__main__":
    main()
