"""
Yield report: hours recorded per station and day, how many of them reached the task sessions and
the episode MCAP files, and where the rest went. From S3 only (the processing logs are not needed):

1. recorded: the day folders' GoPro videos (<raw>/<station>/<day>/head|left|right, MP4 headers only,
   as sort_recordings.py reads them), grouped into recordings (videos overlapping in time);
2. in tasks: the takes of the sessions (<raw>/<station>/industrial_data/<date>/<task>, industrial_data/<task>
   or industrial_data_<date>/<task>, raw_videos/pairs.csv),
   matched to the recordings by file name; per take the time all three cameras recorded. Not in a
   task, per recording: a camera missing, a short clip used as gripper calibration / head mapping,
   another short clip, a complete take outside every task; and in a task, the time the cameras did
   not all record (a camera started late or stopped early);
3. in episodes: the episode MCAP files (<mcap>/<station>/<date>/<task>/, the date from the session's
   path or the day folder its files are in; metadata and time range read
   by ranged reads) placed into their session's takes by their own clock (one offset per session:
   the cameras' timecode against the headers' UTC time, searched within +-14 h); the take time
   outside episodes split into gaps up to --gap_sec (docking and idle between episodes) and longer
   gaps (episodes dropped, or no task done). MCAP files that fit no take of their session (another
   day's episodes in that folder) are counted apart, not as yield. A session without MCAP files:
   not processed or failed (the reason is not on S3; --notes adds it).

Output: <output>_days.csv (station, day), <output>_tasks.csv (session), and a funnel per station.
Inventories are cached in <output>_inventory/ (rerun without reading the headers again).

--notes: CSV station,session,note (session as in the tasks CSV, e.g. 2026-11-22/task01) for what S3
does not say, e.g. why a session has no MCAP files.

python scripts_tag_tracking/report_yield.py --raw s3://bucket/upload_ingestion/GoPro/UMI_Collection \\
    --mcap s3://bucket/umi_data/industrial_data -o ~/yield/yield [--stations station1,station3]
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# %%
import collections
import concurrent.futures
import csv
import datetime
import io
import json
import pathlib
import re
import subprocess
import click
from mcap.reader import make_reader
from sort_recordings import s3_list, s3_video_info, split_s3, group_recordings, span
from report_episodes import S3File, list_mcaps

FOLDERS = ['head', 'left', 'right']
STATION = re.compile(r'station\d+')


def day_folders(raw: str, station: str) -> list:
    """
    Folders of a station with head/ left/ right videos (the uploaded recording days), relative to the station.
    """
    keys = [k for k, _ in s3_list(f"{raw}/{station}") if k.lower().endswith('.mp4')]
    prefix = split_s3(f"{raw}/{station}")[1] + '/'
    days = set()
    for k in keys:
        parts = k[len(prefix):].split('/')
        if len(parts) >= 3 and parts[-2] in FOLDERS and not any(p.startswith('industrial_data') for p in parts):
            days.add('/'.join(parts[:-2]))
    return sorted(days)


def inventory(raw: str, station: str, day: str, cache: pathlib.Path) -> list:
    """
    The day folder's videos (file, folder, serial, start, duration) as sort_recordings.py reads them; cached.
    """
    path = cache.joinpath(f"{station}_{day.replace('/', '_')}.json")
    if path.is_file():
        return json.loads(path.read_text())
    root = f"{raw}/{station}/{day}"
    bucket, prefix = split_s3(root)
    objects = [(k, n) for k, n in s3_list(root) if k.lower().endswith('.mp4') and k[len(prefix) + 1:].split('/')[0] in FOLDERS]
    with concurrent.futures.ThreadPoolExecutor(16) as executor:
        videos = list(executor.map(lambda o: s3_video_info(bucket, prefix, *o), objects))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(videos))
    return videos


def read_text(url: str) -> str:
    return subprocess.run(['aws', 's3', 'cp', url, '-', '--only-show-errors'], check=True, capture_output=True,
        text=True).stdout


def sessions(raw: str, station: str) -> dict:
    """
    session (path below the station: industrial_data/<date>/<task>, industrial_data/<task> without a
    date, or industrial_data_<date>/<task>) -> pairs.csv rows.
    """
    root = f"{raw}/{station}"
    bucket, prefix = split_s3(root)
    out = dict()
    for k, _ in s3_list(root):
        if k.endswith('/raw_videos/pairs.csv'):
            name = k[len(prefix) + 1:-len('/raw_videos/pairs.csv')]
            out[name] = list(csv.DictReader(io.StringIO(read_text(f"s3://{bucket}/{k}"))))
    return out


def file_key(folder: str, name: str) -> tuple:
    name = name.strip()
    return folder, name[len(folder) + 1:] if name.startswith(f"{folder}_") else name


def shared_seconds(group: list) -> float:
    """
    Time all three camera folders recorded in a recording (several files of a camera: first start to last end).
    """
    starts, ends = list(), list()
    for f in FOLDERS:
        vs = [v for v in group if v['folder'] == f]
        if not vs:
            return 0.0
        s0 = [datetime.datetime.fromisoformat(v['start']) for v in vs]
        starts.append(min(s0))
        ends.append(max(s + datetime.timedelta(seconds=v['duration']) for s, v in zip(s0, vs)))
    return max((min(ends) - max(starts)).total_seconds(), 0.0)


def shared_interval(group: list):
    starts, ends = list(), list()
    for f in FOLDERS:
        vs = [v for v in group if v['folder'] == f]
        s0 = [datetime.datetime.fromisoformat(v['start']) for v in vs]
        starts.append(min(s0))
        ends.append(max(s + datetime.timedelta(seconds=v['duration']) for s, v in zip(s0, vs)))
    return max(starts), min(ends)


def match_sessions(session_rows: dict, where: dict, days: list):
    """
    Sessions' takes and clips matched to the recordings, within one day folder (GoPro file names
    repeat across days: each camera numbers its own files): the day with most of the session's
    files; a file only if its length agrees with pairs.csv (within 2 s). where: (day, folder, name)
    -> (recording, duration). Returns (recording -> sessions using it as a take, recording ->
    sessions using it as a clip, session -> [recordings of each take], session -> day).
    """
    take_of, clip_of = collections.defaultdict(set), collections.defaultdict(set)
    session_takes, session_day = dict(), dict()
    for name, rows in session_rows.items():
        files = [file_key(f, x) for r in rows for f in FOLDERS for x in (r.get(f) or '').split(';') if x.strip()]
        votes = collections.Counter(d for d in days for k in files if (d, *k) in where)
        day = votes.most_common(1)[0][0] if votes else ''
        session_day[name] = day
        takes = list()
        for r in rows:
            found = set()
            for f in FOLDERS:
                keys = [file_key(f, x) for x in (r.get(f) or '').split(';') if x.strip()]
                hits = [where[(day, *k)] for k in keys if (day, *k) in where]
                listed = r.get(f'{f}_sec')
                if hits and listed not in (None, '') and abs(sum(d for _, d in hits) - float(listed)) > 2.0:
                    continue                       # another video of the same name
                found |= {k for k, _ in hits}
            for k in found:
                (take_of if r['type'] == 'take' else clip_of)[k].add(name)
            if r['type'] == 'take':
                takes.append(found)
        session_takes[name] = takes
    return take_of, clip_of, session_takes, session_day


def mcap_info(path: str, size: int) -> dict:
    """
    An episode MCAP's duration (steps / fps), good (a pose every step) and end time (its last
    message, the episode's last step; the video starts up to 1 s before the first step).
    """
    with S3File(path, size) as stream:
        reader = make_reader(stream)
        recording = {m.name: m.metadata for m in reader.iter_metadata()}.get('recording', {})
        end = reader.get_summary().statistics.message_end_time / 1e9
    steps, fps = int(recording['episode_length']), float(recording['video_fps'])
    missing = [int(v) for k, v in recording.items() if k.endswith('_pose_missing_steps')]
    duration = steps / fps
    return {'duration': duration, 'end': end, 'start': end - duration, 'good': all(m == 0 for m in missing)}


def place_episodes(takes: list, episodes: list, tolerance: float = 2.0):
    """
    The clock offset (s, added to the episodes' times) that puts the most episodes inside a take,
    searched over the take-start minus episode-start differences within +-14 h (of equally good
    offsets the middle one). Returns (offset, per episode the take index or None).
    """
    scored = list()
    candidates = {round(t0 - e['start'], 1) for t0, _ in takes for e in episodes}
    candidates |= {round(t1 - e['end'], 1) for _, t1 in takes for e in episodes}
    for d in sorted(candidates):
        if abs(d) > 14 * 3600:
            continue
        fit = list()
        for e in episodes:
            s, t = e['start'] + d, e['end'] + d
            fit.append(next((k for k, (t0, t1) in enumerate(takes) if s >= t0 - tolerance and t <= t1 + tolerance), None))
        scored.append((sum(f is not None for f in fit), d, fit))
    if not scored:
        return 0.0, [None] * len(episodes)
    most = max(n for n, _, _ in scored)
    # several offsets fit as many episodes (slack at the take edges): the middle one
    best = [x for x in scored if x[0] == most]
    _, best, best_fit = best[len(best) // 2]
    return best, best_fit


def take_gaps(take: tuple, intervals: list, gap_sec: float):
    """
    (seconds in episodes, seconds of gaps up to gap_sec, seconds of longer gaps) of a take.
    """
    t0, t1 = take
    covered, short, long_, cursor = 0.0, 0.0, 0.0, t0
    for s, e in sorted(intervals):
        s, e = max(s, t0), min(e, t1)
        if e <= cursor:
            continue
        gap = max(s - cursor, 0.0)
        short, long_ = (short + gap, long_) if gap <= gap_sec else (short, long_ + gap)
        covered += e - max(s, cursor)
        cursor = e
    gap = max(t1 - cursor, 0.0)
    short, long_ = (short + gap, long_) if gap <= gap_sec else (short, long_ + gap)
    return covered, short, long_


# %%
@click.command()
@click.option('--raw', required=True, help='s3:// folder with the stations (UMI_Collection)')
@click.option('--mcap', required=True, help='s3:// folder with the episode MCAP files (<station>/.../<task>/)')
@click.option('-o', '--output', required=True, help='Output prefix: <output>_days.csv, <output>_tasks.csv')
@click.option('--stations', default=None, help='Comma separated, default all stationN folders')
@click.option('--gap_sec', type=float, default=30.0, help='Gaps up to this long between episodes: docking / idle')
@click.option('--take_sec', type=float, default=60.0, help='Shorter recordings are clips (calibration, mapping)')
@click.option('--notes', default=None, help='CSV station,session,note')
@click.option('-j', '--jobs', type=int, default=16, help='MCAP files read in parallel')
def main(raw, mcap, output, stations, gap_sec, take_sec, notes, jobs):
    raw, mcap = raw.rstrip('/'), mcap.rstrip('/')
    output = os.path.expanduser(output)
    cache = pathlib.Path(f"{output}_inventory")
    if stations:
        stations = stations.split(',')
    else:
        prefix = split_s3(raw)[1] + '/'
        out = subprocess.run(['aws', 's3', 'ls', raw + '/'], check=True, capture_output=True, text=True).stdout
        stations = sorted(l.split()[-1].rstrip('/') for l in out.splitlines() if STATION.fullmatch(l.split()[-1].rstrip('/')))
    note_of = dict()
    if notes:
        for r in csv.DictReader(open(os.path.expanduser(notes))):
            note_of[(r['station'], r['session'])] = r['note']
    print(f"stations: {', '.join(stations)}", flush=True)
    mcaps = collections.defaultdict(list)
    for path, size, folder in list_mcaps(mcap):
        parts = folder.split('/')
        if parts and parts[0] in stations:
            mcaps[(parts[0], '/'.join(parts[1:]))].append((path, size))
    print(f"{sum(len(v) for v in mcaps.values())} MCAP files in {len(mcaps)} folders; reading their metadata", flush=True)
    keys = [(k, p, n) for k, files in mcaps.items() for p, n in files]
    infos = [None] * len(keys)
    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        futures = {pool.submit(mcap_info, p, n): i for i, (_, p, n) in enumerate(keys)}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            infos[futures[future]] = future.result()
            if done % 100 == 0 or done == len(keys):
                print(f"  {done}/{len(keys)} MCAP files read", flush=True)
    episodes = collections.defaultdict(list)
    for (k, p, _), info in zip(keys, infos):
        episodes[k].append({**info, 'file': p.rsplit('/', 1)[-1]})

    day_rows, task_rows = list(), list()
    h = lambda s: round(s / 3600, 3)
    for station in stations:
        days = day_folders(raw, station)
        session_rows = sessions(raw, station)
        print(f"\n{station}: day folders {', '.join(days) or '-'}; {len(session_rows)} sessions", flush=True)
        # recordings of every day folder, by file
        recordings, where = list(), dict()           # recordings: (day, group); where: (day, folder, name) -> (recording, s)
        for day in days:
            print(f"  {day}: video headers" + (' (cached)' if cache.joinpath(f"{station}_{day.replace('/', '_')}.json").is_file()
                else ''), flush=True)
            videos = inventory(raw, station, day, cache)
            for group in group_recordings(videos):
                for v in group:
                    where[(day, v['folder'], v['file'].split('/')[-1])] = (len(recordings), v['duration'])
                recordings.append((day, group))
        take_of, clip_of, session_takes, session_day = match_sessions(session_rows, where, days)

        per_day = collections.defaultdict(collections.Counter)
        for k, (day, group) in enumerate(recordings):
            start, end = span(group)
            length = (end - start).total_seconds()
            c = per_day[day]
            c['recorded'] += length
            missing = [f for f in FOLDERS if f not in {v['folder'] for v in group}]
            if k in take_of:
                shared = shared_seconds(group)
                c['in_tasks'] += shared
                c['cameras_not_all_recording'] += length - shared
            elif missing:
                c['missing_camera'] += length
            elif k in clip_of:
                c['calibration_mapping'] += length
            elif length < take_sec:
                c['short_clip_unused'] += length
            else:
                c['take_outside_tasks'] += length
        for day in days:
            c = per_day[day]
            day_rows.append({'station': station, 'day': day, **{x: h(c[x]) for x in ('recorded', 'in_tasks',
                'cameras_not_all_recording', 'missing_camera', 'calibration_mapping', 'short_clip_unused',
                'take_outside_tasks')}})

        for name in sorted(session_rows):
            takes = [found for found in session_takes[name] if found]
            groups = [[v for k in found for v in recordings[k][1]] for found in takes]
            day = session_day[name]
            intervals = list()
            for g in groups:
                if all(any(v['folder'] == f for v in g) for f in FOLDERS):
                    s, e = shared_interval(g)
                    intervals.append((s.timestamp(), e.timestamp()))
            in_tasks = sum(max(e - s, 0) for s, e in intervals)
            # the session's MCAP folder: <date>/<task>, the date from the session's path or, without one
            # (sorted before the dated layout), from the day folder its files are in
            date = re.search(r'\d{4}-\d{2}-\d{2}', name) or re.search(r'\d{4}-\d{2}-\d{2}', day)
            task = name.rstrip('/').split('/')[-1]
            folders = ([f"{date.group()}/{task}"] if date else []) + [task]
            best = None
            for folder in folders:
                eps_ = episodes.get((station, folder), [])
                offset_, fit_ = place_episodes(intervals, eps_) if eps_ and intervals else (0.0, [None] * len(eps_))
                if best is None or sum(f is not None for f in fit_) > sum(f is not None for f in best[2]):
                    best = (eps_, offset_, fit_, folder)
            eps, offset, fit, mcap_folder = best
            placed = [e for e, f in zip(eps, fit) if f is not None]
            covered, short, long_ = 0.0, 0.0, 0.0
            for t, take in enumerate(intervals if placed else []):
                a, b, c_ = take_gaps(take, [(e['start'] + offset, e['end'] + offset) for e, f in zip(eps, fit) if f == t],
                    gap_sec)
                covered, short, long_ = covered + a, short + b, long_ + c_
            status = 'processed' if placed else ('MCAP files of other recordings only' if eps else 'no MCAP files')
            task_rows.append({'station': station, 'session': name, 'day': day, 'mcap_folder': mcap_folder if eps else '',
                'takes': len(intervals),
                'in_tasks_h': h(in_tasks), 'mcap_episodes': len(placed), 'mcap_h': h(sum(e['duration'] for e in placed)),
                'mcap_good_h': h(sum(e['duration'] for e in placed if e['good'])),
                'between_episodes_h': h(short), 'long_gaps_h': h(long_ if placed else 0.0),
                'not_processed_h': h(0.0 if placed else in_tasks),
                'other_mcaps': len(eps) - len(placed), 'clock_offset_h': round(offset / 3600, 2) if placed else '',
                'status': status, 'note': note_of.get((station, name), '')})
            print(f"  {name}: {len(intervals)} takes {in_tasks / 3600:.2f} h; {len(placed)} episodes "
                f"{sum(e['duration'] for e in placed) / 3600:.2f} h" + (f"; {len(eps) - len(placed)} MCAP files "
                f"not from these takes" if len(eps) > len(placed) else '') + f" ({status})", flush=True)

    pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
    for suffix, rows in (('days', day_rows), ('tasks', task_rows)):
        if rows:
            with open(f"{output}_{suffix}.csv", 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
                w.writeheader()
                w.writerows(rows)

    print("\n============= yield =============")
    total = collections.Counter()
    for station in stations:
        d = collections.Counter()
        for r in day_rows:
            if r['station'] == station:
                d.update({k: v for k, v in r.items() if isinstance(v, float)})
        t = collections.Counter()
        for r in task_rows:
            if r['station'] == station:
                t.update({k: v for k, v in r.items() if isinstance(v, (int, float)) and k.endswith('_h')})
        total.update(d)
        total.update(t)
        print(f"{station:<9} recorded {d['recorded']:.1f} h -> in tasks {d['in_tasks']:.1f} h "
            f"(missing camera {d['missing_camera']:.1f}, calibration / mapping {d['calibration_mapping']:.1f}, "
            f"other short clips {d['short_clip_unused']:.1f}, takes outside tasks {d['take_outside_tasks']:.1f}, "
            f"cameras not all recording {d['cameras_not_all_recording']:.1f})\n"
            f"{'':<9} -> processed {t['in_tasks_h'] - t['not_processed_h']:.1f} h (not processed {t['not_processed_h']:.1f}) "
            f"-> MCAP episodes {t['mcap_h']:.1f} h, {t['mcap_good_h']:.1f} h without gaps "
            f"(between episodes up to {gap_sec:.0f} s {t['between_episodes_h']:.1f}, longer gaps {t['long_gaps_h']:.1f})")
    print(f"{'total':<9} recorded {total['recorded']:.1f} h -> in tasks {total['in_tasks']:.1f} h -> processed "
        f"{total['in_tasks_h'] - total['not_processed_h']:.1f} h -> MCAP episodes {total['mcap_h']:.1f} h "
        f"({total['mcap_good_h']:.1f} h without gaps)")
    print(f"saved {output}_days.csv, {output}_tasks.csv")

# %%
if __name__ == "__main__":
    main()
