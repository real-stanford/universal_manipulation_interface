"""
Quality checks of every episode (umi.tag_tracking.quality; Odyssey spec 1.4,
1.9, 2.10) and its validation_status: pass / warn / fail, the worst of its
checks. Writes tag_tracking/quality.json; the delivery export takes the
validation_status from it.

Media checks run once per video (resolution, frame rate on the camera clock,
frame timing, skipped frames, keyframe interval, encoding) and count for every
episode that uses the video. Image statistics need one decode of every frame
of every video (cached next to the video as image_stats.npz / .json, so later
runs only read them); -j videos are decoded in parallel.
Episodes: those with a task interval (03t_trim_docking.py); annotations from
tag_tracking/annotations (08 / 09) when present.

python scripts_tag_tracking/11_check_quality.py -i data_workspace/session -c umi_calib/tag_tracking.yaml
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import concurrent.futures
import datetime
import json
import pathlib
import pickle
import click
import numpy as np
from umi.common.image_stats import cached_image_stats
from umi.common.gpmf_util import frame_timing
from umi.tag_tracking.config import load_tag_tracking_config
from umi.tag_tracking.session import get_session_videos, overlapping
from umi.tag_tracking.wrist_sync import wrist_corrections, head_readout, head_to_wrist_time
from umi.tag_tracking.delivery import episode_frames, gripper_streams
from umi.tag_tracking.annotation import gripper_events
from umi.tag_tracking import quality


def video_inputs(path):
    data = cached_image_stats(path)
    return {**data, 'timing': frame_timing(str(path))}


def stats_slice(stats, t0, t1):
    keep = (stats['time'] >= t0) & (stats['time'] < t1)
    return {k: v[keep] for k, v in stats.items()}


def short(c):
    value = c['value'] if not isinstance(c['value'], dict) else ', '.join(f"{k} {v}" for k, v in c['value'].items())
    return f"{c['check']}({c['scope']})={value}"


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Session directory (trim_docking done)')
@click.option('-c', '--config', required=True, help='Tag tracking config yaml')
@click.option('-j', '--jobs', type=int, default=8, help='Videos decoded in parallel for the image statistics')
def main(input, config, jobs):
    session = pathlib.Path(os.path.expanduser(input)).absolute()
    config = load_tag_tracking_config(config)
    tcp = json.load(session.joinpath('tag_tracking', 'gripper_tcp.json').open('r'))
    episodes_json = json.load(session.joinpath('tag_tracking', 'episodes.json').open('r'))['videos']
    sync_path = session.joinpath('tag_tracking', 'wrist_sync.json')
    sync = json.load(sync_path.open('r')) if sync_path.is_file() else None
    annotations = session.joinpath('tag_tracking', 'annotations')
    videos = get_session_videos(session)
    heads = [x for x in videos if x['camera_serial'] == config['head_camera']]

    # media and image statistics per video
    print(f"image statistics of {len(videos)} videos ({jobs} in parallel, cached after the first run)")
    inputs = dict()
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(video_inputs, v['dir'].joinpath('raw_video.mp4')): v for v in videos}
        for f in concurrent.futures.as_completed(futures):
            inputs[futures[f]['name']] = f.result()
    media = dict()
    for v in videos:
        x = inputs[v['name']]
        role = 'head' if v['camera_serial'] == config['head_camera'] else 'wrist'
        media[v['name']] = quality.media_checks(x['facts'], x['timing'], x['stats']['time'], x['stats']['keyframe'], role)
        bad = [c for c in media[v['name']] if c['status'] in ('warn', 'fail')]
        print(f"{v['name']} ({role}): {quality.worst(media[v['name']])}"
            + (': ' + '; '.join(f"{short(c)} {c['detail']}" for c in bad) if bad else ''))

    result = {'parameters': quality.DEFAULTS, 'videos': media, 'episodes': dict(),
        'created_ts_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    counts = {'pass': 0, 'warn': 0, 'fail': 0}
    # head videos 00b_segment_episodes.py skipped (no wrist video overlapping) have no episodes
    for head in [x for x in heads if x['name'] in episodes_json]:
        track = pickle.load(head['dir'].joinpath('tag_tracking.pkl').open('rb'))
        corrections = wrist_corrections(sync, head['name'])
        grippers = gripper_streams(session, config, videos, head, track, tcp, corrections)
        readout = head_readout(head)
        fps = float(head['fps'])
        sync_videos = (sync or {'videos': {}})['videos'].get(head['name'], dict())
        result['episodes'][head['name']] = list()
        print(f"\n{head['name']}")
        for k, e in enumerate(episodes_json[head['name']]['episodes']):
            if e.get('task_start') is None:
                continue
            start, end = e['start'], e['end']
            checks = list()
            out = episode_frames(track['time'], track['head']['T_world_cam'], track['head']['is_valid'], start, end,
                grippers, row_time_offset=readout / 2)
            f = out['frames']
            expected = int(np.sum((track['time'] >= start) & (track['time'] < end)))
            checks += quality.frame_table_checks(f, expected, fps, [g['side'] for g in grippers])
            task = (f['time'] >= e['task_start']) & (f['time'] < e['task_end'])
            checks += quality.length_checks(end - start, e['task_end'] - e['task_start'])
            # cameras: media of the videos used, images of the episode's frames
            used = [(head, 'head', start, end)]
            for g in grippers:
                for wrist in overlapping(videos, head, config['bodies'][g['name']]['gripper_camera']):
                    w0, w1 = (float(head_to_wrist_time(t, head, wrist, corrections)) for t in (start, end))
                    if w1 > 0 and w0 < wrist['end_timestamp'] - wrist['start_timestamp']:
                        used.append((wrist, f"{g['side']}_wrist", w0, w1))
            for video, scope, t0, t1 in used:
                for c in media[video['name']]:
                    if c['status'] in ('warn', 'fail'):
                        checks.append({**c, 'scope': scope})
                checks += quality.image_checks(stats_slice(inputs[video['name']]['stats'], t0, t1), fps, scope)
            for g in grippers:
                side = g['side']
                if not any(s == f"{side}_wrist" for _, s, _, _ in used):
                    checks.append(quality.record('wrist_video', side, 'fail', None, None, 'no wrist video in this episode'))
                pose = f[f'ee_pose_{side}'][task]
                width = f[f'gripper_{side}'][task]
                checks += quality.tracking_checks(~np.isnan(pose[:, 0]), fps, side)
                closes = [x for x in gripper_events(g['width_times'], g['widths'], e['task_start'], e['task_end'])
                    if x['kind'] == 'close']
                r = g['gripper_range']
                checks += quality.gripper_checks(width, r['max_width'] - r['min_width'], len(closes), side)
                checks += quality.motion_checks(pose, width, fps, side)
                seen = track['bodies'][g['name']]['is_valid'][f['head_frame_index'][task]]
                checks.append(quality.visibility_check(seen, side))
                wrist = [v for v, s, t0, _ in used if s == f"{side}_wrist" and t0 >= 0]
                entry = sync_videos.get(wrist[0]['name']) if wrist else None
                checks.append(quality.sync_check(entry['line'] if entry else None,
                    entry['accuracy_ms'] if entry else None, start, side))
            checks.append(quality.static_check([c['value'] for c in checks if c['check'] == 'tcp_path']))
            if not any(c['value'] for c in checks if c['check'] == 'gripper_closes'):
                checks.append(quality.record('grasp_events', 'episode', 'warn', 0, None, 'no gripper close in the task'))
            path = annotations.joinpath(f"{head['name']}_ep{k + 1:03d}.json")
            checks += quality.timeline_checks(json.load(path.open('r')) if path.is_file() else None, end - start)

            status = quality.worst(checks)
            counts[status] += 1
            result['episodes'][head['name']].append({'episode': k + 1, 'start': start, 'end': end,
                'validation_status': status, 'checks': checks})
            bad = [c for c in checks if c['status'] in ('warn', 'fail')]
            print(f"  episode {k + 1:3d} {status:4s}" + (': ' + '; '.join(
                ('FAIL ' if c['status'] == 'fail' else '') + short(c) for c in bad) if bad else ''))

    out_path = session.joinpath('tag_tracking', 'quality.json')
    json.dump(result, out_path.open('w'), indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
    print(f"\n{sum(counts.values())} episodes: {counts['pass']} pass, {counts['warn']} warn, {counts['fail']} fail")
    print(f"Saved to {out_path}")

# %%
if __name__ == "__main__":
    main()
