"""
scripts_tag_tracking/15_export_episode_videos.py on a generated session: three GoPro-like videos
(frame numbers drawn into the images, a keyframe every 30 frames) and a dataset_plan.pkl whose
episode starts in each camera at a different frame (one on a keyframe). Checks: each mp4 is exactly
the plan's frames (frame i = source frame start + i, equal counts), re-encoded start only where the
start is not a keyframe (hev1 / hvc1), gripper CSVs from the plan widths, meta.json, mosaic, the
output check, a plan with unequal frame ranges failing without leaving files, report.json, and the
S3 path (aws faked: uploaded with meta.json last, nothing left in the work folder).

python tests/test_episode_videos.py
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts_tag_tracking'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import csv
import importlib
import json
import pathlib
import shutil
import subprocess
import tempfile
import av
import numpy as np
from test_mcap_exact_start import make_video, read_number, FPS
from test_rhoda_export import probe

videos = importlib.import_module('15_export_episode_videos')
rhoda = importlib.import_module('14_export_rhoda')
N = 40
STARTS = {'right': 45, 'left': 60, 'head': 47}     # left starts on a keyframe
RECORD = {'model': 'GoPro HERO12 Black', 'frame_rate_hz': f"{FPS:.5f}", 'fx': '1200.0', 'serial': 'x'}


def make_session(tmp):
    demos = pathlib.Path(tmp, 'session', 'demos')
    cameras = list()
    for role in ('right', 'left', 'head'):
        d = demos.joinpath(f'demo_{role}')
        d.mkdir(parents=True)
        make_video(str(d.joinpath('raw_video.mp4')))
        cameras.append({'video_path': f'demo_{role}/raw_video.mp4', 'video_start_end': (STARTS[role], STARTS[role] + N),
            'clock_correction_ms': 0.0 if role == 'head' else -3.5})
    episode = {'episode_timestamps': np.arange(N) / FPS,
        'grippers': [{'gripper_width': np.full(N, 0.05)}, {'gripper_width': np.linspace(0.08, 0.01, N)}],
        'cameras': cameras}
    records = {c['video_path']: RECORD for c in cameras}
    return demos, episode, records


def test_local():
    with tempfile.TemporaryDirectory() as tmp:
        demos, episode, records = make_session(tmp)
        out = pathlib.Path(tmp, 'out')
        e = videos.export_episode(3, episode, demos, 'station3/2026-11-22/task01', str(out), out, records,
            True, 'libx264', 2, False)
        assert e == {'task': 'station3/2026-11-22/task01', 'episode': 'episode_000003', 'status': 'ok', 'frames': N,
            'mosaic': True}, e
        rec = '2026-11-22_task01_episode_000003'
        folder = out.joinpath('station3', '2026-11-22', 'task01', 'episode_000003')
        assert sorted(os.listdir(folder)) == sorted(f"{rec}_{x}" for x in ['head.mp4', 'left.mp4', 'right.mp4',
            'meta.json', 'right_gripper.csv', 'left_gripper.csv', 'mosaic.mp4'])
        for role, start in STARTS.items():
            p = probe(folder.joinpath(f"{rec}_{role}.mp4"))
            assert int(p['nb_read_frames']) == N, (role, p)
            assert p['codec_tag_string'] == ('hvc1' if role == 'left' else 'hev1'), (role, p)  # re-encoded start: hev1
            with av.open(str(folder.joinpath(f"{rec}_{role}.mp4"))) as c:
                frames = [f.to_ndarray(format='gray') for f in c.decode(video=0)]
            for i in [0, 1, 12, 13, 14, N - 1]:          # re-encoded start, across the next keyframe, the end
                assert read_number(frames[i]) == start + i, (role, i)
        rows = list(csv.reader(open(folder.joinpath(f"{rec}_left_gripper.csv"))))
        assert rows[0] == ['frame', 'width_m'] and len(rows) == N + 1 and rows[1] == ['0', '0.08000'] \
            and rows[-1] == [str(N - 1), '0.01000'], rows[:2]
        meta = json.load(open(folder.joinpath(f"{rec}_meta.json")))
        assert meta['sync'] == {'method': 'audio cross-correlation, wrist motion refinement',
            'event': "the episode's first step: frame 0 of every file (all files start at it, same frame count)",
            'frame_index': {'head': 0, 'left': 0, 'right': 0}}, meta['sync']
        assert meta['cameras']['head'] == {'model': 'GoPro HERO12 Black', 'frame_rate_hz': f"{FPS:.5f}", 'fx': '1200.0'}
        assert int(probe(folder.joinpath(f"{rec}_mosaic.mp4"))['nb_read_frames']) == N
        # again: skipped
        assert videos.export_episode(3, episode, demos, 'station3/2026-11-22/task01', str(out), out, records,
            False, 'libx264', 2, False)['status'] == 'skipped'

        # unequal frame ranges in the plan: failed, nothing written
        bad = {**episode, 'cameras': [dict(episode['cameras'][0], video_start_end=(45, 45 + N - 1))] + episode['cameras'][1:]}
        e = videos.export_episode(4, bad, demos, 'station3/2026-11-22/task01', str(out), out, records,
            False, 'libx264', 2, False)
        assert e['status'] == 'failed' and 'frame counts differ' in e['detail'], e
        assert not out.joinpath('station3', '2026-11-22', 'task01', 'episode_000004').exists()

        rhoda.write_reports([dict(e, task='station3/2026-11-22/task01'),
            {'task': 'station3/2026-11-22/task01', 'episode': 'episode_000003', 'status': 'ok', 'frames': N, 'mosaic': True}],
            'session', str(out), out)
        report = json.load(open(out.joinpath('station3', '2026-11-22', 'task01', 'report.json')))
        assert report['summary'] == {'total': 2, 'ok': 1, 'failed': 1, 'skipped': 0}, report


def test_s3():
    with tempfile.TemporaryDirectory() as tmp:
        demos, episode, records = make_session(tmp)
        bucket, uploads = dict(), list()
        real_run = subprocess.run

        def fake_run(cmd, *args, **kwargs):
            if cmd[:3] == ['aws', 's3', 'cp'] and not cmd[3].startswith('s3://'):
                keep = pathlib.Path(tmp, 'bucket', cmd[4][5:])
                keep.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(cmd[3], keep)
                bucket[cmd[4]] = str(keep)
                uploads.append(cmd[4])
                return subprocess.CompletedProcess(cmd, 0, '', '')
            return real_run(cmd, *args, **kwargs)
        rhoda.subprocess.run = fake_run
        real_size = rhoda.s3_size
        rhoda.s3_size = lambda url: os.path.getsize(bucket[url]) if url in bucket else None
        try:
            work = pathlib.Path(tmp, 'work')
            e = videos.export_episode(0, episode, demos, 'station3/2026-11-22/task01', 's3://b/mp4', work, records,
                False, 'libx264', 2, False)
        finally:
            rhoda.subprocess.run = real_run
            rhoda.s3_size = real_size
        assert e['status'] == 'ok', e
        assert len(uploads) == 6 and uploads[-1] == \
            's3://b/mp4/station3/2026-11-22/task01/episode_000000/2026-11-22_task01_episode_000000_meta.json', uploads
        assert not [p for p in work.rglob('*') if p.is_file()], list(work.rglob('*'))


if __name__ == '__main__':
    test_local()
    test_s3()
    print('all passed')
