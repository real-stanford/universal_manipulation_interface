"""
scripts_tag_tracking/14_export_rhoda.py on a generated episode MCAP laid out as 07_export_mcap.py
--exact_start writes it: three cameras (frame numbers drawn into the images), each starting exactly at
its frame of the episode's first step (07's video_messages_exact), gripper width every step, a gap in
the right gripper's poses. Checks: three mp4s with the same frame count, frame i = source frame
start + i in each, gripper CSVs one width per frame, meta.json (sync frame index 0 in every file), the
mosaic (H.264, n frames, the three views at frame 0), and the same from an MCAP without
--exact_start (keyframe start: the start re-encoded by the export).

python tests/test_rhoda_export.py
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts_tag_tracking'))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import base64
import csv
import importlib
import json
import pathlib
import subprocess
import tempfile
import av
import cv2
from mcap.writer import Writer
from test_mcap_exact_start import make_video, read_number, FPS

rhoda = importlib.import_module('14_export_rhoda')
export = importlib.import_module('07_export_mcap')
N = 50


def probe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_entries',
        'stream=codec_name,codec_tag_string,nb_read_frames,width,height,pix_fmt,r_frame_rate', '-of', 'json', str(path)],
        check=True, capture_output=True, text=True).stdout
    return json.loads(out)['streams'][0]


def make_mcap(tmp, exact=True, drop=None):
    """
    Episode of N steps at 30 fps from t = 10 s. Camera c: video frames from its keyframe before the
    episode start frame start[c]; frame index i of the source video shows the number i.
    """
    src = os.path.join(tmp, 'src.mp4')
    make_video(src)
    starts = {0: 45, 1: 44, 2: 47}           # camera0 right wrist, camera1 left wrist, camera2 head
    path = os.path.join(tmp, 'episode_000014.mcap' if drop else 'episode_000012.mcap' if exact else 'episode_000013.mcap')
    dt = 1 / FPS
    ns = lambda t: int(round(t * 1e9))
    with open(path, 'wb') as f:
        w = Writer(f)
        w.start()
        s = w.register_schema('x', 'jsonschema', b'{}')
        for c, start in starts.items():
            ch = w.register_channel(f'/cameras/camera{c}/video', 'json', s)
            messages = export.video_messages_exact if exact else export.video_messages
            for index, data in messages(src, start, start + N):
                if drop == (c, index - start):        # a dropped frame: camera c, episode frame k
                    continue
                t = 10 + (index - start) * dt       # step 0 at t = 10 for every camera
                w.add_message(ch, log_time=ns(t), publish_time=ns(t), data=json.dumps(
                    {'data': base64.b64encode(data).decode(), 'format': 'h265'}).encode())
        for g in range(2):
            width = w.register_channel(f'robot{g}/gripper/state', 'json', s)
            pose = w.register_channel(f'robot{g}/eef/pose', 'json', s)
            for i in range(N):
                t = ns(10 + i * dt)
                w.add_message(width, log_time=t, publish_time=t, data=json.dumps({'width': 0.05 - 0.03 * g}).encode())
                if not (g == 0 and 10 <= i < 12):     # right gripper: no pose in steps 10, 11
                    w.add_message(pose, log_time=t, publish_time=t, data=json.dumps(
                        {'pos': [0.1 + g, 0.2, 0.3], 'quat_xyzw': [0, 0, 0, 1]}).encode())
        w.add_metadata('recording', {'episode_length': str(N), 'video_fps': str(FPS), 'camera0': 'robot0_wrist',
            'camera1': 'robot1_wrist', 'camera2': 'head', 'world_frame': 'tag32', 'video_start': 'exact' if exact else 'keyframe'})
        w.add_metadata('sync', {'method': 'audio + motion', 'camera0_offset_ms': '-3.2', 'camera0_sync_sigma_ms': '0.4'})
        w.add_metadata('camera2', {'role': 'head', 'model': 'GoPro HERO12 Black', 'fx': '1200.0'})
        w.finish()
    return path, starts


def test_export_mcap():
    with tempfile.TemporaryDirectory() as tmp:
        path, starts = make_mcap(tmp)
        out = pathlib.Path(tmp, 'out')
        rec = rhoda.recording_name('station3/2026-11-21/task01', path)
        assert rec == '2026-11-21_task01_episode_000012'      # no station in the file names
        r = rhoda.export_mcap(path, rec, out)
        assert r['sync_index'] == {'right': 0, 'left': 0, 'head': 0}, r['sync_index']
        folder = out.joinpath(rec)
        assert sorted(os.listdir(folder)) == sorted(f"{rec}_{x}" for x in
            ['head.mp4', 'left.mp4', 'right.mp4', 'meta.json', 'right_gripper.csv', 'left_gripper.csv'])
        for role, cam in [('right', 0), ('left', 1), ('head', 2)]:
            p = probe(folder.joinpath(f"{rec}_{role}.mp4"))
            assert int(p['nb_read_frames']) == N, (role, p)
            with av.open(str(folder.joinpath(f"{rec}_{role}.mp4"))) as c:
                frames = [f.to_ndarray(format='gray') for f in c.decode(video=0)]
            for i in [0, 1, 14, 15, 16, N - 1]:     # 07's re-encoded start, across the next keyframe, the end
                assert read_number(frames[i]) == starts[cam] + i, (role, i)
        # gripper width per frame: one row per frame, no empty rows
        for side, width in [('right', '0.05000'), ('left', '0.02000')]:
            rows = list(csv.reader(open(folder.joinpath(f"{rec}_{side}_gripper.csv"))))
            assert rows[0] == ['frame', 'width_m'] and len(rows) == 1 + N, (side, len(rows))
            assert rows[1] == ['0', width] and rows[-1] == [str(N - 1), width]
        assert b'\r' not in folder.joinpath(f"{rec}_right_gripper.csv").read_bytes()
        meta = json.load(open(folder.joinpath(f"{rec}_meta.json")))
        assert sorted(meta) == ['cameras', 'files', 'gripper_width', 'sync'], sorted(meta)
        assert meta['sync']['frame_index'] == {'head': 0, 'left': 0, 'right': 0}
        assert meta['sync']['method'] == 'audio + motion'
        assert meta['cameras'] == {'head': {'model': 'GoPro HERO12 Black', 'fx': '1200.0'}}
        assert meta['files'] == {r: f"{rec}_{r}.mp4" for r in ('head', 'left', 'right')}

        mosaic = folder.joinpath(f"{rec}_mosaic.mp4")
        rhoda.mosaic(*((folder.joinpath(r['files'][k]), r['sync_index'][k]) for k in ('head', 'right', 'left')),
            N, mosaic, fps=FPS, width=320)
        p = probe(mosaic)
        assert p['codec_name'] == 'h264' and p['pix_fmt'] == 'yuv420p' and int(p['nb_read_frames']) == N, p
        assert p['r_frame_rate'] == f'{FPS}/1', p        # the cameras' frame rate
        with av.open(str(mosaic)) as c:
            first = next(c.decode(video=0)).to_ndarray(format='gray')
        assert read_number(cv2.resize(first[:240, :320], (320, 240))) == starts[2]          # head at its sync frame
        assert read_number(cv2.resize(first[240:, :160], (320, 240))) == starts[1]          # left wrist, bottom left
        assert read_number(cv2.resize(first[240:, 160:], (320, 240))) == starts[0]          # right wrist, bottom right

        # without --exact_start (keyframe start, lead-in per camera): cut here, same result
        path, _ = make_mcap(tmp, exact=False)
        r = rhoda.export_mcap(path, 'keyframe', out)
        assert r['sync_index'] == {'right': 0, 'left': 0, 'head': 0} and r['n_frames'] == N, r
        assert r['reencoded'] == {'right': True, 'left': True, 'head': True}, r['reencoded']
        for role, cam in [('right', 0), ('left', 1), ('head', 2)]:
            p = probe(out.joinpath('keyframe', f"keyframe_{role}.mp4"))
            assert int(p['nb_read_frames']) == N and p['codec_tag_string'] == 'hev1', (role, p)
            with av.open(str(out.joinpath('keyframe', f"keyframe_{role}.mp4"))) as c:
                frames = [f.to_ndarray(format='gray') for f in c.decode(video=0)]
            for i in [0, 1, 14, 15, 16, N - 1]:
                assert read_number(frames[i]) == starts[cam] + i, (role, i)

def test_jobs_and_s3():
    import shutil
    from click.testing import CliRunner
    with tempfile.TemporaryDirectory() as tmp:
        src = pathlib.Path(tmp, 'mcaps', 'station1', 'task02')
        src.mkdir(parents=True)
        path, starts = make_mcap(tmp)
        shutil.copy(path, src.joinpath('episode_000001.mcap'))
        shutil.copy(path, src.joinpath('episode_000002.mcap'))
        # two recordings in parallel, local output, a mosaic for one
        out = pathlib.Path(tmp, 'rhoda')
        result = CliRunner().invoke(rhoda.main, [str(pathlib.Path(tmp, 'mcaps')), '-o', str(out), '-j', '2',
            '--mosaic_files', 'episode_000002.mcap'])
        assert result.exit_code == 0 and 'done: 2 ok, 0 failed' in result.output, result.output
        report = json.load(open(out.joinpath('station1', 'task02', 'report.json')))
        assert report['summary'] == {'total': 2, 'ok': 2, 'failed': 0, 'skipped': 0}, report
        assert report['episodes'] == [{'episode': 'episode_000001', 'status': 'ok', 'frames': N, 'mosaic': False},
            {'episode': 'episode_000002', 'status': 'ok', 'frames': N, 'mosaic': True}], report['episodes']
        out = out.joinpath('station1', 'task02')      # the input's folder structure, a folder per episode
        assert sorted(os.listdir(out)) == ['episode_000001', 'episode_000002', 'report.json'], os.listdir(out)
        assert sorted(os.listdir(out.joinpath('episode_000002'))) == sorted(f"task02_episode_000002_{x}" for x in
            ['head.mp4', 'left.mp4', 'right.mp4', 'meta.json', 'right_gripper.csv', 'left_gripper.csv', 'mosaic.mp4'])
        assert len(os.listdir(out.joinpath('episode_000001'))) == 6
        assert int(probe(out.joinpath('episode_000002', 'task02_episode_000002_mosaic.mp4'))['nb_read_frames']) == N
        out = pathlib.Path(tmp, 'rhoda')
        # rerun with a dropped head frame in a new episode: it fails (frame_drop) and is not written,
        # the two done ones are skipped and keep their report entries
        shutil.copy(make_mcap(tmp, drop=(2, 20))[0], src.joinpath('episode_000003.mcap'))
        result = CliRunner().invoke(rhoda.main, [str(pathlib.Path(tmp, 'mcaps')), '-o', str(out), '-j', '2'])
        assert result.output.count('in the output, skipped') == 2 and 'FAILED frame_drop' in result.output, result.output
        assert not out.joinpath('station1', 'task02', 'episode_000003').exists()
        report = json.load(open(out.joinpath('station1', 'task02', 'report.json')))
        assert report['summary'] == {'total': 3, 'ok': 2, 'failed': 1, 'skipped': 0}, report
        failed = report['episodes'][2]
        assert failed['episode'] == 'episode_000003' and failed['status'] == 'failed' and failed['reason'] == 'frame_drop' \
            and 'camera2' in failed['detail'], failed
        os.remove(src.joinpath('episode_000003.mcap'))

        # a written file that does not pass the check: output_check
        bad = dict(rhoda.export_mcap(path, 'chk', pathlib.Path(tmp, 'chk')), sync_index={'right': 0, 'left': 0, 'head': 0})
        folder = pathlib.Path(tmp, 'chk', 'chk')
        rhoda.check_outputs(folder, 'chk', bad, False)
        folder.joinpath('chk_left_gripper.csv').write_text('frame,width_m\n0,\n')
        try:
            rhoda.check_outputs(folder, 'chk', bad, False)
            raise RuntimeError('check passed a broken CSV')
        except rhoda.OutputCheckError as e:
            assert 'chk_left_gripper.csv' in str(e) and rhoda.failure_reason('export', e) == 'output_check'

        # --mosaic 2 per task folder: first and last of each task (task03: 0, 2; task04: 0, 4), not the
        # same file names in every task (by name task04 would also get episode 2: 5 mosaics)
        for task, n_files in [('task03', 3), ('task04', 5)]:
            d = pathlib.Path(tmp, 'many', 'station1', task)
            d.mkdir(parents=True)
            for k in range(n_files):
                shutil.copy(path, d.joinpath(f'episode_{k:06d}.mcap'))
        many = pathlib.Path(tmp, 'many_out')
        result = CliRunner().invoke(rhoda.main, [str(pathlib.Path(tmp, 'many')), '-o', str(many), '-j', '4', '--mosaic', '2'])
        mosaics = sorted(p.name[:-len('_mosaic.mp4')] for p in many.rglob('*_mosaic.mp4'))
        assert sorted(os.listdir(many.joinpath('station1'))) == ['task03', 'task04']
        assert result.exit_code == 0 and mosaics == ['task03_episode_000000', 'task03_episode_000002',
            'task04_episode_000000', 'task04_episode_000004'], (mosaics, result.output)

        # S3 in and out (aws faked): downloaded into the workdir, uploaded with meta.json last, all deleted
        bucket, uploads = dict(), list()
        real_run = subprocess.run

        def fake_run(cmd, *args, **kwargs):
            if cmd[:3] == ['aws', 's3', 'cp']:
                a, b = cmd[3], cmd[4]
                if a.startswith('s3://'):
                    if a not in bucket:        # no such key: aws fails
                        return subprocess.CompletedProcess(cmd, 1, '', 'Not Found')
                    shutil.copy(bucket[a], b)
                else:
                    keep = pathlib.Path(tmp, 'bucket', b[5:])
                    keep.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(a, keep)
                    bucket[b] = str(keep)
                    uploads.append(b)
                return subprocess.CompletedProcess(cmd, 0, '', '')
            return real_run(cmd, *args, **kwargs)
        bucket['s3://in/station1/task02/episode_000001.mcap'] = path
        # the station / date / task folders from the stationN folder on, whatever folder is the input
        assert rhoda.task_folder('s3://b/umi_data/industrial_data/station2/2026-11-22/task01/episode_000004.mcap', '.') \
            == 'station2/2026-11-22/task01'
        assert rhoda.task_folder('/data/station3/task01/dataset_mcap/episode_000001.mcap', 'task01/dataset_mcap') \
            == 'station3/task01'
        assert rhoda.task_folder('/data/x/task01/episode_000001.mcap', 'task01') == 'task01'
        result = CliRunner().invoke(rhoda.main, [str(src), '-o', str(pathlib.Path(tmp, 'from_task'))])   # input: the task folder
        assert result.exit_code == 0 and pathlib.Path(tmp, 'from_task', 'station1', 'task02', 'episode_000001',
            'task02_episode_000001_head.mp4').is_file(), result.output
        rhoda.subprocess.run = fake_run
        rhoda.s3_size = lambda url: os.path.getsize(bucket[url]) if url in bucket else None
        try:
            work = pathlib.Path(tmp, 'work')
            work.mkdir()
            entry = rhoda.export_one('s3://in/station1/task02/episode_000001.mcap', 'station1/task02', 's3://out/rhoda',
                work, True, 'libx264', 2, False)
            assert entry == {'task': 'station1/task02', 'episode': 'episode_000001', 'status': 'ok', 'frames': N,
                'mosaic': True}, entry
            rhoda.write_reports([entry], 's3://in/station1', 's3://out/rhoda', work)
        finally:
            rhoda.subprocess.run = real_run
        report = json.load(open(bucket['s3://out/rhoda/station1/task02/report.json']))
        assert report['run']['output'] == 's3://out/rhoda/station1/task02' and report['summary']['ok'] == 1, report
        rec = 'task02_episode_000001'
        assert len(uploads) == 8 and uploads[6] == f's3://out/rhoda/station1/task02/episode_000001/{rec}_meta.json', uploads
        assert sorted(os.listdir(work)) == ['station1'] and not list(work.rglob('*.*')), list(work.rglob('*'))
        assert rhoda.export_one('s3://in/station1/task02/episode_000001.mcap', 'station1/task02',
            's3://out/rhoda', work, True, 'libx264', 2, False)['status'] == 'skipped'


if __name__ == '__main__':
    test_export_mcap()
    test_jobs_and_s3()
    print('all passed')
