"""
scripts_tag_tracking/14_export_rhoda.py on a generated episode MCAP laid out as 07_export_mcap.py
writes it: three cameras (frame numbers drawn into the images), each starting at its keyframe before
the episode, gripper width every step, a gap in the right gripper's poses.
Checks: three hvc1 mp4s with every frame, the frame at each file's sync frame index is the episode's
first frame, frames.csv rows (frame indices, widths, pose_valid, empty TCP in the gap), meta.json,
and the mosaic (H.264, n frames, starting at the sync frames).

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
        'stream=codec_name,codec_tag_string,nb_read_frames,width,height,pix_fmt', '-of', 'json', str(path)],
        check=True, capture_output=True, text=True).stdout
    return json.loads(out)['streams'][0]


def make_mcap(tmp):
    """
    Episode of N steps at 30 fps from t = 10 s. Camera c: video frames from its keyframe before the
    episode start frame start[c]; frame index i of the source video shows the number i.
    """
    src = os.path.join(tmp, 'src.mp4')
    make_video(src)
    starts = {0: 45, 1: 44, 2: 47}           # camera0 right wrist, camera1 left wrist, camera2 head
    path = os.path.join(tmp, 'episode_000012.mcap')
    dt = 1 / FPS
    ns = lambda t: int(round(t * 1e9))
    with open(path, 'wb') as f:
        w = Writer(f)
        w.start()
        s = w.register_schema('x', 'jsonschema', b'{}')
        for c, start in starts.items():
            ch = w.register_channel(f'/cameras/camera{c}/video', 'json', s)
            for index, data in export.video_messages(src, start, start + N):
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
            'camera1': 'robot1_wrist', 'camera2': 'head', 'world_frame': 'tag32', 'video_start': 'keyframe'})
        w.add_metadata('sync', {'method': 'audio + motion', 'camera0_offset_ms': '-3.2', 'camera0_sync_sigma_ms': '0.4'})
        w.add_metadata('camera2', {'role': 'head', 'model': 'GoPro HERO12 Black', 'fx': '1200.0'})
        w.finish()
    return path, starts


def test_export_mcap():
    with tempfile.TemporaryDirectory() as tmp:
        path, starts = make_mcap(tmp)
        out = pathlib.Path(tmp, 'out')
        rec = rhoda.recording_name('station3/2026-11-21/task01', path)
        assert rec == 'station3_2026-11-21_task01_episode_000012'
        r = rhoda.export_mcap(path, rec, out)
        expected = {'right': 15, 'left': 14, 'head': 17}   # start - keyframe before it (keyframes every 30)
        assert r['sync_index'] == expected, r['sync_index']
        folder = out.joinpath(rec)
        assert sorted(os.listdir(folder)) == sorted(f"{rec}_{x}" for x in
            ['head.mp4', 'left.mp4', 'right.mp4', 'meta.json', 'right_gripper.csv', 'left_gripper.csv'])
        for role, cam in [('right', 0), ('left', 1), ('head', 2)]:
            p = probe(folder.joinpath(f"{rec}_{role}.mp4"))
            assert p['codec_tag_string'] == 'hvc1' and int(p['nb_read_frames']) == expected[role] + N, (role, p)
            with av.open(str(folder.joinpath(f"{rec}_{role}.mp4"))) as c:
                frames = [f.to_ndarray(format='gray') for f in c.decode(video=0)]
            assert read_number(frames[expected[role]]) == starts[cam], role      # the sync frame
            assert read_number(frames[expected[role] + 7]) == starts[cam] + 7, role
        # gripper width per frame of its own video: one row per frame, empty before the sync frame
        for side, width, n_lead in [('right', '0.05000', 15), ('left', '0.02000', 14)]:
            rows = list(csv.reader(open(folder.joinpath(f"{rec}_{side}_gripper.csv"))))
            assert rows[0] == ['frame', 'width_m'] and len(rows) == 1 + n_lead + N, (side, len(rows))
            assert rows[1] == ['0', ''] and rows[n_lead] == [str(n_lead - 1), '']
            assert rows[1 + n_lead] == [str(n_lead), width] and rows[-1] == [str(n_lead + N - 1), width]
        meta = json.load(open(folder.joinpath(f"{rec}_meta.json")))
        assert sorted(meta) == ['cameras', 'files', 'gripper_width', 'sync'], sorted(meta)
        assert meta['sync']['frame_index'] == {'head': 17, 'left': 14, 'right': 15}
        assert meta['sync']['method'] == 'audio + motion'
        assert meta['cameras'] == {'head': {'model': 'GoPro HERO12 Black', 'fx': '1200.0'}}
        assert meta['files'] == {r: f"{rec}_{r}.mp4" for r in ('head', 'left', 'right')}

        mosaic = folder.joinpath(f"{rec}_mosaic.mp4")
        rhoda.mosaic(*((folder.joinpath(r['files'][k]), r['sync_index'][k]) for k in ('head', 'right', 'left')),
            N, mosaic, fps=FPS, width=320, source_fps=FPS)
        p = probe(mosaic)
        assert p['codec_name'] == 'h264' and p['pix_fmt'] == 'yuv420p' and int(p['nb_read_frames']) == N, p
        with av.open(str(mosaic)) as c:
            first = next(c.decode(video=0)).to_ndarray(format='gray')
        assert read_number(cv2.resize(first[:240, :320], (320, 240))) == starts[2]          # head at its sync frame
        assert read_number(cv2.resize(first[240:, :160], (320, 240))) == starts[0]          # right wrist
        assert read_number(cv2.resize(first[240:, 160:], (320, 240))) == starts[1]          # left wrist


if __name__ == '__main__':
    test_export_mcap()
    print('all passed')
