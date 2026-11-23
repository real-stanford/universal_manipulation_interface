"""
scripts_tag_tracking/exact_start_mcaps.py on a generated keyframe-start episode MCAP (as 07_export_mcap.py
without --exact_start writes it: each camera's video from the keyframe before the episode, one frame
past its end). Checks: every camera video has one frame per step, starting at the episode's first frame
(re-encoded up to the next keyframe, a start on a keyframe copied as is), message times kept, other
topics unchanged, metadata video_start exact and no camera{j}_sync_frame (moved to source); an exact
MCAP is copied as is; 14_export_rhoda.py accepts the result.

python tests/test_exact_start_mcaps.py
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts_tag_tracking'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import base64
import collections
import filecmp
import importlib
import json
import pathlib
import tempfile
import av
from mcap.reader import make_reader
from mcap.writer import Writer
from test_mcap_exact_start import make_video, read_number, FPS

fix = importlib.import_module('exact_start_mcaps')
export = importlib.import_module('07_export_mcap')
rhoda = importlib.import_module('14_export_rhoda')
N = 50
STARTS = {0: 45, 1: 60, 2: 47}     # camera1 starts on a keyframe (keyframes every 30 frames)


def make_keyframe_mcap(tmp):
    src = os.path.join(tmp, 'src.mp4')
    make_video(src)
    path = os.path.join(tmp, 'episode_000003.mcap')
    ns = lambda t: int(round(t * 1e9))
    with open(path, 'wb') as f:
        w = Writer(f)
        w.start()
        s = w.register_schema('x', 'jsonschema', b'{}')
        for c, start in STARTS.items():
            ch = w.register_channel(f'/cameras/camera{c}/video', 'json', s)
            for index, data in export.video_messages(src, start, start + N + 1):   # lead-in, one frame past the end
                t = 10 + (index - start) / FPS
                w.add_message(ch, log_time=ns(t), publish_time=ns(t), data=json.dumps({'timestamp': {'sec': 0, 'nsec': 0},
                    'frame_id': f'camera{c}', 'format': 'h265', 'data': base64.b64encode(data).decode()}).encode())
        imu = w.register_channel('robot0/imu', 'json', s)
        for k in range(200):                 # IMU from before the video start
            t = ns(8 + k * 0.02)
            w.add_message(imu, log_time=t, publish_time=t, data=json.dumps({'k': k}).encode())
        for g in range(2):
            width = w.register_channel(f'robot{g}/gripper/state', 'json', s)
            for i in range(N):
                t = ns(10 + i / FPS)
                w.add_message(width, log_time=t, publish_time=t, data=json.dumps({'width': 0.01 * i}).encode())
        w.add_metadata('recording', {'episode_length': str(N), 'video_fps': str(FPS), 'camera0': 'robot0_wrist',
            'camera1': 'robot1_wrist', 'camera2': 'head'})
        w.add_metadata('sync', {'method': 'audio + motion', 'camera0_offset_ms': '-3.2',
            **{f'camera{c}_sync_frame': str(1000 + c) for c in STARTS}})
        w.add_metadata('camera2', {'role': 'head', 'model': 'GoPro HERO12 Black'})
        w.finish()
    return path


def read(path):
    out = collections.defaultdict(list)
    with open(path, 'rb') as f:
        reader = make_reader(f)
        metadata = {m.name: dict(m.metadata) for m in reader.iter_metadata()}
        for _, channel, message in reader.iter_messages():
            out[channel.topic].append((message.log_time, message.data))
    return out, metadata


def test_exact_start():
    with tempfile.TemporaryDirectory() as tmp:
        path = make_keyframe_mcap(tmp)
        before, _ = read(path)
        out = os.path.join(tmp, 'exact.mcap')
        summary = fix.exact_start(path, out)
        assert 'camera1 30 (keyframe)' in summary and 'camera0 15 (re-encoded' in summary, summary   # MCAP frame index
        after, metadata = read(out)
        steps = [t for t, _ in after['robot0/gripper/state']]
        for c, start in STARTS.items():
            video = after[f'/cameras/camera{c}/video']
            assert [t for t, _ in video] == steps, c          # one frame per step, at the step times
            dec = av.CodecContext.create('hevc', 'r')
            frames = list()
            for _, data in video:
                frames += dec.decode(av.Packet(base64.b64decode(json.loads(data)['data'])))
            frames += dec.decode(None)
            assert len(frames) == N, (c, len(frames))
            for i in [0, 1, 14, 15, 16, N - 1]:
                assert read_number(frames[i].to_ndarray(format='gray')) == start + i, (c, i)
            assert json.loads(video[0][1])['frame_id'] == f'camera{c}'
        for topic in ['robot0/imu', 'robot0/gripper/state', 'robot1/gripper/state']:
            assert after[topic] == before[topic], topic
        assert metadata['recording']['video_start'] == 'exact'
        assert metadata['sync'] == {'method': 'audio + motion', 'camera0_offset_ms': '-3.2'}, metadata['sync']
        assert metadata['source'] == {**{f'camera{c}_source_first_frame': str(1000 + c) for c in STARTS},
            **{f'camera{c}_source_end_frame': str(1000 + c + N) for c in STARTS}}, metadata['source']
        assert metadata['camera2'] == {'role': 'head', 'model': 'GoPro HERO12 Black'}

        # already exact: copied
        again = os.path.join(tmp, 'again.mcap')
        assert 'copied' in fix.exact_start(out, again) and filecmp.cmp(out, again, shallow=False)

        # the Rhoda export takes it: same frame count, sync frame index 0
        r = rhoda.export_mcap(out, 'rec', pathlib.Path(tmp, 'rhoda'))
        assert r['n_frames'] == N and r['sync_index'] == {'right': 0, 'left': 0, 'head': 0}, r


def test_process_local():
    with tempfile.TemporaryDirectory() as tmp:
        path = make_keyframe_mcap(tmp)
        out_dir = os.path.join(tmp, 'out')
        msg = fix.process(path, 'station1/task02', out_dir, pathlib.Path(tmp), False)
        target = os.path.join(out_dir, 'station1', 'task02', 'episode_000003.mcap')
        assert os.path.isfile(target) and 'frames per camera' in msg, msg
        assert 'skipped' in fix.process(path, 'station1/task02', out_dir, pathlib.Path(tmp), False)
        assert not [p for p in os.listdir(os.path.dirname(target)) if p.endswith('.part')]


if __name__ == '__main__':
    test_exact_start()
    test_process_local()
    print('all passed')
