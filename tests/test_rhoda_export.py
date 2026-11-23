"""
scripts_tag_tracking/14_export_rhoda.py on generated H.265 videos (one keyframe per second, the frame
number drawn into every frame): the copied clip starts at the keyframe before the episode, has every
frame to the end, is tagged hvc1, decodes, and the frame at the sync index is the episode's first
frame; the mosaic has one frame per output frame from the sync frames on.

python tests/test_rhoda_export.py
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts_tag_tracking'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests'))

import importlib
import json
import subprocess
import tempfile
import av
from test_mcap_exact_start import make_video, read_number, FPS

rhoda = importlib.import_module('14_export_rhoda')


def probe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_entries',
        'stream=codec_name,codec_tag_string,nb_read_frames,width,height,pix_fmt', '-of', 'json', path],
        check=True, capture_output=True, text=True).stdout
    return json.loads(out)['streams'][0]


def test_copy_clip():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, 'src.mp4')
        make_video(src)
        for start, stop in [(45, 100), (30, 70), (0, 40)]:
            first = rhoda.first_keyframe(src, start)
            assert first == (start // FPS) * FPS, (start, first)
            out = os.path.join(tmp, f'clip_{start}.mp4')
            n = rhoda.copy_clip(src, first, stop, out)
            assert n == stop - first
            p = probe(out)
            assert p['codec_name'] == 'hevc' and p['codec_tag_string'] == 'hvc1' and int(p['nb_read_frames']) == n, p
            with av.open(out) as c:
                frames = [f.to_ndarray(format='gray') for f in c.decode(video=0)]
            assert len(frames) == n
            assert read_number(frames[0]) == first and read_number(frames[start - first]) == start
            assert read_number(frames[-1]) == stop - 1


def test_mosaic():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, 'src.mp4')
        make_video(src)
        clips = list()
        for start, stop in [(45, 100), (40, 95), (50, 105)]:   # three cameras, different sync indices
            first = rhoda.first_keyframe(src, start)
            out = os.path.join(tmp, f'c{start}.mp4')
            rhoda.copy_clip(src, first, stop, out)
            clips.append((out, start - first))
        path = os.path.join(tmp, 'mosaic.mp4')
        rhoda.mosaic(clips[0], clips[1], clips[2], 55, path, fps=FPS, width=320)
        p = probe(path)
        assert p['codec_name'] == 'h264' and p['pix_fmt'] == 'yuv420p' and int(p['nb_read_frames']) == 55, p
        assert (int(p['width']), int(p['height'])) == (320, 240 + 120), p
        with av.open(path) as c:
            first = next(c.decode(video=0)).to_ndarray(format='gray')
        # head (top) shows its sync frame (45)
        import cv2
        head = cv2.resize(first[:240, :320], (320, 240))
        assert read_number(head) == 45


def test_export_episode():
    import csv
    import numpy as np
    with tempfile.TemporaryDirectory() as tmp:
        demos = os.path.join(tmp, 'demos')
        for name in ['right', 'left', 'head']:
            os.makedirs(os.path.join(demos, name))
            make_video(os.path.join(demos, name, 'raw_video.mp4'))
        n = 50
        pose = np.tile([0.1, 0.2, 0.3, 0.0, 0.0, 0.1], (n, 1))
        pose[10:12] = np.nan                                       # a gap in the right gripper
        episode = {'episode_timestamps': 100 + np.arange(n) / FPS,
            'grippers': [{'tcp_pose': pose, 'gripper_width': np.full(n, 0.05), 'pose_valid': ~np.isnan(pose[:, 0])},
                         {'tcp_pose': np.tile([0.4, 0.2, 0.3, 0.0, 0.0, 0.0], (n, 1)), 'gripper_width': np.full(n, 0.02),
                          'pose_valid': np.ones(n, bool)}],
            'cameras': [{'video_path': 'right/raw_video.mp4', 'video_start_end': (45, 45 + n), 'clock_correction_ms': -3.2},
                        {'video_path': 'left/raw_video.mp4', 'video_start_end': (44, 44 + n), 'clock_correction_ms': 1.5},
                        {'video_path': 'head/raw_video.mp4', 'video_start_end': (47, 47 + n)}]}
        out = os.path.join(tmp, 'out')
        import pathlib
        r = rhoda.export_episode(3, episode, pathlib.Path(demos), pathlib.Path(out), 'st_task', {'session': 'x'}, None, {0: 0.4})
        assert r['sync_index'] == {'right': 15, 'left': 14, 'head': 17}, r['sync_index']
        folder = os.path.join(out, 'st_task_ep0003')
        assert sorted(os.listdir(folder)) == ['st_task_ep0003_frames.csv', 'st_task_ep0003_head.mp4',
            'st_task_ep0003_left.mp4', 'st_task_ep0003_meta.json', 'st_task_ep0003_right.mp4']
        rows = list(csv.DictReader(open(os.path.join(folder, 'st_task_ep0003_frames.csv'))))
        assert len(rows) == n and rows[0]['head_frame'] == '17' and rows[0]['right_frame'] == '15' and rows[5]['left_frame'] == '19'
        assert rows[10]['right_pose_valid'] == '0' and rows[10]['right_tcp_x'] == '' and rows[10]['right_gripper_width_m'] == '0.05000'
        assert rows[0]['right_pose_valid'] == '1' and rows[0]['left_tcp_x'] == '0.40000'
        meta = json.load(open(os.path.join(folder, 'st_task_ep0003_meta.json')))
        assert meta['sync']['sync_frame_index'] == {'head': 17, 'left': 14, 'right': 15}
        assert meta['sync']['clock_correction_ms'] == {'right': -3.2, 'left': 1.5} and meta['sync']['sync_sigma_ms'] == {'right': 0.4}
        # the frame at each file's sync index is the camera's episode start frame
        with av.open(os.path.join(folder, 'st_task_ep0003_head.mp4')) as c:
            frames = [f.to_ndarray(format='gray') for f in c.decode(video=0)]
        assert read_number(frames[17]) == 47 and len(frames) == 17 + n


if __name__ == '__main__':
    test_export_episode()
    test_copy_clip()
    test_mosaic()
    print('all passed')
