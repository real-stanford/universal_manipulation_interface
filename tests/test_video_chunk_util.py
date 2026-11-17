# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import tempfile
from fractions import Fraction
import av
import cv2
import numpy as np
from umi.common.video_chunk_util import process_video_chunked
from umi.tag_tracking.tag_detection import detect_video_tag_corners, detect_tag_corners

N_FRAMES = 300
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def write_tag_video(path):
    """
    Synthetic GoPro-like video (59.94 fps, B-frames, keyframe every 30 frames),
    frame k shows ArUco tag k % 50 at a position moving with k.
    """
    with av.open(path, 'w') as container:
        stream = container.add_stream('libx264', rate=Fraction(60000, 1001))
        stream.width, stream.height, stream.pix_fmt = 640, 480, 'yuv420p'
        stream.options = {'g': '30', 'bf': '2'}
        for k in range(N_FRAMES):
            img = np.full((480, 640), 255, dtype=np.uint8)
            tag = cv2.aruco.generateImageMarker(DICT, k % 50, 160)
            x, y = 40 + (k * 3) % 400, 60 + (k * 2) % 240
            img[y:y + 160, x:x + 160] = tag
            frame = av.VideoFrame.from_ndarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def make_id_reader():
    def read(frame, time):
        return {'time': time, 'ids': sorted(detect_tag_corners(frame.to_ndarray(format='gray'), DICT))}
    return read


def sequential(path):
    out = list()
    with av.open(path) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            out.append({'time': float(frame.pts * stream.time_base),
                'ids': sorted(detect_tag_corners(frame.to_ndarray(format='gray'), DICT))})
    return out


def test_chunked_equals_sequential():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'tags.mp4')
        write_tag_video(path)
        reference = sequential(path)
        assert len(reference) == N_FRAMES
        # every frame decoded shows its own tag
        assert [x['ids'] for x in reference] == [[k % 50] for k in range(N_FRAMES)]
        # several pools in one process, as when a script processes several videos:
        # the second pool used to hang with forked workers
        for num_workers in [1, 3, 7, 16, 7]:
            chunked = process_video_chunked(path, make_id_reader, num_workers=num_workers)
            assert [x['frame_idx'] for x in chunked] == list(range(N_FRAMES))
            assert [x['time'] for x in chunked] == [x['time'] for x in reference], num_workers
            assert [x['ids'] for x in chunked] == [x['ids'] for x in reference], num_workers


def test_detect_video_tag_corners():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'tags.mp4')
        write_tag_video(path)
        frames = detect_video_tag_corners(path, DICT, tag_ids={0, 1, 2}, num_workers=5)
        assert len(frames) == N_FRAMES
        # only the requested tag ids
        assert [sorted(f['corners']) for f in frames] == [[k % 50] if k % 50 < 3 else [] for k in range(N_FRAMES)]
        assert all(c.shape == (4, 2) for f in frames for c in f['corners'].values())


def test():
    test_chunked_equals_sequential()
    test_detect_video_tag_corners()
    print('all passed')


if __name__ == '__main__':
    test()
