"""
scripts_tag_tracking/07_export_mcap.py video_messages_exact on a generated H.265 MP4 (one keyframe
per second, a frame number drawn into every frame): the stream starts exactly at the requested
frame, has every frame up to the stop, decodes in one pass across the switch from the re-encoded
start to the original stream, and each decoded frame shows the frame it should.

python tests/test_mcap_exact_start.py
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts_tag_tracking'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
import tempfile
import av
import cv2
import numpy as np

export = importlib.import_module('07_export_mcap')
W, H, FPS = 320, 240, 30


def frame_image(k):
    img = np.full((H, W, 3), 40, np.uint8)
    cv2.putText(img, str(k), (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 10)
    return img


def read_number(img_gray):
    # the frame whose drawn image matches best
    return int(np.argmin([np.abs(img_gray.astype(int) - cv2.cvtColor(frame_image(k), cv2.COLOR_BGR2GRAY)).mean()
        for k in range(120)]))


def make_video(path, n=120):
    with av.open(path, 'w') as out:
        s = out.add_stream('libx265', rate=FPS)
        s.width, s.height, s.pix_fmt = W, H, 'yuv420p'
        s.options = {'x265-params': f'keyint={FPS}:min-keyint={FPS}:bframes=0:scenecut=0:log-level=error'}
        for k in range(n):
            for p in s.encode(av.VideoFrame.from_ndarray(frame_image(k), format='bgr24')):
                out.mux(p)
        for p in s.encode():
            out.mux(p)


def decode_all(messages):
    dec = av.CodecContext.create('hevc', 'r')
    frames = list()
    for _, data in messages:
        for packet in dec.parse(data):
            frames.extend(dec.decode(packet))
    for packet in dec.parse(b''):
        frames.extend(dec.decode(packet))
    frames.extend(dec.decode(None))
    return frames


def test_exact_start():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'v.mp4')
        make_video(path)
        for start, stop in [(45, 100), (30, 70), (47, 55), (0, 40)]:
            messages = list(export.video_messages_exact(path, start, stop))
            assert [i for i, _ in messages] == list(range(start, stop)), (start, stop, [i for i, _ in messages][:5])
            frames = decode_all(messages)
            assert len(frames) == stop - start, (start, stop, len(frames))
            got = [read_number(f.to_ndarray(format='gray')) for f in frames]
            assert got == list(range(start, stop)), (start, stop, got[:10])
            # the plain export starts at a keyframe at or before the start (lead-in frames)
            plain = list(export.video_messages(path, start, stop))
            assert plain[0][0] <= start and plain[0][0] % FPS == 0


if __name__ == '__main__':
    test_exact_start()
    print('all passed')
