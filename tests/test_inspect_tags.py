"""
umi.tag_tracking.tag_detection.tag_inventory and scripts_tag_tracking/inspect_tags.py
on a generated video with two fixed (finger-like) tags and one that comes and goes.

python tests/test_inspect_tags.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import tempfile
import av
import cv2
import numpy as np
from umi.tag_tracking.tag_detection import tag_inventory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_inventory():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    frames = [{0: square + [100, 400], 1: square + [500, 400]}, {0: square + [102, 400]}, {}, {7: 2 * square}]
    inv = tag_inventory(frames, (640, 480))
    assert list(inv) == [0, 1, 7]
    assert inv[0]['share'] == 0.5 and inv[1]['share'] == 0.25 and inv[7]['side_px'] == 20.0
    assert abs(inv[0]['center'][0] - 106 / 640) < 1e-9 and abs(inv[0]['center'][1] - 405 / 480) < 1e-9


def test_script():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'v.mp4')
        with av.open(path, 'w') as out:
            s = out.add_stream('libx264', rate=30)
            s.width, s.height, s.pix_fmt = 640, 480, 'yuv420p'
            for i in range(120):  # 4 s
                img = np.full((480, 640), 255, np.uint8)
                img[380:460, 100:180] = cv2.aruco.generateImageMarker(d, 0, 80)
                img[380:460, 460:540] = cv2.aruco.generateImageMarker(d, 1, 80)
                if i < 60:  # first 2 s only
                    img[50:150, 270:370] = cv2.aruco.generateImageMarker(d, 33, 100)
                frame = av.VideoFrame.from_ndarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), format='rgb24')
                for p in s.encode(frame):
                    out.mux(p)
            for p in s.encode():
                out.mux(p)
        env = {**os.environ, 'PATH': os.path.dirname(sys.executable) + os.pathsep + os.environ['PATH']}
        r = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts_tag_tracking', 'inspect_tags.py'), '-j', '1',
            '--every_sec', '0.5', path], capture_output=True, text=True, env=env, cwd=ROOT)
        assert r.returncode == 0, r.stderr
        rows = {int(l.split()[0]): l.split() for l in r.stdout.splitlines() if l.strip()[:1].isdigit() and len(l.split()) > 5}
        assert set(rows) == {0, 1, 33}, r.stdout
        assert rows[0][1] == '100%' and rows[1][1] == '100%' and rows[33][1] == '50%', r.stdout
        assert rows[0][3] == '0.22' and rows[1][3] == '0.78', r.stdout  # finger tags left / right of center
        assert '16 mm' in ' '.join(rows[0]) and '80 mm' in ' '.join(rows[33]), r.stdout


if __name__ == '__main__':
    test_inventory()
    test_script()
    print("all passed")
