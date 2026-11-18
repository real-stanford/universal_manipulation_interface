"""
scripts_tag_tracking/render_episodes.py: frame reading and the time series panel.

python tests/test_render_episodes.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util
import subprocess
import tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load():
    spec = importlib.util.spec_from_file_location('render', os.path.join(ROOT, 'scripts_tag_tracking', 'render_episodes.py'))
    module = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    spec.loader.exec_module(module)
    os.chdir(cwd)
    return module


def test_frames_and_plot():
    r = load()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'v.mp4')
        # frame number burned in as brightness: frame k has gray level 2k; no B-frames, as in GoPro files
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=black:s=320x240:r=60:d=2',
            '-vf', "geq=lum='2*N':cb=128:cr=128", '-c:v', 'libx264', '-bf', '0', '-g', '30', '-pix_fmt', 'yuv420p', path], check=True)
        got = list(r.frames(path, 40, 10, (64, 48), 2))
        assert len(got) == 10 and got[0].shape == (48, 64, 3)
        levels = [int(round(f.mean())) for f in got]
        # gray level Y = 2 * frame index, shown in RGB as (Y - 16) * 255 / 219 (video range)
        assert all(abs(lv - (2 * (40 + 2 * k) - 16) * 255 / 219) <= 3 for k, lv in enumerate(levels)), levels
    t = np.linspace(0, 10, 600)
    series = [('/a', 'mm', {'right': (t, np.sin(t)), 'left': (t, np.cos(t))}), ('/b', 'm', {'right': (t, t)})]
    image, (x0, x1), ys = r.plot_background(t, series, 10.0)
    assert image.shape == (r.PLOT[1], r.PLOT[0], 3) and 0 < x0 < x1 < r.PLOT[0] and len(ys) == 2


if __name__ == '__main__':
    test_frames_and_plot()
    print("all passed")
