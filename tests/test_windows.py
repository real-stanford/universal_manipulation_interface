# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import numpy as np
from umi.tag_tracking.windows import episode_windows, window_frames, scatter


def test():
    times = np.arange(0, 100, 0.5)
    # two episodes, the second and third overlap after the margin -> merged
    windows = episode_windows(times, [{'start': 10, 'end': 20}, {'start': 30, 'end': 40}, {'start': 40.6, 'end': 50}], 0.5)
    assert windows == [(19, 42), (59, 102)], windows
    assert episode_windows(times, None, 0.5) == [(0, 200)]
    frames = [{'time': t, 'corners': {}} for t in times]
    selected, idx = window_frames(frames, windows)
    assert len(selected) == len(idx) == (42 - 19) + (102 - 59)
    assert selected[0]['sequence'] == 0 and selected[-1]['sequence'] == 1
    full = scatter(len(times), idx, {'x': np.ones(len(idx)), 'n': np.ones(len(idx), dtype=int),
        'ok': np.ones(len(idx), dtype=bool), 'T': np.ones((len(idx), 4, 4))})
    assert np.isnan(full['x'][0]) and full['x'][19] == 1 and full['n'][0] == 0 and not full['ok'][0]
    assert full['T'].shape == (200, 4, 4) and np.isnan(full['T'][150, 0, 0]) and full['T'][100, 0, 0] == 1
    print('all passed')


def test_split_window():
    from umi.tag_tracking.windows import split_window
    times = np.arange(0.0, 200.0, 1 / 60)
    chunks = split_window(times, 600, 6600, 12.0, 1.0)      # 100 s window
    assert len(chunks) == 9
    keep = [i for lo, hi, a, b in chunks for i in range(a, b)]
    assert keep == list(range(600, 6600))                    # kept parts tile the window exactly
    for lo, hi, a, b in chunks:
        assert lo <= a and b <= hi and 600 <= lo and hi <= 6600
        if a > 600:
            assert times[a] - times[lo] >= 1.0 - 1e-9        # context before
        if b < 6600:
            assert times[hi - 1] - times[b - 1] >= 1.0 - 1e-9  # context after
    assert split_window(times, 0, 600, 12.0, 1.0) == [(0, 600, 0, 600)]


if __name__ == '__main__':
    test_split_window()
    test()
