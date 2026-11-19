"""
scripts_tag_tracking/sort_recordings.py: grouping videos into recordings.

python tests/test_sort_recordings.py
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts_tag_tracking'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sort_recordings import group_recordings, span


def v(folder, start, duration):
    return {'file': f'{folder}/{start}', 'folder': folder, 'serial': folder, 'start': f'2026-11-18T{start}', 'duration': duration}


def test_groups():
    videos = [v('head', '10:00:00', 120), v('left', '10:00:03', 115), v('right', '10:00:02', 118),
              v('head', '10:05:00', 60), v('right', '10:05:01', 58),          # left missing
              v('left', '10:20:00', 4),                                        # stray fragment
              v('head', '10:30:00', 600), v('left', '10:30:05', 300), v('left', '10:35:06', 290)]  # chaptered left
    groups = group_recordings(videos)
    assert [len(g) for g in groups] == [3, 2, 1, 3]
    start, end = span(groups[3])
    assert (end - start).total_seconds() == 600


if __name__ == '__main__':
    test_groups()
    print("all passed")
