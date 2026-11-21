"""
umi.tag_tracking.session.unsynced_recordings: a recording with a video the audio sync rejected
is left out whole (all videos overlapping it in time), other recordings are kept.

python tests/test_session_sync.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from umi.tag_tracking.session import unsynced_recordings


def test_unsynced_recordings():
    # station3 task01: recording 3 (head 49 s from 0, wrists 30 s from 21 s, wrists not synced),
    # recording 4 from 91 s, all synced
    v = lambda name, start, length, synced: {'name': name, 'start': start, 'end': start + length, 'synced': synced}
    videos = [v('head3', 0, 49, True), v('right3', 21, 30, False), v('left3', 21, 30, False),
        v('head4', 91, 31, True), v('right4', 93, 33, True), v('left4', 93, 33, True)]
    assert unsynced_recordings(videos) == {'head3', 'right3', 'left3'}
    assert unsynced_recordings([x for x in videos if x['name'].endswith('4')]) == set()


if __name__ == '__main__':
    test_unsynced_recordings()
    print('all passed')
