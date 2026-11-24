"""
scripts_tag_tracking/report_yield.py: episodes placed into their session's takes across a clock
offset (another day's episodes in the same folder not placed), take time split into episodes, short
gaps and long gaps, the time all three cameras recorded, session file names matched to the day
folder's files.

python tests/test_report_yield.py
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts_tag_tracking'))

import report_yield as ry


def test_place_and_gaps():
    # two takes (head clock, s); episodes on the cameras' clock: 6.5 h later
    takes = [(1000.0, 1600.0), (2000.0, 2600.0)]
    offset = -6.5 * 3600
    eps = [{'start': s - offset, 'end': e - offset} for s, e in [(1010, 1100), (1110, 1300), (1400, 1590), (2005, 2300)]]
    # another day's episodes (24 h earlier): fit no take within +-14 h
    eps += [{'start': 1050 - offset - 86400, 'end': 1150 - offset - 86400}]
    d, fit = ry.place_episodes(takes, eps)
    assert abs(d - offset) <= 12 and fit == [0, 0, 0, 1, None], (d, fit)   # the takes' edge slack (10 s) + tolerance (2 s)
    d = offset      # the gaps with the exact offset
    covered, short, long_ = ry.take_gaps(takes[0], [(e['start'] + d, e['end'] + d) for e in eps[:3]], 30.0)
    # gaps: 10 (start), 10, 100 (long), 10 (end)
    assert abs(covered - 470) < 0.5 and abs(short - 30) < 0.5 and abs(long_ - 100) < 0.5, (covered, short, long_)
    covered, short, long_ = ry.take_gaps(takes[1], [(eps[3]['start'] + d, eps[3]['end'] + d)], 30.0)
    assert abs(covered - 295) < 0.5 and abs(short - 5) < 0.5 and abs(long_ - 300) < 0.5, (covered, short, long_)


def test_shared_and_names():
    group = [{'folder': 'head', 'start': '2026-11-22T10:00:00', 'duration': 600.0},
        {'folder': 'left', 'start': '2026-11-22T10:00:05', 'duration': 300.0},
        {'folder': 'left', 'start': '2026-11-22T10:05:10', 'duration': 290.0},
        {'folder': 'right', 'start': '2026-11-22T10:00:02', 'duration': 590.0}]
    # all three: from 10:00:05 (left) to 10:09:52 (right ends)
    assert abs(ry.shared_seconds(group) - 587) < 0.01, ry.shared_seconds(group)
    assert ry.shared_seconds(group[:3]) == 0.0      # right missing
    assert ry.file_key('left', ' left_GX010136.MP4') == ('left', 'GX010136.MP4')
    assert ry.file_key('head', 'GX010250.MP4') == ('head', 'GX010250.MP4')


def test_match_within_one_day():
    # the same file names on two days (each GoPro numbers its own files): a session matches one day
    days = ['2026-11-20', '2026-11-22']
    where = {('2026-11-20', 'head', 'GX010411.MP4'): (0, 600.0), ('2026-11-20', 'left', 'GX019293.MP4'): (0, 598.0),
        ('2026-11-20', 'right', 'GX010100.MP4'): (0, 599.0),
        ('2026-11-22', 'head', 'GX010411.MP4'): (5, 300.0), ('2026-11-22', 'left', 'GX019293.MP4'): (5, 298.0),
        ('2026-11-22', 'right', 'GX010100.MP4'): (5, 299.0), ('2026-11-22', 'head', 'GX010412.MP4'): (6, 610.0),
        ('2026-11-22', 'left', 'GX019294.MP4'): (6, 608.0), ('2026-11-22', 'right', 'GX010101.MP4'): (6, 609.0)}
    rows = [{'type': 'take', 'head': 'GX010411.MP4', 'head_sec': '300.0', 'left': 'GX019293.MP4', 'left_sec': '298.0',
            'right': 'GX010100.MP4', 'right_sec': '299.0'},
        {'type': 'take', 'head': 'GX010412.MP4', 'head_sec': '610.0', 'left': 'left_GX019294.MP4', 'left_sec': '608.0',
            'right': 'GX010101.MP4', 'right_sec': '609.0'}]
    take_of, clip_of, takes, day = ry.match_sessions({'task01': rows}, where, days)
    assert day == {'task01': '2026-11-22'} and takes == {'task01': [{5}, {6}]}, (day, takes)
    assert dict(take_of) == {5: {'task01'}, 6: {'task01'}} and not clip_of
    # a listed length that does not fit: not matched
    rows[0]['head_sec'] = '600.0'
    _, _, takes, _ = ry.match_sessions({'task01': rows}, where, days)
    assert takes['task01'][0] == {5}, takes      # left and right still match the recording


if __name__ == '__main__':
    test_match_within_one_day()
    test_place_and_gaps()
    test_shared_and_names()
    print('all passed')
