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

import json
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


def test_long_gaps():
    take = (1000.0, 1600.0)
    eps = [(1010, 1100), (1110, 1300), (1400, 1590)]
    assert ry.long_gaps(take, eps, 30.0) == [(1300, 1400)]
    assert ry.long_gaps(take, [], 30.0) == [take]
    assert ry.long_gaps((1000.0, 1600.0), [(1100, 1500)], 30.0) == [(1000.0, 1100), (1500, 1600.0)]


SEGMENT = """demo_C3501326495617_2026.11.21_13.27.49.020600: 16 episodes (27.8-73.9s, total 585s), 16 idle periods, 36 bangs, 1 rejected stretches; wrist IMU to head audio lag 22 ms
  rejected 00:00.0-00:02.1: too_short
demo_C3501326495617_2026.11.21_13.22.53.325200: 8 episodes (11.1-37.1s, total 224s), 7 idle periods, 14 bangs, 0 rejected stretches
"""
TRIM = """python scripts_tag_tracking/03t_trim_docking.py -i session -c config.yaml
  episode 1 00:00.0-00:33.8: task 00:04.9-00:30.2 (25.3s; removed 4.9s + 3.6s; start at dock, end at dock)
  episode 2 00:35.6-01:12.1: task 00:38.4-01:07.6 (29.2s; removed 2.8s + 4.5s; start at bang, end at dock)
  episode 3 01:20.3-01:50.9: cubes never left the dock area, no task
demo_C3501326495617_2026.11.21_13.22.53.325200: usual removal at a bang boundary 3.6s, 0 episodes flagged for review
demo_C3501326495617_2026.11.21_13.22.53.325200: 2 of 3 episodes with a task, 25.3-29.2s, total 54s
  episode 1 00:02.0-00:40.0: task 00:05.0-00:35.0 (30.0s; removed 3.0s + 5.0s; start at bang, end at bang)
  episode 2 01:00.0-01:30.0: task 01:02.0-01:28.0 (26.0s; removed 2.0s + 2.0s; start at bang, end at bang)
demo_C3501326495617_2026.11.21_13.27.49.020600: usual removal at a bang boundary 3.0s
demo_C3501326495617_2026.11.21_13.27.49.020600: 2 of 2 episodes with a task, 26.0-30.0s, total 56s
"""
PLAN = """demo_C3501326501951_2026.11.21_13.22.55.327200: gripper width in 100% of frames
demo_C3501326495617_2026.11.21_13.22.53.325200: 14462 frames, valid poses right_cube 87%, left_cube 87%, all grippers 100%
  episode 3 (80.3-110.9s): dropped, the cubes never left the dock
demo_C3501326495617_2026.11.21_13.27.49.020600: 38001 frames, valid poses right_cube 91%, left_cube 90%, all grippers 100%
  episode 2 (62.0-88.0s): dropped, not recorded by all wrist cameras
3 episodes, 2123.4s of 3219.6s used (66%)
"""


def test_parse_logs():
    p = ry.parse_logs(SEGMENT, TRIM, PLAN)
    a, b = 'demo_C3501326495617_2026.11.21_13.22.53.325200', 'demo_C3501326495617_2026.11.21_13.27.49.020600'
    assert p['kept'] == 3
    assert p['heads'][a]['episodes'] == {1: (4.9, 30.2), 2: (38.4, 67.6)} and p['heads'][a]['no_task'] == {3: (80.3, 110.9)}
    assert p['heads'][b]['episodes'] == {1: (5.0, 35.0), 2: (62.0, 88.0)}
    assert p['heads'][b]['dropped'] == {2: (62.0, 88.0, 'not recorded by all wrist cameras')}
    assert p['heads'][b]['rejected'] == [(0.0, 2.1, 'too_short')]
    assert ry.gap_reasons((60, 90), [(62, 88, 'dropped: x'), (0, 2, 'rejected: y')]) == 'dropped: x'
    assert ry.gap_reasons((200, 300), [(62, 88, 'dropped: x')]) == 'no episode found (no bangs, or idle)'


def test_mp4_session():
    # two takes, one head video each; the logs above: 3 kept episodes, one dropped, one without a task
    files = {'s3://m/report.json': json.dumps({'episodes': [{'episode': f'episode_{k:06d}', 'status': 'ok', 'frames': 1800}
            for k in range(3)]}),
        's3://m/logs/06_segment_episodes.log': SEGMENT, 's3://m/logs/11_trim_docking.log': TRIM,
        's3://m/logs/16_export_plan.log': PLAN}
    real = ry.read_optional
    ry.read_optional = lambda url: files.get(url)
    try:
        heads = [[(1000.0, 200.0, 'head/GX010001.MP4')], [(5000.0, 200.0, 'head/GX010002.MP4')]]
        spans, items = [[], []], [[], []]
        n, sec, dropped, status = ry.mp4_session('s3://m', heads, spans, items)
    finally:
        ry.read_optional = real
    assert n == 3 and abs(sec - 3 * 1800 / (60000 / 1001)) < 1e-6 and status == 'processed (episode videos)', status
    assert spans[0] == [(1004.9, 1030.2, 'episode_000000'), (1038.4, 1067.6, 'episode_000001')], spans
    assert spans[1] == [(5005.0, 5035.0, 'episode_000002')], spans
    assert (5062.0, 5088.0, 'dropped: not recorded by all wrist cameras') in items[1]
    assert (1080.3, 1110.9, 'no task (cubes never left the dock)') in items[0]
    assert 'not recorded by all wrist cameras: 1' in dropped
    # the dropped episode explains the long gap in the second take
    gaps = ry.long_gaps((5000.0, 5200.0), [(s, e) for s, e, _ in spans[1]], 30.0)
    assert ry.gap_reasons(gaps[0], items[1]).startswith('dropped: not recorded by all wrist cameras'), gaps


if __name__ == '__main__':
    test_parse_logs()
    test_mp4_session()
    test_long_gaps()
    test_match_within_one_day()
    test_place_and_gaps()
    test_shared_and_names()
    print('all passed')
