"""
scripts_tag_tracking/sort_recordings.py: grouping videos into recordings.

python tests/test_sort_recordings.py
"""
import sys
import numpy as np
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts_tag_tracking'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sort_recordings import group_recordings, span, top_level_atoms, parse_mvhd, pairs_rows, pairs_csv, propose_tasks, tasks_yaml, \
    session_files, placed_names, is_complete


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


def test_mp4_header():
    import datetime
    def atom(kind, body):
        return (8 + len(body)).to_bytes(4, 'big') + kind + body
    created = int((datetime.datetime(2026, 11, 20, 4, 55, 10) - datetime.datetime(1904, 1, 1)).total_seconds())
    mvhd = atom(b'mvhd', bytes([0, 0, 0, 0]) + created.to_bytes(4, 'big') + created.to_bytes(4, 'big')
        + (1000).to_bytes(4, 'big') + (45500).to_bytes(4, 'big') + bytes(80))
    moov = atom(b'moov', mvhd)
    data = atom(b'ftyp', b'mp41') + atom(b'mdat', bytes(100)) + moov
    atoms = top_level_atoms(data)
    assert [a[0] for a in atoms] == [b'ftyp', b'mdat', b'moov'] and atoms[2][1] == 12 + 108
    t, duration = parse_mvhd(data[atoms[2][1]:])
    assert t.replace(tzinfo=None) == datetime.datetime(2026, 11, 20, 4, 55, 10) and duration == 45.5
    # only the first bytes known: mdat's header gives where moov starts
    head = top_level_atoms(data[:30])
    assert head[-1][0] == b'mdat' and head[-1][1] + head[-1][2] == atoms[2][1]


def test_pairs():
    videos = [v('head', '10:00:00', 600), v('left', '10:00:03', 597), v('right', '10:00:02', 240),
              v('head', '10:20:00', 60), v('left', '10:20:01', 58)]
    groups = group_recordings(videos)
    spec = {'recordings': [0, 1], 'gripper_calibration': ['right/GX1.MP4', 'left/GX2.MP4'], 'head_mapping': ['head/GX3.MP4']}
    rows = pairs_rows(groups, spec, ['head', 'left', 'right'])
    assert rows[0]['head'] == '10:00:00' and rows[0]['right_sec'] == 240 and rows[0]['shared_sec'] == 239.0
    assert 'right stops after 4.0 min' in rows[0]['note']
    assert rows[1]['right'] == '' and rows[1]['shared_sec'] == 0.0 and 'missing: right' in rows[1]['note']
    assert [r['type'] for r in rows[2:]] == ['gripper_calibration', 'gripper_calibration', 'head_mapping']
    assert rows[2]['right'] == 'GX1.MP4' and rows[2]['left'] == '' and rows[4]['head'] == 'GX3.MP4'
    text = pairs_csv(rows, ['head', 'left', 'right'])
    assert text.splitlines()[0] == 'type,recording,start,end,head,head_sec,left,left_sec,right,right_sec,shared_sec,note'
    assert len(text.splitlines()) == 6


def test_split_and_same_names():
    # station3 2026-11-20 recording 11: both wrists restarted (90 s + 511 s) inside one 614 s head video;
    # and two cameras numbering a file GX010136
    def f(folder, start, duration, name):
        return {'file': f'{folder}/{name}', 'folder': folder, 'serial': folder, 'start': f'2026-11-20T{start}',
            'duration': duration}
    videos = [f('head', '15:07:56', 614, 'GX010172.MP4'), f('left', '15:07:58', 90, 'GX019315.MP4'),
        f('left', '15:09:40', 511, 'GX019316.MP4'), f('right', '15:07:58', 90, 'GX010136.MP4'),
        f('right', '15:09:41', 511, 'GX010146.MP4'),
        f('left', '15:30:00', 15, 'GX010136.MP4'), f('right', '15:30:00', 15, 'GX019320.MP4'),
        f('head', '15:30:00', 16, 'GX010180.MP4')]
    groups = group_recordings(videos)
    assert [len(g) for g in groups] == [5, 3]
    spec = {'recordings': [0], 'gripper_calibration': ['left/GX010136.MP4', 'right/GX019320.MP4'],
        'head_mapping': ['head/GX010180.MP4']}
    files = session_files(groups, spec)
    assert len(files) == 8   # every file of the take, two per wrist
    names = placed_names(files)
    # right/GX010136 (take, raw_videos/) and left/GX010136 (calibration, gripper_calibration/) do not meet
    assert names['right/GX010136.MP4'] == 'GX010136.MP4' and names['left/GX010136.MP4'] == 'GX010136.MP4'
    clash = placed_names([('right/GX010136.MP4', ''), ('left/GX010136.MP4', ''), ('head/GX010172.MP4', '')])
    assert clash == {'right/GX010136.MP4': 'right_GX010136.MP4', 'left/GX010136.MP4': 'left_GX010136.MP4',
        'head/GX010172.MP4': 'GX010172.MP4'}
    rows = pairs_rows(groups, spec, ['head', 'left', 'right'], clash)
    assert rows[0]['left'] == 'GX019315.MP4; GX019316.MP4' and rows[0]['left_sec'] == 601
    assert rows[0]['right'] == 'right_GX010136.MP4; GX010146.MP4'
    assert 'left: 2 files, gaps 12 s' in rows[0]['note'], rows[0]['note']
    assert is_complete(groups[0], ['head', 'left', 'right'], 5.0)
    # a wrist file copied into the head folder by mistake: excluded from the files and the sheet
    spec_x = dict(spec, exclude=['left/GX019315.MP4'])
    assert 'left/GX019315.MP4' not in [f for f, _ in session_files(groups, spec_x)]
    assert pairs_rows(groups, spec_x, ['head', 'left', 'right'])[0]['left'] == 'GX019316.MP4'


def test_propose():
    def rec(t, sec, folders=('head', 'left', 'right')):
        return [v(f, t, sec) for f in folders]
    videos = (rec('09:00:00', 300)                        # 0 take before any calibration
        + rec('09:10:00', 30) + rec('09:11:00', 18)        # 1 head mapping, 2 gripper calibration (other order)
        + rec('09:12:00', 600) + rec('09:23:00', 25)       # 3 take, 4 unknown short clip
        + rec('09:24:00', 500) + rec('09:33:00', 200, ('left', 'right'))   # 5 take, 6 head missing
        + rec('09:38:00', 20) + rec('09:39:00', 70)        # 7 right recalibration, 8 take
        + rec('09:45:00', 15) + rec('09:45:30', 20) + rec('09:46:00', 28)  # 9 left calib, 10 right calib, 11 mapping
        + rec('09:47:00', 70))                             # 12 take
    groups = group_recordings(videos)
    roles = {1: ('head_mapping', [], 'e'), 2: ('gripper_calibration', ['left', 'right'], 'e'), 4: ('unknown', [], 'e'),
        7: ('gripper_calibration', ['right'], 'e'), 9: ('gripper_calibration', ['left'], 'e'),
        10: ('gripper_calibration', ['right'], 'e'), 11: ('head_mapping', [], 'e')}
    tasks, comments, left = propose_tasks(groups, ['head', 'left', 'right'], 5.0, 60.0, roles)
    assert list(tasks) == ['task01', 'task02']
    assert tasks['task01']['recordings'] == [3, 5, 8] and tasks['task01']['head_mapping'] == ['head/09:10:00']
    assert tasks['task01']['gripper_calibration'] == ['left/09:11:00', 'right/09:11:00']
    assert 'recalibration right = recording 7' in comments['task01']
    assert tasks['task02']['recordings'] == [12]
    assert tasks['task02']['gripper_calibration'] == ['left/09:45:00', 'right/09:45:30']
    assert tasks['task02']['head_mapping'] == ['head/09:46:00']
    assert [k for k, _ in left] == [0, 4, 6] and 'missing head' in dict(left)[6]
    text = tasks_yaml(tasks, comments, left, groups, 'x', roles)
    import yaml
    assert yaml.safe_load(text) == tasks
    assert '# recording 7 (09:38:00): gripper_calibration right (e)' in text


def test_clip_role():
    from umi.tag_tracking.station_config import finger_opening, open_close_cycles, clip_role
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    frames = [{6: sq + [495 - 300 * o, 900], 7: sq + [1495 + 300 * o, 900]} for o in [0, 1, 0, 1, 0, 1, 0]] + [{}]
    op = finger_opening(frames, (2000, 1000), (6, 7))
    assert np.allclose(op['pair'][:7], [0.5, 0.8] * 3 + [0.5]) and np.isnan(op['pair'][7]), op
    assert np.allclose(op['left'][:7], [0.5, 0.8] * 3 + [0.5])
    c, r = open_close_cycles(op['pair'])
    assert c == 3 and abs(r - 0.3) < 1e-9
    assert all(np.isnan(x).all() for x in finger_opening(frames, (2000, 1000), None).values())
    # still fingers 0 / 1, tag 1 often missed, another gripper's finger 6 at the edge: no motion
    flicker = [{0: sq + [480, 900], 6: sq + [1840, 950], **({1: sq + [1560, 900]} if i % 2 else {})} for i in range(12)]
    still = finger_opening(flicker, (2000, 1000), (0, 1))
    # calibration where the right finger tag is lost whenever open: the left tag shows it
    lost = [{0: sq + [495 - 300 * o, 900], **({} if o else {1: sq + [1495, 900]})} for o in [0, 1] * 5]
    one = finger_opening(lost, (2000, 1000), (0, 1))
    env = {14: {'share': 0.5}, 15: {'share': 0.3}, 16: {'share': 0.2}, 8: {'share': 0.9}}
    assert clip_role(env, {'left': op, 'right': still}, {14, 15, 16})[:2] == ('gripper_calibration', ['left'])
    assert clip_role(env, {'left': one, 'right': still}, {14, 15, 16})[:2] == ('gripper_calibration', ['left'])
    assert clip_role(env, {'left': still, 'right': still}, {14, 15, 16})[:2] == ('head_mapping', [])
    nudged = {k: np.array([0.5, 0.5, 0.63, 0.63, 0.5, 0.5]) for k in ['pair', 'left', 'right']}   # one movement
    assert clip_role(env, {'left': nudged, 'right': still}, {14, 15, 16})[:2] == ('head_mapping', [])
    assert clip_role({14: {'share': 0.5}}, {'left': still, 'right': still}, {14, 15, 16})[0] == 'unknown'


if __name__ == '__main__':
    test_groups()
    test_split_and_same_names()
    test_propose()
    test_clip_role()
    test_pairs()
    test_mp4_header()
    print("all passed")
