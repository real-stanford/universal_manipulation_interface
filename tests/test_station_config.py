"""
umi.tag_tracking.station_config on synthetic detections.

python tests/test_station_config.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from umi.tag_tracking.station_config import tag_classes, finger_pair, cube_groups, assign_cubes, dock_tags


def tag(x, y, side=20.0):
    return np.array([[0, 0], [side, 0], [side, side], [0, side]], float) + [x, y]


def test_classes():
    c = tag_classes({0: 0.016, 1: 0.016, 2: 0.06, 14: 0.08, 12: 0.16}, 0.016, 0.06)
    assert c == {'finger': {0, 1}, 'cube': {2}, 'environment': {12, 14}}


def test_finger_pair():
    inv = {6: {'share': 0.95, 'center': (0.3, 0.85)}, 7: {'share': 0.9, 'center': (0.7, 0.86)},
        0: {'share': 0.1, 'center': (0.6, 0.4)}, 1: {'share': 0.6, 'center': (0.4, 0.2)}}
    assert finger_pair(inv, {0, 1, 6, 7}) == (6, 7)
    assert finger_pair({6: inv[6]}, {6, 7}) is None          # one finger only
    assert finger_pair(inv, {0, 1}) is None                  # not at the bottom / too rare


def test_cube_groups():
    frames = list()
    for i in range(10):
        f = {2: tag(100 + i, 100), 3: tag(125 + i, 100), 8: tag(500, 300 - i), 9: tag(500, 325 - i)}
        if i % 2:
            f[4] = tag(100 + i, 125)       # only with 2 / 3
            del f[2]
        frames.append(f)
    frames.append({3: tag(300, 200), 8: tag(315, 200)})   # hands together once
    groups, pairs = cube_groups(frames, {2, 3, 4, 8, 9})
    assert groups == [[2, 3, 4], [8, 9]], groups
    assert pairs[(3, 8)]['frames'] == 11 and pairs[(3, 8)]['ratio'] > 2


def test_assign_cubes():
    cubes = [[2, 3], [8, 9]]
    inv = {'right': {2: {'share': 0.2}}, 'left': {8: {'share': 0.1}, 9: {'share': 0.05}}}
    owner, seen = assign_cubes(cubes, inv)
    assert owner == {'right': 1, 'left': 0}, owner        # right sees cube 0 -> its own is cube 1
    assert abs(seen['left'][1] - 0.15) < 1e-9
    assert assign_cubes(cubes, {'right': {}, 'left': {}})[0] is None


def test_dock_tags():
    P = lambda *x: np.array(x, float)
    env = {14: P(0, 0, 1), 16: P(0.4, 0, 1), 33: P(1, 0, 1)}
    docked = (P(0.05, 0.1, 1), P(0.4, 0.1, 1))
    frames = [([docked[0] + [0, 0, 0.001 * k], docked[1]], env) for k in range(4)]   # both resting
    frames += [([P(1.0, 0.1, 1), P(0.4 + 0.1 * k, 0.3, 1)], env) for k in range(3)]  # right hand resting at 33,
    frames += [([P(0.05, 0.1, 1), None], env)]                                          # left moving; one unseen
    per_cube, n = dock_tags([frames])
    assert n == 3, n
    assert per_cube[0] == [{'tag': 14, 'share': 1.0, 'distance': per_cube[0][0]['distance']}]
    assert abs(per_cube[0][0]['distance'] - np.hypot(0.05, 0.1)) < 0.01
    assert [x['tag'] for x in per_cube[1]] == [16]
    assert dock_tags([[([None, None], env)]]) == ([], 0)


def test_finger_symmetric():
    # station1 left camera: its fingers 0 / 1, the right gripper's finger 6 at the image edge
    inv = {0: {'share': 1.0, 'center': (0.24, 0.85)}, 1: {'share': 0.53, 'center': (0.78, 0.86)},
        6: {'share': 0.73, 'center': (0.92, 0.9)}}
    assert finger_pair(inv, {0, 1, 6}) == (0, 1)


if __name__ == '__main__':
    test_classes()
    test_finger_pair()
    test_cube_groups()
    test_assign_cubes()
    test_dock_tags()
    test_finger_symmetric()
    print('all passed')
