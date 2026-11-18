"""
umi.tag_tracking.annotation: gripper events from the width, spans covering
the episode.

python tests/test_annotation.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from umi.tag_tracking.annotation import (gripper_events, arm_spans, turning_points, frame_times_for_review,
    measured_items, response_schema, request_text, merge_annotation)

FPS = 60000 / 1001


def test_events():
    t = np.arange(0.0, 20.0, 1 / FPS)
    # open 80 mm; 3-4 s close onto an object (stops at 30 mm); 9-10 s open; 13-14 s close on nothing
    w = np.full(len(t), 0.080)
    w[(t >= 3) & (t < 4)] = np.linspace(0.080, 0.030, np.sum((t >= 3) & (t < 4)))
    w[(t >= 4) & (t < 9)] = 0.030
    w[(t >= 9) & (t < 10)] = np.linspace(0.030, 0.080, np.sum((t >= 9) & (t < 10)))
    w[(t >= 10) & (t < 13)] = 0.080
    w[(t >= 13) & (t < 14)] = np.linspace(0.080, 0.0, np.sum((t >= 13) & (t < 14)))
    w[t >= 14] = 0.0
    rng = np.random.default_rng(0)
    w = w + rng.normal(scale=0.0005, size=len(w))
    w[500:520] = np.nan                                   # finger tags not seen
    events = gripper_events(t, w, 0.0, 20.0)
    assert [e['kind'] for e in events] == ['close', 'open', 'close'], events
    assert abs(events[0]['t'] - 4.0) < 0.1 and abs(events[0]['t_begin'] - 3.0) < 0.1
    assert events[0]['holding'] and not events[2]['holding']
    assert abs(events[1]['t'] - 10.0) < 0.1 and abs(events[2]['t'] - 14.0) < 0.1
    # noise alone: no events
    assert gripper_events(t, 0.08 + rng.normal(scale=0.001, size=len(t)), 0.0, 20.0) == []


def test_turning_points():
    w = np.array([0.0, 0.002, 0.001, 0.02, 0.019, 0.021, 0.0, 0.001])
    assert turning_points(w, 0.01) == [0, 5, 6]


def test_spans():
    events = [{'t': 4.0, 'kind': 'close'}, {'t': 10.0, 'kind': 'open'}, {'t': 14.0, 'kind': 'close'},
        {'t': 25.0, 'kind': 'open'}]
    spans = arm_spans(0.0, 30.0, 2.0, 20.0, events)
    assert [(s['start'], s['end'], s['phase']) for s in spans] == [
        (0.0, 2.0, 'reset'), (2.0, 4.0, 'approach'), (4.0, 10.0, 'hold'), (10.0, 14.0, 'approach'),
        (14.0, 20.0, 'hold'), (20.0, 30.0, 'reset')]
    # spans cover the episode without gaps or overlap
    assert all(a['end'] == b['start'] for a, b in zip(spans[:-1], spans[1:]))
    picks = frame_times_for_review(2.0, 20.0, {'right': events}, max_frames=8)
    assert picks[0] == (2.0, 'task start') and picks[-1] == (20.0, 'task end') and len(picks) <= 8


def merge_inputs():
    arm_events = {
        'right': [{'t': 14.0, 't_begin': 13.5, 'kind': 'close', 'width': 0.02, 'change': -0.06, 'holding': True},
                  {'t': 18.0, 't_begin': 17.0, 'kind': 'open', 'width': 0.08, 'change': 0.06}],
        'left': [{'t': 11.0, 't_begin': 10.5, 'kind': 'close', 'width': 0.0, 'change': -0.05, 'holding': False}],
    }
    # episode 10-30 s, task 12-25 s: the left close at 11 s is in the reset span
    return measured_items(10.0, 30.0, 12.0, 25.0, arm_events)


def test_measured_items():
    spans, events = merge_inputs()
    assert list(events) == ['right_close_1', 'right_open_1']
    right = [(s['phase'], s['start'], s['end']) for i, s in spans.items() if s['arm'] == 'right']
    assert right == [('reset', 10.0, 12.0), ('approach', 12.0, 14.0), ('hold', 14.0, 18.0),
        ('retract', 18.0, 25.0), ('reset', 25.0, 30.0)]
    left = [s['phase'] for s in spans.values() if s['arm'] == 'left']
    assert left == ['reset', 'approach', 'reset']


def test_schema_and_merge():
    spans, events = merge_inputs()
    named = [i for i, s in spans.items() if s['phase'] != 'reset']
    schema = response_schema(named, list(events))

    def strict(node):
        if node.get('type') == 'object':
            assert node['additionalProperties'] is False and set(node['required']) == set(node['properties'])
            for v in node['properties'].values():
                strict(v)
        elif node.get('type') == 'array':
            strict(node['items'])
    strict(schema)
    assert schema['properties']['spans']['items']['properties']['span_id']['enum'] == named
    assert response_schema([], [])['properties']['gripper_events']['items']['properties']['event_id']['enum'] == ['none']
    text = request_text(10.0, 30.0, 12.0, 25.0, spans, events, None)
    assert 'right_close_1: right, 4.00, 20 mm (stopped while open' in text and 'right_1:' not in text

    answer = {
        'task': {'task': 'put the cup on the saucer', 'skill': 'place', 'objects': ['cup', 'saucer'],
            'success_predicate': 'cup rests on saucer', 'outcome': 'success', 'completed_at_s': 8.5,
            'instruction_variants': ['place the cup on the saucer', 'set the cup down on its saucer'], 'confidence': 0.9},
        'spans': [{'span_id': 'right_2', 'verb': 'reach for cup', 'object': 'cup', 'destination': '',
            'contribution': 'advancing', 'progress': 0.2, 'confidence': 0.8},
            # the model naming a reset span is ignored
            {'span_id': 'right_1', 'verb': 'wave', 'object': '', 'destination': '',
            'contribution': 'wasteful', 'progress': 0.0, 'confidence': 0.5}],
        'gripper_events': [{'event_id': 'right_close_1', 'label': 'right gripper grasps cup', 'object': 'cup',
            'grasped_or_released': True, 'outcome': 'success', 'note': ''}],
        'state_changes': [], 'recoveries': [], 'issues': [], 'objects': [{'name': 'cup', 'description': 'white mug'}],
        'scene_graph': [],
    }
    out = merge_annotation(answer, 10.0, 12.0, 25.0, spans, events)
    task = out['tasks'][0]
    assert (task['start_s'], task['end_s'], task['completed_at_s']) == (2.0, 15.0, 8.5)
    assert task['actions'] == ['reach for cup']
    ev = {(e['arm'], e['start_s']): e for e in out['events']}
    assert ev[('right', 0.0)]['verb'] == 'reset' and ev[('right', 0.0)]['contribution'] == 'idle'
    assert ev[('right', 2.0)]['verb'] == 'reach for cup'
    assert ev[('right', 4.0)]['verb'] == '' and ev[('right', 4.0)]['measured_phase'] == 'hold'
    assert sum(e['end_s'] - e['start_s'] for e in out['events'] if e['arm'] == 'right') == 20.0
    kinds = [(k['t_s'], k['kind'], k['outcome']) for k in out['key_events']]
    assert kinds == [(4.0, 'contact', 'success'), (8.0, 'contact', None), (8.5, 'subgoal_complete', 'success')]
    # not a success: no completion time
    answer['task']['outcome'] = 'failure'
    assert merge_annotation(answer, 10.0, 12.0, 25.0, spans, events)['tasks'][0]['completed_at_s'] is None


if __name__ == '__main__':
    test_turning_points()
    test_events()
    test_spans()
    test_measured_items()
    test_schema_and_merge()
    print("all passed")
