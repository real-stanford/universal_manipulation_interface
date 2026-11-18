"""
umi.tag_tracking.coverage, umi.tag_tracking.delivery_config and the coverage
fields of the annotation vocabulary.

python tests/test_coverage.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
from umi.tag_tracking.coverage import normalize_task, episode_coverage, coverage_report
from umi.tag_tracking.delivery_config import load_delivery_config, session_declarations
from umi.tag_tracking.annotation import label_inventory, vocabulary_maps, apply_vocabulary, response_schema
import pathlib


def annotation(task, skill, main, objects, outcome='success', difficulty='multi_step'):
    return {'tasks': [{'task': task, 'skill': skill, 'main_object': main, 'objects': [n for n, _ in objects],
        'difficulty': difficulty, 'outcome': outcome}],
        'objects': [{'name': n, 'description': '', 'object_class': c} for n, c in objects]}


def test_coverage():
    assert normalize_task('  Put the cup  on the saucer. ') == 'put the cup on the saucer'
    decl = {'collection_mode': 'scripted', 'scene_tags': ['tabletop', 'workbench'], 'environment_id': 'lab1',
        'operator_id': None, 'station_id': None, 'consent_reference': None}
    a = annotation('Pack the screwdriver.', 'pack', 'screwdriver', [('screwdriver', 'rigid'), ('lid', 'articulated')])
    rows = [episode_coverage(a, decl, 3600.0, 1800.0),
            episode_coverage(annotation('pack the screwdriver', 'pack', 'lid', [('lid', 'articulated')], 'failure'),
                decl, 1800.0, 900.0),
            episode_coverage(None, decl, 360.0, 180.0)]
    assert rows[0]['object_class'] == 'rigid' and rows[0]['object_tags'] == ['articulated', 'rigid']
    assert rows[2]['task'] is None and rows[2]['task_family'] is None
    r = coverage_report(rows)
    assert r['episodes'] == 3 and r['annotated'] == 2 and r['distinct_tasks'] == 1
    assert abs(r['hours'] - 1.6) < 1e-9 and abs(r['task_hours'] - 0.8) < 1e-9
    assert r['by']['task_family']['pack']['episodes'] == 2 and r['by']['task_family']['(none)']['episodes'] == 1
    assert r['by']['scene']['tabletop']['episodes'] == 3 and r['by']['scene']['workbench']['episodes'] == 3
    assert r['by']['operator_id']['(none)']['episodes'] == 3
    assert r['by']['outcome']['failure']['episodes'] == 1


def test_delivery_config():
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d, 'delivery.yaml')
        p.write_text("sessions:\n  task1:\n    collection_mode: scripted\n    scene_tags: [tabletop]\n")
        c = load_delivery_config(str(p))
        assert session_declarations(c, pathlib.Path('/x/task1'))['collection_mode'] == 'scripted'
        assert session_declarations(c, pathlib.Path('/x/task1'))['operator_id'] is None
        assert session_declarations(c, pathlib.Path('/x/task9'))['scene_tags'] is None
        for bad in ("collection_mode: teleop", "scene_tags: [garden]", "operator: bob"):
            p.write_text(f"sessions:\n  task1:\n    {bad}\n")
            try:
                load_delivery_config(str(p))
                assert False, bad
            except AssertionError as e:
                assert 'task1' in str(e), bad


def test_vocabulary_classes():
    schema = response_schema(['right_2'], ['right_close_1'])
    task = schema['properties']['task']['properties']
    assert 'difficulty' in task and 'main_object' in task
    assert 'failure_kind' in schema['properties']['recoveries']['items']['properties']
    answer = {'task': {'task': 'pack driver', 'skill': 'pack', 'difficulty': 'multi_step', 'objects': ['driver'],
        'main_object': 'driver', 'success_predicate': '', 'outcome': 'success', 'completed_at_s': 1.0,
        'instruction_variants': [], 'confidence': 1.0}, 'spans': [], 'gripper_events': [], 'state_changes': [],
        'recoveries': [], 'issues': [], 'scene_graph': [],
        'objects': [{'name': 'driver', 'description': 'blue', 'object_class': 'tool'}]}
    inv = label_inventory([answer])
    assert inv['objects']['driver']['object_class'] == ['tool'] and inv['tasks']['pack driver']['difficulty'] == ['multi_step']
    vocab = {'objects': [{'name': 'blue screwdriver', 'description': 'blue driver', 'object_class': 'rigid',
        'aliases': ['driver']}], 'phrases': [],
        'tasks': [{'task': 'pack blue screwdriver', 'skill': 'pack', 'difficulty': 'single_step',
            'success_predicate': 'in tray', 'instruction_variants': ['a', 'b'], 'aliases': ['pack driver']}]}
    maps, _ = vocabulary_maps(vocab, inv)
    b = apply_vocabulary(answer, maps)
    assert b['task']['main_object'] == 'blue screwdriver' and b['task']['difficulty'] == 'single_step'
    assert b['objects'] == [{'name': 'blue screwdriver', 'description': 'blue driver', 'object_class': 'rigid'}]
    # an answer written before the coverage fields existed still goes through
    old = {**answer, 'task': {k: v for k, v in answer['task'].items() if k not in ('difficulty', 'main_object')},
        'objects': [{'name': 'driver', 'description': 'blue'}]}
    inv = label_inventory([old])
    maps, _ = vocabulary_maps({'objects': [], 'phrases': [], 'tasks': []}, inv)
    assert apply_vocabulary(old, maps)['objects'] == [{'name': 'driver', 'description': 'blue', 'object_class': None}]


if __name__ == '__main__':
    test_coverage()
    test_delivery_config()
    test_vocabulary_classes()
    print("all passed")
