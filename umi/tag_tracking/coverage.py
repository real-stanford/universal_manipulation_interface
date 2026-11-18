"""
Coverage of a delivery (Odyssey spec 1.6 / 2.9): per episode the coverage
dimensions, from the annotations (task family, main object class, object
tags, difficulty, outcome, task sentence) and the delivery declarations
(scene, environment, operator, station, collection mode); per delivery the
hours and episode counts per dimension and the number of distinct tasks
(task sentences after normalization; run 09_normalize_labels.py first so the
same task has the same sentence).
"""
from typing import Dict, List, Optional
import re

DIMENSIONS = ['task_family', 'object_class', 'scene', 'environment_id', 'operator_id', 'station_id',
    'collection_mode', 'difficulty', 'outcome']
NONE = '(none)'


def normalize_task(sentence: Optional[str]) -> Optional[str]:
    """
    Lower case, single spaces, no surrounding punctuation.
    """
    if not sentence:
        return None
    s = re.sub(r'\s+', ' ', sentence.strip().lower())
    return s.strip(' .,;:!') or None


def episode_coverage(annotation: Optional[dict], declarations: Dict, episode_s: float, task_s: float) -> dict:
    """
    Coverage dimensions of one episode; None where not labeled or not declared.
    """
    row = {'episode_s': episode_s, 'task_s': task_s, 'annotated': annotation is not None,
        'scene': declarations.get('scene_tags') or None}
    for k in ('environment_id', 'operator_id', 'station_id', 'collection_mode'):
        row[k] = declarations.get(k)
    task = annotation['tasks'][0] if annotation else dict()
    objects = {o['name']: o for o in (annotation or dict()).get('objects', [])}
    main = objects.get(task.get('main_object'), dict())
    row.update({
        'task': normalize_task(task.get('task')),
        'task_family': task.get('skill'),
        'object_class': main.get('object_class'),
        'object_tags': sorted({objects[n]['object_class'] for n in task.get('objects', [])
            if objects.get(n, dict()).get('object_class')}),
        'difficulty': task.get('difficulty'),
        'outcome': task.get('outcome'),
    })
    return row


def coverage_report(rows: List[dict]) -> dict:
    """
    Totals, distinct tasks, and per dimension and value: episodes, hours
    (whole episode) and task hours. A list-valued dimension (scene) counts the
    episode under each of its values; not labeled / not declared is '(none)'.
    """
    def add(table, value, row):
        t = table.setdefault(NONE if value is None else value, {'episodes': 0, 'hours': 0.0, 'task_hours': 0.0})
        t['episodes'] += 1
        t['hours'] += row['episode_s'] / 3600.0
        t['task_hours'] += row['task_s'] / 3600.0

    by = {d: dict() for d in DIMENSIONS}
    for row in rows:
        for d in DIMENSIONS:
            values = row[d] if isinstance(row[d], list) else [row[d]]
            for v in values:
                add(by[d], v, row)
    tasks = {r['task'] for r in rows if r['task']}
    return {
        'episodes': len(rows),
        'annotated': sum(r['annotated'] for r in rows),
        'hours': sum(r['episode_s'] for r in rows) / 3600.0,
        'task_hours': sum(r['task_s'] for r in rows) / 3600.0,
        'distinct_tasks': len(tasks),
        'tasks': {t: sum(r['task'] == t for r in rows) for t in sorted(tasks)},
        'by': {d: dict(sorted(v.items(), key=lambda x: -x[1]['episodes'])) for d, v in by.items()},
    }
