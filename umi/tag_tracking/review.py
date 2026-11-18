"""
Human review of an episode (scripts_tag_tracking/12_review_viewer.py),
kept apart from the model's annotation (08 / 09 rebuild that file) in
annotations/<episode>_review.json:

{'reviewed': bool, 'reviewer': str, 'reviewed_ts_utc': str,
 'exclude': bool, 'exclude_reason': str,
 'faces_visible': None | bool,             # privacy, spec 1.10
 'task': {'task': str | None, 'outcome': str | None},
 'events': {'<arm>@<start_s, 3 decimals>': {'verb', 'object', 'destination'}},
 'notes': str}

apply_review gives the annotation as delivered: the reviewer's values over
the model's, annotation_source 'model_reviewed' once reviewed. Spans are
keyed by arm and measured start, which do not change when 08 / 09 rerun.
"""
from typing import Dict, Optional
import copy

OUTCOMES = ['success', 'failure', 'partial', 'aborted', 'unknown']


def empty_review() -> Dict:
    return {'reviewed': False, 'reviewer': '', 'reviewed_ts_utc': None, 'exclude': False, 'exclude_reason': '',
        'faces_visible': None, 'task': {'task': None, 'outcome': None}, 'events': dict(), 'notes': ''}


def span_key(event: dict) -> str:
    return f"{event['arm']}@{event['start_s']:.3f}"


def check_review(review: Dict) -> Dict:
    """
    The review with every key present and values of the right kind; raises on an unknown outcome.
    """
    r = {**empty_review(), **(review or dict())}
    r['task'] = {**empty_review()['task'], **(r.get('task') or dict())}
    outcome = r['task']['outcome']
    assert outcome in OUTCOMES + [None], f"outcome {outcome} not in {OUTCOMES}"
    assert r['faces_visible'] in (None, True, False), "faces_visible must be true, false or null"
    r['task']['task'] = (r['task']['task'] or '').strip() or None
    r['events'] = {k: {f: str(v.get(f) or '').strip() for f in ('verb', 'object', 'destination')}
        for k, v in (r['events'] or dict()).items()}
    return r


def apply_review(annotation: Dict, review: Optional[Dict]) -> Dict:
    """
    The annotation with the review applied (a copy). Only reviewed reviews count.
    """
    a = copy.deepcopy(annotation)
    if not review or not review.get('reviewed'):
        return a
    r = check_review(review)
    task = a['tasks'][0]
    if r['task']['task']:
        task['task'] = r['task']['task']
    if r['task']['outcome'] and r['task']['outcome'] != task['outcome']:
        task['outcome'] = r['task']['outcome']
        if task['outcome'] != 'success':
            task['completed_at_s'] = None
            a['key_events'] = [k for k in a['key_events'] if k['kind'] != 'subgoal_complete']
    for e in a['events']:
        edit = r['events'].get(span_key(e))
        if edit and e.get('measured_phase') != 'reset':
            e.update({f: edit[f] for f in ('verb', 'object', 'destination') if edit[f]})
    task['actions'] = sorted({e['verb'] for e in a['events'] if e['verb'] and e['verb'] != 'reset'})
    a['annotation_source'] = 'model_reviewed' if a.get('annotation_source') == 'model' else a.get('annotation_source')
    a['review'] = {k: r[k] for k in ('reviewer', 'reviewed_ts_utc', 'exclude', 'exclude_reason', 'faces_visible', 'notes')}
    return a
