"""
Measured part of the episode annotations (Odyssey spec 1.7 / 2.7): what the
data shows without looking at images. Times are head video seconds.

- gripper events per arm from the finger-tag width (opening relative to
  fully closed, umi.tag_tracking.delivery.finger_widths): a close / open is a
  width change of at least min_change (zigzag turning points); a close ends
  where the width stops falling: still open there (> holding_width) means
  something is between the fingers, about 0 means closed on nothing;
- per arm, spans covering the whole dock-to-dock episode, half-open,
  bounded by the task interval and the gripper events: 'reset' outside the
  task (docking, bang ritual), inside it the phases between events
  ('approach' before a close, 'hold' between a close and an open, 'retract'
  after an open).
The verb, object and outcome of each span come from the visual check
(scripts_tag_tracking/08_annotate_episodes.py).
"""
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

DEFAULTS = {
    'min_change': 0.010,      # m, width change of a close / open
    'holding_width': 0.005,   # m, a close ending wider than this holds something
    'smooth_samples': 7,      # running median of the width
}


def smooth_width(widths: np.ndarray, n: int) -> np.ndarray:
    """
    Running median over n samples ignoring NaN (not measured stays NaN).
    """
    out = np.full(len(widths), np.nan)
    h = n // 2
    for i in range(len(widths)):
        if np.isnan(widths[i]):
            continue
        w = widths[max(0, i - h):i + h + 1]
        w = w[~np.isnan(w)]
        out[i] = np.median(w)
    return out


def turning_points(w: np.ndarray, h: float) -> List[int]:
    """
    Indices of the zigzag turning points of w with hysteresis h: the signal
    changes by at least h between consecutive points, which alternate
    between local maxima and minima. The first is the extreme before the
    first move, the last the extreme of the last move.
    """
    lo = hi = 0
    trend = 0
    turns = list()
    ext = 0
    for i in range(1, len(w)):
        if trend == 0:
            lo = i if w[i] < w[lo] else lo
            hi = i if w[i] > w[hi] else hi
            if w[hi] - w[i] >= h and hi < i:
                turns, trend, ext = [hi], -1, i
            elif w[i] - w[lo] >= h and lo < i:
                turns, trend, ext = [lo], 1, i
        elif trend == 1:
            if w[i] > w[ext]:
                ext = i
            elif w[ext] - w[i] >= h:
                turns.append(ext)
                trend, ext = -1, i
        else:
            if w[i] < w[ext]:
                ext = i
            elif w[i] - w[ext] >= h:
                turns.append(ext)
                trend, ext = 1, i
    if trend != 0:
        turns.append(ext)
    return turns


def gripper_events(times: np.ndarray, widths: np.ndarray, start: float, end: float,
        params: Optional[Dict] = None) -> List[dict]:
    """
    Close / open events of one gripper in [start, end): zigzag turning points
    of the smoothed width with at least min_change between them.
    Returns [{'t' (end of the motion), 't_begin', 'kind' ('close' | 'open'),
    'width' (m at t), 'change' (m), 'holding' (close only: width > holding_width)}].
    """
    p = {**DEFAULTS, **(params or {})}
    inside = (times >= start) & (times < end) & ~np.isnan(widths)
    t, w = times[inside], smooth_width(widths[inside], p['smooth_samples'])
    if len(t) < 3:
        return list()
    turns = turning_points(w, p['min_change'])
    events = list()
    for a, b in zip(turns[:-1], turns[1:]):
        change = w[b] - w[a]
        if abs(change) < p['min_change']:
            continue
        # end of the motion: first sample within 1 mm of the turning point's width
        segment = np.arange(a, b + 1)
        settled = segment[np.abs(w[segment] - w[b]) <= 0.001]
        k = int(settled[0]) if len(settled) else b
        # start of the motion: last sample still within 1 mm of the previous turning point's width
        resting = segment[np.abs(w[segment] - w[a]) <= 0.001]
        j = int(resting[-1]) if len(resting) else a
        event = {'t': float(t[k]), 't_begin': float(t[j]), 'kind': 'close' if change < 0 else 'open',
            'width': float(w[k]), 'change': float(change)}
        if event['kind'] == 'close':
            event['holding'] = bool(w[k] > p['holding_width'])
        events.append(event)
    return events


def arm_spans(start: float, end: float, task_start: float, task_end: float,
        events: Sequence[dict]) -> List[dict]:
    """
    Spans of one arm covering [start, end), half-open, never overlapping:
    'reset' before task_start and from task_end, inside the task split at the
    events within it: 'approach' (before a close), 'hold' (between a close
    and an open), 'retract' (after an open, or after the last event). A
    gripper that closed before the task (its last event before task_start is
    a close) holds at the start of the task.
    """
    inside = [e for e in events if task_start < e['t'] < task_end]
    before = [e for e in events if e['t'] <= task_start]
    carried = 'close' if before and before[-1]['kind'] == 'close' else None
    spans = list()
    if task_start > start:
        spans.append({'start': start, 'end': task_start, 'phase': 'reset'})
    cuts = [task_start] + [e['t'] for e in inside] + [task_end]
    for k, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        if b <= a:
            continue
        nxt = inside[k] if k < len(inside) else None
        state = inside[k - 1]['kind'] if k > 0 else carried
        if nxt is not None and nxt['kind'] == 'close':
            phase = 'approach'
        elif state == 'close':
            phase = 'hold'
        else:
            phase = 'retract' if k > 0 else 'approach'
        spans.append({'start': a, 'end': b, 'phase': phase})
    if end > task_end:
        spans.append({'start': task_end, 'end': end, 'phase': 'reset'})
    return spans


def frame_times_for_review(task_start: float, task_end: float, events: Dict[str, List[dict]],
        max_frames: int = 12) -> List[Tuple[float, str]]:
    """
    Head video times to show for the visual check: task start and end
    (always), every gripper event given (the whole episode's; evenly thinned
    when there are more than max_frames - 2), filled up with evenly spaced
    task times. Returns sorted [(time, reason)], at most max_frames.
    """
    picks = list()
    for arm, arm_events in events.items():
        for e in arm_events:
            note = (" (fingers stop open: holding)" if e.get('holding') else
                " (fingers nearly closed: empty or a thin object)" if e['kind'] == 'close' else '')
            picks.append((e['t'], f"{arm} gripper {e['kind']}{note}"))
    picks = sorted(picks)
    room = max(0, max_frames - 2)
    if len(picks) > room:
        keep = np.unique(np.round(np.linspace(0, len(picks) - 1, room)).astype(int)) if room else []
        picks = [picks[i] for i in keep]
    picks += [(task_start, 'task start'), (task_end, 'task end')]
    n_fill = max_frames - len(picks)
    if n_fill > 0:
        for t in np.linspace(task_start, task_end, n_fill + 2)[1:-1]:
            if min(abs(t - p[0]) for p in picks) > 0.5:
                picks.append((float(t), 'during the task'))
    return sorted(picks)


# ---------------------------------------------------------------- visual check
# Enums of the Odyssey spec v0.6 (1.7 / 2.7 / 2.9) the model answers in.
TASK_FAMILIES = ['grasp', 'place', 'stack', 'sort', 'insert', 'bin_pick', 'tool_use', 'pour', 'fold', 'wipe',
    'assemble', 'open_close', 'handover', 'restock', 'pack']
OUTCOMES = ['success', 'failure', 'partial', 'aborted', 'unknown']
CONTRIBUTIONS = ['advancing', 'idle', 'wasteful']
ISSUE_CATEGORIES = ['dropped_object', 'incomplete_task', 'idle_stretch', 'other']
SEVERITIES = ['low', 'medium', 'high']
CAMERAS = ['head', 'right_wrist', 'left_wrist']
OBJECT_CLASSES = ['rigid', 'deformable', 'articulated', 'granular', 'liquid', 'tool']
DIFFICULTIES = ['single_step', 'multi_step', 'long_horizon']
FAILURE_KINDS = ['missed_grasp', 'drop', 'slip', 'collision', 'wrong_object', 'timeout', 'other']


def measured_items(start: float, end: float, task_start: float, task_end: float,
        arm_events: Dict[str, List[dict]]) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """
    Ids for what the model names: spans '<arm>_<k>' over the whole episode
    (arm_spans) and gripper events '<arm>_<close|open>_<k>' of both arms over
    the whole episode ('in_task': inside the task interval; a release can
    end just after the last bang, a grasp happen during the undocking).
    """
    spans, events = dict(), dict()
    for arm, ev in arm_events.items():
        for k, s in enumerate(arm_spans(start, end, task_start, task_end, ev)):
            spans[f"{arm}_{k + 1}"] = {**s, 'arm': arm}
        count = {'close': 0, 'open': 0}
        for e in ev:
            if start <= e['t'] < end:
                count[e['kind']] += 1
                events[f"{arm}_{e['kind']}_{count[e['kind']]}"] = {**e, 'arm': arm,
                    'in_task': bool(task_start < e['t'] < task_end)}
    return spans, events


def response_schema(span_ids: Sequence[str], event_ids: Sequence[str]) -> dict:
    """
    JSON schema of the model's answer (structured output: every property
    required, no extra ones). Times it gives are seconds from the episode start.
    """
    string, number, boolean = {'type': 'string'}, {'type': 'number'}, {'type': 'boolean'}

    def enum(values):
        return {'type': 'string', 'enum': list(values) or ['none']}

    def array(properties):
        return {'type': 'array', 'items': obj(properties)}

    def obj(properties):
        return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}

    return obj({
        'task': obj({
            'task': string, 'skill': enum(TASK_FAMILIES), 'difficulty': enum(DIFFICULTIES),
            'objects': {'type': 'array', 'items': string}, 'main_object': string, 'success_predicate': string, 'outcome': enum(OUTCOMES), 'completed_at_s': number,
            'instruction_variants': {'type': 'array', 'items': string}, 'confidence': number}),
        'spans': array({'span_id': enum(span_ids), 'verb': string, 'object': string, 'destination': string,
            'contribution': enum(CONTRIBUTIONS), 'progress': number, 'confidence': number}),
        'gripper_events': array({'event_id': enum(event_ids), 'label': string, 'object': string,
            'grasped_or_released': boolean, 'outcome': enum(['success', 'failure']), 'note': string}),
        'state_changes': array({'t_s': number, 'object': string, 'from': string, 'to': string, 'seen_in': enum(CAMERAS)}),
        'recoveries': array({'failure_t_s': number, 'failure': string, 'failure_kind': enum(FAILURE_KINDS),
            'recovered': boolean, 'recovered_at_s': number, 'correction': string}),
        'issues': array({'t_s': number, 'category': enum(ISSUE_CATEGORIES), 'severity': enum(SEVERITIES), 'evidence': string}),
        'objects': array({'name': string, 'description': string, 'object_class': enum(OBJECT_CLASSES)}),
        'scene_graph': array({'t_s': number, 'relations': {'type': 'array', 'items': string}}),
    })


def request_text(start: float, end: float, task_start: float, task_end: float,
        spans: Dict[str, dict], events: Dict[str, dict], task_prompt: Optional[str]) -> str:
    """
    The measured part of the request: task interval, spans to name, gripper events.
    Times in seconds from the episode start.
    """
    lines = [f"Episode length {end - start:.2f} s. The task part runs {task_start - start:.2f}-{task_end - start:.2f} s; "
        "before and after it the grippers leave / return to the dock and may be knocked together near the head "
        "camera: those spans are fixed as reset and not yours to name."]
    if task_prompt:
        lines.append(f'The operator was given this task: "{task_prompt}".')
    lines.append("Spans to name (span_id: arm, measured phase, start-end s):")
    lines += [f"  {i}: {s['arm']}, {s['phase']}, {s['start'] - start:.2f}-{s['end'] - start:.2f}"
        for i, s in spans.items() if s['phase'] != 'reset'] or ["  none"]
    lines.append("Gripper events (event_id: arm, time s, finger opening at the end of the motion):")
    for i, e in events.items():
        note = ''
        if e['kind'] == 'close':
            note = (' (stopped while open: something between the fingers)' if e['holding'] else
                f" (closed to under {DEFAULTS['holding_width'] * 1000:.0f} mm: empty, or a thin object such as a sheet or an edge)")
        where = '' if e.get('in_task', True) else ' [outside the task part: during docking / bang]'
        lines.append(f"  {i}: {e['arm']}, {e['t'] - start:.2f}, {e['width'] * 1000:.0f} mm{note}{where}")
    if not events:
        lines.append("  none")
    lines.append("For each gripper event, grasped_or_released says whether a close grasped an object / an open let "
        "one go. The frames follow, each after its label.")
    return '\n'.join(lines)


def bang_key_events(bangs: Sequence[float], start: float, end: float, task_start: float, task_end: float) -> List[dict]:
    """
    Key events (kind 'phase') of the bangs in [start, end) (head video
    seconds, episodes.json): the grippers knocked together near the head
    camera, found as a joint wrist IMU spike confirmed by the head audio
    click. The ritual bangs bound the task (03t_trim_docking.py), a bang
    inside the task is labeled as such. Times in seconds from the episode start.
    """
    eps = 1e-6
    out = list()
    for b in sorted(bangs):
        if not start <= b < end:
            continue
        where = 'before the task' if b <= task_start + eps else 'after the task' if b >= task_end - eps else 'during the task'
        out.append({'t_s': b - start, 'label': f"bang {where}: grippers knocked together", 'kind': 'phase',
            'outcome': None, 'note': 'measured: joint wrist IMU spike confirmed by the head audio click',
            'arm': 'both', 'measured_kind': 'bang'})
    return out


def task_objects(task: dict, events: Sequence[dict], key_events: Sequence[dict], state_changes: Sequence[dict]) -> list:
    """
    The task's objects: those the arms act on or put things into (span object / destination,
    grasped or released objects, objects whose state changes), in the model's order. Objects
    only seen in the scene stay in the objects layer. All of them if none is acted on.
    """
    used = {x for e in events for x in (e.get('object'), e.get('destination'))} \
        | {k.get('object') for k in key_events} | {c.get('object') for c in state_changes}
    acted = [o for o in task['objects'] if o in used]
    return acted or list(task['objects'])


def merge_annotation(answer: dict, start: float, task_start: float, task_end: float,
        spans: Dict[str, dict], events: Dict[str, dict], bangs: Sequence[float] = (), end: Optional[float] = None) -> dict:
    """
    Spec 2.7 layers from the model's answer and the measured items: times and
    arms always the measured ones, words from the model. Spans the model did
    not name keep empty words; reset spans are labeled 'reset' (idle). Bangs
    (head video seconds) in the episode become 'phase' key events
    (bang_key_events; end defaults to the end of the last span).
    Times in seconds from the episode start.
    """
    named = {s['span_id']: s for s in answer['spans']}
    labeled = {x['event_id']: x for x in answer['gripper_events']}
    task = answer['task']
    out_events = list()
    for i, s in spans.items():
        m = {} if s['phase'] == 'reset' else named.get(i, {})
        out_events.append({
            'start_s': s['start'] - start, 'end_s': s['end'] - start, 'arm': s['arm'],
            'verb': 'reset' if s['phase'] == 'reset' else m.get('verb', ''),
            'object': m.get('object', ''), 'destination': m.get('destination', ''),
            'contribution': 'idle' if s['phase'] == 'reset' else m.get('contribution'),
            'progress': m.get('progress'), 'confidence': m.get('confidence'),
            'measured_phase': s['phase']})
    key_events = list()
    for i, e in events.items():
        m = labeled.get(i, {})
        # a contact key event only where the gripper grasped or released something (the model's
        # reading of the frames, else the measured hold); closing / opening on nothing is no contact
        # (it stays in the arm's spans)
        touched = m.get('grasped_or_released')
        if touched is None:
            touched = bool(e.get('holding')) if e['kind'] == 'close' else None
        if touched is False:
            continue
        key_events.append({
            't_s': e['t'] - start, 'label': m.get('label') or f"{e['arm']} gripper {e['kind']}s",
            'kind': 'contact', 'outcome': m.get('outcome'), 'note': m.get('note', ''), 'arm': e['arm'],
            'object': m.get('object', ''), 'grasped_or_released': m.get('grasped_or_released'),
            'measured_kind': e['kind'], 'measured_t_begin_s': e['t_begin'] - start,
            'measured_width_mm': e['width'] * 1000, 'measured_holding': e.get('holding')})
    completed = task['completed_at_s'] if task['outcome'] == 'success' else None
    if completed is not None:
        key_events.append({'t_s': completed, 'label': 'task complete', 'kind': 'subgoal_complete',
            'outcome': 'success', 'note': task['success_predicate']})
    if end is None:
        end = max((s['end'] for s in spans.values()), default=task_end)
    key_events += bang_key_events(bangs, start, end, task_start, task_end)
    key_events.sort(key=lambda x: x['t_s'])
    return {
        'tasks': [{
            'start_s': task_start - start, 'end_s': task_end - start, 'task': task['task'], 'skill': task['skill'],
            'actions': sorted({e['verb'] for e in out_events if e['verb'] and e['verb'] != 'reset'}),
            'objects': task_objects(task, out_events, key_events, answer['state_changes']),
            'outcome': task['outcome'], 'success_predicate': task['success_predicate'],
            'completed_at_s': completed, 'instruction_variants': task['instruction_variants'],
            'confidence': task['confidence'],
            # coverage dimensions (spec 2.9); None in answers written before they were asked
            'difficulty': task.get('difficulty'), 'main_object': task.get('main_object')}],
        'events': out_events,
        'key_events': key_events,
        'state_changes': answer['state_changes'],
        'recoveries': answer['recoveries'],
        'issues': answer['issues'],
        'objects': answer['objects'],
        'scene_graph': answer['scene_graph'],
    }


# ---------------------------------------------------------------- vocabulary across episodes
# The visual check labels each episode on its own, so the same object / action /
# task can come back in different words. A text-only pass over all labels of
# the takes groups the strings that name the same thing (vocabulary); the
# grouping is applied here in code, so a label can only become another label
# that was in the same group.
UNNAMED = {'', 'unknown', 'none', 'n/a'}


def _named(s) -> bool:
    return isinstance(s, str) and s.strip().lower() not in UNNAMED


def label_inventory(answers: Sequence[dict]) -> dict:
    """
    Distinct strings of the model answers per category with their counts:
    'objects' (objects and places: task objects, span object / destination,
    gripper event objects, state change objects, object names, with the
    descriptions seen), 'phrases' (span verbs, gripper event labels), 'tasks'
    (task sentences with skill, difficulty, success predicate and objects
    seen). Object classes and difficulties seen go along as context.
    """
    objects, phrases, tasks = dict(), dict(), dict()

    def add(table, s, **context):
        if not _named(s):
            return
        entry = table.setdefault(s, {'count': 0, **{k: list() for k in context}})
        entry['count'] += 1
        for k, v in context.items():
            for x in (v if isinstance(v, list) else [v]):
                if _named(x) and x not in entry[k]:
                    entry[k].append(x)

    for a in answers:
        task = a['task']
        add(tasks, task['task'], skill=task['skill'], difficulty=task.get('difficulty'),
            success_predicate=task['success_predicate'], objects=task['objects'])
        for s in task['objects']:
            add(objects, s, descriptions=[], object_class=[])
        for s in a['spans']:
            add(phrases, s['verb'])
            add(objects, s['object'], descriptions=[], object_class=[])
            add(objects, s['destination'], descriptions=[], object_class=[])
        for x in a['gripper_events']:
            add(phrases, x['label'])
            add(objects, x['object'], descriptions=[], object_class=[])
        for x in a['state_changes']:
            add(objects, x['object'], descriptions=[], object_class=[])
        for x in a['objects']:
            add(objects, x['name'], descriptions=x['description'], object_class=x.get('object_class'))
    return {'objects': objects, 'phrases': phrases, 'tasks': tasks}


def vocabulary_schema(inventory: dict) -> dict:
    """
    Structured output of the vocabulary pass: groups per category, each with a
    canonical label and the inventory strings it covers (enum: only strings
    that exist).
    """
    string = {'type': 'string'}

    def obj(properties):
        return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}

    def aliases(category):
        return {'type': 'array', 'items': {'type': 'string', 'enum': list(inventory[category]) or ['none']}}

    return obj({
        'objects': {'type': 'array', 'items': obj({'name': string, 'description': string,
            'object_class': {'type': 'string', 'enum': OBJECT_CLASSES}, 'aliases': aliases('objects')})},
        'phrases': {'type': 'array', 'items': obj({'phrase': string, 'aliases': aliases('phrases')})},
        'tasks': {'type': 'array', 'items': obj({'task': string, 'skill': {'type': 'string', 'enum': TASK_FAMILIES},
            'difficulty': {'type': 'string', 'enum': DIFFICULTIES},
            'success_predicate': string, 'instruction_variants': {'type': 'array', 'items': string},
            'aliases': aliases('tasks')})},
    })


def vocabulary_maps(vocabulary: dict, inventory: dict) -> Tuple[dict, List[str]]:
    """
    String -> canonical entry per category. Inventory strings in no group map to
    themselves; a string in two groups keeps the first. Returns (maps, warnings).
    """
    keys = {'objects': 'name', 'phrases': 'phrase', 'tasks': 'task'}
    maps, warnings = dict(), list()
    for category, key in keys.items():
        m = dict()
        for entry in vocabulary.get(category, []):
            for s in [entry[key]] + list(entry['aliases']):
                if s in m and m[s][key] != entry[key]:
                    warnings.append(f"{category}: '{s}' in two groups ('{m[s][key]}', '{entry[key]}'), kept the first")
                    continue
                m[s] = entry
        for s in inventory.get(category, {}):
            if s not in m:
                warnings.append(f"{category}: '{s}' not in the vocabulary, kept as is")
        maps[category] = m
    return maps, warnings


def apply_vocabulary(answer: dict, maps: dict) -> dict:
    """
    The model answer with every object, phrase and task replaced by its
    canonical label; a task takes the group's skill, difficulty, success
    predicate and instruction variants, an object the group's description and
    object class.
    """
    import copy
    a = copy.deepcopy(answer)

    def name(s):
        return maps['objects'][s]['name'] if s in maps['objects'] else s

    def phrase(s):
        return maps['phrases'][s]['phrase'] if s in maps['phrases'] else s

    def unique(xs):
        return list(dict.fromkeys(xs))

    t = a['task']
    if t['task'] in maps['tasks']:
        g = maps['tasks'][t['task']]
        t.update({'task': g['task'], 'skill': g['skill'], 'success_predicate': g['success_predicate'],
            'instruction_variants': list(g['instruction_variants'])})
        if 'difficulty' in g:
            t['difficulty'] = g['difficulty']
    t['objects'] = unique(name(s) for s in t['objects'])
    if t.get('main_object') is not None:
        t['main_object'] = name(t['main_object'])
    for s in a['spans']:
        s.update({'verb': phrase(s['verb']), 'object': name(s['object']), 'destination': name(s['destination'])})
    for x in a['gripper_events']:
        x.update({'label': phrase(x['label']), 'object': name(x['object'])})
    for x in a['state_changes']:
        x['object'] = name(x['object'])
    objects = dict()
    for x in a['objects']:
        n = name(x['name'])
        if n not in objects:
            g = maps['objects'].get(x['name'])
            objects[n] = {'name': n, 'description': g['description'] if g else x['description'],
                'object_class': g.get('object_class') if g else x.get('object_class')}
    a['objects'] = list(objects.values())
    return a


# ---------------------------------------------------------------- delivery
SPEC_LAYERS = ['schema_version', 'episode_id', 'annotation_source', 'annotation_model', 'prompt_mode', 'given_prompt',
    'tasks', 'events', 'key_events', 'state_changes', 'recoveries', 'issues', 'objects', 'scene_graph', 'created_ts_utc']


def find_annotation(annotations: Sequence[dict], head_video: str, task_start: float, task_end: float) -> Optional[dict]:
    """
    The annotation (08 / 09, per dock-to-dock episode) of the head video whose measured task
    interval overlaps the given one (head video seconds) most; None if none overlaps.
    """
    best, best_overlap = None, 0.0
    for a in annotations:
        if a.get('source', {}).get('head_video') != head_video:
            continue
        m = a['measured']
        overlap = min(m['task_end'], task_end) - max(m['task_start'], task_start)
        if overlap > best_overlap:
            best, best_overlap = a, overlap
    return best


def delivery_annotation(annotation: dict, first_frame_time: float, duration: float, episode_id: str) -> dict:
    """
    annotations.json (spec 2.7) of a delivered episode from the annotation of the dock-to-dock
    episode it was cut from: times moved from that episode's start (measured start, head video
    seconds) to the delivered episode's first frame (first_frame_time, head video seconds) and
    kept inside [0, duration): spans clipped, moments outside dropped. Only the spec layers.
    """
    shift = annotation['measured']['start'] - first_frame_time
    inside = lambda t: t is not None and 0.0 <= t + shift < duration
    moved = lambda t: None if t is None else t + shift

    def clip(item, a='start_s', b='end_s'):
        s, e = max(item[a] + shift, 0.0), min(item[b] + shift, duration)
        return None if e <= s else {**item, a: s, b: e}

    out = {k: annotation.get(k) for k in SPEC_LAYERS}
    out['episode_id'] = episode_id
    tasks = list()
    for t in annotation['tasks']:
        c = clip(t)
        if c is not None:
            c['completed_at_s'] = moved(t['completed_at_s']) if inside(t['completed_at_s']) else None
            tasks.append(c)
    out['tasks'] = tasks
    out['events'] = [c for c in (clip(e) for e in annotation['events']) if c is not None]
    out['key_events'] = [{**k, 't_s': k['t_s'] + shift,
        **({'measured_t_begin_s': k['measured_t_begin_s'] + shift} if 'measured_t_begin_s' in k else {})}
        for k in annotation['key_events'] if inside(k['t_s'])]
    out['state_changes'] = [{**x, 't_s': x['t_s'] + shift} for x in annotation['state_changes'] if inside(x['t_s'])]
    out['issues'] = [{**x, 't_s': x['t_s'] + shift} for x in annotation['issues'] if inside(x['t_s'])]
    out['scene_graph'] = [{**x, 't_s': x['t_s'] + shift} for x in annotation['scene_graph'] if inside(x['t_s'])]
    out['recoveries'] = [{**x, 'failure_t_s': x['failure_t_s'] + shift, 'recovered_at_s': moved(x.get('recovered_at_s'))}
        for x in annotation['recoveries'] if inside(x['failure_t_s'])]
    return out
