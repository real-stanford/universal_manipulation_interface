"""
Delivery declarations (Odyssey spec 2.3 / 2.4 / 2.9): what only the people
who recorded the data know, per delivery and per session (session directory
name). A key left null is declared as not known and exported as null, never
filled in. See example/tag_tracking/delivery.yaml.
"""
from typing import Dict, Optional
import pathlib
import yaml

COLLECTION_MODES = ['scripted', 'freeform', 'adversarial', 'in_the_wild']
SCENES = ['tabletop', 'kitchen', 'workbench', 'warehouse', 'retail', 'industrial_line', 'household', 'hospitality',
    'laboratory', 'outdoor', 'other']
SESSION_KEYS = ['collection_mode', 'scene_tags', 'environment_id', 'operator_id', 'station_id', 'consent_reference']
# per session, the task (13_export_delivery.py; the review viewer keeps them)
TASK_FAMILIES = ['grasp', 'place', 'stack', 'sort', 'insert', 'bin_pick', 'tool_use', 'pour', 'fold', 'wipe',
    'assemble', 'open_close', 'handover', 'restock', 'pack']
OBJECT_CLASSES = ['rigid', 'deformable', 'articulated', 'granular', 'liquid', 'tool']
TASK_KEYS = ['task_text', 'task_family', 'object_tags']
# per delivery (spec 2.3 / 2.4), 13_export_delivery.py
DELIVERY_KEYS = ['schema_version', 'vendor_name', 'dataset_name', 'delivery_id', 'robot_model', 'gripper_type',
    'source_uri_prefix', 'license_reference', 'anonymization_state', 'consent_reference']
ANONYMIZATION_STATES = ['faces_blurred', 'not_required', 'pending', 'other']


def load_delivery_config(path: Optional[str]) -> Dict:
    """
    The delivery yaml, checked: known session keys only, collection_mode and
    scene_tags from the spec's vocabulary. None -> nothing declared.
    """
    if path is None:
        return {'sessions': dict(), 'delivery': {k: None for k in DELIVERY_KEYS}}
    config = yaml.safe_load(pathlib.Path(path).expanduser().open('r')) or dict()
    config.setdefault('sessions', dict())
    delivery = config.get('delivery') or dict()
    unknown = set(delivery) - set(DELIVERY_KEYS)
    assert not unknown, f"delivery: unknown keys {sorted(unknown)}, known {DELIVERY_KEYS}"
    state = delivery.get('anonymization_state')
    assert state is None or state in ANONYMIZATION_STATES, \
        f"delivery: anonymization_state {state} not in {ANONYMIZATION_STATES}"
    config['delivery'] = {k: delivery.get(k) for k in DELIVERY_KEYS}
    for name, s in config['sessions'].items():
        s = s or dict()
        unknown = set(s) - set(SESSION_KEYS) - set(TASK_KEYS)
        assert not unknown, f"session {name}: unknown keys {sorted(unknown)}, known {SESSION_KEYS + TASK_KEYS}"
        family = s.get('task_family')
        assert family is None or family in TASK_FAMILIES, f"session {name}: task_family {family} not in {TASK_FAMILIES}"
        objects = s.get('object_tags') or []
        assert isinstance(objects, list) and all(t in OBJECT_CLASSES for t in objects), \
            f"session {name}: object_tags {objects} must be a list from {OBJECT_CLASSES}"
        mode = s.get('collection_mode')
        assert mode is None or mode in COLLECTION_MODES, f"session {name}: collection_mode {mode} not in {COLLECTION_MODES}"
        tags = s.get('scene_tags') or []
        assert isinstance(tags, list) and all(t in SCENES for t in tags), \
            f"session {name}: scene_tags {tags} must be a list from {SCENES}"
        config['sessions'][name] = {k: s.get(k) for k in SESSION_KEYS + TASK_KEYS}
    return config


def session_declarations(config: Dict, session: pathlib.Path) -> Dict:
    """
    Declarations of one session (by directory name), all keys present, None where not declared.
    """
    return config['sessions'].get(session.name) or {k: None for k in SESSION_KEYS + TASK_KEYS}
