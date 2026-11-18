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


def load_delivery_config(path: Optional[str]) -> Dict:
    """
    The delivery yaml, checked: known session keys only, collection_mode and
    scene_tags from the spec's vocabulary. None -> nothing declared.
    """
    if path is None:
        return {'sessions': dict()}
    config = yaml.safe_load(pathlib.Path(path).expanduser().open('r')) or dict()
    config.setdefault('sessions', dict())
    for name, s in config['sessions'].items():
        s = s or dict()
        unknown = set(s) - set(SESSION_KEYS)
        assert not unknown, f"session {name}: unknown keys {sorted(unknown)}, known {SESSION_KEYS}"
        mode = s.get('collection_mode')
        assert mode is None or mode in COLLECTION_MODES, f"session {name}: collection_mode {mode} not in {COLLECTION_MODES}"
        tags = s.get('scene_tags') or []
        assert isinstance(tags, list) and all(t in SCENES for t in tags), \
            f"session {name}: scene_tags {tags} must be a list from {SCENES}"
        config['sessions'][name] = {k: s.get(k) for k in SESSION_KEYS}
    return config


def session_declarations(config: Dict, session: pathlib.Path) -> Dict:
    """
    Declarations of one session (by directory name), all keys present, None where not declared.
    """
    return config['sessions'].get(session.name) or {k: None for k in SESSION_KEYS}
