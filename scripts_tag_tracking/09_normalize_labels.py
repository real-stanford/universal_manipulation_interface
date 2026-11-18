"""
One vocabulary for the annotations of one or more sessions: the visual check
(08_annotate_episodes.py) labels every episode on its own, so the same object,
action or task comes back in different words ("blue cup", "the blue mug").
A text-only pass by Claude over all distinct labels groups the ones naming the
same thing under one canonical label; labels that differ in meaning (another
object, colour, size, destination or goal) stay apart. The grouping is applied
in code (umi.tag_tracking.annotation.apply_vocabulary): a label can only be
replaced by the canonical label of its own group, and every group lists only
labels that occur.

Reads <session>/tag_tracking/annotations/*_ep*.json of every -i session, writes
the vocabulary to --vocabulary (default: the first session's
annotations/vocabulary.json, edit it by hand if a grouping is wrong) and rebuilds
the layers of every annotation from its stored model answer and measured items,
so rerunning starts from the original labels each time.
An existing vocabulary is given to the model so its canonical labels stay the
same when episodes are added. --apply_only applies the vocabulary file as it
is (after editing it), no API call. --dry_run writes the request text only.

python scripts_tag_tracking/09_normalize_labels.py -i data_workspace/session_a -i data_workspace/session_b
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import datetime
import json
import pathlib
import click
from umi.tag_tracking.annotation import (label_inventory, vocabulary_schema, vocabulary_maps, apply_vocabulary,
    merge_annotation)

SYSTEM = """You make the labels of a manipulation dataset consistent. The labels were written per episode \
by a model looking at frames, so the same thing is sometimes named in different words. You get every \
distinct label with how often it occurs and its context, in three categories: objects (objects and places), \
phrases (short action phrases and event labels) and tasks (task sentences).

Group the labels of each category that name the same thing, and give each group one canonical label:
- Same thing in other words goes in one group: "blue cup", "the blue mug", "blue cup (mug)".
- A difference in meaning stays apart: another object, colour, size, material, count, place, direction, \
destination or goal ("small box" vs "large box", "place cup in bowl" vs "place cup on plate").
- When unsure, keep labels apart.
- Canonical labels are short, lower case, without articles, and prefer the most specific common wording.
- Phrases and tasks use the canonical object names.
- Every label of the input goes into exactly one group (a label alone is a group of one).
- If existing canonical labels are given, keep them as the canonical labels of their groups.
- A task group's instruction_variants are two to five other natural ways to say that task; its \
success_predicate is the visible end state; its skill the closest task family; its difficulty the one most \
episodes gave unless it is clearly wrong.
- An object group's object_class is the class most episodes gave, unless it is clearly wrong."""


def load_annotations(sessions):
    files = list()
    for s in sessions:
        d = pathlib.Path(os.path.expanduser(s)).absolute().joinpath('tag_tracking', 'annotations')
        files += sorted(p for p in d.glob('*_ep*.json'))
    return files


def request_text(inventory, existing):
    parts = list()
    if existing:
        parts.append("Existing canonical labels (keep these names):\n" + json.dumps(
            {c: [{k: v for k, v in e.items() if k != 'aliases'} for e in existing.get(c, [])]
             for c in ('objects', 'phrases', 'tasks')}, indent=1))
    parts.append("Objects and places (label: count, classes seen, descriptions seen):\n" + '\n'.join(
        f"- {s}: {x['count']}" + (f", classes {x['object_class']}" if x['object_class'] else '')
        + (f", {'; '.join(x['descriptions'])}" if x['descriptions'] else '')
        for s, x in inventory['objects'].items()))
    parts.append("Phrases (label: count):\n" + '\n'.join(f"- {s}: {x['count']}" for s, x in inventory['phrases'].items()))
    parts.append("Tasks (sentence: count, skills, difficulties, success predicates, objects seen):\n" + '\n'.join(
        f"- {s}: {x['count']}, skills {x['skill']}, difficulties {x['difficulty']}, predicates {x['success_predicate']}, objects {x['objects']}"
        for s, x in inventory['tasks'].items()))
    return '\n\n'.join(parts)


# %%
@click.command()
@click.option('-i', '--input', 'sessions', required=True, multiple=True, help='Session directory (repeat for several)')
@click.option('--vocabulary', default=None, help='Default <first session>/tag_tracking/annotations/vocabulary.json')
@click.option('--model', default='claude-opus-5-5')
@click.option('--effort', type=click.Choice(['low', 'medium', 'high', 'xhigh', 'max']), default='high')
@click.option('--apply_only', is_flag=True, default=False, help='Apply the vocabulary file as it is, no API call')
@click.option('--dry_run', is_flag=True, default=False, help='Write the request text only')
def main(sessions, vocabulary, model, effort, apply_only, dry_run):
    files = load_annotations(sessions)
    assert files, "no annotations, run 08_annotate_episodes.py first"
    vocabulary_path = pathlib.Path(os.path.expanduser(vocabulary)) if vocabulary else \
        pathlib.Path(os.path.expanduser(sessions[0])).absolute().joinpath('tag_tracking', 'annotations', 'vocabulary.json')
    annotations = [json.load(p.open('r')) for p in files]
    inventory = label_inventory([a['model_answer'] for a in annotations])
    print(f"{len(files)} episodes: {len(inventory['objects'])} object labels, {len(inventory['phrases'])} phrases, "
        f"{len(inventory['tasks'])} task sentences")
    existing = json.load(vocabulary_path.open('r')) if vocabulary_path.is_file() else None

    if apply_only:
        assert existing is not None, f"no vocabulary at {vocabulary_path}"
        vocab = existing
    else:
        text = request_text(inventory, existing)
        if dry_run:
            out = vocabulary_path.with_name('vocabulary_request.txt')
            out.write_text(SYSTEM + '\n\n' + text + '\n')
            print(f"Saved to {out}")
            return
        import anthropic
        client = anthropic.Anthropic()
        with client.beta.messages.stream(
            model=model,
            max_tokens=64000,
            betas=['server-side-fallback-2026-07-01'],
            fallbacks='default',
            thinking={'type': 'adaptive'},
            output_config={'effort': effort, 'format': {'type': 'json_schema', 'schema': vocabulary_schema(inventory)}},
            system=SYSTEM,
            messages=[{'role': 'user', 'content': text}],
        ) as stream:
            response = stream.get_final_message()
        print(f"{response.usage.input_tokens} input / {response.usage.output_tokens} output tokens")
        assert response.stop_reason == 'end_turn', f"stop reason {response.stop_reason} {response.stop_details or ''}"
        vocab = json.loads(next(b.text for b in response.content if b.type == 'text'))
        vocab['model'] = response.model
        vocab['created_ts_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        json.dump(vocab, vocabulary_path.open('w'), indent=2)
        print(f"Vocabulary saved to {vocabulary_path}")

    maps, warnings = vocabulary_maps(vocab, inventory)
    for w in warnings:
        print(f"  warning: {w}")
    for category, key in (('objects', 'name'), ('phrases', 'phrase'), ('tasks', 'task')):
        merged = [e for e in vocab.get(category, []) if len(set(e['aliases']) - {e[key]}) > 0]
        print(f"{category}: {len(inventory[category])} labels -> {len({e[key] for e in maps[category].values()})} canonical")
        for e in merged:
            print(f"  {e[key]} <- {', '.join(sorted(set(e['aliases']) - {e[key]}))}")

    for p, a in zip(files, annotations):
        m = a['measured']
        a.update(merge_annotation(apply_vocabulary(a['model_answer'], maps), m['start'], m['task_start'],
            m['task_end'], m['spans'], m['events']))
        a['label_normalization'] = {'vocabulary': str(vocabulary_path), 'model': vocab.get('model'),
            'applied_from_file': apply_only}
        json.dump(a, p.open('w'), indent=2)
    print(f"{len(files)} annotations rebuilt with the vocabulary")

# %%
if __name__ == "__main__":
    main()
