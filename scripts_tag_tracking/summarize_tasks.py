"""
One line per task (station, day, task folder) saying exactly what the task is, from the episode
descriptions of describe_mcaps.py, and a report of the episodes that do not fit.

Input: the CSV of describe_mcaps.py (columns task, file, duration_s, good, task_description,
objects, hands, outcome, notes). Rows are grouped by their task folder, e.g.
station3/2026-11-21/task01 -> station station3, day 2026-11-21, task task01 (folders without a
date: day empty).

Per task, Claude reads all its episode descriptions and answers (structured JSON): one line
describing the exact task (action, objects, from where to where), the objects, and the episodes
whose description does not match that task (another action or objects, nothing happening, a
different task). Besides, without Claude: episodes whose outcome is not 'success' and episodes
with notes.

Output:
- <output>_tasks.csv: station, day, task, episodes, minutes, good_episodes, task_line, objects,
  mismatching_episodes
- <output>_mismatches.csv: station, day, task, file, kind (description / outcome / notes), detail
Tasks already in <output>_tasks.csv are skipped (a stopped run continues).
--dry_run: print the request of the first task, no API call.

Needs the anthropic package and Anthropic credentials (ANTHROPIC_API_KEY or `ant auth login`).

python scripts_tag_tracking/summarize_tasks.py mcap_tasks.csv -o task_summary
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# %%
import collections
import csv
import json
import pathlib
import re
import click

SYSTEM = """You get short descriptions of the episodes of one recorded manipulation task: one
person at one station, repeating the same task with two handheld grippers. Each description was
written from a few camera frames of one episode, so details vary and some are vague. Your job is
to say what the task is, exactly, and which episodes do not fit it."""

PROMPT = """Task folder {task}: {n} episodes. Each line: episode file | description | objects seen.

{lines}

Write one line describing exactly this task (the action, the objects, from where to where), list
its objects, and list the episodes whose description does not match that task (a different action
or objects, nothing happening, another task). Vague or less detailed descriptions of the same task
are not mismatches."""

SCHEMA = {
    'type': 'object',
    'properties': {
        'task_line': {'type': 'string', 'description': 'One line: the exact task, its objects, from where to where'},
        'objects': {'type': 'array', 'items': {'type': 'string'}},
        'mismatches': {'type': 'array', 'items': {
            'type': 'object',
            'properties': {'file': {'type': 'string'}, 'reason': {'type': 'string'}},
            'required': ['file', 'reason'], 'additionalProperties': False}},
    },
    'required': ['task_line', 'objects', 'mismatches'],
    'additionalProperties': False,
}

TASK_COLUMNS = ['station', 'day', 'task', 'episodes', 'minutes', 'good_episodes', 'task_line', 'objects',
    'mismatching_episodes', 'model']
MISMATCH_COLUMNS = ['station', 'day', 'task', 'file', 'kind', 'detail']


def split_task(folder: str):
    """
    station, day, task of a task folder (station3/2026-11-21/task01, station1/task03, ...).
    """
    parts = [p for p in folder.split('/') if p and p != 'dataset_mcap']
    day = next((p for p in parts if re.fullmatch(r'\d{4}-\d{2}-\d{2}', p)), '')
    station = next((p for p in parts if p.startswith('station')), parts[0] if parts else '')
    task = parts[-1] if parts else folder
    return station, day, task


def request(task: str, rows: list) -> str:
    lines = '\n'.join(f"{r['file']} | {r['task_description']} | {r['objects']}" for r in rows)
    return PROMPT.format(task=task, n=len(rows), lines=lines)


def ask(client, model: str, effort: str, text: str):
    response = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        betas=['server-side-fallback-2026-07-01'],
        fallbacks='default',
        output_config={'effort': effort, 'format': {'type': 'json_schema', 'schema': SCHEMA}},
        system=SYSTEM,
        messages=[{'role': 'user', 'content': text}],
    )
    if response.stop_reason != 'end_turn':
        raise RuntimeError(f"stop reason {response.stop_reason} {response.stop_details or ''}")
    return json.loads(next(b.text for b in response.content if b.type == 'text')), response.model


# %%
@click.command()
@click.argument('descriptions')
@click.option('-o', '--output', required=True, help='Output prefix: <output>_tasks.csv, <output>_mismatches.csv')
@click.option('--model', default='claude-opus-5-5')
@click.option('--effort', default='medium', type=click.Choice(['low', 'medium', 'high']))
@click.option('--dry_run', is_flag=True, default=False, help='Print the request of the first task, no API call')
def main(descriptions, output, model, effort, dry_run):
    rows = list(csv.DictReader(open(os.path.expanduser(descriptions))))
    assert rows and 'task_description' in rows[0], f"{descriptions}: not a describe_mcaps.py CSV"
    groups = collections.OrderedDict()
    for r in sorted(rows, key=lambda r: (r['task'], r['file'])):
        groups.setdefault(r['task'], list()).append(r)
    tasks_csv = pathlib.Path(os.path.expanduser(f"{output}_tasks.csv"))
    mismatches_csv = pathlib.Path(os.path.expanduser(f"{output}_mismatches.csv"))
    done = set()
    if tasks_csv.is_file():
        done = {(r['station'], r['day'], r['task']) for r in csv.DictReader(tasks_csv.open())}
    todo = [(folder, group) for folder, group in groups.items() if split_task(folder) not in done]
    print(f"{len(rows)} episodes in {len(groups)} tasks, {len(groups) - len(todo)} already in {tasks_csv}, "
        f"{len(todo)} to do", flush=True)
    if dry_run:
        if todo:
            print(SYSTEM + '\n\n' + request(*todo[0]))
        return

    import anthropic
    client = anthropic.Anthropic()
    tasks_csv.parent.mkdir(parents=True, exist_ok=True)
    for path, columns in [(tasks_csv, TASK_COLUMNS), (mismatches_csv, MISMATCH_COLUMNS)]:
        if not path.is_file():
            with path.open('w', newline='') as f:
                csv.writer(f).writerow(columns)

    for folder, group in todo:
        station, day, task = split_task(folder)
        try:
            answer, answered_by = ask(client, model, effort, request(folder, group))
        except Exception as e:
            print(f"{folder}: FAILED {type(e).__name__}: {e} (rerun to retry)", flush=True)
            continue
        files = {r['file'] for r in group}
        mismatches = [m for m in answer['mismatches'] if m['file'] in files]
        mismatch_rows = [[station, day, task, m['file'], 'description', m['reason']] for m in mismatches]
        # without Claude: episodes not judged a success, episodes with notes
        for r in group:
            if r.get('outcome') and r['outcome'] != 'success':
                mismatch_rows.append([station, day, task, r['file'], 'outcome', r['outcome']])
            if r.get('notes'):
                mismatch_rows.append([station, day, task, r['file'], 'notes', r['notes']])
        minutes = sum(float(r['duration_s'] or 0) for r in group) / 60
        good = sum(r.get('good') == 'True' for r in group)
        with tasks_csv.open('a', newline='') as f:
            csv.writer(f).writerow([station, day, task, len(group), round(minutes, 1), good, answer['task_line'],
                '; '.join(answer['objects']), len({m['file'] for m in mismatches}), answered_by])
        with mismatches_csv.open('a', newline='') as f:
            csv.writer(f).writerows(mismatch_rows)
        print(f"{folder}: {answer['task_line']} ({len(group)} episodes, {len(mismatches)} not matching, "
            f"{len(mismatch_rows) - len(mismatches)} outcome / notes flags)", flush=True)
    print(f"saved {tasks_csv}\n      {mismatches_csv}")

# %%
if __name__ == "__main__":
    main()
