"""
Coverage report of one or more sessions (Odyssey spec 1.6 / 2.9): hours and
episode counts per task family, main object class, scene, environment,
operator, station, collection mode, difficulty and outcome, and the number of
distinct tasks. Episodes are those with a task interval (03t_trim_docking.py);
labels from tag_tracking/annotations (08, then 09 for consistent task
sentences), declarations from the delivery yaml (-d, see
example/tag_tracking/delivery.yaml). Hours are dock-to-dock episodes, task
hours the task part.

python scripts_tag_tracking/10_coverage_report.py -i data_workspace/session_a -i data_workspace/session_b -d delivery.yaml
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import json
import pathlib
import click
from umi.tag_tracking.delivery_config import load_delivery_config, session_declarations
from umi.tag_tracking.coverage import episode_coverage, coverage_report


# %%
@click.command()
@click.option('-i', '--input', 'sessions', required=True, multiple=True, help='Session directory (repeat for several)')
@click.option('-d', '--delivery', default=None, help='Delivery declarations yaml')
@click.option('-o', '--output', default=None, help='Also write the report as json')
def main(sessions, delivery, output):
    config = load_delivery_config(delivery)
    rows = list()
    for s in sessions:
        session = pathlib.Path(os.path.expanduser(s)).absolute()
        declarations = session_declarations(config, session)
        undeclared = [k for k, v in declarations.items() if v is None]
        if undeclared:
            print(f"{session.name}: not declared: {', '.join(undeclared)}")
        annotations = session.joinpath('tag_tracking', 'annotations')
        videos = json.load(session.joinpath('tag_tracking', 'episodes.json').open('r'))['videos']
        n = 0
        for head, video in videos.items():
            for k, e in enumerate(video['episodes']):
                if e.get('task_start') is None:
                    continue
                path = annotations.joinpath(f"{head}_ep{k + 1:03d}.json")
                annotation = json.load(path.open('r')) if path.is_file() else None
                rows.append(episode_coverage(annotation, declarations, e['end'] - e['start'], e['task_end'] - e['task_start']))
                n += 1
        print(f"{session.name}: {n} episodes, {sum(r['annotated'] for r in rows[-n:]) if n else 0} annotated")

    report = coverage_report(rows)
    print(f"\n{report['episodes']} episodes ({report['annotated']} annotated), {report['hours']:.2f} h, "
        f"task {report['task_hours']:.2f} h, {report['distinct_tasks']} distinct tasks")
    for task, count in report['tasks'].items():
        print(f"  {count:4d}  {task}")
    for dimension, values in report['by'].items():
        print(f"\n{dimension}")
        for value, x in values.items():
            print(f"  {value:<28s} {x['episodes']:4d} episodes  {x['hours']:6.2f} h  task {x['task_hours']:6.2f} h")
    if output:
        json.dump(report, open(output, 'w'), indent=2)
        print(f"\nSaved to {output}")

# %%
if __name__ == "__main__":
    main()
