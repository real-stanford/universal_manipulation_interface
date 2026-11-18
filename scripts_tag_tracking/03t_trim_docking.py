"""
Remove the docking from dock-to-dock episodes (00b_segment_episodes.py,
boundary 'dock'): per episode, the task runs from the first head frame where
every cube is farther than dock_exit_distance from the nearest dock tag, to
the last such frame (umi.tag_tracking.episodes.task_interval). Picking the
grippers up from the dock and putting them back are outside it.

Cube positions: tracked world poses (tag_tracking.pkl, 03/03b), a cube not
tracked in a frame keeps its last known state. Dock tag positions: the
environment tag map (tag_tracking/bodies.json).
Adds 'task_start' / 'task_end' (head video seconds, or null if the cubes never
left the dock area) to every episode of tag_tracking/episodes.json;
06_export_dataset_plan.py exports that interval.

python scripts_tag_tracking/03t_trim_docking.py -i data_workspace/session -c umi_calib/tag_tracking.yaml
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
import pickle
import click
import numpy as np
from umi.tag_tracking.config import load_tag_tracking_config
from umi.tag_tracking.session import get_session_videos
from umi.tag_tracking.episodes import task_interval


def mmss(t):
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Session directory')
@click.option('-c', '--config', required=True, help='Tag tracking config yaml')
def main(input, config):
    session = pathlib.Path(os.path.expanduser(input)).absolute()
    config = load_tag_tracking_config(config)
    assert 'episodes' in config, "tag tracking config needs an episodes section"
    exit_distance = config['episodes']['dock_exit_distance']
    dock_tags = config['episodes']['dock_tags']
    calibration = json.load(session.joinpath('tag_tracking', 'bodies.json').open('r'))
    static = [b for b in calibration['bodies'].values() if b['static']][0]
    missing = [t for t in dock_tags if str(t) not in static['T_body_tags']]
    assert len(missing) < len(dock_tags), f"no dock tag {dock_tags} in the environment tag map"
    dock_positions = np.array([np.array(static['T_body_tags'][str(t)])[:3, 3] for t in dock_tags if str(t) in static['T_body_tags']])
    names = [name for name, b in config['bodies'].items() if 'gripper_camera' in b]
    episodes_path = session.joinpath('tag_tracking', 'episodes.json')
    data = json.load(episodes_path.open('r'))

    for head in [x for x in get_session_videos(session) if x['camera_serial'] == config['head_camera']]:
        track = pickle.load(head['dir'].joinpath('tag_tracking.pkl').open('rb'))
        times = track['time']
        positions = list()
        for name in names:
            body = track['bodies'][name]
            p = body['T_world_body'][:, :3, 3].copy()
            p[~body['is_valid_world']] = np.nan
            positions.append(p)
        video = data['videos'][head['name']]
        durations = list()
        for k, e in enumerate(video['episodes']):
            interval = task_interval(times, positions, dock_positions, e['start'], e['end'], exit_distance)
            e['task_start'], e['task_end'] = (None, None) if interval is None else interval
            if interval is None:
                print(f"  episode {k + 1} {mmss(e['start'])}-{mmss(e['end'])}: cubes never left the dock area, no task")
                continue
            durations.append(interval[1] - interval[0])
            print(f"  episode {k + 1} {mmss(e['start'])}-{mmss(e['end'])}: task {mmss(interval[0])}-{mmss(interval[1])} "
                f"({interval[1] - interval[0]:.1f}s; docking {interval[0] - e['start']:.1f}s + {e['end'] - interval[1]:.1f}s removed)")
        print(f"{head['name']}: {len(durations)} of {len(video['episodes'])} episodes with a task, "
            + (f"{min(durations):.1f}-{max(durations):.1f}s, total {sum(durations):.0f}s" if durations else 'none'))

    data['parameters']['dock_exit_distance'] = exit_distance
    json.dump(data, episodes_path.open('w'), indent=2)
    print(f"Saved to {episodes_path}")

# %%
if __name__ == "__main__":
    main()
