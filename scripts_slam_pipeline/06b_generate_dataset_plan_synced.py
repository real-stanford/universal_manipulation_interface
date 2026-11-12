"""
Run 06_generate_dataset_plan.py with video start times corrected by
the offsets from 06a_estimate_audio_sync.py.
All other options are passed through to 06_generate_dataset_plan.py.

python scripts_slam_pipeline/06b_generate_dataset_plan_synced.py -i data_workspace/cup_in_the_wild/20240105_zhenjia_packard_2nd_conference_room
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import json
import datetime
import importlib.util
from umi.common.timecode_util import mp4_get_start_datetime


# %%
def load_dataset_plan_module():
    path = pathlib.Path(__file__).parent.joinpath('06_generate_dataset_plan.py')
    spec = importlib.util.spec_from_file_location('generate_dataset_plan', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def get_synced_start_datetime_func(video_offsets):
    """
    video_offsets: video dir name (demo_*) -> offset in seconds
    """
    def synced_mp4_get_start_datetime(mp4_path: str) -> datetime.datetime:
        start_date = mp4_get_start_datetime(mp4_path)
        offset = video_offsets.get(pathlib.Path(mp4_path).parent.name, 0.0)
        return start_date + datetime.timedelta(seconds=offset)
    return synced_mp4_get_start_datetime


# %%
@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option('-i', '--input', required=True, help='Project directory')
@click.option('-so', '--sync_offsets', default=None, help='Default <input>/audio_sync_offsets.json')
@click.pass_context
def main(ctx, input, sync_offsets):
    input_path = pathlib.Path(os.path.expanduser(input)).absolute()
    if sync_offsets is None:
        sync_offsets = input_path.joinpath('audio_sync_offsets.json')
    sync_offsets = pathlib.Path(os.path.expanduser(sync_offsets)).absolute()
    assert sync_offsets.is_file(), f"{sync_offsets} not found, run 06a_estimate_audio_sync.py first"

    video_offsets = dict()
    for name, result in json.load(sync_offsets.open('r'))['videos'].items():
        if result['offset_sec'] is None:
            print(f"Warning: {name} not synced ({result['status']}), using timecode as-is.")
            continue
        video_offsets[name] = result['offset_sec']
    print(f"Loaded audio sync offsets for {len(video_offsets)} videos")

    # 06 reads every video start time through this function
    plan_module = load_dataset_plan_module()
    plan_module.mp4_get_start_datetime = get_synced_start_datetime_func(video_offsets)
    plan_module.main.main(args=['--input', str(input_path)] + ctx.args)

# %%
if __name__ == "__main__":
    main()
