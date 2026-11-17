"""
Detect corners of all tags with a configured size in every video of the
observing (head) camera of a session.
Videos are processed one at a time, each in parallel time chunks on all cores.
Writes tag_corners.pkl next to each raw_video.mp4; videos that already have
one are skipped unless --overwrite (needed after changing the tag sizes/ids in
the aruco config).

python scripts_tag_tracking/00_detect_tag_corners.py -i data_workspace/session -c umi_calib/tag_tracking.yaml
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
import pickle
import multiprocessing
from umi.common.exiftool_util import ExifTool
from umi.tag_tracking.config import load_tag_tracking_config
from umi.tag_tracking.tag_detection import detect_video_tag_corners


# %%
def detect_and_save(video_path, config_path, num_workers):
    config = load_tag_tracking_config(config_path)
    tag_ids = config['known_tag_ids']
    results = detect_video_tag_corners(str(video_path), config['aruco_dict'], tag_ids, num_workers)
    output = pathlib.Path(video_path).parent.joinpath('tag_corners.pkl')
    pickle.dump(results, output.open('wb'))
    return output


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Session directory')
@click.option('-c', '--config', required=True, help='Tag tracking config yaml')
@click.option('-ow', '--overwrite', is_flag=True, default=False, help='Re-detect videos that have tag_corners.pkl')
@click.option('-n', '--num_workers', type=int, default=None, help='Worker processes per video, default all cores')
def main(input, config, overwrite, num_workers):
    session = pathlib.Path(os.path.expanduser(input)).absolute()
    config_path = str(pathlib.Path(os.path.expanduser(config)).absolute())
    head_camera = load_tag_tracking_config(config_path)['head_camera']
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    video_paths = list()
    with ExifTool() as et:
        for video_path in sorted(session.glob('demos/*/raw_video.mp4')):
            meta = list(et.get_metadata(str(video_path)))[0]
            if meta['QuickTime:CameraSerialNumber'] != head_camera:
                continue
            if video_path.parent.joinpath('tag_corners.pkl').is_file() and not overwrite:
                print(f"tag_corners.pkl already exists, skipping {video_path.parent.name}")
                continue
            video_paths.append(video_path)
    print(f"Detecting tags in {len(video_paths)} videos of camera {head_camera}")

    # one video at a time, each split into time chunks on all workers
    for k, video_path in enumerate(video_paths):
        print(f"[{k + 1}/{len(video_paths)}] {video_path.parent.name}")
        print(f"Saved {detect_and_save(video_path, config_path, num_workers)}")

# %%
if __name__ == "__main__":
    main()
