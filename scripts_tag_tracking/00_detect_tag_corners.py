"""
Detect tag corners in every video of the observing (head) camera of a session.
Writes tag_corners.pkl next to each raw_video.mp4.

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
import concurrent.futures
from exiftool import ExifToolHelper
from umi.tag_tracking.config import load_tag_tracking_config
from umi.tag_tracking.tag_detection import detect_video_tag_corners


# %%
def detect_and_save(video_path, config_path):
    config = load_tag_tracking_config(config_path)
    tag_ids = set(config['tag_sizes'].keys())
    results = detect_video_tag_corners(str(video_path), config['aruco_dict'], tag_ids)
    output = pathlib.Path(video_path).parent.joinpath('tag_corners.pkl')
    pickle.dump(results, output.open('wb'))
    return output


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Session directory')
@click.option('-c', '--config', required=True, help='Tag tracking config yaml')
@click.option('-n', '--num_workers', type=int, default=None)
def main(input, config, num_workers):
    session = pathlib.Path(os.path.expanduser(input)).absolute()
    config_path = str(pathlib.Path(os.path.expanduser(config)).absolute())
    head_camera = load_tag_tracking_config(config_path)['head_camera']
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    video_paths = list()
    with ExifToolHelper() as et:
        for video_path in sorted(session.glob('demos/*/raw_video.mp4')):
            meta = list(et.get_metadata(str(video_path)))[0]
            if meta['QuickTime:CameraSerialNumber'] == head_camera:
                video_paths.append(video_path)
    print(f"Found {len(video_paths)} videos of camera {head_camera}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(detect_and_save, x, config_path) for x in video_paths]
        for future in concurrent.futures.as_completed(futures):
            print(f"Saved {future.result()}")

# %%
if __name__ == "__main__":
    main()
