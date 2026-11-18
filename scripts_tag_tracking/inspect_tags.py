"""
Which ArUco tags each video shows, to set up a new station (cameras, cubes,
finger tags or environment changed): per video its camera serial, length
and, per tag id, the share of sampled frames it is in, its median side length
(px) and median image position (0..1, x right, y down), and whether the id
has a size in the aruco config. Needs no tag tracking config.

Reading the output:
- wrist video (gripper calibration or take): the two finger tags are in most
  frames at a fixed spot near the bottom, left and right of the center
  (left_finger_tag_id / right_finger_tag_id of gripper_config.yaml);
- head mapping: the environment tags (bodies.environment.tags);
- head take: the cube tags of both grippers, the environment and dock tags.

python scripts_tag_tracking/inspect_tags.py -a example/tag_tracking/aruco_config.yaml <video.MP4> [<video.MP4> ...]
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import concurrent.futures
import pathlib
import click
import av
import cv2
import numpy as np
import yaml
from umi.common.cv_util import parse_aruco_config
from umi.common.exiftool_util import ExifTool
from umi.tag_tracking.tag_detection import detect_tag_corners, tag_inventory


def sample_detections(path, predefined, every_sec, max_sec):
    """
    Tag corners of one frame every every_sec seconds (up to max_sec).
    """
    cv2.setNumThreads(1)
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, predefined))
    detections = list()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        duration = float(stream.frames / stream.average_rate)
        size = (stream.codec_context.width, stream.codec_context.height)
        end = duration if max_sec is None else min(duration, max_sec)
        for t in np.arange(0.0, end, every_sec):
            container.seek(int(t / stream.time_base), stream=stream)
            for frame in container.decode(stream):
                if float(frame.pts * stream.time_base) >= t - 0.5 / float(stream.average_rate):
                    detections.append(detect_tag_corners(frame.to_ndarray(format='gray'), aruco_dict))
                    break
    return {'duration': duration, 'size': size, 'detections': detections}


# %%
@click.command()
@click.argument('videos', nargs=-1, required=True)
@click.option('-a', '--aruco_config', default='example/tag_tracking/aruco_config.yaml', help='Dictionary and tag sizes')
@click.option('--every_sec', type=float, default=1.0, help='Sample one frame every this many seconds')
@click.option('--max_sec', type=float, default=None, help='Only the first this many seconds of each video')
@click.option('--min_share', type=float, default=0.02, help='Hide tags in fewer sampled frames (spurious detections)')
@click.option('-j', '--jobs', type=int, default=8, help='Videos read in parallel')
def main(videos, aruco_config, every_sec, max_sec, min_share, jobs):
    config = yaml.safe_load(open(os.path.expanduser(aruco_config), 'r'))
    predefined = config['aruco_dict']['predefined']
    sizes = parse_aruco_config(config)['marker_size_map']
    configured = {int(k) for k in config['marker_size_map'] if k != 'default'}
    paths = [pathlib.Path(os.path.expanduser(v)).absolute() for v in videos]
    with ExifTool() as et:
        serials = {p: list(et.get_metadata(str(p)))[0].get('QuickTime:CameraSerialNumber') for p in paths}
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(sample_detections, paths, [predefined] * len(paths), [every_sec] * len(paths),
            [max_sec] * len(paths)))
    for path, r in zip(paths, results):
        inventory = tag_inventory(r['detections'], r['size'])
        shown = {k: v for k, v in inventory.items() if v['share'] >= min_share}
        print(f"\n{path}\n  camera {serials[path]}, {r['duration']:.1f} s, {len(r['detections'])} frames sampled, "
            f"{len(shown)} tags" + (f" ({len(inventory) - len(shown)} more in < {min_share:.0%} of frames)"
            if len(inventory) > len(shown) else ''))
        print(f"  {'tag':>4} {'frames':>7} {'side px':>8} {'x':>5} {'y':>5}  size in aruco config")
        for tag_id, x in shown.items():
            size = f"{sizes[tag_id] * 1000:.0f} mm" if tag_id in configured else f"not listed (default {sizes[tag_id]})"
            print(f"  {tag_id:>4} {x['share']:>7.0%} {x['side_px']:>8.1f} {x['center'][0]:>5.2f} {x['center'][1]:>5.2f}  {size}")

# %%
if __name__ == "__main__":
    main()
