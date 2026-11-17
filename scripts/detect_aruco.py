"""
Detect and localize ArUco tags in every frame of a video, writes a pickle with
per frame {'frame_idx', 'time', 'tag_dict'} (tag id -> corners, rvec, tvec).
Frames are processed in parallel time chunks on num_workers processes
(umi.common.video_chunk_util), one OpenCV thread each.

python scripts/detect_aruco.py -i raw_video.mp4 -o tag_detection.pkl -ij gopro_intrinsics_2_7k.json -ay aruco_config.yaml
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import yaml
import json
import av
import numpy as np
import cv2
import pickle

from umi.common.cv_util import (
    parse_aruco_config,
    parse_fisheye_intrinsics,
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags,
    draw_predefined_mask
)
from umi.common.video_chunk_util import process_video_chunked


# %%
def make_tag_detector(intrinsics_json, aruco_yaml, resolution):
    """
    Per frame detector, built in each worker process (OpenCV objects are not picklable).
    """
    cv2.setNumThreads(1)
    aruco_config = parse_aruco_config(yaml.safe_load(open(aruco_yaml, 'r')))
    aruco_dict = aruco_config['aruco_dict']
    marker_size_map = aruco_config['marker_size_map']
    fisheye_intr = convert_fisheye_intrinsics_resolution(
        opencv_intr_dict=parse_fisheye_intrinsics(json.load(open(intrinsics_json, 'r'))),
        target_resolution=resolution)

    def detect(frame, time):
        img = frame.to_ndarray(format='rgb24')
        # avoid detecting tags in the mirror
        img = draw_predefined_mask(img, color=(0,0,0), mirror=True, gripper=False, finger=False)
        tag_dict = detect_localize_aruco_tags(
            img=img,
            aruco_dict=aruco_dict,
            marker_size_map=marker_size_map,
            fisheye_intr_dict=fisheye_intr,
            refine_subpix=True
        )
        return {'time': time, 'tag_dict': tag_dict}
    return detect


# %%
@click.command()
@click.option('-i', '--input', required=True)
@click.option('-o', '--output', required=True)
@click.option('-ij', '--intrinsics_json', required=True)
@click.option('-ay', '--aruco_yaml', required=True)
@click.option('-n', '--num_workers', type=int, default=None, help='Worker processes, default all cores')
def main(input, output, intrinsics_json, aruco_yaml, num_workers):
    input = os.path.expanduser(input)
    with av.open(input) as container:
        stream = container.streams.video[0]
        resolution = np.array([stream.width, stream.height])

    results = process_video_chunked(input, make_tag_detector,
        (os.path.expanduser(intrinsics_json), os.path.expanduser(aruco_yaml), resolution),
        num_workers=num_workers, desc=os.path.basename(os.path.dirname(os.path.abspath(input))))

    # dump
    pickle.dump(results, open(os.path.expanduser(output), 'wb'))

# %%
if __name__ == "__main__":
    main()
