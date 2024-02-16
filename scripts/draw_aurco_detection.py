# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
from tqdm import tqdm
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

# %%
@click.command()
@click.option('-i', '--input', required=True)
@click.option('-o', '--output', required=True)
@click.option('-ij', '--intrinsics_json', required=True)
@click.option('-ay', '--aruco_yaml', required=True)
@click.option('-n', '--num_workers', type=int, default=4)
def main(input, output, intrinsics_json, aruco_yaml, num_workers):
    cv2.setNumThreads(num_workers)

    # load aruco config
    aruco_config = parse_aruco_config(yaml.safe_load(open(aruco_yaml, 'r')))
    aruco_dict = aruco_config['aruco_dict']
    marker_size_map = aruco_config['marker_size_map']
    
    # load intrinsics
    raw_fisheye_intr = parse_fisheye_intrinsics(json.load(open(intrinsics_json, 'r')))
    
    with av.open(os.path.expanduser(output), mode='w') as out_container:
        with av.open(os.path.expanduser(input)) as in_container:
            in_stream = in_container.streams.video[0]
            in_stream.thread_type = "AUTO"
            in_stream.thread_count = num_workers
            
            out_stream = out_container.add_stream('h264', rate=in_stream.rate)
            out_stream.thread_type = 'AUTO'
            out_stream.thread_count = num_workers
            
            in_res = np.array([in_stream.height, in_stream.width])
            fisheye_intr = convert_fisheye_intrinsics_resolution(
                opencv_intr_dict=raw_fisheye_intr, target_resolution=in_res[::-1])

            out_res = in_res
            
            out_stream.width = out_res[1]
            out_stream.height = out_res[0]
            
            out_codec_context = out_stream.codec_context
            out_codec_context.options = {
                'crf': '20',
                'profile': 'high'
            }

            in_res = np.array([in_stream.height, in_stream.width])[::-1]

            for i, frame in tqdm(enumerate(in_container.decode(in_stream)), total=in_stream.frames):
                img = frame.to_ndarray(format='rgb24')
                
                tag_dict = detect_localize_aruco_tags(
                    img=img,
                    aruco_dict=aruco_dict,
                    marker_size_map=marker_size_map,
                    fisheye_intr_dict=fisheye_intr,
                    refine_subpix=False
                )
                
                ids = list()
                corners = list()
                for k, v in tag_dict.items():
                    ids.append(k)
                    corners.append(v['corners'][None,...])
                ids = np.array(ids, dtype=np.int32)
                ids = ids[...,None]
                # corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image=img, dictionary=aruco_dict)

                # vis
                vis_img = img
                cv2.aruco.drawDetectedMarkers(image=vis_img, corners=corners, ids=ids)
                out_frame = av.VideoFrame.from_ndarray(vis_img, format='rgb24')
                for packet in out_stream.encode(out_frame):
                    out_container.mux(packet)
            
            # flush
            for packet in out_stream.encode():
                out_container.mux(packet)
                
                

# %%
if __name__ == "__main__":
    main()
