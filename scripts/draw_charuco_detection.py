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
    draw_predefined_mask,
    get_charuco_board
)

# %%
@click.command()
@click.option('-i', '--input', required=True)
@click.option('-o', '--output', required=True)
@click.option('-ij', '--intrinsics_json', required=True)
@click.option('-n', '--num_workers', type=int, default=4)
def main(input, output, intrinsics_json, num_workers):
    cv2.setNumThreads(num_workers)

    # load aruco config
    board = get_charuco_board()
    board_detector = cv2.aruco.CharucoDetector(board)

    # load intrinsics
    raw_fisheye_intr = parse_fisheye_intrinsics(json.load(open(intrinsics_json, 'r')))
    intr_dict = raw_fisheye_intr
    
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
                
                # detect charuco
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                charuco_corners, charuco_ids, \
                    marker_corners, marker_ids = board_detector.detectBoard(gray)
                    
                vis_img = img
                cv2.aruco.drawDetectedMarkers(image=vis_img, corners=marker_corners, ids=marker_ids)

                if (charuco_ids is not None) and (charuco_ids.size >= 14):
                
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
                    refined_corners = cv2.cornerSubPix(gray, charuco_corners,(5,5),(-1,-1),criteria)

                    # undistort corners
                    K = intr_dict['K']
                    D = intr_dict['D']
                    undistorted_corners = cv2.fisheye.undistortPoints(refined_corners, K, D, P=np.eye(3))

                    # pose
                    rvec = np.zeros((1,3))
                    tvec = np.zeros((1,3))
                    retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                        charucoCorners=undistorted_corners, 
                        charucoIds=charuco_ids, 
                        board=board, 
                        cameraMatrix=np.eye(3),
                        distCoeffs=np.zeros((1,5)),
                        rvec=rvec,
                        tvec=tvec,
                        useExtrinsicGuess=False)
                    
                    # vis
                    cv2.aruco.drawDetectedCornersCharuco(vis_img, refined_corners, charuco_ids, (255,0,0))
                
                out_frame = av.VideoFrame.from_ndarray(vis_img, format='rgb24')
                for packet in out_stream.encode(out_frame):
                    out_container.mux(packet)
            
            # flush
            for packet in out_stream.encode():
                out_container.mux(packet)
                
                

# %%
if __name__ == "__main__":
    main()
