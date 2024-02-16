import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import cv2
import json
import time
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.multi_uvc_camera import MultiUvcCamera
from umi.real_world.multi_camera_visualizer import MultiCameraVisualizer
from umi.real_world.video_recorder import VideoRecorder
from umi.common.usb_util import reset_all_elgato_devices, get_sorted_v4l_paths
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)

def test():

    # Find and reset all Elgato capture cards.
    # Required to workaround a firmware bug.
    reset_all_elgato_devices()

    # Wait for all v4l cameras to be back online
    time.sleep(0.1)
    v4l_paths = get_sorted_v4l_paths()

    resolution = [
        (3840, 2160),
        (1280, 720),
        (1280, 720)
    ]

    transform = list()
    for in_res in resolution:
        def tf(data, input_res=in_res, output_res=(224,224)):
            f = get_image_transform(input_res=input_res, output_res=output_res)
            data = dict(data)
            data['color'] = f(data['color'])
            return data
        transform.append(tf)

    fps = [30, 60, 60]
    video_recorder = [
    VideoRecorder.create_hevc_nvenc(
        fps=f,
        input_pix_fmt='bgr24',
        bit_rate=6000*1000
    ) for f in fps]


    with SharedMemoryManager() as shm_manager:
        with MultiUvcCamera(
                dev_video_paths=v4l_paths,
                shm_manager=shm_manager,
                resolution=resolution,
                put_fps=10,
                capture_fps=fps,
                record_fps=fps,
                cap_buffer_size=[3,1,1],
                transform=transform,
                recording_transform=None,
                video_recorder=video_recorder,
                verbose=False
            ) as camera:
            # with MultiCameraVisualizer(
            #     camera=camera,
            #     row=row,
            #     col=col,
            #     rgb_to_bgr=False
            #     ) as vis:

                cv2.setNumThreads(1) 

                video_path = 'data_local/test'
                rec_start_time = time.time() + 1
                camera.start_recording(video_path, start_time=rec_start_time)
                camera.restart_put(rec_start_time)

                out = None
                vis_img = None
                while True:
                    out = camera.get(out=out)

                    # bgr = [x['color'] for x in out.values()]
                    # vis_img = np.concatenate(bgr, axis=0, out=vis_img)
                    # cv2.imshow('default', vis_img)
                    # key = cv2.pollKey()
                    # if key == ord('q'):
                    #     break

                    time.sleep(1/60)
                    if time.time() > (rec_start_time + 5.0):
                        break
            
if __name__ == "__main__":
    test()
