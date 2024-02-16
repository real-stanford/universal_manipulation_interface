# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import cv2
import qrcode
import time
import numpy as np
from collections import deque
from tqdm import tqdm
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.uvc_camera import UvcCamera
from umi.common.usb_util import reset_all_elgato_devices, get_sorted_v4l_paths
from matplotlib import pyplot as plt

# %%
@click.command()
@click.option('-ci', '--camera_idx', type=int, default=0)
@click.option('-qs', '--qr_size', type=int, default=720)
@click.option('-f', '--fps', type=int, default=60)
@click.option('-n', '--n_frames', type=int, default=120)
def main(camera_idx, qr_size, fps, n_frames):
    # Find and reset all Elgato capture cards.
    # Required to workaround a firmware bug.
    reset_all_elgato_devices()
    v4l_paths = get_sorted_v4l_paths()
    v4l_path = v4l_paths[camera_idx]
    get_max_k = n_frames
    detector = cv2.QRCodeDetector()
    with SharedMemoryManager() as shm_manager:
        with UvcCamera(
            shm_manager=shm_manager,
            dev_video_path=v4l_path,
            resolution=(1280, 720),
            capture_fps=fps,
            get_max_k=get_max_k
        ) as camera:
            cv2.setNumThreads(1)
            qr_latency_deque = deque(maxlen=get_max_k)
            qr_det_queue = deque(maxlen=get_max_k)
            data = None
            while True:
                t_start = time.time()
                data = camera.get(out=data)
                cam_img = data['color']
                code, corners, _ = detector.detectAndDecodeCurved(cam_img)
                color = (0,0,255)
                if len(code) > 0:
                    color = (0,255,0)
                    ts_qr = float(code)
                    ts_recv = data['camera_receive_timestamp']
                    latency = ts_recv - ts_qr
                    qr_det_queue.append(latency)
                else:
                    qr_det_queue.append(float('nan'))
                if corners is not None:
                    cv2.fillPoly(cam_img, corners.astype(np.int32), color)
                
                qr = qrcode.QRCode(
                    version=1,
                    error_correction=qrcode.constants.ERROR_CORRECT_H,
                )
                t_sample = time.time()
                qr.add_data(str(t_sample))
                qr.make(fit=True)
                pil_img = qr.make_image()
                img = np.array(pil_img).astype(np.uint8) * 255
                img = np.repeat(img[:,:,None], 3, axis=-1)
                img = cv2.resize(img, (qr_size, qr_size), cv2.INTER_NEAREST)
                cv2.imshow('Timestamp QRCode', img)
                t_show = time.time()
                qr_latency_deque.append(t_show - t_sample)
                cv2.imshow('Camera', cam_img)
                keycode = cv2.pollKey()
                t_end = time.time()
                avg_latency = np.nanmean(qr_det_queue) - np.mean(qr_latency_deque)
                det_rate = 1-np.mean(np.isnan(qr_det_queue))
                print("Running at {:.1f} FPS. Recv Latency: {:.3f}. Detection Rate: {:.2f}".format(
                    1/(t_end-t_start),
                    avg_latency,
                    det_rate
                ))

                if keycode == ord('c'):
                    break
                elif keycode == ord('q'):
                    exit(0)
            data = camera.get(k=get_max_k)

        qr_recv_map = dict()
        for i in tqdm(range(len(data['camera_receive_timestamp']))):
            ts_recv = data['camera_receive_timestamp'][i]
            img = data['color'][i]
            code, corners, _ = detector.detectAndDecodeCurved(img)
            if len(code) > 0:
                ts_qr = float(code)
                if ts_qr not in qr_recv_map:
                    qr_recv_map[ts_qr] = ts_recv

        avg_qr_latency = np.mean(qr_latency_deque)
        t_offsets = [v-k-avg_qr_latency for k,v in qr_recv_map.items()]
        avg_latency = np.mean(t_offsets)
        std_latency = np.std(t_offsets)
        print(f'Capture to receive latency: AVG={avg_latency} STD={std_latency}')

        x = np.array(list(qr_recv_map.values()))
        y = np.array(list(qr_recv_map.keys()))
        y -= x[0]
        x -= x[0]
        plt.plot(x, x)
        plt.scatter(x, y)
        plt.xlabel('Receive Timestamp (sec)')
        plt.ylabel('QR Timestamp (sec)')
        plt.show()
        

# %%
if __name__ == "__main__":
    main()
