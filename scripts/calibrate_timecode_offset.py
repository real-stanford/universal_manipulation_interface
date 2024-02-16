# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import os
import av
import cv2
import datetime
from tqdm import tqdm
import json
from umi.common.timecode_util import stream_get_start_datetime

# %%
def parse_qr_datetime(qr_txt):
    return datetime.datetime.strptime(qr_txt, r"#%y%m%d%H%M%S.%f")

# %%
@click.command()
@click.option('-i', '--input', required=True, help='GoPro MP4 file')
def main(input):
    input = os.path.expanduser(input)

    # find the first QR code timestamp
    detector = cv2.QRCodeDetector()
    qr_datetime = None
    tc_datetime = None
    with av.open(input) as container:
        stream = container.streams.video[0]
        tc_datetime = stream_get_start_datetime(stream=stream)
        for i, frame in tqdm(enumerate(container.decode(stream))):
            img = frame.to_ndarray(format='rgb24')
            frame_cts_sec = frame.pts * stream.time_base

            qr_txt, _, _ =detector.detectAndDecodeCurved(img)
            if qr_txt.startswith('#'):
                # found
                qr_datetime = parse_qr_datetime(qr_txt) \
                    - datetime.timedelta(seconds=float(frame_cts_sec))
                break

    if not qr_datetime:
        raise RuntimeError("No valid QR code found.")
    
    dt = (qr_datetime - tc_datetime).total_seconds()

    print("time = date + timecode + dt")
    print(f"dt = {dt}")

if __name__ == '__main__':
    main()

