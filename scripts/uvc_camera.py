import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import cv2
import time
import threadpoolctl
from umi.common.usb_util import create_usb_list, reset_usb_device
'''
apiPreference    preferred Capture API backends to use. 
Can be used to enforce a specific reader implementation 
if multiple are available: 
e.g. cv2.CAP_MSMF or cv2.CAP_DSHOW.
'''

# limiting single thread is critical to reduce
# jitter in capture latency measured by the difference
# between CAP_PROP_POS_MSEC and monotonic time
cv2.setNumThreads(1)
threadpoolctl.threadpool_limits(1)
# getting 0.13-0.14 sec latency witch matches result on MacOS

# enumerate UBS device to find Elgato Capture Card
device_list = create_usb_list()
usb_dev = None
for dev in device_list:
    if 'Elgato' in dev['description']:
        usb_dev = dev
        break

# reset USB device due to Elgato bug
if usb_dev is not None:
    print('Resetting', usb_dev['description'])
    reset_usb_device(usb_dev['path'])
    time.sleep(0.2)

# open video0
cap = cv2.VideoCapture('/dev/video4', cv2.CAP_V4L2)
# set width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
# set fps
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
cap.set(cv2.CAP_PROP_FPS, 29.97)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

# reuse frame buffer
frame = None
while(True):
    t_start = time.monotonic()
    # Capture frame-by-frame
    # ret, frame = cap.read()
    # required for camlink
    ret = cap.grab()
    ret, frame = cap.retrieve(frame)

    t_cap = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    t_now = time.monotonic()
    print(t_now - t_cap)
    
    # Display the resulting frame
    # cv2.imshow('frame', frame)
    # if cv2.pollKey() & 0xFF == ord('q'):
    #     break

    t_end = time.monotonic()
    print('fps:', 1/(t_end - t_start))
    time.sleep(0)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
