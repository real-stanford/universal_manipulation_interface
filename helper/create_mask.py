"""
python scripts_slam_pipeline/00_process_videos.py -i data_workspace/toss_objects/20231113/mapping
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
import subprocess
import multiprocessing
import concurrent.futures
from tqdm import tqdm
import numpy as np
import cv2

# from umi.common.cv_util import draw_predefined_mask
from umi.common.cv_util_realsense import *


def main():
    current_file_path = pathlib.Path(__file__).resolve()
    current_dir = current_file_path.parent

    mode = "im_r_Infrared"

    if mode == "GOPRO":
        slam_mask = np.zeros((2028, 2704), dtype=np.uint8)
    elif mode == "RGB":
        mask_write_path = current_dir.joinpath("slam_mask.png")
        slam_mask = np.zeros(RGB_IMG_SHAPE, dtype=np.uint8)
        slam_mask = draw_rgb_predefined_mask(
            slam_mask, color=255, mirror=False, gripper=False, finger=True
        )
    elif mode == "im_l_Infrared":
        mask_write_path = current_dir.joinpath("im_l_Infrared_mask.png")
        slam_mask = np.zeros(IR_IMG_SHAPE, dtype=np.uint8)
        slam_mask = draw_im_l_infrared_mask(
            slam_mask, color=255, mirror=False, gripper=True, finger=True
        )
    elif mode == "im_r_Infrared":
        mask_write_path = current_dir.joinpath("im_r_Infrared_mask.png")
        slam_mask = np.zeros(IR_IMG_SHAPE, dtype=np.uint8)
        slam_mask = draw_im_r_infrared_mask(
            slam_mask, color=255, mirror=False, gripper=True, finger=True
        )
    else:
        raise Exception()

    cv2.imwrite(str(mask_write_path.absolute()), slam_mask)

    print(f"[INFO] create mask png {str(mask_write_path)}")


if __name__ == "__main__":
    main()
