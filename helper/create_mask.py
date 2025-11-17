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
from umi.common.cv_util_realsense import (
    draw_rgb_predefined_mask,
    RGB_IMG_SHAPE
)

def main():
    current_file_path = pathlib.Path(__file__).resolve()
    current_dir = current_file_path.parent
    mask_write_path = current_dir.joinpath("slam_mask.png")
    
    # slam_mask = np.zeros((2028, 2704), dtype=np.uint8)
    slam_mask = np.zeros(RGB_IMG_SHAPE, dtype=np.uint8)
    slam_mask = draw_rgb_predefined_mask(
        slam_mask, color=255, mirror=False, gripper=False, finger=True)
    cv2.imwrite(str(mask_write_path.absolute()), slam_mask)

    print(f"[INFO] create mask png")


if __name__ == "__main__":
    main()
