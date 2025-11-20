import cv2
import numpy as np
import os
from pathlib import Path


def create_masked_image(video_path, mask_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return

    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame from video file '{video_path}'.")
        cap.release()
        return
    cap.release()

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read the mask as grayscale
    if mask is None:
        print(f"Error: Could not open or read mask file '{mask_path}'.")
        return

    if frame.shape[:2] != mask.shape[:2]:
        print(
            f"Warning: Frame resolution ({frame.shape[:2]}) and mask resolution ({mask.shape[:2]}) are different. Resizing mask to fit frame size."
        )
        mask = cv2.resize(
            mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    mask_inv = cv2.bitwise_not(mask)

    img_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)

    cv2.imwrite(output_path, img_bg)
    print(f"Masked image '{output_path}' created successfully.")


if __name__ == "__main__":
    # video_sample_path = "~/Desktop/study/universal_manipulation_interface/example_demo_session/demos/mapping/raw_video.mp4"
    video_sample_path = "~/Desktop/study/universal_manipulation_interface/my_example_demo_session/demos/mapping/raw_video.mp4"
    video_sample_path = Path(video_sample_path).expanduser()
    video_sample_path = str(video_sample_path)
    mask_sample_path = "./slam_mask.png"
    output_masked_path = "./masked.png"

    if not os.path.exists(video_sample_path):
        print(f"Error: Video file '{video_sample_path}' does not exist.")
    if not os.path.exists(mask_sample_path):
        print(f"Error: Mask file '{mask_sample_path}' does not exist.")

    create_masked_image(video_sample_path, mask_sample_path, output_masked_path)
