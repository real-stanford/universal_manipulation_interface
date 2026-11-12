"""
Measure tag corner detection noise (pixels) from a video of a static
camera looking at static tags, e.g. an IMU bias calibration video of a gripper camera,
where the finger tags stay fixed in view. Only tags listed in the
aruco config are used.

Every tag corner should stay at the same pixel; its deviation from the
per-corner median is detection noise. sigma is the median absolute
deviation scaled by 1.4826, the consistency constant that makes it
equal the standard deviation for Gaussian noise, and is robust to the
short handling motion at the start and end of the recording.

python scripts_tag_tracking/01_measure_corner_noise.py -i gopro_calib/gripper1/GX019256.MP4 -c umi_calib/tag_tracking.yaml
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import json
import click
import numpy as np
from umi.tag_tracking.config import load_tag_tracking_config
from umi.tag_tracking.tag_detection import detect_video_tag_corners

MAD_TO_SIGMA = 1.4826


# %%
@click.command()
@click.option('-i', '--input', required=True, multiple=True, help='Video(s) of a static camera viewing static tags')
@click.option('-c', '--config', required=True, help='Tag tracking config yaml')
@click.option('-o', '--output', default=None, help='Default: corner_noise path in config')
def main(input, config, output):
    config = load_tag_tracking_config(config)
    if output is None:
        output = config['corner_noise_path']

    deviations = list()
    tag_stats = dict()
    n_frames = 0
    for video_path in input:
        frames = detect_video_tag_corners(video_path, config['aruco_dict'], config['known_tag_ids'])
        n_frames += len(frames)
        name = os.path.basename(video_path)
        for tag_id in sorted({k for f in frames for k in f['corners']}):
            corners = np.array([f['corners'][tag_id] for f in frames if tag_id in f['corners']])
            dev = (corners - np.median(corners, axis=0)).flatten()
            deviations.append(dev)
            tag_stats[f"{name}:{tag_id}"] = {
                'n_frames': len(corners),
                'sigma_px': float(MAD_TO_SIGMA * np.median(np.abs(dev)))
            }
            print(f"{name} tag {tag_id}: {len(corners)} frames, sigma {tag_stats[f'{name}:{tag_id}']['sigma_px']:.3f} px")
    assert len(deviations) > 0, "no tags detected"
    deviations = np.concatenate(deviations)
    sigma_px = float(MAD_TO_SIGMA * np.median(np.abs(deviations)))
    print(f"Corner noise sigma: {sigma_px:.3f} px from {len(deviations)} corner coordinates")

    json.dump({
        'sigma_px': sigma_px,
        'source_videos': [os.path.abspath(os.path.expanduser(x)) for x in input],
        'n_frames': n_frames,
        'tags': tag_stats
    }, open(output, 'w'), indent=2)
    print(f"Saved to {output}")

# %%
if __name__ == "__main__":
    main()
