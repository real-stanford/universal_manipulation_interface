# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import collections
import pickle
import json
import numpy as np
from umi.common.cv_util import get_gripper_width, get_finger_center_x


# %%
def identify_gripper(tag_detection_results, tag_det_threshold):
    """
    Default UMI convention: gripper g uses finger tags 6g (left) and 6g+1 (right).
    """
    # identify gripper hardware id
    n_frames = len(tag_detection_results)
    tag_counts = collections.defaultdict(lambda: 0)
    for frame in tag_detection_results:
        for key in frame['tag_dict'].keys():
            tag_counts[key] += 1
    tag_stats = collections.defaultdict(lambda: 0.0)
    for k, v in tag_counts.items():
        tag_stats[k] = v / n_frames
    
    max_tag_id = np.max(list(tag_stats.keys()))
    tag_per_gripper = 6
    max_gripper_id = max_tag_id // tag_per_gripper
    
    gripper_prob_map = dict()
    for gripper_id in range(max_gripper_id+1):
        left_id = gripper_id * tag_per_gripper
        right_id = left_id + 1
        left_prob = tag_stats[left_id]
        right_prob = tag_stats[right_id]
        gripper_prob = min(left_prob, right_prob)
        if gripper_prob <= 0:
            continue
        gripper_prob_map[gripper_id] = gripper_prob
    if len(gripper_prob_map) == 0:
        print("No grippers detected!")
        exit(1)
    
    gripper_probs = sorted(gripper_prob_map.items(), key=lambda x:x[1])
    gripper_id = gripper_probs[-1][0]
    gripper_prob = gripper_probs[-1][1]
    print(f"Detected gripper id: {gripper_id} with probability {gripper_prob}")
    if gripper_prob < tag_det_threshold:
        print(f"Detection rate {gripper_prob} < {tag_det_threshold} threshold.")
        exit(1)
        
    left_id = gripper_id * tag_per_gripper
    right_id = left_id + 1
    return gripper_id, left_id, right_id

# %%
@click.command()
@click.option('-i', '--input', required=True, help='Tag detection pkl')
@click.option('-o', '--output', required=True, help='output json')
@click.option('-t', '--tag_det_threshold', type=float, default=0.8)
@click.option('-nz', '--nominal_z', type=float, default=0.072, help="nominal Z value for gripper finger tag")
@click.option('-g', '--gripper_id', type=int, default=None, help="Known gripper id, skips identification by tag ids")
@click.option('-l', '--left_id', type=int, default=None, help="Finger tag id on image left, with --gripper_id")
@click.option('-r', '--right_id', type=int, default=None, help="Finger tag id on image right, with --gripper_id")
def main(input, output, tag_det_threshold, nominal_z, gripper_id, left_id, right_id):
    tag_detection_results = pickle.load(open(input, 'rb'))
    if gripper_id is None:
        gripper_id, left_id, right_id = identify_gripper(
            tag_detection_results, tag_det_threshold)
    else:
        assert (left_id is not None) and (right_id is not None)
    print(f"Gripper id {gripper_id}, finger tags left {left_id} right {right_id}")

    # run calibration
    gripper_widths = list()
    for i, dt in enumerate(tag_detection_results):
        tag_dict = dt['tag_dict']
        width = get_gripper_width(tag_dict, left_id, right_id, nominal_z=nominal_z)
        if width is None:
            width = float('Nan')
        gripper_widths.append(width)
    gripper_widths = np.array(gripper_widths)
    print(f"Gripper width measured in {np.isfinite(gripper_widths).mean() * 100:.1f}% of frames")
    max_width = np.nanmax(gripper_widths)
    min_width = np.nanmin(gripper_widths)
    # midpoint of the fingers in camera x, from frames with both finger tags: a width
    # from one tag is twice its distance from it (UMI assumed the camera axis, 0)
    centers = [get_finger_center_x(dt['tag_dict'], left_id, right_id, nominal_z=nominal_z)
        for dt in tag_detection_results]
    centers = np.array([c for c in centers if c is not None])
    assert len(centers) > 0, "no frame with both finger tags"
    finger_center_x = float(np.median(centers))
    print(f"Finger midpoint {finger_center_x * 1000:+.2f} mm from the camera axis "
        f"(5-95% {np.percentile(centers, 5) * 1000:+.2f}..{np.percentile(centers, 95) * 1000:+.2f} mm, "
        f"{len(centers)} frames with both finger tags)")

    result = {
        'gripper_id': gripper_id,
        'left_finger_tag_id': left_id,
        'right_finger_tag_id': right_id,
        'max_width': max_width,
        'min_width': min_width,
        'finger_center_x': finger_center_x,
        'finger_center_x_p5_p95': [float(np.percentile(centers, 5)), float(np.percentile(centers, 95))],
        'n_center_frames': int(len(centers))
    }
    json.dump(result, open(output, 'w'), indent=2)

# %%
if __name__ == "__main__":
    main()
