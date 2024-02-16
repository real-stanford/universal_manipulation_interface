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
from umi.common.cv_util import get_gripper_width


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Tag detection pkl')
@click.option('-o', '--output', required=True, help='output json')
@click.option('-t', '--tag_det_threshold', type=float, default=0.8)
@click.option('-nz', '--nominal_z', type=float, default=0.072, help="nominal Z value for gripper finger tag")
def main(input, output, tag_det_threshold, nominal_z):
    tag_detection_results = pickle.load(open(input, 'rb'))
    
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
        
    # run calibration
    left_id = gripper_id * tag_per_gripper
    right_id = left_id + 1

    gripper_widths = list()
    for i, dt in enumerate(tag_detection_results):
        tag_dict = dt['tag_dict']
        width = get_gripper_width(tag_dict, left_id, right_id, nominal_z=nominal_z)
        if width is None:
            width = float('Nan')
        gripper_widths.append(width)
    gripper_widths = np.array(gripper_widths)
    max_width = np.nanmax(gripper_widths)
    min_width = np.nanmin(gripper_widths)

    result = {
        'gripper_id': gripper_id,
        'left_finger_tag_id': left_id,
        'right_finger_tag_id': right_id,
        'max_width': max_width,
        'min_width': min_width
    }
    json.dump(result, open(output, 'w'), indent=2)

# %%
if __name__ == "__main__":
    main()
