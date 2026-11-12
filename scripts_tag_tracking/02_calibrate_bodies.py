"""
Calibrate the pose of every tag on its rigid body (gripper cubes, static
environment) by bundle adjustment over all head camera videos of a session,
together with the corner noise model and the camera's rolling shutter
line delay. Requires tag_corners.pkl from 00_detect_tag_corners.py.

0. Corner noise model sigma = s_tag * cos(theta)^-p by median regression
   of single-tag fits (umi.tag_tracking.noise_model).
1. Calibrate bodies without rolling shutter.
2. Line delay: maximum likelihood (minimum robust tracking error) within the
   physical bound |line_delay| <= frame period / image rows; the sign leaves
   the readout direction to the data.
3. Calibrate bodies with rolling shutter.

Writes <session>/tag_tracking/bodies.json

python scripts_tag_tracking/02_calibrate_bodies.py -i data_workspace/session -c umi_calib/tag_tracking.yaml
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import json
import pathlib
import pickle
import click
import numpy as np
import scipy.optimize
from umi.tag_tracking.config import load_tag_tracking_config
from umi.tag_tracking.noise_model import TagNoiseModel
from umi.tag_tracking.rigid_body import calibrate_rigid_body, track_rigid_body


# %%
def load_sequences(session):
    corner_paths = sorted(session.glob('demos/*/tag_corners.pkl'))
    assert len(corner_paths) > 0, "run 00_detect_tag_corners.py first"
    frames = list()
    for sequence, path in enumerate(corner_paths):
        for x in pickle.load(path.open('rb')):
            frames.append({'time': x['time'], 'corners': x['corners'], 'sequence': sequence})
    return frames, len(corner_paths)

def calibrate_all(camera, frames, config, noise_model, line_delay):
    return {name: calibrate_rigid_body(camera, frames, body['tags'], config['tag_sizes'],
        noise_model, body['anchor_tag'], line_delay) for name, body in config['bodies'].items()}

def print_results(results):
    for name, x in results.items():
        print(f"  {name}: anchor {x['anchor_tag']}, tags {sorted(x['T_body_tags'])}, missing {x['missing_tags']}, "
            f"{x['n_frames_consistent']}/{x['n_frames_used']} multi-tag frames consistent, "
            f"variance factor {x['variance_factor']:.2f}")


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Session directory')
@click.option('-c', '--config', required=True, help='Tag tracking config yaml')
def main(input, config):
    session = pathlib.Path(os.path.expanduser(input)).absolute()
    config = load_tag_tracking_config(config)
    camera = config['cameras'][config['head_camera']]
    frames, n_videos = load_sequences(session)
    print(f"{len(frames)} frames from {n_videos} videos")

    print("0. Corner noise model")
    noise_model = TagNoiseModel.fit(camera, frames, config['tag_sizes'])
    print(f"  sigma = s_tag * cos(theta)^-{noise_model.exponent:.2f}, s_tag px "
        f"{ {k: round(v, 3) for k, v in sorted(noise_model.tag_scales.items())} }")
    if 'sigma_px' in config:
        print(f"  static measurement (01_measure_corner_noise.py): {config['sigma_px']:.3f} px")

    print("1. Calibration without rolling shutter")
    results = calibrate_all(camera, frames, config, noise_model, 0.0)
    print_results(results)

    print("2. Rolling shutter line delay")
    frame_period = np.median(np.diff([x['time'] for x in frames if x['sequence'] == 0]))
    max_line_delay = frame_period / camera.image_size[1]

    def tracking_error(ratio):
        error = sum(track_rigid_body(camera, frames, x['T_body_tags'], config['tag_sizes'],
            noise_model, ratio * max_line_delay)['error'].sum() for x in results.values())
        print(f"  line delay {ratio * max_line_delay * 1e6:7.3f} us: error {error:.1f}")
        return error

    opt = scipy.optimize.minimize_scalar(tracking_error, bounds=(-1, 1), method='bounded')
    line_delay = float(opt.x * max_line_delay)
    print(f"  line delay {line_delay * 1e6:.3f} us "
        f"(readout {abs(line_delay) * camera.image_size[1] * 1e3:.2f} ms, frame period {frame_period * 1e3:.2f} ms)")

    print("3. Calibration with rolling shutter")
    results = calibrate_all(camera, frames, config, noise_model, line_delay)
    print_results(results)

    output = session.joinpath('tag_tracking', 'bodies.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        'noise_model': noise_model.to_dict(),
        'line_delay': line_delay,
        'bodies': {name: {
            'static': config['bodies'][name]['static'],
            'anchor_tag': x['anchor_tag'],
            'T_body_tags': {str(k): v.tolist() for k, v in x['T_body_tags'].items()},
            'missing_tags': x['missing_tags'],
            'n_frames_used': x['n_frames_used'],
            'n_frames_consistent': x['n_frames_consistent'],
            'variance_factor': x['variance_factor'],
            'tag_rms_px': {str(k): v for k, v in x['tag_rms_px'].items()}
        } for name, x in results.items()}
    }, output.open('w'), indent=2)
    print(f"Saved to {output}")

# %%
if __name__ == "__main__":
    main()
