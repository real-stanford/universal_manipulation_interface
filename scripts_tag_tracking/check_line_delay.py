"""
Check the head camera's rolling shutter line delay on the frames where it
matters: per body, the head frames with the fastest relative motion
(tag_tracking.pkl), tracked again with line delay +readout/rows (top row
first), 0 and -readout/rows; the tag reprojection error of those frames
(robust tracking error and corner residual RMS in pixels) per line delay.
On mostly static frames the line delay changes nothing, so a choice made
over all frames (calibrate_setup.py) can miss it.

Reads tag_tracking/bodies.json (tag poses, noise model), head demos/*/tag_corners.pkl
and tag_tracking.pkl; readout from SensorReadoutTime. Writes nothing.

python scripts_tag_tracking/check_line_delay.py -i data_workspace/session -c umi_calib/tag_tracking.yaml
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import json
import multiprocessing
import pathlib
import pickle
import click
import numpy as np
from scipy.spatial.transform import Rotation
from umi.common.exiftool_util import ExifTool
from umi.tag_tracking.config import load_tag_tracking_config
from umi.tag_tracking.session import get_session_videos
from umi.tag_tracking.noise_model import TagNoiseModel
from umi.tag_tracking.parallel import make_executor, submit_track, gather_track, CHUNKS_PER_WORKER
from umi.tag_tracking.setup_calibration import frames_with_neighbors


def motion_speed(T_cam_body: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Relative angular speed (rad/s) per frame from its two neighbors, 0 where unknown.
    """
    n = len(T_cam_body)
    speed = np.zeros(n)
    ok = ~np.isnan(T_cam_body[:, 0, 0])
    k = np.flatnonzero(ok[:-2] & ok[1:-1] & ok[2:]) + 1
    R0 = Rotation.from_matrix(T_cam_body[k - 1, :3, :3])
    R1 = Rotation.from_matrix(T_cam_body[k + 1, :3, :3])
    speed[k] = np.linalg.norm((R0.inv() * R1).as_rotvec(), axis=1) / (times[k + 1] - times[k - 1])
    return speed


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Session directory (tracked)')
@click.option('-c', '--config', required=True, help='Tag tracking config yaml')
@click.option('-n', '--n_frames', type=int, default=1500, help='Fastest frames per body')
@click.option('-j', '--num_workers', type=int, default=None, help='Worker processes, default all cores')
def main(input, config, n_frames, num_workers):
    session = pathlib.Path(os.path.expanduser(input)).absolute()
    config = load_tag_tracking_config(config)
    camera = config['cameras'][config['head_camera']]
    rows = camera.image_size[1]
    calibration = json.load(session.joinpath('tag_tracking', 'bodies.json').open('r'))
    noise_model = TagNoiseModel.from_dict(calibration['noise_model'])
    num_workers = num_workers or multiprocessing.cpu_count()
    executor = make_executor(num_workers)

    for head in [x for x in get_session_videos(session) if x['camera_serial'] == config['head_camera']]:
        with ExifTool() as et:
            readout = float(list(et.get_metadata(str(head['dir'].joinpath('raw_video.mp4'))))[0]['QuickTime:SensorReadoutTime']) * 1e-3
        corners = pickle.load(head['dir'].joinpath('tag_corners.pkl').open('rb'))
        track = pickle.load(head['dir'].joinpath('tag_tracking.pkl').open('rb'))
        times = track['time']
        magnitude = readout / rows
        delays = [magnitude, 0.0, -magnitude]
        print(f"{head['name']}: readout {readout * 1e3:.1f} ms, line delays (us) "
            + ', '.join(f"{d * 1e6:+.3f}" for d in delays))
        for name, body in calibration['bodies'].items():
            T_body_tags = {int(k): np.array(v) for k, v in body['T_body_tags'].items()}
            speed = motion_speed(track['bodies'][name]['T_cam_body'], times)
            n_tags = np.array([sum(t in T_body_tags for t in f['corners']) for f in corners])
            candidates = np.flatnonzero((speed > 0) & (n_tags >= 2))
            fastest = candidates[np.argsort(speed[candidates])[::-1][:n_frames]]
            frames, position = frames_with_neighbors(corners, fastest, head['name'])
            frames = [{**f, 'corners': {k: v for k, v in f['corners'].items() if k in T_body_tags}} for f in frames]
            centre = np.array([position[i] for i in fastest])
            submitted = [submit_track(executor, camera, frames, T_body_tags, config['tag_sizes'], noise_model, d,
                num_workers * CHUNKS_PER_WORKER) for d in delays]
            results = [gather_track(s) for s in submitted]
            print(f"  {name}: {len(fastest)} fastest frames, relative speed {np.median(speed[fastest]):.2f} rad/s median "
                f"(min {speed[fastest].min():.2f})")
            for d, r in zip(delays, results):
                error = r['error'][centre]
                ok = np.isfinite(error)
                print(f"    line delay {d * 1e6:+.3f} us: tracking error sum {np.sum(error[ok]):.1f}, "
                    f"median {np.median(error[ok]):.3f}, frames tracked {ok.sum()}")
    executor.shutdown()

# %%
if __name__ == "__main__":
    main()
