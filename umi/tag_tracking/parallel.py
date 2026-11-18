"""
Tag tracking and body calibration on several CPU cores.

parallel_track_rigid_body: per frame tracking (umi.tag_tracking.rigid_body.
track_rigid_body) split into chunks of consecutive frames, one worker process
per chunk. Each chunk is extended by one frame on each side within the same
sequence, so the rolling shutter motion (from neighboring frames' first pass
poses) is the same as tracking all frames at once; results are identical.

parallel_calibrate_bodies: calibrate_rigid_body of several bodies at once.

Workers are spawned (not forked, see umi.common.video_chunk_util) and rebuild
the camera from its intrinsics json and the noise model from its dict, since
GTSAM objects cannot be sent between processes. Pass an executor from
make_executor to reuse the worker processes across calls.
"""
from typing import Dict, List, Optional
import concurrent.futures
import multiprocessing
import numpy as np
from umi.tag_tracking.camera_model import FisheyeCamera
from umi.tag_tracking.noise_model import TagNoiseModel
from umi.tag_tracking.rigid_body import track_rigid_body, calibrate_rigid_body

CHUNKS_PER_WORKER = 4
# IMU samples sent with a fusion window beyond its first / last frame (s)
IMU_MARGIN_SEC = 1.0


def make_executor(num_workers: Optional[int] = None) -> concurrent.futures.ProcessPoolExecutor:
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    return concurrent.futures.ProcessPoolExecutor(max_workers=num_workers,
        mp_context=multiprocessing.get_context('spawn'))


def _track_chunk(camera_json, frames, T_body_tags, tag_sizes, noise_model_dict, line_delay):
    return track_rigid_body(FisheyeCamera(camera_json), frames, T_body_tags, tag_sizes,
        TagNoiseModel.from_dict(noise_model_dict), line_delay)


def _calibrate_body(camera_json, frames, tag_ids, tag_sizes, noise_model_dict, anchor_tag, line_delay):
    return calibrate_rigid_body(FisheyeCamera(camera_json), frames, tag_ids, tag_sizes,
        TagNoiseModel.from_dict(noise_model_dict), anchor_tag, line_delay)


def chunk_bounds(frames: List[dict], n_chunks: int) -> List[tuple]:
    """
    (start, stop, lo, hi) per chunk: frames [start, stop) are kept, frames
    [lo, hi) are tracked (one extra neighbor per side within the same sequence).
    """
    n = len(frames)
    n_chunks = max(1, min(n_chunks, n))
    edges = np.linspace(0, n, n_chunks + 1).round().astype(int)
    seq = lambda i: frames[i].get('sequence')
    out = list()
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop <= start:
            continue
        lo = start - 1 if start > 0 and seq(start - 1) == seq(start) else start
        hi = stop + 1 if stop < n and seq(stop) == seq(stop - 1) else stop
        out.append((int(start), int(stop), int(lo), int(hi)))
    return out


def parallel_track_rigid_body(camera: FisheyeCamera, frames: List[dict], T_body_tags: Dict[int, np.ndarray],
        tag_sizes: Dict[int, float], noise_model: TagNoiseModel, line_delay: float,
        executor: concurrent.futures.Executor, n_chunks: int) -> Dict[str, np.ndarray]:
    """
    track_rigid_body on chunks of frames in the executor's worker processes.
    """
    return gather_track(submit_track(executor, camera, frames, T_body_tags, tag_sizes,
        noise_model, line_delay, n_chunks))


def submit_track(executor, camera, frames, T_body_tags, tag_sizes, noise_model, line_delay, n_chunks):
    """
    Submit the chunks of one tracking without waiting; collect with gather_track.
    Lets several trackings (bodies, line delay candidates) share the workers.
    """
    bounds = chunk_bounds(frames, n_chunks)
    futures = [executor.submit(_track_chunk, camera.intrinsics_json, frames[lo:hi], T_body_tags,
        tag_sizes, noise_model.to_dict(), line_delay) for _, _, lo, hi in bounds]
    return bounds, futures


def gather_track(submitted) -> Dict[str, np.ndarray]:
    bounds, futures = submitted
    parts = list()
    for (start, stop, lo, hi), future in zip(bounds, futures):
        result = future.result()
        parts.append({k: v[start - lo:start - lo + (stop - start)] for k, v in result.items()})
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def parallel_calibrate_bodies(camera: FisheyeCamera, body_frames: Dict[str, List[dict]],
        body_configs: Dict[str, dict], tag_sizes: Dict[int, float], noise_model: TagNoiseModel,
        line_delay: float, executor: concurrent.futures.Executor) -> Dict[str, dict]:
    """
    calibrate_rigid_body of every body at once, one worker process each.
    """
    futures = {name: executor.submit(_calibrate_body, camera.intrinsics_json, body_frames[name],
        body['tags'], tag_sizes, noise_model.to_dict(), body.get('anchor_tag'), line_delay)
        for name, body in body_configs.items()}
    return {name: future.result() for name, future in futures.items()}


def _fuse_window(camera_json, frames, T_world_tags, tag_sizes, noise_model_dict, imu, T_imu_cam,
        time_offset, imu_noise, T_world_cam_init, pose_measurements=None):
    from umi.tag_tracking.visual_inertial import fuse_camera_trajectory
    return fuse_camera_trajectory(FisheyeCamera(camera_json), frames, T_world_tags, tag_sizes,
        TagNoiseModel.from_dict(noise_model_dict), imu, T_imu_cam, time_offset, imu_noise, T_world_cam_init,
        pose_measurements=pose_measurements)


def submit_fuse(executor, camera: FisheyeCamera, frames: List[dict], T_world_tags, tag_sizes,
        noise_model: TagNoiseModel, imu: dict, T_imu_cam, time_offset: float, imu_noise: dict, T_world_cam_init,
        pose_measurements=None):
    """
    Visual-inertial smoothing (umi.tag_tracking.visual_inertial.fuse_camera_trajectory)
    of one window of frames in a worker process; returns the future.
    """
    # only the IMU samples around the window's frames go to the worker (camera time = IMU time + offset)
    t0, t1 = frames[0]['time'] - time_offset - IMU_MARGIN_SEC, frames[-1]['time'] - time_offset + IMU_MARGIN_SEC
    lo, hi = np.searchsorted(imu['time'], [t0, t1])
    imu = {k: v[lo:hi] for k, v in imu.items()}
    return executor.submit(_fuse_window, camera.intrinsics_json, frames, T_world_tags, tag_sizes,
        noise_model.to_dict(), imu, T_imu_cam, time_offset, imu_noise, T_world_cam_init, pose_measurements)
