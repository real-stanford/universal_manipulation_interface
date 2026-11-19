"""
Independent check of the frame sync between the head and the wrist videos,
as exported (dataset_plan.pkl: 07_export_mcap.py / 13_export_delivery.py
timestamp wrist video frame i at frame0_time + i / fps, head frames at the
head video's own times). Writes nothing.

The sync chain being checked: audio sync (06a) -> wrist IMU fitted to the
head-tracked cube rotation (03s) -> wrist video placed from its IMU by the
camera-IMU time offset (imu_time_offset). The wrist video itself is never
compared with the head video in that chain; here it is, without the IMUs:

1. video: the wrist camera's orientation from the environment tags its own
   video sees (per wrist frame, at the time the tags' image rows were read)
   against the same camera's orientation from the head video (cube pose x
   T_cube_wrist, at the head observation times). The shift d (ms) of the
   head-side curve that best matches the wrist-side one is the wrist video's
   frame sync error: its exported timestamps are d too early (d > 0) or too
   late (d < 0). A constant calibration rotation is removed before
   comparing, so only the timing remains. 0 ms = the frames are in sync.
2. imu: the wrist gyro (as exported: frame0_time + IMU time + imu_time_offset)
   against the rotation rate of that camera's own env-tag orientation:
   d > 0: the exported IMU samples are d too late relative to the wrist video.

Per plan episode and wrist camera; the error curve's value at 0 and at d
shows how strongly the data prefer d. Episodes need motion and wrist frames
with >= --min_tags environment tags.

python scripts_tag_tracking/check_frame_sync.py -i data_workspace/session -c umi_calib/tag_tracking.yaml -e 0,5,10
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
from scipy.spatial.transform import Rotation
from umi.tag_tracking.config import load_tag_tracking_config
from umi.tag_tracking.session import get_session_videos
from umi.tag_tracking.noise_model import TagNoiseModel
from umi.tag_tracking.parallel import make_executor, parallel_track_rigid_body
from umi.tag_tracking.imu import load_imu
from umi.tag_tracking.delivery import align_poses
from umi.tag_tracking.wrist_sync import observation_times, cube_image_rows, head_readout, gyro_integral, mean_rate

SHIFTS_MS = np.arange(-100.0, 100.5, 0.5)


def rotation_mismatch(R_head: Rotation, R_vis: Rotation) -> float:
    """
    RMS angle (deg) between two orientation series after removing their mean relative rotation.
    """
    rel = R_head.inv() * R_vis
    residual = rel.mean().inv() * rel
    return float(np.degrees(np.sqrt(np.mean(residual.magnitude() ** 2))))


def best_shift(shifts: np.ndarray, cost: np.ndarray):
    """
    Minimum of a cost curve, refined by a parabola through its neighbors.
    """
    k = int(np.nanargmin(cost))
    if 0 < k < len(shifts) - 1 and np.all(np.isfinite(cost[k - 1:k + 2])):
        a, b, c = cost[k - 1:k + 2]
        denominator = a - 2 * b + c
        if denominator > 0:
            return float(shifts[k] + 0.5 * (a - c) / denominator * (shifts[1] - shifts[0])), float(cost[k])
    return float(shifts[k]), float(cost[k])


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Session directory with dataset_plan.pkl')
@click.option('-c', '--config', required=True, help='Tag tracking config yaml')
@click.option('-e', '--episodes', default=None, help='Comma separated plan episode indices, default all')
@click.option('--min_tags', type=int, default=1, help='Environment tags a wrist frame needs')
@click.option('-j', '--num_workers', type=int, default=None, help='Tracking processes, default all cores')
def main(input, config, episodes, min_tags, num_workers):
    session = pathlib.Path(os.path.expanduser(input)).absolute()
    config = load_tag_tracking_config(config)
    plan = pickle.load(session.joinpath('dataset_plan.pkl').open('rb'))
    indices = list(range(len(plan))) if episodes is None else [int(x) for x in episodes.split(',')]
    bodies = json.load(session.joinpath('tag_tracking', 'bodies.json').open('r'))
    line_delay = bodies['line_delay']
    T_env_tags = {int(k): np.array(v) for b in bodies['bodies'].values() if b['static']
        for k, v in b['T_body_tags'].items()}
    tcp = json.load(session.joinpath('tag_tracking', 'gripper_tcp.json').open('r'))
    names = [n for _, n in sorted((x['robot_index'], n) for n, x in tcp.items())]
    videos = {v['name']: v for v in get_session_videos(session)}
    head_camera = config['cameras'][config['head_camera']]
    height = head_camera.image_size[1]
    accuracy = config['accuracy']
    executor = make_executor(num_workers)
    n_chunks = 4 * (num_workers or os.cpu_count())
    wrist_noise = TagNoiseModel(0.0, {t: config['sigma_px'] for t in T_env_tags})
    cache, readouts, results = dict(), dict(), list()

    for k in indices:
        e = plan[k]
        cameras = e['cameras']
        n_grippers = len(e['grippers'])
        assert len(cameras) > n_grippers, "the plan has no head camera"
        head = videos[pathlib.Path(cameras[-1]['video_path']).parent.name]
        if head['name'] not in cache:
            track = pickle.load(head['dir'].joinpath('tag_tracking.pkl').open('rb'))
            cache[head['name']] = track
            readouts[head['name']] = head_readout(head)
        track = cache[head['name']]
        t_first, t_last = float(e['episode_timestamps'][0]), float(e['episode_timestamps'][-1])
        for g in range(n_grippers):
            name = names[g]
            camera = cameras[g]
            wrist = videos[pathlib.Path(camera['video_path']).parent.name]
            serial = wrist['camera_serial']
            wrist_cam = config['cameras'][serial]
            if wrist['name'] not in readouts:
                readouts[wrist['name']] = head_readout(wrist)
            # head side: the wrist camera's world pose through the cube, at the head observation times
            body = track['bodies'][name]
            rows = cube_image_rows(body['T_cam_body'], head_camera, height)
            head_t = head['start_timestamp'] + observation_times(track['time'], rows, readouts[head['name']],
                line_delay, height)
            T_head = body['T_world_body'] @ np.array(tcp[name]['T_cube_wrist'])
            head_ok = body['is_valid_world'] & ~np.isnan(T_head[:, 0, 0])
            order = np.argsort(head_t)
            head_t, T_head, head_ok = head_t[order], T_head[order], head_ok[order]
            spacing = 1.5 * float(np.median(np.diff(track['time'])))

            # wrist side: its own env-tag pose per frame, at the time its tag rows were read (no rolling
            # shutter correction for the wrist camera: line delay 0)
            detections = pickle.load(wrist['dir'].joinpath('tag_detection.pkl').open('rb'))
            frames, times = list(), list()
            for d in detections:
                t = camera['frame0_time'] + d['time']
                if not (t_first <= t <= t_last):
                    continue
                corners = {i: v['corners'] for i, v in d['tag_dict'].items() if i in T_env_tags}
                if len(corners) < min_tags:
                    continue
                row = float(np.mean([c[:, 1].mean() for c in corners.values()])) / wrist_cam.image_size[1]
                frames.append({'time': d['time'], 'corners': corners})
                times.append(t + readouts[wrist['name']] * row)
            label = f"episode {k} {name} ({serial})"
            if len(frames) < 30:
                print(f"{label}: {len(frames)} wrist frames with >= {min_tags} environment tags, skipped")
                continue
            vis = parallel_track_rigid_body(wrist_cam, frames, T_env_tags, config['tag_sizes'], wrist_noise, 0.0,
                executor, n_chunks)
            ok = (vis['position_std'] <= accuracy['position_std']) & (vis['rotation_std'] <= accuracy['rotation_std_rad']) \
                & vis['is_consistent'] & ~np.isnan(vis['T_cam_body'][:, 0, 0])
            times = np.array(times)[ok]
            T_world_wrist = np.linalg.inv(vis['T_cam_body'][ok])
            if len(times) < 30:
                print(f"{label}: {len(times)} accurate wrist env-tag poses, skipped")
                continue
            R_vis = Rotation.from_matrix(T_world_wrist[:, :3, :3])

            # 1. video: shift of the head-side curve matching the wrist video's own poses
            cost = np.full(len(SHIFTS_MS), np.nan)
            n_used = 0
            for s, shift in enumerate(SHIFTS_MS):
                pose, aligned = align_poses(head_t, T_head, head_ok, times + shift * 1e-3, spacing)
                if aligned.sum() >= 30:
                    cost[s] = rotation_mismatch(Rotation.from_quat(pose[aligned, 3:]), R_vis[aligned])
                    if shift == 0:
                        n_used = int(aligned.sum())
            if np.all(np.isnan(cost)):
                print(f"{label}: no head-tracked poses at the wrist frames, skipped")
                continue
            video_ms, video_cost = best_shift(SHIFTS_MS, cost)
            zero_cost = float(cost[np.argmin(np.abs(SHIFTS_MS))])
            # how well the data separate a one-frame error from none: mismatch at +-1 frame
            frame_ms = 1000.0 / float(wrist['fps'])
            at = lambda ms: float(np.interp(ms, SHIFTS_MS, cost))
            one_frame = min(at(-frame_ms), at(frame_ms))

            # 2. imu: gyro (as exported) against the rotation rate of the wrist video's own poses
            consecutive = np.flatnonzero(np.diff(times) < 1.5 / float(wrist['fps']))
            angle = (R_vis[consecutive].inv() * R_vis[consecutive + 1]).magnitude()
            dt = times[consecutive + 1] - times[consecutive]
            rate_vis = angle / dt
            imu = load_imu(str(wrist['dir'].joinpath('imu_data.json')))
            imu_t = camera['frame0_time'] + imu['time'] + (camera.get('imu_time_offset') or 0.0)
            integral = gyro_integral(imu_t, imu['gyro'])
            imu_cost = np.full(len(SHIFTS_MS), np.nan)
            for s, shift in enumerate(SHIFTS_MS):
                t0 = times[consecutive] + shift * 1e-3
                t1 = times[consecutive + 1] + shift * 1e-3
                inside = (t0 >= imu_t[0]) & (t1 <= imu_t[-1])
                if inside.sum() >= 30:
                    rate = np.linalg.norm(mean_rate(imu_t, integral, t0[inside], t1[inside]), axis=1)
                    imu_cost[s] = float(np.sqrt(np.mean((rate - rate_vis[inside]) ** 2)))
            imu_ms = best_shift(SHIFTS_MS, imu_cost)[0] if np.any(np.isfinite(imu_cost)) else np.nan

            speed = float(np.degrees(np.median(rate_vis))) if len(rate_vis) else np.nan
            results.append({'episode': k, 'name': name, 'video': wrist['name'], 'video_ms': video_ms, 'imu_ms': imu_ms,
                'n': n_used, 'separation': one_frame / max(video_cost, 1e-9)})
            print(f"{label}: {n_used} frames, median rotation {speed:.0f} deg/s | video sync error "
                f"{video_ms:+.1f} ms ({video_ms / frame_ms:+.2f} frames) | mismatch {video_cost:.2f} deg at the best "
                f"shift, {zero_cost:.2f} at 0 ms, {one_frame:.2f} at +-1 frame | IMU vs wrist video {imu_ms:+.1f} ms")
        # the three camera pairs of this episode (head is the reference of both wrist measurements)
        here = {r['name']: r for r in results if r['episode'] == k and r['separation'] > 1.2}
        if len(here) == 2:
            a, b = (here[n] for n in names[:2])
            print(f"episode {k} pairs: head-{a['name']} {a['video_ms']:+.1f} ms, head-{b['name']} {b['video_ms']:+.1f} ms, "
                f"{a['name']}-{b['name']} {a['video_ms'] - b['video_ms']:+.1f} ms")
    executor.shutdown()

    if results:
        print("\nper wrist video (median over episodes, 5-95%):")
        for video in sorted({r['video'] for r in results}):
            # episodes whose mismatch at +-1 frame is clearly above the best (enough motion to time it)
            v = np.array([r['video_ms'] for r in results if r['video'] == video and r['separation'] > 1.2])
            m = np.array([r['imu_ms'] for r in results if r['video'] == video])
            m = m[np.isfinite(m)]
            name = [r['name'] for r in results if r['video'] == video][0]
            if len(v) == 0:
                print(f"  {name} {video}: no episode with enough motion to time a one-frame error")
                continue
            print(f"  {name} {video}: video sync error {np.median(v):+.1f} ms ({np.percentile(v, 5):+.1f}..{np.percentile(v, 95):+.1f}, "
                f"{len(v)} episodes with mismatch at +-1 frame > 1.2x the best); IMU vs wrist video "
                f"{np.median(m) if len(m) else np.nan:+.1f} ms")
        pairs = list()
        for k in sorted({r['episode'] for r in results}):
            here = {r['name']: r for r in results if r['episode'] == k and r['separation'] > 1.2}
            if len(here) == 2:
                pairs.append(here[names[0]]['video_ms'] - here[names[1]]['video_ms'])
        if pairs:
            print(f"  {names[0]}-{names[1]} (wrist to wrist, episodes where both are timed): {np.median(pairs):+.1f} ms "
                f"({np.percentile(pairs, 5):+.1f}..{np.percentile(pairs, 95):+.1f}, {len(pairs)} episodes)")
        print("video sync error: + = the exported wrist frames are stamped too early (the head shows that moment later); "
            "0 = in sync. One frame = 16.7 ms.")

# %%
if __name__ == "__main__":
    main()
