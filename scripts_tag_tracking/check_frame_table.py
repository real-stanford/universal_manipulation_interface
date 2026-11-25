"""
Build the per-frame delivery data of every episode (umi.tag_tracking.delivery)
and check it. Writes nothing.

Checks:
A. Structure: rows = head frames of the task interval, frame intervals uniform
   (max deviation from 1/fps), quaternions unit, values null exactly where
   pose_valid / gripper validity says so.
B. Independent cross-check: the aligned (fused, wrist-clock) TCP against the
   head camera's own cube measurement x T_cube_tcp at the same head frames:
   position / rotation difference, and the remaining time offset between them
   (difference regressed on the head-measured velocity: a sync error of d ms
   shows as difference = velocity * d), from positions and from rotations
   (a constant calibration error is a constant angle and drops out). ~0 ms when
   the wrist sync, the IMU time offset and the head timing are consistent.
C. Physics: TCP speed, lowest TCP height (fingertips at the table: ~0 cm),
   gripper width (opening relative to fully closed, as in UMI) within 0 .. its
   calibrated open width, share of widths from one finger tag.
D. Coverage: valid share of aligned rows and of the native streams.

python scripts_tag_tracking/check_frame_table.py -i data_workspace/session -c umi_calib/tag_tracking.yaml
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
from umi.tag_tracking.wrist_sync import wrist_corrections, head_readout, cube_image_rows, observation_times
from umi.tag_tracking.delivery import episode_frames, gripper_streams, pose7, align_poses, SPACING_FACTOR


def residual_time_offset(diff: np.ndarray, velocity: np.ndarray) -> float:
    """
    d (s) in diff = velocity * d + bias (per axis bias), least squares over rows.
    """
    v = velocity - velocity.mean(axis=0)
    r = diff - diff.mean(axis=0)
    denominator = np.sum(v * v)
    return float(np.sum(v * r) / denominator) if denominator > 0 else np.nan


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Session directory (tracked)')
@click.option('-c', '--config', required=True, help='Tag tracking config yaml')
def main(input, config):
    session = pathlib.Path(os.path.expanduser(input)).absolute()
    config = load_tag_tracking_config(config)
    tcp = json.load(session.joinpath('tag_tracking', 'gripper_tcp.json').open('r'))
    episodes = json.load(session.joinpath('tag_tracking', 'episodes.json').open('r'))['videos']
    sync_path = session.joinpath('tag_tracking', 'wrist_sync.json')
    sync = json.load(sync_path.open('r')) if sync_path.is_file() else None
    videos = get_session_videos(session)
    line_delay = json.load(session.joinpath('tag_tracking', 'bodies.json').open('r'))['line_delay']
    head_camera = config['cameras'][config['head_camera']]
    height = head_camera.image_size[1]

    # head videos 00b_segment_episodes.py skipped (no wrist video overlapping) have no episodes
    for head in [x for x in videos if x['camera_serial'] == config['head_camera'] and x['name'] in episodes]:
        track = pickle.load(head['dir'].joinpath('tag_tracking.pkl').open('rb'))
        corrections = wrist_corrections(sync, head['name'])
        grippers = gripper_streams(session, config, videos, head, track, tcp, corrections)
        fps = float(head['fps'])
        readout = head_readout(head)
        cube_times = {g['name']: observation_times(track['time'], cube_image_rows(
            track['bodies'][g['name']]['T_cam_body'], head_camera, height), readout, line_delay, height) for g in grippers}
        print(f"{head['name']}: wrist sync {'applied' if corrections else 'NOT applied (no wrist_sync.json)'}, "
            f"poses {', '.join(g['side'] + ' ' + g['pose_source'] for g in grippers)}")
        totals = {g['side']: {'rows': 0, 'aligned': 0, 'native': 0, 'native_valid': 0, 'width': 0, 'one_tag': 0,
            'width_measured': 0, 'dp': [], 'dr': [], 'diff': [], 'vel': [], 'ang_diff': [], 'ang_vel': [], 'speed': [], 'z': [], 'width_out': 0}
            for g in grippers}
        n_rows = interval_issues = 0
        frame_jitter = 0.0
        quat_error = 0.0
        null_mismatch = 0
        for k, e in enumerate(episodes[head['name']]['episodes']):
            start, end = (e['task_start'], e['task_end']) if e.get('task_start') is not None else (e['start'], e['end'])
            if 'task_start' in e and e['task_start'] is None:
                continue
            out = episode_frames(track['time'], track['head']['T_world_cam'], track['head']['is_valid'], start, end,
                grippers, row_time_offset=readout / 2)
            f = out['frames']
            n_rows += len(f['time'])
            # A. structure
            expected = np.sum((track['time'] >= start) & (track['time'] < end))
            interval_issues += int(expected != len(f['time']))
            if len(f['time']) > 1:
                frame_jitter = max(frame_jitter, float(np.max(np.abs(np.diff(f['time']) - 1 / fps))))
            for g in grippers:
                side = g['side']
                pose = f[f'ee_pose_{side}']
                ok = ~np.isnan(pose[:, 0])
                null_mismatch += int(np.sum(ok != ~np.any(np.isnan(pose), axis=1)))
                if ok.any():
                    quat_error = max(quat_error, float(np.max(np.abs(np.linalg.norm(pose[ok, 3:], axis=1) - 1))))
                t = totals[side]
                t['rows'] += len(ok)
                t['aligned'] += int(ok.sum())
                nat = out['native'][side]
                t['native'] += len(nat['pose_valid'])
                t['native_valid'] += int(nat['pose_valid'].sum())
                t['width'] += len(nat['width'])
                t['width_measured'] += int(np.sum(~np.isnan(nat['width'])))
                t['one_tag'] += int(np.sum(nat['width_tags'] == 1))
                w = nat['width'][~np.isnan(nat['width'])]
                # opening relative to fully closed (UMI): 0 .. max - min finger tag distance
                r = g['gripper_range']
                t['width_out'] += int(np.sum((w < -0.005) | (w > r['max_width'] - r['min_width'] + 0.005)))
                # B. the head's own cube measurement, against the fused pose at the moment the head observed it
                body = track['bodies'][g['name']]
                rows = f['head_frame_index']
                head_tcp = body['T_world_body'][rows] @ g['T_cube_tcp']
                obs = cube_times[g['name']][rows]
                period = float(np.median(np.diff(g['pose_times'])))
                fused, fused_ok = align_poses(g['pose_times'], g['T_world_tcp'],
                    g['pose_valid'] & ~np.isnan(g['T_world_tcp'][:, 0, 0]), obs, SPACING_FACTOR * period)
                head_ok = body['is_valid_world'][rows] & ~np.isnan(head_tcp[:, 0, 0])
                both = fused_ok & head_ok
                if both.sum() > 10:
                    hp_all = pose7(head_tcp)
                    t['dp'].append(np.linalg.norm(fused[both, :3] - hp_all[both, :3], axis=1))
                    t['dr'].append((Rotation.from_quat(fused[both, 3:]).inv() * Rotation.from_quat(hp_all[both, 3:])).magnitude())
                    # velocity from the head measurement (central differences on consecutive measured frames)
                    idx = np.flatnonzero(both)
                    consecutive = (idx[2:] - idx[:-2]) == 2
                    if consecutive.any():
                        mid = idx[1:-1][consecutive]
                        prev, nxt = idx[:-2][consecutive], idx[2:][consecutive]
                        vel = (hp_all[nxt, :3] - hp_all[prev, :3]) / (obs[nxt] - obs[prev])[:, None]
                        t['diff'].append(fused[mid, :3] - hp_all[mid, :3])
                        t['vel'].append(vel)
                        # rotation only: a constant calibration error is a constant angle, removed by the fit
                        R_prev, R_next = Rotation.from_quat(hp_all[prev, 3:]), Rotation.from_quat(hp_all[nxt, 3:])
                        t['ang_vel'].append((R_next * R_prev.inv()).as_rotvec() / (obs[nxt] - obs[prev])[:, None])
                        t['ang_diff'].append((Rotation.from_quat(fused[mid, 3:]) * Rotation.from_quat(hp_all[mid, 3:]).inv()).as_rotvec())
                # C. physics
                if ok.sum() > 2:
                    k_ok = np.flatnonzero(ok)
                    step = np.diff(k_ok) == 1
                    speed = np.linalg.norm(np.diff(pose[k_ok, :3], axis=0), axis=1)[step] * fps
                    t['speed'].append(speed)
                    t['z'].append(pose[ok, 2])

        print(f"  A structure: {n_rows} rows, interval mismatches {interval_issues}, frame interval max deviation "
            f"{frame_jitter * 1e6:.1f} us, quaternion norm max error {quat_error:.1e}, null pattern mismatches {null_mismatch}")
        for side, t in totals.items():
            dp = np.concatenate(t['dp']) if t['dp'] else np.array([])
            dr = np.concatenate(t['dr']) if t['dr'] else np.array([])
            offset = residual_time_offset(np.concatenate(t['diff']), np.concatenate(t['vel'])) if t['diff'] else np.nan
            rot_offset = residual_time_offset(np.concatenate(t['ang_diff']), np.concatenate(t['ang_vel'])) \
                if t['ang_diff'] else np.nan
            speed = np.concatenate(t['speed']) if t['speed'] else np.array([0.0])
            z = np.concatenate(t['z']) if t['z'] else np.array([np.nan])
            print(f"  {side}:")
            if len(dp):
                print(f"    B vs head cube measurement ({len(dp)} frames): position median {np.median(dp) * 1e3:.1f} mm, "
                    f"95% {np.percentile(dp, 95) * 1e3:.1f} mm; rotation median {np.degrees(np.median(dr)):.2f} deg, "
                    f"95% {np.degrees(np.percentile(dr, 95)):.2f} deg; remaining time offset {offset * 1e3:+.2f} ms (position), "
                    f"{rot_offset * 1e3:+.2f} ms (rotation, independent of the calibration)")
            print(f"    C physics: TCP speed 99% {np.percentile(speed, 99):.2f} m/s max {speed.max():.2f} m/s, "
                f"lowest TCP {np.nanmin(z) * 100:.1f} cm (1st percentile {np.nanpercentile(z, 1) * 100:.1f} cm), "
                f"widths outside the calibrated range {t['width_out']}, from one finger tag "
                f"{t['one_tag'] / max(t['width_measured'], 1) * 100:.1f}%")
            g = [x for x in grippers if x['side'] == side][0]
            calibrated = g['gripper_range'].get('finger_center_x')
            print(f"    finger midpoint: calibration video "
                + ('not measured (rerun gripper_range), one-tag widths assume the camera axis' if calibrated is None
                    else f"{calibrated * 1000:+.2f} mm") + ", this session's two-tag frames "
                + ('-' if g['session_center_x'] is None else f"{g['session_center_x'] * 1000:+.2f} mm")
                + ('' if calibrated is None or g['session_center_x'] is None else
                    f" (difference {(g['session_center_x'] - calibrated) * 1000:+.2f} mm, one-tag width error up to "
                    f"{2 * abs(g['session_center_x'] - calibrated) * 1000:.1f} mm)"))
            print(f"    D coverage: aligned pose {t['aligned'] / max(t['rows'], 1) * 100:.1f}% of rows, native pose valid "
                f"{t['native_valid'] / max(t['native'], 1) * 100:.1f}% of {t['native']} wrist frames, width measured "
                f"{t['width_measured'] / max(t['width'], 1) * 100:.1f}%")

# %%
if __name__ == "__main__":
    main()
