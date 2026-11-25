"""
Export tag tracking results as dataset_plan.pkl in the format of
scripts_slam_pipeline/06_generate_dataset_plan.py, so that
07_generate_replay_buffer.py and training run unchanged.

Per head camera video and combination of overlapping wrist videos (a wrist
camera stopped and restarted during the take records several), one demo
spans the time all its cameras record (synchronized start times); episodes
partly outside it (a wrist video stopped during them) are dropped. Following 06:
- frames of all cameras are aligned on a common grid of the robot0 camera;
- TCP pose per gripper: fused wrist camera pose (03c_fuse_gripper_imu.py) *
  T_wrist_tcp if available (world reference), else
  T_ref_cube (tag tracking) * T_cube_tcp (05),
  interpolated to the grid only between two valid tracked head frames;
- gripper width from the wrist camera finger tags (04 tag_detection.pkl)
  and gripper_range.json, as in 06;
- wrist clocks: audio sync start times + the motion sync correction of
  tag_tracking/wrist_sync.json (03s_sync_wrists.py) at each episode: steps at
  the robot0 camera frames' corrected times, each camera's frames shifted by
  whole frames; per camera frame0_time (corrected session time of frame 0),
  clock_correction_ms and imu_time_offset (wrist cameras) for 07;
- a step is valid if all grippers' poses are valid; valid segments
  shorter than min_episode_length are dropped; one episode per segment.
With an 'episodes' section in the config (continuous takes), episodes come
from 00b_segment_episodes.py (tag_tracking/episodes.json) instead, their task
interval (task_start / task_end, 03t_trim_docking.py: docking removed) if present:
- poses are interpolated across tracking gaps up to episodes.max_gap_sec;
- invalid steps at an episode's start / end (the bang motions) are trimmed,
  at most --max_trim_sec per side; the rest must be fully valid, else the
  episode is dropped (not split); demo_start_pose / demo_end_pose are per episode;
- --keep_gaps: episodes with longer gaps inside are kept whole instead; at
  those steps the pose is not available: tcp_pose NaN, pose_valid False
  (per gripper, every step). Such plans are for the per-frame deliveries
  (07_export_mcap.py writes no pose message there), not for
  07_generate_replay_buffer.py.
--ignore_poses (episode videos only, 15_export_episode_videos.py; run_tag_tracking_pipeline.py
--videos_only, without fuse_grippers): every episode is kept whole, whatever its poses (no pose
based trimming or dropping); its frame ranges per camera (wrist clocks as above) and gripper widths
are what is used, the poses are not.
Grippers are ordered by robot_index; cameras are the grippers' wrist
cameras in that order, then optionally the head camera.

--reference world: poses in the static body (environment) frame.
--reference head: poses in the head camera frame, only meaningful if
the head camera did not move.

python scripts_tag_tracking/06_export_dataset_plan.py -i data_workspace/session -c umi_calib/tag_tracking.yaml
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import itertools
import json
import math
import pathlib
import pickle
import click
import numpy as np
from umi.common.pose_util import mat_to_pose
from umi.common.interpolation_util import PoseInterpolator, get_interp1d
from umi.tag_tracking.config import load_tag_tracking_config
from umi.tag_tracking.session import get_session_videos, overlapping
from umi.tag_tracking.wrist_sync import wrist_corrections, correction_seconds
from umi.tag_tracking.imu import load_cam_imu_calibration, wrist_imu_time_offset
from umi.tag_tracking.delivery import load_gripper_range, finger_widths


# %%
def get_bool_segments(bool_seq):
    """
    Same as scripts_slam_pipeline/06_generate_dataset_plan.py
    """
    bool_seq = np.array(bool_seq, dtype=bool)
    segment_ends = (np.nonzero(np.diff(bool_seq))[0] + 1).tolist()
    segment_bounds = [0] + segment_ends + [len(bool_seq)]
    segments = list()
    segment_type = list()
    for i in range(len(segment_bounds) - 1):
        segments.append(slice(segment_bounds[i], segment_bounds[i + 1]))
        segment_type.append(bool_seq[segment_bounds[i]])
    return segments, np.array(segment_type, dtype=bool)

def interpolate_valid_poses(times, poses, is_valid, query_times, max_gap_sec=None):
    """
    Poses (N,4,4) at times interpolated to query_times; a query time is
    valid only if the two samples around it are both valid, or, with
    max_gap_sec, if the valid samples around it are at most max_gap_sec apart.
    """
    query_valid = np.zeros(len(query_times), dtype=bool)
    if max_gap_sec is None:
        idx = np.searchsorted(times, query_times)
        inside = (idx > 0) & (idx < len(times))
        query_valid[inside] = is_valid[idx[inside] - 1] & is_valid[idx[inside]]
    else:
        valid_times = times[is_valid]
        idx = np.searchsorted(valid_times, query_times)
        inside = (idx > 0) & (idx < len(valid_times))
        query_valid[inside] = (valid_times[idx[inside]] - valid_times[idx[inside] - 1]) <= max_gap_sec
    out = np.full((len(query_times), 6), np.nan)
    if is_valid.sum() >= 2 and query_valid.any():
        interp = PoseInterpolator(times[is_valid], mat_to_pose(poses[is_valid]))
        out[query_valid] = interp(query_times[query_valid])
    return out, query_valid

def bridged_gaps(times, is_valid, query_times):
    """
    Per query time, the time between the valid samples around it, and whether
    that spans lost samples (more than 1.5 sample periods): the pose there is
    interpolated across a tracking gap, not measured.
    """
    valid_times = times[is_valid]
    idx = np.clip(np.searchsorted(valid_times, query_times), 1, max(len(valid_times) - 1, 1))
    span = valid_times[idx] - valid_times[idx - 1] if len(valid_times) >= 2 else np.full(len(query_times), np.inf)
    period = np.median(np.diff(times)) if len(times) >= 2 else np.inf
    return span, span > 1.5 * period


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Session directory')
@click.option('-c', '--config', required=True, help='Tag tracking config yaml')
@click.option('-o', '--output', default=None, help='Default <input>/dataset_plan.pkl')
@click.option('-ws', '--wrist_sync', default=None, help='Default <input>/tag_tracking/wrist_sync.json (if present)')
@click.option('-r', '--reference', type=click.Choice(['world', 'head']), default='world')
@click.option('-ml', '--min_episode_length', type=int, default=24, help='Same default as 06')
@click.option('-nh', '--no_head_camera', is_flag=True, default=False, help='Exclude head camera images')
@click.option('-mt', '--max_trim_sec', type=float, default=1.0,
    help='Episodes: trim invalid steps at the start / end, at most this long per side (s)')
@click.option('-kg', '--keep_gaps', is_flag=True, default=False,
    help='Episodes: keep episodes with tracking gaps over max_gap_sec inside, pose not available (NaN) there')
@click.option('-ip', '--ignore_poses', is_flag=True, default=False,
    help='Episode videos only: keep every episode whole whatever its poses (frame ranges and widths are used)')
def main(input, config, output, wrist_sync, reference, min_episode_length, no_head_camera, max_trim_sec, keep_gaps,
        ignore_poses):
    session = pathlib.Path(os.path.expanduser(input)).absolute()
    if output is None:
        output = session.joinpath('dataset_plan.pkl')
    config = load_tag_tracking_config(config)
    hardware = config['gripper_hardware']
    tcp_calibration = json.load(session.joinpath('tag_tracking', 'gripper_tcp.json').open('r'))
    gripper_bodies = sorted([(b['robot_index'], name) for name, b in config['bodies'].items()
        if 'gripper_camera' in b])
    videos = get_session_videos(session)
    # continuous takes: episodes from 00b_segment_episodes.py
    episodes = None
    max_gap_sec = None
    if 'episodes' in config:
        episodes_path = session.joinpath('tag_tracking', 'episodes.json')
        assert episodes_path.is_file(), "run 00b_segment_episodes.py first"
        episodes = json.load(episodes_path.open('r'))['videos']
        max_gap_sec = config['episodes']['max_gap_sec']

    # wrist clocks: audio sync start times + motion sync correction (03s_sync_wrists.py)
    sync_path = session.joinpath('tag_tracking', 'wrist_sync.json') if wrist_sync is None else pathlib.Path(wrist_sync)
    sync = json.load(sync_path.open('r')) if sync_path.is_file() else None
    if sync is None:
        print("no tag_tracking/wrist_sync.json (03s_sync_wrists.py): wrist clocks from the audio sync only")

    all_plans = list()
    total_time = 0.0
    used_time = 0.0
    bridge_stats = [{'steps': 0, 'bridged': 0, 'longest': 0.0, 'missing': 0} for _ in gripper_bodies]
    # head videos 00b_segment_episodes.py skipped (no wrist video overlapping) have no episodes
    for head in [x for x in videos if x['camera_serial'] == config['head_camera']
            and (episodes is None or x['name'] in episodes)]:
        track = pickle.load(head['dir'].joinpath('tag_tracking.pkl').open('rb'))
        head_times = head['start_timestamp'] + track['time']
        corrections = wrist_corrections(sync, head['name'])

        def imu_time_offset(video):
            serial = video['camera_serial']
            if serial == config['head_camera'] or serial not in config.get('gripper_imu', dict()):
                return None
            cam_imu = load_cam_imu_calibration(config['gripper_imu'][serial]['cam_imu_calibration'])
            return wrist_imu_time_offset(config['gripper_imu'][serial], cam_imu,
                str(video['dir'].joinpath('raw_video.mp4')), float(1 / video['fps']))

        def correction(video, t_session):
            """
            Seconds to add to a video's times (session clock) at session time(s) t; 0 for the head.
            """
            if video['name'] not in corrections:
                return np.zeros_like(np.asarray(t_session, dtype=float))
            return correction_seconds(corrections[video['name']], np.asarray(t_session) - head['start_timestamp'])

        # a wrist camera may have split the take into several videos (stopped and restarted):
        # one demo per combination of wrist videos (one per gripper) that overlap in time
        gripper_wrists = list()
        for _, name in gripper_bodies:
            wrists = overlapping(videos, head, config['bodies'][name]['gripper_camera'])
            assert len(wrists) > 0, f"{head['name']}: no {name} video"
            gripper_wrists.append(wrists)
        combinations = [c for c in itertools.product(*gripper_wrists)
            if max(x['start_timestamp'] for x in c) < min(x['end_timestamp'] for x in c)]
        seen_episodes = set()
        for combination in combinations:
            # cameras of this demo: wrist cameras in robot order, then head
            demo_videos = list(combination)
            if len(combinations) > 1:
                print(f"{head['name']} with wrist videos {', '.join(x['name'] for x in demo_videos)}:")
            if not no_head_camera:
                demo_videos.append(head)
            fps = demo_videos[0]['fps']
            assert all(x['fps'] == fps for x in demo_videos), "cameras must record at the same fps"
            dt = 1 / fps

            # common frame grid aligned to the robot0 camera, as in 06
            start_timestamp = max(x['start_timestamp'] for x in demo_videos + [head])
            end_timestamp = min(x['end_timestamp'] for x in demo_videos + [head])
            align_start = demo_videos[0]['start_timestamp']
            start_timestamp += float(dt - ((start_timestamp - align_start) % dt))
            n_frames = int((end_timestamp - start_timestamp) / dt)
            video_start_frames = list()
            for x in demo_videos:
                video_start_frame = math.ceil((start_timestamp - x['start_timestamp']) / dt)
                video_n_frames = math.floor((x['end_timestamp'] - start_timestamp) / dt) - 1
                if video_start_frame < 0:
                    video_n_frames += video_start_frame
                    video_start_frame = 0
                video_start_frames.append(video_start_frame)
                n_frames = min(n_frames, video_n_frames)
            demo_timestamps = np.arange(n_frames) * float(dt) + start_timestamp
            total_time += n_frames * float(dt)

            # TCP poses and gripper widths per gripper
            tcp_poses = list()
            gripper_widths = list()
            is_valid = list()
            pose_inputs = list()
            width_interpolators = list()
            for (_, name), wrist in zip(gripper_bodies, demo_videos):
                body = track['bodies'][name]
                T_cube_tcp = np.array(tcp_calibration[name]['T_cube_tcp'])
                if reference == 'world' and name in track.get('grippers', dict()):
                    # fused wrist camera poses (03c_fuse_gripper_imu.py), on the wrist frames
                    fused = track['grippers'][name]
                    T_wrist_tcp = np.array(tcp_calibration[name]['T_wrist_tcp'])
                    valid = fused['is_valid'] & ~np.isnan(fused['T_world_wrist'][:, 0, 0])
                    T_ref_tcp = np.where(valid[:, None, None], fused['T_world_wrist'] @ T_wrist_tcp, np.nan)
                    pose_times = head['start_timestamp'] + fused['time']
                else:
                    if reference == 'world':
                        T_ref_cube, valid = body['T_world_body'], body['is_valid_world']
                    else:
                        T_ref_cube, valid = body['T_cam_body'], body['is_valid']
                    valid = valid & ~np.isnan(T_ref_cube[:, 0, 0])
                    T_ref_tcp = np.where(valid[:, None, None], T_ref_cube @ T_cube_tcp, np.nan)
                    pose_times = head_times
                pose, pose_valid = interpolate_valid_poses(pose_times, T_ref_tcp, valid, demo_timestamps, max_gap_sec)
                pose_inputs.append((pose_times, T_ref_tcp, valid))
                tcp_poses.append(pose)
                is_valid.append(pose_valid)

                gripper = hardware['grippers'][wrist['camera_serial']]
                gripper_range = load_gripper_range(session, wrist['camera_serial'], gripper)
                detections = pickle.load(wrist['dir'].joinpath('tag_detection.pkl').open('rb'))
                # same widths as the delivery: one-tag widths from the calibrated finger midpoint
                all_widths, _, _ = finger_widths(detections, gripper['left_finger_tag_id'],
                    gripper['right_finger_tag_id'], hardware['nominal_z'], gripper_range)
                measured = ~np.isnan(all_widths)
                width_times = [wrist['start_timestamp'] + d['time'] for d, m in zip(detections, measured) if m]
                widths = all_widths[measured].tolist()
                if 'finger_center_x' not in gripper_range:
                    print(f"{wrist['name']}: gripper_range.json has no finger_center_x (rerun gripper_range), "
                        f"one-tag widths assume the camera axis")
                print(f"{wrist['name']}: gripper width in {len(widths) / len(detections) * 100:.0f}% of frames"
                    + ('' if wrist['name'] in corrections or sync is None else '; no wrist sync correction, audio sync only'))
                width_times = np.array(width_times)
                width_times = width_times + correction(wrist, width_times)
                width_interpolators.append(get_interp1d(width_times, np.array(widths)))
                gripper_widths.append(width_interpolators[-1](demo_timestamps))

            is_step_valid = np.all(is_valid, axis=0)
            if ignore_poses:                  # episode videos only: every step counts
                is_step_valid = np.ones(n_frames, dtype=bool)
            print(f"{head['name']}: {n_frames} frames, valid poses "
                + ', '.join(f"{name} {v.mean() * 100:.0f}%" for (_, name), v in zip(gripper_bodies, is_valid))
                + f", all grippers {is_step_valid.mean() * 100:.0f}%")
            if not is_step_valid.any():
                continue

            if episodes is not None:
                # one episode per segmented episode, fully valid (gaps up to max_gap_sec bridged)
                assert head['name'] in episodes, f"{head['name']} not in episodes.json, rerun 00b_segment_episodes.py"
                segments = list()
                for k, e in enumerate(episodes[head['name']]['episodes']):
                    if 'task_start' in e:
                        # dock-to-dock episode: the task without the docking (03t_trim_docking.py)
                        if e['task_start'] is None:
                            seen_episodes.add(k)
                            print(f"  episode {k + 1} ({e['start']:.1f}-{e['end']:.1f}s): dropped, the cubes never left the dock")
                            continue
                        e = {**e, 'start': e['task_start'], 'end': e['task_end']}
                    e_start, e_end = head['start_timestamp'] + e['start'], head['start_timestamp'] + e['end']
                    if e_end < demo_timestamps[0] or e_start > demo_timestamps[-1]:
                        continue  # recorded by other wrist videos
                    seen_episodes.add(k)
                    in_episode = np.flatnonzero((demo_timestamps >= e_start) & (demo_timestamps <= e_end))
                    label = f"  episode {k + 1} ({e['start']:.1f}-{e['end']:.1f}s)"
                    if e_start < demo_timestamps[0] - max_trim_sec or e_end > demo_timestamps[-1] + max_trim_sec:
                        print(f"{label}: dropped, partly outside the recording of "
                            f"{', '.join(x['name'] for x in demo_videos)} (a wrist video stopped during it)")
                        continue
                    if len(in_episode) < min_episode_length:
                        print(f"{label}: dropped, {len(in_episode)} steps on the common frame grid")
                        continue
                    # trim invalid steps at the edges (bang motions), at most max_trim_sec per side
                    valid = is_step_valid[in_episode[0]:in_episode[-1] + 1]
                    step = float(dt)
                    max_trim = int(round(max_trim_sec / step))
                    first_valid = np.flatnonzero(valid)
                    if len(first_valid) == 0:
                        print(f"{label}: dropped, no valid poses")
                        continue
                    trim_start, trim_end = first_valid[0], len(valid) - 1 - first_valid[-1]
                    if trim_start > max_trim or trim_end > max_trim:
                        print(f"{label}: dropped, invalid for {trim_start * step:.2f}s at the start / "
                            f"{trim_end * step:.2f}s at the end (more than {max_trim_sec}s)")
                        continue
                    s = slice(in_episode[0] + trim_start, in_episode[-1] + 1 - trim_end)
                    if not is_step_valid[s].all():
                        gap_segments, gap_types = get_bool_segments(~is_step_valid[s])
                        longest = max((g.stop - g.start) * step for g, is_gap in zip(gap_segments, gap_types) if is_gap)
                        if not keep_gaps:
                            print(f"{label}: dropped, gaps over {max_gap_sec}s inside after trimming the edges "
                                f"(poses valid in {is_step_valid[s].mean() * 100:.0f}% of steps, longest gap {longest:.2f}s)")
                            continue
                        print(f"{label}: kept with pose not available in {(~is_step_valid[s]).mean() * 100:.1f}% "
                            f"of steps ({int(sum(gap_types))} gaps over {max_gap_sec}s, longest {longest:.2f}s)")
                    if s.stop - s.start < min_episode_length:
                        print(f"{label}: dropped, {s.stop - s.start} steps after trimming")
                        continue
                    if trim_start > 0 or trim_end > 0:
                        print(f"{label}: trimmed {trim_start * step:.2f}s at the start, {trim_end * step:.2f}s at the end")
                    segments.append(s)
                # start and end poses of each episode
                segment_poses = [(s, [x[s.start] for x in tcp_poses], [x[s.stop - 1] for x in tcp_poses])
                    for s in segments]
            else:
                first = np.flatnonzero(is_step_valid)[0]
                last = np.flatnonzero(is_step_valid)[-1]
                demo_start_poses = [x[first] for x in tcp_poses]
                demo_end_poses = [x[last] for x in tcp_poses]

                # drop short valid segments, one episode per remaining segment
                segments, segment_type = get_bool_segments(is_step_valid)
                for s, valid in zip(segments, segment_type):
                    if valid and (s.stop - s.start) < min_episode_length:
                        is_step_valid[s] = False
                segments, segment_type = get_bool_segments(is_step_valid)
                segment_poses = [(s, demo_start_poses, demo_end_poses)
                    for s, valid in zip(segments, segment_type) if valid]

            for s, demo_start_poses, demo_end_poses in segment_poses:
                # wrist clock correction of the segment: the steps are the robot0 camera's frames, on
                # the head clock at their corrected times; poses and widths there; each camera's frames
                # shifted by its correction relative to robot0 (whole frames)
                t_mid = 0.5 * (demo_timestamps[s.start] + demo_timestamps[s.stop - 1])
                shift = [float(correction(x, t_mid)) for x in demo_videos]
                times = demo_timestamps[s] + shift[0]
                poses, valids = zip(*[interpolate_valid_poses(*pose_inputs[g], times, max_gap_sec)
                    for g in range(len(gripper_bodies))])
                widths = [width_interpolators[g](times) for g in range(len(gripper_bodies))]
                valid = np.ones(len(times), dtype=bool) if ignore_poses else np.all(valids, axis=0)
                if not valid.all():
                    # the shift (a few ms) can move an edge step out of a tracked stretch: trim it
                    # (with keep_gaps, gaps inside stay: pose not available there)
                    ok = np.flatnonzero(valid)
                    a_, b_ = (ok[0], ok[-1] + 1) if len(ok) else (0, 0)
                    if len(ok) == 0 or (not keep_gaps and not valid[a_:b_].all()) \
                            or a_ > 3 or len(valid) - b_ > 3 or b_ - a_ < min_episode_length:
                        print(f"  segment {times[0] - head['start_timestamp']:.1f}s: dropped, poses invalid after the wrist clock correction")
                        continue
                    s = slice(s.start + a_, s.start + b_)
                    times, widths = times[a_:b_], [w[a_:b_] for w in widths]
                    poses = [p[a_:b_] for p in poses]
                    valids = [v[a_:b_] for v in valids]
                # steps whose pose is interpolated across lost tracking (not measured), per gripper
                gaps = [bridged_gaps(pose_inputs[g][0], pose_inputs[g][2], times) for g in range(len(gripper_bodies))]
                for g, (span, bridged) in enumerate(gaps):
                    bridged = bridged & valids[g]  # within max_gap_sec: interpolated; longer: not available
                    gaps[g] = (span, bridged)
                    bridge_stats[g]['steps'] += len(times)
                    bridge_stats[g]['bridged'] += int(bridged.sum())
                    bridge_stats[g]['missing'] += int((~valids[g]).sum())
                    if bridged.any():
                        bridge_stats[g]['longest'] = max(bridge_stats[g]['longest'], float(span[bridged].max()))
                frame_shifts = [int(round((shift[0] - c) / float(dt))) for c in shift]
                starts = [s.start + f + d for f, d in zip(video_start_frames, frame_shifts)]
                if any(a_ < 0 or a_ + (s.stop - s.start) > x['n_frames'] for a_, x in zip(starts, demo_videos)):
                    print(f"  segment {times[0] - head['start_timestamp']:.1f}s: dropped, outside a video after the correction")
                    continue
                if episodes is not None:
                    demo_start_poses = [p[0] for p in poses]
                    demo_end_poses = [p[-1] for p in poses]
                used_time += (s.stop - s.start) * float(dt)
                all_plans.append({
                    'episode_timestamps': times,
                    'grippers': [{
                        'tcp_pose': poses[g],
                        'gripper_width': widths[g],
                        # True where the pose is interpolated across a tracking gap (up to max_gap_sec)
                        'pose_bridged': gaps[g][1],
                        # False where the pose is not available (gap over max_gap_sec, --keep_gaps): tcp_pose NaN
                        'pose_valid': valids[g],
                        'demo_start_pose': demo_start_poses[g],
                        'demo_end_pose': demo_end_poses[g]
                    } for g in range(len(gripper_bodies))],
                    'cameras': [{
                        'video_path': str(x['dir'].joinpath('raw_video.mp4').relative_to(x['dir'].parent)),
                        'video_start_end': (a_, a_ + (s.stop - s.start)),
                        # session clock time of the video's frame 0 (wrist clock correction applied)
                        'frame0_time': float(x['start_timestamp'] + c),
                        'clock_correction_ms': c * 1e3,
                        # wrist IMU: video time = IMU time + offset (OpenICC camera-IMU calibration)
                        'imu_time_offset': imu_time_offset(x),
                        # 07 masks the gripper only in wrist camera images
                        'is_wrist_camera': x['camera_serial'] != config['head_camera']
                    } for x, a_, c in zip(demo_videos, starts, shift)]
                })
        if episodes is not None:
            for k, e in enumerate(episodes.get(head['name'], dict()).get('episodes', [])):
                if k not in seen_episodes:
                    print(f"  episode {k + 1} ({e['start']:.1f}-{e['end']:.1f}s): dropped, "
                        f"not recorded by all wrist cameras")

    print(f"{len(all_plans)} episodes, {used_time:.1f}s of {total_time:.1f}s used "
        f"({used_time / max(total_time, 1e-9) * 100:.0f}%)")
    for (_, name), x in zip(gripper_bodies, bridge_stats):
        print(f"  {name}: pose interpolated across lost tracking (not measured) in {x['bridged']} of {x['steps']} steps "
            f"({x['bridged'] / max(x['steps'], 1) * 100:.2f}%), longest gap {x['longest'] * 1e3:.0f} ms"
            + (f"; pose not available in {x['missing']} steps ({x['missing'] / max(x['steps'], 1) * 100:.2f}%)"
                if keep_gaps else ''))
    pickle.dump(all_plans, open(output, 'wb'))
    print(f"Saved to {output}")

# %%
if __name__ == "__main__":
    main()
