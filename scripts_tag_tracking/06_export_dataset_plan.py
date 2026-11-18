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
from 00b_segment_episodes.py (tag_tracking/episodes.json) instead:
- poses are interpolated across tracking gaps up to episodes.max_gap_sec;
- invalid steps at an episode's start / end (the bang motions) are trimmed,
  at most --max_trim_sec per side; the rest must be fully valid, else the
  episode is dropped (not split); demo_start_pose / demo_end_pose are per episode.
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
from umi.common.exiftool_util import ExifTool
from umi.common.cv_util import get_gripper_width
from umi.common.pose_util import mat_to_pose
from umi.common.interpolation_util import (
    PoseInterpolator,
    get_interp1d,
    get_gripper_calibration_interpolator
)
from umi.tag_tracking.config import load_tag_tracking_config
from umi.tag_tracking.session import get_session_videos, overlapping
from umi.tag_tracking.wrist_sync import wrist_corrections, correction_seconds
from umi.tag_tracking.imu import load_cam_imu_calibration


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
    if is_valid.sum() >= 2:
        interp = PoseInterpolator(times[is_valid], mat_to_pose(poses[is_valid]))
        out[query_valid] = interp(query_times[query_valid])
    return out, query_valid

def load_gripper_width_interpolator(session, serial):
    with ExifTool() as et:
        for path in session.glob('demos/gripper_calibration*/gripper_range.json'):
            meta = list(et.get_metadata(str(path.parent.joinpath('raw_video.mp4'))))[0]
            if meta['QuickTime:CameraSerialNumber'] == serial:
                data = json.load(path.open('r'))
                return get_gripper_calibration_interpolator(
                    aruco_measured_width=[data['min_width'], data['max_width']],
                    aruco_actual_width=[data['min_width'], data['max_width']])
    raise RuntimeError(f"no gripper_range.json for camera {serial}, run 05_run_calibrations.py")


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
def main(input, config, output, wrist_sync, reference, min_episode_length, no_head_camera, max_trim_sec):
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
    for head in [x for x in videos if x['camera_serial'] == config['head_camera']]:
        track = pickle.load(head['dir'].joinpath('tag_tracking.pkl').open('rb'))
        head_times = head['start_timestamp'] + track['time']
        corrections = wrist_corrections(sync, head['name'])

        def imu_time_offset(video):
            serial = video['camera_serial']
            if serial == config['head_camera'] or serial not in config.get('gripper_imu', dict()):
                return None
            return float(load_cam_imu_calibration(config['gripper_imu'][serial]['cam_imu_calibration'])['time_offset'])

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
                width_calibration = load_gripper_width_interpolator(session, wrist['camera_serial'])
                detections = pickle.load(wrist['dir'].joinpath('tag_detection.pkl').open('rb'))
                width_times = list()
                widths = list()
                for d in detections:
                    width = get_gripper_width(d['tag_dict'], gripper['left_finger_tag_id'],
                        gripper['right_finger_tag_id'], nominal_z=hardware['nominal_z'])
                    if width is not None:
                        width_times.append(wrist['start_timestamp'] + d['time'])
                        widths.append(width_calibration(width))
                print(f"{wrist['name']}: gripper width in {len(widths) / len(detections) * 100:.0f}% of frames"
                    + ('' if wrist['name'] in corrections or sync is None else '; no wrist sync correction, audio sync only'))
                width_times = np.array(width_times)
                width_times = width_times + correction(wrist, width_times)
                width_interpolators.append(get_interp1d(width_times, np.array(widths)))
                gripper_widths.append(width_interpolators[-1](demo_timestamps))

            is_step_valid = np.all(is_valid, axis=0)
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
                        print(f"{label}: dropped, gaps over {max_gap_sec}s inside after trimming the edges "
                            f"(poses valid in {is_step_valid[s].mean() * 100:.0f}% of steps)")
                        continue
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
                valid = np.all(valids, axis=0)
                if not valid.all():
                    # the shift (a few ms) can move an edge step out of a tracked stretch: trim it
                    ok = np.flatnonzero(valid)
                    a_, b_ = (ok[0], ok[-1] + 1) if len(ok) else (0, 0)
                    if len(ok) == 0 or not valid[a_:b_].all() or a_ > 3 or len(valid) - b_ > 3 or b_ - a_ < min_episode_length:
                        print(f"  segment {times[0] - head['start_timestamp']:.1f}s: dropped, poses invalid after the wrist clock correction")
                        continue
                    s = slice(s.start + a_, s.start + b_)
                    times, widths = times[a_:b_], [w[a_:b_] for w in widths]
                    poses = [p[a_:b_] for p in poses]
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
    pickle.dump(all_plans, open(output, 'wb'))
    print(f"Saved to {output}")

# %%
if __name__ == "__main__":
    main()
