"""
Export episodes of dataset_plan.pkl (06_export_dataset_plan.py) as one MCAP
file per episode (JSON schemas umi.msg.* and foxglove.*), viewable in Foxglove:

  robot{i}/eef/state              umi.msg.EefState      TCP pos + rot_axis_angle per step; no message
                                                        where the pose is not available (tracking gap,
                                                        count in metadata robot{i}_pose_missing_steps)
  robot{i}/eef/pose               umi.msg.PoseQuat      the same TCP poses as position + quaternion xyzw
  robot{i}/gripper/state          umi.msg.GripperState  gripper width per step
  robot{i}/eef_demo_start/state   umi.msg.EefState      once, at the first step
  robot{i}/eef_demo_end/state     umi.msg.EefState      once, at the last step
  robot{i}/imu                    umi.msg.ImuSample     wrist GoPro IMU, raw GPMF axes, all samples
  /cameras/camera{j}/video        foxglove.CompressedVideo  H.265, the GoPro's own stream
  /head/pose                      umi.msg.PoseQuat      (with -c) head camera pose in the world per head frame
                                                        where tracked (position + quaternion xyzw, world from
                                                        camera; the instant of the middle image row)
  /head/imu                       umi.msg.ImuSample     (with -c and head_imu) head GoPro IMU, raw GPMF axes
  metadata 'recording'            episode, robot, camera and frame information
  metadata 'camera{j}'             (with -c) role, serial, model, firmware, lens, stabilization,
                                   field of view, resolution, frame rate, Kannala-Brandt intrinsics
  metadata 'sync'                  (with -c) method, per camera its clock offset, per wrist camera the
                                   motion sync uncertainty; without --exact_start also per camera the
                                   frame of its video at the episode's first step (camera{j}_sync_frame)
  metadata 'source'                per camera the original GoPro video (demos/ folder name) and the
                                   episode's frame range in it (provenance, not needed for sync)

Robots in robot_index order (UMI: 0 right, 1 left); cameras as in the plan:
wrist cameras in robot order, then the head camera.

Video is not re-encoded: the GoPro HEVC packets (no B-frames) are copied,
converted from MP4 length-prefixed NAL units to Annex B, with VPS/SPS/PPS
from the MP4 header inserted into every keyframe (after the access unit
delimiter), as Foxglove CompressedVideo requires. Each camera's video starts
at the keyframe before the episode (GoPro: one keyframe per second), so the
first video messages can precede the first pose message by up to 1 s.
--exact_start: every camera's video starts exactly at its frame of the episode's
first step instead; the frames from there to the next keyframe are re-encoded
(libx265, no B-frames, first one a keyframe), the rest is the original stream.
Metadata recording video_start: exact / keyframe.

python scripts_tag_tracking/07_export_mcap.py -i data_workspace/session -c umi_calib/tag_tracking.yaml --episodes 0,1,2
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import base64
import concurrent.futures
import json
import multiprocessing
import pathlib
import pickle
import click
import av
import numpy as np
from mcap.writer import Writer, CompressionType
from umi.tag_tracking.imu import load_imu
from umi.tag_tracking.config import load_tag_tracking_config
from umi.tag_tracking.camera_metadata import camera_record
from umi.common.exiftool_util import ExifTool
from umi.common.hevc_util import cut_from
from scipy.spatial.transform import Rotation

SYNC_METHOD = 'audio cross-correlation, wrist motion refinement'

PROFILE = 'umi'
LIBRARY = 'umi tag tracking'
TIME_SCHEMA = {"type": "object", "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}},
    "required": ["sec", "nsec"]}
VEC3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
VEC4 = {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}
SCHEMAS = {
    'umi.msg.EefState': {"type": "object", "properties": {"pos": VEC3, "rot_axis_angle": VEC3},
        "required": ["pos", "rot_axis_angle"]},
    'umi.msg.PoseQuat': {"type": "object", "properties": {"pos": VEC3, "quat_xyzw": VEC4},
        "required": ["pos", "quat_xyzw"]},
    'umi.msg.GripperState': {"type": "object", "properties": {"width": {"type": "number"}}, "required": ["width"]},
    'umi.msg.ImuSample': {"type": "object", "properties": {"timestamp": TIME_SCHEMA, "linear_acceleration": VEC3,
        "angular_velocity": VEC3}, "required": ["timestamp", "linear_acceleration", "angular_velocity"]},
    'foxglove.CompressedVideo': {"type": "object", "properties": {"timestamp": TIME_SCHEMA, "frame_id": {"type": "string"},
        "format": {"type": "string"}, "data": {"type": "string", "contentEncoding": "base64"}},
        "required": ["timestamp", "frame_id", "format", "data"]},
}
START_CODE = b'\x00\x00\x00\x01'
# UMI convention: robot0 right, robot1 left
ROBOT_SIDES = ('right', 'left')
HEVC_AUD = 35


def ns(t: float) -> int:
    return int(round(t * 1e9))


def stamp(t: float) -> dict:
    n = ns(t)
    return {'sec': n // 1_000_000_000, 'nsec': n % 1_000_000_000}


def hevc_parameter_sets(extradata: bytes) -> list:
    """
    VPS, SPS, PPS NAL units from an MP4 hvcC record.
    """
    assert extradata[0] == 1, "expected an hvcC record (MP4 HEVC)"
    out, i = list(), 23
    for _ in range(extradata[22]):
        count = int.from_bytes(extradata[i + 1:i + 3], 'big')
        i += 3
        for _ in range(count):
            length = int.from_bytes(extradata[i:i + 2], 'big')
            out.append(extradata[i + 2:i + 2 + length])
            i += 2 + length
    return out


def to_annexb(packet: bytes, length_size: int, parameter_sets: list, keyframe: bool) -> bytes:
    """
    MP4 length-prefixed NAL units -> Annex B; on keyframes the parameter sets
    follow the access unit delimiter (or lead if there is none).
    """
    nals, i = list(), 0
    while i + length_size <= len(packet):
        length = int.from_bytes(packet[i:i + length_size], 'big')
        nals.append(packet[i + length_size:i + length_size + length])
        i += length_size + length
    if keyframe:
        at = 1 if nals and ((nals[0][0] >> 1) & 0x3f) == HEVC_AUD else 0
        nals = nals[:at] + parameter_sets + nals[at:]
    return b''.join(START_CODE + n for n in nals)


def video_messages(video_path: str, start_frame: int, stop_frame: int):
    """
    (frame index, Annex B data) of frames [keyframe before start_frame, stop_frame).
    """
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == 'hevc', f"{video_path}: {stream.codec_context.name}, expected hevc"
        extradata = bytes(stream.codec_context.extradata)
        length_size = (extradata[21] & 3) + 1
        parameter_sets = hevc_parameter_sets(extradata)
        fps = float(stream.average_rate)
        frame_time = lambda pts: float(pts * stream.time_base) - float((stream.start_time or 0) * stream.time_base)
        container.seek(int(max(0.0, start_frame / fps - 0.001) / stream.time_base), stream=stream,
            backward=True, any_frame=False)
        started = False
        for packet in container.demux(stream):
            if packet.size == 0 or packet.pts is None:
                continue
            index = int(round(frame_time(packet.pts) * fps))
            if index >= stop_frame:
                break
            if not started:
                if not packet.is_keyframe:
                    continue
                assert index <= start_frame, f"{video_path}: first keyframe {index} after episode start {start_frame}"
                started = True
            yield index, to_annexb(bytes(packet), length_size, parameter_sets, packet.is_keyframe)


def video_messages_exact(video_path: str, start_frame: int, stop_frame: int, crf: int = 16):
    """
    (frame index, Annex B data) of frames [start_frame, stop_frame) exactly, no frames before
    start_frame: the frames from start_frame up to the next keyframe re-encoded (libx265, first frame
    a keyframe, no B-frames, full range like the GoPro stream), the rest the original stream (from
    its next keyframe, which carries the original VPS / SPS / PPS); umi.common.hevc_util.cut_from.
    Only the first GOP is re-encoded (at most one second at GoPro's one keyframe per second).
    """
    with av.open(video_path) as container:
        rate = container.streams.video[0].average_rate
    yield from cut_from(video_messages(video_path, start_frame, stop_frame), start_frame, rate, crf)


def frame0_time(camera: dict, times: np.ndarray, dt: float) -> float:
    """
    Session clock time of the camera video's frame 0: from the plan (06, wrist
    clock correction applied), else from the first step (plans without it).
    """
    if 'frame0_time' in camera:
        return camera['frame0_time']
    return float(times[0] - camera['video_start_end'][0] * dt)


def head_extras(plan_episode, demos_dir, config, tracks: dict) -> dict:
    """
    The head camera's tracked poses over the episode's head frames (tag_tracking.pkl, head IMU fused),
    the instant they refer to (frame time + readout / 2, the middle image row) and the head IMU time
    offset (head_imu of the config; None without). None if the plan has no head camera.
    tracks: cache of loaded tag_tracking.pkl per head video.
    """
    from umi.tag_tracking.imu import load_cam_imu_calibration, wrist_imu_time_offset
    cameras = plan_episode['cameras']
    if len(cameras) <= len(plan_episode['grippers']):
        return None
    video_dir = demos_dir.joinpath(cameras[-1]['video_path']).parent
    if video_dir.name not in tracks:
        track = pickle.load(video_dir.joinpath('tag_tracking.pkl').open('rb'))
        raw = str(video_dir.joinpath('raw_video.mp4'))
        with ExifTool() as et:
            readout = float(list(et.get_metadata(raw))[0]['QuickTime:SensorReadoutTime']) * 1e-3
        with av.open(raw) as container:
            fps = float(container.streams.video[0].average_rate)
        offset = None
        if 'head_imu' in config:
            offset = wrist_imu_time_offset(config['head_imu'],
                load_cam_imu_calibration(config['head_imu']['cam_imu_calibration']), raw, 1 / fps)
        tracks[video_dir.name] = {'T': track['head']['T_world_cam'], 'valid': track['head']['is_valid'],
            'readout': readout, 'imu_time_offset': offset}
    x = tracks[video_dir.name]
    a, b = cameras[-1]['video_start_end']
    T = x['T'][a:b]
    return {'T_world_cam': T, 'valid': x['valid'][a:b] & ~np.isnan(T[:, 0, 0]),
        'row_time_offset': x['readout'] / 2, 'imu_time_offset': x['imu_time_offset']}


def write_episode(path, episode_index, plan_episode, demos_dir, world_tag, video_fps, tcps_in_camera,
        camera_records=None, sync_record=None, time_offset=0.0, head=None, exact_start=False):
    """
    head (head_extras): head camera poses of the episode's head frames and its IMU time offset, or None.
    """
    times = np.asarray(plan_episode['episode_timestamps'], dtype=np.float64)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1 / video_fps
    cameras = plan_episode['cameras']
    grippers = plan_episode['grippers']
    with open(path, 'wb') as stream:
        writer = Writer(stream, compression=CompressionType.ZSTD)
        writer.start(profile=PROFILE, library=LIBRARY)
        schema_ids = {name: writer.register_schema(name=name, encoding='jsonschema', data=json.dumps(s).encode())
            for name, s in SCHEMAS.items()}

        def channel(topic, schema, metadata=None):
            return writer.register_channel(topic=topic, message_encoding='json', schema_id=schema_ids[schema],
                metadata=metadata or {})

        # message times: session clock + time_offset (13_export_delivery.py: UTC)
        stamp_ = lambda t: stamp(t + time_offset)

        def write(channel_id, t, message):
            writer.add_message(channel_id, log_time=ns(t + time_offset), publish_time=ns(t + time_offset),
                data=json.dumps(message).encode())

        def pose_quat(pos, quat):
            return {'pos': list(map(float, pos)), 'quat_xyzw': list(map(float, quat))}

        for g, gripper in enumerate(grippers):
            eef = channel(f'robot{g}/eef/state', 'umi.msg.EefState')
            eef_quat = channel(f'robot{g}/eef/pose', 'umi.msg.PoseQuat')
            width = channel(f'robot{g}/gripper/state', 'umi.msg.GripperState')
            demo_start = channel(f'robot{g}/eef_demo_start/state', 'umi.msg.EefState')
            demo_end = channel(f'robot{g}/eef_demo_end/state', 'umi.msg.EefState')
            write(demo_start, times[0], {'pos': list(map(float, gripper['demo_start_pose'][:3])),
                'rot_axis_angle': list(map(float, gripper['demo_start_pose'][3:]))})
            write(demo_end, times[-1], {'pos': list(map(float, gripper['demo_end_pose'][:3])),
                'rot_axis_angle': list(map(float, gripper['demo_end_pose'][3:]))})
            for t, p, w in zip(times, gripper['tcp_pose'], gripper['gripper_width']):
                # no message where the pose is not available (tracking gap, 06 --keep_gaps)
                if not np.isnan(p).any():
                    write(eef, t, {'pos': list(map(float, p[:3])), 'rot_axis_angle': list(map(float, p[3:]))})
                    write(eef_quat, t, pose_quat(p[:3], Rotation.from_rotvec(p[3:]).as_quat()))
                write(width, t, {'width': float(np.asarray(w).reshape(-1)[0])})
            # wrist camera g is camera g: its IMU, on the session clock via the plan's frame alignment
            camera = cameras[g]
            video_dir = demos_dir.joinpath(camera['video_path']).parent
            imu = load_imu(str(video_dir.joinpath('imu_data.json')))
            # wrist IMU on the session clock: video frame 0 (wrist clock correction applied by 06)
            # + IMU time + camera-IMU offset (video time = IMU time + offset)
            imu_channel = channel(f'robot{g}/imu', 'umi.msg.ImuSample')
            imu_times = frame0_time(camera, times, dt) + imu['time'] + (camera.get('imu_time_offset') or 0.0)
            inside = (imu_times >= times[0]) & (imu_times <= times[-1] + dt)
            for t, a, w in zip(imu_times[inside], imu['accel'][inside], imu['gyro'][inside]):
                write(imu_channel, t, {'timestamp': stamp_(t), 'linear_acceleration': list(map(float, a)),
                    'angular_velocity': list(map(float, w))})

        if head is not None:
            # head camera (last camera of the plan): its own tracked pose per head frame, and its IMU
            camera = cameras[-1]
            t0 = frame0_time(camera, times, dt)
            first = camera['video_start_end'][0]
            pose_channel = channel('/head/pose', 'umi.msg.PoseQuat')
            for i, (T, ok) in enumerate(zip(head['T_world_cam'], head['valid'])):
                if ok:
                    write(pose_channel, t0 + (first + i) * dt + head['row_time_offset'],
                        pose_quat(T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat()))
            if head.get('imu_time_offset') is not None:
                imu = load_imu(str(demos_dir.joinpath(camera['video_path']).parent.joinpath('imu_data.json')))
                imu_channel = channel('/head/imu', 'umi.msg.ImuSample')
                imu_times = t0 + imu['time'] + head['imu_time_offset']
                inside = (imu_times >= times[0]) & (imu_times <= times[-1] + dt)
                for t, a, w in zip(imu_times[inside], imu['accel'][inside], imu['gyro'][inside]):
                    write(imu_channel, t, {'timestamp': stamp_(t), 'linear_acceleration': list(map(float, a)),
                        'angular_velocity': list(map(float, w))})

        image_size = None
        for c, camera in enumerate(cameras):
            video_path = demos_dir.joinpath(camera['video_path'])
            start, stop = camera['video_start_end']
            if image_size is None:
                with av.open(str(video_path)) as container:
                    image_size = (container.streams.video[0].width, container.streams.video[0].height)
            video = channel(f'/cameras/camera{c}/video', 'foxglove.CompressedVideo', {'stream_type': 'color'})
            messages = video_messages_exact if exact_start else video_messages
            for index, data in messages(str(video_path), start, stop):
                t = frame0_time(camera, times, dt) + index * dt
                write(video, t, {'timestamp': stamp_(t), 'frame_id': f'camera{c}', 'format': 'h265',
                    'data': base64.b64encode(data).decode()})

        writer.add_metadata('recording', {
            'episode_index': str(episode_index),
            'episode_length': str(len(times)),
            'n_grippers': str(len(grippers)),
            'n_cameras': str(len(cameras)),
            'image_width': str(image_size[0]),
            'image_height': str(image_size[1]),
            'video_fps': str(video_fps),
            'video_codec': 'h265',
            'robot_type': 'umi_bimanual' if len(grippers) == 2 else 'umi',
            **{f'robot{g}': side for g, side in zip(range(len(grippers)), ROBOT_SIDES)},
            **{f'camera{c}': f'robot{c}_wrist' if c < len(grippers) else 'head' for c in range(len(cameras))},
            'world_frame': f'tag{world_tag}',
            # exact: every camera's video starts at its frame of the episode's first step (start re-encoded);
            # keyframe: at the keyframe before it (up to 1 s earlier, original stream only)
            'video_start': 'exact' if exact_start else 'keyframe',
            'eef_frame': 'tcp',
            # TCP in each gripper's wrist camera frame, one key if all grippers share it
            **({'tcp_in_camera_m': ','.join(f'{x:g}' for x in tcps_in_camera[0])} if len(set(tcps_in_camera)) == 1
                else {f'robot{g}_tcp_in_camera_m': ','.join(f'{x:g}' for x in t) for g, t in enumerate(tcps_in_camera)}),
            'rotation': 'axis_angle',
            'units': 'm,rad',
            'gripper_width_unit': 'm',
            'gripper_width_zero': 'fully closed',
            # steps without a pose message: pose not available (tracking gap)
            **{f'robot{g}_pose_missing_steps': str(int(np.isnan(np.asarray(gripper['tcp_pose'])).any(axis=1).sum()))
                for g, gripper in enumerate(grippers)},
        })
        for c, record in enumerate(camera_records or []):
            writer.add_metadata(f'camera{c}', record)
        if sync_record is not None:
            writer.add_metadata('sync', {
                **sync_record,
                # keyframe start: where the episode starts in each camera's video (the MCAP video starts earlier);
                # exact start: not written, every MCAP video starts at the episode start (its frame 0)
                **({} if exact_start else {f'camera{c}_sync_frame': str(int(camera['video_start_end'][0]))
                    for c, camera in enumerate(cameras)}),
                **{f'camera{c}_offset_ms': f"{camera['clock_correction_ms']:.3f}"
                    for c, camera in enumerate(cameras) if camera.get('clock_correction_ms') is not None},
            })
        # provenance, not needed for sync: the original GoPro video of each camera and the frame range of
        # the episode in it (frame 0 of the MCAP video = source_first_frame with exact start)
        writer.add_metadata('source', {
            **{f'camera{c}_source_video': pathlib.Path(camera['video_path']).parent.name
                for c, camera in enumerate(cameras)},
            **{f'camera{c}_source_first_frame': str(int(camera['video_start_end'][0])) for c, camera in enumerate(cameras)},
            **{f'camera{c}_source_end_frame': str(int(camera['video_start_end'][1])) for c, camera in enumerate(cameras)},
        })
        writer.finish()
    return path


def verify(path) -> str:
    """
    Decode the first video message of every camera (must be a keyframe) and
    count messages per topic.
    """
    from mcap.reader import make_reader
    counts, decoded = dict(), dict()
    with open(path, 'rb') as stream:
        for schema, channel, message in make_reader(stream).iter_messages():
            counts[channel.topic] = counts.get(channel.topic, 0) + 1
            if schema.name == 'foxglove.CompressedVideo' and channel.topic not in decoded:
                codec = av.CodecContext.create('hevc', 'r')
                frames = codec.decode(av.Packet(base64.b64decode(json.loads(message.data)['data'])))
                frames += codec.decode(None)
                decoded[channel.topic] = f"{frames[0].width}x{frames[0].height}" if frames else 'NOT DECODABLE'
    return ', '.join(f"{t} {n}" + (f" ({decoded[t]})" if t in decoded else '') for t, n in sorted(counts.items()))


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Session directory with dataset_plan.pkl')
@click.option('-o', '--output', default=None, help='Output directory, default <session>/dataset_mcap')
@click.option('-e', '--episodes', default=None, help='Comma separated episode indices of the plan, default all')
@click.option('-c', '--config', default=None,
    help='Tag tracking config: adds the camera (model, lens, intrinsics) and sync metadata records')
@click.option('--exact_start', is_flag=True, default=False,
    help='Videos start exactly at the episode start frame (first second re-encoded), no keyframe lead-in')
@click.option('-j', '--num_workers', type=int, default=None, help='Episodes written in parallel, default all cores')
def main(input, output, episodes, config, exact_start, num_workers):
    session = pathlib.Path(os.path.expanduser(input)).absolute()
    output = session.joinpath('dataset_mcap') if output is None else pathlib.Path(os.path.expanduser(output)).absolute()
    output.mkdir(parents=True, exist_ok=True)
    plan = pickle.load(session.joinpath('dataset_plan.pkl').open('rb'))
    indices = list(range(len(plan))) if episodes is None else [int(x) for x in episodes.split(',')]
    assert all(0 <= k < len(plan) for k in indices), f"plan has {len(plan)} episodes"
    demos_dir = session.joinpath('demos')
    with av.open(str(demos_dir.joinpath(plan[0]['cameras'][0]['video_path']))) as container:
        video_fps = float(container.streams.video[0].average_rate)
    # world frame: anchor tag of the static environment body (calibration)
    bodies = json.load(session.joinpath('tag_tracking', 'bodies.json').open('r'))['bodies']
    # TCP in each gripper's wrist camera frame (T_wrist_tcp of the setup calibration), robot order
    tcps_in_camera = [tuple(float(v) for v in np.round(np.array(x['T_wrist_tcp'])[:3, 3], 6)) for x in sorted(
        json.load(session.joinpath('tag_tracking', 'gripper_tcp.json').open('r')).values(), key=lambda x: x['robot_index'])]
    print(f"TCP in the wrist camera frame per robot: {tcps_in_camera} m")
    world_tag = [b['anchor_tag'] for b in bodies.values() if b['static']][0]
    print(f"{len(plan)} episodes in the plan, writing {len(indices)} to {output}")
    # camera and sync metadata (camera order of the plan: wrist cameras in robot order, then the head)
    camera_records, sync_records = {k: None for k in indices}, {k: None for k in indices}
    heads = {k: None for k in indices}
    if config is not None:
        config = load_tag_tracking_config(config)
        sync_path = session.joinpath('tag_tracking', 'wrist_sync.json')
        sync = json.load(sync_path.open('r')) if sync_path.is_file() else {'videos': {}}
        with ExifTool() as et:
            exif = {str(p): list(et.get_metadata(str(demos_dir.joinpath(p))))[0]
                for p in {c['video_path'] for k in indices for c in plan[k]['cameras']}}
        for k in indices:
            cameras = plan[k]['cameras']
            n_grippers = len(plan[k]['grippers'])
            records = list()
            for c, camera in enumerate(cameras):
                info = exif[camera['video_path']]
                serial = info['QuickTime:CameraSerialNumber']
                role = f"{ROBOT_SIDES[c]}_wrist" if c < n_grippers else 'head'
                records.append(camera_record(role, serial, info, video_fps, config['cameras'][serial]))
            camera_records[k] = records
            head = pathlib.Path(cameras[-1]['video_path']).parent.name
            # per wrist camera: the motion sync fit's uncertainty, where it was fitted
            fits = sync['videos'].get(head, dict())
            sync_records[k] = {'method': SYNC_METHOD if any(c.get('clock_correction_ms') for c in cameras[:n_grippers])
                else 'audio cross-correlation'}
            for c, camera in enumerate(cameras[:n_grippers]):
                line = (fits.get(pathlib.Path(camera['video_path']).parent.name) or dict()).get('line')
                if line is not None:
                    sync_records[k][f'camera{c}_sync_sigma_ms'] = f"{line['residual_ms']:.2f}"
        tracks = dict()
        for k in indices:
            heads[k] = head_extras(plan[k], demos_dir, config, tracks)
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(num_workers, len(indices)),
            mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = {executor.submit(write_episode, str(output.joinpath(f'episode_{k:06d}.mcap')), k, plan[k],
            demos_dir, world_tag, video_fps, tcps_in_camera, camera_records[k], sync_records[k], 0.0, heads[k],
            exact_start): k
            for k in indices}
        for future in concurrent.futures.as_completed(futures):
            path = future.result()
            print(f"{pathlib.Path(path).name}: {os.path.getsize(path) / 1e6:.0f} MB, {verify(path)}")

# %%
if __name__ == "__main__":
    main()
