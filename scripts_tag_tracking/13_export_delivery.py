"""
Odyssey delivery (Robotics Data Capture Spec v0.6, container 'mcap') of the
episodes of a session's dataset_plan.pkl (06_export_dataset_plan.py), one
directory per episode:

  <output>/<episode_id>/episode.mcap     07_export_mcap.py content, message times in UTC
  <output>/<episode_id>/episode.json     identity, provenance, clocks, task, outcome (2.4)
  <output>/<episode_id>/sensors.json     cameras, wrist IMUs, TCP poses, grippers, tag tracker (2.5)
  <output>/<episode_id>/frames.parquet   one row per head camera frame (primary camera) (2.6)
  <output>/<episode_id>/annotations.json event timeline (2.7), where the session is annotated
  <output>/manifest.parquet              one row per episode of every session in <output>, written last (2.3)

Only measured or declared values: frames.parquet rows stay where tracking is
lost, with null pose and pose_valid false (umi.tag_tracking.delivery, nothing
filled in); what only the recording team knows (vendor, dataset, task text,
consent, license, ...) comes from the delivery yaml (umi.tag_tracking.delivery_config)
and is null where not declared.

annotations.json: from the session's annotations (08_annotate_episodes.py / 09, per
dock-to-dock episode, tag_tracking/annotations), the one whose task interval overlaps the
episode's, with its times moved to the episode's first frame and clipped to the episode
(umi.tag_tracking.annotation.delivery_annotation). It also gives the task text and family
where the delivery yaml declares none, the outcome / success, the failure spans
(episode.json events) and annotation_source. Episodes without an annotation: no
annotations.json, outcome 'unknown', success null. validation_status 'not_run' (no
quality checks).

Clock: utc_unix. Session times (audio + motion synchronized, 06a / 03s) are
mapped to UTC by the head camera's own start time (GoPro creation_time +
timecode, umi.common.timecode_util.mp4_get_start_utc); clock_sync_accuracy_ns
is the largest head-to-wrist sync accuracy of the motion sync (wrist_sync.json).

Episode ids: recording_id = head camera serial + UTC start of the head video,
episode_id = recording_id + head video frame of the episode start: stable when
re-exported. The manifest keeps the rows of episodes exported by earlier runs
(other sessions) and replaces those exported again; written after every file.

Needs pyarrow (pip install pyarrow).

python scripts_tag_tracking/13_export_delivery.py -i data_workspace/task1 -c umi_calib/tag_tracking.yaml -d delivery.yaml -o delivery_dir
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import concurrent.futures
import datetime
import hashlib
import importlib.util
import json
import multiprocessing
import pathlib
import pickle
import click
import av
import numpy as np
from umi.common.exiftool_util import ExifTool
from umi.common.timecode_util import mp4_get_start_utc
from umi.tag_tracking.camera_metadata import camera_record
from umi.tag_tracking.config import load_tag_tracking_config
from umi.tag_tracking.delivery import episode_frames, gripper_streams
from umi.tag_tracking.delivery_config import load_delivery_config, session_declarations, TASK_FAMILIES
from umi.tag_tracking.annotation import find_annotation, delivery_annotation
from umi.tag_tracking.imu import load_imu, load_cam_imu_calibration, wrist_imu_time_offset
from umi.tag_tracking.session import get_session_videos
from umi.tag_tracking.wrist_sync import wrist_corrections, head_readout

CONTAINER = 'mcap'
TIER = 'T2'
SOURCE_TYPE = 'handheld_gripper'
EMBODIMENT = 'handheld_gripper'
SIDES = ('right', 'left')  # robot0, robot1 (UMI)


def load_mcap_module():
    spec = importlib.util.spec_from_file_location('export_mcap', os.path.join(ROOT_DIR, 'scripts_tag_tracking', '07_export_mcap.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_mcap(path, episode_index, plan_episode, demos_dir, world_tag, video_fps, tcps_in_camera,
        camera_records, sync_record, time_offset, head):
    module = load_mcap_module()
    module.write_episode(path, episode_index, plan_episode, demos_dir, world_tag, video_fps, tcps_in_camera,
        camera_records, sync_record, time_offset, head)
    return path


def first_video_frame(video_path: str, start_frame: int) -> int:
    """
    Frame index of the first video message 07_export_mcap.py writes: the keyframe before start_frame.
    """
    module = load_mcap_module()
    for index, _ in module.video_messages(video_path, start_frame, start_frame + 1):
        return index
    raise RuntimeError(f"{video_path}: no keyframe before frame {start_frame}")


def utc_iso(t: float) -> str:
    return datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).isoformat().replace('+00:00', 'Z')


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def mat44(T) -> list:
    return [float(x) for x in np.asarray(T, dtype=float).reshape(-1)]


def quat_to_mat(q: dict) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat([q['x'], q['y'], q['z'], q['w']]).as_matrix()


def video_encoding(path: str) -> dict:
    with av.open(path) as container:
        s = container.streams.video[0]
        return {'codec': s.codec_context.name, 'bit_rate': int(s.bit_rate or 0), 'pix_fmt': s.codec_context.pix_fmt,
            # PyAV 10 does not expose the color range; the yuvj pixel formats are full range by definition
            'color_range': 'full' if s.codec_context.pix_fmt.startswith('yuvj') else 'unstated',
            'width': s.codec_context.width, 'height': s.codec_context.height}


def channel_counts(path: pathlib.Path) -> dict:
    from mcap.reader import make_reader
    with path.open('rb') as stream:
        summary = make_reader(stream).get_summary()
        topics = {c.id: c.topic for c in summary.channels.values()}
        return {topics[k]: int(n) for k, n in summary.statistics.channel_message_counts.items()}


def frames_table(episode_id: str, rows: dict, utc_shift: float, head_start: float, first_frame: int,
        created: datetime.datetime):
    """
    frames.parquet (2.6): one row per head frame, null where not measured.
    """
    import pyarrow as pa
    n = len(rows['time'])
    nullable_list = lambda a: [None if np.isnan(x).any() else [float(v) for v in x] for x in a]
    nullable = lambda a: [None if np.isnan(x) else float(x) for x in a]
    columns = {
        'episode_id': pa.array([episode_id] * n, pa.string()),
        'frame_id': pa.array(np.arange(n), pa.int64()),
        # the instant the row's poses refer to: head frame time + readout / 2 (middle image row)
        'timestamp_ns': pa.array(np.round((head_start + rows['time'] + utc_shift) * 1e9).astype(np.int64), pa.int64()),
        'primary_camera_id': pa.array(['head'] * n, pa.string()),
        # index of the frame among the head video messages of episode.mcap (they start at a keyframe)
        'camera_frame_index': pa.array(rows['head_frame_index'] - first_frame, pa.int64()),
        'media_uri': pa.array(['episode.mcap'] * n, pa.string()),
        # the frame is in the video; image quality checks not run
        'image_valid': pa.array(np.ones(n, dtype=bool), pa.bool_()),
        'ee_pose_left': pa.array(nullable_list(rows['ee_pose_left']), pa.list_(pa.float64())),
        'ee_pose_right': pa.array(nullable_list(rows['ee_pose_right']), pa.list_(pa.float64())),
        'pose_valid': pa.array(rows['pose_valid'], pa.bool_()),
        'gripper_left': pa.array(nullable(rows['gripper_left']), pa.float64()),
        'gripper_right': pa.array(nullable(rows['gripper_right']), pa.float64()),
        # handheld rig: the state is the TCP poses and gripper openings (no joints)
        'proprio_valid': pa.array(rows['pose_valid'] & ~np.isnan(rows['gripper_left']) & ~np.isnan(rows['gripper_right']),
            pa.bool_()),
        # moving head camera (R11): its own pose per frame
        'camera_pose_head': pa.array(nullable_list(rows['camera_pose_head']), pa.list_(pa.float64())),
        'camera_pose_head_valid': pa.array(rows['camera_pose_head_valid'], pa.bool_()),
        'created_at_utc': pa.array([created.date()] * n, pa.date32()),
        'created_ts_utc': pa.array([created] * n, pa.timestamp('us', tz='UTC')),
    }
    return pa.table(columns)


MANIFEST_COLUMNS = ['schema_version', 'container_format', 'vendor_name', 'dataset_name', 'delivery_id', 'episode_id',
    'recording_id', 'tier', 'source_type', 'embodiment_class', 'robot_model', 'capture_start_utc', 'capture_end_utc',
    'duration_ns', 'primary_camera_id', 'primary_frame_count', 'cameras', 'streams', 'files', 'total_byte_size',
    'has_proprio', 'has_actions', 'has_ee_pose', 'has_gripper', 'has_calibration', 'has_kinematics', 'task_id',
    'task_text', 'task_family', 'success', 'outcome', 'has_failure_spans', 'has_event_timeline', 'collection_mode',
    'n_tasks', 'environment_id', 'operator_id', 'station_id', 'pose_source', 'gripper_width_source',
    'nominal_frame_rate_hz', 'object_tags', 'scene_tags', 'annotation_source', 'anonymization_state',
    'consent_reference', 'license_reference', 'validation_status', 'created_at_utc', 'created_ts_utc']


# manifest columns the spec marks optional: left out when no episode has a value
OPTIONAL_COLUMNS = ['environment_id', 'operator_id', 'station_id', 'object_tags', 'scene_tags']


def write_manifest(output: pathlib.Path, rows: list):
    """
    manifest.parquet (2.3), one row per episode; rows of earlier runs kept unless re-exported.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = output.joinpath('manifest.parquet')
    previous = pq.read_table(path).to_pylist() if path.is_file() else list()
    ids = {r['episode_id'] for r in rows}
    kept = [r for r in previous if r['episode_id'] not in ids and output.joinpath(r['episode_id']).is_dir()]
    # new rows carry ISO strings, rows read back from the manifest datetimes
    for r in kept + rows:
        for k in ('capture_start_utc', 'capture_end_utc', 'created_ts_utc'):
            if isinstance(r[k], str):
                r[k] = datetime.datetime.fromisoformat(r[k].replace('Z', '+00:00'))
    all_rows = sorted(kept + rows, key=lambda r: (r['capture_start_utc'], r['episode_id']))
    camera = pa.struct([('id', pa.string()), ('role', pa.string()), ('width', pa.int64()), ('height', pa.int64()),
        ('frame_count', pa.int64())])
    stream = pa.struct([('id', pa.string()), ('type', pa.string()), ('sample_count', pa.int64()),
        ('native_rate_hz', pa.float64())])
    file = pa.struct([('path', pa.string()), ('byte_size', pa.int64()), ('sha256', pa.string())])
    ts = pa.timestamp('us', tz='UTC')
    types = {'cameras': pa.list_(camera), 'streams': pa.list_(stream), 'files': pa.list_(file),
        'capture_start_utc': ts, 'capture_end_utc': ts, 'created_ts_utc': ts, 'created_at_utc': pa.date32(),
        'duration_ns': pa.int64(), 'primary_frame_count': pa.int64(), 'total_byte_size': pa.int64(), 'n_tasks': pa.int64(),
        'nominal_frame_rate_hz': pa.float64(), 'object_tags': pa.list_(pa.string()), 'scene_tags': pa.list_(pa.string()),
        **{k: pa.bool_() for k in ('has_proprio', 'has_actions', 'has_ee_pose', 'has_gripper', 'has_calibration',
            'has_kinematics', 'success', 'has_failure_spans', 'has_event_timeline')}}
    columns = [k for k in MANIFEST_COLUMNS if k not in OPTIONAL_COLUMNS
        or any(r.get(k) not in (None, []) for r in all_rows)]
    table = pa.table({k: pa.array([r.get(k) for r in all_rows], types.get(k, pa.string())) for k in columns})
    tmp = path.with_suffix('.parquet.tmp')
    pq.write_table(table, tmp)
    tmp.replace(path)
    return len(all_rows)


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Session directory with dataset_plan.pkl')
@click.option('-c', '--config', required=True, help='Tag tracking config yaml')
@click.option('-d', '--delivery', required=True, help='Delivery yaml (example/tag_tracking/delivery.yaml)')
@click.option('-o', '--output', required=True, help='Delivery directory (episode directories + manifest.parquet)')
@click.option('-e', '--episodes', default=None, help='Comma separated episode indices of the plan, default all')
@click.option('-j', '--num_workers', type=int, default=None, help='MCAP files written in parallel, default all cores')
@click.option('--no_annotations', is_flag=True, default=False,
    help='Leave the annotations out: no annotations.json, task and outcome only as declared in the delivery yaml')
def main(input, config, delivery, output, episodes, num_workers, no_annotations):
    import pyarrow.parquet as pq
    session = pathlib.Path(os.path.expanduser(input)).absolute()
    output = pathlib.Path(os.path.expanduser(output)).absolute()
    output.mkdir(parents=True, exist_ok=True)
    config = load_tag_tracking_config(config)
    declared = load_delivery_config(delivery)
    d = declared['delivery']
    s = session_declarations(declared, session)
    plan = pickle.load(session.joinpath('dataset_plan.pkl').open('rb'))
    indices = list(range(len(plan))) if episodes is None else [int(x) for x in episodes.split(',')]
    assert all(0 <= k < len(plan) for k in indices), f"plan has {len(plan)} episodes"
    assert all(len(plan[k]['cameras']) == len(plan[k]['grippers']) + 1 for k in indices), \
        "the plan has no head camera (06 --no_head_camera); the head camera is the primary camera"
    demos_dir = session.joinpath('demos')
    created = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    mcap = load_mcap_module()

    tcp = json.load(session.joinpath('tag_tracking', 'gripper_tcp.json').open('r'))
    bodies = json.load(session.joinpath('tag_tracking', 'bodies.json').open('r'))
    world_tag = [b['anchor_tag'] for b in bodies['bodies'].values() if b['static']][0]
    tcps = sorted(tcp.values(), key=lambda x: x['robot_index'])
    tcps_in_camera = [tuple(float(v) for v in np.round(np.array(x['T_wrist_tcp'])[:3, 3], 6)) for x in tcps]
    sync_path = session.joinpath('tag_tracking', 'wrist_sync.json')
    sync = json.load(sync_path.open('r')) if sync_path.is_file() else None
    offsets = json.load(session.joinpath('audio_sync_offsets.json').open('r'))['videos']
    videos = get_session_videos(session)
    by_name = {v['name']: v for v in videos}
    with av.open(str(demos_dir.joinpath(plan[indices[0]]['cameras'][0]['video_path']))) as container:
        video_fps = float(container.streams.video[0].average_rate)
    with ExifTool() as et:
        exif = {p: list(et.get_metadata(str(demos_dir.joinpath(p))))[0]
            for p in {c['video_path'] for k in indices for c in plan[k]['cameras']}}

    heads = dict()  # per head video: tracking, gripper streams, UTC shift
    jobs, episode_info = list(), dict()
    for k in indices:
        e = plan[k]
        n_grippers = len(e['grippers'])
        head_name = pathlib.Path(e['cameras'][-1]['video_path']).parent.name
        head = by_name[head_name]
        if head_name not in heads:
            track = pickle.load(head['dir'].joinpath('tag_tracking.pkl').open('rb'))
            corrections = wrist_corrections(sync, head_name)
            raw = str(head['dir'].joinpath('raw_video.mp4'))
            # session clock -> UTC: the head camera's own start (creation_time + timecode)
            utc_shift = mp4_get_start_utc(raw).timestamp() - head['start_timestamp']
            fits = (sync or {'videos': {}})['videos'].get(head_name, dict())
            accuracy = [x['accuracy_ms'] for x in fits.values() if x.get('accuracy_ms') is not None]
            heads[head_name] = {'track': track, 'utc_shift': utc_shift, 'readout': head_readout(head),
                'grippers': gripper_streams(session, config, videos, head, track, tcp, corrections),
                'sync_accuracy_ns': int(round(max(accuracy) * 1e6)) if accuracy else None,
                'recording_id': f"{head['camera_serial']}_{datetime.datetime.fromtimestamp(head['start_timestamp'] + utc_shift, tz=datetime.timezone.utc):%Y%m%dT%H%M%S}Z",
                'fits': fits,
                # head IMU on the head video clock: video time = IMU time + offset (head_imu of the config)
                'imu_time_offset': None if 'head_imu' not in config else wrist_imu_time_offset(config['head_imu'],
                    load_cam_imu_calibration(config['head_imu']['cam_imu_calibration']), raw, float(1 / head['fps']))}
        h = heads[head_name]
        a, b = e['cameras'][-1]['video_start_end']
        episode_id = f"{h['recording_id']}_f{a:06d}"
        directory = output.joinpath(episode_id)
        directory.mkdir(parents=True, exist_ok=True)
        # camera and sync metadata records of the MCAP, as 07_export_mcap.py -c
        records = list()
        for c, camera in enumerate(e['cameras']):
            info = exif[camera['video_path']]
            serial = info['QuickTime:CameraSerialNumber']
            role = f"{SIDES[c]}_wrist" if c < n_grippers else 'head'
            records.append(camera_record(role, serial, info, video_fps, config['cameras'][serial]))
        sync_record = {'method': mcap.SYNC_METHOD if any(c.get('clock_correction_ms') for c in e['cameras'][:n_grippers])
            else 'audio cross-correlation'}
        for c, camera in enumerate(e['cameras'][:n_grippers]):
            line = (h['fits'].get(pathlib.Path(camera['video_path']).parent.name) or dict()).get('line')
            if line is not None:
                sync_record[f'camera{c}_sync_sigma_ms'] = f"{line['residual_ms']:.2f}"
        T_head = h['track']['head']['T_world_cam'][a:b]
        head_poses = {'T_world_cam': T_head, 'valid': h['track']['head']['is_valid'][a:b] & ~np.isnan(T_head[:, 0, 0]),
            'row_time_offset': h['readout'] / 2, 'imu_time_offset': h['imu_time_offset']}
        jobs.append((str(directory.joinpath('episode.mcap')), k, e, demos_dir, world_tag, video_fps, tcps_in_camera,
            records, sync_record, h['utc_shift'], head_poses))
        episode_info[k] = {'episode_id': episode_id, 'dir': directory, 'head': head, 'head_range': (a, b),
            'records': records}

    print(f"{len(indices)} episodes of {session.name} -> {output}")
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(num_workers, len(jobs)),
            mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = {executor.submit(write_mcap, *job): job[1] for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            future.result()

    manifest_rows = list()
    # annotations (08 / 09) per dock-to-dock episode, moved onto the delivered episodes
    annotations = list() if no_annotations else \
        [json.load(p.open('r')) for p in sorted(session.glob('tag_tracking/annotations/*_ep*.json'))]
    for k in indices:
        e, x = plan[k], episode_info[k]
        head, (a, b) = x['head'], x['head_range']
        h = heads[head['name']]
        track = h['track']
        n_grippers = len(e['grippers'])
        directory = x['dir']
        # frames.parquet: one row per head frame in episode.mcap. The head video there starts at the
        # keyframe before the task (not re-encoded), so the rows start at that keyframe too: the lead-in
        # frames before the task have rows (null pose where not tracked), camera_frame_index starts at 0
        first_frame = first_video_frame(str(head['dir'].joinpath('raw_video.mp4')), a)
        start = float(track['time'][first_frame])
        end = float(track['time'][b]) if b < len(track['time']) else float(track['time'][-1]) + 1e-3
        out = episode_frames(track['time'], track['head']['T_world_cam'], track['head']['is_valid'], start, end,
            h['grippers'], row_time_offset=h['readout'] / 2)
        rows = out['frames']
        n_rows = b - first_frame
        assert len(rows['time']) == n_rows, f"episode {k}: {len(rows['time'])} rows for head frames {first_frame}-{b}"
        table = frames_table(x['episode_id'], rows, h['utc_shift'], head['start_timestamp'], first_frame, created)
        pq.write_table(table, directory.joinpath('frames.parquet'))
        row_ns = table['timestamp_ns'].to_numpy()
        frame_rate = float(1e9 / np.median(np.diff(row_ns))) if len(row_ns) > 1 else None
        capture_start = (head['start_timestamp'] + float(track['time'][first_frame]) + h['utc_shift'])
        capture_end = capture_start + n_rows / float(head['fps'])
        # the task (bang to bang) inside the episode, seconds on the primary camera clock from the first frame
        task_interval_s = [(a - first_frame) / float(head['fps']), (b - first_frame) / float(head['fps'])]
        source_annotation = find_annotation(annotations, head['name'], float(track['time'][a]), float(track['time'][b - 1]))
        annotation = None if source_annotation is None else delivery_annotation(source_annotation,
            float(track['time'][first_frame]), n_rows / float(head['fps']), x['episode_id'])
        if annotation is not None:
            json.dump(annotation, directory.joinpath('annotations.json').open('w'), indent=2)
        elif directory.joinpath('annotations.json').is_file():
            directory.joinpath('annotations.json').unlink()  # from an earlier export

        # sensors.json
        counts = channel_counts(directory.joinpath('episode.mcap'))
        sensors, cameras, streams = list(), list(), list()
        for c, camera in enumerate(e['cameras']):
            is_head = c == n_grippers
            serial = x['records'][c]['serial']
            cam = config['cameras'][serial]
            sensor_id = 'head' if is_head else f'wrist_{SIDES[c]}'
            role = 'ego_head' if is_head else f'wrist_{SIDES[c]}'
            video_name = pathlib.Path(camera['video_path']).parent.name
            enc = video_encoding(str(demos_dir.joinpath(camera['video_path'])))
            # this camera's clock: audio sync offset + motion sync correction (wrist cameras)
            offset_s = offsets[video_name]['offset_sec'] + (camera.get('clock_correction_ms') or 0.0) / 1e3
            topic = f'/cameras/camera{c}/video'
            record = {
                'sensor_id': sensor_id, 'sensor_type': 'camera', 'role': role,
                'serial': serial, 'model': x['records'][c].get('model'), 'lens': x['records'][c].get('lens'),
                'parent_frame': 'world' if is_head else f'tcp_{SIDES[c]}',
                'child_frame': f'{sensor_id}_optical',
                'sample_rate_hz': float(head['fps']) if is_head else float(by_name[video_name]['fps']),
                'sample_rate_source': 'nominal (container frame rate)',
                'time_offset_ns': int(round(offset_s * 1e9)),
                'encoding': f"{enc['codec']}, {enc['bit_rate']} bit/s, {enc['pix_fmt']}, {enc['color_range']} range, "
                    "Annex B access units, not re-encoded",
                'mcap_topic': topic, 'message_schema': 'foxglove.CompressedVideo',
                'width_px': int(cam.image_size[0]), 'height_px': int(cam.image_size[1]),
                'intrinsics_33': [float(v) for v in np.asarray(cam.K).reshape(-1)],
                'distortion_model': 'kannala_brandt',
                'distortion_coeffs': [float(v) for v in np.asarray(cam.D).reshape(-1)],
                'extrinsic_44': None if is_head else mat44(np.linalg.inv(np.array(tcps[c]['T_wrist_tcp']))),
                'extrinsic_direction': ('none: moving camera, per frame camera_pose_head in frames.parquet '
                    '(world from camera)') if is_head else f'tcp_{SIDES[c]} from camera (child in parent)',
                'has_mirror': False,
                'is_primary': is_head,
                'calibration_source': 'measured',
            }
            sensors.append(record)
            cameras.append({'id': sensor_id, 'role': role, 'width': int(cam.image_size[0]),
                'height': int(cam.image_size[1]), 'frame_count': counts.get(topic, 0)})
        for g in range(n_grippers):
            side = SIDES[g]
            camera = e['cameras'][g]
            serial = x['records'][g]['serial']
            video_dir = demos_dir.joinpath(camera['video_path']).parent
            imu = load_imu(str(video_dir.joinpath('imu_data.json')))
            imu_rate = float(1 / np.median(np.diff(imu['time'])))
            offset_s = offsets[video_dir.name]['offset_sec'] + (camera.get('clock_correction_ms') or 0.0) / 1e3 \
                + (camera.get('imu_time_offset') or 0.0)
            imu_calibration = config.get('gripper_imu', dict()).get(serial)
            T_cam_imu = None
            if imu_calibration is not None:
                ci = json.load(open(imu_calibration['cam_imu_calibration'], 'r'))
                T_imu_cam = np.eye(4)
                T_imu_cam[:3, :3] = quat_to_mat(ci['q_i_c'])
                T_imu_cam[:3, 3] = [ci['t_i_c']['x'], ci['t_i_c']['y'], ci['t_i_c']['z']]
                T_cam_imu = np.linalg.inv(T_imu_cam)
            sensors += [{
                'sensor_id': f'imu_{side}', 'sensor_type': 'imu', 'role': f'wrist_{side}',
                'parent_frame': f'wrist_{side}_optical', 'child_frame': f'imu_{side}',
                'sample_rate_hz': imu_rate, 'sample_rate_source': 'measured (median sample interval)',
                'time_offset_ns': int(round(offset_s * 1e9)),
                'encoding': 'json: linear_acceleration m/s^2, angular_velocity rad/s, GoPro GPMF axes',
                'mcap_topic': f'robot{g}/imu', 'message_schema': 'umi.msg.ImuSample',
                'extrinsic_44': None if T_cam_imu is None else mat44(T_cam_imu),
                'extrinsic_direction': f'wrist_{side}_optical from imu_{side} (child in parent)',
                'is_primary': False,
                'calibration_source': 'none' if T_cam_imu is None else
                    ('approximate' if 't_i_c_source' in ci or 'q_i_c_source' in ci else 'measured'),
            }, {
                'sensor_id': f'tcp_{side}', 'sensor_type': 'pose_tracker', 'role': f'wrist_{side}',
                'parent_frame': 'world', 'child_frame': f'tcp_{side}',
                'sample_rate_hz': frame_rate, 'sample_rate_source': 'measured (episode steps)',
                'time_offset_ns': 0,
                'encoding': 'json: pos m, quat_xyzw (world from tcp); no message where the pose is not available',
                'mcap_topic': f'robot{g}/eef/pose', 'message_schema': 'umi.msg.PoseQuat',
                # the same poses in the UMI form (axis-angle rotation, rad)
                'mcap_topic_axis_angle': f'robot{g}/eef/state', 'message_schema_axis_angle': 'umi.msg.EefState',
                'is_primary': False, 'calibration_source': 'measured',
            }, {
                'sensor_id': f'gripper_{side}', 'sensor_type': 'gripper', 'role': f'wrist_{side}',
                'parent_frame': f'tcp_{side}', 'child_frame': f'gripper_{side}',
                'sample_rate_hz': frame_rate, 'sample_rate_source': 'measured (episode steps)',
                'time_offset_ns': 0,
                'encoding': 'json: width m, finger opening relative to fully closed, from the finger tags in the wrist camera',
                'mcap_topic': f'robot{g}/gripper/state', 'message_schema': 'umi.msg.GripperState',
                'is_primary': False, 'calibration_source': 'measured',
            }]
            streams += [{'id': f'imu_{side}', 'type': 'imu', 'sample_count': counts.get(f'robot{g}/imu', 0),
                'native_rate_hz': imu_rate},
                {'id': f'tcp_{side}', 'type': 'pose', 'sample_count': counts.get(f'robot{g}/eef/pose', 0),
                'native_rate_hz': frame_rate},
                {'id': f'gripper_{side}', 'type': 'gripper', 'sample_count': counts.get(f'robot{g}/gripper/state', 0),
                'native_rate_hz': frame_rate}]
        # head camera pose (moving camera) and head IMU
        sensors.append({
            'sensor_id': 'head_pose', 'sensor_type': 'pose_tracker', 'role': 'ego_head',
            'parent_frame': 'world', 'child_frame': 'head_optical',
            'sample_rate_hz': float(head['fps']), 'sample_rate_source': 'nominal (head camera frames)',
            'time_offset_ns': 0,
            'encoding': 'json: pos m, quat_xyzw (world from camera); per tracked head frame, at the middle image '
                'row; no message where the head is not tracked',
            'mcap_topic': '/head/pose', 'message_schema': 'umi.msg.PoseQuat',
            'is_primary': False, 'calibration_source': 'measured',
        })
        streams.append({'id': 'head_pose', 'type': 'pose', 'sample_count': counts.get('/head/pose', 0),
            'native_rate_hz': float(head['fps'])})
        if h['imu_time_offset'] is not None:
            head_imu = load_imu(str(head['dir'].joinpath('imu_data.json')))
            head_imu_rate = float(1 / np.median(np.diff(head_imu['time'])))
            ci = json.load(open(config['head_imu']['cam_imu_calibration'], 'r'))
            T_imu_cam = np.eye(4)
            T_imu_cam[:3, :3] = quat_to_mat(ci['q_i_c'])
            T_imu_cam[:3, 3] = [ci['t_i_c']['x'], ci['t_i_c']['y'], ci['t_i_c']['z']]
            sensors.append({
                'sensor_id': 'imu_head', 'sensor_type': 'imu', 'role': 'ego_head',
                'parent_frame': 'head_optical', 'child_frame': 'imu_head',
                'sample_rate_hz': head_imu_rate, 'sample_rate_source': 'measured (median sample interval)',
                'time_offset_ns': int(round((offsets[head['name']]['offset_sec'] + h['imu_time_offset']) * 1e9)),
                'encoding': 'json: linear_acceleration m/s^2, angular_velocity rad/s, GoPro GPMF axes',
                'mcap_topic': '/head/imu', 'message_schema': 'umi.msg.ImuSample',
                'extrinsic_44': mat44(np.linalg.inv(T_imu_cam)),
                'extrinsic_direction': 'head_optical from imu_head (child in parent)',
                'is_primary': False, 'calibration_source': 'approximate' if 't_i_c_source' in ci or 'q_i_c_source' in ci else 'measured',
            })
            streams.append({'id': 'imu_head', 'type': 'imu', 'sample_count': counts.get('/head/imu', 0),
                'native_rate_hz': head_imu_rate})
        sensors.append({
            'sensor_id': 'rig', 'sensor_type': 'pose_tracker', 'role': 'other',
            'description': 'handheld bimanual grippers, each with a wrist GoPro and a 70 mm ArUco cube, '
                'tracked by the head GoPro against the environment tag map, fused with the wrist IMUs and '
                'the environment tags seen by the wrist cameras',
            'parent_frame': 'world', 'child_frame': 'world',
            'world_frame': f'environment tag map, origin at tag {world_tag}',
            'sample_rate_hz': frame_rate, 'sample_rate_source': 'measured (episode steps)',
            'time_offset_ns': 0, 'encoding': 'none', 'is_primary': False, 'calibration_source': 'measured',
        })
        for r in sensors:
            r['created_at_utc'] = created.date().isoformat()
            r['created_ts_utc'] = utc_iso(created.timestamp())
        json.dump(sensors, directory.joinpath('sensors.json').open('w'), indent=2)

        # task and outcome: declared in the delivery yaml first, else from the annotation
        task = annotation['tasks'][0] if annotation and annotation['tasks'] else None
        # per episode from its annotation, else as declared for the session
        outcome = task['outcome'] if task and task.get('outcome') else (s.get('outcome') or 'unknown')
        success = {'success': True, 'failure': False, 'partial': False, 'aborted': False}.get(outcome)
        task_family = s.get('task_family') or (task['skill'] if task and task.get('skill') in TASK_FAMILIES else None)
        object_classes = sorted({o.get('object_class') for o in (annotation or {}).get('objects', [])
            if task and o.get('name') in task.get('objects', []) and o.get('object_class')})
        failures = [{'start_s': r['failure_t_s'], 'end_s': r['recovered_at_s'] if r.get('recovered_at_s') is not None
            else n_rows / float(head['fps']), 'kind': 'failure', 'failure_kind': r.get('failure_kind'),
            'text': r.get('failure'), 'recovered': r.get('recovered')} for r in (annotation or {}).get('recoveries', [])]
        # episode.json
        widths = {SIDES[g]: [0.0, float(h['grippers'][g]['gripper_range']['max_width']
            - h['grippers'][g]['gripper_range']['min_width'])] for g in range(n_grippers)}
        source_uri = None if d['source_uri_prefix'] is None else \
            f"{d['source_uri_prefix'].rstrip('/')}/{session.name}/raw_videos/"
        episode = {
            'schema_version': d['schema_version'], 'vendor_name': d['vendor_name'], 'dataset_name': d['dataset_name'],
            'delivery_id': d['delivery_id'], 'episode_id': x['episode_id'], 'recording_id': h['recording_id'],
            'tier': TIER, 'source_type': SOURCE_TYPE, 'embodiment_class': EMBODIMENT, 'robot_model': d['robot_model'],
            'source_uri': source_uri,
            'source_files': {c: pathlib.Path(camera['video_path']).parent.name for c, camera in
                zip([*(f'wrist_{SIDES[g]}' for g in range(n_grippers)), 'head'], e['cameras'])},
            # frames of the original videos: the task, and the head frames the episode (rows) starts at
            'source_frames': {c: [int(v) for v in camera['video_start_end']] for c, camera in
                zip([*(f'wrist_{SIDES[g]}' for g in range(n_grippers)), 'head'], e['cameras'])},
            'source_first_frame_head': int(first_frame),
            'task_interval_s': task_interval_s,
            'gripper_type': d['gripper_type'],
            'gripper_range': [0.0, max(v[1] for v in widths.values())],
            'gripper_range_per_arm': widths,
            'gripper_units': 'm',
            'capture_start_utc': utc_iso(capture_start), 'capture_end_utc': utc_iso(capture_end),
            'duration_ns': int(round((capture_end - capture_start) * 1e9)),
            'primary_camera_id': 'head',
            'nominal_frame_rate_hz': frame_rate,
            'clock_domain': 'utc_unix',
            'clock_sync_accuracy_ns': h['sync_accuracy_ns'],
            'clock_sync_method': sync_record['method'],
            'base_frame': f'world: environment tag map, origin at tag {world_tag}',
            'pose_transform_direction': 'world_from_tcp (pose maps tcp coordinates to world)',
            'task_text': s.get('task_text') or (task['task'] if task else None), 'task_family': task_family,
            'success': success, 'outcome': outcome,
            **({'events': failures} if failures else {}),
            'operator_id': s.get('operator_id'), 'station_id': s.get('station_id'),
            'environment_id': s.get('environment_id'),
            'scene_tags': s.get('scene_tags') or [], 'object_tags': s.get('object_tags') or object_classes,
            'consent_reference': s.get('consent_reference') or d['consent_reference'],
            'license_reference': d['license_reference'], 'anonymization_state': d['anonymization_state'],
            'created_at_utc': created.date().isoformat(), 'created_ts_utc': utc_iso(created.timestamp()),
        }
        # fields without a value are left out (not written as null)
        json.dump({k: v for k, v in episode.items() if v is not None}, directory.joinpath('episode.json').open('w'), indent=2)

        files = [{'path': f"{x['episode_id']}/{p.name}", 'byte_size': p.stat().st_size, 'sha256': sha256(p)}
            for p in sorted(directory.iterdir()) if p.is_file()]
        manifest_rows.append({
            **{key: episode[key] for key in ('schema_version', 'vendor_name', 'dataset_name', 'delivery_id', 'episode_id',
                'recording_id', 'tier', 'source_type', 'embodiment_class', 'robot_model', 'capture_start_utc',
                'capture_end_utc', 'duration_ns', 'primary_camera_id', 'task_text', 'task_family', 'success', 'outcome',
                'operator_id', 'station_id', 'environment_id', 'nominal_frame_rate_hz', 'object_tags', 'scene_tags',
                'anonymization_state', 'consent_reference', 'license_reference', 'created_ts_utc')},
            'container_format': CONTAINER,
            'primary_frame_count': int(n_rows),
            'cameras': cameras, 'streams': streams, 'files': files,
            'total_byte_size': int(sum(f['byte_size'] for f in files)),
            'has_proprio': False, 'has_actions': False, 'has_ee_pose': True, 'has_gripper': True,
            'has_calibration': True, 'has_kinematics': False,
            # the session (one task per session); prefixed with the dataset name where declared
            'task_id': f"{d['dataset_name']}:{session.name}" if d['dataset_name'] else session.name,
            'has_failure_spans': bool(failures), 'has_event_timeline': annotation is not None,
            'collection_mode': s.get('collection_mode'), 'n_tasks': len(annotation['tasks']) if annotation else 1,
            'pose_source': 'visual_inertial', 'gripper_width_source': 'marker',
            # who wrote the task text, spans and event timeline: the annotation's source, a person
            # where only the declared task text exists, else none
            'annotation_source': annotation['annotation_source'] if annotation else ('human' if s.get('task_text') else None),
            'validation_status': 'not_run',
            'created_at_utc': created.date(),
        })
        print(f"{x['episode_id']} (plan episode {k}): {n_rows} frames ({a - first_frame} lead-in), {utc_iso(capture_start)}, pose valid "
            f"{rows['pose_valid'].mean() * 100:.1f}% (task part {rows['pose_valid'][a - first_frame:].mean() * 100:.1f}%), "
            f"{', '.join(f'{t} {n}' for t, n in sorted(counts.items()))}")

    n = write_manifest(output, manifest_rows)
    missing = [k for k in ('vendor_name', 'dataset_name', 'delivery_id', 'robot_model', 'license_reference',
        'anonymization_state') if d[k] is None] + [k for k in ('task_text', 'task_family') if s.get(k) is None]
    print(f"Saved {len(indices)} episodes, manifest.parquet with {n} episodes in {output}")
    if missing:
        print(f"not declared (null in the delivery, see {delivery}): {', '.join(missing)}")

# %%
if __name__ == "__main__":
    main()
