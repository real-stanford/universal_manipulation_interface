"""
Export the episodes of a processed session (dataset_plan.pkl) as one folder per recording with
three video files, per-frame data and metadata, as the data PRD's UMI delivery asks for:

  <out>/<name>_ep<k>/
    <name>_ep<k>_head.mp4, _left.mp4, _right.mp4
        the cameras' original H.265 video, not re-encoded: copied from the keyframe at or before
        the episode's first step (GoPro: one keyframe per second) to its last step; tagged hvc1
        (QuickTime / Safari need it), moov first (faststart)
    <name>_ep<k>_frames.csv
        one row per episode step (= one frame of each camera): the frame index of that step in each
        of the three files, time from the first step, per gripper its width (m, 0 = fully closed),
        pose_valid and TCP pose (world frame, m + quaternion xyzw; empty where pose_valid is false)
    <name>_ep<k>_meta.json
        per camera: model, lens, field of view, resolution, frame rate, Kannala-Brandt intrinsics and
        their source; sync: method, the sync event (the episode's first step) and its frame index in
        each file, wrist clock corrections and their uncertainty; gripper width and TCP conventions
    <name>_ep<k>_mosaic.mp4 (optional, --mosaic / --mosaic_episodes)
        review video: head on top, right and left wrist below, all three from the sync frame on;
        H.264 (yuv420p, faststart, any player), --mosaic_fps (30), head 1280x960; the only part that
        decodes and encodes video. Not a substitute for the three files.

The sync event is the episode's first step: the same instant in all three files (each camera's frame
nearest to it, within half a frame); frame f of a file belongs to step f - sync_frame_index.

python scripts_tag_tracking/14_export_rhoda.py -i <session> -c <config> -o <out> --name station3_2026-11-21_task01 [--mosaic 3]
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import concurrent.futures
import csv
import json
import pathlib
import pickle
import subprocess
import click
import av
import numpy as np
from scipy.spatial.transform import Rotation
from umi.common.exiftool_util import ExifTool
from umi.tag_tracking.config import load_tag_tracking_config
from umi.tag_tracking.camera_metadata import camera_record

ROBOT_SIDES = ('right', 'left')   # UMI: robot0 right, robot1 left
SYNC_METHOD = 'audio cross-correlation of all cameras, wrist clocks refined per episode from wrist IMU against head-tracked motion'


def first_keyframe(video_path: str, start_frame: int) -> int:
    """
    Index of the keyframe at or before start_frame.
    """
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        t0 = float((stream.start_time or 0) * stream.time_base)
        container.seek(int(max(0.0, start_frame / fps - 0.001) / stream.time_base), stream=stream,
            backward=True, any_frame=False)
        best = None
        for packet in container.demux(stream):
            if packet.pts is None:
                continue
            index = int(round((float(packet.pts * stream.time_base) - t0) * fps))
            if index > start_frame:
                break
            if packet.is_keyframe:
                best = index
        assert best is not None, f"{video_path}: no keyframe at or before frame {start_frame}"
        return best


def copy_clip(video_path: str, first_frame: int, stop_frame: int, out_path: str) -> int:
    """
    Frames [first_frame, stop_frame) of the video stream into an mp4, packets copied (no decoding),
    first_frame must be a keyframe; tag hvc1, faststart. Returns the number of frames written.
    """
    with av.open(video_path) as src, av.open(out_path, 'w', format='mp4', options={'movflags': '+faststart'}) as dst:
        stream = src.streams.video[0]
        fps = float(stream.average_rate)
        t0 = float((stream.start_time or 0) * stream.time_base)
        out = dst.add_stream(template=stream)
        out.codec_tag = 'hvc1'
        src.seek(int(max(0.0, first_frame / fps - 0.001) / stream.time_base), stream=stream,
            backward=True, any_frame=False)
        offset, n = None, 0
        for packet in src.demux(stream):
            if packet.pts is None or packet.size == 0:
                continue
            index = int(round((float(packet.pts * stream.time_base) - t0) * fps))
            if index < first_frame:
                continue
            if index >= stop_frame:
                break
            if offset is None:
                assert packet.is_keyframe and index == first_frame, f"{video_path}: frame {index} is not the keyframe {first_frame}"
                offset = packet.pts
            packet.pts -= offset
            packet.dts = packet.pts if packet.dts is None else packet.dts - offset
            packet.stream = out
            dst.mux(packet)
            n += 1
    return n


def mosaic(head: tuple, right: tuple, left: tuple, n_frames: int, out_path: str, fps: float,
        encoder: str = 'libx264', width: int = 1280):
    """
    Review mosaic (ffmpeg): head on top (width x 3/4 width), right and left wrist below (half width
    each), each input from its sync frame on (file, sync_frame_index), n_frames frames of the source
    rate, written at fps, H.264 yuv420p faststart.
    """
    w2 = width // 2
    h1, h2 = width * 3 // 4, w2 * 3 // 4
    inputs = list()
    for path, _ in (head, right, left):
        inputs += ['-i', str(path)]
    trims = [f"[{k}:v]trim=start_frame={sync}:end_frame={sync + n_frames},setpts=PTS-STARTPTS,"
        f"scale={size},setsar=1[v{k}]" for k, ((_, sync), size) in
        enumerate(zip((head, right, left), (f'{width}:{h1}', f'{w2}:{h2}', f'{w2}:{h2}')))]
    graph = ';'.join(trims) + f";[v1][v2]hstack=inputs=2[bottom];[v0][bottom]vstack=inputs=2,fps={fps},format=yuv420p[out]"
    codec = ['-c:v', encoder] + (['-preset', 'veryfast', '-crf', '23'] if encoder == 'libx264' else ['-preset', 'p4', '-cq', '23'])
    subprocess.run(['ffmpeg', '-v', 'error', '-y', *inputs, '-filter_complex', graph, '-map', '[out]', *codec,
        '-movflags', '+faststart', '-an', str(out_path)], check=True)


def export_episode(k: int, episode: dict, demos_dir: pathlib.Path, out_dir: pathlib.Path, name: str,
        meta_base: dict, camera_records: list, sync_sigmas: dict) -> dict:
    """
    One recording folder: three mp4s, frames.csv, meta.json. Returns the files and sync indices.
    """
    rec = f"{name}_ep{k:04d}"
    folder = out_dir.joinpath(rec)
    folder.mkdir(parents=True, exist_ok=True)
    cameras = episode['cameras']
    grippers = episode['grippers']
    n_steps = len(episode['episode_timestamps'])
    roles = [ROBOT_SIDES[c] if c < len(grippers) else 'head' for c in range(len(cameras))]
    files, sync_index = dict(), dict()
    for c, (camera, role) in enumerate(zip(cameras, roles)):
        start, stop = (int(x) for x in camera['video_start_end'])
        assert stop - start == n_steps, f"episode {k} camera {role}: {stop - start} frames for {n_steps} steps"
        video = str(demos_dir.joinpath(camera['video_path']))
        first = first_keyframe(video, start)
        path = folder.joinpath(f"{rec}_{role}.mp4")
        n = copy_clip(video, first, stop, str(path))
        assert n == stop - first, f"{path}: {n} frames written, {stop - first} expected"
        files[role], sync_index[role] = path.name, start - first

    times = np.asarray(episode['episode_timestamps'], dtype=np.float64)
    with folder.joinpath(f"{rec}_frames.csv").open('w', newline='') as f:
        w = csv.writer(f)
        header = ['step', 'time_s'] + [f'{role}_frame' for role in ('head', 'left', 'right') if role in files]
        for g in range(len(grippers)):
            side = ROBOT_SIDES[g]
            header += [f'{side}_gripper_width_m', f'{side}_pose_valid'] + \
                [f'{side}_tcp_{a}' for a in ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw')]
        w.writerow(header)
        quats = [Rotation.from_rotvec(np.nan_to_num(np.asarray(g['tcp_pose'])[:, 3:])).as_quat() for g in grippers]
        for i in range(n_steps):
            row = [i, f"{times[i] - times[0]:.6f}"] + [sync_index[role] + i for role in ('head', 'left', 'right') if role in files]
            for g, gripper in enumerate(grippers):
                pose = np.asarray(gripper['tcp_pose'])[i]
                valid = bool(np.asarray(gripper.get('pose_valid', ~np.isnan(np.asarray(gripper['tcp_pose'])[:, 0])))[i]) \
                    and not np.isnan(pose).any()
                width = float(np.asarray(gripper['gripper_width']).reshape(-1)[i])
                row += [f"{width:.5f}" if np.isfinite(width) else '', int(valid)]
                row += ([f"{v:.5f}" for v in pose[:3]] + [f"{v:.6f}" for v in quats[g][i]]) if valid else [''] * 7
            w.writerow(row)

    meta = {
        **meta_base,
        'recording': rec,
        'episode_index': k,
        'n_frames': n_steps,
        'duration_s': round(float(times[-1] - times[0]) + float(np.median(np.diff(times))) if n_steps > 1 else 0.0, 3),
        'files': {role: files[role] for role in ('head', 'left', 'right') if role in files},
        'cameras': {role: rec_ for role, rec_ in zip(roles, camera_records)} if camera_records else None,
        'sync': {
            'method': SYNC_METHOD,
            'event': "the episode's first step: the same instant in all three files (each camera's frame nearest "
                     "to it, within half a frame); frame f of a file belongs to step f - sync_frame_index",
            'sync_frame_index': {role: sync_index[role] for role in ('head', 'left', 'right') if role in sync_index},
            'clock_correction_ms': {role: round(float(cam['clock_correction_ms']), 3) for cam, role in zip(cameras, roles)
                if cam.get('clock_correction_ms') is not None},
            'sync_sigma_ms': {roles[c]: s for c, s in sync_sigmas.items()},
        },
        'frames_csv': f"{rec}_frames.csv",
    }
    folder.joinpath(f"{rec}_meta.json").write_text(json.dumps(meta, indent=2))
    return {'recording': rec, 'folder': folder, 'files': files, 'sync_index': sync_index, 'n_frames': n_steps}


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Session directory with dataset_plan.pkl')
@click.option('-c', '--config', required=True, help='Tag tracking config (camera intrinsics, gripper config)')
@click.option('-o', '--output', required=True, help='Output directory, one folder per recording')
@click.option('--name', default=None, help='Recording name prefix, e.g. station3_2026-11-21_task01 (default: session folder)')
@click.option('-e', '--episodes', default=None, help='Comma separated plan episodes, default all')
@click.option('--mosaic', 'n_mosaic', type=int, default=0, help='Also a mosaic for this many episodes, evenly over the session')
@click.option('--mosaic_episodes', default=None, help='Comma separated plan episodes with a mosaic (instead of --mosaic)')
@click.option('--mosaic_fps', type=float, default=30.0)
@click.option('--mosaic_encoder', default='libx264', type=click.Choice(['libx264', 'h264_nvenc']))
@click.option('-j', '--jobs', type=int, default=4, help='Recordings written in parallel (copying, disk bound)')
def main(input, config, output, name, episodes, n_mosaic, mosaic_episodes, mosaic_fps, mosaic_encoder, jobs):
    session = pathlib.Path(os.path.expanduser(input)).absolute()
    out_dir = pathlib.Path(os.path.expanduser(output)).absolute()
    name = name or session.name
    plan = pickle.load(session.joinpath('dataset_plan.pkl').open('rb'))
    indices = list(range(len(plan))) if episodes is None else [int(x) for x in episodes.split(',')]
    assert all(0 <= k < len(plan) for k in indices), f"plan has {len(plan)} episodes"
    demos_dir = session.joinpath('demos')
    config = load_tag_tracking_config(config)

    bodies = json.load(session.joinpath('tag_tracking', 'bodies.json').open('r'))['bodies']
    world_tag = [b['anchor_tag'] for b in bodies.values() if b['static']][0]
    tcp = {x['robot_index']: x for x in json.load(session.joinpath('tag_tracking', 'gripper_tcp.json').open('r')).values()}
    meta_base = {
        'session': session.name,
        'world_frame': f"environment tag {world_tag} (anchor of the static tag map), z up from the table",
        'gripper': {
            'width_unit': 'm', 'width_zero': 'fully closed',
            'width_source': 'finger tag distance measured in the wrist camera every frame, '
                            'mapped to the gripper range of its calibration clip',
            'right': 'robot0', 'left': 'robot1',
        },
        'tcp': {
            'frame': 'world', 'position_unit': 'm', 'orientation': 'quaternion xyzw',
            'in_wrist_camera_m': {ROBOT_SIDES[r]: [round(float(v), 5) for v in np.array(t['T_wrist_tcp'])[:3, 3]]
                for r, t in tcp.items()},
        },
    }

    # per camera metadata (cameras are the same for all episodes of a session)
    first_cameras = plan[indices[0]]['cameras']
    with ExifTool() as et:
        exif = {c['video_path']: list(et.get_metadata(str(demos_dir.joinpath(c['video_path']))))[0] for c in first_cameras}
    with av.open(str(demos_dir.joinpath(first_cameras[0]['video_path']))) as container:
        fps = float(container.streams.video[0].average_rate)
    n_grippers = len(plan[indices[0]]['grippers'])
    camera_records = list()
    for c, camera in enumerate(first_cameras):
        info = exif[camera['video_path']]
        serial = info['QuickTime:CameraSerialNumber']
        role = f"{ROBOT_SIDES[c]}_wrist" if c < n_grippers else 'head'
        rec = camera_record(role, serial, info, fps, config['cameras'][serial])
        rec['intrinsics_source'] = 'OpenICC calibration of this camera: ' + os.path.basename(config['cameras'][serial].intrinsics_json)
        camera_records.append(rec)

    sync_path = session.joinpath('tag_tracking', 'wrist_sync.json')
    sync = json.load(sync_path.open('r')) if sync_path.is_file() else {'videos': {}}

    def sigmas(episode):
        head = pathlib.Path(episode['cameras'][-1]['video_path']).parent.name
        fits = sync['videos'].get(head, dict())
        out = dict()
        for c, camera in enumerate(episode['cameras'][:n_grippers]):
            line = (fits.get(pathlib.Path(camera['video_path']).parent.name) or dict()).get('line')
            if line is not None:
                out[c] = round(float(line['residual_ms']), 2)
        return out

    print(f"{len(plan)} episodes in the plan, exporting {len(indices)} to {out_dir}", flush=True)
    results = dict()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, min(jobs, len(indices)))) as pool:
        futures = {pool.submit(export_episode, k, plan[k], demos_dir, out_dir, name, meta_base, camera_records,
            sigmas(plan[k])): k for k in indices}
        for future in concurrent.futures.as_completed(futures):
            r = future.result()
            results[futures[future]] = r
            print(f"{r['recording']}: {r['n_frames']} frames, sync frame index "
                + ', '.join(f"{role} {i}" for role, i in r['sync_index'].items()), flush=True)

    if mosaic_episodes:
        chosen = [int(x) for x in mosaic_episodes.split(',')]
    elif n_mosaic > 0:
        chosen = sorted({indices[int(round(i))] for i in np.linspace(0, len(indices) - 1, min(n_mosaic, len(indices)))})
    else:
        chosen = list()
    for k in chosen:
        if k not in results:
            print(f"episode {k}: no mosaic, not exported in this run")
            continue
        r = results[k]
        path = r['folder'].joinpath(f"{r['recording']}_mosaic.mp4")
        mosaic(*((r['folder'].joinpath(r['files'][role]), r['sync_index'][role]) for role in ('head', 'right', 'left')),
            r['n_frames'], path, mosaic_fps, mosaic_encoder)
        print(f"{path.name}: mosaic", flush=True)

# %%
if __name__ == "__main__":
    main()
