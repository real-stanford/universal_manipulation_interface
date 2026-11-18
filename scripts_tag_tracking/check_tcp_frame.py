"""
Visual check of the TCP (fingertip point) on one head camera frame: the
gripper's TCP (green, T_cube_tcp of tag_tracking/gripper_tcp.json), a line from
it straight down to the table plane z = 0 of the world frame (yellow) and the
cube axes.

The frame: --time (seconds of the head video), else the step of the plan
(dataset_plan.pkl) where the robot's TCP is lowest, e.g. fingertips on the table;
without a plan, the tracked head frame where it is lowest (cube pose x T_cube_tcp).
Head video: the longest of the session, or --head_video (demos/ directory name).
Reads tag_tracking.pkl (cube and head poses), tag_tracking/gripper_tcp.json
(T_cube_wrist, T_wrist_tcp), the tag tracking config.
Also the gripper's own wrist camera frame at the same time with the TCP
projected (the TCP is defined in the wrist camera frame, so this checks the
camera -> TCP geometry directly, independent of the head and the world frame).
--compare_tcp Y Z: also draw (magenta) a TCP at (0, Y, Z) in the wrist camera
frame (m, e.g. measured: Y below the lens centre, Z in front of the camera).
Default both grippers in one image: the head frame with both TCPs on top, the
grippers' wrist camera frames at the same time below (robot0, robot1).
Writes <session>/tag_tracking/tcp_check_robots<k..>_<time>s.png

python scripts_tag_tracking/check_tcp_frame.py -i data_workspace/session -c umi_calib/tag_tracking.yaml --compare_tcp 0.075 0.192
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
import av
import cv2
import numpy as np
from umi.tag_tracking.config import load_tag_tracking_config
from umi.tag_tracking.session import get_session_videos, overlapping


def read_frame(video_path, time_sec):
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        container.seek(int(max(0, time_sec - 1) / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            if abs(float(frame.pts * stream.time_base) - time_sec) < 0.5 / float(stream.average_rate):
                return frame.to_ndarray(format='bgr24')
    raise RuntimeError(f"no frame at {time_sec}s in {video_path}")


def put_labels(img, labels):
    for k, text in enumerate(labels):
        cv2.putText(img, text, (30, 70 + 60 * k), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 8)
        cv2.putText(img, text, (30, 70 + 60 * k), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)


def project(camera, T_cam_world, points_world):
    points = (T_cam_world[:3, :3] @ np.asarray(points_world).T).T + T_cam_world[:3, 3]
    return camera.project(points), points[:, 2]


# %%
def wrist_view(config, session, head, track_time, name, T_wrist_tcp, T_wrist_compare):
    """
    The gripper's wrist camera frame at head video time track_time, TCPs projected
    (the TCP is defined in the wrist camera frame: checks camera -> TCP directly).
    """
    serial = config['bodies'][name]['gripper_camera']
    # the wrist video recording at that time (a camera may have split the take into several)
    t_session = head['start_timestamp'] + track_time
    wrist = [x for x in overlapping(get_session_videos(session), head, serial)
        if x['start_timestamp'] <= t_session <= x['end_timestamp']][0]
    wrist_time = t_session - wrist['start_timestamp']
    img = read_frame(wrist['dir'].joinpath('raw_video.mp4'), wrist_time)
    wrist_camera = config['cameras'][serial]
    px = np.round(wrist_camera.project(T_wrist_tcp[None, :3, 3])).astype(int)
    cv2.circle(img, tuple(px[0]), 14, (0, 255, 0), -1)
    if T_wrist_compare is not None:
        cpx = np.round(wrist_camera.project(T_wrist_compare[None, :3, 3])).astype(int)
        cv2.circle(img, tuple(cpx[0]), 14, (255, 0, 255), -1)
    put_labels(img, [f"{name} wrist camera t={wrist_time:.2f}s"])
    return img


@click.command()
@click.option('-i', '--input', required=True, help='Session directory')
@click.option('-c', '--config', required=True, help='Tag tracking config yaml')
@click.option('-r', '--robot', type=int, default=None, help='Robot index (0 right, 1 left), default both')
@click.option('-t', '--time', 'time_sec', type=float, default=None,
    help='Head video time (s), default the plan step with the lowest TCP (of both: lowest higher one)')
@click.option('--head_video', default=None, help='Head video (demos/ directory name), default the longest')
@click.option('--compare_tcp', type=float, nargs=2, default=None,
    help='Also draw a TCP at (0, Y, Z) in the wrist camera frame (m): Y Z')
def main(input, config, robot, time_sec, compare_tcp, head_video):
    session = pathlib.Path(os.path.expanduser(input)).absolute()
    config = load_tag_tracking_config(config)
    camera = config['cameras'][config['head_camera']]
    names = [name for _, name in sorted((b['robot_index'], name)
        for name, b in config['bodies'].items() if 'gripper_camera' in b)]
    robots = list(range(len(names))) if robot is None else [robot]
    tcp_calibration = json.load(session.joinpath('tag_tracking', 'gripper_tcp.json').open('r'))

    heads = [x for x in get_session_videos(session) if x['camera_serial'] == config['head_camera']]
    head = [x for x in heads if x['name'] == head_video][0] if head_video else \
        max(heads, key=lambda x: x['end_timestamp'] - x['start_timestamp'])
    print(f"head video {head['name']}")
    track = pickle.load(head['dir'].joinpath('tag_tracking.pkl').open('rb'))
    if time_sec is None and not session.joinpath('dataset_plan.pkl').is_file():
        # lowest TCP (higher of the selected robots) over tracked head frames
        ok = track['head']['is_valid'].copy()
        z = np.full(len(track['time']), -np.inf)
        for r in robots:
            body = track['bodies'][names[r]]
            ok &= body['is_valid_world']
            tcp_z = (body['T_world_body'] @ np.array(tcp_calibration[names[r]]['T_cube_tcp']))[:, 2, 3]
            z = np.maximum(z, tcp_z)
        z[~ok] = np.inf
        i = int(np.argmin(z))
        time_sec = float(track['time'][i])
        print(f"tracked frame with the lowest TCP (no dataset_plan.pkl): z {z[i] * 100:.1f} cm at head video {time_sec:.2f}s")
    if time_sec is None:
        plan = pickle.load(session.joinpath('dataset_plan.pkl').open('rb'))
        z, t = min((max(e['grippers'][r]['tcp_pose'][k, 2] for r in robots), e['episode_timestamps'][k])
            for e in plan for k in range(len(e['episode_timestamps'])))
        time_sec = float(t - head['start_timestamp'])
        print(f"plan step with the lowest TCP ({'higher of both' if len(robots) > 1 else names[robots[0]]}): "
            f"z {z * 100:.1f} cm at head video {time_sec:.2f}s")
    valid = track['head']['is_valid'].copy()
    for r in robots:
        valid &= track['bodies'][names[r]]['is_valid']
    candidates = np.flatnonzero(valid)
    i = candidates[np.argmin(np.abs(track['time'][candidates] - time_sec))]
    T_world_cam = track['head']['T_world_cam'][i]
    T_cam_world = np.linalg.inv(T_world_cam)
    img = read_frame(head['dir'].joinpath('raw_video.mp4'), track['time'][i])

    labels = [f"head t={track['time'][i]:.2f}s, yellow: down to the table z=0"]
    wrist_imgs = list()
    for r in robots:
        name = names[r]
        tcp = tcp_calibration[name]
        T_cube_wrist = np.array(tcp['T_cube_wrist'])
        T_wrist_tcp = np.array(tcp['T_wrist_tcp'])
        T_wrist_compare = None
        if compare_tcp:
            T_wrist_compare = T_wrist_tcp.copy()
            T_wrist_compare[1:3, 3] = compare_tcp
        T_world_cube = T_world_cam @ track['bodies'][name]['T_cam_body'][i]
        tcp_world = (T_world_cube @ T_cube_wrist @ T_wrist_tcp)[:3, 3]
        px, _ = project(camera, T_cam_world, [tcp_world, tcp_world * np.array([1, 1, 0])])
        px = np.round(px).astype(int)
        origin = T_world_cube[:3, 3]
        axes, _ = project(camera, T_cam_world, [origin] + [origin + 0.05 * T_world_cube[:3, k] for k in range(3)])
        axes = np.round(axes).astype(int)
        for end, color in zip(axes[1:], [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            cv2.line(img, tuple(axes[0]), tuple(end), color, 4)
        cv2.line(img, tuple(px[0]), tuple(px[1]), (0, 255, 255), 3)
        cv2.circle(img, tuple(px[0]), 12, (0, 255, 0), -1)
        if r == robots[0]:
            labels.append(f"green TCP {T_wrist_tcp[2, 3] * 1000:.1f} mm fwd / {T_wrist_tcp[1, 3] * 1000:.1f} mm below camera"
                + (f", magenta {compare_tcp[1] * 1000:.1f} / {compare_tcp[0] * 1000:.1f} mm" if compare_tcp else ''))
        label = f"robot{r} {name}: green z={tcp_world[2] * 100:.1f} cm"
        if T_wrist_compare is not None:
            compare_world = (T_world_cube @ T_cube_wrist @ T_wrist_compare)[:3, 3]
            cpx, _ = project(camera, T_cam_world, [compare_world])
            cv2.circle(img, tuple(np.round(cpx[0]).astype(int)), 12, (255, 0, 255), -1)
            label += f", magenta z={compare_world[2] * 100:.1f} cm"
        labels.append(label)
        print(label)
        wrist_imgs.append(wrist_view(config, session, head, track['time'][i], name, T_wrist_tcp, T_wrist_compare))
    put_labels(img, labels)

    # head on top, wrist views below side by side (robot order)
    h, w = img.shape[:2]
    tile_w = w // len(wrist_imgs)
    tiles = [cv2.resize(x, (tile_w, int(round(x.shape[0] * tile_w / x.shape[1])))) for x in wrist_imgs]
    tile_h = max(x.shape[0] for x in tiles)
    bottom = np.zeros((tile_h, w, 3), dtype=img.dtype)
    for k, x in enumerate(tiles):
        bottom[:x.shape[0], k * tile_w:k * tile_w + x.shape[1]] = x
    combined = np.concatenate([img, bottom], axis=0)
    tag = 'robots' + ''.join(str(r) for r in robots)
    output = session.joinpath('tag_tracking', f"tcp_check_{tag}_{track['time'][i]:.1f}s.png")
    cv2.imwrite(str(output), combined)
    print(f"saved {output}")

# %%
if __name__ == "__main__":
    main()
