"""
Render tracked body poses onto the head camera video for visual checking.
Each body's frame axes are projected (x red, y green, z blue), labeled with
the body name; the label is green if the pose is valid (uncertainty within
the configured accuracy), red if estimated but too uncertain. Detected tag corners are drawn as dots. Gripper TCP frames are drawn if
05_calibrate_gripper_tcp.py has been run. The episode state (EPISODE k,
BANG, IDLE, REJECTED) is written top left if 00b_segment_episodes.py has been run.
Writes tag_tracking_render.mp4 next to tag_tracking.pkl.

python scripts_tag_tracking/04_render_tracking.py -i data_workspace/session -c umi_calib/tag_tracking.yaml
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
from tqdm import tqdm
from umi.tag_tracking.config import load_tag_tracking_config
from umi.tag_tracking.initialization import transform_points
from umi.tag_tracking.episodes import frame_labels


# %%
def draw_body(img, camera, T_cam_body, axis_length, label, is_consistent, scale):
    points = transform_points(T_cam_body, np.array([
        [0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]))
    if np.any(points[:, 2] <= 0):
        return
    pixels = np.round(camera.project(points) * scale).astype(int)
    for end, color in zip(pixels[1:], [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        cv2.line(img, tuple(pixels[0]), tuple(end), color, 3)
    text_color = (0, 255, 0) if is_consistent else (0, 0, 255)
    cv2.putText(img, label, tuple(pixels[0] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Session directory')
@click.option('-c', '--config', required=True, help='Tag tracking config yaml')
@click.option('-s', '--scale', type=float, default=0.5, help='Output video scale')
def main(input, config, scale):
    session = pathlib.Path(os.path.expanduser(input)).absolute()
    config = load_tag_tracking_config(config)
    camera = config['cameras'][config['head_camera']]
    calibration = json.load(session.joinpath('tag_tracking', 'bodies.json').open('r'))
    axis_lengths = {name: max(config['tag_sizes'][int(k)] for k in body['T_body_tags'])
        for name, body in calibration['bodies'].items()}
    # gripper TCP frames, if calibrated by 05_calibrate_gripper_tcp.py
    tcp_path = session.joinpath('tag_tracking', 'gripper_tcp.json')
    T_cube_tcps = dict()
    if tcp_path.is_file():
        T_cube_tcps = {name: np.array(x['T_cube_tcp']) for name, x in json.load(tcp_path.open('r')).items()}
    tcp_axis_length = min(config['tag_sizes'].values())
    # episode state, if segmented by 00b_segment_episodes.py
    episodes_path = session.joinpath('tag_tracking', 'episodes.json')
    episodes = json.load(episodes_path.open('r'))['videos'] if episodes_path.is_file() else dict()
    label_colors = {'EPISODE': (0, 200, 0), 'BANG': (0, 255, 255), 'IDLE': (200, 200, 200), 'REJECTED': (0, 0, 255)}

    for track_path in sorted(session.glob('demos/demo_*/tag_tracking.pkl')):
        track = pickle.load(track_path.open('rb'))
        corners = pickle.load(track_path.parent.joinpath('tag_corners.pkl').open('rb'))
        labels = None
        if track_path.parent.name in episodes:
            labels = frame_labels(track['time'], episodes[track_path.parent.name])
        output = track_path.parent.joinpath('tag_tracking_render.mp4')
        with av.open(str(track_path.parent.joinpath('raw_video.mp4'))) as in_container, \
                av.open(str(output), 'w') as out_container:
            in_stream = in_container.streams.video[0]
            in_stream.thread_type = 'AUTO'
            out_stream = out_container.add_stream('h264', rate=in_stream.average_rate)
            out_stream.width = int(in_stream.codec_context.width * scale) // 2 * 2
            out_stream.height = int(in_stream.codec_context.height * scale) // 2 * 2
            out_stream.pix_fmt = 'yuv420p'
            for i, frame in tqdm(enumerate(in_container.decode(in_stream)),
                    total=in_stream.frames, desc=track_path.parent.name):
                img = cv2.resize(frame.to_ndarray(format='bgr24'),
                    (out_stream.width, out_stream.height))
                for tag_corners in corners[i]['corners'].values():
                    for pixel in np.round(tag_corners * scale).astype(int):
                        cv2.circle(img, tuple(pixel), 3, (0, 255, 255), -1)
                for name, body_track in track['bodies'].items():
                    T = body_track['T_cam_body'][i]
                    if not np.isnan(T[0, 0]):
                        draw_body(img, camera, T, axis_lengths[name], name,
                            body_track['is_valid'][i], scale)
                        if name in T_cube_tcps:
                            draw_body(img, camera, T @ T_cube_tcps[name], tcp_axis_length,
                                'tcp', body_track['is_valid'][i], scale)
                if labels is not None and labels[i]:
                    color = label_colors[labels[i].split(' ')[0]]
                    cv2.putText(img, labels[i], (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
                    cv2.putText(img, labels[i], (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                out_frame = av.VideoFrame.from_ndarray(img, format='bgr24')
                for packet in out_stream.encode(out_frame):
                    out_container.mux(packet)
            for packet in out_stream.encode():
                out_container.mux(packet)
        print(f"Saved to {output}")

# %%
if __name__ == "__main__":
    main()
