"""
Render each exported episode (dataset_plan.pkl) as an MP4 dashboard in the
style of a Rerun viewer: the three synchronized camera views, time series
with a playhead and a log of the episode's events.

  head camera      the gripper TCPs (right orange, left blue) and their last
                   second of path projected with the tracked head pose
  wrist cameras    right and left, same instant
  time series      gripper width (mm), TCP height above the table (m), TCP
                   speed (m/s), wrist gyro magnitude (rad/s), per gripper
  log              episode start / end, camera sync (frame of the first step
                   and clock offset per camera), gripper close / open with the
                   resulting width, bangs

One output frame every --step steps of the plan (default 2: 29.97 fps from the
59.94 fps head video). Writes <output>/episode_<k>.mp4 (default
<session>/dataset_render).

python scripts_tag_tracking/render_episodes.py -i data_workspace/session -c umi_calib/tag_tracking.yaml
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import concurrent.futures
import json
import multiprocessing
import pathlib
import pickle
import subprocess
import click
import av
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from umi.tag_tracking.config import load_tag_tracking_config
from umi.tag_tracking.imu import load_imu
from umi.tag_tracking.annotation import gripper_events

W, H = 1920, 1080
HEAD = (1152, 864)                       # head view, top left
WRIST = (384, 288)                       # each wrist view, top right
LOG_H = H - HEAD[1]                      # log panel under the head view
PLOT = (W - HEAD[0], H - WRIST[1])       # time series under the wrist views
BG = (24, 24, 27)
PANEL = (32, 33, 37)
GREY = (150, 150, 155)
TEXT = (225, 225, 228)
SIDES = ('right', 'left')
COLORS = {'right': (255, 150, 40), 'left': (70, 160, 255)}   # RGB
FONT = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', 'DejaVuSansMono.ttf')


def rgb(color):
    return tuple(c / 255 for c in color)


def mmss(t):
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def frames(path, start, count, size, step):
    """
    RGB frames start, start + step, ... (count of them, video frame indices), scaled to size.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        fps = float(stream.average_rate)
        container.seek(int(max(0.0, (start - 2) / fps - 1.0) / stream.time_base), stream=stream)
        wanted, last = start, None
        for frame in container.decode(stream):
            index = int(round(float(frame.pts * stream.time_base) * fps))
            if index < wanted:
                continue
            while index > wanted and last is not None:   # missing frame: repeat (none expected in GoPro files)
                yield last
                wanted += step
                count -= 1
                if count == 0:
                    return
            last = frame.reformat(width=size[0], height=size[1], format='rgb24').to_ndarray()
            yield last
            wanted += step
            count -= 1
            if count == 0:
                return


def plot_background(t, series, duration):
    """
    The time series panel as an image and the pixel x of t = 0 and t = duration,
    and per subplot its pixel y range (top, bottom).
    """
    dpi = 100
    fig, axes = plt.subplots(len(series), 1, figsize=(PLOT[0] / dpi, PLOT[1] / dpi), dpi=dpi, sharex=True)
    fig.patch.set_facecolor(rgb(PANEL))
    for ax, (title, unit, lines) in zip(axes, series):
        ax.set_facecolor(rgb(BG))
        for side, (x, y) in lines.items():
            ax.plot(x, y, color=rgb(COLORS[side]), linewidth=1.2, label=side)
        ax.set_title(f"{title}  [{unit}]", color=rgb(GREY), fontsize=9, loc='left', family='monospace')
        ax.tick_params(colors=rgb(GREY), labelsize=8)
        for spine in ax.spines.values():
            spine.set_color((0.3, 0.3, 0.32))
        ax.grid(color=(0.25, 0.25, 0.27), linewidth=0.5)
        ax.set_xlim(0, duration)
    axes[0].legend(loc='upper right', fontsize=8, facecolor=rgb(PANEL), edgecolor='none',
        labelcolor=rgb(TEXT))
    axes[-1].set_xlabel('s', color=rgb(GREY), fontsize=8)
    fig.tight_layout(pad=0.6, h_pad=0.8)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    x0 = axes[0].transData.transform((0, 0))[0]
    x1 = axes[0].transData.transform((duration, 0))[0]
    ys = [(PLOT[1] - ax.get_window_extent().y1, PLOT[1] - ax.get_window_extent().y0) for ax in axes]
    plt.close(fig)
    return image, (x0, x1), ys


def episode_data(session, config, plan_episode, episodes_json):
    times = np.asarray(plan_episode['episode_timestamps'], dtype=float)
    t = times - times[0]
    cameras = plan_episode['cameras']
    out = {'t': t, 'duration': float(t[-1]), 'cameras': cameras, 'grippers': list(), 'events': list()}
    head_cam = cameras[-1]
    head_name = pathlib.Path(head_cam['video_path']).parent.name
    for g, gripper in enumerate(plan_episode['grippers']):
        side = SIDES[g]
        pos = np.asarray(gripper['tcp_pose'])[:, :3]
        width = np.asarray(gripper['gripper_width'], dtype=float).reshape(-1)
        speed = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1) / np.maximum(np.diff(t), 1e-6)]
        camera = cameras[g]
        imu = load_imu(str(session.joinpath('demos', camera['video_path']).parent.joinpath('imu_data.json')))
        imu_t = camera['frame0_time'] + imu['time'] + (camera.get('imu_time_offset') or 0.0) - times[0]
        inside = (imu_t >= 0) & (imu_t <= t[-1])
        out['grippers'].append({'side': side, 'pos': pos, 'width': width, 'speed': speed,
            'imu_t': imu_t[inside], 'gyro': np.linalg.norm(imu['gyro'][inside], axis=1)})
        for e in gripper_events(t, width, 0.0, t[-1] + 1e-6):
            note = f"{e['width'] * 1000:.0f} mm" + (" (holding)" if e.get('holding') else '')
            out['events'].append((e['t'], side, f"gripper {e['kind']} -> {note}"))
    # bangs of this head video inside the episode
    bangs = episodes_json['videos'].get(head_name, dict()).get('bangs', [])
    for b in bangs:
        tb = head_cam['frame0_time'] + b - times[0]
        if 0 <= tb <= t[-1]:
            out['events'].append((tb, 'both', 'bang'))
    roles = [f"{SIDES[c]}_wrist" for c in range(len(plan_episode['grippers']))] + ['head']
    start_lines = [(0.0, 'episode', f"start, {t[-1]:.1f} s, {len(t)} steps")]
    for c, (camera, role) in enumerate(zip(cameras, roles)):
        serial = pathlib.Path(camera['video_path']).parent.name.split('_')[1]
        offset = camera.get('clock_correction_ms')
        start_lines.append((0.0, role, f"{serial} sync frame {camera['video_start_end'][0]}"
            + (f", offset {offset:+.2f} ms" if offset is not None else '')))
    out['events'] = start_lines + sorted(out['events']) + [(float(t[-1]), 'episode', 'end')]
    out['roles'] = roles
    track = pickle.load(session.joinpath('demos', head_name, 'tag_tracking.pkl').open('rb'))
    out['head_T_world_cam'] = track['head']['T_world_cam']
    out['head_camera'] = config['cameras'][config['head_camera']]
    return out


def draw_text(draw, xy, text, fill, font):
    draw.text(xy, text, fill=fill, font=font)


def render_episode(session, config_path, plan_episode, episodes_json, path, step, title):
    config = load_tag_tracking_config(config_path)
    d = episode_data(session, config, plan_episode, episodes_json)
    t, duration = d['t'], d['duration']
    series = [
        ('/robot/gripper/width', 'mm', {g['side']: (t, g['width'] * 1000) for g in d['grippers']}),
        ('/robot/eef/height', 'm', {g['side']: (t, g['pos'][:, 2]) for g in d['grippers']}),
        ('/robot/eef/speed', 'm/s', {g['side']: (t, g['speed']) for g in d['grippers']}),
        ('/robot/imu/gyro', 'rad/s', {g['side']: (g['imu_t'], g['gyro']) for g in d['grippers']}),
    ]
    plot_bg, (x0, x1), _ = plot_background(t, series, duration)
    font = ImageFont.truetype(FONT, 15)
    small = ImageFont.truetype(FONT, 13)
    n_out = (len(t) + step - 1) // step
    cams = d['cameras']
    head_start = cams[-1]['video_start_end'][0]
    streams = [frames(session.joinpath('demos', c['video_path']), c['video_start_end'][0], n_out,
        HEAD if k == len(cams) - 1 else WRIST, step) for k, c in enumerate(cams)]
    camera = d['head_camera']
    sx, sy = HEAD[0] / camera.image_size[0], HEAD[1] / camera.image_size[1]
    ffmpeg = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
        '-r', f'{59.94006 / step:.5f}', '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
    trail = int(round(1.0 * 59.94))
    for n in range(n_out):
        i = n * step
        canvas = Image.new('RGB', (W, H), BG)
        images = [next(s) for s in streams]
        # head view with the TCPs and their last second
        head = Image.fromarray(images[-1])
        hd = ImageDraw.Draw(head)
        T = d['head_T_world_cam'][head_start + i]
        if not np.isnan(T[0, 0]):
            T_cam_world = np.linalg.inv(T)
            for g in d['grippers']:
                p = g['pos'][max(0, i - trail):i + 1]
                p = p[~np.isnan(p[:, 0])]
                if len(p) == 0:
                    continue
                pc = (T_cam_world[:3, :3] @ p.T).T + T_cam_world[:3, 3]
                pc = pc[pc[:, 2] > 0.05]
                if len(pc) == 0:
                    continue
                px = camera.project(pc) * np.array([sx, sy])
                hd.line([tuple(x) for x in px], fill=COLORS[g['side']], width=3)
                cx, cy = px[-1]
                hd.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], outline=(255, 255, 255), width=2, fill=COLORS[g['side']])
        canvas.paste(head, (0, 0))
        for k in range(len(cams) - 1):
            canvas.paste(Image.fromarray(images[k]), (HEAD[0] + k * WRIST[0], 0))
        canvas.paste(Image.fromarray(plot_bg), (HEAD[0], WRIST[1]))
        draw = ImageDraw.Draw(canvas)
        # labels in the style of the viewer's space views
        draw_text(draw, (8, 6), f"/cameras/head   {title}", TEXT, small)
        for k in range(len(cams) - 1):
            draw_text(draw, (HEAD[0] + k * WRIST[0] + 6, 6), f"/cameras/{d['roles'][k]}", TEXT, small)
        # playhead on the time series
        x = HEAD[0] + x0 + (x1 - x0) * t[i] / max(duration, 1e-6)
        draw.line([(x, WRIST[1] + 4), (x, H - 4)], fill=(235, 235, 240), width=1)
        # current values next to the playhead
        values = '   '.join(f"{g['side']} {g['width'][i] * 1000:5.1f} mm" for g in d['grippers'])
        draw_text(draw, (HEAD[0] + 8, WRIST[1] + 2), values, TEXT, small)
        # log panel
        draw.rectangle([0, HEAD[1], HEAD[0], H], fill=PANEL)
        draw_text(draw, (10, HEAD[1] + 6), f"log   t = {mmss(t[i])} / {mmss(duration)}", GREY, small)
        shown = [e for e in d['events'] if e[0] <= t[i] + 1e-9][-9:]
        for row, (te, entity, message) in enumerate(shown):
            color = TEXT if te > t[i] - 1.0 else GREY
            draw_text(draw, (10, HEAD[1] + 28 + row * 20), f"{mmss(te):>7}  {entity:<12} {message}", color, font)
        ffmpeg.stdin.write(np.asarray(canvas).tobytes())
    ffmpeg.stdin.close()
    assert ffmpeg.wait() == 0, f"ffmpeg failed for {path}"
    return path


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Session directory with dataset_plan.pkl')
@click.option('-c', '--config', required=True, help='Tag tracking config yaml')
@click.option('-o', '--output', default=None, help='Output directory, default <session>/dataset_render')
@click.option('-e', '--episodes', default=None, help='Comma separated episode indices of the plan, default all')
@click.option('--step', type=int, default=2, help='Plan steps per output frame (2: 29.97 fps)')
@click.option('-j', '--num_workers', type=int, default=4, help='Episodes rendered in parallel')
def main(input, config, output, episodes, step, num_workers):
    session = pathlib.Path(os.path.expanduser(input)).absolute()
    output = session.joinpath('dataset_render') if output is None else pathlib.Path(os.path.expanduser(output)).absolute()
    output.mkdir(parents=True, exist_ok=True)
    plan = pickle.load(session.joinpath('dataset_plan.pkl').open('rb'))
    episodes_json = json.load(session.joinpath('tag_tracking', 'episodes.json').open('r'))
    indices = list(range(len(plan))) if episodes is None else [int(x) for x in episodes.split(',')]
    print(f"{len(plan)} episodes in the plan, rendering {len(indices)} to {output}")
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(num_workers, len(indices)),
            mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = {executor.submit(render_episode, session, os.path.abspath(config), plan[k], episodes_json,
            output.joinpath(f'episode_{k:06d}.mp4'), step, f"{session.name} episode {k}"): k for k in indices}
        for future in concurrent.futures.as_completed(futures):
            path = future.result()
            print(f"{pathlib.Path(path).name}: {os.path.getsize(path) / 1e6:.0f} MB")

# %%
if __name__ == "__main__":
    main()
