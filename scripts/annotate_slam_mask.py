"""
Annotate the gripper outline on wrist camera frames and save it as a
SLAM mask image (255 = ignored), replacing the predefined polygons in
umi.common.cv_util for grippers or lenses that differ from the UMI default.

1. Frame selection: 'n'/'p' next/previous frame, ']'/'[' +-10 frames,
   'a' add the frame. Add frames with fingers fully open and fully closed.
   Enter when done.
2. Annotation, polygons are shared by all selected frames:
   click: add vertex, 'c': close polygon and start a new one,
   'u': undo vertex (reopens the last polygon), 'd': delete last polygon,
   'f': next frame, 'm': toggle filled mask view,
   's': save (window stays open), 'q': quit.

The mask is the convex hull of all polygons, see render_mask.
Polygons are saved next to the mask as json (pixel coordinates).
If that json exists, its polygons are loaded for further editing.
--polygons_json re-renders the mask from a json without the GUI.

python scripts/annotate_slam_mask.py -i sample_003/left/calibration_gripper.MP4 -o umi_calib/slam_mask.png
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
import click
import av
import cv2
import numpy as np
from tqdm import tqdm

WINDOW = 'annotate_slam_mask'
# plain window: Qt backend otherwise adds a toolbar and a right-click menu
WINDOW_FLAGS = cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL


# %%
def read_frames(video_path, n_frames, display_scale):
    """
    n_frames evenly spaced BGR frames at display scale, and full resolution (w, h).
    """
    with av.open(os.path.expanduser(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        image_size = (stream.codec_context.width, stream.codec_context.height)
        sample_idxs = set(np.linspace(0, stream.frames - 1, n_frames).round().astype(int).tolist())
        frames = list()
        for i, frame in tqdm(enumerate(container.decode(stream)), total=stream.frames,
                desc=f"Loading {pathlib.Path(video_path).name}"):
            if i in sample_idxs:
                frames.append(cv2.resize(frame.to_ndarray(format='bgr24'), None,
                    fx=display_scale, fy=display_scale))
    return frames, image_size

def put_lines(img, lines):
    for i, line in enumerate(lines):
        cv2.putText(img, line, (10, 30 + 30 * i),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def is_window_closed():
    return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1

def select_frames(frames):
    selected = list()
    idx = 0
    cv2.namedWindow(WINDOW, WINDOW_FLAGS)
    while True:
        img = frames[idx].copy()
        put_lines(img, [
            f"frame {idx + 1}/{len(frames)}  'n'/'p' next/prev, ']'/'[' +-10",
            f"'a' add frame ({len(selected)} added), Enter: done"])
        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(30) & 0xFF
        steps = {ord('n'): 1, ord('p'): -1, ord(']'): 10, ord('['): -10}
        if key in steps:
            idx = int(np.clip(idx + steps[key], 0, len(frames) - 1))
        elif key == ord('a') and idx not in selected:
            selected.append(idx)
        elif key in (13, 10) and len(selected) > 0:
            break
        elif is_window_closed():
            return list()
    cv2.destroyWindow(WINDOW)
    return [frames[i] for i in selected]

def render_mask(polygons, image_size):
    """
    Convex hull of all polygons.
    gopro_slam paints masked pixels black, so the mask outline itself becomes
    a static edge in every frame. Each corner of a detailed outline yields
    ORB features that move with the camera and break SLAM initialization;
    the convex hull keeps the outline smooth with few corners.
    """
    w, h = image_size
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(polygons) > 0:
        points = np.concatenate([np.array(x) for x in polygons]).round().astype(np.int32)
        cv2.fillPoly(mask, [cv2.convexHull(points)], color=255)
    return mask

def annotate_polygons(frames, polygons, save_fn):
    """
    Edit polygons in frame pixel coordinates, shared by all frames.
    save_fn(polygons) is called on 's'.
    """
    polygons = [list(x) for x in polygons]
    current = list()
    frame_idx = 0
    show_fill = False
    is_saved = True

    def on_mouse(event, x, y, flags, param):
        nonlocal is_saved
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))
            is_saved = False

    cv2.namedWindow(WINDOW, WINDOW_FLAGS)
    cv2.setMouseCallback(WINDOW, on_mouse)
    h, w = frames[0].shape[:2]
    while True:
        img = frames[frame_idx].copy()
        if show_fill:
            mask = render_mask(polygons, (w, h)) > 0
            img[mask] = (img[mask] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
        for polygon in polygons:
            pts = np.round(np.array(polygon)).astype(np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
        if len(current) > 0:
            pts = np.round(np.array(current)).astype(np.int32)
            cv2.polylines(img, [pts], isClosed=False, color=(0, 255, 0), thickness=2)
            for pt in pts:
                cv2.circle(img, tuple(pt), 3, (0, 255, 0), -1)
        put_lines(img, [
            f"click: add vertex, 'c': close polygon ({len(polygons)} polygons)",
            "'u': undo vertex, 'd': delete last polygon, 'm': toggle fill",
            f"'f': next frame ({frame_idx + 1}/{len(frames)}), "
            f"'s': save{'' if is_saved else ' (unsaved)'}, 'q': quit"])
        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('f'):
            frame_idx = (frame_idx + 1) % len(frames)
        elif key == ord('m'):
            show_fill = not show_fill
        elif key == ord('c') and len(current) >= 3:
            polygons.append(list(current))
            current.clear()
        elif key == ord('u'):
            if len(current) > 0:
                current.pop()
            elif len(polygons) > 0:
                current.extend(polygons.pop())
            is_saved = False
        elif key == ord('d') and len(polygons) > 0:
            polygons.pop()
            is_saved = False
        elif key == ord('s'):
            if len(current) >= 3:
                polygons.append(list(current))
                current.clear()
            save_fn(polygons)
            is_saved = True
        elif key == ord('q') or is_window_closed():
            break
    cv2.destroyAllWindows()
    if not is_saved:
        print("Quit with unsaved changes.")


# %%
@click.command()
@click.option('-i', '--input', multiple=True, help='Gripper calibration video(s), fingers fully opening and closing')
@click.option('-o', '--output', required=True, help='Output mask png, 255 = masked')
@click.option('-pj', '--polygons_json', default=None, help='Render mask from saved polygons json, without GUI')
@click.option('-n', '--n_frames', type=int, default=200, help='Frames sampled per video for selection')
@click.option('-ds', '--display_scale', type=float, default=0.4, help='Window size relative to video, 0.4 fits a 1080p screen')
def main(input, output, polygons_json, n_frames, display_scale):
    output = pathlib.Path(os.path.expanduser(output)).absolute()
    json_path = output.with_suffix('.json')

    def save_mask(polygons, image_size):
        mask = render_mask(polygons, image_size)
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), mask)
        print(f"Mask covers {(mask > 0).mean() * 100:.1f}% of image, saved to {output}")

    if polygons_json is not None:
        data = json.load(open(os.path.expanduser(polygons_json), 'r'))
        save_mask(data['polygons'], tuple(data['image_size']))
        return

    assert len(input) > 0, "--input or --polygons_json required"
    frames = list()
    image_size = None
    for path in input:
        video_frames, size = read_frames(path, n_frames, display_scale)
        assert image_size is None or image_size == size
        image_size = size
        frames.extend(video_frames)

    # continue editing previously saved polygons
    polygons = list()
    if json_path.is_file():
        polygons = [[(x * display_scale, y * display_scale) for x, y in polygon]
            for polygon in json.load(json_path.open('r'))['polygons']]
        print(f"Loaded {len(polygons)} polygons from {json_path}")

    def save_fn(display_polygons):
        # display to full resolution pixel coordinates
        polygons = [[(x / display_scale, y / display_scale) for x, y in polygon]
            for polygon in display_polygons]
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump({
            'image_size': list(image_size),
            'source_videos': [str(pathlib.Path(os.path.expanduser(x)).absolute()) for x in input],
            'polygons': polygons
        }, json_path.open('w'), indent=2)
        print(f"Saved {len(polygons)} polygons to {json_path}")
        save_mask(polygons, image_size)

    selected = select_frames(frames)
    if len(selected) == 0:
        print("No frames selected.")
        exit(1)
    annotate_polygons(selected, polygons, save_fn)

# %%
if __name__ == "__main__":
    main()
