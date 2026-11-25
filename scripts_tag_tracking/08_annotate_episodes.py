"""
Draft annotations.json of every episode (Odyssey spec 1.7 / 2.7): times and
arms measured from the data (umi.tag_tracking.annotation), the words from a
visual check by Claude.

Measured per dock-to-dock episode (tag_tracking/episodes.json, task interval
from 03t_trim_docking.py):
- gripper close / open events per arm from the finger-tag width; a close that
  stops open holds something, one that closes fully holds nothing;
- per arm, spans covering the episode: 'reset' outside the task (docking,
  bang ritual), 'approach' / 'hold' / 'retract' between the events.
Visual check: head camera frames at the task start, every gripper event and
the task end (plus evenly spaced ones up to --max_frames), and each arm's wrist
camera frame at its events, each labeled with its time. Claude answers as
structured JSON: task sentence, skill, objects, success predicate, outcome,
instruction variants; verb / object / destination per measured span; per
gripper event whether it grasped / released an object; state changes,
recoveries, issues, objects, scene graph. Claude does not set any time of a
span or event.

Writes per episode, in <output> (default <session>/tag_tracking/annotations):
<head video>_ep<k>.json (spec 2.7 layers, annotation_source 'model', times in
seconds from the episode's first head frame; episode_id is set by the delivery
export) and <head video>_ep<k>_frames.jpg (the frames Claude saw, for review).
Episodes already annotated are skipped unless --overwrite.
Labels are per episode; 09_normalize_labels.py then gives the same thing the
same name across episodes.
--dry_run writes the frame sheet and the request text, no API call.

Needs the anthropic package and ANTHROPIC_API_KEY.

python scripts_tag_tracking/08_annotate_episodes.py -i data_workspace/session -c umi_calib/tag_tracking.yaml
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import base64
import datetime
import json
import pathlib
import pickle
import click
import av
import cv2
import numpy as np
from umi.tag_tracking.config import load_tag_tracking_config
from umi.tag_tracking.session import get_session_videos, overlapping
from umi.tag_tracking.wrist_sync import wrist_corrections, head_to_wrist_time
from umi.tag_tracking.delivery import gripper_streams
from umi.tag_tracking.annotation import (gripper_events, measured_items, frame_times_for_review,
    response_schema, request_text, merge_annotation)

SCHEMA_VERSION = '0.6'
IMAGE_WIDTH = 1024
JPEG_QUALITY = 85

SYSTEM = """You annotate manipulation demonstrations recorded with two handheld grippers: a person holds \
one gripper in each hand. Each gripper has a fisheye wrist camera looking along its fingers, and the person \
wears a head camera looking down at the workspace. "right" and "left" are the person's hands.

You get frames of one episode, each labeled with its camera and time (seconds from the episode start), and \
the gripper events and spans measured from the finger opening. The measured times and arms are fixed. Your \
job is to say what the frames show: the task, the objects, what each arm does in each measured span, whether \
each gripper event grasped or released an object, and whether the task succeeded.

Rules:
- Describe only what the frames show. If something cannot be seen, write "unknown" or leave it empty and \
lower the confidence; do not guess an object or an outcome.
- Verbs are short phrases naming the object and destination: "grasp blue screwdriver", \
"carry cup to saucer", "release screwdriver into tray".
- Use the same object names everywhere in the answer.
- destination is where the object ends up or goes (a place or another object); leave it empty when nothing \
is carried, e.g. while approaching.
- outcome is "success" only if the frames show the goal reached; completed_at_s is then the time it was \
reached (use the label time of the first frame showing it).
- progress is how far the whole task is done at the end of the span, from 0 (not started) to 1 (done); \
confidence is from 0 to 1.
- main_object is the object the task is about (one of objects). object_class of each object: rigid, \
deformable (cloth, bag, cable), articulated (lid on a hinge, drawer, box flaps), granular, liquid, or tool \
(an object used to act on another, like a screwdriver used to drive a screw; a screwdriver only moved is rigid).
- difficulty: single_step (one grasp and place), multi_step (a few dependent steps), long_horizon (many \
steps or subgoals).
- failure_kind of a recovery: missed_grasp, drop, slip, collision, wrong_object, timeout or other."""


def read_frames(video_path, times):
    """
    BGR frames at the given video times (s): the first frame whose time is
    within half a frame of it, None if there is none.
    """
    out = [None] * len(times)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        half = 0.5 / float(stream.average_rate)
        for k in np.argsort(times):
            t = times[k]
            container.seek(int(max(0.0, t - 1.0) / stream.time_base), stream=stream)
            for frame in container.decode(stream):
                ft = float(frame.pts * stream.time_base)
                if ft >= t - half:
                    if ft <= t + half:
                        out[k] = frame.to_ndarray(format='bgr24')
                    break
    return out


def labeled(img, text):
    h, w = img.shape[:2]
    img = cv2.resize(img, (IMAGE_WIDTH, int(round(h * IMAGE_WIDTH / w))), interpolation=cv2.INTER_AREA)
    for color, thickness in (((0, 0, 0), 5), ((255, 255, 255), 2)):
        cv2.putText(img, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness)
    return img


def contact_sheet(images, columns=4):
    w = IMAGE_WIDTH // 2
    tiles = [cv2.resize(x, (w, int(round(x.shape[0] * w / x.shape[1]))), interpolation=cv2.INTER_AREA) for x in images]
    h = max(t.shape[0] for t in tiles)
    rows = list()
    for i in range(0, len(tiles), columns):
        row = [np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))) for t in tiles[i:i + columns]]
        row += [np.zeros((h, w, 3), dtype=np.uint8)] * (columns - len(row))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Session directory (trim_docking done)')
@click.option('-c', '--config', required=True, help='Tag tracking config yaml')
@click.option('-o', '--output', default=None, help='Default <input>/tag_tracking/annotations')
@click.option('-e', '--episodes', default=None, help='Comma separated episode numbers (1-based, as printed by 03t), default all')
@click.option('--task_prompt', default=None, help='The task as issued to the operator (prompt_mode given), else inferred')
@click.option('--model', default='claude-opus-5-5')
@click.option('--effort', type=click.Choice(['low', 'medium', 'high', 'xhigh', 'max']), default='high')
@click.option('--max_frames', type=int, default=12, help='Head camera frames per episode')
@click.option('--dry_run', is_flag=True, default=False, help='Frame sheet and request text only, no API call')
@click.option('--overwrite', is_flag=True, default=False, help='Redo episodes already annotated')
def main(input, config, output, episodes, task_prompt, model, effort, max_frames, dry_run, overwrite):
    session = pathlib.Path(os.path.expanduser(input)).absolute()
    config = load_tag_tracking_config(config)
    output = session.joinpath('tag_tracking', 'annotations') if output is None else pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tcp = json.load(session.joinpath('tag_tracking', 'gripper_tcp.json').open('r'))
    episodes_json = json.load(session.joinpath('tag_tracking', 'episodes.json').open('r'))['videos']
    sync_path = session.joinpath('tag_tracking', 'wrist_sync.json')
    sync = json.load(sync_path.open('r')) if sync_path.is_file() else None
    selected = None if episodes is None else {int(x) for x in episodes.split(',')}
    videos = get_session_videos(session)
    client = None
    if not dry_run:
        import anthropic
        client = anthropic.Anthropic()
    usage = {'input_tokens': 0, 'output_tokens': 0, 'episodes': 0}

    # head videos 00b_segment_episodes.py skipped (no wrist video overlapping) have no episodes
    for head in [x for x in videos if x['camera_serial'] == config['head_camera'] and x['name'] in episodes_json]:
        track = pickle.load(head['dir'].joinpath('tag_tracking.pkl').open('rb'))
        corrections = wrist_corrections(sync, head['name'])
        grippers = gripper_streams(session, config, videos, head, track, tcp, corrections)
        wrists = {g['side']: overlapping(videos, head, config['bodies'][g['name']]['gripper_camera']) for g in grippers}
        for k, e in enumerate(episodes_json[head['name']]['episodes']):
            number = k + 1
            if selected is not None and number not in selected:
                continue
            stem = f"{head['name']}_ep{number:03d}"
            if e.get('task_start') is None:
                print(f"{stem}: no task interval (run 03t_trim_docking.py), skipped")
                continue
            out_path = output.joinpath(stem + '.json')
            if out_path.is_file() and not overwrite and not dry_run:
                print(f"{stem}: already annotated, skipped")
                continue
            t0 = e['start']

            # measured
            arm_events = {g['side']: gripper_events(g['width_times'], g['widths'], e['start'], e['end']) for g in grippers}
            spans, events = measured_items(e['start'], e['end'], e['task_start'], e['task_end'], arm_events)
            bangs = [float(b) for b in episodes_json[head['name']]['bangs'] if e['start'] <= b < e['end']]

            # frames: head at the task start / end and every event, each wrist at its events
            picks = frame_times_for_review(e['task_start'], e['task_end'], arm_events, max_frames)
            images, labels = list(), list()
            for (t, reason), img in zip(picks, read_frames(head['dir'].joinpath('raw_video.mp4'), [t for t, _ in picks])):
                if img is not None:
                    labels.append(f"head camera, t={t - t0:.2f} s: {reason}")
                    images.append(labeled(img, labels[-1]))
            for g in grippers:
                arm = [x for x in events.values() if x['arm'] == g['side']]
                for wrist in wrists[g['side']]:
                    def to_wrist(t):
                        return float(head_to_wrist_time(t, head, wrist, corrections))
                    length = wrist['end_timestamp'] - wrist['start_timestamp']
                    inside = [x for x in arm if 0 <= to_wrist(x['t']) < length]
                    frames = read_frames(wrist['dir'].joinpath('raw_video.mp4'), [to_wrist(x['t']) for x in inside])
                    for x, img in zip(inside, frames):
                        if img is not None:
                            labels.append(f"{g['side']} wrist camera, t={x['t'] - t0:.2f} s: gripper {x['kind']}")
                            images.append(labeled(img, labels[-1]))
            if not images:
                print(f"{stem}: no frames read, skipped")
                continue
            cv2.imwrite(str(output.joinpath(stem + '_frames.jpg')), contact_sheet(images), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            text = request_text(e['start'], e['end'], e['task_start'], e['task_end'], spans, events, task_prompt)
            if dry_run:
                output.joinpath(stem + '_request.txt').write_text('\n\n'.join([SYSTEM, text, '\n'.join(labels)]) + '\n')
                print(f"{stem}: {len(images)} frames, {sum(s['phase'] != 'reset' for s in spans.values())} spans, "
                    f"{len(events)} gripper events (dry run)")
                continue

            # visual check
            content = [{'type': 'text', 'text': text}]
            for img, label in zip(images, labels):
                jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1]
                content.append({'type': 'text', 'text': label})
                content.append({'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg',
                    'data': base64.standard_b64encode(jpg.tobytes()).decode('ascii')}})
            schema = response_schema([i for i, s in spans.items() if s['phase'] != 'reset'], list(events))
            with client.beta.messages.stream(
                model=model,
                max_tokens=32000,
                betas=['server-side-fallback-2026-07-01'],
                fallbacks='default',
                thinking={'type': 'adaptive'},
                output_config={'effort': effort, 'format': {'type': 'json_schema', 'schema': schema}},
                system=SYSTEM,
                messages=[{'role': 'user', 'content': content}],
            ) as stream:
                response = stream.get_final_message()
            usage['input_tokens'] += response.usage.input_tokens
            usage['output_tokens'] += response.usage.output_tokens
            if response.stop_reason != 'end_turn':
                print(f"{stem}: not annotated, stop reason {response.stop_reason} {response.stop_details or ''}")
                continue
            answer = json.loads(next(b.text for b in response.content if b.type == 'text'))
            usage['episodes'] += 1

            annotation = {
                'schema_version': SCHEMA_VERSION,
                'episode_id': None,
                'source': {'head_video': head['name'], 'episode': number, 'start_head_video_s': e['start'],
                    'end_head_video_s': e['end'], 'task_boundary': e.get('task_boundary'),
                    'task_review': e.get('task_review', [])},
                'annotation_source': 'model',
                'annotation_model': response.model,
                'prompt_mode': 'given' if task_prompt else 'inferred',
                'given_prompt': task_prompt,
                **merge_annotation(answer, e['start'], e['task_start'], e['task_end'], spans, events, bangs, e['end']),
                'frames_seen': labels,
                # inputs of merge_annotation, to rebuild the layers after 09_normalize_labels.py
                'measured': {'start': e['start'], 'task_start': e['task_start'], 'task_end': e['task_end'],
                    'spans': spans, 'events': events, 'end': e['end'], 'bangs': bangs},
                'model_answer': answer,
                'created_ts_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            json.dump(annotation, out_path.open('w'), indent=2)
            task = annotation['tasks'][0]
            print(f"{stem}: {task['skill']}, {task['outcome']} ({task['confidence']:.2f}): {task['task']} "
                f"[{response.usage.input_tokens} in, {response.usage.output_tokens} out tokens]")

    if usage['episodes']:
        print(f"{usage['episodes']} episodes annotated, {usage['input_tokens']} input / {usage['output_tokens']} output tokens")
    print(f"Saved to {output}")

# %%
if __name__ == "__main__":
    main()
