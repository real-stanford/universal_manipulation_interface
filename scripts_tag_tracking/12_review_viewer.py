"""
Review viewer for one session, in the browser: per episode the head and both
wrist videos played in sync, the measured spans and gripper events, the
gripper opening and tracking validity, the model's annotations (08 / 09), the
quality checks (11), and the fields a person fills in:
- per episode (annotations/<episode>_review.json, umi.tag_tracking.review):
  reviewed, reviewer, task sentence and outcome corrections, span word
  corrections, faces visible, exclude + reason, notes;
- per session (with -d): the delivery declarations (collection mode, scene,
  environment / operator / station ids, consent reference) in the delivery
  yaml. The yaml is rewritten without its comments.

The source videos are HEVC, which browsers do not reliably play: the first
time an episode is opened, its clips are cut and converted to small H.264
files in tag_tracking/viewer/ (a few seconds per episode; --prepare does all
episodes up front).

python scripts_tag_tracking/12_review_viewer.py -i data_workspace/session -c umi_calib/tag_tracking.yaml -d delivery.yaml
then open http://127.0.0.1:8765
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import concurrent.futures
import datetime
import http.server
import json
import math
import pathlib
import pickle
import subprocess
import threading
import urllib.parse
import click
import numpy as np
import yaml
from umi.tag_tracking.config import load_tag_tracking_config
from umi.tag_tracking.session import get_session_videos, overlapping
from umi.tag_tracking.wrist_sync import wrist_corrections, head_readout, head_to_wrist_time
from umi.tag_tracking.delivery import episode_frames, gripper_streams
from umi.tag_tracking.annotation import gripper_events, measured_items, bang_key_events
from umi.tag_tracking.review import check_review, empty_review, OUTCOMES
from umi.tag_tracking.delivery_config import load_delivery_config, COLLECTION_MODES, SCENES, SESSION_KEYS

PAGE = pathlib.Path(os.path.abspath(__file__)).parent.joinpath('viewer', 'index.html')
WIDTH = {'head': 1280, 'wrist': 640}


def clean(x):
    """
    JSON-safe: NaN / inf -> None, numpy -> python.
    """
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


class Session:
    def __init__(self, session, config, delivery):
        self.session = session
        self.config = config
        self.delivery = delivery
        tt = session.joinpath('tag_tracking')
        self.tcp = json.load(tt.joinpath('gripper_tcp.json').open('r'))
        self.episodes = json.load(tt.joinpath('episodes.json').open('r'))['videos']
        sync_path = tt.joinpath('wrist_sync.json')
        self.sync = json.load(sync_path.open('r')) if sync_path.is_file() else None
        quality_path = tt.joinpath('quality.json')
        self.quality = json.load(quality_path.open('r'))['episodes'] if quality_path.is_file() else dict()
        self.annotations = tt.joinpath('annotations')
        self.proxies = tt.joinpath('viewer')
        self.proxies.mkdir(exist_ok=True)
        self.videos = get_session_videos(session)
        self.heads = {x['name']: x for x in self.videos if x['camera_serial'] == config['head_camera']}
        self.cache = dict()
        self.lock = threading.Lock()
        self.proxy_locks = dict()

    def head_data(self, name):
        with self.lock:
            if name not in self.cache:
                head = self.heads[name]
                track = pickle.load(head['dir'].joinpath('tag_tracking.pkl').open('rb'))
                corrections = wrist_corrections(self.sync, name)
                grippers = gripper_streams(self.session, self.config, self.videos, head, track, self.tcp, corrections)
                self.cache[name] = {'track': track, 'grippers': grippers, 'corrections': corrections,
                    'readout': head_readout(head)}
            return self.cache[name]

    def stem(self, head, number):
        return f"{head}_ep{number:03d}"

    def list(self):
        out = list()
        for name in self.heads:
            quality = {q['episode']: q['validation_status'] for q in self.quality.get(name, [])}
            for k, e in enumerate(self.episodes[name]['episodes']):
                if e.get('task_start') is None:
                    continue
                stem = self.stem(name, k + 1)
                review = self.read_review(stem)
                out.append({'head': name, 'number': k + 1, 'duration': e['end'] - e['start'],
                    'status': quality.get(k + 1), 'annotated': self.annotations.joinpath(stem + '.json').is_file(),
                    'reviewed': review['reviewed'], 'exclude': review['exclude']})
        return {'session': self.session.name, 'episodes': out}

    def read_review(self, stem):
        path = self.annotations.joinpath(stem + '_review.json')
        return check_review(json.load(path.open('r'))) if path.is_file() else empty_review()

    def write_review(self, head, number, review):
        assert head in self.heads and 0 < number <= len(self.episodes[head]['episodes'])
        review = check_review(review)
        review['reviewed_ts_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.annotations.mkdir(exist_ok=True)
        path = self.annotations.joinpath(self.stem(head, number) + '_review.json')
        tmp = path.with_suffix('.tmp')
        json.dump(review, tmp.open('w'), indent=2)
        tmp.replace(path)
        return review

    def proxy(self, source, name, t0, duration, width):
        """
        H.264 copy of source from t0 (s) for duration, width px wide; made once.
        """
        path = self.proxies.joinpath(name + '.mp4')
        with self.lock:
            lock = self.proxy_locks.setdefault(name, threading.Lock())
        with lock:
            if not path.is_file():
                tmp = path.with_suffix('.tmp.mp4')
                subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f"{t0:.4f}", '-i', str(source), '-t', f"{duration:.4f}",
                    '-vf', f"scale={width}:-2", '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '24', '-g', '30',
                    '-pix_fmt', 'yuv420p', '-an', '-movflags', '+faststart', str(tmp)], check=True)
                tmp.replace(path)
        return '/video/' + path.name

    def episode_videos(self, head_name, number, e, data):
        """
        Head clip and per arm the wrist clip; offset = episode time of the clip's first frame.
        """
        head = self.heads[head_name]
        stem = self.stem(head_name, number)
        duration = e['end'] - e['start']
        jobs = [('head', 'head camera', head, e['start'], 0.0, WIDTH['head'])]
        for g in data['grippers']:
            camera = self.config['bodies'][g['name']]['gripper_camera']
            found = None
            for wrist in overlapping(self.videos, head, camera):
                w0 = float(head_to_wrist_time(e['start'], head, wrist, data['corrections']))
                length = wrist['end_timestamp'] - wrist['start_timestamp']
                if w0 < length and w0 + duration > 0:
                    found = (g['side'], f"{g['side']} wrist", wrist, max(0.0, w0), max(0.0, -w0), WIDTH['wrist'])
                    break
            jobs.append(found or (g['side'], f"{g['side']} wrist", None, 0, 0, 0))
        with concurrent.futures.ThreadPoolExecutor(len(jobs)) as pool:
            futures = [pool.submit(self.proxy, video['dir'].joinpath('raw_video.mp4'), f"{stem}_{key}", t0,
                duration - offset, width) if video else None for key, _, video, t0, offset, width in jobs]
            urls = [f.result() if f else None for f in futures]
        return [{'label': label, 'url': url, 'offset': offset} for (_, label, _, _, offset, _), url in zip(jobs, urls)]

    def episode(self, head_name, number):
        e = self.episodes[head_name]['episodes'][number - 1]
        data = self.head_data(head_name)
        track, grippers = data['track'], data['grippers']
        start, end = e['start'], e['end']
        frames = episode_frames(track['time'], track['head']['T_world_cam'], track['head']['is_valid'], start, end,
            grippers, row_time_offset=data['readout'] / 2)['frames']
        signals = {'time': frames['time'] - start}
        for g in grippers:
            signals[g['side']] = {'width': frames[f"gripper_{g['side']}"] * 1000,
                'pose_valid': ~np.isnan(frames[f"ee_pose_{g['side']}"][:, 0])}
        signals['max_width_mm'] = max((g['gripper_range']['max_width'] - g['gripper_range']['min_width']) * 1000
            for g in grippers)
        arm_events = {g['side']: gripper_events(g['width_times'], g['widths'], start, end) for g in grippers}
        spans, events = measured_items(start, end, e['task_start'], e['task_end'], arm_events)
        measured = {
            'spans': [{'arm': s['arm'], 'phase': s['phase'], 'start_s': s['start'] - start, 'end_s': s['end'] - start}
                for s in spans.values()],
            'events': sorted([{'arm': x['arm'], 'kind': 'contact', 'measured_kind': x['kind'], 't_s': x['t'] - start,
                'label': f"gripper {x['kind']}" + (' (holding)' if x.get('holding') else ''), 'outcome': None}
                for x in events.values()], key=lambda x: x['t_s']),
            'bangs': bang_key_events(self.episodes[head_name]['bangs'], start, end, e['task_start'], e['task_end'])}
        stem = self.stem(head_name, number)
        path = self.annotations.joinpath(stem + '.json')
        annotation = json.load(path.open('r')) if path.is_file() else None
        if annotation is not None:
            annotation = {k: v for k, v in annotation.items() if k not in ('model_answer', 'measured')}
        quality = [q for q in self.quality.get(head_name, []) if q['episode'] == number]
        return clean({
            'head': head_name, 'number': number, 'stem': stem, 'duration': end - start, 'fps': float(self.heads[head_name]['fps']),
            'task_start': e['task_start'] - start, 'task_end': e['task_end'] - start,
            'task_boundary': e.get('task_boundary'), 'task_review': e.get('task_review', []),
            'arms': [g['side'] for g in grippers], 'signals': signals, 'measured': measured,
            'annotation': annotation, 'review': self.read_review(stem), 'outcomes': OUTCOMES,
            'quality': {'status': quality[0]['validation_status'],
                'checks': [c for c in quality[0]['checks'] if c['status'] in ('warn', 'fail')]} if quality else None,
            'videos': self.episode_videos(head_name, number, e, data),
        })

    def declarations(self):
        if self.delivery is None:
            return {'path': None}
        config = load_delivery_config(str(self.delivery)) if self.delivery.is_file() else {'sessions': dict()}
        current = config['sessions'].get(self.session.name) or {k: None for k in SESSION_KEYS}
        return {'path': str(self.delivery), 'session': self.session.name, 'declarations': current,
            'collection_modes': COLLECTION_MODES, 'scenes': SCENES}

    def write_declarations(self, values):
        assert self.delivery is not None, "no delivery yaml (-d)"
        raw = yaml.safe_load(self.delivery.open('r')) if self.delivery.is_file() else dict()
        raw = raw or dict()
        raw.setdefault('sessions', dict())
        raw['sessions'][self.session.name] = {k: values.get(k) for k in SESSION_KEYS}
        tmp = self.delivery.with_suffix('.tmp')
        yaml.safe_dump(raw, tmp.open('w'), sort_keys=False)
        load_delivery_config(str(tmp))  # validate before replacing
        tmp.replace(self.delivery)
        return self.declarations()


def handler_for(state: Session):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, body, content_type='application/json'):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def query(self):
            url = urllib.parse.urlparse(self.path)
            return url.path, {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}

        def do_GET(self):
            path, q = self.query()
            try:
                if path == '/':
                    self.send(200, PAGE.read_bytes(), 'text/html; charset=utf-8')
                elif path == '/api/episodes':
                    self.send(200, state.list())
                elif path == '/api/episode':
                    self.send(200, state.episode(q['head'], int(q['n'])))
                elif path == '/api/session':
                    self.send(200, state.declarations())
                elif path.startswith('/video/'):
                    self.video(state.proxies.joinpath(pathlib.Path(path).name))
                else:
                    self.send(404, {'error': 'not found'})
            except Exception as e:
                self.send(500, f"{type(e).__name__}: {e}".encode(), 'text/plain')

        def do_POST(self):
            path, q = self.query()
            try:
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if path == '/api/review':
                    self.send(200, state.write_review(q['head'], int(q['n']), body))
                elif path == '/api/session':
                    self.send(200, state.write_declarations(body))
                else:
                    self.send(404, {'error': 'not found'})
            except Exception as e:
                self.send(400, f"{type(e).__name__}: {e}".encode(), 'text/plain')

        def video(self, path):
            """
            The clip, with byte ranges (the browser seeks with them).
            """
            if not path.is_file():
                return self.send(404, {'error': 'no clip'})
            size = path.stat().st_size
            start, end = 0, size - 1
            ranged = self.headers.get('Range', '').startswith('bytes=')
            if ranged:
                a, b = self.headers['Range'][6:].split(',')[0].split('-')
                start = int(a) if a else max(0, size - int(b))
                end = min(int(b), size - 1) if a and b else size - 1
            self.send_response(206 if ranged else 200)
            self.send_header('Content-Type', 'video/mp4')
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(end - start + 1))
            if ranged:
                self.send_header('Content-Range', f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open('rb') as f:
                f.seek(start)
                remaining = end - start + 1
                try:
                    while remaining > 0:
                        chunk = f.read(min(1 << 20, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass
    return Handler


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Session directory (trim_docking done)')
@click.option('-c', '--config', required=True, help='Tag tracking config yaml')
@click.option('-d', '--delivery', default=None, help='Delivery declarations yaml to edit (created if missing)')
@click.option('-p', '--port', type=int, default=8765)
@click.option('--prepare', is_flag=True, default=False, help='Convert the clips of every episode first')
def main(input, config, delivery, port, prepare):
    session = pathlib.Path(os.path.expanduser(input)).absolute()
    state = Session(session, load_tag_tracking_config(config),
        pathlib.Path(os.path.expanduser(delivery)).absolute() if delivery else None)
    if prepare:
        listed = state.list()['episodes']
        for k, e in enumerate(listed):
            state.episode(e['head'], e['number'])
            print(f"clips {k + 1}/{len(listed)}", end='\r')
        print()
    server = http.server.ThreadingHTTPServer(('127.0.0.1', port), handler_for(state))
    print(f"{session.name}: open http://127.0.0.1:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

# %%
if __name__ == "__main__":
    main()
