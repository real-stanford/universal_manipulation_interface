"""
How many episodes, and how long, in a folder of episode MCAP files (07_export_mcap.py), per
task folder and in total, split into good episodes (a pose for every step of both grippers)
and episodes kept with tracking gaps (--keep_gaps plans: pose not available in some steps).

Reads only each MCAP's metadata 'recording' (episode_length steps, video_fps,
robot<i>_pose_missing_steps) from the end of the file: on S3 by ranged reads (aws CLI and its
credentials), the videos are not downloaded. Duration = episode_length / video_fps (the
episode's steps, without the video lead-in). MCAP files without robot<i>_pose_missing_steps
(exported before --keep_gaps existed) had no gaps: counted as good.

python scripts_tag_tracking/report_episodes.py s3://my-bucket/umi_data/industrial_data [--csv report.csv]
python scripts_tag_tracking/report_episodes.py ~/umi_data/station1/industrial_data
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# %%
import collections
import concurrent.futures
import csv
import json
import pathlib
import subprocess
import click
from mcap.reader import make_reader


def split_s3(url: str):
    bucket, _, key = url[len('s3://'):].partition('/')
    return bucket, key


class S3File:
    """
    Read-only seekable view of an S3 object by ranged reads (block cache), enough for the MCAP
    reader to read the summary, the metadata and a few chunks. Ranges are fetched over HTTP
    from one presigned URL per file (aws CLI once), not one aws CLI process per range.
    """
    BLOCK = 4 << 20

    def __init__(self, url: str, size: int):
        self.url = url
        self.bucket, self.key = split_s3(url)
        self.size, self.pos, self.blocks, self.presigned = size, 0, dict(), None

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def block(self, k: int) -> bytes:
        if k not in self.blocks:
            import urllib.request
            if self.presigned is None:
                self.presigned = subprocess.run(['aws', 's3', 'presign', self.url, '--expires-in', '3600'],
                    check=True, capture_output=True, text=True).stdout.strip()
            start, end = k * self.BLOCK, min(self.size, (k + 1) * self.BLOCK) - 1
            request = urllib.request.Request(self.presigned, headers={'Range': f'bytes={start}-{end}'})
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(request, timeout=60) as response:
                        self.blocks[k] = response.read()
                    break
                except OSError:
                    if attempt == 2:
                        raise RuntimeError(f"{self.url}: bytes {start}-{end} not readable") from None
        return self.blocks[k]

    def read(self, n=-1):
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        out = bytearray()
        while self.pos < end:
            k = self.pos // self.BLOCK
            b = self.block(k)
            off = self.pos - k * self.BLOCK
            chunk = b[off:off + (end - self.pos)]
            out += chunk
            self.pos += len(chunk)
        return bytes(out)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.blocks.clear()


def list_mcaps(root: str) -> list:
    """
    (path or URL, size, task folder relative to root) of every .mcap under root.
    """
    if root.startswith('s3://'):
        bucket, prefix = split_s3(root.rstrip('/') + '/')
        out = subprocess.run(['aws', 's3api', 'list-objects-v2', '--bucket', bucket, '--prefix', prefix,
            '--query', 'Contents[].[Key,Size]', '--output', 'json'], check=True, capture_output=True, text=True).stdout
        items = [(k, n) for k, n in (json.loads(out) or []) if k.endswith('.mcap')]
        return [(f"s3://{bucket}/{k}", n, os.path.dirname(k[len(prefix):])) for k, n in items]
    base = pathlib.Path(os.path.expanduser(root)).absolute()
    return [(str(p), p.stat().st_size, str(p.parent.relative_to(base))) for p in sorted(base.rglob('*.mcap'))]


def episode_info(path: str, size: int) -> dict:
    """
    One MCAP's steps, fps, duration (s) and missing pose steps per robot, from its metadata.
    """
    stream = S3File(path, size) if path.startswith('s3://') else open(path, 'rb')
    with stream:
        recording = {m.name: m.metadata for m in make_reader(stream).iter_metadata()}.get('recording')
    assert recording is not None, f"{path}: no 'recording' metadata (not from 07_export_mcap.py)"
    steps, fps = int(recording['episode_length']), float(recording['video_fps'])
    missing = [int(recording[f'robot{i}_pose_missing_steps']) for i in range(int(recording.get('n_grippers', 2)))
        if f'robot{i}_pose_missing_steps' in recording]
    return {'steps': steps, 'fps': fps, 'duration_s': steps / fps, 'missing_steps': missing,
        'good': all(m == 0 for m in missing)}


# %%
@click.command()
@click.argument('root')
@click.option('--csv', 'csv_path', default=None, help='Also write one row per episode here')
@click.option('-j', '--jobs', type=int, default=16, help='Files read in parallel')
def main(root, csv_path, jobs):
    files = list_mcaps(root)
    assert files, f"no .mcap files under {root}"
    print(f"{len(files)} MCAP files under {root}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        infos = list(pool.map(lambda f: episode_info(f[0], f[1]), files))

    per_task = collections.OrderedDict()
    for (path, size, task), info in sorted(zip(files, infos), key=lambda x: (x[0][2], x[0][0])):
        per_task.setdefault(task or '.', list()).append((path, size, info))

    hms = lambda s: f"{int(s // 3600)}:{int(s % 3600 // 60):02d}:{int(s % 60):02d}"
    header = f"{'task':<36} {'episodes':>8} {'good':>6} {'good duration':>14} {'with gaps':>10} {'gaps duration':>14}"
    print('\n' + header + '\n' + '-' * len(header))
    totals = collections.Counter()
    for task, rows in per_task.items():
        good = [i for _, _, i in rows if i['good']]
        gaps = [i for _, _, i in rows if not i['good']]
        t = {'episodes': len(rows), 'good': len(good), 'good_s': sum(i['duration_s'] for i in good),
            'gaps': len(gaps), 'gaps_s': sum(i['duration_s'] for i in gaps)}
        totals.update(t)
        print(f"{task:<36} {t['episodes']:>8} {t['good']:>6} {hms(t['good_s']):>14} {t['gaps']:>10} {hms(t['gaps_s']):>14}")
    print('-' * len(header))
    print(f"{'total':<36} {totals['episodes']:>8} {totals['good']:>6} {hms(totals['good_s']):>14} "
        f"{totals['gaps']:>10} {hms(totals['gaps_s']):>14}")
    print(f"\ngood: a pose for every step of both grippers; duration: episode steps / video fps "
        f"(good {totals['good_s'] / 60:.1f} min, {totals['good_s'] / 3600:.2f} h)")

    if csv_path:
        with open(os.path.expanduser(csv_path), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['task', 'file', 'size_mb', 'steps', 'fps', 'duration_s', 'good', 'missing_pose_steps'])
            for task, rows in per_task.items():
                for path, size, i in rows:
                    w.writerow([task, path.rsplit('/', 1)[-1], round(size / 1e6, 1), i['steps'], round(i['fps'], 3),
                        round(i['duration_s'], 2), i['good'], ';'.join(str(m) for m in i['missing_steps'])])
        print(f"saved {csv_path}")

# %%
if __name__ == "__main__":
    main()
