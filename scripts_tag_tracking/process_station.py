"""
Process all (or some) sessions of a station, one after the other: per session
1. download it if the sessions are on S3 (aws s3 sync, videos already there are skipped);
2. run_tag_tracking_pipeline.py (all steps, no render);
3. the dataset plan again with --keep_gaps (episodes with a short tracking gap kept,
   pose not available there), unless --no_keep_gaps;
4. with --mcap: the episode MCAP files (07_export_mcap.py -c) to <session>/dataset_mcap/
   (MCAP files of an earlier export are moved to dataset_mcap_<date>_<time>/ first);
5. with --upload: the MCAP files to <upload>/<session>/;
6. with --cleanup (sessions on S3 and --upload): once every local MCAP file is on S3 with
   the same size, the local videos (demos/*/raw_video.mp4 and raw_videos/: the originals stay
   in the S3 session) and the local MCAP files are deleted; logs, pairs.csv, tag tracking
   outputs and dataset_plan.pkl are kept. Not deleted if anything does not match.
A session that fails does not stop the others. At the end, per session: status, the
failed step and its log, and the yield (episodes, seconds used). The output of each
session is also written to <session>/logs/process_station.log.

Sessions: the sub-folders of SESSIONS with raw_videos/ (from sort_recordings.py --tasks),
local or s3:// (then -w, the local folder to download to). The station's tag tracking
config must have its setup calibration (README, step 2).

python scripts_tag_tracking/process_station.py s3://my-bucket/UMI_Collection/station3/industrial_data \\
    -c example/tag_tracking/tag_tracking_station3.yaml -w ~/umi_data/station3/industrial_data \\
    --tasks task02,task03,task04 --mcap --upload s3://my-bucket/mcaps/station3

Several stations one after the other (--jobs, e.g. example/tag_tracking/process_jobs.yaml):
    - sessions: s3://my-bucket/UMI_Collection/station1/industrial_data
      config: example/tag_tracking/tag_tracking_station1.yaml
      workdir: ~/umi_data/station1/industrial_data
      tasks: [task03, task04, task05]      # optional, default all
      upload: s3://my-bucket/mcaps/station1
    - sessions: ...
python scripts_tag_tracking/process_station.py --jobs jobs.yaml --mcap --cleanup
All stations are checked (configs, sessions found) before the first starts; one summary at the end.
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# %%
import json
import pathlib
import re
import subprocess
import time
import click
import yaml


def s3_sizes(url: str) -> dict:
    """
    File name -> size of the objects directly under an s3:// prefix.
    """
    bucket, _, prefix = url[len('s3://'):].partition('/')
    prefix = prefix.rstrip('/') + '/'
    out = subprocess.run(['aws', 's3api', 'list-objects-v2', '--bucket', bucket, '--prefix', prefix,
        '--query', 'Contents[].[Key,Size]', '--output', 'json'], check=True, capture_output=True, text=True).stdout
    return {k[len(prefix):]: int(n) for k, n in (json.loads(out) or []) if '/' not in k[len(prefix):]}


def cleanup(session: pathlib.Path, upload_url: str) -> str:
    """
    Delete the local videos and MCAP files of a session whose MCAP files are all on S3
    (same names and sizes). Returns what was done.
    """
    local = {p.name: p.stat().st_size for p in session.joinpath('dataset_mcap').glob('*.mcap')}
    remote = s3_sizes(upload_url)
    mismatch = sorted(n for n, size in local.items() if remote.get(n) != size)
    if not local or mismatch:
        return f"cleanup skipped: MCAP files not all on {upload_url} with the same size: {mismatch or 'no local MCAP files'}"
    videos = [p for p in session.joinpath('demos').glob('*/raw_video.mp4') if p.is_file() and not p.is_symlink()]
    links = [p for p in session.joinpath('raw_videos').rglob('*') if p.suffix.lower() == '.mp4']
    mcaps = list(session.joinpath('dataset_mcap').glob('*.mcap'))
    freed = sum(p.stat().st_size for p in videos + mcaps) + sum(p.stat().st_size for p in links
        if p.is_file() and not p.is_symlink())
    for p in videos + links + mcaps:
        p.unlink()
    return f"cleanup: deleted {len(videos)} videos, {len(links)} raw_videos entries, {len(mcaps)} MCAP files " \
        f"({freed / 1e9:.1f} GB); videos stay in the S3 session, MCAP files on {upload_url}"


def s3_sessions(url: str) -> list:
    """
    Session names under an s3:// prefix: sub-folders with raw_videos/.
    """
    bucket, _, prefix = url[len('s3://'):].partition('/')
    prefix = prefix.rstrip('/') + '/'
    out = subprocess.run(['aws', 's3api', 'list-objects-v2', '--bucket', bucket, '--prefix', prefix,
        '--query', 'Contents[].Key', '--output', 'json'], check=True, capture_output=True, text=True).stdout
    keys = json.loads(out) or []
    return sorted({k[len(prefix):].split('/')[0] for k in keys if k[len(prefix):].split('/')[1:2] == ['raw_videos']})


def run(cmd: list, log) -> tuple:
    """
    Run a command, its output to the console and the log; returns (return code, output lines).
    """
    print('$ ' + ' '.join(str(c) for c in cmd), flush=True)
    log.write('$ ' + ' '.join(str(c) for c in cmd) + '\n')
    lines = list()
    with subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            cwd=ROOT_DIR) as p:
        for line in p.stdout:
            sys.stdout.write(line)
            log.write(line)
            lines.append(line.rstrip('\n'))
    log.flush()
    return p.returncode, lines


def station_job(sessions: str, config: str, workdir, tasks, mcap: bool, upload, cleanup_after: bool, name=None) -> dict:
    """
    One station's run, checked before anything starts: its sessions (all, or tasks), config,
    local folder, upload prefix.
    """
    config = os.path.abspath(os.path.expanduser(config))
    assert os.path.isfile(config), f"config not found: {config}"
    on_s3 = sessions.startswith('s3://')
    sessions = sessions.rstrip('/')
    if on_s3:
        assert workdir, f"{sessions}: a local folder (-w / workdir) is needed for sessions on S3"
        names = s3_sessions(sessions)
        root = pathlib.Path(os.path.expanduser(workdir)).absolute()
    else:
        root = pathlib.Path(os.path.expanduser(sessions)).absolute()
        names = sorted(p.name for p in root.iterdir() if p.joinpath('raw_videos').is_dir())
    if tasks:
        wanted = tasks.split(',') if isinstance(tasks, str) else [str(t) for t in tasks]
        missing = [t for t in wanted if t not in names]
        assert not missing, f"not found in {sessions}: {missing} (sessions: {names})"
        names = wanted
    assert names, f"no sessions (folders with raw_videos/) in {sessions}"
    assert upload is None or mcap, "upload uploads the MCAP files: needs --mcap"
    # the videos are deleted only where they can be downloaded again
    assert not cleanup_after or (on_s3 and upload), f"{sessions}: --cleanup needs sessions on s3:// and an upload"
    if name is None:   # station name: the folder above the sessions folder, e.g. station3/industrial_data
        parts = sessions.split('/')
        name = parts[-2] if len(parts) > 1 and parts[-1] == 'industrial_data' else parts[-1]
    return {'name': name, 'sessions': sessions, 'config': config, 'on_s3': on_s3, 'root': root, 'names': names,
        'upload': upload}


def process(job: dict, no_keep_gaps: bool, mcap: bool, cleanup_after: bool, num_workers) -> list:
    """
    Run one station's sessions; returns per session (name, status, failed step, yield, minutes).
    """
    sessions, config, upload = job['sessions'], job['config'], job['upload']
    workers = [] if num_workers is None else ['-j', num_workers]
    py = sys.executable
    summary = list()
    for name in job['names']:
        session = job['root'].joinpath(name)
        session.joinpath('logs').mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        status, failed, plan_yield = 'ok', None, None
        with session.joinpath('logs', 'process_station.log').open('a') as log:
            log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            print(f"\n============= {job['name']} {name} =============", flush=True)
            steps = list()
            if job['on_s3']:
                steps.append(('download', ['aws', 's3', 'sync', f"{sessions}/{name}", session, '--only-show-errors']))
            steps.append(('pipeline', [py, 'run_tag_tracking_pipeline.py', '-c', config, '-nr', *workers, session]))
            if not no_keep_gaps:
                steps.append(('plan --keep_gaps', [py, 'scripts_tag_tracking/06_export_dataset_plan.py',
                    '-i', session, '-c', config, '--keep_gaps']))
            if mcap:
                # MCAP files of an earlier export (other plan, other episode count) are kept aside,
                # so the folder (and the upload) holds only this export
                previous = session.joinpath('dataset_mcap')
                if any(previous.glob('*.mcap')):
                    keep = session.joinpath(f"dataset_mcap_{time.strftime('%Y%m%d_%H%M%S')}")
                    previous.rename(keep)
                    print(f"earlier MCAP files moved to {keep}", flush=True)
                steps.append(('mcap', [py, 'scripts_tag_tracking/07_export_mcap.py', '-i', session, '-c', config,
                    *(['-j', min(num_workers, 4)] if num_workers else ['-j', 4])]))
            if upload:
                steps.append(('upload', ['aws', 's3', 'sync', session.joinpath('dataset_mcap'),
                    f"{upload.rstrip('/')}/{name}/", '--only-show-errors']))
            for step, cmd in steps:
                code, lines = run(cmd, log)
                if step in ('pipeline', 'plan --keep_gaps'):
                    found = [l for l in lines if re.match(r'\d+ episodes, ', l.strip())]
                    plan_yield = found[-1].strip() if found else plan_yield
                if code != 0:
                    status = 'failed'
                    # the pipeline names the failed step and its log
                    reason = [l for l in lines if 'failed, see' in l]
                    failed = f"{step}: {reason[-1].split('AssertionError: ')[-1]}" if reason else \
                        f"{step} (exit {code}), see {session.joinpath('logs', 'process_station.log')}"
                    break
            if mcap and status == 'ok':
                n = len(list(session.joinpath('dataset_mcap').glob('*.mcap')))
                plan_yield = f"{plan_yield}; {n} MCAP files" if plan_yield else f"{n} MCAP files"
            if cleanup_after and status == 'ok':
                done = cleanup(session, f"{upload.rstrip('/')}/{name}")
                print(done, flush=True)
                log.write(done + '\n')
                plan_yield = f"{plan_yield}\n  {done}"
        summary.append((f"{job['name']} {name}", status, failed, plan_yield, (time.time() - t0) / 60))
    return summary


# %%
@click.command()
@click.argument('sessions', required=False)
@click.option('-c', '--config', default=None, help="The station's tag tracking config")
@click.option('-w', '--workdir', default=None, help='Local folder for the sessions (required for s3:// sessions)')
@click.option('--tasks', default=None, help='Comma separated session names, default all')
@click.option('--jobs', 'jobs_yaml', default=None,
    help='Several stations, one after the other: yaml list of {sessions, config, workdir, tasks, upload} (instead of SESSIONS)')
@click.option('--no_keep_gaps', is_flag=True, default=False, help='Drop episodes with tracking gaps (no --keep_gaps plan)')
@click.option('--mcap', is_flag=True, default=False, help='Export the episode MCAP files')
@click.option('--upload', default=None, help='s3:// prefix: upload each session\'s MCAP files to <upload>/<session>/')
@click.option('--cleanup', 'cleanup_after', is_flag=True, default=False,
    help='After a verified upload: delete the local videos and MCAP files (sessions on S3 only)')
@click.option('-j', '--num_workers', type=int, default=None, help='Parallel processes per step, default all cores')
def main(sessions, config, workdir, tasks, jobs_yaml, no_keep_gaps, mcap, upload, cleanup_after, num_workers):
    if jobs_yaml:
        assert sessions is None and config is None, "--jobs replaces SESSIONS / -c (give them per station in the yaml)"
        entries = yaml.safe_load(open(os.path.expanduser(jobs_yaml), 'r'))
        assert isinstance(entries, list) and entries, f"{jobs_yaml}: a list of stations expected"
        # every station checked before the first starts
        jobs = [station_job(e['sessions'], e['config'], e.get('workdir'), e.get('tasks'), mcap, e.get('upload'),
            cleanup_after, e.get('name')) for e in entries]
    else:
        assert sessions and config, "SESSIONS and -c, or --jobs"
        jobs = [station_job(sessions, config, workdir, tasks, mcap, upload, cleanup_after)]
    for job in jobs:
        print(f"{job['name']}: {len(job['names'])} sessions: {', '.join(job['names'])}", flush=True)

    summary = list()
    for job in jobs:
        summary += process(job, no_keep_gaps, mcap, cleanup_after, num_workers)

    print("\n============= summary =============")
    for name, status, failed, plan_yield, minutes in summary:
        print(f"{name}: {status} ({minutes:.0f} min)" + (f", {failed}" if failed else '')
            + (f"\n  {plan_yield}" if plan_yield else ''))

# %%
if __name__ == "__main__":
    main()
