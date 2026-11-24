"""
Process all (or some) sessions of a station, one after the other: per session
1. download it if the sessions are on S3 (aws s3 sync, videos already there are skipped); skipped
   if the free disk is less than what is left to download plus --spare_gb (processing outputs and
   one MCAP batch), before anything is downloaded;
2. run_tag_tracking_pipeline.py (all steps, no render);
3. the dataset plan again with --keep_gaps (episodes with a short tracking gap kept,
   pose not available there), unless --no_keep_gaps;
4. with --mcap: the episode MCAP files (07_export_mcap.py -c) to <session>/dataset_mcap/
   (MCAP files of an earlier or failed export are deleted first); with --upload in batches of
   --batch_size episodes: export, upload to <upload>/<session>/, check on S3 (name and size),
   delete locally, so the disk holds the videos and one batch; without --upload all at once
   (stops if the free disk is less than 1.2 x the session's videos);
5. with --upload and --start_from upload: the MCAP files already in dataset_mcap/;
6. with --cleanup (sessions on S3 and --upload): once every MCAP file is on S3 with
   the same size, the local videos (demos/*/raw_video.mp4 and raw_videos/: the originals stay
   in the S3 session) and the local MCAP files are deleted; logs, pairs.csv, tag tracking
   outputs and dataset_plan.pkl are kept. Not deleted if anything does not match.
--videos_only --mp4 <prefix> (episode videos, no poses, no MCAP files): instead of 2.-5., the
pipeline with --videos_only (episodes, head cube tracking for the motion sync, wrist sync, gripper
widths, plan with --ignore_poses) and 15_export_episode_videos.py to <mp4>/<station>/<date>/<task>/
(three mp4s, gripper CSVs, meta.json per episode, --mosaic N, report.json per task); with --cleanup
the local videos are deleted once the export has run (the originals stay in the S3 session).
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
Resume after a failure: --start_from pipeline / plan / mcap / upload / export (e.g. after freeing disk).

python scripts_tag_tracking/process_station.py s3://my-bucket/UMI_Collection/station3/industrial_data/2026-11-22 \\
    -c example/tag_tracking/tag_tracking_station3.yaml -w ~/umi_data/station3/2026-11-22 \\
    --videos_only --mp4 s3://my-bucket/umi_data/industrial_data_mp4 --mosaic 1000 --cleanup
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# %%
import json
import pickle
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


def s3_total_bytes(url: str) -> int:
    """
    Total size of all objects under an s3:// prefix (recursive).
    """
    bucket, _, prefix = url[len('s3://'):].partition('/')
    out = subprocess.run(['aws', 's3api', 'list-objects-v2', '--bucket', bucket, '--prefix', prefix.rstrip('/') + '/',
        '--query', 'sum(Contents[].Size)', '--output', 'json'], check=True, capture_output=True, text=True).stdout
    return int(json.loads(out) or 0)


def local_bytes(path: pathlib.Path) -> int:
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink()) if path.exists() else 0


def not_uploaded(paths: list, upload_url: str) -> list:
    """
    The local files not on S3 under upload_url with the same name and size.
    """
    remote = s3_sizes(upload_url)
    return sorted(p.name for p in paths if remote.get(p.name) != p.stat().st_size)


def cleanup(session: pathlib.Path, upload_url: str, mcaps_verified: bool = False, what: str = 'MCAP files') -> str:
    """
    Delete the local videos and MCAP files of a session whose MCAP files are all on S3
    (same names and sizes; mcaps_verified: already checked and deleted batch by batch).
    Returns what was done.
    """
    local = list(session.joinpath('dataset_mcap').glob('*.mcap'))
    if not mcaps_verified:
        mismatch = not_uploaded(local, upload_url)
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
        f"({freed / 1e9:.1f} GB); videos stay in the S3 session, {what} on {upload_url}"


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


def task_path(sessions: str, name: str) -> str:
    """
    A session's folders in the episode video output: from the station folder on, without
    industrial_data (s3://b/UMI_Collection/station3/industrial_data/2026-11-22 + task01 ->
    station3/2026-11-22/task01); without a station folder: the session name.
    """
    parts = [p for p in f"{sessions.rstrip('/')}/{name}".split('://', 1)[-1].split('/') if p and p != 'industrial_data']
    station = next((k for k, p in enumerate(parts) if re.fullmatch(r'station\d+', p)), None)
    return '/'.join(parts[station:]) if station is not None else name


def station_job(sessions: str, config: str, workdir, tasks, mcap: bool, upload, cleanup_after: bool, name=None,
        mp4=None) -> dict:
    """
    One station's run, checked before anything starts: its sessions (all, or tasks), config,
    local folder, upload prefix.
    """
    config = os.path.abspath(os.path.expanduser(config))
    assert os.path.isfile(config), f"config not found: {config}"
    # without a setup calibration every session would calibrate its own bodies and TCP (slow, one
    # process, and different per session): build it once first (README step 2)
    assert yaml.safe_load(open(config)).get('setup_calibration'), \
        f"{config}: no setup_calibration; build it once on one session (calibrate_setup.py, README step 2)"
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
    assert not (mp4 and (mcap or upload)), "--videos_only --mp4 replaces --mcap / --upload"
    # the videos are deleted only where they can be downloaded again
    assert not cleanup_after or (on_s3 and (upload or mp4)), \
        f"{sessions}: --cleanup needs sessions on s3:// and an upload (--upload or --mp4)"
    if name is None:   # station name: the folder above the sessions folder, e.g. station3/industrial_data
        parts = sessions.split('/')
        name = parts[-2] if len(parts) > 1 and parts[-1] == 'industrial_data' else parts[-1]
    return {'name': name, 'sessions': sessions, 'config': config, 'on_s3': on_s3, 'root': root, 'names': names,
        'upload': upload, 'mp4': mp4}


STEPS = ['download', 'pipeline', 'plan', 'mcap', 'upload', 'export']


def free_gb(path: pathlib.Path) -> float:
    import shutil
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 1e9


def process(job: dict, no_keep_gaps: bool, mcap: bool, cleanup_after: bool, num_workers, start_from='download',
        batch_size: int = 10, spare_gb: float = 5.0, n_mosaic: int = 0, mp4_jobs: int = 4) -> list:
    """
    Run one station's sessions; returns per session (name, status, failed step, yield, minutes).
    """
    sessions, config, upload, mp4 = job['sessions'], job['config'], job['upload'], job.get('mp4')
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
            if job['on_s3'] and STEPS.index(start_from) <= STEPS.index('download'):
                # the session must fit: what is not local yet, plus room for the processing outputs and one
                # MCAP batch; otherwise skipped before downloading anything (no half-full disk)
                missing_gb = max(0.0, s3_total_bytes(f"{sessions}/{name}") - local_bytes(session)) / 1e9
                need_gb = missing_gb + spare_gb
                if free_gb(session) < need_gb:
                    status, failed = 'skipped', f"download: {free_gb(session):.1f} GB free, about {need_gb:.1f} GB " \
                        f"needed ({missing_gb:.1f} GB to download + {spare_gb:.0f} GB for processing); enlarge the disk"
                    print(failed, flush=True)
                    log.write(failed + '\n')
                    summary.append((f"{job['name']} {name}", status, failed, None, (time.time() - t0) / 60))
                    continue
                steps.append(('download', ['aws', 's3', 'sync', f"{sessions}/{name}", session, '--only-show-errors']))
            if STEPS.index(start_from) <= STEPS.index('pipeline'):
                steps.append(('pipeline', [py, 'run_tag_tracking_pipeline.py', '-c', config, '-nr',
                    *(['--videos_only'] if mp4 else []), *workers, session]))
            if mp4:
                # episode videos straight from the session (no poses, no MCAP files)
                steps.append(('export', [py, 'scripts_tag_tracking/15_export_episode_videos.py', '-i', session,
                    '-c', config, '--task', task_path(sessions, name), '-o', mp4,
                    '-w', session.joinpath('mp4_work'), '-j', mp4_jobs, '--mosaic', n_mosaic]))
            if not mp4 and not no_keep_gaps and STEPS.index(start_from) <= STEPS.index('plan'):
                steps.append(('plan --keep_gaps', [py, 'scripts_tag_tracking/06_export_dataset_plan.py',
                    '-i', session, '-c', config, '--keep_gaps']))
            batched = mcap and upload and STEPS.index(start_from) <= STEPS.index('mcap')
            if mcap and STEPS.index(start_from) <= STEPS.index('mcap'):
                # MCAP files of an earlier or failed export are deleted (they are made again from the plan and
                # the videos), so the folder (and the upload) holds only this export and the disk is free for it
                previous = list(session.joinpath('dataset_mcap').glob('*.mcap'))
                if previous:
                    for p in previous:
                        p.unlink()
                    print(f"deleted {len(previous)} MCAP files of an earlier export", flush=True)
            if mcap and not batched and STEPS.index(start_from) <= STEPS.index('mcap'):
                # all MCAP files at once (not uploaded): about as much free disk as the videos, kept 20% spare
                videos_gb = sum(p.stat().st_size for p in session.joinpath('demos').glob('*/raw_video.mp4')) / 1e9
                if free_gb(session) < 1.2 * videos_gb:
                    status, failed = 'failed', f"mcap: {free_gb(session):.1f} GB free, about {1.2 * videos_gb:.1f} GB " \
                        f"needed (videos {videos_gb:.1f} GB); free disk, or upload (--upload exports in batches)"
                    print(failed, flush=True)
                    log.write(failed + '\n')
                    summary.append((f"{job['name']} {name}", status, failed, None, (time.time() - t0) / 60))
                    continue
                steps.append(('mcap', [py, 'scripts_tag_tracking/07_export_mcap.py', '-i', session, '-c', config,
                    *(['-j', min(num_workers, 4)] if num_workers else ['-j', 4])]))
            if upload and not batched and STEPS.index(start_from) <= STEPS.index('upload'):
                steps.append(('upload', ['aws', 's3', 'sync', session.joinpath('dataset_mcap'),
                    f"{upload.rstrip('/')}/{name}/", '--only-show-errors']))
            for step, cmd in steps:
                code, lines = run(cmd, log)
                if step in ('pipeline', 'plan --keep_gaps'):
                    found = [l for l in lines if re.match(r'\d+ episodes, ', l.strip())]
                    plan_yield = found[-1].strip() if found else plan_yield
                if step == 'export':
                    found = [l.strip() for l in lines if l.startswith('done: ')]
                    exported = f"episode videos {found[-1][len('done: '):]}, {mp4.rstrip('/')}/{task_path(sessions, name)}" \
                        if found else None
                    plan_yield = f"{plan_yield}; {exported}" if plan_yield and exported else exported or plan_yield
                if code != 0:
                    status = 'failed'
                    # the pipeline names the failed step and its log
                    reason = [l for l in lines if 'failed, see' in l]
                    failed = f"{step}: {reason[-1].split('AssertionError: ')[-1]}" if reason else \
                        f"{step} (exit {code}), see {session.joinpath('logs', 'process_station.log')}"
                    break
            n_uploaded = 0
            if batched and status == 'ok':
                # export, upload, check on S3 and delete batch_size episodes at a time: the disk holds the
                # videos and one batch of MCAP files, not all of them
                url = f"{upload.rstrip('/')}/{name}"
                n_episodes = len(pickle.load(session.joinpath('dataset_plan.pkl').open('rb')))
                for b0 in range(0, n_episodes, batch_size):
                    batch = list(range(b0, min(b0 + batch_size, n_episodes)))
                    print(f"MCAP episodes {batch[0]}-{batch[-1]} of {n_episodes}", flush=True)
                    code, _ = run([py, 'scripts_tag_tracking/07_export_mcap.py', '-i', session, '-c', config,
                        '-e', ','.join(str(k) for k in batch),
                        *(['-j', min(num_workers, 4)] if num_workers else ['-j', 4])], log)
                    if code == 0:
                        code, _ = run(['aws', 's3', 'sync', session.joinpath('dataset_mcap'), url + '/',
                            '--only-show-errors'], log)
                    files = list(session.joinpath('dataset_mcap').glob('*.mcap'))
                    missing = not_uploaded(files, url) if code == 0 else None
                    if code != 0 or missing:
                        status = 'failed'
                        failed = f"mcap / upload of episodes {batch[0]}-{batch[-1]}" + \
                            (f": not on S3 {missing}" if missing else f" (exit {code})") + \
                            f", see {session.joinpath('logs', 'process_station.log')}"
                        break
                    for p in files:
                        p.unlink()
                    n_uploaded += len(files)
            if mcap and status == 'ok':
                n = n_uploaded if batched else len(list(session.joinpath('dataset_mcap').glob('*.mcap')))
                n = f"{n} MCAP files" + (' (uploaded and checked in batches, deleted locally)' if batched else '')
                plan_yield = f"{plan_yield}; {n}" if plan_yield else n
            if cleanup_after and status == 'ok' and mp4:
                done = cleanup(session, f"{mp4.rstrip('/')}/{task_path(sessions, name)}", mcaps_verified=True,
                    what='episode videos')
                print(done, flush=True)
                log.write(done + '\n')
                plan_yield = f"{plan_yield}\n  {done}"
            elif cleanup_after and status == 'ok':
                done = cleanup(session, f"{upload.rstrip('/')}/{name}", mcaps_verified=batched)
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
@click.option('--start_from', type=click.Choice(STEPS), default='download',
    help='First step per session (resume after a failure; earlier steps\' outputs must exist)')
@click.option('--spare_gb', type=float, default=5.0,
    help='Free disk needed beyond a session\'s download (processing outputs, one MCAP batch)')
@click.option('--batch_size', type=int, default=10,
    help='With --mcap --upload: episodes exported, uploaded, checked and deleted locally per batch')
@click.option('--videos_only', is_flag=True, default=False,
    help='Episode videos only (no poses, no MCAP files): pipeline --videos_only and 15_export_episode_videos.py to --mp4')
@click.option('--mp4', default=None, help='With --videos_only: output prefix (s3:// or local), <mp4>/<station>/<date>/<task>/')
@click.option('--mosaic', 'n_mosaic', type=int, default=0, help='With --videos_only: mosaics per task (1000: all)')
@click.option('--mp4_jobs', type=int, default=4, help='With --videos_only: episodes exported at the same time')
@click.option('-j', '--num_workers', type=int, default=None, help='Parallel processes per step, default all cores')
def main(sessions, config, workdir, tasks, jobs_yaml, no_keep_gaps, mcap, upload, cleanup_after, start_from,
        spare_gb, batch_size, videos_only, mp4, n_mosaic, mp4_jobs, num_workers):
    assert bool(videos_only) == bool(mp4), "--videos_only and --mp4 go together"
    if jobs_yaml:
        assert sessions is None and config is None, "--jobs replaces SESSIONS / -c (give them per station in the yaml)"
        entries = yaml.safe_load(open(os.path.expanduser(jobs_yaml), 'r'))
        assert isinstance(entries, list) and entries, f"{jobs_yaml}: a list of stations expected"
        for k, e in enumerate(entries):
            missing = [key for key in ('sessions', 'config') if not e.get(key)]
            assert not missing, f"{jobs_yaml}, station {k + 1} ({e.get('sessions')}): {', '.join(missing)} not set " \
                "(a station needs its tag tracking config with setup_calibration, README step 2)"
        # every station checked before the first starts
        jobs = [station_job(e['sessions'], e['config'], e.get('workdir'), e.get('tasks'), mcap, e.get('upload'),
            cleanup_after, e.get('name'), mp4) for e in entries]
    else:
        assert sessions and config, "SESSIONS and -c, or --jobs"
        jobs = [station_job(sessions, config, workdir, tasks, mcap, upload, cleanup_after, mp4=mp4)]
    for job in jobs:
        print(f"{job['name']}: {len(job['names'])} sessions: {', '.join(job['names'])}", flush=True)

    summary = list()
    for job in jobs:
        summary += process(job, no_keep_gaps, mcap, cleanup_after, num_workers, start_from, batch_size, spare_gb,
            n_mosaic, mp4_jobs)

    print("\n============= summary =============")
    for name, status, failed, plan_yield, minutes in summary:
        print(f"{name}: {status} ({minutes:.0f} min)" + (f", {failed}" if failed else '')
            + (f"\n  {plan_yield}" if plan_yield else ''))

# %%
if __name__ == "__main__":
    main()
