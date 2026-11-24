"""
scripts_tag_tracking/process_station.py cleanup: deletes the local videos and MCAP files
only when every MCAP file is on S3 with the same size; keeps everything else.

python tests/test_process_station.py
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts_tag_tracking'))

import pathlib
import tempfile
import process_station


def make_session(root: pathlib.Path) -> pathlib.Path:
    s = root.joinpath('task01')
    for d in ['raw_videos/gripper_calibration', 'demos/demo_A', 'demos/gripper_calibration_A', 'dataset_mcap',
            'tag_tracking', 'logs']:
        s.joinpath(d).mkdir(parents=True)
    s.joinpath('demos/demo_A/raw_video.mp4').write_bytes(b'v' * 100)
    s.joinpath('demos/demo_A/tag_corners.pkl').write_bytes(b'x')
    s.joinpath('demos/gripper_calibration_A/raw_video.mp4').write_bytes(b'v' * 50)
    s.joinpath('raw_videos/GX01.MP4').symlink_to('../demos/demo_A/raw_video.mp4')
    s.joinpath('raw_videos/gripper_calibration/GX02.MP4').symlink_to('../../demos/gripper_calibration_A/raw_video.mp4')
    s.joinpath('raw_videos/pairs.csv').write_text('type\n')
    s.joinpath('dataset_mcap/episode_000000.mcap').write_bytes(b'm' * 10)
    s.joinpath('dataset_mcap/episode_000001.mcap').write_bytes(b'm' * 20)
    s.joinpath('dataset_plan.pkl').write_bytes(b'p')
    return s


def test_cleanup():
    with tempfile.TemporaryDirectory() as tmp:
        s = make_session(pathlib.Path(tmp))
        # one MCAP file missing on S3: nothing deleted
        process_station.s3_sizes = lambda url: {'episode_000000.mcap': 10}
        assert 'skipped' in process_station.cleanup(s, 's3://b/x/task01')
        # a size differs: nothing deleted
        process_station.s3_sizes = lambda url: {'episode_000000.mcap': 10, 'episode_000001.mcap': 19}
        assert 'skipped' in process_station.cleanup(s, 's3://b/x/task01')
        assert s.joinpath('demos/demo_A/raw_video.mp4').is_file() and len(list(s.rglob('*.mcap'))) == 2
        # all there: videos, raw_videos entries and MCAP files deleted, the rest kept
        process_station.s3_sizes = lambda url: {'episode_000000.mcap': 10, 'episode_000001.mcap': 20, 'other': 1}
        done = process_station.cleanup(s, 's3://b/x/task01')
        assert 'deleted 2 videos, 2 raw_videos entries, 2 MCAP files' in done, done
        assert not list(s.rglob('*.mp4')) and not list(s.rglob('*.MP4')) and not list(s.rglob('*.mcap'))
        for kept in ['raw_videos/pairs.csv', 'demos/demo_A/tag_corners.pkl', 'dataset_plan.pkl']:
            assert s.joinpath(kept).is_file(), kept
        # nothing local any more: skipped, not an error
        assert 'skipped' in process_station.cleanup(s, 's3://b/x/task01')


def test_station_job():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp, 'station3', 'industrial_data')
        for t in ['task01', 'task02']:
            root.joinpath(t, 'raw_videos').mkdir(parents=True)
        root.joinpath('notes').mkdir()
        config = pathlib.Path(tmp, 'config.yaml')
        config.write_text('setup_calibration: setup_x\n')
        job = process_station.station_job(str(root), str(config), None, None, True, None, False)
        assert job['name'] == 'station3' and job['names'] == ['task01', 'task02'] and not job['on_s3']
        assert process_station.station_job(str(root), str(config), None, ['task02'], True, None, False)['names'] == ['task02']
        for bad in [dict(tasks='task03'), dict(cleanup_after=True), dict(upload='s3://b/x', mcap=False)]:
            args = dict(tasks=None, mcap=True, upload=None, cleanup_after=False)
            args.update(bad)
            try:
                process_station.station_job(str(root), str(config), None, args['tasks'], args['mcap'], args['upload'],
                    args['cleanup_after'])
                raise RuntimeError(f"accepted {bad}")
            except AssertionError:
                pass


def test_batched_mcap():
    import pickle
    with tempfile.TemporaryDirectory() as tmp:
        s = make_session(pathlib.Path(tmp))
        pickle.dump([{}] * 23, s.joinpath('dataset_plan.pkl').open('wb'))
        uploaded, exports = dict(), list()

        def fake_run(cmd, log):
            cmd = [str(c) for c in cmd]
            if '07_export_mcap.py' in ' '.join(cmd):
                batch = [int(k) for k in cmd[cmd.index('-e') + 1].split(',')]
                exports.append(batch)
                assert len(list(s.joinpath('dataset_mcap').glob('*.mcap'))) == 0, "earlier batch not deleted"
                for k in batch:
                    s.joinpath('dataset_mcap', f'episode_{k:06d}.mcap').write_bytes(b'm' * (k + 1))
            elif cmd[:3] == ['aws', 's3', 'sync']:
                for p in s.joinpath('dataset_mcap').glob('*.mcap'):
                    uploaded[p.name] = p.stat().st_size
            return 0, []
        process_station.run = fake_run
        process_station.s3_sizes = lambda url: dict(uploaded)
        job = {'name': 'station1', 'sessions': tmp, 'config': 'c.yaml', 'on_s3': False, 'root': s.parent,
            'names': ['task01'], 'upload': 's3://b/x'}
        summary = process_station.process(job, False, True, True, None, start_from='mcap', batch_size=10)
        assert exports == [list(range(10)), list(range(10, 20)), list(range(20, 23))], exports
        name, status, failed, plan_yield, _ = summary[0]
        assert status == 'ok', failed
        assert len(uploaded) == 23 and '23 MCAP files' in plan_yield and 'cleanup: deleted 2 videos' in plan_yield, plan_yield
        assert not list(s.rglob('*.mcap')) and not list(s.rglob('*.mp4'))
        # an upload that does not arrive: the batch stays local, the session fails, videos kept
        s2 = make_session(pathlib.Path(tmp, 'other'))
        pickle.dump([{}] * 5, s2.joinpath('dataset_plan.pkl').open('wb'))
        s = s2
        uploaded.clear()
        process_station.s3_sizes = lambda url: {}
        job.update({'root': s2.parent})
        summary = process_station.process(job, False, True, True, None, start_from='mcap', batch_size=10)
        assert summary[0][1] == 'failed' and 'not on S3' in summary[0][2], summary
        assert len(list(s2.rglob('*.mcap'))) == 5 and s2.joinpath('demos/demo_A/raw_video.mp4').is_file()


def test_download_space_check():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp, 'sessions')
        calls = list()
        process_station.run = lambda cmd, log: calls.append(cmd) or (0, [])
        process_station.s3_total_bytes = lambda url: 10 ** 15   # 1 PB: never fits
        job = {'name': 'station3', 'sessions': 's3://b/station3/industrial_data', 'config': 'c.yaml', 'on_s3': True,
            'root': root, 'names': ['task01', 'task02'], 'upload': 's3://b/up'}
        summary = process_station.process(job, False, True, True, None)
        assert [x[1] for x in summary] == ['skipped', 'skipped'] and 'enlarge the disk' in summary[0][2], summary
        assert calls == [], "nothing may run (no download) when the session does not fit"


def test_videos_only():
    assert process_station.task_path('s3://b/UMI_Collection/station3/industrial_data/2026-11-22', 'task01') \
        == 'station3/2026-11-22/task01'
    assert process_station.task_path('/data/station1/industrial_data', 'task02') == 'station1/task02'
    assert process_station.task_path('/data/x', 'task02') == 'task02'
    assert process_station.task_path('s3://b/UMI_Collection/station1/industrial_data_2026-11-22', 'task06') \
        == 'station1/2026-11-22/task06'
    assert process_station.task_path('s3://b/UMI_Collection/station1/industrial_data', 'task01', '2026-11-20') \
        == 'station1/2026-11-20/task01'
    assert process_station.task_path('s3://b/UMI_Collection/station3/industrial_data/2026-11-22', 'task01', '2026-11-20') \
        == 'station3/2026-11-22/task01'      # a date in the path wins
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp, 'work')
        s = make_session(root)
        calls = list()

        def fake_run(cmd, log):
            cmd = [str(c) for c in cmd]
            calls.append(cmd)
            if '15_export_episode_videos.py' in ' '.join(cmd):
                return 0, ['1/2 ...', 'done: 2 ok, 0 failed, 0 skipped (rerun to retry ...)']
            if 'run_tag_tracking_pipeline.py' in ' '.join(cmd):
                return 0, ['2 episodes, 30.0s of 40.0s used (75%)']
            return 0, []
        process_station.run = fake_run
        process_station.s3_total_bytes = lambda url: 0
        job = {'name': 'station3', 'sessions': 's3://b/UMI_Collection/station3/industrial_data/2026-11-22',
            'config': 'c.yaml', 'on_s3': True, 'root': root, 'names': ['task01'], 'upload': None,
            'mp4': 's3://b/umi_data/industrial_data_mp4'}
        summary = process_station.process(job, False, False, True, None, n_mosaic=1000, mp4_jobs=4)
        steps = [c[1] if c[0] != 'aws' else 'aws' for c in calls]
        assert steps == ['aws', 'run_tag_tracking_pipeline.py', 'scripts_tag_tracking/15_export_episode_videos.py', 'aws'], steps
        # the logs next to the output, after the session
        assert calls[3][3].endswith('task01/logs') and \
            calls[3][4] == 's3://b/umi_data/industrial_data_mp4/station3/2026-11-22/task01/logs/', calls[3]
        assert '--videos_only' in calls[1]
        export = calls[2]
        assert export[export.index('--task') + 1] == 'station3/2026-11-22/task01' and export[export.index('-o') + 1] == \
            's3://b/umi_data/industrial_data_mp4' and export[export.index('--mosaic') + 1] == '1000', export
        name, status, failed, plan_yield, _ = summary[0]
        assert status == 'ok', failed
        assert 'episode videos 2 ok, 0 failed, 0 skipped' in plan_yield and 'cleanup: deleted 2 videos' in plan_yield \
            and 'episode videos on s3://b/umi_data/industrial_data_mp4/station3/2026-11-22/task01' in plan_yield, plan_yield
        assert not list(s.rglob('*.mp4')) and not list(s.rglob('*.MP4'))
        # --videos_only replaces --mcap / --upload
        config = pathlib.Path(tmp, 'config.yaml')
        config.write_text('setup_calibration: setup_x\n')
        local = pathlib.Path(tmp, 'local', 'station3', 'industrial_data')
        local.joinpath('task01', 'raw_videos').mkdir(parents=True)
        try:
            process_station.station_job(str(local), str(config), None, None, True, None, False, mp4='s3://b/x')
            raise RuntimeError('accepted --mcap with --mp4')
        except AssertionError:
            pass


def test_jobs_videos_only():
    # --jobs with --videos_only --mp4 --cleanup: every job, its own work folder, dated output folders,
    # the videos deleted after each session's export
    from click.testing import CliRunner
    with tempfile.TemporaryDirectory() as tmp:
        config = pathlib.Path(tmp, 'config.yaml')
        config.write_text('setup_calibration: setup_x\n')
        jobs = pathlib.Path(tmp, 'jobs.yaml')
        jobs.write_text(f"""
- sessions: s3://b/UMI_Collection/station1/industrial_data
  config: {config}
  workdir: {tmp}/w1
  tasks: [task01]
  date: 2026-11-20
- sessions: s3://b/UMI_Collection/station1/industrial_data_2026-11-22
  config: {config}
  workdir: {tmp}/w2
  tasks: [task06]
""")
        calls = list()

        def fake_run(cmd, log):
            cmd = [str(c) for c in cmd]
            calls.append(cmd)
            if cmd[:3] == ['aws', 's3', 'sync'] and cmd[3].startswith('s3://'):     # download: a session with videos
                s = pathlib.Path(cmd[4])
                for d in ['demos/demo_A', 'raw_videos']:
                    s.joinpath(d).mkdir(parents=True, exist_ok=True)
                s.joinpath('demos/demo_A/raw_video.mp4').write_bytes(b'v' * 100)
                s.joinpath('raw_videos/GX01.MP4').write_bytes(b'v' * 100)
            if '15_export_episode_videos.py' in ' '.join(cmd):
                return 0, ['done: 3 ok, 0 failed, 0 skipped']
            return 0, []
        process_station.run = fake_run
        process_station.s3_sessions = lambda url: ['task01', 'task06']
        process_station.s3_total_bytes = lambda url: 0
        result = CliRunner().invoke(process_station.main, ['--jobs', str(jobs), '--videos_only', '--mp4', 's3://b/mp4',
            '--mosaic', '1000', '--cleanup'])
        assert result.exit_code == 0, (result.output, result.exception)
        exports = [c for c in calls if '15_export_episode_videos.py' in ' '.join(c)]
        assert [c[c.index('--task') + 1] for c in exports] == ['station1/2026-11-20/task01', 'station1/2026-11-22/task06'], exports
        assert exports[0][exports[0].index('-i') + 1] == f'{tmp}/w1/task01' and exports[1][exports[1].index('-i') + 1] == f'{tmp}/w2/task06'
        for task in ['station1/2026-11-20/task01', 'station1/2026-11-22/task06']:
            assert f"cleanup: deleted 1 videos, 1 raw_videos entries, 0 MCAP files (0.0 GB); videos stay in the S3 session, " \
                f"episode videos on s3://b/mp4/{task}" in result.output, result.output
        assert not list(pathlib.Path(tmp).rglob('*.mp4')) and not list(pathlib.Path(tmp).rglob('*.MP4'))


if __name__ == '__main__':
    test_jobs_videos_only()
    test_videos_only()
    test_download_space_check()
    test_batched_mcap()
    test_station_job()
    test_cleanup()
    print('all passed')
